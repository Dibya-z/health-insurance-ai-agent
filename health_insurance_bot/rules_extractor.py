"""Per-policy rule extractor.

For each canonical condition (from data/conditions_to_check.json), retrieves
relevant chunks via RAG and asks the LLM to extract structured coverage rules.
Output is saved to policies/<pdf_hash>/rules.json.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from ingest import pdf_fingerprint, search

load_dotenv()

POLICIES_DIR = Path(__file__).parent / "policies"
CONDITIONS_FILE = Path(__file__).parent / "data" / "conditions_to_check.json"
EXTRACTION_MODEL = "llama-3.3-70b-versatile"
MAX_WORKERS = 5

EXTRACTION_PROMPT = """You extract structured coverage rules from health insurance policies.

CONDITION: {condition}

RELEVANT POLICY EXCERPTS (with page numbers):
---
{excerpts}
---

Determine if this condition is covered, and what conditions/limits apply.

CRITICAL DISTINCTION — read carefully:
- A condition that is COVERED but has a WAITING PERIOD (e.g. cataract has 24-month wait, diabetes has 36-month PED wait) ===> covered: true, waiting_period_days: <days>. The wait does NOT mean it is excluded — it means the user must wait that long before claiming.
- A condition that appears in the policy's EXCLUSIONS section and is NEVER paid (e.g. IVF, cosmetic surgery, maternity in base plans, OPD, hearing aids) ===> covered: false.
- The phrase "shall be excluded until the expiry of X months" means there is a WAITING PERIOD — set covered: true with the appropriate waiting_period_days. Do NOT mark such conditions as covered: false.
- Permanent exclusions usually appear under "Standard Exclusions" or "Specific Exclusions" sections without any time-based phrase like "until X months".

Output STRICT JSON with this schema:
{{
  "covered": true|false,
  "sub_limit_inr": <integer or null>,
  "co_pay_percent": <number between 0 and 100>,
  "waiting_period_days": <integer>,
  "evidence_quote": "<short quote from the excerpts, max 200 chars>",
  "page_reference": <integer page number>,
  "notes": "<one sentence: 'Covered after Xd waiting period' OR 'Excluded permanently because Y'>"
}}

Mapping rules:
- "At actuals" or no specific monetary cap -> sub_limit_inr: null
- No condition-specific co-pay mentioned -> co_pay_percent: 0
- Waiting periods:
   * Initial 30-day general waiting (illnesses other than accidents, in first 30 days of policy) -> 30
   * Specified-disease list (cataract, hernia, joint replacement, cholecystectomy, hysterectomy, tonsillectomy, fibroids, kidney stones, etc.; commonly 24 months) -> 730
   * Pre-Existing Disease - PED (diabetes, hypertension, thyroid, asthma if pre-existing; commonly 36 months) -> 1095
   * Accident-related or no waiting -> 0
- If excerpts are insufficient to decide, set covered: true, waiting_period_days: 30, and explain uncertainty in notes."""


def get_policy_dir(pdf_path: Path) -> Path:
    d = POLICIES_DIR / pdf_fingerprint(pdf_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_one(client: Groq, pdf_path: Path, condition: str) -> dict:
    hits = search(pdf_path, f"{condition} coverage waiting period sub-limit exclusion", k=5)
    excerpts = "\n\n".join(f"[Page {h['page']}]: {h['text']}" for h in hits) or "[no excerpts found]"
    prompt = EXTRACTION_PROMPT.format(condition=condition, excerpts=excerpts)

    try:
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "You output only valid JSON. No markdown, no commentary."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {
            "covered": True,
            "sub_limit_inr": None,
            "co_pay_percent": 0,
            "waiting_period_days": 0,
            "evidence_quote": "",
            "page_reference": 0,
            "notes": f"extraction error: {e}",
        }


PERMANENT_EXCLUSION_KEYWORDS = {
    "maternity", "pregnancy", "childbirth", "ivf", "infertility", "sterility",
    "cosmetic", "plastic surgery", "lasik", "refractive",
    "hearing aid", "spectacles", "opd", "outpatient",
    "vaccination", "immunization", "self-inflicted", "suicide",
    "alcohol", "substance abuse", "hazardous", "adventure sports",
    "change of gender", "obesity", "bariatric",
}


def normalize_rule(condition: str, rule: dict) -> dict:
    """Apply semantic normalization the LLM tends to get wrong.

    1. Items matching permanent-exclusion keywords are forced covered=false (no wait).
    2. Items with waiting_period > 0 but covered=false are flipped — the LLM
       misread 'shall be excluded until the expiry of X months' as a permanent
       exclusion when it actually means a waiting period.
    """
    cond_l = condition.lower()
    if any(ex in cond_l for ex in PERMANENT_EXCLUSION_KEYWORDS):
        rule["covered"] = False
        rule["waiting_period_days"] = 0
        if "extraction error" in (rule.get("notes") or ""):
            rule["notes"] = f"Permanently excluded under standard exclusions (not covered under base plan)."
        return rule

    wait = rule.get("waiting_period_days") or 0
    if wait > 0 and rule.get("covered") is False:
        rule["covered"] = True
        original = rule.get("notes", "")
        wait_months = wait // 30
        rule["notes"] = f"Covered after {wait_months}-month waiting period. {original}".strip()
    return rule


def extract_all(pdf_path: Path, force: bool = False) -> Path:
    pol_dir = get_policy_dir(pdf_path)
    rules_path = pol_dir / "rules.json"

    if rules_path.exists() and not force:
        existing = json.loads(rules_path.read_text())
        n = existing.get("_meta", {}).get("num_conditions", 0)
        print(f"[skip] rules already extracted at {rules_path} ({n} conditions)")
        return rules_path

    meta = json.loads(CONDITIONS_FILE.read_text())
    todo = []
    for category, items in meta.items():
        if category == "_meta":
            continue
        for c in items:
            todo.append((category, c))

    print(f"[extract] {len(todo)} conditions to extract via {EXTRACTION_MODEL}")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")
    client = Groq(api_key=api_key)

    rules = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(extract_one, client, pdf_path, cond): (cat, cond) for cat, cond in todo}
        done = 0
        for fut in as_completed(futures):
            cat, cond = futures[fut]
            r = fut.result()
            r["_category"] = cat
            r = normalize_rule(cond, r)
            rules[cond] = r
            done += 1
            cov = "Y" if r.get("covered") else "N"
            wait = r.get("waiting_period_days", 0)
            print(f"  [{done:>2}/{len(todo)}] {cond:<45} covered={cov}  wait={wait}d")

    output = {
        "_meta": {
            "pdf_hash": pdf_fingerprint(pdf_path),
            "pdf_filename": pdf_path.name,
            "extracted_at": datetime.utcnow().isoformat() + "Z",
            "extraction_model": EXTRACTION_MODEL,
            "num_conditions": len(rules),
        },
        "rules": rules,
    }
    rules_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nrules saved -> {rules_path}")
    return rules_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python rules_extractor.py <path-to-pdf> [--force]")
        sys.exit(1)
    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"error: not found: {pdf}")
        sys.exit(1)
    force = "--force" in sys.argv
    extract_all(pdf, force=force)
