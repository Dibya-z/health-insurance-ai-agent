"""Coverage check + claim calculation tools."""

import json
from langchain_core.tools import tool

from ._context import CTX, DATA_DIR


def _find_condition(condition: str) -> dict | None:
    """Fuzzy-match a condition name against the per-policy rules dict."""
    if not CTX.rules:
        return None
    cl = condition.lower().strip()
    # exact
    for k, v in CTX.rules.items():
        if k.lower() == cl:
            return v
    # substring either direction
    for k, v in CTX.rules.items():
        if cl in k.lower() or k.lower() in cl:
            return v
    # token overlap
    cl_tokens = set(cl.replace(",", " ").split())
    best, best_score = None, 0
    for k, v in CTX.rules.items():
        kt = set(k.lower().replace(",", " ").split())
        score = len(cl_tokens & kt)
        if score > best_score:
            best, best_score = v, score
    return best if best_score >= 1 else None


@tool
def check_coverage(condition: str) -> dict:
    """Check if a specific medical condition or procedure is covered under the user's policy.

    Use this when the user asks 'is X covered?' or wants to know sub-limits, co-pay,
    or waiting periods for a specific condition (e.g. 'cataract surgery', 'diabetes',
    'maternity', 'IVF', 'knee replacement', 'hernia', 'cosmetic surgery').

    Returns a dict with: covered (bool), sub_limit_inr, co_pay_percent,
    waiting_period_days, evidence_quote, page_reference, notes.
    Returns covered=null if the condition is not in the rules — in that case,
    follow up with search_policy_docs.
    """
    if not CTX.rules:
        return {"error": "No policy loaded. Upload a PDF first."}
    rule = _find_condition(condition)
    if rule is None:
        return {
            "covered": None,
            "condition": condition,
            "notes": f"No structured rule found for '{condition}'. Use search_policy_docs to look it up in the policy text.",
        }
    return {"condition": condition, **rule}


@tool
def calculate_claim(condition: str, bill_amount: float) -> dict:
    """Calculate the expected claim payout for a specific condition and bill amount.

    Use when the user asks 'how much will I get?', 'what's my reimbursement?', or
    similar. Combines the condition's coverage rule with the user's policy
    (sum_insured, co_pay).

    Math: covered = min(bill, sub_limit OR sum_insured)
          payable = covered * (1 - effective_co_pay)
    """
    from .user import _get_user_data

    if not CTX.rules:
        return {"error": "No policy loaded."}
    rule = _find_condition(condition)
    if rule is None:
        return {"error": f"No rule found for '{condition}'. Try search_policy_docs."}
    if not rule.get("covered"):
        return {
            "condition": condition,
            "covered": False,
            "covered_amount": 0,
            "co_pay_amount": 0,
            "payable_to_user": 0,
            "reason": rule.get("notes") or "Excluded under the policy.",
            "policy_reference": rule.get("evidence_quote", "")[:200],
        }

    user = _get_user_data(CTX.user_id) or {}
    sum_insured = float(user.get("sum_insured", 500_000))
    sub_limit = rule.get("sub_limit_inr")
    cap = float(sub_limit) if sub_limit is not None else sum_insured

    covered = min(float(bill_amount), cap)

    cond_copay = float(rule.get("co_pay_percent") or 0) / 100.0
    user_copay = float(user.get("co_pay") or 0)
    effective_copay = max(cond_copay, user_copay)

    copay_amount = covered * effective_copay
    payable = covered - copay_amount

    return {
        "condition": condition,
        "bill_amount": float(bill_amount),
        "sub_limit_applied": sub_limit if sub_limit is not None else f"sum_insured ({int(sum_insured)})",
        "covered_amount": round(covered, 2),
        "co_pay_percent": round(effective_copay * 100, 2),
        "co_pay_amount": round(copay_amount, 2),
        "payable_to_user": round(payable, 2),
        "waiting_period_days": rule.get("waiting_period_days", 0),
        "policy_reference": rule.get("evidence_quote", "")[:200],
    }


if __name__ == "__main__":
    # smoke test (requires a processed policy + user set)
    from ingest import pdf_fingerprint
    from pathlib import Path
    from . import _context as ctx_mod

    pdf = Path(__file__).parent.parent / "data" / "policy.pdf"
    ctx_mod.set_policy(pdf_fingerprint(pdf), pdf)
    ctx_mod.set_user("U1001")
    print("check_coverage('cataract surgery'):")
    print(check_coverage.invoke({"condition": "cataract surgery"}))
    print("\ncalculate_claim('cataract surgery', 50000):")
    print(calculate_claim.invoke({"condition": "cataract surgery", "bill_amount": 50000}))
