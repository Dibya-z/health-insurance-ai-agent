"""Single entry point to process a policy PDF: ingest + extract.

Usage:
    python policy_processor.py <path-to-pdf>

Steps:
    1. Parse + chunk + embed the PDF -> ChromaDB collection (per-PDF)
    2. Extract structured coverage rules -> policies/<hash>/rules.json

Both steps are idempotent (skip if already done for this PDF hash).
"""

import sys
from pathlib import Path

from ingest import ingest, pdf_fingerprint
from rules_extractor import extract_all, get_policy_dir


def process(pdf_path: Path) -> dict:
    print(f"=== processing {pdf_path.name} (hash={pdf_fingerprint(pdf_path)}) ===\n")

    print("--- step 1: RAG ingestion ---")
    n_chunks = ingest(pdf_path)

    print("\n--- step 2: rule extraction ---")
    rules_path = extract_all(pdf_path)

    pol_dir = get_policy_dir(pdf_path)
    return {
        "pdf_hash": pdf_fingerprint(pdf_path),
        "policy_dir": str(pol_dir),
        "rules_path": str(rules_path),
        "num_chunks": n_chunks,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python policy_processor.py <path-to-pdf>")
        sys.exit(1)
    pdf = Path(sys.argv[1])
    if not pdf.exists():
        print(f"error: not found: {pdf}")
        sys.exit(1)
    result = process(pdf)
    print("\n=== done ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
