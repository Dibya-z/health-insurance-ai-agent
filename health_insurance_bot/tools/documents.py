"""Required-documents lookup tool."""

import json
from langchain_core.tools import tool

from ._context import DATA_DIR

DOCS_FILE = DATA_DIR / "documents_rules.json"


@tool
def list_required_documents(claim_type: str) -> dict:
    """List the paperwork required to file a specific type of insurance claim.

    Use this when the user asks 'what documents do I need?', 'how do I file a
    claim?', or 'what paperwork is required for X?'.

    Args:
        claim_type: One of:
          - 'cashless_hospitalization' (planned cashless treatment in network hospital)
          - 'reimbursement_hospitalization' (paid out of pocket, claiming back)
          - 'day_care_procedure' (under 24h surgical procedures)
          - 'pre_post_hospitalization' (consultations/tests before or after stay)
          - 'maternity' (childbirth-related)
        If user just says 'hospitalization', default to 'reimbursement_hospitalization'.
        If user says 'cashless', use 'cashless_hospitalization'.
    """
    docs = json.loads(DOCS_FILE.read_text())
    ct = claim_type.lower().strip().replace(" ", "_").replace("-", "_")
    if ct in docs:
        return {"claim_type": ct, "documents": docs[ct]}
    for k in docs:
        if ct in k or k in ct:
            return {"claim_type": k, "documents": docs[k]}
    return {"error": f"Unknown claim type '{claim_type}'. Valid options: {list(docs.keys())}"}
