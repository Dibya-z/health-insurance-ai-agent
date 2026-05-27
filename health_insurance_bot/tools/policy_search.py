"""RAG search over the uploaded policy PDF."""

from langchain_core.tools import tool

from ._context import CTX
from ingest import search


@tool
def search_policy_docs(query: str, k: int = 5) -> list[dict]:
    """Search the user's uploaded policy PDF for clauses matching a natural-language query (RAG).

    Use this for open-ended 'why' or 'explain' questions about policy wording,
    definitions, or clauses NOT directly answered by check_coverage. Examples:
      - "what's the definition of pre-existing disease?"
      - "what's the claim notification timeline?"
      - "explain the proportionate deduction rule for room rent"
      - "what does 'at actuals' mean?"

    Returns up to k chunks with text, page number (for citation), and a relevance score.
    """
    if CTX.pdf_path is None:
        return [{"error": "No policy PDF loaded in this session."}]
    hits = search(CTX.pdf_path, query, k=k)
    return [
        {"text": h["text"], "page": h["page"], "relevance_score": round(1 - h["distance"], 3)}
        for h in hits
    ]
