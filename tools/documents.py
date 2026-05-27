from langchain_core.tools import tool

@tool
def list_required_documents(claim_type: str) -> list:
    """Returns the paperwork checklist for a specific claim type (e.g., 'hospitalization', 'daycare', 'reimbursement')."""
    claim_type = claim_type.lower()
    docs = {
        "hospitalization": [
            "Pre-authorization form from the hospital",
            "Original discharge summary",
            "All original bills and payment receipts",
            "Diagnostic reports (X-rays, blood work, etc.)",
            "Valid ID proof"
        ],
        "reimbursement": [
            "Duly filled and signed claim form",
            "Original discharge summary",
            "Original final hospital bill with detailed breakdown",
            "Original payment receipts",
            "All investigation reports",
            "Pharmacy bills with prescriptions",
            "Cancelled cheque for NEFT transfer",
            "KYC documents / Valid ID proof"
        ],
        "daycare": [
            "Daycare summary",
            "Original bills and receipts",
            "Doctor's prescription for the procedure",
            "Valid ID proof"
        ]
    }
    
    return docs.get(claim_type, ["Standard claim form", "Original bills", "Discharge summary", "ID proof"])
