import json
from langchain_core.tools import tool

@tool
def check_coverage(condition: str) -> dict:
    """Returns whether a specific condition or procedure is covered, its sub-limits, and waiting periods."""
    try:
        with open('data/coverage_rules.json', 'r') as f:
            rules = json.load(f)
        return rules.get(condition.lower(), {'covered': False, 'sub_limit': 0, 'waiting_period_months': 0})
    except Exception as e:
        return {"error": str(e)}

@tool
def calculate_claim(condition: str, bill_amount: float, co_pay_percentage: float = 0.0) -> dict:
    """Computes expected claim amount given the condition, bill amount, and optionally user's co-pay percentage."""
    try:
        with open('data/coverage_rules.json', 'r') as f:
            rules = json.load(f)
        rule = rules.get(condition.lower(), {'covered': False, 'sub_limit': 0})
        
        if not rule['covered']:
            return {"covered_amount": 0, "co_pay": 0, "payable_to_user": 0, "message": "Condition not covered"}
            
        sub_limit = rule.get('sub_limit', 0)
        
        # If sub_limit is 0, we assume no sub-limit applies and full bill amount is considered.
        if sub_limit > 0:
            covered_amount = min(bill_amount, sub_limit)
        else:
            covered_amount = bill_amount
            
        co_pay_amount = covered_amount * co_pay_percentage
        payable_to_user = covered_amount - co_pay_amount
        
        return {
            "covered_amount": covered_amount,
            "co_pay": co_pay_amount,
            "payable_to_user": payable_to_user
        }
    except Exception as e:
        return {"error": str(e)}
