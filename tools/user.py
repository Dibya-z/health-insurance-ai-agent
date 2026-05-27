import json
from langchain_core.tools import tool

@tool
def get_user_policy(user_id: str) -> dict:
    """Returns user's policy details including sum insured, co-pay percentage, and no-claim bonus."""
    try:
        with open('data/users.json', 'r') as f:
            users = json.load(f)
        
        for user in users:
            if user['user_id'].lower() == user_id.lower():
                return user
                
        return {"error": "User not found"}
    except Exception as e:
        return {"error": str(e)}
