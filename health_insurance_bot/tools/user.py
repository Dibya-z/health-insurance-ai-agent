"""User policy lookup tool."""

import json
from langchain_core.tools import tool

from ._context import CTX, DATA_DIR

USERS_FILE = DATA_DIR / "users.json"


def _get_user_data(user_id: str | None) -> dict | None:
    # Custom user (typed in by the user themselves) takes precedence
    if CTX.custom_user is not None:
        return CTX.custom_user
    if not user_id:
        return None
    users = json.loads(USERS_FILE.read_text())
    return users.get(user_id)


@tool
def get_user_policy(user_id: str = "") -> dict:
    """Get the user's policy details: sum insured, co-pay percentage, no-claim bonus,
    policy start date, city, pre-existing conditions.

    Use this whenever you need user-specific values for a calculation (sum insured
    for claim cap, user's co-pay rate, eligibility for waiting periods based on
    policy_start), or to answer profile questions like 'what's my sum insured?'.

    If user_id is empty, defaults to the currently selected user in the session.
    """
    uid = user_id or CTX.user_id
    user = _get_user_data(uid)
    if user is None:
        return {"error": f"No user found with id '{uid}'"}
    return {"user_id": uid, **user}
