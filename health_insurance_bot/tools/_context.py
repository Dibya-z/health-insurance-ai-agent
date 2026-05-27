"""Shared session-level context: which policy + user is currently active.

App.py (Streamlit) sets these at the start of each request; tools read from
the module-level CTX object. This is mutable global state, appropriate for
a single-session Streamlit app.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
POLICIES_DIR = Path(__file__).parent.parent / "policies"


@dataclass
class PolicyContext:
    pdf_path: Path | None = None
    pdf_hash: str | None = None
    rules: dict = field(default_factory=dict)
    user_id: str | None = None
    custom_user: dict | None = None  # set when user enters their own info instead of picking a demo profile


CTX = PolicyContext()


def set_policy(pdf_hash: str, pdf_path: Path | None = None) -> None:
    CTX.pdf_hash = pdf_hash
    CTX.pdf_path = pdf_path
    rules_file = POLICIES_DIR / pdf_hash / "rules.json"
    if rules_file.exists():
        CTX.rules = json.loads(rules_file.read_text()).get("rules", {})
    else:
        CTX.rules = {}


def set_user(user_id: str) -> None:
    CTX.user_id = user_id
    CTX.custom_user = None  # picking a demo profile clears any custom user


def set_custom_user(profile: dict) -> None:
    """Use the user's own typed-in profile instead of a demo entry from users.json."""
    CTX.custom_user = profile
    CTX.user_id = profile.get("user_id", "self")
