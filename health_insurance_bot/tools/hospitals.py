"""Network hospital lookup tool."""

import pandas as pd
from langchain_core.tools import tool

from ._context import DATA_DIR

HOSP_FILE = DATA_DIR / "hospitals.csv"


@tool
def find_network_hospitals(city: str, specialty: str = "") -> list[dict]:
    """Find CASHLESS network hospitals in a given city, optionally filtered by specialty.

    Use this when the user asks 'which hospitals can I go to?', 'find a cashless
    hospital for X near me', or 'is hospital Y in my network?'. Always returns
    ONLY cashless hospitals (filtered automatically).

    Args:
        city: One of Bangalore, Mumbai, Delhi, Chennai, Hyderabad, Pune, Kolkata.
        specialty: Optional. e.g. cardiology, oncology, orthopedics, ophthalmology,
            neurology, urology, gynecology, pediatrics, gastroenterology. Empty = any.

    Returns a list of hospitals with name, area, specialties, rating, phone.
    """
    df = pd.read_csv(HOSP_FILE)
    df = df[df["cashless"] == True]  # noqa: E712
    df = df[df["city"].str.lower() == city.lower()]
    if specialty:
        df = df[df["specialties"].str.lower().str.contains(specialty.lower(), na=False)]
    if df.empty:
        return [{
            "message": f"No cashless network hospitals found in {city}"
                       + (f" for {specialty}" if specialty else "")
        }]
    return df.drop(columns=["cashless"]).to_dict(orient="records")
