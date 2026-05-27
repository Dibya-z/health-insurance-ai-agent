import pandas as pd
from langchain_core.tools import tool

@tool
def find_network_hospitals(city: str, specialty: str) -> list:
    """Lists cashless network hospitals in a specific city for a given specialty."""
    try:
        df = pd.read_csv('data/hospitals.csv')
        df['city'] = df['city'].str.lower()
        df['specialties'] = df['specialties'].str.lower()
        
        filtered = df[
            (df['city'] == city.lower()) & 
            (df['specialties'].str.contains(specialty.lower(), na=False)) &
            (df['cashless'] == True)
        ]
        
        hospitals = filtered.to_dict('records')
        return hospitals if hospitals else [{"message": "No cashless network hospitals found for this criteria."}]
    except Exception as e:
        return [{"error": str(e)}]
