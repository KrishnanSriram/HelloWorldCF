from mock_data import (OFFICERS_DATA, PORTFOLIO_DATA, ATM_DATA,
                       RETIREMENT_DATA, LOCATIONS_DATA)

def get_atm_centers(zipcode: str) -> dict:
    data = ATM_DATA.get(str(zipcode).strip())
    if not data:
        return {
            "error": f"No ATM data for zipcode '{zipcode}'.",
            "available_zipcodes": list(ATM_DATA.keys()),
        }
    return {"zipcode": zipcode, "atm_centers": data, "total": len(data)}


def get_investment_officers(city: str) -> dict:
    key = city.strip().lower()
    data = OFFICERS_DATA.get(key)
    if not data:
        return {
            "error": f"No investment officers found in '{city}'.",
            "available_cities": [c.title() for c in OFFICERS_DATA],
        }
    return {"city": city, "officers": data, "total": len(data)}


def get_portfolio_companies(sector: str = "all") -> dict:
    companies = PORTFOLIO_DATA
    if sector and sector.lower() != "all":
        companies = [c for c in companies if sector.lower() in c["sector"].lower()]
    return {"companies": companies, "total": len(companies), "total_aum": "$6.35B"}


def get_retirement_plans(plan_type: str = "all") -> dict:
    plans = RETIREMENT_DATA
    if plan_type and plan_type.lower() != "all":
        plans = [
            p for p in plans
            if plan_type.lower() in p["name"].lower() or plan_type.lower() in p["type"].lower()
        ]
    return {"plans": plans, "total": len(plans)}


def get_bank_locations(city: str = "all") -> dict:
    locations = LOCATIONS_DATA
    if city and city.lower() != "all":
        locations = [l for l in locations if city.lower() in l["city"].lower()]
    total_branches = sum(l["branches"] for l in locations)
    return {"locations": locations, "total_branches": total_branches}