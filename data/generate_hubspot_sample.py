"""Generates a second synthetic customer, in a different CRM's shape.

Where `generate_sample_data.py` produces a Salesforce-style export (Northwind, North America), this one
produces a HubSpot-style export for a different company, region, currency set and fiscal quarter:

    * HubSpot property names   (hs_object_id, dealstage, hs_deal_stage_probability, amount_in_home_currency)
    * HubSpot stage ids        (qualifiedtobuy, presentationscheduled, closedwon ...)
    * HubSpot forecast values  (COMMIT, BEST_CASE, PIPELINE, OMITTED)
    * probability as 0-1       (Salesforce uses 0-100)

The toolkit reads it with no code change: `customers/acme_hubspot.json` maps the columns and the values.

    python data/generate_hubspot_sample.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 5
FISCAL_QUARTER = "2027-Q3"
QUARTER_START = pd.Timestamp("2027-05-01")
QUARTER_END = pd.Timestamp("2027-07-31")
SNAPSHOT = pd.Timestamp("2027-07-19")            # 12 days before the quarter ends
SOFTEN = 0.94

FX = {"EUR": 1.09, "SEK": 0.096, "NOK": 0.094, "DKK": 0.146}
CURRENCY = {"IE": "EUR", "FI": "EUR", "SE": "SEK", "NO": "NOK", "DK": "DKK"}
OPEN_STAGES = ["qualifiedtobuy", "presentationscheduled", "decisionmakerboughtin", "contractsent"]
STAGE_MIX = [0.18, 0.28, 0.31, 0.23]
STAGE_PROB = {"qualifiedtobuy": (0.10, 0.35), "presentationscheduled": (0.25, 0.55),
              "decisionmakerboughtin": (0.40, 0.70), "contractsent": (0.55, 0.92)}
DEAL_TYPES = ["newbusiness", "renewal", "expansion"]
DEAL_TYPE_MIX = [0.45, 0.33, 0.22]
LOSS_REASONS = ["Price", "Lost to competitor", "No decision", "Timing", "Budget frozen"]
CHURN_REASONS = ["Churn - competitor", "Churn - budget", "Churn - low usage", "Non-renewal - no response"]

# manager, country, team, quota per seat, share of stale pipeline, owners:
# (name, attainment this quarter, last quarter, two quarters ago, tenure months, status)
TEAMS = [
    dict(manager="Sigrid Halvorsen", country="NO", team="Nordics Enterprise", quota=540_000, stale=0.41, reps=[
        ("Anders Bergstrom", 1.04, 1.07, 0.98, 27, "ACTIVE"),
        ("Linnea Falk", 1.09, 1.00, 1.06, 20, "ACTIVE"),
        ("Mikkel Sorensen", 0.93, 0.92, 1.03, 33, "ACTIVE")]),
    dict(manager="Ronan Gallagher", country="IE", team="Ireland Mid-Market", quota=300_000, stale=0.46, reps=[
        ("Saoirse Donnelly", 1.07, 1.09, 1.01, 29, "ACTIVE"),
        ("Cathal Mulvaney", 0.99, 0.95, 1.04, 17, "ACTIVE"),
        ("Aoife Brennan", 0.96, 1.03, 0.94, 23, "ACTIVE"),
        ("VACANT (Ferris territory, req open)", 0.36, 0.97, 0.91, 0, "VACANT")]),
    dict(manager="Elin Waller", country="SE", team="Sweden Mid-Market", quota=280_000, stale=0.88, reps=[
        ("Johan Ostberg", 0.94, 1.01, 0.97, 21, "ACTIVE"),
        ("Petra Lindholm", 0.86, 0.85, 0.87, 26, "ACTIVE"),
        ("Gustav Ahlin", 1.01, 0.98, 1.05, 14, "ACTIVE"),
        ("Nora Ekstrom", 0.97, 1.02, 0.99, 31, "ACTIVE")]),
    dict(manager="Freja Lundqvist", country="DK", team="Denmark SMB", quota=185_000, stale=0.35, reps=[
        ("Emil Kofoed", 1.06, 1.05, 1.02, 24, "ACTIVE"),
        ("Ida Jorgensen", 0.95, 0.97, 0.96, 13, "ACTIVE"),
        ("Rasmus Vang", 1.02, 0.99, 1.08, 19, "ACTIVE"),
        ("Clara Bech", 0.90, 0.93, 0.95, 8, "ACTIVE")]),
    dict(manager="Tuomas Leino", country="FI", team="Finland SMB", quota=175_000, stale=0.44, reps=[
        ("Aino Virtanen", 1.05, 1.04, 1.01, 22, "ACTIVE"),
        ("Eero Nieminen", 0.92, 0.90, 0.96, 11, "ACTIVE"),
        ("Helmi Rautio", 1.00, 1.06, 0.97, 28, "ACTIVE")]),
]
ACCOUNT_FIRST = ["Aurora", "Baltic", "Cinder", "Drift", "Everly", "Fjord", "Glacier", "Hollow", "Isla",
                 "Juniper", "Kelvin", "Lumen", "Mistral", "Nordic", "Orbit", "Polar", "Quartz", "Ravel"]
ACCOUNT_LAST = ["Logistik", "Systems", "Care", "Media", "Foods", "Bank", "Works", "Retail", "Energi",
                "Labs", "Group", "Telecom"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    owners, deals, acct = [], [], {}
    obj_id = 14_500_000

    def account() -> tuple[str, str]:
        name = f"{rng.choice(ACCOUNT_FIRST)} {rng.choice(ACCOUNT_LAST)}"
        if name in acct and rng.random() < 0.7:
            return acct[name], name
        acct[name] = str(int(rng.integers(7_000_000, 7_999_999)))
        return acct[name], name

    def a_date(low: pd.Timestamp, high: pd.Timestamp) -> pd.Timestamp:
        return low + pd.Timedelta(days=int(rng.integers(0, max(1, (high - low).days))))

    def forecast_value(stage: str) -> str:
        late = stage in ("decisionmakerboughtin", "contractsent")
        weights = [0.31, 0.21, 0.34, 0.14] if late else [0.09, 0.18, 0.58, 0.15]
        return str(rng.choice(["COMMIT", "BEST_CASE", "PIPELINE", "OMITTED"], p=weights))

    def add(owner, team, stage, amount_usd, created, close, probability, deal_type, forecast, loss=None):
        nonlocal obj_id
        obj_id += 1
        ccy = CURRENCY[team["country"]]
        acc_id, acc_name = account()
        amount_usd = float(round(amount_usd, 0))
        deals.append({
            "hs_object_id": obj_id,
            "dealname": f"{acc_name} ({FISCAL_QUARTER})",
            "associatedcompanyid": acc_id,
            "company_name": acc_name,
            "country": team["country"],
            "team__c": team["team"],
            "hubspot_owner_name": owner,
            "owner_manager": team["manager"],
            "dealtype": deal_type,
            "dealstage": stage,
            "hs_forecast_category": forecast,
            "hs_deal_stage_probability": round(float(probability), 2),
            "amount": int(round(amount_usd / FX[ccy])),
            "deal_currency_code": ccy,
            "amount_in_home_currency": amount_usd,
            "createdate": created,
            "closedate": close,
            "days_open": int((SNAPSHOT - created).days),
            "closed_lost_reason": loss,
            "notes_next_step": np.nan,
            "fiscal_period": FISCAL_QUARTER,
        })

    for team in TEAMS:
        for owner, att, h1, h2, tenure, status in team["reps"]:
            owners.append({"owner_name": owner, "manager_name": team["manager"], "country_code": team["country"],
                           "team_name": team["team"], "quota_amount_usd": team["quota"],
                           "months_in_seat": tenure, "employment_status": status,
                           "attainment_last_quarter": h1, "attainment_two_quarters_ago": h2})
            quota, vacant = team["quota"], status != "ACTIVE"
            forecast_target = att * SOFTEN * quota
            won_target = forecast_target * rng.uniform(0.40, 0.52)
            open_target = forecast_target - won_target

            sizes = rng.lognormal(0, 0.42, int(rng.integers(1, 3) if vacant else rng.integers(4, 9)))
            for amt in won_target * sizes / sizes.sum():
                created = a_date(QUARTER_START - pd.Timedelta(days=110), SNAPSHOT - pd.Timedelta(days=8))
                add(owner, team, "closedwon", amt, created, a_date(max(QUARTER_START, created), SNAPSHOT),
                    1.0, str(rng.choice(DEAL_TYPES, p=DEAL_TYPE_MIX)), "CLOSED")

            for _ in range(0 if vacant else int(rng.integers(0, 3))):
                created = a_date(QUARTER_START - pd.Timedelta(days=140), SNAPSHOT - pd.Timedelta(days=18))
                deal_type = str(rng.choice(DEAL_TYPES, p=DEAL_TYPE_MIX))
                reason = str(rng.choice(CHURN_REASONS if deal_type == "renewal" else LOSS_REASONS))
                add(owner, team, "closedlost", rng.uniform(12_000, 60_000) * (quota / 400_000), created,
                    a_date(max(QUARTER_START, created), SNAPSHOT), 0.0, deal_type, "OMITTED", reason)

            n_open = int(rng.integers(2, 4) if vacant else rng.integers(6, 12))
            stages = rng.choice(OPEN_STAGES, n_open, p=STAGE_MIX)
            probs = np.array([rng.uniform(*STAGE_PROB[s]) for s in stages])
            sizes = rng.lognormal(0, 0.48, n_open)
            for stage, prob, amt in zip(stages, probs, open_target * sizes / (sizes * probs).sum()):
                stale = rng.random() < team["stale"]
                created = (a_date(QUARTER_START - pd.Timedelta(days=150), QUARTER_START) if stale
                           else a_date(QUARTER_START, SNAPSHOT - pd.Timedelta(days=3)))
                close = (a_date(max(QUARTER_START, created + pd.Timedelta(days=18)), SNAPSHOT) if stale
                         else a_date(SNAPSHOT + pd.Timedelta(days=1), QUARTER_END))
                add(owner, team, str(stage), amt, created, close, prob,
                    str(rng.choice(DEAL_TYPES, p=DEAL_TYPE_MIX)), forecast_value(str(stage)))

    opp = pd.DataFrame(deals)
    roster = pd.DataFrame(owners)

    ahead = opp[opp["dealstage"].isin(OPEN_STAGES) & (opp["closedate"] > SNAPSHOT)].index
    opp.loc[ahead[0], ["createdate", "days_open"]] = [SNAPSHOT, 0]
    opp.loc[ahead[1], "closedate"] = QUARTER_END

    # ---- planted data-quality problems (different counts from the other customer) ---------
    planted = {}
    idx = opp[opp["deal_currency_code"].isin(["EUR", "DKK"])].sample(6, random_state=SEED + 1).index
    opp.loc[idx, "amount_in_home_currency"] = opp.loc[idx, "amount"].astype(float)
    planted["rows never converted to USD"] = len(idx)

    spelled = {"SE": "Sweden", "DK": "Denmark", "IE": "Ireland"}
    idx = opp[opp["country"].isin(spelled)].sample(27, random_state=SEED + 2).index
    opp.loc[idx, "country"] = opp.loc[idx, "country"].map(spelled)
    planted["rows with a spelled-out country"] = len(idx)

    idx = opp.sample(120, random_state=SEED + 3).index
    opp.loc[idx, "days_open"] = (opp.loc[idx, "closedate"] - opp.loc[idx, "createdate"]).dt.days
    opp.loc[opp.sample(2, random_state=SEED + 4).index, "days_open"] = -1
    planted["rows with a wrong days_open"] = len(idx)

    idx = opp[opp["dealstage"].isin(OPEN_STAGES)].sample(17, random_state=SEED + 5).index
    opp.loc[idx, "hs_forecast_category"] = np.nan
    planted["open deals with no forecast category"] = len(idx)
    lost = opp[opp["dealstage"] == "closedlost"].sample(4, random_state=SEED + 6).index
    opp.loc[lost, "closed_lost_reason"] = np.nan
    planted["lost deals with no reason"] = len(lost)

    dupes = opp.sample(4, random_state=SEED)
    opp = pd.concat([opp, dupes], ignore_index=True)
    planted["duplicate rows"] = len(dupes)

    opp = opp.sample(frac=1, random_state=SEED + 7).reset_index(drop=True)
    for col in ("createdate", "closedate"):
        opp[col] = pd.to_datetime(opp[col]).dt.strftime("%Y-%m-%d")

    opp.to_csv(HERE / "hubspot_deals_export.csv", index=False)
    roster.to_csv(HERE / "hubspot_owner_quotas_export.csv", index=False)
    print(f"{len(opp)} deal rows, {len(roster)} owner rows")
    print("planted:", ", ".join(f"{v} {k}" for k, v in planted.items()))


if __name__ == "__main__":
    main()
