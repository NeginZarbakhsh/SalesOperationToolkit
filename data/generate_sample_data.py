"""Generates the synthetic sample data used by this repo.

The files are shaped like a CRM report export (Salesforce-style field names: `Id`, `StageName`,
`ForecastCategoryName`, `Probability`, `Account.BillingCountry`, custom `__c` fields), because that is what a
Sales Ops team actually receives. `config.json` maps those columns onto the toolkit's internal names.

Nothing here comes from a real company: the region, teams, people, accounts and amounts are invented, and the
data-quality problems (duplicate rows, amounts never converted, spelled-out countries, a broken age field,
stale close dates, seats with nobody in them) are planted on purpose so the diagnostic has something to find.

    python data/generate_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 24
FISCAL_QUARTER = "FY27-Q4"
QUARTER_START = pd.Timestamp("2027-02-01")
QUARTER_END = pd.Timestamp("2027-04-30")
SNAPSHOT = pd.Timestamp("2027-04-16")            # "today" in the data: 14 days before the quarter ends
SOFTEN = 0.96                                    # scales the region so it lands just under target

FX = {"USD": 1.0, "CAD": 0.73, "MXN": 0.058, "BRL": 0.19}
CURRENCY = {"US": "USD", "CA": "CAD", "MX": "MXN", "BR": "BRL"}
OPEN_STAGES = ["Prospecting", "Qualification", "Proposal/Price Quote", "Negotiation/Review"]
STAGE_MIX = [0.11, 0.23, 0.37, 0.29]
STAGE_PROB = {"Prospecting": (5, 30), "Qualification": (15, 45),
              "Proposal/Price Quote": (25, 60), "Negotiation/Review": (45, 90)}
DEAL_TYPES = ["New Business", "Renewal", "Upsell", "Pilot"]
DEAL_TYPE_MIX = [0.44, 0.29, 0.17, 0.10]
LOSS_REASONS = ["Price", "Lost to competitor", "No decision", "Timing - pushed out", "Budget cut"]
CHURN_REASONS = ["Churn - to competitor", "Churn - budget cut", "Churn - product fit", "Non-renewal - no response"]
PRODUCTS = ["Platform Expansion", "Annual Subscription", "Analytics Add-on", "Enterprise Rollout",
            "Support Upgrade", "Pilot Programme", "Seat Expansion", "Data Migration"]

# manager, country, segment, quota per seat, how stale the team's pipeline is, and the reps:
# (name, attainment this quarter, last quarter, two quarters ago, tenure months, user status)
TEAMS = [
    dict(manager="Carmen Dorsey", country="US", segment="Enterprise", quota=750_000, stale=0.44, reps=[
        ("Dexter Moyo", 1.03, 1.06, 0.97, 28, "Active"),
        ("Priscilla Vance", 1.10, 1.01, 1.09, 19, "Active"),
        ("Emmett Sorrell", 0.95, 0.93, 1.02, 34, "Active"),
        ("Rosalind Abiola", 0.88, 0.96, 0.92, 11, "Active")]),
    dict(manager="Theo Vandermeer", country="CA", segment="Commercial", quota=380_000, stale=0.48, reps=[
        ("Imogen Blackwood", 1.05, 1.11, 0.99, 31, "Active"),
        ("Rashid Benally", 1.00, 0.97, 1.05, 16, "Active"),
        ("Colette Ferrand", 0.97, 1.02, 0.93, 22, "Active"),
        ("OPEN REQ (Marcus Iwu backfill pending)", 0.38, 0.98, 0.89, 0, "Open Req"),
        ("OPEN REQ (Delia Santoro backfill pending)", 0.31, 0.92, 1.02, 0, "Open Req")]),
    dict(manager="Beatriz Salgado", country="BR", segment="Enterprise", quota=620_000, stale=0.40, reps=[
        ("Caio Brandao", 0.98, 1.04, 0.95, 24, "Active"),
        ("Larissa Pequeno", 0.84, 0.87, 0.85, 29, "Active"),
        ("Otavio Rennard", 0.92, 0.94, 1.01, 13, "Active"),
        ("Nayara Quintal", 0.93, 0.98, 0.96, 9, "Active")]),
    dict(manager="Harriet Okonjo", country="US", segment="SMB", quota=240_000, stale=0.52, reps=[
        ("Sybil Trent", 1.09, 1.07, 1.13, 26, "Active"),
        ("Malik Doyle", 0.96, 1.00, 0.94, 12, "Active"),
        ("Anouk Ferreira", 0.94, 0.93, 0.98, 17, "Active"),
        ("Jonas Whitlock", 1.05, 0.99, 1.07, 7, "Active"),
        ("Tabitha Nyong", 0.98, 1.08, 0.95, 21, "Active")]),
    dict(manager="Rafael Ibarra", country="MX", segment="Public Sector", quota=195_000, stale=0.93, reps=[
        ("Ximena Robledo", 0.95, 1.03, 0.98, 20, "Active"),
        ("Bruno Cavazos", 0.87, 0.86, 0.88, 15, "Active"),
        ("Paulina Esquivel", 1.02, 0.96, 1.06, 27, "Active"),
        ("Iker Montalvo", 0.91, 1.05, 0.94, 8, "Active"),
        ("Renata Villalobos", 0.98, 0.99, 1.00, 32, "Active"),
        ("Santiago Ruelas", 1.04, 1.02, 0.97, 10, "Active")]),
    dict(manager="Nadine Lockhart", country="CA", segment="SMB", quota=225_000, stale=0.36, reps=[
        ("Wren Tessier", 1.06, 1.04, 1.03, 25, "Active"),
        ("Oskar Lindqvist", 0.94, 0.98, 0.95, 14, "Active"),
        ("Marguerite Chow", 1.00, 0.96, 1.09, 18, "Active"),
        ("Elias Tremblay", 0.89, 0.90, 0.97, 6, "Active"),
        ("Yusra Kaddour", 1.03, 1.05, 0.98, 30, "Active")]),
]
ACCOUNT_FIRST = ["Alderway", "Brightmoor", "Cascadia", "Dunbarton", "Elmwood", "Fairhaven", "Granite Peak",
                 "Harborview", "Ironwood", "Juniper", "Kingsport", "Larkspur", "Monarch", "Northgate",
                 "Oakhurst", "Pinehollow", "Quarry Hill", "Rivendale", "Stonebridge", "Tallgrass"]
ACCOUNT_LAST = ["Freight", "Systems", "Health Group", "Broadcasting", "Grocers", "Credit Union",
                "Manufacturing", "Outfitters", "Utilities", "Biosciences", "Advisors", "Telecom"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    users, deals, acct_pool = [], [], {}

    def sf_id(prefix: str) -> str:
        """A Salesforce-looking 18-character record id."""
        chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
        return prefix + "Pg00000" + "".join(rng.choice(chars, 5)) + "QAG"

    def account() -> tuple[str, str]:
        name = f"{rng.choice(ACCOUNT_FIRST)} {rng.choice(ACCOUNT_LAST)}"
        if name in acct_pool and rng.random() < 0.7:
            return acct_pool[name], name
        acct_pool[name] = sf_id("001")            # sometimes a second id for the same account name
        return acct_pool[name], name

    def a_date(low: pd.Timestamp, high: pd.Timestamp) -> pd.Timestamp:
        return low + pd.Timedelta(days=int(rng.integers(0, max(1, (high - low).days))))

    def category_for(stage: str) -> str:
        late = stage in ("Proposal/Price Quote", "Negotiation/Review")
        weights = [0.30, 0.20, 0.36, 0.14] if late else [0.08, 0.17, 0.60, 0.15]
        return str(rng.choice(["Commit", "Best Case", "Pipeline", "Omitted"], p=weights))

    def add(rep, team, stage, amount_usd, created, close, probability, deal_type, category, loss=None):
        ccy = CURRENCY[team["country"]]
        acc_id, acc_name = account()
        amount_usd = float(round(amount_usd, 0))
        deals.append({
            "Id": sf_id("006"),
            "Name": f"{acc_name} - {rng.choice(PRODUCTS)}",
            "AccountId": acc_id,
            "Account.Name": acc_name,
            "Account.BillingCountry": team["country"],
            "Segment__c": team["segment"],
            "Owner.Name": rep,
            "Owner.Manager__c": team["manager"],
            "Type": deal_type,
            "StageName": stage,
            "ForecastCategoryName": category,
            "Probability": int(round(probability)),
            "Amount": int(round(amount_usd / FX[ccy])),
            "CurrencyIsoCode": ccy,
            "ConvertedAmount": amount_usd,
            "CreatedDate": created,
            "CloseDate": close,
            "Age_Days__c": int((SNAPSHOT - created).days),
            "Loss_Reason__c": loss,
            "NextStep": np.nan,
            "Fiscal_Quarter__c": FISCAL_QUARTER,
        })

    for team in TEAMS:
        for rep, att, h1, h2, tenure, status in team["reps"]:
            users.append({"User.Name": rep, "Manager.Name": team["manager"],
                          "Territory_Country__c": team["country"], "Segment__c": team["segment"],
                          "Quota_USD__c": team["quota"], "Tenure_Months__c": tenure,
                          "User_Status__c": status, "Attainment_Prior_Q1__c": h1,
                          "Attainment_Prior_Q2__c": h2})
            quota, vacant = team["quota"], status != "Active"
            forecast_target = att * SOFTEN * quota
            won_target = forecast_target * rng.uniform(0.38, 0.50)
            open_target = forecast_target - won_target

            # closed won (for an empty seat, the deals it closed before it emptied)
            sizes = rng.lognormal(0, 0.45, int(rng.integers(1, 3) if vacant else rng.integers(4, 9)))
            for amt in won_target * sizes / sizes.sum():
                created = a_date(QUARTER_START - pd.Timedelta(days=120), SNAPSHOT - pd.Timedelta(days=10))
                add(rep, team, "Closed Won", amt, created, a_date(max(QUARTER_START, created), SNAPSHOT),
                    100, str(rng.choice(DEAL_TYPES, p=DEAL_TYPE_MIX)), "Closed")

            # closed lost
            for _ in range(0 if vacant else int(rng.integers(0, 3))):
                created = a_date(QUARTER_START - pd.Timedelta(days=150), SNAPSHOT - pd.Timedelta(days=20))
                deal_type = str(rng.choice(DEAL_TYPES, p=DEAL_TYPE_MIX))
                reason = str(rng.choice(CHURN_REASONS if deal_type == "Renewal" else LOSS_REASONS))
                add(rep, team, "Closed Lost", rng.uniform(15_000, 70_000) * (quota / 450_000), created,
                    a_date(max(QUARTER_START, created), SNAPSHOT), 0, deal_type, "Omitted", reason)

            # open pipeline, scaled so that sum(amount x probability) matches the rep's target
            n_open = int(rng.integers(2, 4) if vacant else rng.integers(7, 13))
            stages = rng.choice(OPEN_STAGES, n_open, p=STAGE_MIX)
            probs = np.array([rng.integers(*STAGE_PROB[s]) for s in stages])
            sizes = rng.lognormal(0, 0.5, n_open)
            for stage, prob, amt in zip(stages, probs, open_target * sizes / (sizes * probs / 100).sum()):
                stale = rng.random() < team["stale"]
                created = (a_date(QUARTER_START - pd.Timedelta(days=170), QUARTER_START) if stale
                           else a_date(QUARTER_START, SNAPSHOT - pd.Timedelta(days=3)))
                close = (a_date(max(QUARTER_START, created + pd.Timedelta(days=20)), SNAPSHOT) if stale
                         else a_date(SNAPSHOT + pd.Timedelta(days=1), QUARTER_END))
                add(rep, team, str(stage), amt, created, close, prob,
                    str(rng.choice(DEAL_TYPES, p=DEAL_TYPE_MIX)), category_for(str(stage)))

    opp = pd.DataFrame(deals)
    roster = pd.DataFrame(users)

    # one deal created today and one closing on the last day of the quarter, so both dates can be
    # inferred from the data itself
    ahead = opp[opp["StageName"].isin(OPEN_STAGES) & (opp["CloseDate"] > SNAPSHOT)].index
    opp.loc[ahead[0], ["CreatedDate", "Age_Days__c"]] = [SNAPSHOT, 0]
    opp.loc[ahead[1], "CloseDate"] = QUARTER_END

    # ---- planted data-quality problems ---------------------------------------------------
    planted = {}
    near_one = opp[opp["CurrencyIsoCode"].isin(["CAD", "BRL"])]      # 1. converted amount never converted
    idx = near_one.sample(9, random_state=SEED + 1).index
    opp.loc[idx, "ConvertedAmount"] = opp.loc[idx, "Amount"].astype(float)
    planted["rows where ConvertedAmount is still local currency"] = len(idx)

    spelled = {"US": "United States", "CA": "Canada", "MX": "Mexico"}  # 2. country typed out by hand
    idx = opp[opp["Account.BillingCountry"].isin(spelled)].sample(34, random_state=SEED + 2).index
    opp.loc[idx, "Account.BillingCountry"] = opp.loc[idx, "Account.BillingCountry"].map(spelled)
    planted["rows with a spelled-out billing country"] = len(idx)

    idx = opp.sample(160, random_state=SEED + 3).index               # 3. age measured to the close date
    opp.loc[idx, "Age_Days__c"] = (opp.loc[idx, "CloseDate"] - opp.loc[idx, "CreatedDate"]).dt.days
    neg = opp.sample(3, random_state=SEED + 4).index                 #    and three impossible values
    opp.loc[neg, "Age_Days__c"] = -1
    planted["rows with a wrong Age_Days__c"] = len(idx)

    idx = opp[opp["StageName"].isin(OPEN_STAGES)].sample(24, random_state=SEED + 5).index
    opp.loc[idx, "ForecastCategoryName"] = np.nan                    # 4. values only a person can fill in
    planted["open deals with no forecast category"] = len(idx)
    lost = opp[opp["StageName"] == "Closed Lost"].sample(6, random_state=SEED + 6).index
    opp.loc[lost, "Loss_Reason__c"] = np.nan
    planted["lost deals with no loss reason"] = len(lost)

    dupes = opp.sample(6, random_state=SEED)                         # 5. the same export row twice
    opp = pd.concat([opp, dupes], ignore_index=True)
    planted["duplicate rows"] = len(dupes)

    opp = opp.sample(frac=1, random_state=SEED + 7).reset_index(drop=True)
    for col in ("CreatedDate", "CloseDate"):
        opp[col] = pd.to_datetime(opp[col]).dt.strftime("%Y-%m-%d")

    opp.to_csv(HERE / "crm_opportunities_export.csv", index=False)
    roster.to_csv(HERE / "crm_user_quotas_export.csv", index=False)
    print(f"{len(opp)} opportunity rows, {len(roster)} user rows")
    print("planted:", ", ".join(f"{v} {k}" for k, v in planted.items()))


if __name__ == "__main__":
    main()
