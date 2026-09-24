"""Generates the synthetic sample dataset used by this repo.

Nothing here comes from a real company: teams, reps, accounts and amounts are made up, and the
data-quality problems (duplicate rows, unconverted currency, spelled-out country names, a broken age
field, stale close dates, vacant seats) are planted on purpose so the diagnostic has something to find.

    python data/generate_sample_data.py     # writes sample_opportunities.csv and sample_rep_roster.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 11
QUARTER = "FY27 Q1"
QUARTER_START = pd.Timestamp("2026-08-01")
QUARTER_END = pd.Timestamp("2026-10-31")
SNAPSHOT = pd.Timestamp("2026-10-17")           # "today" in the data: 14 days before quarter end
SOFTEN = 0.955                                  # scales the whole region so it lands just under target
FX = {"AUD": 0.66, "NZD": 0.60, "SGD": 0.74, "JPY": 0.0068, "INR": 0.012, "KRW": 0.00073}
CURRENCY = {"AU": "AUD", "NZ": "NZD", "SG": "SGD", "JP": "JPY", "IN": "INR", "KR": "KRW"}
OPEN_STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation"]
STAGE_MIX = [0.10, 0.24, 0.38, 0.28]
STAGE_PROB = {"Prospecting": (0.06, 0.28), "Qualification": (0.15, 0.45),
              "Proposal": (0.25, 0.62), "Negotiation": (0.45, 0.88)}
DEAL_TYPES = ["New Business", "Renewal", "Add-on", "Pilot"]
DEAL_TYPE_MIX = [0.46, 0.28, 0.16, 0.10]
LOSS_REASONS = ["Price", "Lost to competitor", "No decision / stalled", "Timing - pushed out",
                "Budget/priority shift"]
CHURN_REASONS = ["Churned - to competitor", "Churned - budget cut", "Churned - product fit",
                 "Non-renewal - no response"]

# manager, country, segment, quota per seat, how stale that team's pipeline is, and its reps:
# (name, attainment this quarter, last quarter, two quarters ago, tenure months, headcount status)
TEAMS = [
    dict(manager="Priya Raman", country="AU", segment="ENT", quota=600_000, stale=0.45, reps=[
        ("Noel Adeyemi", 1.02, 1.05, 0.96, 26, "Active"),
        ("Ivy Chen", 1.11, 1.02, 1.08, 18, "Active"),
        ("Rafael Dias", 0.94, 0.94, 1.01, 36, "Active"),
        ("Hana Oyelaran", 0.89, 0.97, 0.93, 9, "Active")]),
    dict(manager="Daniel Okafor", country="IN", segment="MME", quota=320_000, stale=0.5, reps=[
        ("Arjun Mehta", 1.06, 1.12, 0.98, 30, "Active"),
        ("Sara Villanueva", 1.01, 0.96, 1.04, 15, "Active"),
        ("Leo Fischer", 0.98, 1.01, 0.92, 21, "Active"),
        ("OPEN REQ (Nikhil Rao backfill pending)", 0.40, 0.99, 0.88, 0, "Open Req"),
        ("OPEN REQ (Grace Lim backfill pending)", 0.33, 0.91, 1.03, 0, "Open Req")]),
    dict(manager="Mei Lin", country="JP", segment="ENT", quota=560_000, stale=0.42, reps=[
        ("Kenji Sato", 0.99, 1.03, 0.94, 22, "Active"),
        ("Yuki Tanaka", 0.85, 0.88, 0.86, 27, "Active"),
        ("Diego Moreno", 0.91, 0.93, 1.02, 12, "Active"),
        ("Amara Nwosu", 0.94, 0.99, 0.95, 8, "Active")]),
    dict(manager="Tom Baker", country="SG", segment="SMB", quota=200_000, stale=0.5, reps=[
        ("Wei Zhang", 1.08, 1.06, 1.12, 24, "Active"),
        ("Farah Aziz", 0.97, 1.01, 0.95, 10, "Active"),
        ("Oliver Grant", 0.93, 0.92, 0.97, 16, "Active"),
        ("Nadia Haddad", 1.04, 0.98, 1.06, 6, "Active"),
        ("Samuel Adeyemi", 0.99, 1.09, 0.94, 20, "Active")]),
    dict(manager="Aisha Khan", country="KR", segment="Acquisition", quota=170_000, stale=0.95, reps=[
        ("Min-Jun Park", 0.96, 1.02, 0.99, 19, "Active"),
        ("Clara Bianchi", 0.88, 0.87, 0.89, 14, "Active"),
        ("Tobias Lang", 1.03, 0.95, 1.07, 28, "Active"),
        ("Ji-Woo Han", 0.92, 1.04, 0.93, 7, "Active"),
        ("Elena Marchetti", 0.99, 0.98, 1.01, 33, "Active"),
        ("Bilal Rahman", 1.05, 1.03, 0.96, 11, "Active")]),
    dict(manager="Lucas Ferreira", country="NZ", segment="SMB", quota=190_000, stale=0.38, reps=[
        ("Ruby Callaghan", 1.07, 1.05, 1.02, 23, "Active"),
        ("Andre Silva", 0.95, 0.99, 0.94, 13, "Active"),
        ("Mia Thompson", 1.01, 0.97, 1.08, 17, "Active"),
        ("Hugo Marais", 0.90, 0.91, 0.96, 5, "Active"),
        ("Zara Iqbal", 1.02, 1.06, 0.99, 29, "Active")]),
]
ACCOUNT_FIRST = ["Northwind", "Bluepeak", "Kestrel", "Marlow", "Quantia", "Verdant", "Halcyon", "Orbital",
                 "Redwood", "Lumen", "Sable", "Crestline", "Tidewater", "Ironbark", "Nimbus", "Solstice",
                 "Aurora", "Foundry", "Copperline", "Meridian"]
ACCOUNT_LAST = ["Logistics", "Systems", "Health", "Media", "Foods", "Bank", "Industries", "Retail",
                "Energy", "Labs", "Partners", "Networks"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    roster_rows, deals, acct_pool = [], [], {}
    opp_no = 200_100

    def account() -> tuple[str, str]:
        name = f"{rng.choice(ACCOUNT_FIRST)} {rng.choice(ACCOUNT_LAST)}"
        if name in acct_pool and rng.random() < 0.7:      # same name, same id
            return acct_pool[name], name
        acct_pool[name] = f"ACC-{rng.integers(1000, 9999)}"   # sometimes a second id for the same name
        return acct_pool[name], name

    def a_date(low: pd.Timestamp, high: pd.Timestamp) -> pd.Timestamp:
        return low + pd.Timedelta(days=int(rng.integers(0, max(1, (high - low).days))))

    def category_for(stage: str) -> str:
        weights = [0.34, 0.30, 0.26, 0.10] if stage in ("Proposal", "Negotiation") else [0.12, 0.24, 0.52, 0.12]
        return str(rng.choice(["Commit", "Best Case", "Pipeline", "Omitted"], p=weights))

    def add(rep, team, stage, amount_usd, created, close, prob, deal_type, category, loss=None):
        nonlocal opp_no
        opp_no += 1
        ccy = CURRENCY[team["country"]]
        acc_id, acc_name = account()
        amount_usd = float(round(amount_usd, 0))
        deals.append({
            "opp_id": f"OPP-{opp_no}", "account_id": acc_id, "account_name": acc_name,
            "segment": team["segment"], "country": team["country"], "rep_name": rep,
            "manager_name": team["manager"], "deal_type": deal_type, "pipeline_stage": stage,
            "forecast_category": category, "created_date": created, "close_date": close,
            "age_days": int((SNAPSHOT - created).days), "amount_local": int(round(amount_usd / FX[ccy])),
            "currency": ccy, "amount_usd": amount_usd, "win_probability": round(float(prob), 2),
            "quarter": QUARTER, "note": np.nan, "loss_reason": loss})

    for team in TEAMS:
        for rep, att, h1, h2, tenure, status in team["reps"]:
            roster_rows.append({"rep_name": rep, "manager_name": team["manager"], "country": team["country"],
                                "segment": team["segment"], "quota_usd": team["quota"],
                                "tenure_months": tenure, "headcount_status": status,
                                "historical_attainment_pct_q_minus_1": h1,
                                "historical_attainment_pct_q_minus_2": h2})
            quota = team["quota"]
            forecast_target = att * SOFTEN * quota
            won_target = forecast_target * rng.uniform(0.38, 0.50)
            open_target = forecast_target - won_target

            vacant = status != "Active"

            # closed won (the seat's last deals before it emptied)
            sizes = rng.lognormal(0, 0.45, int(rng.integers(1, 3) if vacant else rng.integers(4, 9)))
            for amt in won_target * sizes / sizes.sum():
                created = a_date(QUARTER_START - pd.Timedelta(days=120), SNAPSHOT - pd.Timedelta(days=10))
                add(rep, team, "Closed Won", amt, created, a_date(max(QUARTER_START, created), SNAPSHOT),
                    1.0, str(rng.choice(DEAL_TYPES, p=DEAL_TYPE_MIX)), "Commit")

            # closed lost
            for _ in range(0 if vacant else int(rng.integers(0, 3))):
                created = a_date(QUARTER_START - pd.Timedelta(days=150), SNAPSHOT - pd.Timedelta(days=20))
                deal_type = str(rng.choice(DEAL_TYPES, p=DEAL_TYPE_MIX))
                reason = str(rng.choice(CHURN_REASONS if deal_type == "Renewal" else LOSS_REASONS))
                add(rep, team, "Closed Lost", rng.uniform(15_000, 70_000) * (quota / 450_000), created,
                    a_date(max(QUARTER_START, created), SNAPSHOT), 0.0, deal_type, reason, reason)

            # open pipeline, scaled so that sum(amount x win probability) matches the rep's target
            n_open = int(rng.integers(2, 4) if vacant else rng.integers(7, 13))
            stages = rng.choice(OPEN_STAGES, n_open, p=STAGE_MIX)
            probs = np.array([rng.uniform(*STAGE_PROB[s]) for s in stages])
            sizes = rng.lognormal(0, 0.5, n_open)
            for stage, prob, amt in zip(stages, probs, open_target * sizes / (sizes * probs).sum()):
                stale = rng.random() < team["stale"]
                created = (a_date(QUARTER_START - pd.Timedelta(days=170), QUARTER_START) if stale
                           else a_date(QUARTER_START, SNAPSHOT - pd.Timedelta(days=3)))
                close = (a_date(max(QUARTER_START, created + pd.Timedelta(days=20)), SNAPSHOT) if stale
                         else a_date(SNAPSHOT + pd.Timedelta(days=1), QUARTER_END))
                add(rep, team, str(stage), amt, created, close, prob,
                    str(rng.choice(DEAL_TYPES, p=DEAL_TYPE_MIX)), category_for(str(stage)))

    opp = pd.DataFrame(deals)
    roster = pd.DataFrame(roster_rows)

    # one deal created today and one closing on the last day of the quarter, so a reader (and the
    # toolkit) can infer the snapshot date and the quarter end from the data itself
    fresh = opp[opp["pipeline_stage"].isin(OPEN_STAGES) & (opp["close_date"] > SNAPSHOT)].index[0]
    opp.loc[fresh, ["created_date", "age_days"]] = [SNAPSHOT, 0]
    last = opp[opp["pipeline_stage"].isin(OPEN_STAGES) & (opp["close_date"] > SNAPSHOT)].index[1]
    opp.loc[last, "close_date"] = QUARTER_END

    # ---- planted data-quality problems ---------------------------------------------------
    planted = {}

    near_one = opp[opp["currency"].isin(["AUD", "NZD", "SGD"])]   # 2. currency never converted
    idx = near_one.sample(7, random_state=SEED + 1).index
    opp.loc[idx, "amount_usd"] = opp.loc[idx, "amount_local"].astype(float)
    planted["unconverted currency rows"] = len(idx)

    spelled = {"AU": "Australia", "JP": "Japan", "SG": "Singapore"}   # 3. country spelled out
    idx = opp[opp["country"].isin(spelled)].sample(31, random_state=SEED + 2).index
    opp.loc[idx, "country"] = opp.loc[idx, "country"].map(spelled)
    planted["spelled-out country rows"] = len(idx)

    idx = opp.sample(150, random_state=SEED + 3).index             # 4. age measured to the close date
    opp.loc[idx, "age_days"] = (opp.loc[idx, "close_date"] - opp.loc[idx, "created_date"]).dt.days
    neg = opp.sample(3, random_state=SEED + 4).index               #    and three impossible values
    opp.loc[neg, "age_days"] = -1
    planted["rows with a wrong age_days"] = len(idx)

    idx = opp[opp["pipeline_stage"].isin(OPEN_STAGES)].sample(22, random_state=SEED + 5).index
    opp.loc[idx, "forecast_category"] = np.nan                     # 5. missing values for a person to fix
    planted["open deals with no forecast category"] = len(idx)
    lost = opp[opp["pipeline_stage"] == "Closed Lost"].sample(5, random_state=SEED + 6).index
    opp.loc[lost, "loss_reason"] = np.nan
    planted["lost deals with no reason"] = len(lost)

    dupes = opp.sample(5, random_state=SEED)                      # 6. exact duplicate rows, added last
    opp = pd.concat([opp, dupes], ignore_index=True)              #    so both copies are identical
    planted["duplicate rows"] = len(dupes)

    opp = opp.sample(frac=1, random_state=SEED + 7).reset_index(drop=True)
    for col in ("created_date", "close_date"):
        opp[col] = pd.to_datetime(opp[col]).dt.strftime("%Y-%m-%d")

    opp.to_csv(HERE / "sample_opportunities.csv", index=False)
    roster.to_csv(HERE / "sample_rep_roster.csv", index=False)
    print(f"{len(opp)} opportunity rows, {len(roster)} roster seats")
    print("planted:", ", ".join(f"{v} {k}" for k, v in planted.items()))


if __name__ == "__main__":
    main()
