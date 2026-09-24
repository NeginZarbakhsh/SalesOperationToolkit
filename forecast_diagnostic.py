"""
Forecast Diagnostic Toolkit
===========================

Re-runnable quarter-end diagnostic for "why is the region missing its number?".

    clean  ->  validate  ->  reconcile  ->  diagnose  ->  flag actions  ->  draft summary (AI, verified)

Usage (command line):
    python forecast_diagnostic.py                      # uses config.json next to this file
    python forecast_diagnostic.py --config q4.json     # a different quarter / region
    python forecast_diagnostic.py --no-ai              # skip the AI summary step

Usage (notebook / Python):
    import forecast_diagnostic as fd
    res = fd.run("config.json")
    res.by_manager, res.dq_log, res.actions["past_due"], res.summary_text ...

Design principles
-----------------
* Never trust a rollup before validating its inputs: every fix is logged in `dq_log`
  with the rows and dollars it touched, so the reconciled number is auditable.
* The forecast is built from pipeline stage + win probability, NOT the self-reported
  forecast_category -- the category is audited separately (see `category_audit`).
* Everything that is quarter-specific (file paths, dates, FX, thresholds, stage names,
  country spellings) lives in config.json, not in code.
* The AI step only writes prose from a pre-computed facts pack, and every number it
  writes is checked back against that pack before the summary is accepted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

REQUIRED_OPP_COLS = [
    "opp_id", "account_id", "account_name", "segment", "country", "rep_name", "manager_name",
    "deal_type", "pipeline_stage", "forecast_category", "created_date", "close_date", "age_days",
    "amount_local", "currency", "amount_usd", "win_probability", "loss_reason",
]
REQUIRED_ROSTER_COLS = [
    "rep_name", "manager_name", "country", "segment", "quota_usd", "tenure_months",
    "headcount_status", "historical_attainment_pct_q_minus_1", "historical_attainment_pct_q_minus_2",
]


# --------------------------------------------------------------------------------------
# Config & loading
# --------------------------------------------------------------------------------------
def load_config(path: str | Path | None = None, **overrides) -> dict:
    """Read config.json and resolve file paths relative to the config's own folder."""
    path = Path(path) if path else HERE / "config.json"
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.update(overrides)
    base = Path(path).resolve().parent
    for key in ("opportunity_file", "roster_file", "output_dir"):
        cfg[key] = str((base / cfg[key]).resolve())
    cfg["ai_summary"]["prompt_file"] = str((base / cfg["ai_summary"]["prompt_file"]).resolve())
    return cfg


def _check_schema(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns {missing}. "
            f"If your CRM export uses different headers, rename them before running."
        )


def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    opp = pd.read_csv(cfg["opportunity_file"])
    roster = pd.read_csv(cfg["roster_file"])
    _check_schema(opp, REQUIRED_OPP_COLS, "opportunity file")
    _check_schema(roster, REQUIRED_ROSTER_COLS, "roster file")
    opp["created_date"] = pd.to_datetime(opp["created_date"], errors="coerce")
    opp["close_date"] = pd.to_datetime(opp["close_date"], errors="coerce")
    return opp, roster


def infer_dates(opp: pd.DataFrame, cfg: dict) -> dict:
    """Snapshot = latest created_date in the extract; quarter end = latest close_date."""
    out = {}
    snap = cfg.get("snapshot_date", "auto")
    out["snapshot"] = opp["created_date"].max().normalize() if snap == "auto" else pd.Timestamp(snap)
    qend = cfg.get("quarter_end", "auto")
    out["quarter_end"] = opp["close_date"].max().normalize() if qend == "auto" else pd.Timestamp(qend)
    qstart = cfg.get("quarter_start", "auto")
    out["quarter_start"] = (
        (out["quarter_end"] - pd.offsets.MonthBegin(3)).normalize() if qstart == "auto" else pd.Timestamp(qstart)
    )
    out["days_left"] = int((out["quarter_end"] - out["snapshot"]).days)
    return out


def infer_fx(opp: pd.DataFrame, cfg: dict) -> dict:
    """Dominant USD rate per currency (median of amount_usd / amount_local), unless given in config."""
    if isinstance(cfg.get("fx_to_usd"), dict):
        return {k: float(v) for k, v in cfg["fx_to_usd"].items()}
    ratio = (opp["amount_usd"] / opp["amount_local"]).replace([np.inf, -np.inf], np.nan)
    return ratio.groupby(opp["currency"]).median().round(4).to_dict()


# --------------------------------------------------------------------------------------
# Validation & cleaning
# --------------------------------------------------------------------------------------
class DQLog:
    """Collects one row per data-quality check so every fix is auditable."""

    def __init__(self):
        self.rows = []

    def add(self, check_id, check, severity, action, rows, usd=0.0, ids=None, detail=""):
        ids = list(ids)[:8] if ids is not None else []
        self.rows.append({
            "check_id": check_id, "check": check, "severity": severity, "action": action,
            "rows_affected": int(rows), "usd_affected": float(round(usd, 0)),
            "example_ids": ", ".join(map(str, ids)), "detail": detail,
        })

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        order = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}
        return df.sort_values(["severity", "check_id"], key=lambda s: s.map(order) if s.name == "severity" else s,
                              ignore_index=True)


def clean_and_validate(opp_raw: pd.DataFrame, roster: pd.DataFrame, cfg: dict, dates: dict, fx: dict):
    log = DQLog()
    st = cfg["stages"]
    th = cfg["thresholds"]
    opp = opp_raw.copy()
    snap = dates["snapshot"]

    # DQ01 exact duplicate rows -------------------------------------------------------
    exact = opp.duplicated(keep="first")
    if exact.any():
        log.add("DQ01", "Exact duplicate opportunity rows", "High", "Removed", exact.sum(),
                opp.loc[exact, "amount_usd"].sum(), opp.loc[exact, "opp_id"],
                "Same opp_id and identical values - would double-count pipeline and bookings.")
    opp = opp.loc[~exact].copy()

    # DQ02 conflicting duplicates (same id, different values) -> keep first, flag ------
    conflict = opp.duplicated(subset="opp_id", keep="first")
    if conflict.any():
        log.add("DQ02", "Duplicate opp_id with conflicting values", "High", "Removed (kept first) - review",
                conflict.sum(), opp.loc[conflict, "amount_usd"].sum(), opp.loc[conflict, "opp_id"],
                "Same opp_id, different field values - needs a CRM owner to decide which is correct.")
    opp = opp.loc[~conflict].copy()

    # DQ03 FX conversion ---------------------------------------------------------------
    expected_rate = opp["currency"].map(fx)
    actual_rate = opp["amount_usd"] / opp["amount_local"]
    bad_fx = (actual_rate - expected_rate).abs() > cfg.get("fx_tolerance", 0.02)
    if bad_fx.any():
        before = opp.loc[bad_fx, "amount_usd"].sum()
        opp.loc[bad_fx, "amount_usd"] = (opp.loc[bad_fx, "amount_local"] * expected_rate[bad_fx]).round(0)
        after = opp.loc[bad_fx, "amount_usd"].sum()
        log.add("DQ03", "amount_usd not converted at the currency's rate", "High", "Corrected (recomputed)",
                bad_fx.sum(), after - before, opp.loc[bad_fx, "opp_id"],
                f"Rates used: {fx}. Typically local amount copied into the USD field unconverted.")
    unknown_ccy = expected_rate.isna()
    if unknown_ccy.any():
        log.add("DQ04", "Currency with no known USD rate", "High", "Flagged", unknown_ccy.sum(),
                opp.loc[unknown_ccy, "amount_usd"].sum(), opp.loc[unknown_ccy, "opp_id"])

    # DQ05 country naming ------------------------------------------------------------
    valid = set(roster["country"].dropna().unique())
    off = ~opp["country"].isin(valid)
    if off.any():
        raw_vals = opp.loc[off, "country"].value_counts().to_dict()
        opp.loc[off, "country"] = opp.loc[off, "country"].map(cfg.get("country_aliases", {})).fillna(opp.loc[off, "country"])
        still = ~opp["country"].isin(valid)
        rep_country = opp["rep_name"].map(roster.set_index("rep_name")["country"])
        opp.loc[still & rep_country.notna(), "country"] = rep_country[still & rep_country.notna()]
        log.add("DQ05", "Country spelled out instead of roster ISO code", "Medium", "Corrected (mapped)",
                off.sum(), opp.loc[off, "amount_usd"].sum(), opp.loc[off, "opp_id"],
                f"Values found: {raw_vals}. Unfixed, these rows fall into a phantom country with zero quota.")
        unresolved = ~opp["country"].isin(valid)
        if unresolved.any():
            log.add("DQ06", "Country could not be mapped to roster", "High", "Flagged", unresolved.sum(),
                    opp.loc[unresolved, "amount_usd"].sum(), opp.loc[unresolved, "opp_id"])

    # DQ07 rep / hierarchy consistency with roster --------------------------------------
    r = roster.set_index("rep_name")
    not_on_roster = ~opp["rep_name"].isin(r.index)
    if not_on_roster.any():
        log.add("DQ07", "Opportunity owner not on roster", "High", "Flagged", not_on_roster.sum(),
                opp.loc[not_on_roster, "amount_usd"].sum(), opp.loc[not_on_roster, "opp_id"])
    for col in ("manager_name", "segment", "country"):
        mism = opp["rep_name"].isin(r.index) & (opp["rep_name"].map(r[col]) != opp[col])
        if mism.any():
            log.add("DQ08", f"{col} on opportunity differs from roster", "Medium", "Flagged", mism.sum(),
                    opp.loc[mism, "amount_usd"].sum(), opp.loc[mism, "opp_id"])

    # DQ09 vacant seats: quota with nobody behind it, and live deals with no owner -------
    vacant = roster[roster["headcount_status"] != cfg["active_headcount_status"]]
    if len(vacant):
        log.add("DQ09", "Quota assigned to vacant seats (Open Req)", "High", "Flagged", len(vacant),
                vacant["quota_usd"].sum(), vacant["rep_name"],
                "Quota with no active rep - a coverage gap, not a performance gap.")
        orphan = opp["rep_name"].isin(vacant["rep_name"]) & opp["pipeline_stage"].isin(st["open"])
        if orphan.any():
            log.add("DQ10", "Open deals owned by a vacant seat", "High", "Flagged - reassign", orphan.sum(),
                    opp.loc[orphan, "amount_usd"].sum(), opp.loc[orphan, "opp_id"],
                    "Live pipeline that nobody is actively working.")
        hist = vacant[["historical_attainment_pct_q_minus_1", "historical_attainment_pct_q_minus_2"]].notna().any(axis=1)
        if hist.any():
            log.add("DQ11", "Vacant seat still shows historical attainment", "Low", "Flagged - exclude from rep trends",
                    hist.sum(), 0, vacant.loc[hist, "rep_name"],
                    "History belongs to the departed rep; exclude from performance analysis.")

    # DQ12 forecast category blank / inconsistent with stage ----------------------------
    fc = opp["forecast_category"]
    blank = fc.isna()
    if blank.any():
        log.add("DQ12", "Forecast category blank", "Medium", "Flagged", blank.sum(),
                opp.loc[blank, "amount_usd"].sum(), opp.loc[blank, "opp_id"])
    is_open = opp["pipeline_stage"].isin(st["open"])
    commit_early = is_open & (fc == "Commit") & opp["pipeline_stage"].isin(st["early"])
    if commit_early.any():
        log.add("DQ13", "'Commit' on an early-stage deal", "Medium", "Flagged - not used in forecast",
                commit_early.sum(), opp.loc[commit_early, "amount_usd"].sum(), opp.loc[commit_early, "opp_id"],
                "Self-reported category contradicts pipeline stage.")
    omitted_late = is_open & (fc.isna() | (fc == "Omitted")) & opp["pipeline_stage"].isin(st["late"])
    if omitted_late.any():
        log.add("DQ14", "Late-stage deal marked Omitted / blank", "Medium", "Flagged - not used in forecast",
                omitted_late.sum(), opp.loc[omitted_late, "amount_usd"].sum(), opp.loc[omitted_late, "opp_id"])

    # DQ15 dates & age --------------------------------------------------------------
    recomputed_age = (snap - opp["created_date"]).dt.days
    bad_age = (opp["age_days"] < 0) | (is_open & (opp["age_days"] != recomputed_age))
    if bad_age.any():
        log.add("DQ15", "age_days inconsistent with dates (incl. negatives)", "Medium",
                "Corrected (recomputed from created_date)", bad_age.sum(), opp.loc[bad_age, "amount_usd"].sum(),
                opp.loc[bad_age, "opp_id"],
                "Field mixes 'age at snapshot' and 'close minus created'. Do not use for stall analysis.")
    bad_order = opp["close_date"] < opp["created_date"]
    if bad_order.any():
        log.add("DQ16", "Close date before created date", "Medium", "Flagged", bad_order.sum(),
                opp.loc[bad_order, "amount_usd"].sum(), opp.loc[bad_order, "opp_id"])
    past_due = is_open & (opp["close_date"] < snap)
    if past_due.any():
        log.add("DQ17", "Open deal with a close date already in the past", "High",
                "Flagged - confirm or push close date", past_due.sum(), opp.loc[past_due, "amount_usd"].sum(),
                opp.loc[past_due, "opp_id"],
                "Stale close dates inflate coverage; win probability on these is likely overstated.")

    # DQ18 win probability sanity -------------------------------------------------------
    wp = opp["win_probability"]
    wp_bad = ((opp["pipeline_stage"] == st["won"]) & (wp != 1)) | ((opp["pipeline_stage"] == st["lost"]) & (wp != 0)) \
        | (is_open & ((wp <= 0) | (wp >= 1)))
    if wp_bad.any():
        log.add("DQ18", "Win probability inconsistent with stage", "Medium", "Flagged", wp_bad.sum(),
                opp.loc[wp_bad, "amount_usd"].sum(), opp.loc[wp_bad, "opp_id"])

    # DQ19 closed-lost without a reason -----------------------------------------------
    lost_no_reason = (opp["pipeline_stage"] == st["lost"]) & opp["loss_reason"].isna()
    if lost_no_reason.any():
        log.add("DQ19", "Closed Lost with no loss reason", "Low", "Flagged", lost_no_reason.sum(),
                opp.loc[lost_no_reason, "amount_usd"].sum(), opp.loc[lost_no_reason, "opp_id"],
                "Limits the loss analysis.")

    # DQ20 account master & empty columns (informational) ------------------------------
    multi = opp.groupby("account_name")["account_id"].nunique()
    multi = multi[multi > 1]
    if len(multi):
        log.add("DQ20", "Same account_name under several account_ids", "Info", "Noted", len(multi), 0, multi.index,
                "Account master needs de-duplication before any account-level analysis.")
    empty_cols = [c for c in opp_raw.columns if opp_raw[c].isna().all()]
    if empty_cols:
        log.add("DQ21", "Column entirely empty", "Info", "Ignored", len(empty_cols), 0, empty_cols)

    opp["age_at_snapshot"] = recomputed_age
    return opp, log.frame()


# --------------------------------------------------------------------------------------
# Enrichment & analysis
# --------------------------------------------------------------------------------------
def enrich(opp: pd.DataFrame, roster: pd.DataFrame, cfg: dict, dates: dict) -> pd.DataFrame:
    st = cfg["stages"]
    o = opp.copy()
    o["is_open"] = o["pipeline_stage"].isin(st["open"])
    o["is_won"] = o["pipeline_stage"] == st["won"]
    o["is_lost"] = o["pipeline_stage"] == st["lost"]
    o["won_usd"] = np.where(o["is_won"], o["amount_usd"], 0.0)
    o["lost_usd"] = np.where(o["is_lost"], o["amount_usd"], 0.0)
    o["open_usd"] = np.where(o["is_open"], o["amount_usd"], 0.0)
    o["weighted_open_usd"] = np.where(o["is_open"], o["amount_usd"] * o["win_probability"], 0.0)
    o["forecast_usd"] = o["won_usd"] + o["weighted_open_usd"]
    o["past_due"] = o["is_open"] & (o["close_date"] < dates["snapshot"])
    o["days_past_due"] = np.where(o["past_due"], (dates["snapshot"] - o["close_date"]).dt.days, 0)
    o["past_due_weighted_usd"] = np.where(o["past_due"], o["weighted_open_usd"], 0.0)
    o["cohort"] = np.where(o["created_date"] < dates["quarter_start"], "Created before quarter", "Created in quarter")
    o["headcount_status"] = o["rep_name"].map(roster.set_index("rep_name")["headcount_status"])
    o["vacant_seat"] = o["headcount_status"] != cfg["active_headcount_status"]
    o["forecast_category_clean"] = o["forecast_category"].fillna("(blank)")
    return o


def reconcile(opp_clean: pd.DataFrame, opp_raw: pd.DataFrame, roster: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compare every plausible way of rolling up 'the forecast' against quota."""
    st = cfg["stages"]
    quota = roster["quota_usd"].sum()

    def methods(o):
        won = o.loc[o["pipeline_stage"] == st["won"], "amount_usd"].sum()
        op = o[o["pipeline_stage"].isin(st["open"])]
        cat = op["forecast_category"]
        return {
            "Closed Won only": won,
            "Won + open 'Commit' (category)": won + op.loc[cat == "Commit", "amount_usd"].sum(),
            "Won + open 'Commit' + 'Best Case' (category)": won + op.loc[cat.isin(["Commit", "Best Case"]), "amount_usd"].sum(),
            "Won + open pipeline x win probability (stage-based)": won + (op["amount_usd"] * op["win_probability"]).sum(),
        }

    raw, clean = methods(opp_raw), methods(opp_clean)
    df = pd.DataFrame({"method": list(clean), "raw_usd": list(raw.values()), "clean_usd": list(clean.values())})
    df["raw_pct_of_quota"] = df["raw_usd"] / quota
    df["clean_pct_of_quota"] = df["clean_usd"] / quota
    rep = cfg.get("reported_forecast_pct")
    if rep:
        df["matches_reported"] = (df["clean_pct_of_quota"] - rep).abs() < 0.01
    df["uses_forecast_category"] = df["method"].str.contains("category")
    return df


def category_audit(o: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Does the self-reported category carry signal? Compare it with stage-based win probability."""
    op = o[o["is_open"]]
    th = cfg["thresholds"]
    g = op.groupby("forecast_category_clean")
    df = g.agg(deals=("opp_id", "count"), open_usd=("amount_usd", "sum"), weighted_usd=("weighted_open_usd", "sum"),
               avg_win_prob=("win_probability", "mean"))
    early = op["pipeline_stage"].isin(cfg["stages"]["early"])
    df["share_in_early_stage"] = early.groupby(op["forecast_category_clean"]).mean()
    df["share_past_due"] = op["past_due"].groupby(op["forecast_category_clean"]).mean()
    order = ["Commit", "Best Case", "Pipeline", "Omitted", "(blank)"]
    return df.reindex([c for c in order if c in df.index] + [c for c in df.index if c not in order])


def gap_by(o: pd.DataFrame, roster: pd.DataFrame, dim: str, cfg: dict) -> pd.DataFrame:
    active = roster["headcount_status"] == cfg["active_headcount_status"]
    q = roster.assign(vacant_quota=np.where(active, 0, roster["quota_usd"])).groupby(dim).agg(
        quota_usd=("quota_usd", "sum"), vacant_quota_usd=("vacant_quota", "sum"), seats=("rep_name", "count"),
        vacant_seats=("headcount_status", lambda s: int((s != cfg["active_headcount_status"]).sum())))
    f = o.groupby(dim).agg(
        won_usd=("won_usd", "sum"), weighted_open_usd=("weighted_open_usd", "sum"), forecast_usd=("forecast_usd", "sum"),
        open_pipeline_usd=("open_usd", "sum"), lost_usd=("lost_usd", "sum"),
        past_due_weighted_usd=("past_due_weighted_usd", "sum"), open_deals=("is_open", "sum"),
        past_due_deals=("past_due", "sum"))
    df = q.join(f, how="outer").fillna(0)
    df["attainment"] = df["forecast_usd"] / df["quota_usd"].replace(0, np.nan)
    df["gap_usd"] = df["forecast_usd"] - df["quota_usd"]
    total_gap = df["gap_usd"].clip(upper=0).sum()
    df["share_of_gap"] = df["gap_usd"].clip(upper=0) / total_gap if total_gap else 0
    df["coverage_of_remaining"] = df["open_pipeline_usd"] / (df["quota_usd"] - df["won_usd"]).replace(0, np.nan)
    df["past_due_share_of_weighted"] = df["past_due_weighted_usd"] / df["weighted_open_usd"].replace(0, np.nan)
    closed = df["won_usd"] + df["lost_usd"]
    df["win_rate_usd"] = df["won_usd"] / closed.replace(0, np.nan)
    return df.sort_values("gap_usd")


def rep_table(o: pd.DataFrame, roster: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    th = cfg["thresholds"]
    f = o.groupby("rep_name").agg(won_usd=("won_usd", "sum"), forecast_usd=("forecast_usd", "sum"),
                                  open_deals=("is_open", "sum"), past_due_deals=("past_due", "sum"),
                                  lost_usd=("lost_usd", "sum"))
    df = roster.set_index("rep_name").join(f).fillna({"won_usd": 0, "forecast_usd": 0, "open_deals": 0,
                                                        "past_due_deals": 0, "lost_usd": 0})
    df["attainment"] = df["forecast_usd"] / df["quota_usd"]
    df["gap_usd"] = df["forecast_usd"] - df["quota_usd"]
    active = df["headcount_status"] == cfg["active_headcount_status"]
    df["chronic_underperformer"] = active & (df["historical_attainment_pct_q_minus_1"] < th["chronic_underperformance"]) \
        & (df["historical_attainment_pct_q_minus_2"] < th["chronic_underperformance"])
    df["three_quarters_below"] = df["chronic_underperformer"] & (df["attainment"] < th["chronic_underperformance"])
    return df.sort_values("attainment")


def pipeline_health(o: pd.DataFrame, cfg: dict) -> dict:
    op = o[o["is_open"]]
    order = cfg["stages"]["open"]
    by_stage = op.groupby("pipeline_stage").agg(
        deals=("opp_id", "count"), open_usd=("amount_usd", "sum"), weighted_usd=("weighted_open_usd", "sum"),
        avg_win_prob=("win_probability", "mean"), past_due_deals=("past_due", "sum"),
        past_due_weighted_usd=("past_due_weighted_usd", "sum"), median_age_days=("age_at_snapshot", "median"),
    ).reindex(order)
    by_cohort = op.groupby("cohort").agg(deals=("opp_id", "count"), open_usd=("amount_usd", "sum"),
                                         weighted_usd=("weighted_open_usd", "sum"), past_due_deals=("past_due", "sum"))
    by_mgr_stage = pd.crosstab(op["manager_name"], op["pipeline_stage"], values=op["amount_usd"], aggfunc="sum") \
        .reindex(columns=order).fillna(0)
    return {"by_stage": by_stage, "by_cohort": by_cohort, "by_manager_stage": by_mgr_stage}


def action_lists(o: pd.DataFrame, roster: pd.DataFrame, cfg: dict) -> dict:
    """Deal- and rep-level lists a manager can act on this week."""
    th = cfg["thresholds"]
    st = cfg["stages"]
    cols = ["opp_id", "account_name", "rep_name", "manager_name", "country", "segment", "pipeline_stage",
            "forecast_category_clean", "close_date", "days_past_due", "age_at_snapshot", "amount_usd",
            "win_probability", "weighted_open_usd"]
    op = o[o["is_open"]]
    out = {}
    out["past_due"] = op[op["past_due"]].sort_values("weighted_open_usd", ascending=False)[cols]
    out["orphaned_deals"] = op[op["vacant_seat"]].sort_values("weighted_open_usd", ascending=False)[cols]
    out["stalled_early_stage"] = op[op["pipeline_stage"].isin(st["early"]) &
                                    (op["age_at_snapshot"] >= th["stalled_age_days"])] \
        .sort_values("age_at_snapshot", ascending=False)[cols]
    fc = op["forecast_category_clean"]
    reason = pd.Series("", index=op.index)
    reason[(fc == "Commit") & op["pipeline_stage"].isin(st["early"])] = "Commit but early stage"
    reason[(fc == "Commit") & (op["win_probability"] < th["commit_min_win_prob"]) & (reason == "")] = "Commit but low win probability"
    reason[fc.isin(["Omitted", "(blank)"]) & op["pipeline_stage"].isin(st["late"])] = "Late stage but Omitted/blank"
    reason[fc.isin(["Pipeline", "Omitted", "(blank)"]) & (op["win_probability"] >= th["uncommitted_high_win_prob"])
           & (reason == "")] = "High win probability but not called"
    out["category_fixes"] = op.assign(issue=reason)[reason != ""][["issue"] + cols].sort_values(["issue", "amount_usd"],
                                                                                                ascending=[True, False])
    # concentration: share of each team's weighted open pipeline held by its top rep
    w = op.groupby(["manager_name", "rep_name"])["weighted_open_usd"].sum().reset_index()
    w["share_of_team"] = w["weighted_open_usd"] / w.groupby("manager_name")["weighted_open_usd"].transform("sum")
    top = w.sort_values("share_of_team", ascending=False).groupby("manager_name").head(1)
    out["concentration"] = top[top["share_of_team"] >= th["rep_concentration_share"]].sort_values("share_of_team",
                                                                                                    ascending=False)
    lost = o[o["is_lost"]]
    out["losses"] = lost.assign(loss_reason=lost["loss_reason"].fillna("(blank)")).pivot_table(
        index=["manager_name", "deal_type"], columns="loss_reason", values="amount_usd", aggfunc="sum", fill_value=0)
    return out


def scenarios(o: pd.DataFrame, roster: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    quota = roster["quota_usd"].sum()
    won = o["won_usd"].sum()
    w_current = o.loc[~o["past_due"], "weighted_open_usd"].sum()
    w_pd = o["past_due_weighted_usd"].sum()
    h = cfg["thresholds"]["past_due_haircut"]
    op = o[o["is_open"]]
    rows = [
        ("Stage-weighted forecast (the reconciled number)", won + w_current + w_pd,
         "Closed Won + every open deal x its win probability"),
        (f"Risk-adjusted: past-due deals at {int((1 - h) * 100)}% of stated probability", won + w_current + w_pd * (1 - h),
         "Deals that already missed their close date are unlikely to close at their stated odds"),
        ("Downside: past-due deals slip out entirely", won + w_current, "Only deals with a current close date count"),
        ("Rep call: Won + 'Commit' category", won + op.loc[op["forecast_category"] == "Commit", "amount_usd"].sum(),
         "What the self-reported category implies"),
    ]
    df = pd.DataFrame(rows, columns=["scenario", "forecast_usd", "assumption"])
    df["pct_of_quota"] = df["forecast_usd"] / quota
    df["gap_usd"] = df["forecast_usd"] - quota
    return df


def gap_bridge(o: pd.DataFrame, roster: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Quota -> forecast walk, one step per team, with vacant seats split out as their own step."""
    active_status = cfg["active_headcount_status"]
    seat_type = np.where(roster["headcount_status"] == active_status, "active reps", "vacant seats")
    rq = roster.assign(seat=seat_type).groupby(["manager_name", "seat"])["quota_usd"].sum()
    of = o.assign(seat=np.where(o["vacant_seat"], "vacant seats", "active reps")) \
        .groupby(["manager_name", "seat"])["forecast_usd"].sum()
    df = pd.concat([rq, of], axis=1).fillna(0)
    df["gap_usd"] = df["forecast_usd"] - df["quota_usd"]
    df = df.reset_index()
    team = roster.groupby("manager_name")[["country", "segment"]].first()
    df["label"] = df.apply(lambda r: f"{r.manager_name} ({team.loc[r.manager_name, 'country']} "
                                     f"{team.loc[r.manager_name, 'segment']})"
                                     + (" - vacant seats" if r.seat == "vacant seats" else ""), axis=1)
    has_vacancy = df.groupby("manager_name")["seat"].transform(lambda s: (s == "vacant seats").any())
    df.loc[has_vacancy & (df["seat"] == "active reps"), "label"] += " - active reps"
    return df.sort_values("gap_usd")[["label", "manager_name", "seat", "quota_usd", "forecast_usd", "gap_usd"]]


# --------------------------------------------------------------------------------------
# Facts pack, AI summary and number verification
# --------------------------------------------------------------------------------------
def _r(x, nd=4):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    return x


def build_facts(res: "Results") -> dict:
    """Everything the summary is allowed to say, as plain numbers. Single source of truth for the AI step."""
    o, r, cfg = res.opp, res.roster, res.cfg
    quota = r["quota_usd"].sum()
    fc = res.scenarios.iloc[0]["forecast_usd"]
    teams = res.by_manager.reset_index()
    ca = res.category_audit
    vac = r[r["headcount_status"] != cfg["active_headcount_status"]]
    orphan = res.actions["orphaned_deals"]
    op = o[o["is_open"]]
    facts = {
        "region": cfg["region_name"],
        "snapshot_date": str(res.dates["snapshot"].date()),
        "quarter_end": str(res.dates["quarter_end"].date()),
        "days_left_in_quarter": res.dates["days_left"],
        "action_horizons_days": [14, 30, 60],
        "thresholds": cfg["thresholds"],
        "total_quota_usd": _r(quota, 0),
        "closed_won_usd": _r(o["won_usd"].sum(), 0),
        "weighted_open_pipeline_usd": _r(o["weighted_open_usd"].sum(), 0),
        "forecast_usd": _r(fc, 0),
        "forecast_pct_of_quota": _r(fc / quota),
        "gap_usd": _r(fc - quota, 0),
        "reported_forecast_pct": cfg.get("reported_forecast_pct"),
        "open_pipeline_usd": _r(o["open_usd"].sum(), 0),
        "coverage_of_remaining_quota_x": _r(o["open_usd"].sum() / (quota - o["won_usd"].sum()), 2),
        "data_quality": {
            "checks_triggered": int(len(res.dq_log)),
            "high_severity_checks": int((res.dq_log["severity"] == "High").sum()),
            "duplicate_rows_removed": int(res.dq_log.loc[res.dq_log.check_id.isin(["DQ01", "DQ02"]), "rows_affected"].sum()),
            "fx_rows_corrected": int(res.dq_log.loc[res.dq_log.check_id == "DQ03", "rows_affected"].sum()),
            "country_rows_remapped": int(res.dq_log.loc[res.dq_log.check_id == "DQ05", "rows_affected"].sum()),
        },
        "rollup_methods_pct_of_quota": {m: _r(p) for m, p in zip(res.reconciliation["method"],
                                                                 res.reconciliation["clean_pct_of_quota"])},
        "teams": [
            {"manager": t.manager_name, "country": t.country, "segment": t.segment, "quota_usd": _r(t.quota_usd, 0),
             "forecast_usd": _r(t.forecast_usd, 0), "attainment": _r(t.attainment), "gap_usd": _r(t.gap_usd, 0),
             "share_of_gap": _r(t.share_of_gap), "vacant_seats": int(t.vacant_seats),
             "vacant_quota_usd": _r(t.vacant_quota_usd, 0), "past_due_share_of_weighted": _r(t.past_due_share_of_weighted),
             "win_rate_usd": _r(t.win_rate_usd), "lost_usd": _r(t.lost_usd, 0)}
            for t in teams.itertuples()
        ],
        "vacancy": {
            "vacant_seats": int(len(vacant := vac)),
            "vacant_quota_usd": _r(vacant["quota_usd"].sum(), 0),
            "forecast_on_vacant_seats_usd": _r(o.loc[o["vacant_seat"], "forecast_usd"].sum(), 0),
            "gap_on_vacant_seats_usd": _r(o.loc[o["vacant_seat"], "forecast_usd"].sum() - vacant["quota_usd"].sum(), 0),
            "orphaned_open_deals": int(len(orphan)),
            "orphaned_open_usd": _r(orphan["amount_usd"].sum(), 0),
            "orphaned_weighted_usd": _r(orphan["weighted_open_usd"].sum(), 0),
        },
        "pipeline_hygiene": {
            "open_deals": int(len(op)),
            "past_due_deals": int(op["past_due"].sum()),
            "past_due_share_of_deals": _r(op["past_due"].mean()),
            "past_due_open_usd": _r(op.loc[op["past_due"], "amount_usd"].sum(), 0),
            "past_due_weighted_usd": _r(op["past_due_weighted_usd"].sum(), 0),
            "past_due_share_of_weighted_pipeline": _r(op["past_due_weighted_usd"].sum() / op["weighted_open_usd"].sum()),
            "past_due_share_of_forecast": _r(op["past_due_weighted_usd"].sum() / fc),
            "median_days_past_due": _r(op.loc[op["past_due"], "days_past_due"].median(), 0),
            "stalled_early_stage_deals": int(len(res.actions["stalled_early_stage"])),
        },
        "forecast_category": {
            "avg_win_prob_by_category": {k: _r(v, 3) for k, v in ca["avg_win_prob"].items()},
            "commit_on_early_stage_deals": int(((op["forecast_category"] == "Commit") &
                                                op["pipeline_stage"].isin(cfg["stages"]["early"])).sum()),
            "commit_on_early_stage_usd": _r(op.loc[(op["forecast_category"] == "Commit") &
                                                   op["pipeline_stage"].isin(cfg["stages"]["early"]), "amount_usd"].sum(), 0),
            "late_stage_omitted_or_blank_deals": int((op["forecast_category_clean"].isin(["Omitted", "(blank)"]) &
                                                      op["pipeline_stage"].isin(cfg["stages"]["late"])).sum()),
            "late_stage_omitted_or_blank_usd": _r(op.loc[op["forecast_category_clean"].isin(["Omitted", "(blank)"]) &
                                                         op["pipeline_stage"].isin(cfg["stages"]["late"]), "amount_usd"].sum(), 0),
        },
        "scenarios": [{"scenario": s.scenario, "forecast_usd": _r(s.forecast_usd, 0), "pct_of_quota": _r(s.pct_of_quota)}
                      for s in res.scenarios.itertuples()],
        "reps_below_threshold_prior_two_quarters": [
            {"rep": n, "manager": row.manager_name, "attainment": _r(row.attainment),
             "q_minus_1": _r(row.historical_attainment_pct_q_minus_1), "q_minus_2": _r(row.historical_attainment_pct_q_minus_2)}
            for n, row in res.by_rep[res.by_rep["chronic_underperformer"]].iterrows()
        ],
        "rep_concentration": [
            {"manager": c.manager_name, "rep": c.rep_name, "share_of_team_weighted_pipeline": _r(c.share_of_team)}
            for c in res.actions["concentration"].itertuples()
        ],
        "lost_by_manager_and_type_usd": {
            f"{m} | {t}": _r(v, 0) for (m, t), v in res.actions["losses"].sum(axis=1).items() if v > 0
        },
    }
    # Derived values a writer legitimately needs. Pre-computed here so the verifier can stay strict.
    gaps = sorted(facts["teams"], key=lambda t: t["gap_usd"])
    facts["derived"] = {
        "top2_teams_share_of_gap": _r(sum(t["share_of_gap"] for t in gaps[:2])),
        "vacant_seats_share_of_gap": _r(facts["vacancy"]["gap_on_vacant_seats_usd"] / facts["gap_usd"]),
        "active_reps_attainment_in_teams_with_vacancies": _r(
            o.loc[~o["vacant_seat"] & o["manager_name"].isin(vac["manager_name"]), "forecast_usd"].sum()
            / r.loc[(r["headcount_status"] == cfg["active_headcount_status"]) & r["manager_name"].isin(vac["manager_name"]),
                    "quota_usd"].sum()) if len(vac) else None,
        "total_lost_usd": _r(o["lost_usd"].sum(), 0),
        "past_due_negotiation_weighted_usd": _r(op.loc[op["past_due"] & (op["pipeline_stage"] == "Negotiation"),
                                                       "weighted_open_usd"].sum(), 0),
        "past_due_negotiation_deals": int((op["past_due"] & (op["pipeline_stage"] == "Negotiation")).sum()),
    }
    return facts


_NUM = re.compile(r"(?P<cur>\$)?(?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?(?P<suf>%|[KkMm](?![a-z])|x(?![a-z]))?")


def _flatten_numbers(obj) -> list[float]:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _flatten_numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    return out


_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
_DATES = re.compile(rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\bFY\d{{2}}\b|\bQ[1-4]\b|\b\d{{1,2}}\s{_MONTH}\b|\b{_MONTH}\s\d{{1,2}}\b"
                    r"|(?<![\d$.,])20\d{2}(?!\d|%|[KkMm]\b|[.,]\d)"
                    r"|\b[A-Z]{2,5}-\d+\b"                      # record IDs such as OPP-100185, ACC-1234
                    r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")       # dates such as 7/31 or 31/07/2026


def verify_numbers(text: str, facts: dict) -> pd.DataFrame:
    """Check every number in AI-written text against the facts pack, allowing only display rounding.

    Type-aware: a percentage must match a fraction in the pack, a $ amount a money value, a bare
    number a count. Derived values are NOT inferred here - if the summary needs one (e.g. the share
    of the gap held by the top two teams) it must be pre-computed in build_facts(). Dates are ignored.
    """
    vals = np.array(sorted(set(_flatten_numbers(facts))))
    frac = vals[np.abs(vals) <= 2]
    money = np.abs(vals[np.abs(vals) >= 1000])
    counts = np.abs(vals[(vals == np.round(vals)) & (np.abs(vals) < 1000)])
    rows = []
    for m in _NUM.finditer(_DATES.sub(" ", text)):
        raw, num, cur, suf = m.group(0).strip(), m.group("num"), m.group("cur"), (m.group("suf") or "")
        decimals = len(num.split(".")[1]) if "." in num else 0
        val = float(num.replace(",", ""))
        if not cur and not suf and val <= 12 and decimals == 0:
            continue  # small counts like "two teams", "3 drivers", list numbering
        mult = {"k": 1e3, "m": 1e6}.get(suf.lower(), 1)
        if suf == "%":
            target, tol, pool = val / 100, (0.5 * 10 ** -decimals) / 100 + 1e-9, frac
        elif cur or mult > 1:
            target, tol, pool = val * mult, 0.5 * 10 ** -decimals * mult + 1e-9, money
        elif suf == "x":
            target, tol, pool = val, 0.5 * 10 ** -decimals + 1e-9, vals[np.abs(vals) < 100]
        else:
            target, tol, pool = val, 0.5 * 10 ** -decimals + 1e-9, np.concatenate([counts, money])
        ok = bool(len(pool)) and bool((np.abs(pool - target) <= tol).any())
        rows.append({"number_in_text": raw, "value": target, "verified": ok})
    return pd.DataFrame(rows, columns=["number_in_text", "value", "verified"])


def action_plan(facts: dict) -> dict:
    """Recommended actions by horizon ('14', '30', '60'), each {"action", "owner"}. Only includes what the data supports."""
    v, ph, fcat, d = facts["vacancy"], facts["pipeline_hygiene"], facts["forecast_category"], facts["derived"]
    m = lambda x: f"${abs(x) / 1e6:.2f}M" if abs(x) >= 1e6 else f"${abs(x) / 1e3:,.0f}K"
    p = lambda x: f"{x * 100:.0f}%"
    thr = p(facts.get("thresholds", {}).get("chronic_underperformance", 0.9))
    most_lost = max(facts["teams"], key=lambda x: x["lost_usd"])
    vac_mgrs = ", ".join(t["manager"] for t in facts["teams"] if t["vacant_seats"])
    plan = {"14": [], "30": [], "60": []}
    if v["orphaned_open_deals"]:
        plan["14"].append({"action": f"Reassign the {v['orphaned_open_deals']} deals with no owner "
                                     f"({m(v['orphaned_open_usd'])}) today", "owner": vac_mgrs or "Managers"})
    if d["past_due_negotiation_deals"]:
        plan["14"].append({"action": f"Review the {d['past_due_negotiation_deals']} past-due Negotiation deals "
                                     f"({m(d['past_due_negotiation_weighted_usd'])} weighted): confirm a dated next step "
                                     f"or move them out of the quarter", "owner": "All managers"})
    if ph["past_due_weighted_usd"] > 0:
        risk = facts["scenarios"][1]
        plan["14"].append({"action": f"Call the quarter as a range, {p(risk['pct_of_quota'])} to "
                                     f"{p(facts['forecast_pct_of_quota'])}, not on the Commit category",
                           "owner": "VP + Sales Ops"})
    if not plan["14"]:
        plan["14"].append({"action": "Walk each manager's largest late-stage deals and confirm a dated next step",
                           "owner": "All managers"})
    if v["vacant_seats"]:
        plan["30"].append({"action": "Backfill the vacant seats and cover their territories in the meantime",
                           "owner": "VP + Recruiting"})
    if ph["past_due_deals"]:
        plan["30"].append({"action": "No open deal may carry a past close date (weekly automatic flag)",
                           "owner": "Sales Ops"})
    if fcat["commit_on_early_stage_deals"]:
        plan["30"].append({"action": f"Set entry criteria for 'Commit' (start with the "
                                     f"{fcat['commit_on_early_stage_deals']} early-stage Commit deals)",
                           "owner": "Sales Ops"})
    if most_lost["lost_usd"] > 0:
        plan["60"].append({"action": f"Loss review with {most_lost['manager']}'s team "
                                     f"({m(most_lost['lost_usd'])} lost this quarter)", "owner": most_lost["manager"]})
    coach = ", ".join(r["rep"] for r in facts["reps_below_threshold_prior_two_quarters"])
    if coach:
        plan["60"].append({"action": f"Coaching plans for reps below {thr} in both prior quarters ({coach})",
                           "owner": "Managers"})
    for c in facts["rep_concentration"][:1]:
        plan["60"].append({"action": f"Reduce reliance on {c['rep']}, who holds "
                                     f"{p(c['share_of_team_weighted_pipeline'])} of their team's weighted pipeline",
                           "owner": c["manager"]})
    return plan


def template_summary(facts: dict) -> str:
    """Deterministic fallback summary (no AI) - always available, always correct."""
    t = sorted(facts["teams"], key=lambda x: x["gap_usd"])
    behind = [x for x in t if x["gap_usd"] < 0]
    v, ph, fcat, d = facts["vacancy"], facts["pipeline_hygiene"], facts["forecast_category"], facts["derived"]
    m = lambda x: f"${abs(x) / 1e6:.2f}M" if abs(x) >= 1e6 else f"${abs(x) / 1e3:,.0f}K"
    p = lambda x: f"{x * 100:.0f}%"
    name = lambda x: f"{x['manager']} ({x['country']} {x['segment']})"
    risk = facts["scenarios"][1]
    most_lost = max(facts["teams"], key=lambda x: x["lost_usd"])
    stalest = max(facts["teams"], key=lambda x: x["past_due_share_of_weighted"] or 0)
    thr = p(facts.get("thresholds", {}).get("chronic_underperformance", 0.9))
    region, fc_pct = facts["region"], facts["forecast_pct_of_quota"]

    # ---- headline -------------------------------------------------------------------
    head = (f"**Headline.** {region} is forecasting {m(facts['forecast_usd'])} against {m(facts['total_quota_usd'])} "
            f"quota ({fc_pct * 100:.1f}%), ")
    if facts["gap_usd"] >= 0:
        head += f"{m(facts['gap_usd'])} ahead of quota."
    elif len(behind) >= 2:
        head += (f"a gap of {m(facts['gap_usd'])}. {p(d['top2_teams_share_of_gap'])} of the gap sits in two teams: "
                 f"{name(behind[0])} at {p(behind[0]['attainment'])} and {name(behind[1])} at {p(behind[1]['attainment'])}.")
    else:
        head += f"a gap of {m(facts['gap_usd'])}, mostly in {name(behind[0])} at {p(behind[0]['attainment'])}."
    lines = [f"## {region} forecast summary - snapshot {facts['snapshot_date']} "
             f"({facts['days_left_in_quarter']} days to quarter end)", "", head, "",
             "**What is driving the gap**" if facts["gap_usd"] < 0 else "**What to watch**"]

    # ---- drivers: only the ones present in this data ---------------------------------
    if v["vacant_seats"]:
        orphan = (f" {v['orphaned_open_deals']} open deals ({m(v['orphaned_open_usd'])}) have no active owner."
                  if v["orphaned_open_deals"] else "")
        lines.append(f"- **Empty seats:** {v['vacant_seats']} vacant seats carry {m(v['vacant_quota_usd'])} of quota with "
                     f"only {m(v['forecast_on_vacant_seats_usd'])} forecast against it "
                     f"({p(d['vacant_seats_share_of_gap'])} of the gap).{orphan}")
    if ph["past_due_deals"]:
        lines.append(f"- **Stale pipeline:** {ph['past_due_deals']} of {ph['open_deals']} open deals are past their close "
                     f"date, holding {m(ph['past_due_weighted_usd'])} ({p(ph['past_due_share_of_weighted_pipeline'])}) of "
                     f"weighted pipeline. Worst: {name(stalest)}, {p(stalest['past_due_share_of_weighted'])} past due.")
    if most_lost["lost_usd"] > 0:
        lines.append(f"- **Losses:** {name(most_lost)} lost the most this quarter ({m(most_lost['lost_usd'])}).")
    if fcat["commit_on_early_stage_deals"] or fcat["late_stage_omitted_or_blank_deals"]:
        lines.append(f"- **Forecast category not reliable:** {fcat['commit_on_early_stage_deals']} 'Commit' deals are "
                     f"still early stage and {fcat['late_stage_omitted_or_blank_deals']} late-stage deals are Omitted or "
                     f"blank, so the call is built from stage and win probability instead.")

    # ---- risk -------------------------------------------------------------------------
    lines.append("")
    if ph["past_due_weighted_usd"] > 0:
        lines.append(f"**Risk.** If past-due deals close at half their stated odds, the region lands at "
                     f"{p(risk['pct_of_quota'])} ({m(risk['forecast_usd'])}). Call the quarter as a range: "
                     f"{p(risk['pct_of_quota'])} to {p(fc_pct)}.")
    else:
        lines.append("**Risk.** No open deal is past its close date, so the forecast does not lean on stale deals.")

    # ---- actions ----------------------------------------------------------------------
    plan = action_plan(facts)
    lines += ["", "**Recommended actions**"]
    for horizon, label in [("14", "Next 14 days"), ("30", "By day 30"), ("60", "By day 60")]:
        if plan[horizon]:
            lines.append(f"- **{label}:** {'; '.join(a['action'][0].lower() + a['action'][1:] for a in plan[horizon])}.")

    # ---- data confidence --------------------------------------------------------------
    dq = facts["data_quality"]
    fixes = [f"{n} {what}" for n, what in [(dq["duplicate_rows_removed"], "duplicate rows removed"),
                                           (dq["fx_rows_corrected"], "currency conversions corrected"),
                                           (dq["country_rows_remapped"], "country codes fixed")] if n]
    fixed = (f"{', '.join(fixes)} before any number was calculated." if fixes
             else "No duplicates, currency or country-code errors were found.")
    lines += ["", f"**Data confidence.** {dq['checks_triggered']} data checks fired ({dq['high_severity_checks']} high "
                  f"severity). {fixed[0].upper() + fixed[1:]}"]
    return "\n".join(lines)


def ai_summary(facts: dict, cfg: dict) -> tuple[str, str]:
    """Ask Claude to write the VP summary from the facts pack. Returns (text, source)."""
    import anthropic  # optional dependency: only needed for this step

    has_creds = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")) \
        or (Path.home() / ".config" / "anthropic").exists()
    if not has_creds:
        raise RuntimeError("no Anthropic credentials found - set ANTHROPIC_API_KEY to enable the AI summary")
    system = Path(cfg["ai_summary"]["prompt_file"]).read_text(encoding="utf-8")
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=cfg["ai_summary"]["model"],
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": cfg["ai_summary"].get("effort", "medium")},
        system=system,
        messages=[{"role": "user", "content": "FACTS PACK (JSON):\n" + json.dumps(facts, indent=1)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Model declined the request.")
    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    return text, f"AI ({response.model})"


def draft_summary(facts: dict, cfg: dict, use_ai: bool | None = None) -> tuple[str, str, pd.DataFrame]:
    """AI summary if enabled and credentials exist; otherwise the deterministic template.
    Returns (text, source, number_check). An AI draft with any unverified number is rejected."""
    use_ai = cfg["ai_summary"]["enabled"] if use_ai is None else use_ai
    if use_ai:
        try:
            text, source = ai_summary(facts, cfg)
            check = verify_numbers(text, facts)
            if len(check) and not check["verified"].all():
                bad = ", ".join(check.loc[~check["verified"], "number_in_text"])
                fallback = template_summary(facts)
                note = (f"\n\n> AI draft rejected: {int((~check['verified']).sum())} number(s) not traceable to the "
                        f"facts pack ({bad}). Deterministic summary shown instead.")
                return fallback + note, "Template (AI draft failed verification)", check
            return text, source, check
        except Exception as e:  # no SDK / no credentials / network - never block the diagnostic
            text = template_summary(facts)
            return text, f"Template (AI unavailable: {str(e)[:120] or type(e).__name__})", verify_numbers(text, facts)
    text = template_summary(facts)
    return text, "Template (AI disabled)", verify_numbers(text, facts)


# --------------------------------------------------------------------------------------
# Charts & export
# --------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, RED, GREEN, GREY = "#2a78d6", "#d03b3b", "#0ca30c", "#c3c2b7"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_charts(res: "Results", out_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    kfmt = FuncFormatter(lambda x, _: f"${x / 1e3:,.0f}K")

    # 1. Gap bridge (quota -> forecast)
    b = res.bridge
    quota = res.roster["quota_usd"].sum()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels = ["Quota"] + b["label"].tolist() + ["Forecast"]
    level = quota
    for i, g in enumerate(b["gap_usd"], start=1):
        start = level
        level += g
        ax.barh(i, g, left=start, color=RED if g < 0 else GREEN, height=0.6)
        ax.text(min(start, level) - 8e3, i, f"{g / 1e3:+,.0f}K", va="center", ha="right", fontsize=9, color=INK)
    ax.barh(0, quota, color=GREY, height=0.6)
    ax.barh(len(labels) - 1, level, color=BLUE, height=0.6)
    ax.text(quota + 8e3, 0, f"${quota / 1e6:.2f}M", va="center", fontsize=9, color=INK)
    ax.text(level + 8e3, len(labels) - 1, f"${level / 1e6:.2f}M ({level / quota:.1%})", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(quota * 0.85, quota * 1.03)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x / 1e6:.1f}M"))
    ax.set_title("Where the gap comes from: quota to forecast, by team", loc="left", fontsize=12, color=INK)
    _style(ax)
    fig.tight_layout()
    paths["bridge"] = out_dir / "01_gap_bridge.png"
    fig.savefig(paths["bridge"], dpi=160)
    plt.close(fig)

    # 2. Attainment by team
    t = res.by_manager.sort_values("attainment")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names = [f"{m} ({c} {s})" for m, c, s in zip(t.index, t["country"], t["segment"])]
    ax.barh(names, t["attainment"], color=[RED if a < 0.9 else BLUE for a in t["attainment"]], height=0.6)
    ax.axvline(1.0, color=INK2, linewidth=1, linestyle="--")
    for i, (a, g) in enumerate(zip(t["attainment"], t["gap_usd"])):
        ax.text(a + 0.005, i, f"{a:.0%}  ({g / 1e3:+,.0f}K)", va="center", fontsize=9, color=INK)
    ax.set_xlim(0.6, 1.12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_title("Forecast attainment by team (stage-weighted)", loc="left", fontsize=12, color=INK)
    _style(ax)
    fig.tight_layout()
    paths["attainment"] = out_dir / "02_attainment_by_team.png"
    fig.savefig(paths["attainment"], dpi=160)
    plt.close(fig)

    # 3. Rollup methods
    rc = res.reconciliation
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh(rc["method"], rc["clean_pct_of_quota"], height=0.55,
            color=[GREY if u else BLUE for u in rc["uses_forecast_category"]])
    rep = res.cfg.get("reported_forecast_pct")
    if rep:
        ax.axvline(rep, color=INK2, linestyle="--", linewidth=1)
        ax.text(rep + 0.01, -0.4, f"reported {rep:.0%}", ha="left", va="top", fontsize=8, color=INK2)
    for i, v in enumerate(rc["clean_pct_of_quota"]):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1.1)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.suptitle("Same data, four ways to add up the forecast (grey = uses the reps' forecast category)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    _style(ax)
    fig.tight_layout()
    paths["methods"] = out_dir / "03_rollup_methods.png"
    fig.savefig(paths["methods"], dpi=160)
    plt.close(fig)

    # 4. Past-due weighted pipeline by team
    t = res.by_manager.sort_values("past_due_share_of_weighted")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = [f"{m} ({c} {s})" for m, c, s in zip(t.index, t["country"], t["segment"])]
    cur = t["weighted_open_usd"] - t["past_due_weighted_usd"]
    ax.barh(names, cur, color=BLUE, height=0.6, label="Close date still ahead")
    ax.barh(names, t["past_due_weighted_usd"], left=cur, color=RED, height=0.6, label="Close date already passed")
    for i, (tot, s) in enumerate(zip(t["weighted_open_usd"], t["past_due_share_of_weighted"])):
        ax.text(tot + 1e4, i, f"{s:.0%} past due", va="center", fontsize=9, color=INK)
    ax.xaxis.set_major_formatter(kfmt)
    ax.set_xlim(0, t["weighted_open_usd"].max() * 1.3)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Weighted open pipeline: how much is already past its close date", loc="left", fontsize=12, color=INK)
    _style(ax)
    fig.tight_layout()
    paths["past_due"] = out_dir / "04_past_due_pipeline.png"
    fig.savefig(paths["past_due"], dpi=160)
    plt.close(fig)

    # 5. Category vs win probability
    ca = res.category_audit
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(ca.index, ca["avg_win_prob"], color=BLUE, width=0.55)
    for i, (v, n) in enumerate(zip(ca["avg_win_prob"], ca["deals"])):
        ax.text(i, v + 0.01, f"{v:.2f}\n({n} deals)", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Avg win probability", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID)
    ax.set_title("Open deals: forecast category vs. win probability", loc="left", fontsize=12, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    paths["category"] = out_dir / "05_category_vs_winprob.png"
    fig.savefig(paths["category"], dpi=160)
    plt.close(fig)
    return paths


def export(res: "Results", out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = res.dates["snapshot"].strftime("%Y-%m-%d")
    xlsx = out_dir / f"{res.cfg['region_name']}_forecast_diagnostic_{stamp}.xlsx"
    sheets = {
        "Scenarios": res.scenarios, "Reconciliation": res.reconciliation, "DQ_Log": res.dq_log,
        "By_Manager": res.by_manager.reset_index(), "By_Rep": res.by_rep.reset_index(),
        "By_Segment": res.by_segment.reset_index(), "By_Country": res.by_country.reset_index(),
        "Category_Audit": res.category_audit.reset_index(), "Stage_Health": res.health["by_stage"].reset_index(),
        "Cohorts": res.health["by_cohort"].reset_index(), "Bridge": res.bridge,
        "Act_PastDue": res.actions["past_due"], "Act_Orphaned": res.actions["orphaned_deals"],
        "Act_CategoryFix": res.actions["category_fixes"], "Act_Stalled": res.actions["stalled_early_stage"],
        "Concentration": res.actions["concentration"], "Losses": res.actions["losses"].reset_index(),
        "Clean_Data": res.opp,
    }
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df = df.copy()
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = df[c].dt.date
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max([len(str(c.value or "")) for c in col[:60]])  # header + first rows (sheet may be empty)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
    md = out_dir / f"{res.cfg['region_name']}_summary_{stamp}.md"
    check = res.number_check
    verified = f"{int(check['verified'].sum())}/{len(check)}" if len(check) else "n/a"
    md.write_text(f"{res.summary_text}\n\n---\n_Source: {res.summary_source}. Numbers traced to facts pack: {verified}._\n",
                  encoding="utf-8")
    facts_path = out_dir / f"{res.cfg['region_name']}_facts_{stamp}.json"
    facts_path.write_text(json.dumps(res.facts, indent=2), encoding="utf-8")
    charts = make_charts(res, out_dir / "charts")
    return {"workbook": xlsx, "summary": md, "facts": facts_path, **{f"chart_{k}": v for k, v in charts.items()}}


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
@dataclass
class Results:
    cfg: dict
    dates: dict
    fx: dict
    opp_raw: pd.DataFrame
    roster: pd.DataFrame
    opp: pd.DataFrame
    dq_log: pd.DataFrame
    reconciliation: pd.DataFrame
    category_audit: pd.DataFrame
    by_manager: pd.DataFrame
    by_segment: pd.DataFrame
    by_country: pd.DataFrame
    by_rep: pd.DataFrame
    health: dict
    actions: dict
    scenarios: pd.DataFrame
    bridge: pd.DataFrame
    facts: dict = field(default_factory=dict)
    summary_text: str = ""
    summary_source: str = ""
    number_check: pd.DataFrame = field(default_factory=pd.DataFrame)
    outputs: dict = field(default_factory=dict)


def run(config: str | Path | dict | None = None, use_ai: bool | None = None, export_outputs: bool = True) -> Results:
    cfg = config if isinstance(config, dict) else load_config(config)
    opp_raw, roster = load_data(cfg)
    dates = infer_dates(opp_raw, cfg)
    fx = infer_fx(opp_raw, cfg)
    opp_clean, dq = clean_and_validate(opp_raw, roster, cfg, dates, fx)
    o = enrich(opp_clean, roster, cfg, dates)
    teams = roster.groupby("manager_name")[["country", "segment"]].first()
    by_manager = gap_by(o, roster, "manager_name", cfg).join(teams)
    res = Results(
        cfg=cfg, dates=dates, fx=fx, opp_raw=opp_raw, roster=roster, opp=o, dq_log=dq,
        reconciliation=reconcile(o, opp_raw, roster, cfg), category_audit=category_audit(o, cfg),
        by_manager=by_manager, by_segment=gap_by(o, roster, "segment", cfg),
        by_country=gap_by(o, roster, "country", cfg), by_rep=rep_table(o, roster, cfg),
        health=pipeline_health(o, cfg), actions=action_lists(o, roster, cfg),
        scenarios=scenarios(o, roster, cfg), bridge=gap_bridge(o, roster, cfg),
    )
    res.facts = build_facts(res)
    res.summary_text, res.summary_source, res.number_check = draft_summary(res.facts, cfg, use_ai)
    if export_outputs:
        res.outputs = export(res, cfg["output_dir"])
    return res


def main():
    ap = argparse.ArgumentParser(description="Quarter-end forecast gap diagnostic")
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--no-ai", action="store_true", help="skip the AI summary and use the deterministic template")
    args = ap.parse_args()
    res = run(args.config, use_ai=False if args.no_ai else None)
    s = res.scenarios.iloc[0]
    print(f"\n{res.cfg['region_name']} | snapshot {res.dates['snapshot'].date()} | "
          f"{res.dates['days_left']} days to quarter end")
    print(f"Forecast ${s.forecast_usd:,.0f} = {s.pct_of_quota:.1%} of quota (gap ${s.gap_usd:,.0f})")
    print(f"Data-quality checks triggered: {len(res.dq_log)} "
          f"({(res.dq_log.severity == 'High').sum()} high severity)\n")
    print(res.summary_text)
    print(f"\n[summary source: {res.summary_source}]")
    print("\nOutputs:")
    for k, v in res.outputs.items():
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
