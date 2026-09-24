"""End-to-end checks on the synthetic sample export.

The sample data is generated with a fixed seed, so these numbers are stable: if a change to the pipeline
moves the forecast, a test fails instead of the number quietly changing in a report.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import forecast_diagnostic as fd  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return fd.load_config(ROOT / "config.json")


@pytest.fixture(scope="module")
def res(cfg):
    return fd.run(cfg, use_ai=False, export_outputs=False)


# ---- reading a CRM export ----------------------------------------------------------------
def test_the_export_keeps_its_own_column_names(cfg):
    raw = pd.read_csv(cfg["opportunity_file"], nrows=1)
    assert {"Id", "StageName", "ForecastCategoryName", "Probability", "ConvertedAmount"} <= set(raw.columns)


def test_columns_are_mapped_onto_the_toolkit_fields(cfg):
    opp, roster = fd.load_data(cfg)
    assert set(fd.REQUIRED_OPP_COLS) <= set(opp.columns)
    assert set(fd.REQUIRED_ROSTER_COLS) <= set(roster.columns)
    assert opp["win_probability"].between(0, 1).all()      # the CRM stores 0-100, the toolkit works in 0-1


def test_a_missing_column_says_which_one_to_map(cfg, tmp_path):
    broken = pd.read_csv(cfg["opportunity_file"]).drop(columns=["StageName"])
    path = tmp_path / "broken.csv"
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="StageName"):
        fd.load_data({**cfg, "opportunity_file": str(path)})


# ---- cleaning -----------------------------------------------------------------------------
def test_duplicates_are_removed(res):
    assert len(res.opp_raw) == 479
    assert len(res.opp) == 473                      # the 6 planted duplicate rows are gone
    assert res.opp["opp_id"].is_unique


def test_every_planted_problem_is_caught(res):
    fired = dict(zip(res.dq_log["check_id"], res.dq_log["rows_affected"]))
    assert fired["DQ01"] == 6                       # duplicate rows
    assert fired["DQ03"] == 9                       # ConvertedAmount still in local currency
    assert fired["DQ05"] == 34                      # billing country spelled out
    assert fired["DQ09"] == 2                       # quota on seats with nobody in them
    assert fired["DQ12"] == 24                      # blank forecast category
    assert fired["DQ19"] == 6                       # closed lost with no reason
    assert "DQ21" in fired                          # the empty NextStep column


def test_currency_is_repaired_not_dropped(res):
    implied = res.opp["amount_usd"] / res.opp["amount_local"]
    assert (implied - res.opp["currency"].map(res.fx)).abs().max() < 0.02


def test_country_codes_match_the_roster(res):
    assert set(res.opp["country"]) <= set(res.roster["country"])


def test_dates_are_inferred_from_the_data(res):
    assert str(res.dates["snapshot"].date()) == "2027-04-16"
    assert str(res.dates["quarter_end"].date()) == "2027-04-30"
    assert res.dates["days_left"] == 14


# ---- the forecast -------------------------------------------------------------------------
def test_forecast_is_stage_weighted(res):
    o = res.opp
    expected = o.loc[o["is_won"], "amount_usd"].sum() + o.loc[o["is_open"], "weighted_open_usd"].sum()
    assert res.facts["forecast_usd"] == pytest.approx(expected, abs=1)
    assert res.facts["forecast_pct_of_quota"] == pytest.approx(0.8909, abs=0.0005)
    assert res.facts["gap_usd"] == pytest.approx(-1_186_538, abs=1)


def test_only_the_stage_based_rollup_reproduces_the_reported_number(res):
    rc = res.reconciliation.set_index("method")
    reproduces = rc.loc[rc["matches_reported"]].index
    assert len(reproduces) == 1
    assert "win probability" in reproduces[0]        # not the self-reported category


def test_the_final_stage_comes_from_the_config(res):
    d = res.facts["derived"]
    assert d["final_stage"] == "Negotiation/Review"  # the CRM's name for it, not a hardcoded one
    assert d["past_due_final_stage_deals"] > 0


def test_scenarios_bracket_the_forecast(res):
    pcts = list(res.scenarios["pct_of_quota"])
    assert pcts[1] < pcts[0]                         # risk-adjusted sits below the headline
    assert pcts[2] < pcts[1]                         # downside below that


# ---- the summary and its guardrail ---------------------------------------------------------
def test_every_number_in_the_summary_traces_back_to_the_data(res):
    check = fd.verify_numbers(fd.template_summary(res.facts), res.facts)
    assert len(check) > 20
    assert check["verified"].all()


def test_invented_numbers_are_rejected(res):
    text = "The region is forecasting $12.34M, which is 77.7% of quota, with 4321 open deals."
    check = fd.verify_numbers(text, res.facts)
    assert not check["verified"].all()               # this is what stops a hallucination shipping


# ---- optional AI credentials ---------------------------------------------------------------
def test_a_local_env_file_is_read_without_overriding_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(chr(10).join(["# a comment line",
                                 'ANTHROPIC_API_KEY="sk-ant-from-file"',
                                 "OTHER=plain", ""]), encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OTHER", "already-set")
    fd.load_env_file(env)
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"   # picked up for the optional AI step
    assert os.environ["OTHER"] == "already-set"                    # a real environment variable still wins


def test_the_facts_pack_carries_the_haircut_value(res):
    """The summary writer must never derive a number: the downside figure is pre-computed for it."""
    d = res.facts["derived"]
    expected = res.facts["pipeline_hygiene"]["past_due_weighted_usd"] * 0.5
    assert d["past_due_weighted_at_haircut_usd"] == pytest.approx(expected, abs=1)
