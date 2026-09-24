"""End-to-end checks on the synthetic sample data.

The sample data is generated with a fixed seed, so these numbers are stable: if a change to the pipeline
moves the forecast, a test fails instead of the number quietly changing in a report.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import forecast_diagnostic as fd  # noqa: E402


@pytest.fixture(scope="module")
def res():
    return fd.run(ROOT / "config.json", use_ai=False, export_outputs=False)


# ---- cleaning ---------------------------------------------------------------------------
def test_duplicates_are_removed(res):
    assert len(res.opp_raw) == 469
    assert len(res.opp) == 464                      # the 5 planted duplicate rows are gone
    assert res.opp["opp_id"].is_unique


def test_every_planted_problem_is_caught(res):
    fired = dict(zip(res.dq_log["check_id"], res.dq_log["rows_affected"]))
    assert fired["DQ01"] == 5                       # duplicate rows
    assert fired["DQ03"] == 7                       # amounts never converted to USD
    assert fired["DQ05"] == 31                      # country spelled out instead of the roster code
    assert fired["DQ09"] == 2                       # quota on seats with nobody in them
    assert fired["DQ12"] == 22                      # blank forecast category
    assert fired["DQ19"] == 5                       # closed lost with no reason
    assert "DQ21" in fired                          # the empty column


def test_currency_is_repaired_not_dropped(res):
    rate = res.opp["currency"].map(res.fx)
    implied = res.opp["amount_usd"] / res.opp["amount_local"]
    assert (implied - rate).abs().max() < 0.02      # every row now sits at its currency's rate


def test_country_codes_match_the_roster(res):
    assert set(res.opp["country"]) <= set(res.roster["country"])


def test_dates_are_inferred_from_the_data(res):
    assert str(res.dates["snapshot"].date()) == "2026-10-17"
    assert str(res.dates["quarter_end"].date()) == "2026-10-31"
    assert res.dates["days_left"] == 14


# ---- the forecast -----------------------------------------------------------------------
def test_forecast_is_stage_weighted(res):
    o = res.opp
    expected = o.loc[o["is_won"], "amount_usd"].sum() + o.loc[o["is_open"], "weighted_open_usd"].sum()
    assert res.facts["forecast_usd"] == pytest.approx(expected, abs=1)
    assert res.facts["forecast_pct_of_quota"] == pytest.approx(0.8903, abs=0.0005)
    assert res.facts["gap_usd"] == pytest.approx(-1_009_902, abs=1)


def test_only_the_stage_based_rollup_reproduces_the_reported_number(res):
    rc = res.reconciliation.set_index("method")
    reproduces = rc.loc[rc["matches_reported"]].index
    assert len(reproduces) == 1
    assert "win probability" in reproduces[0]        # not the self-reported category


def test_scenarios_bracket_the_forecast(res):
    pcts = list(res.scenarios["pct_of_quota"])
    assert pcts[1] < pcts[0]                         # risk-adjusted is below the headline
    assert pcts[2] < pcts[1]                         # downside is below that


# ---- the summary and its guardrail -------------------------------------------------------
def test_every_number_in_the_summary_traces_back_to_the_data(res):
    text = fd.template_summary(res.facts)
    check = fd.verify_numbers(text, res.facts)
    assert len(check) > 20
    assert check["verified"].all()


def test_invented_numbers_are_rejected(res):
    text = "The region is forecasting $12.34M, which is 77.7% of quota, with 4321 open deals."
    check = fd.verify_numbers(text, res.facts)
    assert not check["verified"].all()               # the verifier is what stops a hallucination shipping
