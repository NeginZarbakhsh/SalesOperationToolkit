"""The same engine, a second customer, no code change.

customers/acme_hubspot.json points at a HubSpot-shaped export: different property names, different stage
ids, forecast values in SCREAMING_CASE, probability stored 0-1 instead of 0-100, different currencies and
a different fiscal quarter. If onboarding a customer ever needs a code change, these tests fail.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import forecast_diagnostic as fd  # noqa: E402

CONFIG = ROOT / "customers" / "acme_hubspot.json"


@pytest.fixture(scope="module")
def res():
    return fd.run(CONFIG, use_ai=False, export_outputs=False)


def test_the_export_is_in_the_other_crms_shape():
    raw = pd.read_csv(fd.load_config(CONFIG)["opportunity_file"], nrows=5)
    assert {"hs_object_id", "dealstage", "hs_forecast_category", "hs_deal_stage_probability"} <= set(raw.columns)
    assert raw["dealstage"].str.islower().all()                 # closedwon, contractsent ...


def test_its_vocabulary_is_translated(res):
    assert res.opp["forecast_category"].dropna().isin(
        ["Commit", "Best Case", "Pipeline", "Omitted", "Closed"]).all()
    assert res.opp["deal_type"].isin(["New Business", "Renewal", "Expansion"]).all()
    assert res.opp["win_probability"].between(0, 1).all()       # already 0-1 here, so it is left alone


def test_it_produces_a_full_diagnostic(res):
    F = res.facts
    assert F["region"] == "Nordics & Ireland"
    assert F["total_quota_usd"] == 5_205_000
    assert 0.85 < F["forecast_pct_of_quota"] < 0.95
    assert F["pipeline_hygiene"]["past_due_deals"] > 0
    assert F["vacancy"]["vacant_seats"] == 1                    # employment_status "VACANT", not "Open Req"


def test_only_the_stage_based_rollup_reproduces_the_reported_number(res):
    rc = res.reconciliation.set_index("method")
    assert list(rc.loc[rc["matches_reported"]].index) == [i for i in rc.index if "win probability" in i]


def test_the_briefing_names_the_stage_the_way_a_human_would(res):
    """The CRM calls it 'contractsent'; stage_labels in the config makes the briefing say 'Contract Sent'."""
    assert res.facts["derived"]["final_stage"] == "Contract Sent"
    text = fd.template_summary(res.facts)
    assert "contractsent" not in text
    assert fd.verify_numbers(text, res.facts)["verified"].all()
