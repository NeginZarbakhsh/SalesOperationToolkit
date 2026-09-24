"""Builds notebooks/walkthrough.ipynb. Run from the repo root: python notebooks/build_walkthrough.py"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent / "walkthrough.ipynb"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Forecast diagnostic: walkthrough

This notebook runs the toolkit on the synthetic sample data and shows each step of the pipeline:

**clean → validate → reconcile → diagnose → act → summarise**

Everything comes from `forecast_diagnostic.py`; the settings come from `config.json`. To run it on your own
export, point `config.json` at your two CSVs and run this notebook again.
""")
code(r"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import forecast_diagnostic as fd

res = fd.run(ROOT / "config.json", use_ai=False, export_outputs=False)
F = res.facts
print(f"{res.cfg['region_name']}  |  snapshot {F['snapshot_date']}  |  {F['days_left_in_quarter']} days to quarter end")
print(f"{len(res.opp_raw)} rows in, {len(res.opp)} after cleaning  |  FX rates inferred: {res.fx}")
""")

md(r"""
## 1. Clean and validate

Every check that found something, with the rows and dollars it touched. Fixes the data can answer itself
(duplicates, currency, country codes, deal age) are applied and logged; anything that needs a human decision
is flagged instead of guessed.
""")
code(r"""
res.dq_log[["check_id", "check", "severity", "action", "rows_affected", "usd_affected"]]
""")

md(r"""
## 2. Which roll-up reproduces the reported number?

The self-reported `forecast_category` and the stage-based calculation give very different answers. Only one
of them reconciles with the number leadership is quoting, and only after cleaning.
""")
code(r"""
res.reconciliation[["method", "raw_pct_of_quota", "clean_pct_of_quota", "matches_reported"]]
""")

md(r"""
Is the category worth trusting? If it meant anything, `Commit` deals would carry a much higher win
probability than `Pipeline` deals.
""")
code(r"""
res.category_audit[["deals", "open_usd", "avg_win_prob", "share_in_early_stage", "share_past_due"]].round(2)
""")

md(r"""
## 3. The forecast, and how fragile it is

Forecast = Closed Won + every open deal × its win probability. The scenarios show what happens if the deals
that already missed their close date are worth less than the pipeline claims.
""")
code(r"""
res.scenarios[["scenario", "forecast_usd", "pct_of_quota", "gap_usd"]]
""")

md(r"""
## 4. Where the gap is

Quota comes from the roster, the forecast from the deals. Vacant seats are split out from active reps,
because a coverage gap and a performance gap need different fixes.
""")
code(r"""
cols = ["country", "segment", "quota_usd", "vacant_quota_usd", "forecast_usd", "attainment", "gap_usd",
        "share_of_gap", "coverage_of_remaining", "past_due_share_of_weighted", "win_rate_usd"]
res.by_manager[cols].round(3)
""")
code(r"""
res.bridge   # quota -> forecast, one row per team, vacant seats on their own line
""")

md(r"""
## 5. Pipeline health: stage, cohort and hygiene
""")
code(r"""
res.health["by_stage"]
""")
code(r"""
res.health["by_cohort"]   # deals created before the quarter vs during it
""")

md(r"""
## 6. Lists a manager can act on this week
""")
code(r"""
show = ["opp_id", "account_name", "rep_name", "manager_name", "pipeline_stage", "close_date",
        "days_past_due", "amount_usd", "win_probability", "weighted_open_usd"]
res.actions["past_due"][show].head(10)
""")
code(r"""
res.actions["orphaned_deals"][show]     # open deals owned by a seat nobody sits in
""")

md(r"""
## 7. The summary

`template_summary()` writes the briefing from the facts pack with no AI involved. With an API key set,
`fd.draft_summary(F, res.cfg, use_ai=True)` asks Claude to write it instead, then checks every number it
wrote against the same facts pack and falls back to this template if anything cannot be traced.
""")
code(r"""
from IPython.display import Markdown

Markdown(fd.template_summary(F))
""")
code(r"""
check = fd.verify_numbers(fd.template_summary(F), F)
print(f"{int(check['verified'].sum())}/{len(check)} numbers in the summary trace back to the facts pack")
""")

md(r"""
## 8. Run it on your own data

1. Export your opportunities and rep roster as CSV, with the columns listed in the README.
2. Point `opportunity_file` and `roster_file` in `config.json` at them (stage names, thresholds, FX and
   country spellings live there too).
3. Run this notebook again, or `python forecast_diagnostic.py`, which also writes the Excel workbook,
   the charts and the summary into `outputs/`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("wrote", OUT)
