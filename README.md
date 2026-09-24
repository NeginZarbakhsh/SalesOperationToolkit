# Sales Operation Toolkit

### A forecast diagnostic that cleans the data, reconciles the number, finds the gap, and lets AI explain it — never calculate it.

[![CI](https://github.com/NeginZarbakhsh/SalesOperationToolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/NeginZarbakhsh/SalesOperationToolkit/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-10%20passing-brightgreen)](tests/test_pipeline.py)

Give it two ordinary CRM exports — opportunities and a rep roster — and it answers the question a VP of
Sales actually asks: **why are we missing the number, where exactly, and what do we do in the next 14 days?**

![The dashboard](docs/img/dashboard_overview.png)

---

## The problem

A forecast roll-up is only as good as the fields under it. In a typical export, some deals are duplicated,
some amounts never got converted from local currency, country names are typed by hand, a date field means
two different things in different rows, and the rep's own "Commit" flag does not track reality. Add that up
without looking and you get a number nobody can defend — and a leadership team focused on the wrong team.

## The answer this produces

Run it on the bundled synthetic data and you get, in about two seconds:

| | |
|---|---|
| **Forecast** | **$8.20M against $9.21M quota = 89.0%**, a gap of $1.01M with 14 days left |
| **Where** | 70% of the gap is in two of six teams |
| **Why (structural)** | 2 empty seats carry $640K of quota with $223K forecast against them: 41% of the gap, while that team's active reps are at 97% |
| **Why (pipeline)** | 161 of 271 open deals are past their close date, holding 60% of the weighted pipeline |
| **Why (data)** | 6 "Commit" deals are still early stage and 36 late-stage deals are marked Omitted or blank, so a category roll-up lands 18 points off the stage-based number (71% vs 89%) |
| **So what** | Call the quarter as a range, **74% to 89%**, and work the 6 ownerless deals and 51 past-due Negotiation deals first |

Every one of those numbers is computed in code, logged, and covered by a test.

## How it works

```mermaid
flowchart LR
    A["opportunities.csv<br/>rep_roster.csv"] --> B["1 · Clean &amp; validate<br/>21 checks, every fix logged"]
    B --> C["2 · Reconcile<br/>which roll-up gives the reported %?"]
    C --> D["3 · Diagnose<br/>team · market · segment · rep · stage · cohort"]
    D --> E["4 · Act<br/>14/30/60-day plan, deal lists"]
    E --> F["5 · Summarise<br/>facts pack → AI draft → number check"]
    F --> G["Dashboard · Excel · charts · summary.md · facts.json"]
```

**1 · Clean and validate.** 21 automated checks. Problems the data itself can answer are fixed and logged
with the rows and dollars they moved (duplicate rows, unconverted currency, country names, a broken age
field). Problems only a person can answer are flagged, never guessed (blank categories, close dates already
in the past, deals owned by an empty seat).

**2 · Reconcile.** Every plausible way of adding up the forecast is computed on both the raw and the cleaned
data, and compared with the number leadership is quoting, so the analysis starts from a number that can be
defended:

```
forecast = Closed Won (full value) + Σ(open deal amount × win probability)
```

**3 · Diagnose.** A quota → forecast bridge by team, with vacant seats split out from active reps, plus cuts
by market, segment, rep, pipeline stage and deal cohort. Structural causes are separated from execution
causes because a VP fixes them differently.

**4 · Act.** Rule-based 14 / 30 / 60-day plan with an owner per action, and per-manager deal lists (no
owner, past due, label mismatch, stalled) exported to Excel and CSV.

**5 · Summarise.** A briefing a VP can read in a minute, with the AI kept on a short leash (below).

## Where AI is used — and where it deliberately is not

| Step | Who does it |
|---|---|
| Cleaning, validation, every calculation, the action plan | **Code only.** No model involved |
| The VP briefing | Claude writes prose **from a pre-computed facts pack**, then `verify_numbers()` re-checks every figure in the text against that pack. A draft containing a number that cannot be traced is **rejected** and the deterministic template is shipped instead |
| The "Ask AI" page | A tool-using agent: Claude picks among **8 analysis tools** that query the validated data, the code computes every number, and each answer shows the tools it called plus a per-number trace |

This is the part I care most about. An LLM that writes a confident wrong number is worse than no summary at
all, so the model never sees a calculator and never sees raw rows it did not ask for:

```python
text  = ai_summary(facts, cfg)          # the model only writes prose
check = verify_numbers(text, facts)     # every $ , % and count is matched back to the facts pack
if not check["verified"].all():         # one untraceable number and the draft is dropped
    text = template_summary(facts)
```

The guardrail is covered by a test that feeds it invented numbers and asserts they are caught
([`tests/test_pipeline.py`](tests/test_pipeline.py)).

The AI features are **optional**: with no API key, the toolkit, the dashboard and the deterministic summary
all still work, and the output records which path was used.

## The dashboard

`streamlit run app.py`, then drop the two CSVs in. Six pages, in the order the questions get asked:

| Page | Answers |
|---|---|
| **Overview** | Will we hit the number? What the forecast is made of, and a what-if slider for stale deals |
| **Teams** | Where is the gap? By manager, market or segment, down to each rep and the deals to work |
| **Pipeline** | How healthy are the deals? Late deals by stage, by cohort, and a team × stage heatmap |
| **Data trust** | Can we trust it? Every check, what was fixed, which roll-up reproduces the reported % |
| **Actions** | What do we do? The 14/30/60-day plan and per-manager deal lists |
| **Ask AI** | Ask in plain English; the code answers, the model explains |

| Data trust | Ask AI |
|---|---|
| ![Data trust page](docs/img/dashboard_data_trust.png) | ![Ask AI page](docs/img/dashboard_ask_ai.png) |

## Quickstart

```bash
git clone https://github.com/NeginZarbakhsh/SalesOperationToolkit.git
cd SalesOperationToolkit
pip install -r requirements.txt

python forecast_diagnostic.py     # full run → outputs/ (Excel, charts, summary.md, facts.json)
streamlit run app.py              # the dashboard
pytest -q                         # 10 tests on the sample data
```

Optional AI features: `cp .env.example .env` and add your own `ANTHROPIC_API_KEY` (`.env` is git-ignored).

Prefer to read rather than run? [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) has the whole
pipeline with results already rendered, and [`docs/sample_summary.md`](docs/sample_summary.md) is the
briefing it produces.

## The sample data is synthetic

`data/` is generated by [`data/generate_sample_data.py`](data/generate_sample_data.py) with a fixed seed:
an invented APAC region, 6 teams, 29 seats, 469 deals, 6 currencies. The data-quality problems are **planted
on purpose** so the diagnostic has something to find, and the planted counts are exactly what the tests
assert. No company data is used anywhere in this repo.

To run it on a real export, point `config.json` at your two CSVs. Required columns:

- **Opportunities:** `opp_id, account_id, account_name, segment, country, rep_name, manager_name, deal_type,
  pipeline_stage, forecast_category, created_date, close_date, age_days, amount_local, currency, amount_usd,
  win_probability, loss_reason`
- **Rep roster:** `rep_name, manager_name, country, segment, quota_usd, tenure_months, headcount_status,
  historical_attainment_pct_q_minus_1, historical_attainment_pct_q_minus_2`

## Design decisions worth calling out

- **Config over code.** Stage names, FX handling, country spellings, thresholds, the reported forecast % and
  the AI settings all live in `config.json`. `"auto"` means the toolkit infers the value (snapshot date,
  quarter dates, FX rates) and prints what it inferred, so next quarter is a config edit, not a code edit.
- **Fix vs flag.** A fix is only applied when the data can prove the right answer. Everything else is
  surfaced as a decision for a human, with the rows and dollars attached.
- **Stage × win probability, never the self-reported category.** The category is audited in its own view so
  the reader can see *why* it is not trusted.
- **Structural ≠ execution.** Vacant seats are separated from rep performance everywhere, including in the
  bridge chart, because they lead to different actions (hire vs coach).
- **Deterministic first, AI second.** Every output exists without an API key; the model improves the prose,
  not the numbers.
- **Tested and wired to CI.** The sample data has a fixed seed, so a change that moves the forecast fails a
  test instead of quietly changing a number in a report.

## Project structure

```
forecast_diagnostic.py        the engine: clean → reconcile → diagnose → act → summarise
app.py                        six-page Streamlit dashboard
agent.py                      8 analysis tools + the tool-using agent behind "Ask AI"
config.json                   everything that changes per quarter or per region
prompts/vp_summary_prompt.md  system prompt for the AI briefing (tone and format live here, not in code)
data/                         synthetic sample data + the generator that makes it
notebooks/walkthrough.ipynb   the pipeline step by step, with results
tests/test_pipeline.py        10 end-to-end tests, run in CI on 3.11 and 3.12
docs/                         sample briefing and the screenshots above
```

## Limitations, and what I would build next

- The number verifier catches figures that do not exist in the data, but not a real figure attached to the
  wrong claim; a human still reviews the briefing.
- FX rates are inferred from the data. In production, read the finance rate table.
- Column names are fixed; a column-mapping block in the config would cover other CRM exports.
- Not yet built: a scheduled weekly run that emails each manager their own list, and hosting behind SSO.

## About

Built by **Negin Zarbakhsh** as a study of how far a Sales Ops workflow can be automated while keeping every
number auditable. Feedback and issues are welcome.

MIT licensed — see [LICENSE](LICENSE).
