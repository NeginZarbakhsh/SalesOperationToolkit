"""
Forecast agent: Claude answers questions about the data by calling tools that run on the
validated toolkit output. The model decides WHICH analysis to run; the code does every calculation.

    question -> Claude picks tools -> tools query the cleaned data (forecast_diagnostic.Results)
             -> Claude writes the answer -> every number in it is checked against the tool outputs

Used by app.py (Streamlit UI). Can also be run from a terminal for a quick test:
    python agent.py "Which deals should the German team chase this week?"
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import forecast_diagnostic as fd

HERE = Path(__file__).resolve().parent
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are the forecast analyst agent for a regional Sales Operations team. You answer questions from a
VP of Sales or a front-line manager about the uploaded opportunity and rep-roster data.

How to work:
- Get every number from a tool. Never estimate, and never add up figures yourself when a tool can return them
  (use run_scenario for what-ifs, find_deals for totals of any deal list).
- The forecast is Closed Won + open pipeline x win probability, on data the toolkit has already cleaned
  (duplicates removed, currency fixed, country codes mapped). The self-reported forecast_category is unreliable;
  never present a category-based rollup as the forecast.
- Separate structural causes (vacant seats with quota) from execution causes (stale pipeline, losses, rep performance).
- Talk about individual reps neutrally, as coaching or pipeline observations, not judgements.
- If the data cannot answer the question, say so and say what data would be needed.

How to answer:
- Lead with the answer in one or two sentences, then at most five bullets or one small table.
- Money as $K or $M (at most two decimals), percentages as whole numbers or one decimal.
- End with one concrete next step when the question is about action."""


# --------------------------------------------------------------------------------------
# Credentials: read from the environment or a local .env file. The key never lives in code.
# --------------------------------------------------------------------------------------
def load_env_file(path: Path = HERE / ".env") -> None:
    """Same .env loader the toolkit uses, so the app and the command line behave identically."""
    fd.load_env_file(path)


def ai_available() -> bool:
    load_env_file()
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------
def _clean(v):
    if isinstance(v, (pd.Timestamp,)):
        return str(v.date())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 4)
    return v


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    df = df.head(limit) if limit else df
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _match(series: pd.Series, value: str) -> pd.Series:
    return series.astype(str).str.contains(str(value), case=False, regex=False)


TOOLS = [
    {
        "name": "get_headline",
        "description": "Region-level headline: quota, closed won, weighted open pipeline, forecast, % of quota, gap, "
                       "pipeline coverage, forecast scenarios (incl. past-due risk), data-quality summary and key derived "
                       "shares. Call this first for any 'overall' or 'why are we missing' question.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_gap_breakdown",
        "description": "Forecast vs quota by team (manager), segment, country or rep, sorted by largest gap. Includes "
                       "attainment, gap, share of gap, vacant-seat quota, past-due share of weighted pipeline, win rate "
                       "and losses (team/segment/country), or tenure, prior-quarter attainment and coaching flags (rep).",
        "input_schema": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string", "enum": ["team", "segment", "country", "rep"]},
                "limit": {"type": "integer", "description": "max rows, default 10"},
            },
            "required": ["dimension"], "additionalProperties": False,
        },
    },
    {
        "name": "find_deals",
        "description": "List and total opportunities matching filters. Use for 'which deals...' questions, past-due or "
                       "stalled deal lists, deals owned by vacant seats (orphaned), or a manager's pipeline. Returns the "
                       "count and totals of ALL matches plus the top rows.",
        "input_schema": {
            "type": "object",
            "properties": {
                "manager": {"type": "string", "description": "manager name or part of it"},
                "rep": {"type": "string", "description": "rep name or part of it"},
                "country": {"type": "string", "description": "country code or name as it appears in the roster"},
                "segment": {"type": "string"},
                "status": {"type": "string", "enum": ["open", "won", "lost", "any"], "description": "default open"},
                "stage": {"type": "string"},
                "past_due": {"type": "boolean", "description": "only open deals whose close date has already passed"},
                "orphaned": {"type": "boolean", "description": "only open deals owned by a vacant (Open Req) seat"},
                "stalled": {"type": "boolean", "description": "only early-stage deals older than the stall threshold"},
                "forecast_category": {"type": "string", "enum": ["Commit", "Best Case", "Pipeline", "Omitted", "(blank)"]},
                "min_amount_usd": {"type": "number"},
                "sort_by": {"type": "string", "enum": ["weighted", "amount", "days_past_due", "age"],
                            "description": "default weighted"},
                "limit": {"type": "integer", "description": "rows to return, default 10, max 40"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "run_scenario",
        "description": "What-if on the forecast. Scale the win probability of past-due deals (e.g. 0.5 = they close at half "
                       "their stated odds, 0 = they all slip), mark specific deals as won or lost, or drop deals owned by "
                       "vacant seats. Returns the new forecast, % of quota, gap, and change vs the base forecast.",
        "input_schema": {
            "type": "object",
            "properties": {
                "past_due_probability_multiplier": {"type": "number", "description": "0 to 1, default 1"},
                "assume_won": {"type": "array", "items": {"type": "string"}, "description": "opp_ids to treat as won"},
                "assume_lost": {"type": "array", "items": {"type": "string"}, "description": "opp_ids to treat as lost"},
                "exclude_orphaned_deals": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_rep_profile",
        "description": "One rep: roster details (manager, segment, tenure, quota, prior two quarters' attainment), current "
                       "forecast and attainment, coaching flags, losses, and their largest open deals.",
        "input_schema": {"type": "object", "properties": {"rep_name": {"type": "string"}},
                         "required": ["rep_name"], "additionalProperties": False},
    },
    {
        "name": "get_data_quality_issues",
        "description": "The data-quality log: every check that fired, severity, rows and dollars affected, and the action "
                       "taken (removed, corrected, flagged). Use for 'can we trust this data' questions.",
        "input_schema": {
            "type": "object",
            "properties": {"min_severity": {"type": "string", "enum": ["High", "Medium", "Low", "Info"]}},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_forecast_category_audit",
        "description": "Is the self-reported forecast category reliable? Average win probability, early-stage share and "
                       "past-due share per category, every way of rolling up the forecast vs quota, and the count of "
                       "category fixes needed by team.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_losses",
        "description": "Closed-lost deals this quarter by loss reason and deal type, optionally for one team (manager).",
        "input_schema": {"type": "object", "properties": {"manager": {"type": "string"}}, "additionalProperties": False},
    },
]


class ForecastTools:
    """Executes tool calls against one diagnostic run (fd.Results)."""

    def __init__(self, res: fd.Results):
        self.res = res
        self.quota = float(res.roster["quota_usd"].sum())
        self.base_forecast = float(res.opp["forecast_usd"].sum())

    def run(self, name: str, args: dict) -> dict:
        fn = getattr(self, name, None)
        if fn is None or name.startswith("_") or name == "run":
            raise ValueError(f"Unknown tool: {name}")
        return fn(**(args or {}))

    # ---- tools -----------------------------------------------------------------------
    def get_headline(self) -> dict:
        f = self.res.facts
        keys = ["region", "snapshot_date", "quarter_end", "days_left_in_quarter", "total_quota_usd", "closed_won_usd",
                "weighted_open_pipeline_usd", "forecast_usd", "forecast_pct_of_quota", "gap_usd", "open_pipeline_usd",
                "coverage_of_remaining_quota_x", "scenarios", "vacancy", "pipeline_hygiene", "data_quality", "derived"]
        return {k: f[k] for k in keys}

    def get_gap_breakdown(self, dimension: str, limit: int = 10) -> dict:
        if dimension == "rep":
            df = self.res.by_rep.reset_index()[[
                "rep_name", "manager_name", "country", "segment", "headcount_status", "tenure_months", "quota_usd",
                "forecast_usd", "attainment", "gap_usd", "historical_attainment_pct_q_minus_1",
                "historical_attainment_pct_q_minus_2", "chronic_underperformer", "open_deals", "past_due_deals"]]
            df = df.sort_values("gap_usd")
        else:
            src = {"team": self.res.by_manager, "segment": self.res.by_segment, "country": self.res.by_country}[dimension]
            df = src.reset_index()
            cols = [df.columns[0]] + [c for c in ["country", "segment", "quota_usd", "vacant_quota_usd", "forecast_usd",
                                                  "attainment", "gap_usd", "share_of_gap", "coverage_of_remaining",
                                                  "past_due_share_of_weighted", "win_rate_usd", "lost_usd"]
                                      if c in df.columns and c != df.columns[0]]
            df = df[cols]
        return {"dimension": dimension, "rows": _records(df, min(int(limit or 10), 40))}

    def find_deals(self, manager=None, rep=None, country=None, segment=None, status="open", stage=None, past_due=None,
                   orphaned=None, stalled=None, forecast_category=None, min_amount_usd=None, sort_by="weighted",
                   limit=10) -> dict:
        o = self.res.opp
        m = pd.Series(True, index=o.index)
        status = status or "open"
        m &= {"open": o["is_open"], "won": o["is_won"], "lost": o["is_lost"], "any": m}[status]
        if manager:
            m &= _match(o["manager_name"], manager)
        if rep:
            m &= _match(o["rep_name"], rep)
        if country:
            code = self.res.cfg.get("country_aliases", {}).get(country.strip().title(), country.strip().upper())
            m &= o["country"].str.upper() == code
        if segment:
            m &= _match(o["segment"], segment)
        if stage:
            m &= o["pipeline_stage"] == stage
        if past_due:
            m &= o["past_due"]
        if orphaned:
            m &= o["vacant_seat"] & o["is_open"]
        if stalled:
            th = self.res.cfg["thresholds"]["stalled_age_days"]
            m &= o["pipeline_stage"].isin(self.res.cfg["stages"]["early"]) & (o["age_at_snapshot"] >= th)
        if forecast_category:
            m &= o["forecast_category_clean"] == forecast_category
        if min_amount_usd:
            m &= o["amount_usd"] >= float(min_amount_usd)
        hits = o[m]
        sort_col = {"weighted": "weighted_open_usd", "amount": "amount_usd", "days_past_due": "days_past_due",
                    "age": "age_at_snapshot"}.get(sort_by or "weighted", "weighted_open_usd")
        if status != "open" and sort_col == "weighted_open_usd":
            sort_col = "amount_usd"
        cols = ["opp_id", "account_name", "rep_name", "manager_name", "country", "pipeline_stage", "forecast_category_clean",
                "close_date", "days_past_due", "age_at_snapshot", "amount_usd", "win_probability", "weighted_open_usd"]
        if status in ("lost", "any"):
            cols.append("loss_reason")
        rows = hits.sort_values(sort_col, ascending=False)[cols]
        return {"matching_deals": int(len(hits)), "total_amount_usd": round(float(hits["amount_usd"].sum()), 0),
                "total_weighted_usd": round(float(hits["weighted_open_usd"].sum()), 0),
                "rows_shown": min(len(hits), min(int(limit or 10), 40)),
                "rows": _records(rows, min(int(limit or 10), 40))}

    def run_scenario(self, past_due_probability_multiplier: float = 1.0, assume_won=None, assume_lost=None,
                     exclude_orphaned_deals: bool = False) -> dict:
        o = self.res.opp.copy()
        mult = float(np.clip(past_due_probability_multiplier if past_due_probability_multiplier is not None else 1, 0, 1))
        w = np.where(o["past_due"], o["weighted_open_usd"] * mult, o["weighted_open_usd"])
        if exclude_orphaned_deals:
            w = np.where(o["vacant_seat"] & o["is_open"], 0.0, w)
        f = o["won_usd"] + w
        unknown = []
        for oid in assume_won or []:
            idx = o.index[o["opp_id"] == oid]
            if len(idx) == 0 or not o.loc[idx[0], "is_open"]:
                unknown.append(oid)
                continue
            f.loc[idx] = o.loc[idx, "amount_usd"]
        for oid in assume_lost or []:
            idx = o.index[o["opp_id"] == oid]
            if len(idx) == 0 or not o.loc[idx[0], "is_open"]:
                unknown.append(oid)
                continue
            f.loc[idx] = 0.0
        new = float(f.sum())
        out = {"base_forecast_usd": round(self.base_forecast, 0), "base_pct_of_quota": round(self.base_forecast / self.quota, 4),
               "scenario_forecast_usd": round(new, 0), "scenario_pct_of_quota": round(new / self.quota, 4),
               "scenario_gap_usd": round(new - self.quota, 0), "change_vs_base_usd": round(new - self.base_forecast, 0),
               "quota_usd": round(self.quota, 0)}
        if unknown:
            out["ignored_opp_ids_not_open_or_not_found"] = unknown
        return out

    def get_rep_profile(self, rep_name: str) -> dict:
        br = self.res.by_rep
        hits = [n for n in br.index if rep_name.lower() in n.lower()]
        if not hits:
            return {"error": f"No rep matching '{rep_name}'. Reps: {', '.join(br.index[:40])}"}
        n = hits[0]
        r = br.loc[n]
        conc = self.res.actions["concentration"]
        deals = self.find_deals(rep=n, limit=5)
        return {"rep": n, **{k: _clean(r[k]) for k in [
            "manager_name", "country", "segment", "headcount_status", "tenure_months", "quota_usd", "won_usd",
            "forecast_usd", "attainment", "gap_usd", "historical_attainment_pct_q_minus_1",
            "historical_attainment_pct_q_minus_2", "chronic_underperformer", "open_deals", "past_due_deals", "lost_usd"]},
            "holds_majority_of_team_pipeline": bool((conc["rep_name"] == n).any()),
            "largest_open_deals": deals["rows"], "other_matches": hits[1:]}

    def get_data_quality_issues(self, min_severity: str = "Info") -> dict:
        order = ["High", "Medium", "Low", "Info"]
        keep = order[: order.index(min_severity or "Info") + 1]
        dq = self.res.dq_log[self.res.dq_log["severity"].isin(keep)]
        return {"checks_fired": int(len(dq)), "rows": _records(dq)}

    def get_forecast_category_audit(self) -> dict:
        cf = self.res.actions["category_fixes"]
        return {"by_category": _records(self.res.category_audit.reset_index()),
                "rollup_methods": _records(self.res.reconciliation[["method", "clean_usd", "clean_pct_of_quota",
                                                                    "uses_forecast_category"]]),
                "category_fixes_by_team": _records(pd.crosstab(cf["manager_name"], cf["issue"]).reset_index())}

    def get_losses(self, manager: str | None = None) -> dict:
        o = self.res.opp
        lost = o[o["is_lost"]]
        closed = o[o["is_won"] | o["is_lost"]]
        if manager:
            lost = lost[_match(lost["manager_name"], manager)]
            closed = closed[_match(closed["manager_name"], manager)]
        lost = lost.assign(loss_reason=lost["loss_reason"].fillna("(blank)"))
        by = lost.groupby(["loss_reason", "deal_type"])["amount_usd"].agg(["count", "sum"]).reset_index() \
            .rename(columns={"count": "deals", "sum": "lost_usd"}).sort_values("lost_usd", ascending=False)
        won = closed.loc[closed["is_won"], "amount_usd"].sum()
        return {"lost_deals": int(len(lost)), "lost_usd": round(float(lost["amount_usd"].sum()), 0),
                "win_rate_usd": round(float(won / closed["amount_usd"].sum()), 4) if len(closed) else None,
                "by_reason_and_type": _records(by)}


# --------------------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------------------
class ForecastAgent:
    """Manual tool-use loop so the UI can show each step and every number can be traced to a tool output."""

    def __init__(self, res: fd.Results, client=None, model: str | None = None, effort: str | None = None):
        self.tools = ForecastTools(res)
        self.res = res
        agent_cfg = res.cfg.get("agent", {})
        self.model = model or agent_cfg.get("model", "claude-opus-5")
        self.effort = effort or agent_cfg.get("effort", "medium")
        self.messages: list[dict] = []
        if client is None:
            import anthropic
            load_env_file()
            client = anthropic.Anthropic()
        self.client = client

    def _tool_spec(self) -> list[dict]:
        """The tool list, with the stage names and countries of the data that was actually loaded."""
        import copy

        spec = copy.deepcopy(TOOLS)
        stages = self.res.cfg["stages"]
        countries = sorted(self.res.roster["country"].dropna().unique())
        for tool in spec:
            props = tool["input_schema"]["properties"]
            if "stage" in props:
                props["stage"]["enum"] = list(stages["open"]) + [stages["won"], stages["lost"]]
            if "country" in props:
                props["country"]["description"] = f"one of: {', '.join(countries)}"
        return spec

    def _call(self):
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            system=SYSTEM_PROMPT,
            tools=self._tool_spec(),
            messages=self.messages,
        )

    def ask(self, question: str, on_step=None) -> dict:
        """Returns {"text", "steps": [{"tool", "input", "output"}], "number_check": DataFrame, "stop_reason"}."""
        self.messages.append({"role": "user", "content": question})
        steps, outputs = [], []
        response = None
        for _ in range(MAX_TOOL_ROUNDS):
            response = self._call()
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                break
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    out = self.tools.run(block.name, block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out)})
                    outputs.append(out)
                except Exception as e:  # report the error to the model instead of crashing the chat
                    out = {"error": f"{type(e).__name__}: {e}"}
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out),
                                    "is_error": True})
                steps.append({"tool": block.name, "input": block.input, "output": out})
                if on_step:
                    on_step(steps[-1])
            self.messages.append({"role": "user", "content": results})
        else:
            # hit the round limit while still calling tools: ask for an answer with what it has
            self.messages.append({"role": "user", "content": "Please answer now with the information you have."})
            response = self._call()
            self.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            text = "The model declined this request. Try rephrasing the question."
        else:
            text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        check = fd.verify_numbers(text, {"facts": self.res.facts, "tools": outputs})
        return {"text": text, "steps": steps, "number_check": check, "stop_reason": response.stop_reason}

    def reset(self):
        self.messages = []


# --------------------------------------------------------------------------------------
# Offline answers (no API key / no internet): the same tools, formatted without an LLM
# --------------------------------------------------------------------------------------
QUICK_QUESTIONS = {
    "Why is the region missing its number?": ("get_headline", {}),
    "Where is the gap, by team?": ("get_gap_breakdown", {"dimension": "team"}),
    "Which past-due deals should managers chase first?": ("find_deals", {"past_due": True, "limit": 15}),
    "What if past-due deals close at half their odds?": ("run_scenario", {"past_due_probability_multiplier": 0.5}),
    "Which live deals have no owner?": ("find_deals", {"orphaned": True}),
    "Can we trust the Commit category?": ("get_forecast_category_audit", {}),
}


def _m(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x) / 1e6:.2f}M" if abs(x) >= 1e6 else f"{sign}${abs(x) / 1e3:,.0f}K"


def offline_answer(tools: ForecastTools, question: str) -> tuple[str, pd.DataFrame | None]:
    name, args = QUICK_QUESTIONS[question]
    out = tools.run(name, args)
    if name == "get_headline":
        return fd.template_summary(tools.res.facts), None
    if name == "get_gap_breakdown":
        return "Forecast vs quota by team, largest gap first:", pd.DataFrame(out["rows"])
    if name == "find_deals":
        return (f"**{out['matching_deals']} deals**, {_m(out['total_amount_usd'])} open, "
                f"{_m(out['total_weighted_usd'])} weighted. Top {out['rows_shown']}:"), pd.DataFrame(out["rows"])
    if name == "run_scenario":
        return (f"Base forecast **{out['base_pct_of_quota']:.1%}** -> scenario **{out['scenario_pct_of_quota']:.1%}** "
                f"({_m(out['scenario_forecast_usd'])}, change {_m(out['change_vs_base_usd'])})."), None
    if name == "get_forecast_category_audit":
        return ("Commit, Best Case and Pipeline carry almost the same average win probability, so the category "
                "is not a reliable basis for the call:"), pd.DataFrame(out["by_category"])
    return json.dumps(out)[:2000], None


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Why is the region missing its number, and where should the VP focus?"
    res = fd.run(use_ai=False, export_outputs=False)
    if not ai_available():
        print("No ANTHROPIC_API_KEY found (environment or the .env file next to this script). Offline answer:\n")
        print(offline_answer(ForecastTools(res), "Why is the region missing its number?")[0])
        sys.exit(0)
    agent = ForecastAgent(res)
    result = agent.ask(q, on_step=lambda s: print(f"  -> {s['tool']}({json.dumps(s['input'])})"))
    print("\n" + result["text"])
    chk = result["number_check"]
    print(f"\n[{int(chk['verified'].sum())}/{len(chk)} numbers traced to tool outputs]")
    if len(chk) and not chk["verified"].all():
        print("Not traced:", ", ".join(chk.loc[~chk["verified"], "number_in_text"]))
