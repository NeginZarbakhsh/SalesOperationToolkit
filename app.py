"""
Forecast Diagnostic: VP view.

Upload the two CSVs and the app answers, in this order:
    Will we hit the number?  ->  Why not, and where?  ->  What should we do?

Pages: Overview (the answer + what-if) | Teams | Pipeline | Data trust | Actions
Start: double-click Start_Forecast_App.bat, or `py -m streamlit run app.py`.
All numbers come from forecast_diagnostic.py; the optional Claude rewrite is number-checked against the data.
"""
import base64
import html
import io
import json
import os
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import agent as ag
import forecast_diagnostic as fd
from agent import ForecastTools, ai_available

HERE = Path(__file__).resolve().parent
LOGO = HERE / "assets" / "logo.png"
LOGO_B64 = base64.b64encode(LOGO.read_bytes()).decode()

# One blue for the brand and the data, red only for "behind", green only for "ahead"
BLUE, BLUE_DARK, RED, GREEN, AMBER = "#0A66C2", "#004182", "#CC1016", "#057642", "#E7A33E"
GREY, INK, INK2, GRID = "#CFCCC6", "#191919", "#5E5E5E", "#EEECE8"
FONT = "Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, Helvetica Neue, Arial, sans-serif"
PAGES = ["Overview", "Teams", "Pipeline", "Data trust", "Actions", "Ask AI"]

st.set_page_config(page_title="Forecast Diagnostic", page_icon=Image.open(LOGO), layout="wide")

st.html("""<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1360px;}
header[data-testid="stHeader"] {background: transparent;}
div[class*="st-key-card"] {background: #FFFFFF; border: none !important;
  box-shadow: 0 0 0 1px rgba(0,0,0,.08), 0 4px 14px rgba(0,0,0,.04);}
.topbar {display:flex; flex-wrap:wrap; align-items:center; gap:14px; background:#fff; border-radius:12px; padding:12px 18px;
  box-shadow: 0 0 0 1px rgba(0,0,0,.08), 0 4px 14px rgba(0,0,0,.04);}
.topbar img {width:42px; height:42px; border-radius:6px;}
.topbar .t1 {font-size:20px; font-weight:700; color:#191919; line-height:1.15;}
.topbar .t2 {font-size:13px; color:#5E5E5E;}
.chips {margin-left:auto; display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end;}
.chip {background:#E8F1FB; color:#0A66C2; border-radius:999px; padding:4px 12px; font-size:13px; font-weight:600;
  white-space:nowrap;}
.chip.grey {background:#F3F2EF; color:#404040;}
.chip.red {background:#FDECEC; color:#CC1016;}
.kicker {font-size:12px; letter-spacing:.07em; text-transform:uppercase; color:#5E5E5E; font-weight:700;}
.hero-num {font-size:64px; font-weight:700; line-height:1; color:#191919; margin-top:6px;}
.hero-num small {font-size:20px; font-weight:600; color:#5E5E5E; margin-left:10px;}
.hero-sub {font-size:15px; color:#404040; margin-top:8px;}
.bar {position:relative; display:flex; height:24px; border-radius:6px; background:#F3F2EF; margin:30px 0 10px;}
.seg {height:100%;}
.seg:first-child {border-radius:6px 0 0 6px;}
.won {background:#004182;} .exp {background:#0A66C2;} .risk {background:#E7A33E;}
.gap {background:repeating-linear-gradient(45deg,#E3E0DB,#E3E0DB 6px,#F3F2EF 6px,#F3F2EF 12px); border-radius:0 6px 6px 0;}
.quota {position:absolute; top:-8px; bottom:-8px; width:2px; background:#191919;}
.quota span {position:absolute; top:-22px; transform:translateX(-50%); font-size:12px; color:#191919;
  white-space:nowrap; font-weight:700;}
.legend {display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:#404040;}
.legend i {display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px;}
.answer {font-size:17px; color:#191919; margin-top:18px; line-height:1.5; padding-top:14px; border-top:1px solid #EEECE8;}
.scen-num {font-size:44px; font-weight:700; color:#191919; line-height:1.05;}
.scen-sub {font-size:14px; color:#5E5E5E;}
.neg {color:#CC1016; font-weight:700;} .pos {color:#057642; font-weight:700;}
.insight {min-height:262px; display:flex; flex-direction:column; gap:4px;}
.insight .impact {font-size:30px; font-weight:700; color:#CC1016; line-height:1.1; margin-top:8px;}
.insight .impact.neutral {color:#191919;}
.insight .impact-label {font-size:12px; color:#5E5E5E;}
.insight .title {font-size:16px; font-weight:700; color:#191919; margin-top:8px;}
.insight .body {font-size:14px; color:#404040; line-height:1.45;}
.tag {display:inline-block; font-size:11px; font-weight:700; letter-spacing:.06em; padding:3px 8px; border-radius:4px;}
.tag.structural {background:#FDECEC; color:#CC1016;} .tag.execution {background:#FCEFE6; color:#B24020;}
.tag.pipeline {background:#FFF4DA; color:#8A5A00;} .tag.data {background:#E8F1FB; color:#0A66C2;}
.h2 {font-size:22px; font-weight:700; color:#191919; margin:0;}
.muted {color:#5E5E5E; font-size:14px;}
.act {display:flex; gap:10px; align-items:flex-start; padding:10px 0; border-bottom:1px solid #EEECE8;}
.act:last-child {border-bottom:none;}
.act .dot {width:8px; height:8px; border-radius:50%; background:#0A66C2; margin-top:7px; flex:none;}
.act .txt {font-size:14px; color:#191919; line-height:1.45;}
.act .own {font-size:12px; color:#0A66C2; font-weight:600; margin-top:3px;}
.callout {background:#FDF3E1; border-radius:8px; padding:10px 14px; font-size:14px; color:#5A3B00;}
</style>""")


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Checking and analysing the data...")
def analyse(opp_bytes: bytes, roster_bytes: bytes, region: str, reported_pct: float | None) -> fd.Results:
    tmp = Path(tempfile.mkdtemp(prefix="forecast_"))
    (tmp / "opportunities.csv").write_bytes(opp_bytes)
    (tmp / "roster.csv").write_bytes(roster_bytes)
    cfg = fd.load_config(HERE / "config.json", opportunity_file=str(tmp / "opportunities.csv"),
                         roster_file=str(tmp / "roster.csv"), output_dir=str(tmp / "outputs"),
                         region_name=region, reported_forecast_pct=reported_pct)
    return fd.run(cfg, use_ai=False, export_outputs=True)


def identify(files) -> tuple:
    """Work out which uploaded CSV is the deals file and which is the roster, from their column names."""
    opp = roster = None
    notes = []
    for f in files:
        try:
            cols = set(pd.read_csv(io.BytesIO(f.getvalue()), nrows=0).columns.str.strip())
        except Exception:
            notes.append(f"**{f.name}** could not be read as a CSV file.")
            continue
        if {"opp_id", "pipeline_stage"} <= cols:
            opp = f
        elif {"rep_name", "quota_usd"} <= cols:
            roster = f
        else:
            notes.append(f"**{f.name}** doesn't look like a deals file (needs `opp_id`, `pipeline_stage`) or a roster "
                         f"(needs `rep_name`, `quota_usd`).")
    return opp, roster, notes


class LocalFile:
    """Test/screenshot mode only: when FORECAST_DEMO_FILES="deals.csv;roster.csv" is set, those files stand in for
    uploads. Never set in normal use, so the app always starts empty and waits for an upload."""

    def __init__(self, path):
        self.name, self._bytes = Path(path).name, Path(path).read_bytes()

    def getvalue(self):
        return self._bytes


DEMO_FILES = [LocalFile(p) for p in os.environ.get("FORECAST_DEMO_FILES", "").split(";") if p.strip()]


def m(x) -> str:
    x = float(x)
    s = "-" if x < 0 else ""
    return f"{s}${abs(x) / 1e6:.2f}M" if abs(x) >= 1e6 else f"{s}${abs(x) / 1e3:,.0f}K"


esc = html.escape


def md(text: str):
    st.markdown(text.replace("$", r"\$"))  # "$...$" would otherwise render as a LaTeX formula


def table(df: pd.DataFrame, money=(), pct=(), dec=(), height="auto", column_config=None):
    usd = lambda v: "-" if pd.isna(v) else (f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}")
    f = {c: usd for c in money if c in df.columns}
    f.update({c: "{:.0%}" for c in pct if c in df.columns})
    f.update({c: "{:.2f}" for c in dec if c in df.columns})
    st.dataframe(df.style.format(f, na_rep="-"), hide_index=True, width="stretch", height=height,
                 column_config=column_config)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def show(chart, key=None, on_select="ignore"):
    """One chart style everywhere: quiet axes, hairline grid, the app font."""
    chart = chart.configure(background="#FFFFFF").configure_view(strokeWidth=0).configure_axis(
        labelFont=FONT, titleFont=FONT, labelColor=INK2, titleColor=INK2, labelFontSize=12, titleFontSize=12,
        titleFontWeight="normal", gridColor=GRID, domainColor="#DDDAD5", tickColor="#DDDAD5", labelPadding=6,
    ).configure_legend(labelFont=FONT, labelColor=INK2, labelFontSize=12, orient="bottom", symbolType="square",
                       labelLimit=0) \
        .configure_text(font=FONT)
    return st.altair_chart(chart, theme=None, width="stretch", key=key, on_select=on_select)


def go(page: str, team: str | None = None):
    st.session_state["nav"] = page
    if team:
        st.session_state["team_sel"] = team
        st.session_state["view_sel"] = "Manager"


def sync(widget_key: str, state_key: str):
    st.session_state[state_key] = st.session_state[widget_key]


def on_pick(chart_key: str):
    """Click a team on a chart -> open that team's page."""
    event = st.session_state.get(chart_key) or {}
    points = (event.get("selection") or {}).get("pick") or []
    if points and points[0].get("manager"):
        go("Teams", points[0]["manager"])


def team_code(t) -> str:
    return f"{t['country']} {t['segment']}"


def chip(text, kind=""):
    return f'<span class="chip {kind}">{esc(str(text))}</span>'


# ---------------------------------------------------------------------------------------
# Top bar + data source
# ---------------------------------------------------------------------------------------
st.session_state.setdefault("nav", "Overview")
opp_up, roster_up, notes = identify(st.session_state.get("uploads") or DEMO_FILES)
ready = bool(opp_up and roster_up)

top_chips = ""
if ready and "last_facts" in st.session_state:
    lf = st.session_state["last_facts"]
    top_chips = ((chip(lf["region"]) if lf["region"] != "Region" else "") + chip(f"Data as of {lf['snapshot_date']}", "grey")
                 + chip(f"{lf['days_left_in_quarter']} days left in quarter", "red"))
st.html(f"""<div class="topbar"><img src="data:image/png;base64,{LOGO_B64}">
  <div><div class="t1">Forecast Diagnostic</div><div class="t2">Sales Operations</div></div>
  <div class="chips">{top_chips}</div></div>""")

label = (f"📁  Data: {opp_up.name}  +  {roster_up.name}   (click to change files or settings)" if ready
         else "📁  Upload your data")
with st.expander(label, expanded=not ready):
    u1, u2 = st.columns([3, 1.3])
    with u1:
        uploads = st.file_uploader("Drop both CSV files here: the deals file and the sales-team file, in any order",
                                   type="csv", accept_multiple_files=True, key="uploads")
    with u2:
        region = st.text_input("Region name (optional)", placeholder="e.g. APAC", key="region").strip() or "Region"
        reported = st.number_input("Reported forecast % (optional)", min_value=0.0, max_value=300.0, value=None,
                                   step=0.1, format="%.1f", placeholder="e.g. 91", key="reported",
                                   help="If leadership quotes a forecast %, the app checks which way of adding up "
                                        "the pipeline reproduces it.")
opp_up, roster_up, notes = identify(uploads or DEMO_FILES)
for n in notes:
    st.warning(n)

if not (opp_up and roster_up):
    if uploads:
        missing = "sales-team roster" if opp_up else "deals file" if roster_up else "deals file and sales-team roster"
        st.info(f"Got it. Now add the **{missing}**.")
    with st.container(border=True, key="card_welcome"):
        st.html("""<div class="kicker">For the VP of Sales</div>
          <div class="h2" style="font-size:30px;margin:6px 0 4px">Will we hit the number, why not, and what do we do?</div>
          <div class="muted">Upload the deals export and the sales-team roster. In a few seconds you get the answer,
          where the gap is, and an action plan. Every number is checked against the data.</div>""")
        w1, w2, w3 = st.columns(3)
        for col, (k, t, b) in zip((w1, w2, w3), [
                ("1", "The answer", "Forecast vs quota, how much of it is at risk, and a what-if you can play with."),
                ("2", "Where the gap is", "By team, market, segment, rep, pipeline stage and cohort. Click a team to "
                                          "drill in."),
                ("3", "What to do", "A 14 / 30 / 60-day plan with owners, plus deal lists each manager can download.")]):
            col.html(f"""<div style="padding:14px 0"><span class="chip">{k}</span>
              <div style="font-weight:700;font-size:16px;margin-top:10px">{t}</div>
              <div class="muted" style="margin-top:4px">{b}</div></div>""")
        with st.expander("What the two files need to contain"):
            c1, c2 = st.columns(2)
            c1.markdown("**Deals file** (one row per opportunity)  \n" + ", ".join(f"`{c}`" for c in fd.REQUIRED_OPP_COLS))
            c2.markdown("**Sales-team file** (one row per rep or open seat)  \n"
                        + ", ".join(f"`{c}`" for c in fd.REQUIRED_ROSTER_COLS))
    st.stop()

try:
    res = analyse(opp_up.getvalue(), roster_up.getvalue(), region, reported / 100 if reported else None)
except Exception as e:
    st.error(f"These files could not be analysed: {e}")
    st.stop()

F = res.facts
if st.session_state.get("last_facts") is not F:  # refresh the top-bar chips once for new data
    st.session_state["last_facts"] = F
    st.rerun()
tools = ForecastTools(res)
TEAMS = {t["manager"]: t for t in F["teams"]}
ORDERED_TEAMS = [t["manager"] for t in sorted(F["teams"], key=lambda t: t["gap_usd"])]
if st.session_state.get("team_sel") not in TEAMS:  # first load, or new files with different teams
    st.session_state["team_sel"] = ORDERED_TEAMS[0]
st.session_state.setdefault("view_sel", "Manager")

st.segmented_control("Page", PAGES, key="nav", required=True, label_visibility="collapsed", width="stretch")
page = st.session_state["nav"]


# ---------------------------------------------------------------------------------------
# Insights: the 3-4 things a VP must know, generated from this data and ranked by $ impact
# ---------------------------------------------------------------------------------------
def build_insights() -> list[dict]:
    out = []
    v, ph, b = F["vacancy"], F["pipeline_hygiene"], res.bridge
    if v["vacant_seats"]:
        vac = b[b["seat"] == "vacant seats"].sort_values("gap_usd").iloc[0]
        mgr = vac["manager_name"]
        act = b[(b["manager_name"] == mgr) & (b["seat"] == "active reps")]
        body = f"{m(v['vacant_quota_usd'])} of quota has no rep behind it."
        if v["orphaned_open_deals"]:
            body += f" {v['orphaned_open_deals']} live deals ({m(v['orphaned_open_usd'])}) have no owner."
        if len(act) and act["quota_usd"].sum():
            body += f" The team's active reps are at {act['forecast_usd'].sum() / act['quota_usd'].sum():.0%} of quota."
        out.append(dict(tag="structural", title=f"{team_code(TEAMS[mgr])}: {v['vacant_seats']} empty seats",
                        impact=m(v["gap_on_vacant_seats_usd"]), label=f"{F['derived']['vacant_seats_share_of_gap']:.0%} "
                        f"of the gap", body=body, team=mgr, rank=abs(v["gap_on_vacant_seats_usd"])))
    act = b[b["seat"] == "active reps"].sort_values("gap_usd")
    if len(act) and act.iloc[0]["gap_usd"] < 0:
        mgr = act.iloc[0]["manager_name"]
        t = TEAMS[mgr]
        lost = res.opp[res.opp["is_lost"] & (res.opp["manager_name"] == mgr)]
        body = f"Forecast at {t['attainment']:.0%} of quota."
        title = f"{team_code(t)}: furthest behind"
        if len(lost):
            dtype = lost.groupby("deal_type")["amount_usd"].sum().idxmax()
            reason = lost.assign(r=lost["loss_reason"].fillna("reason not logged")).groupby("r")["amount_usd"].sum().idxmax()
            title = f"{team_code(t)}: {m(lost['amount_usd'].sum())} lost this quarter"
            body += f" Most losses are {dtype.lower()} deals; the top reason is '{reason.lower()}'."
        conc = res.actions["concentration"]
        if (conc["manager_name"] == mgr).any():
            c = conc[conc["manager_name"] == mgr].iloc[0]
            body += f" {c['rep_name']} holds {c['share_of_team']:.0%} of the team's pipeline."
        out.append(dict(tag="execution", title=title, impact=m(act.iloc[0]["gap_usd"]),
                        label=f"{t['share_of_gap']:.0%} of the gap", body=body, team=mgr,
                        rank=abs(act.iloc[0]["gap_usd"])))
    if ph["past_due_weighted_usd"] > 0:
        stalest = max(F["teams"], key=lambda x: x["past_due_share_of_weighted"] or 0)
        at_risk = ph["past_due_weighted_usd"] * res.cfg["thresholds"]["past_due_haircut"]
        out.append(dict(tag="pipeline", title="Stale deals prop up the forecast", impact=m(-at_risk),
                        label="at risk if half of them slip",
                        body=f"{ph['past_due_deals']} of {ph['open_deals']} open deals are past their close date "
                             f"({m(ph['past_due_weighted_usd'])} expected, {ph['past_due_share_of_forecast']:.0%} of the "
                             f"forecast). Worst: {team_code(stalest)}, {stalest['past_due_share_of_weighted']:.0%} past due.",
                        team=stalest["manager"], rank=at_risk))
    # what explains the gap (largest first), then what puts the forecast at risk, then data caveats
    out.sort(key=lambda x: (x["tag"] == "pipeline", -x["rank"]))
    fc = F["forecast_category"]
    if fc["commit_on_early_stage_deals"] or fc["late_stage_omitted_or_blank_deals"]:
        cat = {k: v for k, v in F["rollup_methods_pct_of_quota"].items() if "category" in k}
        wp = F["forecast_category"]["avg_win_prob_by_category"]
        lo, hi = min(cat.values()), max(cat.values())
        out.append(dict(tag="data", title="The 'Commit' label can't be trusted",
                        impact=f"{lo:.0%} to {hi:.0%}", label="what the reps' labels imply", neutral=True,
                        body=f"'Commit' deals average {wp.get('Commit', 0):.0%} win probability, the same as "
                             f"'Pipeline' ({wp.get('Pipeline', 0):.0%}). {fc['commit_on_early_stage_deals']} Commit deals "
                             f"are still early stage. The forecast here uses stage and win probability instead.",
                        team=None, rank=0))
    return out[:4]


# ---------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------
def bridge_chart():
    q = F["total_quota_usd"]
    rows, level = [dict(label=f"Quota   {m(q)}", start=0, end=q, kind="Quota", manager="", value=q)], q
    for r in res.bridge.itertuples():
        t = TEAMS[r.manager_name]
        suffix = " · empty seats" if r.seat == "vacant seats" else (
            " · active reps" if t["vacant_seats"] else "")
        rows.append(dict(label=f"{team_code(t)}{suffix}   {'+' if r.gap_usd > 0 else ''}{m(r.gap_usd)}",
                         start=level, end=level + r.gap_usd, kind="Ahead" if r.gap_usd > 0 else "Behind",
                         manager=r.manager_name, value=r.gap_usd))
        level += r.gap_usd
    rows.append(dict(label=f"Forecast   {m(level)}", start=0, end=level, kind="Forecast", manager="", value=level))
    d = pd.DataFrame(rows)
    lo = min(d.loc[d["kind"].isin(["Ahead", "Behind"]), ["start", "end"]].min().min(), level) - 0.04 * q
    hi = max(q, level) * 1.01
    sel = alt.selection_point(fields=["manager"], name="pick", on="click")
    return alt.Chart(d).mark_bar(cornerRadius=3, height=20).encode(
        y=alt.Y("label:N", sort=d["label"].tolist(), title=None, axis=alt.Axis(labelLimit=320, domain=False, ticks=False)),
        x=alt.X("start:Q", scale=alt.Scale(domain=[lo, hi], clamp=True), title=None, axis=alt.Axis(format="$~s")),
        x2="end:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Quota", "Behind", "Ahead", "Forecast"],
                                                  range=[GREY, RED, GREEN, BLUE]), legend=None),
        opacity=alt.condition(sel, alt.value(1), alt.value(0.35)),
        tooltip=[alt.Tooltip("label:N", title=" "), alt.Tooltip("value:Q", format="$,.0f", title="Amount")],
    ).add_params(sel).properties(height=44 * len(d))


def attainment_chart(d: pd.DataFrame, name_col: str, pick_field: str | None = None):
    d = d.assign(txt=[f"{a:.0%}  ({m(g)})" for a, g in zip(d["attainment"], d["gap_usd"])],
                 status=np.where(d["attainment"] < 0.9, "Below 90% of quota", "90% or more"))
    order = d.sort_values("attainment")[name_col].tolist()
    y = alt.Y(f"{name_col}:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, domain=False, ticks=False))
    base = alt.Chart(d).encode(y=y)
    bars = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("attainment:Q", title="Forecast as % of quota", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, max(1.25, float(d["attainment"].max()) + 0.15)])),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Below 90% of quota", "90% or more"], range=[RED, BLUE]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip(f"{name_col}:N", title=" "), alt.Tooltip("quota_usd:Q", format="$,.0f", title="Quota"),
                 alt.Tooltip("forecast_usd:Q", format="$,.0f", title="Forecast"),
                 alt.Tooltip("attainment:Q", format=".1%", title="% of quota")])
    if pick_field:
        sel = alt.selection_point(fields=[pick_field], name="pick", on="click")
        bars = bars.encode(opacity=alt.condition(sel, alt.value(1), alt.value(0.35))).add_params(sel)
    text = base.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(x="attainment:Q", text="txt:N")
    rule = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(strokeDash=[4, 4], color=INK2).encode(x="x:Q")
    return (bars + text + rule).properties(height=44 * len(d) + 20)


def stacked_pipeline_chart(d: pd.DataFrame, cat: str, order):
    long = pd.DataFrame({
        cat: np.repeat(d[cat].values, 2),
        "status": ["Close date still ahead", "Close date already passed"] * len(d),
        "usd": np.ravel(np.column_stack([d["weighted_usd"] - d["past_due_weighted_usd"], d["past_due_weighted_usd"]])),
    })
    return alt.Chart(long).mark_bar(height=26, cornerRadius=2).encode(
        y=alt.Y(f"{cat}:N", sort=order, title=None, axis=alt.Axis(domain=False, ticks=False)),
        x=alt.X("usd:Q", stack=True, title="Expected value of open deals (amount x win probability)",
                axis=alt.Axis(format="$~s")),
        color=alt.Color("status:N", scale=alt.Scale(domain=["Close date still ahead", "Close date already passed"],
                                                     range=[BLUE, AMBER]), legend=alt.Legend(title=None)),
        order=alt.Order("status:N", sort="descending"),
        tooltip=[alt.Tooltip(f"{cat}:N", title=" "), "status", alt.Tooltip("usd:Q", format="$,.0f", title="Expected $")],
    ).properties(height=62 * len(d) + 90)


# ---------------------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------------------
def page_overview():
    q, fc, won = F["total_quota_usd"], F["forecast_usd"], F["closed_won_usd"]
    ph = F["pipeline_hygiene"]
    pdw = ph["past_due_weighted_usd"]
    exp = F["weighted_open_pipeline_usd"] - pdw
    gap = max(q - fc, 0)
    scale = max(q, fc) * 1.02
    w = lambda x: f"{100 * x / scale:.2f}%"
    behind = [t for t in sorted(F["teams"], key=lambda t: t["gap_usd"]) if t["gap_usd"] < 0]
    region = esc(F["region"])
    if gap > 0:
        answer = f"<b>{region} is {m(gap)} short</b> with {F['days_left_in_quarter']} days left."
        if len(behind) >= 2:
            answer += (f" {F['derived']['top2_teams_share_of_gap']:.0%} of the gap sits in <b>{esc(team_code(behind[0]))}</b>"
                       f" and <b>{esc(team_code(behind[1]))}</b>.")
        if pdw > 0:
            answer += (f" {ph['past_due_share_of_forecast']:.0%} of the forecast depends on deals that have already "
                       f"missed their close date.")
    else:
        answer = f"<b>{region} is on track</b>: {m(fc - q)} ahead of quota with {F['days_left_in_quarter']} days left."

    with st.container(border=True, key="card_hero"):
        c1, c2 = st.columns([1.75, 1], gap="large")
        with c1:
            st.html(f"""
            <div class="kicker">Forecast vs quota</div>
            <div class="hero-num">{F['forecast_pct_of_quota'] * 100:.1f}%<small>{m(fc)} of {m(q)}</small></div>
            <div class="hero-sub">{'<span class="neg">' + m(-gap) + ' vs quota</span>' if gap > 0 else
                                   '<span class="pos">+' + m(fc - q) + ' vs quota</span>'}
              &nbsp;·&nbsp; {F['days_left_in_quarter']} days left in the quarter</div>
            <div class="bar">
              <div class="seg won" style="width:{w(won)}" title="Closed won {m(won)}"></div>
              <div class="seg exp" style="width:{w(exp)}" title="Expected from on-time deals {m(exp)}"></div>
              <div class="seg risk" style="width:{w(pdw)}" title="Expected from deals past their close date {m(pdw)}"></div>
              <div class="seg gap" style="width:{w(gap)}" title="Gap to quota {m(gap)}"></div>
              <div class="quota" style="left:{w(q)}"><span>Quota {m(q)}</span></div>
            </div>
            <div class="legend">
              <span><i style="background:{BLUE_DARK}"></i>Already won {m(won)}</span>
              <span><i style="background:{BLUE}"></i>Expected, on time {m(exp)}</span>
              <span><i style="background:{AMBER}"></i>Expected, but past close date {m(pdw)}</span>
              {'<span><i style="background:#E3E0DB"></i>Gap ' + m(gap) + '</span>' if gap > 0 else ''}
            </div>
            <div class="answer">{answer}</div>""")
        with c2:
            st.html('<div class="kicker">What if...</div>')
            mult = st.slider("Deals past their close date close at this share of their stated odds", 0, 100, 100,
                             step=10, format="%d%%", key="whatif_mult",
                             help="100% = the forecast as reported. Try 50%: stale deals rarely close at full odds.")
            drop = st.toggle(f"Deals with no owner are lost ({F['vacancy']['orphaned_open_deals']})",
                             key="whatif_drop", disabled=F["vacancy"]["orphaned_open_deals"] == 0)
            s = tools.run_scenario(past_due_probability_multiplier=mult / 100, exclude_orphaned_deals=drop)
            chg = s["change_vs_base_usd"]
            st.html(f"""<div style="margin-top:6px"><div class="scen-num">{s['scenario_pct_of_quota'] * 100:.1f}%</div>
              <div class="scen-sub">{m(s['scenario_forecast_usd'])} forecast &nbsp;·&nbsp;
              <span class="{'neg' if chg < 0 else 'muted'}">{m(chg) if chg else 'no change'} vs reported</span></div>
              <div class="scen-sub" style="margin-top:10px">Realistic range to call:
              <b>{F['scenarios'][1]['pct_of_quota']:.0%} to {F['forecast_pct_of_quota']:.0%}</b></div></div>""")

    st.html('<div class="kicker" style="margin:22px 0 4px">Where to focus, ranked by impact</div>')
    ins = build_insights()
    cols = st.columns(len(ins) or 1)
    for i, (col, x) in enumerate(zip(cols, ins)):
        with col.container(border=True, key=f"card_ins{i}"):
            st.html(f"""<div class="insight"><div><span class="tag {x['tag']}">{x['tag'].upper()}</span></div>
              <div class="impact {'neutral' if x.get('neutral') else ''}">{esc(x['impact'])}</div>
              <div class="impact-label">{esc(x['label'])}</div>
              <div class="title">{esc(x['title'])}</div><div class="body">{esc(x['body'])}</div></div>""")
            if x["team"]:
                st.button(f"Open {team_code(TEAMS[x['team']])} →", key=f"ins_btn{i}", on_click=go,
                          args=("Teams", x["team"]), width="stretch")
            else:
                st.button("See the data checks →", key=f"ins_btn{i}", on_click=go, args=("Data trust",),
                          width="stretch")

    c1, c2 = st.columns([1.15, 1])
    with c1.container(border=True, key="card_bridge"):
        st.html('<div class="h2" style="font-size:17px">Quota to forecast, team by team</div>'
                '<div class="muted">Click a team to open it</div>')
        show(bridge_chart(), key="bridge_pick", on_select=lambda: on_pick("bridge_pick"))
    with c2.container(border=True, key="card_att"):
        st.html('<div class="h2" style="font-size:17px">Forecast as % of quota</div>'
                '<div class="muted">Click a team to open it</div>')
        d = res.by_manager.reset_index().rename(columns={"manager_name": "manager"})
        d["team"] = [f"{team_code(TEAMS[x])} · {x}" for x in d["manager"]]
        show(attainment_chart(d, "team", pick_field="manager"), key="att_pick", on_select=lambda: on_pick("att_pick"))


# ---------------------------------------------------------------------------------------
# Page: Teams
# ---------------------------------------------------------------------------------------
def page_teams():
    # widget keys include the current value, so a selection made elsewhere (e.g. a chart click) always shows
    vk = f"view_{st.session_state['view_sel']}"
    view = st.segmented_control("View by", ["Manager", "Market", "Segment"], default=st.session_state["view_sel"],
                                key=vk, required=True, on_change=sync, args=(vk, "view_sel"))
    if view in ("Market", "Segment"):
        src = (res.by_country if view == "Market" else res.by_segment).reset_index()
        key = src.columns[0]
        worst = src.sort_values("gap_usd").iloc[0]
        with st.container(border=True, key="card_dim"):
            md(f"**{worst[key]}** has the largest gap: **{m(worst['gap_usd'])}** at **{worst['attainment']:.0%}** of "
               f"quota ({worst['share_of_gap']:.0%} of the total gap).")
            show(attainment_chart(src, key))
            show_cols = src[[key, "quota_usd", "forecast_usd", "attainment", "gap_usd", "share_of_gap",
                             "past_due_share_of_weighted", "win_rate_usd", "lost_usd"]]
            show_cols.columns = [view, "Quota", "Forecast", "% of quota", "Gap", "Share of gap", "Pipeline past due",
                                 "Win rate", "Lost this quarter"]
            table(show_cols.sort_values("Gap"), money=("Quota", "Forecast", "Gap", "Lost this quarter"),
                  pct=("% of quota", "Share of gap", "Pipeline past due", "Win rate"))
        return

    tk = f"team_{st.session_state['team_sel']}"
    st.segmented_control("Team", ORDERED_TEAMS, default=st.session_state["team_sel"], key=tk, required=True,
                         on_change=sync, args=(tk, "team_sel"),
                         format_func=lambda x: f"{team_code(TEAMS[x])} · {x.split()[-1]}")
    mgr = st.session_state["team_sel"]
    t = TEAMS[mgr]
    r = res.by_manager.loc[mgr]
    with st.container(border=True, key="card_team"):
        st.html(f"""<div class="kicker">Team</div><div style="display:flex;align-items:center;gap:10px;margin-top:4px">
          <div class="h2">{esc(mgr)}</div>{chip(team_code(t))}{chip(f"{int(r['seats'])} seats", "grey")}
          {chip(f"{int(r['vacant_seats'])} vacant", "red") if r['vacant_seats'] else ''}</div>""")
        k = st.columns(5)
        k[0].metric("Forecast vs quota", f"{r['attainment']:.0%}", f"{m(r['gap_usd'])} vs quota")
        k[1].metric("Quota", m(r["quota_usd"]))
        k[2].metric("Open pipeline", m(r["open_pipeline_usd"]), f"{int(r['open_deals'])} deals", delta_color="off",
                    delta_arrow="off")
        k[3].metric("Pipeline past close date", f"{(r['past_due_share_of_weighted'] or 0):.0%}",
                    f"{int(r['past_due_deals'])} deals", delta_color="off", delta_arrow="off")
        k[4].metric("Win rate (by $)", f"{(r['win_rate_usd'] or 0):.0%}", f"{m(r['lost_usd'])} lost", delta_color="off",
                    delta_arrow="off")
        if r["vacant_seats"]:
            b = res.bridge[(res.bridge["manager_name"] == mgr)]
            a = b[b["seat"] == "active reps"]
            v = b[b["seat"] == "vacant seats"]
            st.html(f"""<div class="callout"><b>Coverage gap, not a performance gap:</b> {int(r['vacant_seats'])} empty
              seats carry {m(v['quota_usd'].sum())} of quota ({m(v['gap_usd'].sum())} vs quota). The active reps are
              at {a['forecast_usd'].sum() / a['quota_usd'].sum():.0%} of their own quota.</div>""")

    c1, c2 = st.columns([1.2, 1])
    with c1.container(border=True, key="card_reps"):
        st.html('<div class="h2" style="font-size:17px">Reps: this quarter vs the last two</div>'
                '<div class="muted">Bar = forecast this quarter. Marks = attainment in the previous two quarters.</div>')
        reps = res.by_rep[res.by_rep["manager_name"] == mgr].reset_index()
        reps["rep"] = [("Empty seat (" + n.split("(")[1].split(" backfill")[0] + ")") if "OPEN REQ" in n else n
                       for n in reps["rep_name"]]
        reps["status"] = np.where(reps["headcount_status"] != res.cfg["active_headcount_status"], "Empty seat",
                                  np.where(reps["attainment"] < 0.9, "Below 90%", "90% or more"))
        order = reps.sort_values("attainment")["rep"].tolist()
        y = alt.Y("rep:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, domain=False, ticks=False))
        bars = alt.Chart(reps).mark_bar(cornerRadiusEnd=4, height=20).encode(
            y=y, x=alt.X("attainment:Q", title="% of quota", axis=alt.Axis(format="%"),
                         scale=alt.Scale(domain=[0, max(1.5, float(reps["attainment"].max()) + 0.1)])),
            color=alt.Color("status:N", scale=alt.Scale(domain=["Below 90%", "90% or more", "Empty seat"],
                                                        range=[RED, BLUE, GREY]), legend=alt.Legend(title=None)),
            tooltip=["rep", alt.Tooltip("attainment:Q", format=".0%", title="This quarter"),
                     alt.Tooltip("historical_attainment_pct_q_minus_1:Q", format=".0%", title="Last quarter"),
                     alt.Tooltip("historical_attainment_pct_q_minus_2:Q", format=".0%", title="Two quarters ago"),
                     "tenure_months"])
        hist = reps[reps["status"] != "Empty seat"].melt(
            id_vars=["rep"], value_vars=["historical_attainment_pct_q_minus_1", "historical_attainment_pct_q_minus_2"],
            var_name="quarter", value_name="att")
        hist["quarter"] = hist["quarter"].map({"historical_attainment_pct_q_minus_1": "Last quarter",
                                               "historical_attainment_pct_q_minus_2": "Two quarters ago"})
        ticks = alt.Chart(hist).mark_tick(thickness=3, size=18).encode(
            y=y, x="att:Q", color=alt.Color("quarter:N", scale=alt.Scale(domain=["Last quarter", "Two quarters ago"],
                                                                         range=[INK, "#9A9791"]),
                                            legend=alt.Legend(title=None)))
        rule = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(strokeDash=[4, 4], color=INK2).encode(x="x:Q")
        show((bars + ticks + rule).resolve_scale(color="independent").properties(height=40 * len(reps) + 30))
    with c2.container(border=True, key="card_losses"):
        st.html('<div class="h2" style="font-size:17px">Lost this quarter, by reason</div>')
        lost = res.opp[res.opp["is_lost"] & (res.opp["manager_name"] == mgr)]
        if len(lost):
            lz = lost.assign(reason=lost["loss_reason"].fillna("Not logged")).groupby(["reason", "deal_type"],
                                                                                        as_index=False)["amount_usd"].sum()
            order = lz.groupby("reason")["amount_usd"].sum().sort_values(ascending=False).index.tolist()
            show(alt.Chart(lz).mark_bar(height=18, cornerRadius=2).encode(
                y=alt.Y("reason:N", sort=order, title=None, axis=alt.Axis(labelLimit=200, domain=False, ticks=False)),
                x=alt.X("amount_usd:Q", title=None, axis=alt.Axis(format="$~s")),
                color=alt.Color("deal_type:N", legend=alt.Legend(title=None),
                                scale=alt.Scale(range=[BLUE, "#70B5F9", BLUE_DARK, "#9A9791"])),
                tooltip=["reason", "deal_type", alt.Tooltip("amount_usd:Q", format="$,.0f", title="Lost")],
            ).properties(height=36 * lz["reason"].nunique() + 40))
        else:
            st.caption("No lost deals this quarter.")

    with st.container(border=True, key="card_deals"):
        st.html('<div class="h2" style="font-size:17px">Deals to work, largest expected value first</div>')
        o = res.opp[res.opp["is_open"] & (res.opp["manager_name"] == mgr)].copy()
        fix_ids = set(res.actions["category_fixes"]["opp_id"])
        stall_ids = set(res.actions["stalled_early_stage"]["opp_id"])
        o["Flags"] = [", ".join(f for f, on in [("No owner", vs), (f"Past due {int(dp)}d", pd_), ("Label mismatch", i in fix_ids),
                                                 ("Stalled", i in stall_ids)] if on)
                      for vs, pd_, dp, i in zip(o["vacant_seat"], o["past_due"], o["days_past_due"], o["opp_id"])]
        o = o.sort_values("weighted_open_usd", ascending=False)
        d = pd.DataFrame({"Deal": o["opp_id"], "Account": o["account_name"], "Owner": o["rep_name"],
                          "Stage": o["pipeline_stage"], "Close date": o["close_date"].dt.date, "Flags": o["Flags"],
                          "Amount": o["amount_usd"], "Win probability": (o["win_probability"] * 100).round(0),
                          "Expected value": o["weighted_open_usd"]})
        table(d, money=("Amount", "Expected value"), height=min(38 * len(d) + 40, 460), column_config={
            "Win probability": st.column_config.ProgressColumn("Win probability", min_value=0, max_value=100,
                                                               format="%d%%")})


# ---------------------------------------------------------------------------------------
# Page: Pipeline (stage + cohort)
# ---------------------------------------------------------------------------------------
def page_pipeline():
    ph = F["pipeline_hygiene"]
    k = st.columns(4)
    k[0].metric("Open pipeline", m(F["open_pipeline_usd"]), f"{ph['open_deals']} deals", delta_color="off",
                delta_arrow="off")
    k[1].metric("Expected value", m(F["weighted_open_pipeline_usd"]), "amount x win probability", delta_color="off",
                delta_arrow="off")
    k[2].metric("Coverage of remaining quota", f"{F['coverage_of_remaining_quota_x']:.1f}x")
    k[3].metric("Expected value past close date", f"{ph['past_due_share_of_weighted_pipeline']:.0%}",
                f"{ph['past_due_deals']} deals", delta_color="off", delta_arrow="off")

    c1, c2 = st.columns([1, 1])
    with c1.container(border=True, key="card_stage"):
        s = res.health["by_stage"].reset_index()
        top = s.sort_values("past_due_weighted_usd", ascending=False).iloc[0]
        st.html('<div class="h2" style="font-size:17px">By stage</div>')
        md(f"The biggest block of stale value is in **{top['pipeline_stage']}**: {int(top['past_due_deals'])} deals, "
           f"{m(top['past_due_weighted_usd'])} expected, already past their close date.")
        show(stacked_pipeline_chart(s, "pipeline_stage", res.cfg["stages"]["open"]))
    with c2.container(border=True, key="card_cohort"):
        c = res.health["by_cohort"].reset_index()
        c["past_due_weighted_usd"] = [res.opp.loc[res.opp["is_open"] & (res.opp["cohort"] == x),
                                                  "past_due_weighted_usd"].sum() for x in c["cohort"]]
        st.html('<div class="h2" style="font-size:17px">By cohort (when the deal was created)</div>')
        pre = c[c["cohort"].str.contains("before")]
        if len(pre):
            md(f"**{int(pre.iloc[0]['past_due_deals'])} of the {ph['past_due_deals']}** past-due deals were created "
               f"before the quarter started. They are carried forward with stale close dates.")
        show(stacked_pipeline_chart(c, "cohort", ["Created before quarter", "Created in quarter"]))

    with st.container(border=True, key="card_heat"):
        st.html('<div class="h2" style="font-size:17px">Team x stage: where the stale value sits</div>'
                '<div class="muted">Number = open pipeline. Colour = share of expected value past its close date.</div>')
        op = res.opp[res.opp["is_open"]]
        hm = op.groupby(["manager_name", "pipeline_stage"], as_index=False).agg(
            open_usd=("amount_usd", "sum"), w=("weighted_open_usd", "sum"), pdw=("past_due_weighted_usd", "sum"),
            deals=("opp_id", "count"))
        hm["share"] = hm["pdw"] / hm["w"].replace(0, np.nan)
        hm["team"] = [f"{team_code(TEAMS[x])} · {x}" for x in hm["manager_name"]]
        hm["txt"] = hm["open_usd"].map(m)
        base = alt.Chart(hm).encode(
            x=alt.X("pipeline_stage:N", sort=res.cfg["stages"]["open"], title=None,
                    axis=alt.Axis(orient="top", labelAngle=0, domain=False, ticks=False)),
            y=alt.Y("team:N", sort=[f"{team_code(TEAMS[x])} · {x}" for x in ORDERED_TEAMS], title=None,
                    axis=alt.Axis(labelLimit=260, domain=False, ticks=False)))
        rect = base.mark_rect(cornerRadius=4, stroke="#FFFFFF", strokeWidth=3).encode(
            color=alt.Color("share:Q", title="Past close date", scale=alt.Scale(domain=[0, 1], range=["#FFF6E5", "#B24020"]),
                            legend=alt.Legend(format="%", gradientLength=220)),
            tooltip=["team", "pipeline_stage", "deals", alt.Tooltip("open_usd:Q", format="$,.0f", title="Open pipeline"),
                     alt.Tooltip("share:Q", format=".0%", title="Past close date")])
        text = base.mark_text(fontSize=12, fontWeight=600).encode(
            text="txt:N", color=alt.condition("datum.share > 0.6", alt.value("#FFFFFF"), alt.value(INK)))
        show((rect + text).properties(height=52 * hm["team"].nunique() + 30))

    with st.container(border=True, key="card_cat"):
        st.html('<div class="h2" style="font-size:17px">Can we trust the reps\' forecast category?</div>'
                '<div class="muted">If the labels meant something, Commit deals would have a much higher win '
                'probability than Pipeline deals.</div>')
        ca = res.category_audit.reset_index().rename(columns={"forecast_category_clean": "category"})
        ca["txt"] = [f"{a:.0%}  ({n} deals)" for a, n in zip(ca["avg_win_prob"], ca["deals"])]
        b = alt.Chart(ca).encode(y=alt.Y("category:N", sort=None, title=None, axis=alt.Axis(domain=False, ticks=False)))
        show((b.mark_bar(height=22, cornerRadiusEnd=4, color=BLUE).encode(
            x=alt.X("avg_win_prob:Q", title="Average win probability", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1]))) + b.mark_text(align="left", dx=6, color=INK).encode(
            x="avg_win_prob:Q", text="txt:N")).properties(height=44 * len(ca) + 20))


# ---------------------------------------------------------------------------------------
# Page: Data trust
# ---------------------------------------------------------------------------------------
def page_data():
    dq = res.dq_log
    high = dq[dq["severity"] == "High"]
    rc = res.reconciliation.set_index("method")
    stage_row = rc.loc[[i for i in rc.index if "win probability" in i][0]]
    with st.container(border=True, key="card_trust"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Data checks that found something", len(dq), f"{len(high)} high severity", delta_color="off",
                  delta_arrow="off")
        c2.metric("Forecast after cleaning", f"{stage_row.clean_pct_of_quota:.1%}",
                  f"{stage_row.raw_pct_of_quota:.1%} on the raw file", delta_color="off", delta_arrow="off")
        fixed = dq[dq["action"].str.startswith(("Removed", "Corrected"))]["rows_affected"].sum()
        c3.metric("Rows corrected automatically", int(fixed), "before any number was calculated", delta_color="off",
                  delta_arrow="off")

    if len(high):
        cols = st.columns(min(len(high), 5))
        for i, r in enumerate(high.itertuples()):
            with cols[i % len(cols)].container(border=True, key=f"card_dq{i}"):
                icon = "🛠️ Fixed" if r.action.startswith(("Removed", "Corrected")) else "🚩 Needs follow-up"
                st.html(f"""<div style="min-height:150px"><span class="tag {'data' if 'Fixed' in icon else 'structural'}">
                  {icon.split(' ', 1)[1].upper()}</span><div class="title" style="font-weight:700;margin-top:8px">
                  {esc(r.check)}</div><div class="muted" style="margin-top:6px">{r.rows_affected} rows ·
                  {m(r.usd_affected)}</div><div class="muted" style="margin-top:4px">{esc(r.action)}</div></div>""")

    with st.container(border=True, key="card_rollup"):
        st.html('<div class="h2" style="font-size:17px">Which forecast number can we trust?</div>')
        r2 = res.reconciliation.assign(
            kind=np.where(res.reconciliation["uses_forecast_category"], "Uses the reps' forecast category",
                          "Uses deal data"),
            txt=res.reconciliation["clean_pct_of_quota"].map("{:.1%}".format))
        base = alt.Chart(r2).encode(y=alt.Y("method:N", sort=None, title=None,
                                            axis=alt.Axis(labelLimit=360, domain=False, ticks=False)))
        ch = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
            x=alt.X("clean_pct_of_quota:Q", title="% of quota", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1.15])),
            color=alt.Color("kind:N", scale=alt.Scale(range=[BLUE, GREY]), legend=alt.Legend(title=None))) + \
            base.mark_text(align="left", dx=6, color=INK).encode(x="clean_pct_of_quota:Q", text="txt:N")
        rep = res.cfg.get("reported_forecast_pct")
        if rep:
            ch = ch + alt.Chart(pd.DataFrame({"x": [rep]})).mark_rule(strokeDash=[4, 4], color=INK).encode(x="x:Q")
        show(ch.properties(height=200))
        if rep:
            hits = res.reconciliation.loc[res.reconciliation["matches_reported"], "method"].tolist()
            st.caption(f"Reported forecast {rep:.1%} (dashed line). " + (
                f"Reproduced by: {', '.join(hits)}." if hits else "None of these reproduces it within 1 point."))
        else:
            st.caption("Enter a reported forecast % in the data settings to check which rollup reproduces it.")

    with st.container(border=True, key="card_dqall"):
        st.html('<div class="h2" style="font-size:17px">Every check, what it found and what was done</div>')
        table(dq[["severity", "check", "action", "rows_affected", "usd_affected", "detail"]].rename(columns={
            "severity": "Severity", "check": "Check", "action": "Action", "rows_affected": "Rows",
            "usd_affected": "Amount", "detail": "Why it matters"}), money=("Amount",))
        st.download_button("⬇️ Download the cleaned deals file (CSV)", csv_bytes(res.opp), "cleaned_deals.csv")


# ---------------------------------------------------------------------------------------
# Page: Actions
# ---------------------------------------------------------------------------------------
def page_actions():
    plan = fd.action_plan(F)
    cols = st.columns(3)
    for col, (h, title) in zip(cols, [("14", "Next 14 days: land the quarter"), ("30", "By day 30: fix the causes"),
                                      ("60", "By day 60: make it stick")]):
        with col.container(border=True, key=f"card_plan{h}"):
            items = "".join(f'<div class="act"><div class="dot"></div><div><div class="txt">{esc(a["action"])}</div>'
                            f'<div class="own">Owner: {esc(a["owner"])}</div></div></div>' for a in plan[h]) \
                or '<div class="muted">Nothing needed on this horizon.</div>'
            st.html(f'<div class="kicker">{title}</div><div style="margin-top:6px">{items}</div>')

    ai_on = ai_available()
    key = f"summary_{id(res)}"
    if key not in st.session_state:
        text = fd.template_summary(F)
        st.session_state[key] = (text, "Written automatically from the checked numbers", fd.verify_numbers(text, F))
    with st.container(border=True, key="card_summary"):
        h1, h2, h3, h4 = st.columns([2.2, 1.2, 1, 1.2], vertical_alignment="center")
        h1.html('<div class="h2" style="font-size:17px">Summary for the VP</div>')
        if h2.button("✨ Rewrite with Claude" if ai_on else "✨ Claude (needs API key)", disabled=not ai_on,
                     width="stretch"):
            with st.spinner("Claude is writing; every number is then checked against the data..."):
                st.session_state[key] = fd.draft_summary(F, res.cfg, use_ai=True)
        text, source, chk = st.session_state[key]
        h3.download_button("⬇️ Summary", text.encode("utf-8"), "forecast_summary.md", width="stretch")
        h4.download_button("⬇️ Excel workbook", Path(res.outputs["workbook"]).read_bytes(),
                           Path(res.outputs["workbook"]).name, width="stretch")
        lines = text.strip().splitlines()
        if lines and lines[0].startswith("#"):  # the card already has a title; show the report heading small
            st.html(f'<div class="kicker">{esc(lines[0].lstrip("# "))}</div>')
            lines = lines[1:]
        md("\n".join(lines))
        ok = int(chk["verified"].sum()) if len(chk) else 0
        st.caption(f"{source}  |  ✅ {ok}/{len(chk)} numbers checked against the data")

    with st.container(border=True, key="card_lists"):
        st.html('<div class="h2" style="font-size:17px">Deal lists for managers</div>')
        manager = st.selectbox("Show lists for", ["All managers"] + ORDERED_TEAMS)
        nice = {"opp_id": "Deal", "account_name": "Account", "rep_name": "Owner", "manager_name": "Manager",
                "pipeline_stage": "Stage", "close_date": "Close date", "days_past_due": "Days past due",
                "amount_usd": "Amount", "win_probability": "Win prob.", "weighted_open_usd": "Expected value",
                "issue": "Issue"}
        cols = list(nice)[:-1]
        lists = [("🚨 Deals with no owner: reassign now", res.actions["orphaned_deals"], cols, "deals_with_no_owner.csv"),
                 ("⏰ Past their close date: confirm a date or move out", res.actions["past_due"], cols,
                  "past_due_deals.csv"),
                 ("🏷️ Forecast label doesn't match the deal", res.actions["category_fixes"], ["issue"] + cols,
                  "forecast_category_fixes.csv"),
                 ("🐢 Early-stage deals stuck for 90+ days", res.actions["stalled_early_stage"], cols, "stalled_deals.csv")]
        for i, (title, df, c, fname) in enumerate(lists):
            d = df if manager == "All managers" else df[df["manager_name"] == manager]
            d = d[c].rename(columns=nice)
            d["Close date"] = pd.to_datetime(d["Close date"]).dt.date
            with st.expander(f"{title}  ({len(d)})", expanded=i == 0 and len(d) > 0):
                table(d, money=("Amount", "Expected value"), dec=("Win prob.",), height=min(38 * len(d) + 40, 400))
                st.download_button("Download this list (CSV)", csv_bytes(d), fname, key=f"dl_{i}")


# ---------------------------------------------------------------------------------------
# Page: Ask AI (chat with the forecast; same engine as agent_app.py)
# ---------------------------------------------------------------------------------------
MORE_QUESTIONS = [
    "Which deals should the India team chase this week?",
    "Why is Japan Enterprise behind, and what did they lose?",
    "What if the biggest open deal in the region slips out of the quarter?",
    "Which reps need coaching, and why?",
    "Why did we lose deals this quarter?",
    "Which Negotiation deals are past their close date?",
    "How much of the gap comes from the empty seats?",
    "Why is almost all of the Korea pipeline past due?",
    "What data problems did you find, and what did you fix?",
    "What should the VP do in the next 14 days?",
]
CHAT_MONEY = ("amount_usd", "weighted_open_usd", "total_amount_usd", "quota_usd", "forecast_usd", "gap_usd",
              "lost_usd", "open_usd", "weighted_usd", "vacant_quota_usd")
CHAT_PCT = ("attainment", "share_of_gap", "past_due_share_of_weighted", "win_rate_usd", "share_in_early_stage",
            "share_past_due")


def render_msg(msg):
    with st.chat_message(msg["role"]):
        md(msg["text"])
        if msg.get("df") is not None:
            table(msg["df"], money=CHAT_MONEY, pct=CHAT_PCT, dec=("win_probability", "avg_win_prob"))
        if msg.get("steps"):
            with st.expander(f"How I got this: {len(msg['steps'])} look-up(s) on the checked data"):
                for s in msg["steps"]:
                    st.markdown(f"**`{s['tool']}`** `{json.dumps(s['input'])}`")
                    st.json(s["output"], expanded=False)
        chk = msg.get("check")
        if chk is not None and len(chk):
            ok = int(chk["verified"].sum())
            if ok == len(chk):
                st.caption(f"✅ {ok}/{len(chk)} numbers traced to the data")
            else:
                bad = ", ".join(chk.loc[~chk["verified"], "number_in_text"])
                st.warning(f"⚠️ {len(chk) - ok} number(s) not traced to the data: {bad}. Check before sharing.")
        if msg.get("note"):
            st.caption(msg["note"])


def ask(question: str, ai_on: bool):
    chat = st.session_state["ask_chat"]
    chat.append({"role": "user", "text": question})
    render_msg(chat[-1])
    if ai_on:
        with st.chat_message("assistant"):
            status = st.status("Working on it...", expanded=True)
            try:
                if "ask_agent" not in st.session_state:
                    st.session_state["ask_agent"] = ag.ForecastAgent(res)
                out = st.session_state["ask_agent"].ask(
                    question, on_step=lambda s: status.write(f"🔧 `{s['tool']}` {json.dumps(s['input'])}"))
                status.update(label=f"Done: {len(out['steps'])} look-up(s)", state="complete", expanded=False)
                chat.append({"role": "assistant", "text": out["text"], "steps": out["steps"],
                             "check": out["number_check"]})
            except Exception as e:  # no internet, bad key, rate limit: fall back rather than break the demo
                status.update(label="Claude unavailable", state="error", expanded=False)
                st.session_state.pop("ask_agent", None)
                if question in ag.QUICK_QUESTIONS:
                    text, df = ag.offline_answer(tools, question)
                    chat.append({"role": "assistant", "text": text, "df": df,
                                 "note": f"Answered without AI: the Claude call failed ({type(e).__name__})."})
                else:
                    chat.append({"role": "assistant", "text": f"The Claude call failed ({type(e).__name__}: {e}). "
                                                              "The quick answers still work without AI."})
    elif question in ag.QUICK_QUESTIONS:
        text, df = ag.offline_answer(tools, question)
        chat.append({"role": "assistant", "text": text, "df": df,
                     "note": "Answered without AI: computed directly from the checked data."})
    else:
        chat.append({"role": "assistant", "text": "This question needs Claude. Add an API key to "
                                                  "`Part_B_Toolkit/.env` and restart the app."})
    st.rerun()


def page_ask():
    ai_on = ai_available()
    if st.session_state.get("ask_data") != id(res):  # new files: start a new conversation
        st.session_state.update(ask_data=id(res), ask_chat=[])
        st.session_state.pop("ask_agent", None)
    with st.container(border=True, key="card_ask"):
        c1, c2 = st.columns([3.2, 1], vertical_alignment="center")
        c1.html('<div class="h2" style="font-size:20px">Ask the forecast</div><div class="muted">Ask in plain English. '
                'Claude decides what to look up, the code does the maths on the checked data, and every number in '
                'the answer is traced back to it.</div>')
        if c2.button("🗑️ New conversation", width="stretch"):
            st.session_state["ask_chat"] = []
            st.session_state.pop("ask_agent", None)
        st.caption(f"🟢 Claude connected ({res.cfg['agent']['model']})" if ai_on else
                   "⚪ Offline: no API key found. The quick answers work; the other questions and free chat need a key "
                   "in Part_B_Toolkit/.env.")
        st.html('<div class="kicker" style="margin-top:6px">Quick answers (work even without AI)</div>')
        cols = st.columns(3)
        for i, q in enumerate(ag.QUICK_QUESTIONS):
            if cols[i % 3].button(q, key=f"ask_q{i}", width="stretch"):
                st.session_state["ask_pending"] = q
        st.html('<div class="kicker" style="margin-top:6px">More questions to ask Claude</div>')
        cols = st.columns(2)
        for i, q in enumerate(MORE_QUESTIONS):
            if cols[i % 2].button(q, key=f"ask_m{i}", width="stretch", disabled=not ai_on):
                st.session_state["ask_pending"] = q

    history = st.container()  # created before the input box so the conversation renders above it
    typed = st.chat_input("Ask about the forecast, a team, a rep or a deal..." if ai_on
                          else "Add an API key to chat freely (the quick answers work without one)", disabled=not ai_on)
    question = typed or st.session_state.pop("ask_pending", None)
    with history:
        for msg in st.session_state["ask_chat"]:
            render_msg(msg)
        if question:
            ask(question, ai_on)


{"Overview": page_overview, "Teams": page_teams, "Pipeline": page_pipeline, "Data trust": page_data,
 "Actions": page_actions, "Ask AI": page_ask}[page]()

st.caption("Forecast = deals already won + each open deal x its win probability, after the data checks. "
           "Same engine as the Part B notebook (forecast_diagnostic.py).")
