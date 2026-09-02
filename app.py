"""market-risk-data-lab: what bad data does to VaR, and a pipeline that
detects it, repairs it under guardrails, and proves the repair.

Read the tabs left to right; they are the pipeline in order.
"""
import altair as alt
import pandas as pd
import streamlit as st

from data.generate import generate_market_data
from src import agent, detection, evaluation, remediation, risk
from src.corruption import apply_default_faults, inject_stale

st.set_page_config(page_title="market-risk-data-lab", layout="wide")

# Streamlit hashes a cached function's OWN source, not the source of what it
# calls. Changing the generator or the repair logic therefore leaves a stale
# cached result behind. Bumping this string is what actually invalidates it.
PIPELINE_VERSION = "2026-09-01-unresolved-and-ml-benchmark"


def table(df):
    """st.dataframe at full width across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.dataframe(df, use_container_width=True)


def chart(c):
    """st.altair_chart at full width across Streamlit versions."""
    try:
        st.altair_chart(c, width="stretch")
    except Exception:
        st.altair_chart(c, use_container_width=True)


CLEAN_COLOR, CORRUPT_COLOR = "#9aa5b1", "#d9480f"


def overlay(view: pd.DataFrame, height: int = 300):
    """Clean as a thick grey band, corrupted as a thin orange line on top.

    Two lines that are identical almost everywhere hide each other, so the
    clean one is drawn wide and the corrupted one narrow: where they agree
    you see orange inside grey, where they disagree the grey shows on its
    own. The y axis does not start at zero, so small faults stay visible.
    """
    long = (view.rename_axis("date").reset_index()
            .melt(id_vars="date", var_name="source", value_name="value"))
    color = alt.Color("source:N", title=None,
                      scale=alt.Scale(domain=["clean", "corrupted"],
                                      range=[CLEAN_COLOR, CORRUPT_COLOR]))
    base = alt.Chart(long).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)),
        color=color,
        tooltip=[alt.Tooltip("date:T"), "source:N",
                 alt.Tooltip("value:Q", format=".4f")])
    clean_layer = base.transform_filter(alt.datum.source == "clean") \
        .mark_line(strokeWidth=5, opacity=0.9)
    corrupt_layer = base.transform_filter(alt.datum.source == "corrupted") \
        .mark_line(strokeWidth=1.6)
    return alt.layer(clean_layer, corrupt_layer).properties(height=height)


@st.cache_data
def load_all(version: str):
    clean = generate_market_data()
    corrupted, fault_log = apply_default_faults(clean)
    findings = detection.run_all(corrupted)
    staged, flags, proposals, checks, unresolved = remediation.run(
        corrupted, findings)
    return (clean, corrupted, fault_log, findings, proposals, staged, flags,
            checks, unresolved)


(clean, corrupted, fault_log, findings, proposals, staged, flags, checks,
 unresolved) = load_all(PIPELINE_VERSION)


@st.cache_data
def staleness_sweep(version: str):
    """How long must every feed stall before 99% VaR actually moves?"""
    base = risk.var99(risk.pnl_vector(clean))
    rows = []
    for days in (20, 60, 100, 200, 300):
        c = clean.copy()
        for col in clean.columns:
            c, _ = inject_stale(c, col, "2025-06-02", days)
        v = risk.var99(risk.pnl_vector(c.ffill()))
        rows.append({"days stalled": days,
                     "share of 500 day window": f"{days/5:.0f}%",
                     "99% VaR": f"${v/1e6:,.2f}M",
                     "vs clean": f"{(v-base)/base*100:+.1f}%"})
    return pd.DataFrame(rows)


var_clean = risk.var99(risk.pnl_vector(clean))
var_corrupt = risk.var99(risk.pnl_vector(corrupted.ffill()))
var_repaired = risk.var99(risk.pnl_vector(staged))
es_clean = risk.expected_shortfall(risk.pnl_vector(clean))
es_corrupt = risk.expected_shortfall(risk.pnl_vector(corrupted.ffill()))
es_repaired = risk.expected_shortfall(risk.pnl_vector(staged))

st.title("market-risk-data-lab")
st.caption(
    "One synthetic trading book, six risk factor series, four injected data "
    "faults. The pipeline detects them, repairs them under deterministic "
    "guardrails, and proves the repairs statistically. Tabs are the pipeline "
    "in order."
)

st.subheader("The finding: your risk number is not a data alarm")
c1, c2 = st.columns(2)
c1.metric("99% VaR with corrupted data", f"${var_corrupt/1e6:,.2f}M",
          f"{(var_corrupt-var_clean)/var_clean*100:+.1f}% vs clean, barely moves")
c2.metric("Expected shortfall, same corrupted data",
          f"${es_corrupt/1e6:,.2f}M",
          f"{(es_corrupt-es_clean)/es_clean*100:+.1f}% vs clean")
st.write(
    f"Same book, same four data faults, two risk measures. VaR reads "
    f"\\${var_corrupt/1e6:,.2f}M against a true \\${var_clean/1e6:,.2f}M and "
    f"looks perfectly normal. Expected shortfall reads "
    f"\\${es_corrupt/1e6:,.2f}M against a true \\${es_clean/1e6:,.2f}M. "
    "The reason is structural: 99% VaR "
    "is the 5th worst of 500 days, so one corrupt print moves the ranking by "
    "a single place and the number shrugs. Expected shortfall averages the "
    "worst days, so it absorbs the whole fake loss. Two consequences. You "
    "cannot use the headline risk number to tell you your data broke, which "
    "is the argument for dedicated monitoring. And as the industry shifts "
    "from VaR toward expected shortfall, data quality gets more load bearing, "
    "not less."
)
st.write(
    f"After detection and repair: VaR \\${var_repaired/1e6:,.2f}M "
    f"({(var_repaired-var_clean)/var_clean*100:+.1f}% vs clean), expected "
    f"shortfall \\${es_repaired/1e6:,.2f}M "
    f"({(es_repaired-es_clean)/es_clean*100:+.1f}% vs clean)."
)

tabs = st.tabs(["1 Data health", "2 Detection", "3 Remediation",
                "4 VaR and sVaR", "5 Sensitivities and stress",
                "6 Evaluation", "About"])

with tabs[0]:
    st.subheader("Six risk factor series, four injected faults")
    st.write(
        "The golden copy is clean seeded synthetic history (2020 to 2026, "
        "with an engineered high volatility era in 2022). Four realistic "
        "faults are injected: a stale feed, a spike, a gap, and a vendor "
        "splice. Pick a series to compare clean vs corrupted. The same data "
        "is exported as CSV in the repo's data folder."
    )
    table(fault_log)
    col = st.selectbox("series", list(clean.columns), index=1)
    view = pd.DataFrame({"clean": clean[col], "corrupted": corrupted[col]})
    # focus the chart on the stretch where this series was damaged
    focus = {"swaption_vol": ("2022-06-01", "2023-06-30")}
    lo, hi = focus.get(col, ("2025-06-01", None))
    faults_here = fault_log[fault_log["series"] == col]
    if len(faults_here):
        described = "; ".join(
            f"{r.fault} ({r.start.date()}" + (f" to {r.end.date()})" if r.end != r.start else ")")
            for r in faults_here.itertuples())
        st.write(f"Faults injected on this series: {described}.")
    else:
        st.write("No faults were injected on this series, so the two lines "
                 "sit exactly on top of each other.")
    st.write(
        "Grey is the clean truth drawn thick. Orange is the corrupted feed "
        "drawn thin on top. Where they agree you see orange inside grey. "
        "Where grey shows on its own, the feed is wrong there. Notice how "
        "normal the picture looks with eight percent of this history's "
        "cells altered: that is why eyeballing feeds does not work and a "
        "detector is needed."
    )
    chart(overlay(view.loc[lo:hi]))

with tabs[1]:
    st.subheader("Finding the breaks with statistics, no model needed yet")
    n_stale = int((findings["type"] == "stale").sum())
    n_gap = int((findings["type"] == "gap").sum())
    spikes = findings[findings["type"] == "spike"]
    n_err = int((spikes["verdict"] == detection.VERDICT_ERROR).sum())
    n_real = int((spikes["verdict"] == detection.VERDICT_REAL).sum())
    n_held = int((spikes["verdict"] == detection.VERDICT_BREAK).sum())
    st.write(
        f"**{len(findings)} findings:** {n_stale} stale feed, {n_gap} gap, "
        f"{len(spikes)} big moves. Of the big moves, {n_err} look like data "
        f"errors, {n_real} are confirmed real by correlated series, and "
        f"{n_held} are level breaks held for a human."
    )
    st.write(
        "Three detectors: a run of identical prints (stale), a move far "
        "outside the series' own recent volatility (spike), missing days "
        "(gap). A big move alone is not proof of an error, so each spike "
        "gets two tiebreakers. Did correlated series move too? Then it is "
        "real. Was it undone the next day? Then it was a bad print. Neither? "
        "Then it is a level break, and it is held rather than repaired, "
        "because interpolating a real repricing destroys real history."
    )
    show = findings.copy()
    show["start"] = show["start"].dt.date
    show["verdict"] = show["verdict"].fillna("data fault, repair proposed")
    table(show[["series", "type", "start", "length", "detail", "verdict"]]
          .rename(columns={"length": "days", "verdict": "decision"}))

with tabs[2]:
    st.subheader("Every finding gets one decision, and only accepted repairs "
                 "touch the data")
    n_acc = sum(c["accepted"] for c in checks)
    n_rev = sum(c["needs_review"] for c in checks)
    n_rej = len(checks) - n_acc - n_rev
    held = findings[findings["verdict"] == detection.VERDICT_BREAK]
    a, b, c_, d_ = st.columns(4)
    a.metric("repairs applied", n_acc)
    b.metric("sent to human review", n_rev)
    c_.metric("rejected by guardrail", n_rej)
    d_.metric("level breaks held", len(held))
    st.write(
        "Fixes follow a trust ladder: interpolate short problems, rebuild "
        "long stretches from correlated series in change space. Each "
        "proposal is scored alone, using only data the pipeline would "
        "really have, against two deterministic guardrails: does the "
        "repaired stretch keep the series' own return distribution (KS "
        "test), and does the repair move VaR materially (routes to a "
        "human). The model proposes, the controls dispose, and the golden "
        "copy is never edited in place."
    )
    rows = []
    for p_, c in zip(proposals, checks):
        if c["accepted"]:
            decision, why = "applied", "passed both guardrails"
        elif c["needs_review"]:
            decision, why = "human review", f"VaR impact {c['var_impact_pct']}% is material"
        else:
            decision, why = "rejected", f"distribution changed (KS p = {c['ks_pvalue']})"
        rows.append({"series": p_["series"], "issue": p_["type"],
                     "days": len(p_["dates"]), "fix": p_["method"],
                     "KS p": c["ks_pvalue"], "VaR impact %": c["var_impact_pct"],
                     "decision": decision, "reason": why})
    for r in held.itertuples():
        rows.append({"series": r.series, "issue": "level break", "days": 1,
                     "fix": "none", "KS p": None, "VaR impact %": None,
                     "decision": "held for review",
                     "reason": "big move, no peer confirmation, no reversal"})
    st.write("Decision log:")
    table(pd.DataFrame(rows))
    st.write(
        f"The rejected one is worth a look: a {rows[[r['decision'] for r in rows].index('rejected')]['days']} "
        "day straight line fill has no volatility, and a guardrail that "
        "compares return distributions catches exactly that. The "
        f"{len(unresolved)} points it would have filled stay on the "
        "exception report, carried forward as a stopgap and clearly "
        "labeled, until a person picks a proxy. The pipeline never fills "
        "silently."
        if n_rej else
        f"{len(unresolved)} points have no accepted repair; they are carried "
        "forward as a stopgap and listed on the exception report."
    )
    with st.expander(f"Audit trail: {len(flags)} applied points, each with its method"):
        table(flags)
    with st.expander(f"Exception report: {len(unresolved)} unresolved points"):
        table(unresolved)
    facts = agent.build_facts(findings, checks, var_corrupt, var_repaired)
    text, source = agent.narrative(facts)
    st.info(f"Morning report ({source}): {text}")

with tabs[3]:
    st.subheader("VaR and stressed VaR on the repaired data")
    pnl = risk.pnl_vector(staged)
    es = risk.expected_shortfall(pnl)
    svar, ws, we = risk.svar99(staged)
    a, b, d = st.columns(3)
    a.metric("99% 1 day VaR", f"${var_repaired/1e6:,.2f}M",
             "5th worst of the last 500 replayed days")
    b.metric("Expected shortfall", f"${es/1e6:,.2f}M",
             "average of the days worse than VaR")
    d.metric("Stressed VaR", f"${svar/1e6:,.2f}M",
             f"worst 12 months: {ws.date()} to {we.date()}")
    st.write(
        "Historical simulation: replay each of the last 500 days of market "
        "moves against today's book, sort the 500 results, read the 5th "
        "worst. Stressed VaR runs the same engine over every 12 month window "
        "in history and keeps the worst; it finds the engineered 2022 stress "
        "era by itself. That search needs clean history all the way back, "
        "which is why backfilling is a capital problem, not housekeeping."
    )
    st.write("**How broken must the data be before VaR notices?** Every "
             "series frozen for a growing stretch:")
    table(staleness_sweep(PIPELINE_VERSION))
    st.write(
        "VaR does not move until the outage covers most of the window, "
        "because until then the five worst days are still in there. A "
        "stalled feed is close to invisible in this number."
    )
    with st.expander("Simulated P&L distribution, last 500 days"):
        st.bar_chart(pnl.round(-4).value_counts().sort_index(), height=220)
    bt = risk.backtest(staged)
    n_exc = int(bt["exceedance"].sum())
    with st.expander(f"Backtest, last 250 days: {n_exc} days worse than VaR "
                     "(about 2.5 expected)"):
        st.write("Too many misses means the model understates risk; misses "
                 "bunched together are worse than the count alone.")
        st.line_chart(bt[["pnl", "var"]], height=240)

with tabs[4]:
    st.subheader("Sensitivities: what the book cares about, factor by factor")
    st.write(
        "The book is held as sensitivities: bump one market number by one "
        "unit, and this is the dollar response. Traders hedge off these, "
        "risk sets limits on them, and when VaR moves overnight this is the "
        "layer that says whether positions changed, markets changed, or a "
        "series' history changed."
    )
    labels = {"DV01 2y": ("2 year rate up 0.01%", -1),
              "DV01 5y": ("5 year rate up 0.01%", -1),
              "DV01 10y": ("10 year rate up 0.01%", -1),
              "Vega": ("volatility up 1 point", 1),
              "FX delta": ("euro up 1%", 0.01),
              "Spread DV01": ("credit spreads wider 0.01%", -1)}
    sens = risk.sensitivities_table()
    def plain(row):
        what, mult = labels[row["sensitivity"]]
        pnl = row["value"] * mult
        verb = "gains" if pnl > 0 else "loses"
        return f"{what}: book {verb} ${abs(pnl):,.0f}"
    sens["in plain English"] = sens.apply(plain, axis=1)
    table(sens)
    st.subheader("Stress scenarios: designed shocks, several factors at once")
    st.write(
        "No probabilities: each scenario says 'if this happens, here is the "
        "damage'. The craft is coherence, moving the factors together the "
        "way a real event would. Not every scenario is a loss; the point of "
        "several is finding which direction hurts."
    )
    table(risk.stress_pnl().rename(columns={"pnl_musd": "P&L ($M)"}))

with tabs[5]:
    st.subheader("Which fill method deserves trust, and the metric that decides")
    st.write(
        "Mask and recover: hide points that are actually known, rebuild each "
        "outage the way the pipeline would, score against the truth. MAE is "
        "average accuracy. tail_ratio is repaired volatility over true "
        "volatility, and it is the score that matters for risk: below 1 "
        "means the method smooths, and smoothed history understates VaR. "
        "A random forest gets the same inputs as the linear proxy, so the "
        "test isolates the model, not the features."
    )
    col = st.selectbox("series to evaluate", list(clean.columns), index=1,
                       key="evalcol")
    ev = evaluation.mask_and_recover(clean, col)
    table(ev)
    by_mae = ev["mae"].idxmin()
    by_tail = ev["tail_ratio"].idxmax()
    if by_mae == by_tail:
        st.write(f"**On this series {by_mae} wins on both average error and "
                 "tail preservation.**")
    else:
        st.write(f"**Rank by average error and you would ship {by_mae}. Rank "
                 f"by tail preservation and you ship {by_tail}.** The metric "
                 "you choose decides the model you deploy, and for risk data "
                 "average accuracy is the wrong metric.")
    st.write(
        "No method reaches a tail ratio of 1: every fill flattens volatility "
        "somewhat, which is why filled points stay flagged. Carry forward "
        "scores exactly zero, and it is what the pipeline falls back to for "
        "points it refuses to repair. A series with no correlated peers has "
        "no proxy or forest row at all: a factor with no proxy is an "
        "escalation, not a computation."
    )

with tabs[6]:
    st.subheader("What this is")
    st.write(
        "A compact demonstration of the data layer under trading book risk "
        "models. All data is synthetic and seeded; the portfolio is a toy "
        "book held as sensitivities. The design principle throughout: "
        "automated proposals, deterministic guardrails, flagged and "
        "reversible changes, and statistical evaluation before trust. "
        "Nothing here claims production scale; the architecture is the "
        "point."
    )
    st.write(
        "Pipeline: generate golden copy > inject faults > detect "
        "(run length, EWMA z-score, calendar, peer and reversal tiebreakers) "
        "> propose fixes (interpolation or change-space peer regression, "
        "with a random forest benchmarked against both) > "
        "per-proposal guardrail checks (KS distribution test, VaR impact "
        "routing) > apply accepted only, with flags > risk engine (hist sim "
        "VaR, sVaR window search, sensitivities, coherent stress scenarios, "
        "backtesting) > mask and recover evaluation harness."
    )
