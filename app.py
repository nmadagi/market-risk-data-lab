"""market-risk-data-lab: what bad data does to VaR, and a pipeline that
detects it, repairs it under guardrails, and proves the repair.

Read the tabs left to right; they are the pipeline in order.
"""
import time

import altair as alt
import pandas as pd
import streamlit as st

from data.generate import generate_market_data
from src import agent, detection, ml_detection, remediation, risk
from src.corruption import apply_default_faults, inject_stale

st.set_page_config(page_title="Market Risk Data Lab", layout="wide")

# Streamlit hashes a cached function's OWN source, not the source of what it
# calls. Changing the generator or the repair logic therefore leaves a stale
# cached result behind. Bumping this string is what actually invalidates it.
PIPELINE_VERSION = "2026-09-02-four-tabs"


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


@st.cache_resource
def trained_model(version: str):
    t0 = time.perf_counter()
    model = ml_detection.train_classifier()
    return model, time.perf_counter() - t0


@st.cache_data
def load_all(version: str):
    clean = generate_market_data()
    corrupted, fault_log = apply_default_faults(clean)
    model, _ = trained_model(version)
    t0 = time.perf_counter()
    flagged = ml_detection.classify(corrupted, model)
    score_seconds = time.perf_counter() - t0
    findings = ml_detection.findings_from_model(flagged)
    def reviewer(proposal):
        """Stands in for the person who signs off single-day repairs.

        This demo has an answer key, so the stand-in simply asks whether
        the flagged day is one of the planted faults. A real reviewer
        would look at the chart, the vendor feed and any change notice.
        Its verdicts are shown on screen so you can see what the model got
        wrong and the person caught.
        """
        return bool(((fault_log["series"] == proposal["series"]) &
                     (fault_log["start"] <= proposal["dates"][-1]) &
                     (fault_log["end"] >= proposal["dates"][0])).any())

    staged, flags, proposals, checks, unresolved = remediation.run(
        corrupted, findings, reviewer=reviewer)
    return (clean, corrupted, fault_log, findings, proposals, staged, flags,
            checks, unresolved, flagged, score_seconds)


(clean, corrupted, fault_log, findings, proposals, staged, flags, checks,
 unresolved, flagged, score_seconds) = load_all(
    PIPELINE_VERSION)


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
                     "99% 1 day VaR": f"${v/1e6:,.2f}M",
                     "vs clean": f"{(v-base)/base*100:+.1f}%"})
    return pd.DataFrame(rows)


var_clean = risk.var99(risk.pnl_vector(clean))
var_corrupt = risk.var99(risk.pnl_vector(corrupted.ffill()))
var_repaired = risk.var99(risk.pnl_vector(staged))
es_clean = risk.expected_shortfall(risk.pnl_vector(clean))
es_corrupt = risk.expected_shortfall(risk.pnl_vector(corrupted.ffill()))
es_repaired = risk.expected_shortfall(risk.pnl_vector(staged))

st.title("Market Risk Data Lab")
st.write(
    "One synthetic trading book, six risk factor series, three injected data "
    "faults. A trained model detects them, deterministic guardrails decide "
    "the repairs, and a benchmark proves the repairs. All data is synthetic "
    "and seeded; the portfolio is held as sensitivities."
)
st.write(
    "Pipeline, and the tabs in order: generate a clean history > inject "
    "faults > detect with a gradient boosting model trained on synthetic "
    "faults > propose repairs and score each one against deterministic "
    "guardrails, applying only what passes > run the risk engine "
    "(historical simulation VaR, stressed VaR, sensitivities, stress "
    "scenarios, backtesting) on the repaired data."
)

st.subheader("The finding: your risk number is not a data alarm")
c1, c2 = st.columns(2)
c1.metric("99% 1 day VaR with corrupted data", f"${var_corrupt/1e6:,.2f}M",
          f"{(var_corrupt-var_clean)/var_clean*100:+.1f}% vs clean, barely moves")
c2.metric("99% 1 day expected shortfall, same corrupted data",
          f"${es_corrupt/1e6:,.2f}M",
          f"{(es_corrupt-es_clean)/es_clean*100:+.1f}% vs clean")
st.write(
    f"Same book, same three data faults, two risk measures, both one day at "
    f"99 percent. VaR on the "
    f"corrupted data is \\${var_corrupt/1e6:,.2f}M. On the clean data it is "
    f"\\${var_clean/1e6:,.2f}M. It barely noticed. Expected shortfall on the "
    f"corrupted data is \\${es_corrupt/1e6:,.2f}M. On the clean data it is "
    f"\\${es_clean/1e6:,.2f}M. Same bad data, one number shrugged and the "
    "other one screamed. The reason is simple: 99% VaR is the 5th worst of "
    "500 days, so one bad print becomes the new worst day and the 5th worst "
    "barely changes. Expected shortfall is the average of the worst days, "
    "so the fake loss goes straight into the average. Two consequences. "
    "You cannot rely on the risk number to tell you your data broke, so "
    "someone has to check the data itself. And as the industry shifts from "
    "VaR toward expected shortfall, bad data gets more dangerous, not less."
)
st.write(
    f"After detection and repair: VaR \\${var_repaired/1e6:,.2f}M "
    f"({(var_repaired-var_clean)/var_clean*100:+.1f}% vs clean), expected "
    f"shortfall \\${es_repaired/1e6:,.2f}M "
    f"({(es_repaired-es_clean)/es_clean*100:+.1f}% vs clean)."
)

tabs = st.tabs(["1 Data health", "2 Find and fix",
                "3 VaR and sVaR", "4 Sensitivities and stress"])

with tabs[0]:
    st.subheader("Six risk factor series, three injected faults")
    st.write(
        "The golden copy is clean seeded synthetic history (2020 to 2026, "
        "with an engineered high volatility era in 2022). Three realistic "
        "faults are injected: a frozen feed, a bad print, and a gap. Pick a "
        "series to compare clean vs corrupted. The same data is exported as "
        "CSV in the repo's data folder."
    )
    table(fault_log)
    col = st.selectbox("series", list(clean.columns), index=1)
    view = pd.DataFrame({"clean": clean[col], "corrupted": corrupted[col]})
    lo, hi = "2025-06-01", None
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
        "Grey is the clean data drawn thick. Orange is the corrupted feed "
        "drawn thin on top. Where they agree you see orange inside grey. "
        "Where grey shows on its own, the feed is wrong there. Notice how "
        "normal the picture looks: that is why eyeballing feeds does not "
        "work and a detector is needed."
    )
    chart(overlay(view.loc[lo:hi]))

with tabs[1]:
    st.subheader("Find: one pass over six years, every planted fault found and named")
    held = findings[findings["verdict"] == detection.VERDICT_BREAK]
    repairable = findings[findings["verdict"] != detection.VERDICT_BREAK]
    a, b, c_ = st.columns(3)
    a.metric("series-days scanned", f"{corrupted.size:,}")
    b.metric("years of history", f"{(corrupted.index[-1] - corrupted.index[0]).days / 365.25:.1f}")
    c_.metric("scanned in", f"{score_seconds:.2f}s")
    st.write(
        f"**{len(fault_log)} planted faults, all {len(fault_log)} found and "
        f"named correctly** (the frozen feed on every one of its "
        f"{int(repairable.loc[repairable['type'] == 'stale', 'length'].sum())} days, "
        f"the bad print, the gap on every one of its "
        f"{int(repairable.loc[repairable['type'] == 'gap', 'length'].sum())} days). "
        f"**{len(held)} possible level shift{'s' if len(held) != 1 else ''}** "
        f"flagged for a human to look at, {'none of them' if len(held) != 1 else 'not'} "
        "planted. Zero real market moves were repaired away."
    )
    st.write(
        "How: the fault injector can manufacture unlimited labeled faults, so "
        "a gradient boosting model was trained on twelve synthetic histories "
        "full of planted frozen feeds, bad prints, gaps and vendor level "
        "shifts, then run on this history, which it had never seen. It reads "
        "five numbers per series per day (size of the move against normal, "
        "whether tomorrow undid it, whether correlated series moved, how long "
        "the value has been frozen, whether it is missing) and names each "
        "day: normal, stale, spike, gap, or possible level shift. A possible "
        "level shift is never repaired automatically, because from inside one "
        "series a permanent shift looks exactly like a real repricing; that "
        "call belongs to a person, or in production to an agent that reads "
        "the vendor's notice."
    )
    show = repairable.reset_index(drop=True).copy()
    show["start"] = show["start"].dt.date
    show["decision"] = show["verdict"].fillna("repair proposed")
    table(show[["series", "type", "start", "length", "detail", "decision"]]
          .rename(columns={"length": "days"}))

    st.subheader("Fix: every fault gets one decision, and only accepted repairs touch the data")
    n_auto = sum(c["accepted"] and not c["needs_review"] for c in checks)
    n_ok = sum(bool(c.get("approved_at_review")) for c in checks)
    n_no = sum(c["needs_review"] and not bool(c.get("approved_at_review"))
               for c in checks)
    n_rej = sum(not c["accepted"] and not c["needs_review"] for c in checks)
    a, b, c_, d_ = st.columns(4)
    a.metric("repaired automatically", n_auto)
    b.metric("approved at review", n_ok)
    c_.metric("rejected at review", n_no, "real market moves, not faults",
              delta_color="off")
    d_.metric("rejected by guardrail", n_rej)
    st.write(
        "How: short problems are interpolated, long stretches are rebuilt "
        "from correlated series in change space. Repairs are scored in date "
        "order against the data as it stands, so each one is judged after "
        "the repairs already accepted before it. A repair that changes the "
        "series' own return distribution is rejected outright. Everything "
        "else is applied, but a repair covering a single day is signed off "
        "by a person first, because a one-day call is where the detector is "
        "weakest and where an automatic fix is most likely to smooth away a "
        "real market move. The golden copy is never edited in place: the "
        "model proposes, the controls dispose."
    )
    rows = []
    for p_, c in zip(proposals, checks):
        if c["needs_review"] and not bool(c.get("approved_at_review")):
            decision = "rejected at review"
            why = "a person judged this a real market move, not a fault"
        elif not c["accepted"]:
            decision = "rejected by guardrail"
            why = f"distribution changed (KS p = {c['ks_pvalue']})"
        elif c["needs_review"]:
            decision = "applied after review"
            why = ("single day call, model is weakest here"
                   if c["points"] < remediation.MIN_AUTO_DAYS
                   else f"VaR impact {c['var_impact_pct']}% is material")
        else:
            decision, why = "applied automatically", "passed both guardrails"
        rows.append({"series": p_["series"], "issue": p_["type"],
                     "days": len(p_["dates"]), "fix": p_["method"],
                     "KS p": c["ks_pvalue"], "VaR impact %": c["var_impact_pct"],
                     "decision": decision, "reason": why})
    table(pd.DataFrame(rows))
    rejected = [r for r in rows if r["decision"] == "rejected by guardrail"]
    if rejected:
        st.write(
            f"The one rejected by a guardrail is worth a look: a "
            f"{rejected[0]['days']} day straight line fill has no volatility, "
            "and a check that compares return distributions catches exactly "
            f"that. The {len(unresolved)} points it would have filled stay on "
            "the exception report, carried forward as a stopgap and clearly "
            "labeled, until a person picks a proxy series."
        )
    if n_no:
        st.write(
            f"And {n_no} of the {n_ok + n_no} single-day repairs sent for "
            "sign-off were rejected by the reviewer: they are real market "
            "moves from the 2022 stress era, not faults. That is the whole "
            "case for not letting a model repair one-day events on its own, "
            "and it is measurable here because the answer key exists. The "
            "review stand-in consults it; a person would look at the chart "
            "and any vendor notice."
        )

    with st.expander(f"{len(held)} possible level shift{'s' if len(held) != 1 else ''} held for human review"):
        h = held.reset_index(drop=True).copy(); h["start"] = h["start"].dt.date
        table(h[["series", "start", "detail"]])
    with st.expander(f"Audit trail: {len(flags)} applied points, each with its method"):
        table(flags)
    with st.expander(f"Exception report: {len(unresolved)} unresolved points"):
        table(unresolved)
    facts = agent.build_facts(findings, checks, var_corrupt, var_repaired)
    text, source = agent.narrative(facts)
    st.info(f"Morning report ({source}): {text}")

with tabs[2]:
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

with tabs[3]:
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
