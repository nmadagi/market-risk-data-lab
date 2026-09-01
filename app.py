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


def difference(clean_s: pd.Series, corrupt_s: pd.Series, height: int = 160):
    """corrupted minus clean over the whole history. Zero means identical,
    a break means the value is missing, anything else is a fault."""
    d = (corrupt_s - clean_s).rename("corrupted minus clean") \
        .rename_axis("date").reset_index()
    return alt.Chart(d).mark_line(color=CORRUPT_COLOR, strokeWidth=1.5).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("corrupted minus clean:Q", title=None),
        tooltip=[alt.Tooltip("date:T"),
                 alt.Tooltip("corrupted minus clean:Q", format=".4f")]
    ).properties(height=height)


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
    st.write(
        "Grey is the clean history drawn thick; orange is the corrupted "
        "feed drawn thin on top. Where the two agree you see orange inside "
        "grey. Where grey shows on its own, the feed is wrong there. "
        "Last fifteen months:"
    )
    chart(overlay(view.loc["2025-06-01":]))
    st.write("Around the vendor switch in January 2023 (the splice):")
    chart(overlay(view.loc["2022-06-01":"2023-06-30"], height=220))
    st.write(
        "Corrupted minus clean over the whole history. Zero means the two "
        "files agree; a break means the value is missing; anything else is "
        "a fault. This is the map of everything that was done to this "
        "series."
    )
    chart(difference(clean[col], corrupted[col]))

with tabs[1]:
    st.subheader("Statistical detection, no model needed yet")
    st.write(
        "Three detectors: run length on zero returns (stale), EWMA z-score "
        "on daily moves (spike), calendar completeness (gap). A big move on "
        "its own is not evidence of an error, so spikes get two tiebreakers. "
        "Peers: if correlated series moved the same day it is a market "
        "event. Reversal: a corrupt print is undone the next day when the "
        "feed returns to reality; a genuine regime move or a vendor level "
        "shift is not. Spike plus reversal is a data error and gets a "
        "repair proposal. Spike with neither is a level break, and it is "
        "held for a human rather than interpolated away, because "
        "interpolating a real repricing destroys real history."
    )
    table(findings)

with tabs[2]:
    st.subheader("Repairs are proposed, checked one at a time, and only the "
                 "accepted ones are applied")
    st.write(
        "The fix ladder by trust: interpolate short problems, rebuild long "
        "stretches by regression on correlated peers in change space. Each "
        "proposal is scored alone, using only data the pipeline would really "
        "have. Two deterministic guardrails decide: a KS test that the "
        "repaired region keeps the series' own return distribution, and a "
        "VaR impact check that routes material changes to human review. "
        "Rejected and held proposals are kept for the audit trail but never "
        "touch the staging copy. The golden copy is never edited in place."
    )
    n_acc = sum(c["accepted"] for c in checks)
    n_rev = sum(c["needs_review"] for c in checks)
    n_rej = len(checks) - n_acc - n_rev
    held = findings[findings.get("verdict", pd.Series(dtype=str))
                    == detection.VERDICT_BREAK] if not findings.empty else findings
    a, b, c_, d_ = st.columns(4)
    a.metric("accepted", n_acc)
    b.metric("needs human review", n_rev)
    c_.metric("rejected by guardrail", n_rej)
    d_.metric("level breaks held", len(held))
    for p, c in zip(proposals, checks):
        badge = "auto-accepted" if c["accepted"] else (
            "needs human review" if c["needs_review"] else "rejected")
        with st.expander(f"{p['series']}: {p['rationale']}  [{badge}]"):
            st.json(c)
    if len(held):
        st.write("Held for human review (no automatic repair):")
        table(held[["series", "start", "detail", "verdict"]])
    st.write("Applied point flags (audit trail):")
    table(flags)

    st.subheader("What the pipeline refused to fix")
    st.write(
        f"{len(unresolved)} points have no accepted repair. They are carried "
        "forward so the risk engine can run at all, but carrying a value "
        "forward is the method this project's own evaluation shows has zero "
        "volatility, so it is a stopgap and not a fix. Nothing here is "
        "presented as repaired data: these points stay on the exception "
        "report until a person resolves them, which for the credit spread "
        "gap means choosing a proxy series, because that factor has no "
        "correlated peer to rebuild from. A pipeline that silently filled "
        "these would be worse than one that leaves them visible."
    )
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
    a.metric("99% 1d VaR", f"${var_repaired/1e6:,.2f}M")
    b.metric("Expected shortfall", f"${es/1e6:,.2f}M",
             "the average of the days worse than VaR")
    d.metric("Stressed VaR", f"${svar/1e6:,.2f}M",
             f"window {ws.date()} to {we.date()}")
    st.write(
        "VaR replays the last 500 days of factor moves against the book's "
        "sensitivities and reads the 1st percentile. sVaR runs the same "
        "engine over every rolling 250 day window and keeps the worst: the "
        "search lands on the engineered 2022 stress era, which is the Basel "
        "2.5 idea that capital should not soften just because markets are "
        "calm. Note the data implication: sVaR needs clean history all the "
        "way back, which is why backfilling is a capital problem."
    )
    st.write("Simulated P&L distribution (repaired data, last 500 days):")
    st.bar_chart(pnl.round(-4).value_counts().sort_index(), height=220)

    st.write("How broken does the data have to be before VaR reacts?")
    st.write(
        "Every one of the six series is frozen for a stretch, and the "
        "stretch gets longer. VaR does not meaningfully move until the "
        "outage covers most of the lookback window, because until then the "
        "worst five days are still in there. A stalled feed is close to "
        "invisible in this number."
    )
    table(staleness_sweep(PIPELINE_VERSION))

    bt = risk.backtest(staged)
    n_exc = int(bt["exceedance"].sum())
    st.write(
        f"Backtest, last 250 days: {n_exc} days where the loss was bigger "
        "than the prior day's VaR, against about 2.5 expected at 99 pct. "
        "Too many means the model understates risk; clustered misses are "
        "worse than the count alone."
    )
    st.line_chart(bt[["pnl", "var"]], height=240)

with tabs[4]:
    st.subheader("Sensitivities: the book, factor by factor")
    st.write(
        "The book is held as sensitivities, the standard fast approximation: "
        "bump one factor, the P&L response is the sensitivity times the "
        "move. This is also the diagnostic layer: when VaR moves overnight, "
        "decomposing by factor shows whether positions changed, markets "
        "changed, or one series' history changed."
    )
    table(risk.sensitivities_table())
    st.subheader("Stress scenarios: designed, coherent, no probabilities")
    st.write(
        "Each scenario moves several factors at once, the way real events "
        "do. Severity plus simultaneity; a scenario that shocks rates but "
        "leaves vol untouched is fiction. Not every scenario is a loss: the "
        "point of running several is finding which direction hurts."
    )
    table(risk.stress_pnl())

with tabs[5]:
    st.subheader("Mask and recover: proving a fill method deserves trust")
    st.write(
        "Hide observed points, reconstruct them with each method, score "
        "against truth. Each outage is rebuilt on its own, exactly the way "
        "the pipeline would be called. MAE is plain accuracy. The KS "
        "p-value asks whether the repair keeps the return distribution "
        "shape. tail_ratio is the one that matters most for risk: repaired "
        "volatility over true volatility. Below 1 means the method smooths, "
        "and smoothed history understates VaR and weakens stress "
        "calibration."
    )
    st.write(
        "**The result worth reading twice.** Rank by MAE and you ship "
        "interpolation. Rank by tail preservation, which is what risk "
        "actually needs, and the random forest wins on every series that "
        "has peers to learn from, and it is the only method whose repaired "
        "region comes close to passing the distribution test. The metric "
        "you choose decides the model you deploy, and for risk data average "
        "accuracy is the wrong metric. Note also that no method reaches a "
        "tail ratio of 1: every fill flattens volatility to some degree, "
        "which is the reason filled points stay flagged and never quietly "
        "drive stress calibration."
    )
    st.write(
        "Carry forward wins no prizes on purpose: it is the baseline, its "
        "tail ratio is exactly zero, and it is what the pipeline falls back "
        "to for points it refuses to repair. Series without correlated "
        "peers show no proxy or forest row at all, which is itself a "
        "finding: a factor with no proxy is a factor whose gaps cannot be "
        "rebuilt safely, and that is an escalation, not a computation."
    )
    col = st.selectbox("series to evaluate", list(clean.columns), index=1,
                       key="evalcol")
    table(evaluation.mask_and_recover(clean, col))

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
