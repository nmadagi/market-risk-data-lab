"""market-risk-data-lab: what bad data does to VaR, and a pipeline that
detects it, repairs it under guardrails, and proves the repair.

Read the tabs left to right; they are the pipeline in order.
"""
import pandas as pd
import streamlit as st

from data.generate import generate_market_data
from src import agent, detection, evaluation, remediation, risk
from src.corruption import apply_default_faults, inject_stale

st.set_page_config(page_title="market-risk-data-lab", layout="wide")


@st.cache_data
def load_all():
    clean = generate_market_data()
    corrupted, fault_log = apply_default_faults(clean)
    findings = detection.run_all(corrupted)
    proposals = remediation.propose(corrupted, findings)
    staged, flags = remediation.apply_to_staging(corrupted.ffill(), proposals)
    checks = [remediation.guardrail_check(clean, staged, corrupted, p)
              for p in proposals]
    return clean, corrupted, fault_log, findings, proposals, staged, flags, checks


clean, corrupted, fault_log, findings, proposals, staged, flags, checks = load_all()


@st.cache_data
def staleness_sweep():
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
        "splice. Pick a series to compare clean vs corrupted."
    )
    st.dataframe(fault_log, use_container_width=True)
    col = st.selectbox("series", list(clean.columns), index=1)
    view = pd.DataFrame({"clean": clean[col], "corrupted": corrupted[col]})
    st.line_chart(view.loc["2025-06-01":])
    st.line_chart(view.loc["2022-06-01":"2023-06-30"],
                  height=200)

with tabs[1]:
    st.subheader("Statistical detection, no model needed yet")
    st.write(
        "Three detectors: run length on zero returns (stale), EWMA z-score "
        "on daily moves (spike and splice seam), calendar completeness "
        "(gap). Spikes get the cross sectional tiebreaker: if correlated "
        "peers moved the same day it is likely a real market event, not an "
        "error. That check is what keeps a detector from deleting real "
        "history."
    )
    st.dataframe(findings, use_container_width=True)

with tabs[2]:
    st.subheader("Repairs are proposed, checked, staged, and flagged")
    st.write(
        "The fix ladder by trust: interpolate short problems, rebuild long "
        "stretches by regression on correlated peers. Nothing edits the "
        "golden copy: fixes land in a staging copy, every filled point is "
        "flagged, and two deterministic guardrails decide acceptance: a KS "
        "test that the repair preserves the return distribution, and a VaR "
        "impact check that routes material changes to human review."
    )
    for p, c in zip(proposals, checks):
        badge = "auto-accepted" if c["accepted"] else (
            "needs human review" if c["needs_review"] else "rejected")
        with st.expander(f"{p['series']}: {p['rationale']}  [{badge}]"):
            st.json(c)
    st.write("Filled point flags (audit trail):")
    st.dataframe(flags, use_container_width=True)
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
    st.dataframe(staleness_sweep(), use_container_width=True)

    bt = risk.backtest(staged)
    n_exc = int(bt["exceedance"].sum())
    st.write(
        f"Backtest, last 250 days: {n_exc} exceedances vs about 2.5 "
        "expected at 99 pct. Too many means the model understates risk; "
        "clustered exceedances are worse than the count alone."
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
    st.dataframe(risk.sensitivities_table(), use_container_width=True)
    st.subheader("Stress scenarios: designed, coherent, no probabilities")
    st.write(
        "Each scenario moves several factors at once, the way real events "
        "do. Severity plus simultaneity; a scenario that shocks rates but "
        "leaves vol untouched is fiction."
    )
    st.dataframe(risk.stress_pnl(), use_container_width=True)

with tabs[5]:
    st.subheader("Mask and recover: proving a fill method deserves trust")
    st.write(
        "Hide observed points, reconstruct them with each method, score "
        "against truth. MAE is plain accuracy. The KS p-value asks whether "
        "the repair keeps the return distribution shape. tail_ratio is the "
        "one that matters most for risk: repaired volatility over true "
        "volatility. Below 1 means the method smooths, and smoothed history "
        "understates VaR and weakens stress calibration. Carry forward wins "
        "no prizes here on purpose: it is the baseline that shows why "
        "evaluation must happen before trust."
    )
    col = st.selectbox("series to evaluate", list(clean.columns), index=1,
                       key="evalcol")
    st.dataframe(evaluation.mask_and_recover(clean, col),
                 use_container_width=True)

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
        "(run length, EWMA z-score, calendar, cross sectional tiebreaker) > "
        "propose fixes (interpolation or peer regression) > guardrail "
        "checks (KS distribution test, VaR impact routing) > staged apply "
        "with flags > risk engine (hist sim VaR, sVaR window search, "
        "sensitivities, coherent stress scenarios, backtesting) > mask and "
        "recover evaluation harness."
    )
