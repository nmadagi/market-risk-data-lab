# market-risk-data-lab

I built this to demonstrate the layer of market risk that rarely gets demo
apps: the data underneath the models. Trading book risk numbers (VaR,
stressed VaR, sensitivities, stress tests) all consume the same risk factor
time series, and when one series goes bad the numbers go quietly wrong. This
app injects realistic data faults into a synthetic six-factor history,
detects them statistically, repairs them under deterministic guardrails, and
proves the repairs with a mask-and-recover evaluation harness.

The design principle throughout: automated proposals, deterministic
acceptance, flagged and reversible changes. The LLM writes the morning
report; it never touches data, and a number-check guardrail rejects any
draft containing a figure not present in the computed facts.

## The finding

I built this expecting to show that bad data visibly distorts VaR. It
mostly does not, and that turned out to be the more useful result.

With all four faults injected, 99% VaR moves +0.1% and looks completely
normal. Expected shortfall on the same data moves +56%. The reason is
structural: 99% VaR over a 500 day window is the 5th worst day, so one
corrupt print shifts the ranking by a single place and the number
absorbs it. Expected shortfall averages the tail, so it takes the full
weight of the fake loss. Freezing every series for 20% of the lookback
window still leaves VaR unchanged.

Two things follow. You cannot use the headline risk number as your data
alarm, which is the argument for dedicated data quality monitoring. And
as the industry shifts from VaR toward expected shortfall, data errors
get more load bearing, not less.

After detection and repair, expected shortfall returns to within 5% of
its clean value with every filled point flagged. The stressed VaR window
search independently lands on the engineered 2022 high volatility era at
roughly 2.9x ordinary VaR.

## What it covers

| Capability | Where |
|---|---|
| Time series construction (golden copy, seeded synthetic) | data/generate.py |
| Fault injection: stale, spike, gap, vendor splice | src/corruption.py |
| Anomaly detection: run-length, EWMA z-score, calendar, cross-sectional tiebreaker | src/detection.py |
| Remediation ladder with staging, flags, KS and VaR-impact guardrails | src/remediation.py |
| Historical simulation VaR, sVaR window search, ES, sensitivities, stress scenarios, backtesting | src/risk.py |
| Mask-and-recover evaluation (MAE, KS, tail preservation) | src/evaluation.py |
| LLM narrative with number-check guardrail and template fallback | src/agent.py |
| 36 pytest cases | tests/ |

## Run it

    pip install -r requirements.txt
    streamlit run app.py
    python -m pytest tests/

Optional: set ANTHROPIC_API_KEY to enable the LLM morning report; without
it the deterministic template is used, which is the point of the fallback.

All data is synthetic and seeded. The portfolio is a toy book held as
sensitivities. Nothing here claims production scale; the architecture is
the point.
