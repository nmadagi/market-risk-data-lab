# Market Risk Data Lab

I built this to demonstrate the layer of market risk that rarely gets demo
apps: the data underneath the models. Trading book risk numbers (VaR,
stressed VaR, sensitivities, stress tests) all consume the same risk factor
time series, and when one series goes bad the numbers go quietly wrong. This
app injects realistic data faults into a synthetic six-factor history,
detects them with a model trained on synthetic faults, repairs them under
deterministic guardrails, and proves the repairs with a mask-and-recover
evaluation harness.

Live: https://market-risk-data-lab.streamlit.app

The design principle throughout: automated proposals, deterministic
acceptance, flagged and reversible changes. The LLM writes the morning
report; it never touches data, and a number-check guardrail rejects any
draft containing a figure not present in the computed facts.

## The finding

I built this expecting to show that bad data visibly distorts VaR. It
mostly does not, and that turned out to be the more useful result.

With all three faults injected, 99% VaR moves +0.1% and looks completely
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
its clean value with every applied point flagged. The stressed VaR window
search independently lands on the engineered 2022 high volatility era at
roughly 2.9x ordinary VaR.

## How a big move gets decided

A big move on its own is not evidence of an error. The model reads two
tiebreakers among its five features: did correlated series move the same
day (a market event), and was the move undone the next day (a bad print).
A big move with neither is a possible level shift, and it is held for a
human rather than repaired, because interpolating a real repricing
destroys real history. In the demo, one such day across six years is
held; it was not planted.

## How repairs are accepted

Each proposal is scored alone, using only data the pipeline would really
have (no clean reference anywhere). A KS test checks that the repaired
region keeps the series' own return distribution; a VaR impact check
routes material changes to human review. Only accepted proposals touch
the staging copy. In the demo, a 20 day linear interpolation of the credit
spread gap is rejected by the KS test because it flattens volatility,
which is exactly the failure that guardrail exists to catch.

## What it covers

| Capability | Where |
|---|---|
| Time series construction (golden copy, seeded synthetic) | data/generate.py |
| The dataset as CSV, with a column dictionary | data/ |
| Fault injection: stale, spike, gap (vendor splice used in training worlds) | src/corruption.py |
| Anomaly detection: run-length, EWMA z-score, calendar, peer and reversal tiebreakers | src/detection.py |
| Remediation ladder, per-proposal KS and VaR-impact guardrails, accepted-only staging, flags | src/remediation.py |
| Historical simulation VaR, sVaR window search, ES, sensitivities, stress scenarios, backtesting | src/risk.py |
| Mask-and-recover evaluation (MAE, KS, tail preservation) | src/evaluation.py |
| LLM narrative with number-check guardrail and template fallback | src/agent.py |
| Random forest imputation benchmarked against the simple methods | src/remediation.py, src/evaluation.py |
| Supervised fault classifier trained on synthetic worlds, tested on a held-out world | src/ml_detection.py |
| 75 tests, including a headless run of the app through every tab and widget | tests/ |

## Run it

    pip install -r requirements.txt
    streamlit run app.py
    python -m pytest tests/
    python -m data.export

Optional: set ANTHROPIC_API_KEY to enable the LLM morning report; without
it the deterministic template is used, which is the point of the fallback.

All data is synthetic and seeded. The portfolio is a toy book held as
sensitivities. Nothing here claims production scale; the architecture is
the point.
