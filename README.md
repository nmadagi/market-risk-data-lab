# Market Risk Data Lab

I built this to demonstrate the layer of market risk that rarely gets demo
apps: the data underneath the models. Trading book risk numbers (VaR,
stressed VaR, sensitivities, stress tests) all consume the same risk factor
time series, and when one series goes bad the numbers go quietly wrong. This
app injects realistic data faults into a synthetic six-factor history,
detects them statistically, repairs them under deterministic guardrails, and
proves the repairs with a mask-and-recover evaluation harness.

Live: https://market-risk-data-lab.streamlit.app

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
its clean value with every applied point flagged. The stressed VaR window
search independently lands on the engineered 2022 high volatility era at
roughly 2.9x ordinary VaR.

## How detection decides what a big move is

A big move on its own is not evidence of an error. Every spike gets two
tiebreakers. Peers: if correlated series moved the same day, it is a
market event. Reversal: a corrupt print is undone the next day when the
feed returns to reality; a genuine regime move or a vendor level shift is
not. Spike plus reversal is a data error and gets a repair proposal. Spike
with neither is a level break and is held for a human, because
interpolating a real repricing destroys real history. In the demo, the
2022 stress onset and the vendor splice seam are both held, not repaired.

## Rules first, machine learning as a scored second opinion

Detection is three plain statistical rules plus two tiebreakers. An
Isolation Forest runs alongside them on five scale-free features per
series-day and is scored against the rules on the four planted faults.
At a realistic alert budget of about 30 flags the rules find 4 of 4 and
the forest finds 2 of 4, with about half its false alarms in the 2022
stress era. Given a budget of about 200 flags it finds everything, at the
cost of roughly 170 false alarms. And which two it catches at the small
budget changed when the fit changed from one joint model to one per
series, while the rules found all four both ways. That is the honest
shape of unsupervised anomaly detection on market data: it cannot tell a
regime change from a data error, its answer depends on fitting choices a
rule does not have, and the rules encode what a person already knows. The
forest earns its weight on that scorecard, not by assumption.

## Which fill method to trust, and the metric that decides

A mask-and-recover harness hides observed points, rebuilds each outage the
way the pipeline would, and scores four methods against the clean data: carry
forward, linear interpolation, an OLS regression on correlated peers, and
a random forest on the same peer inputs. The forest gets identical
features and identical anchoring, so the benchmark isolates the functional
form rather than flattering the fancier model.

The result is worth reading twice. Rank by average error and you ship
interpolation. Rank by tail preservation, which is what risk data actually
needs, and the random forest wins on every series that has peers to learn
from, and it is the only method whose repaired region comes close to
passing the distribution test. The metric you choose decides the model you
deploy. No method reaches a tail ratio of 1: every fill flattens
volatility to some degree, which is exactly why filled points stay flagged
and never quietly drive stress calibration.

## What the pipeline refuses to fix

A repair that fails its guardrail is not applied. Those points are carried
forward so the risk engine can run, but carry forward is the method with a
tail ratio of zero, so it is a stopgap and not a fix. Every such point is
listed on an exception report with the reason, and the credit spread gap
sits there permanently in the demo because that factor has no correlated
peer to rebuild from. A factor with no usable proxy is an escalation, not
a computation. A pipeline that silently filled these would be worse than
one that leaves them visible.

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
| Fault injection: stale, spike, gap, vendor splice | src/corruption.py |
| Anomaly detection: run-length, EWMA z-score, calendar, peer and reversal tiebreakers | src/detection.py |
| Remediation ladder, per-proposal KS and VaR-impact guardrails, accepted-only staging, flags | src/remediation.py |
| Historical simulation VaR, sVaR window search, ES, sensitivities, stress scenarios, backtesting | src/risk.py |
| Mask-and-recover evaluation (MAE, KS, tail preservation) | src/evaluation.py |
| LLM narrative with number-check guardrail and template fallback | src/agent.py |
| Random forest imputation benchmarked against the simple methods | src/remediation.py, src/evaluation.py |
| Isolation Forest anomaly detection scored against the rules | src/ml_detection.py |
| 65 tests, including a headless run of the app through every tab and widget | tests/ |

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
