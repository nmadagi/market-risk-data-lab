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

## The headline numbers

A 15-day stale feed on one curve node understates 99% VaR by about 4%
without any error being visible in the reported number. The pipeline
detects it, repairs it via peer regression, and restores VaR to within
rounding of the clean value, with every filled point flagged. The stressed
VaR window search lands on the engineered 2022 high-volatility era at
roughly 2.4x ordinary VaR.

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
| 32 pytest cases | tests/ |

## Run it

    pip install -r requirements.txt
    streamlit run app.py
    python -m pytest tests/

Optional: set ANTHROPIC_API_KEY to enable the LLM morning report; without
it the deterministic template is used, which is the point of the fallback.

All data is synthetic and seeded. The portfolio is a toy book held as
sensitivities. Nothing here claims production scale; the architecture is
the point.
