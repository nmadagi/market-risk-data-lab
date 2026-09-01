# build notes

Decisions made while building, in order.

- Sensitivity-based P&L instead of full revaluation. A toy book priced by
  bump-times-move keeps the risk engine honest and small; full reval would
  add pricing-model surface area without changing the data story.
- Rates move in absolute bp, FX in log returns. Standard practice and it
  matters near zero rates; also gives the mask-and-recover harness the
  right units to score against.
- Stress era engineered into 2022 in the generator. The sVaR window search
  has to find something real, otherwise the tab is a tautology.
- Splice detection rides on the spike detector. A vendor level-shift
  surfaces as one huge z-score move at the seam. A dedicated CUSUM
  detector was considered and dropped for scope; the seam is caught either
  way. TODO: proper level-break test if this grows.
- Guardrails route rather than reject on VaR materiality. A big VaR impact
  from a repair is not evidence the repair is wrong; it is evidence a human
  should sign it. KS failure rejects, materiality routes.
- Carry-forward kept as an evaluated method on purpose. It loses on
  tail_ratio, which is the demonstration that average accuracy is not the
  right score for risk data.
- Number-check guardrail compares floats, not strings, and strips series
  names first (usd5y contains a digit). Both were real bugs found by tests.
- No database, no API needed to run. Streamlit cache holds the one scenario;
  deterministic seed means every viewer sees the same story.
