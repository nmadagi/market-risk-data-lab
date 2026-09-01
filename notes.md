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

Three bugs worth recording, all found by inspecting output rather than by
a failing test, which is why each now has a test.

- Implied vol was a plain random walk and drifted to 1000+ vol points over
  six years. No market has ever produced that. Implied vol is anchored, so
  the generator now mean reverts in log space around 80. Caught by actually
  reading the generated table instead of trusting the chart shape.
- The proxy fill regressed LEVELS. The fitted level did not meet the last
  real observation, so the repair injected a fake 44bp move on its first
  day, which the risk engine read as an 8.3M loss: the repair was creating
  a worse artifact than the fault. Now it regresses CHANGES, anchors to the
  last good value, and linear-bridges the residual so the far edge lands on
  the next real observation. Risk models consume returns, so a repair has
  to be right in return space, not just look right on a chart.
- The original headline claim (stale feed understates VaR ~4%) did not
  survive the vol fix: it had been an artifact of vol dominating P&L. The
  honest result is that VaR barely moves at all, for a structural reason,
  and that is now the app's actual thesis. Kept as a lesson: verify the
  demonstration reproduces after any change to the data generator.
