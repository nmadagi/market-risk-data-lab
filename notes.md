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

QC round before showing it to anyone. Ran the real app headless through
every tab and widget, read every table the way a viewer would, and fixed
what a careful reader would have asked about.

- Spike detector flagged 28 sigma moves in the first week of 2020. That
  is the EWMA vol estimate with no history, not a finding. Added a warmup.
- The 2022 stress onset and the splice seam were labeled data errors and
  interpolated away. Added the reversal test: a corrupt print is undone
  next day, a regime move or level shift is not. Those are now held as
  level breaks for a human. Interpolating real history is worse than
  leaving a fault in.
- Every proposal showed the same VaR impact because the guardrail measured
  the whole staged frame. Now each proposal is scored alone.
- Rejected proposals were applied to staging anyway. Now only accepted
  ones are, and the audit trail keeps the rest.
- The KS guardrail compared against the clean truth, which production
  never has. Now it compares the repaired region to the series' own
  returns everywhere else.
- A visible consequence of the honest guardrail: the 20 day linear fill of
  the credit spread gap is rejected because it flattens volatility. Kept,
  because that is the guardrail doing its job on screen.
- Added tests/test_app.py: Streamlit's AppTest runs the actual script and
  drives every selectbox option. Unit tests cannot catch a tab that only
  throws when rendered.
- Exported the dataset to data/ as CSV with a column dictionary, and a
  test asserts the snapshot matches the generator so it cannot drift.
- Streamlit's cache hashes a function's own source, not its callees; a
  version string passed into cached functions is what invalidates them.

Second QC round, prompted by a fair question: is any of this ML?

- It was not, so I added one. A random forest now competes with the linear
  proxy on identical peer inputs and identical anchoring, and the harness
  reports it honestly. It loses on average error and wins on tail
  preservation, which for risk data is the criterion that matters. That
  split is the whole argument for benchmarking rather than assuming.
- Answering the question exposed a worse bug than the missing ML. The
  20 day credit spread gap had its repair rejected, correctly, and then
  the points were carried forward into the final data with no flag on
  them. Carry forward has a tail ratio of exactly zero, so the pipeline
  was silently shipping the single worst fill it knows about. There is now
  an explicit unresolved report: any faulty point without an accepted
  repair is listed with the reason and what value is standing in.
- The evaluation harness was handing every hidden day to a method at once.
  Anchored methods then drifted across years of untouched data and scored
  a tail ratio near 10, a mistake the pipeline would never make because
  real outages are contiguous. It now rebuilds one outage at a time.
- The forest was refitting for each of 27 outage blocks, 15 seconds per
  series. Now it fits once and applies everywhere: 0.9 seconds, same
  numbers, and it is also the correct methodology.

Isolation Forest as a second opinion, added when asked whether the
detector could be ML.

- It can, and the honest answer is that it should not be the first line.
  Five scale-free features, scored against the planted faults. One joint
  model at ~30 flags found only the spike; one model per series at the
  same budget found the stale run and the splice seam and missed the
  spike and the gap. Same budget, different answer. The rules found all
  four both ways. At ~200 flags the forest finds everything with ~170
  false alarms, about half of them in the 2022 stress era. Rules encode
  what a person knows; the forest has to rediscover it, and what it
  rediscovers depends on how it is fit.
