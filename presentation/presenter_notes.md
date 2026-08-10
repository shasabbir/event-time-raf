# Event-TimeRAF Presentation Notes

## Core explanation

The task uses 168 hourly PM2.5 observations plus source-audited context to
forecast the next 24 hourly values. SACB constructs the aligned 85-feature
origin context. LSER first applies a strict temporal embargo, then ranks 1,521
training-history windows at the primary 24-hour stride using PM2.5 shape,
weather, calendar, and event
similarity. DFEH evaluates 24 direct XGBoost horizon models and a frozen
Chronos-Bolt forecast-fusion path while storing traceable evidence.

The full M09 input is 151 per horizon: 85 context features, 51 retrieval
features, six drift fields, and nine future-calendar features.

## Result statement

M04 context XGBoost is the strongest verified model: MSE 26.185, MAE 3.125,
RMSE 5.117, and R-squared 0.379. Full M09 records MSE 26.734, so it does not beat
M04 overall. M11 changes frozen Chronos-Bolt MSE from 28.941 to 28.733, but its
paired interval crosses zero. The climatology fusion control reaches 27.450 and
significantly outperforms M11. The contribution is therefore an audited
evaluation and a bounded negative result.

## Likely questions

**Why is M04 better than the full model?**
Weather and calendar features carry strong local predictive signal. Dense
retrieval improves retrieval alone, but the extra retrieval, event, and drift
fields do not add value beyond that context in the tested feature model.

**What is novel?**
The study combines source and availability audits, a candidate-target embargo,
event-conditioned transparent retrieval, robust drift evidence, frozen-TSFM
fusion, and per-origin evidence records in one reproducible environmental
forecasting protocol.

**Is this a reproduction of TimeRAF?**
No. It adapts TimeRAF's retrieve-then-forecast principle. It does not implement
the learned dual encoder or internal Channel Prompting.

**How is leakage prevented?**
Every retrieved candidate's target end must be earlier than the query input
start, and the knowledge base contains training windows only.

**Were ablations actually run?**
Yes. The final run includes all stride and event controls. The verification
notebook recalculates the six primary comparisons using 2,000 resamples of
168-origin blocks, HAC lag 167, and Holm adjustment. A negative A-minus-B
difference favors A.

**What does event mean?**
Official NOAA Storm Events records mapped to Los Angeles County. The 39 model
features include rolling event counts, active-state indicators, category
counts, recent-event counts, and an event-burst ratio.

**Are event features real-time?**
Not established. NOAA detail files lack machine-readable publication times, so
the study uses event start as a retrospective availability proxy.

**Did the event channel affect retrieval?**
Yes. Weight 0.2 changed at least one top-eight candidate for 91.4% of test
queries, but MSE increased from 38.080 to 38.411. The channel is active but the
tested representation is not useful for forecast accuracy.

**Can the PC run the live notebook?**
Yes. The verification notebook takes about 2-3 minutes locally because it loads
archived models rather than retraining. Full training remains the Kaggle job.

## Live-run checklist

1. Place `event_timeraf_final_run_20260810T103436161252Z.zip` in the repository root.
2. Open `notebooks/03_paper_claim_verification.ipynb`.
3. Restart the kernel and run all cells.
4. Show the 54/54 manifest check, main metrics, primary inference, stride and
   event controls, and final `PASS WITH EXPLICIT LIMITATIONS` message.
5. Keep `paper/verification_log.pdf` open for claim-level questions.
# Presentation Build Status

The generated PPTX/PDF in this directory predates final run
`20260810T103436161252Z`. The source notes below are synchronized, but the deck
must be regenerated and visually verified before it is used for submission or
live presentation.
