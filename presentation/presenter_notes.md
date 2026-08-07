# Event-TimeRAF Presentation Notes

## Core explanation

The task uses 168 hourly PM2.5 observations plus source-audited context to
forecast the next 24 hourly values. SACB constructs the aligned 85-feature
origin context. LSER first applies a strict temporal embargo, then ranks 191
training-history windows using PM2.5 shape, weather, calendar, and event
similarity. DFEH evaluates 24 direct XGBoost horizon models and a frozen
Chronos-Bolt forecast-fusion path while storing traceable evidence.

The full M09 input is 151 per horizon: 85 context features, 51 retrieval
features, six drift fields, and nine future-calendar features.

## Result statement

M04 context XGBoost is the strongest verified model: MSE 26.185, MAE 3.125,
RMSE 5.117, and R-squared 0.379. Full M09 records MSE 26.712, so it does not beat
M04 overall. M11 changes frozen Chronos-Bolt MSE from 28.941 to 28.709, but its
paired interval crosses zero. The contribution is therefore an audited
framework with selective gains, not universal baseline superiority.

## Likely questions

**Why is M04 better than the full model?**
Weather and calendar features carry strong local predictive signal. The
retrieval knowledge base has only 191 non-overlapping windows, storm-event
context is sparse, and the extra retrieval/drift fields can add variance.

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
Yes. The verification notebook reruns saved predictors and calculates eight
paired comparisons using 500 resamples of 24-hour blocks. A negative A-minus-B
difference favors A.

**What does event mean?**
Official NOAA Storm Events records mapped to Los Angeles County. The 39 model
features include rolling event counts, active-state indicators, category
counts, recent-event counts, and an event-burst ratio.

**Are event features real-time?**
Not established. NOAA detail files lack machine-readable publication times, so
the study uses event start as a retrospective availability proxy.

**Why was M12 removed?**
It was created after the immutable run and the edited result files no longer
matched manifest hashes. Keeping it would violate the evidence contract.

**Why does random retrieval need archived evidence?**
The run used NumPy 2.0.2. NumPy 2.4.1 produces a different seeded `choice`
stream. The notebook therefore reconstructs M05/A01 from hash-verified saved
candidate rows; exact retraining requires the archived environment.

**Can the PC run the live notebook?**
Yes. The verification notebook takes about 2-3 minutes locally because it loads
archived models rather than retraining. Full training remains the Kaggle job.

## Live-run checklist

1. Place `event_timeraf_final_run.zip` in the repository root.
2. Open `notebooks/03_paper_claim_verification.ipynb`.
3. Restart the kernel and run all cells.
4. Show the 43/43 manifest check, model reconciliation table, main metrics,
   ablation table, and final `PASS WITH EXPLICIT LIMITATIONS` message.
5. Keep `paper/verification_log.pdf` open for claim-level questions.
