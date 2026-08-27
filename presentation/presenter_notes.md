# TRACE-RAF Presentation Notes

Authoritative run: `20260827T043457543402Z`

## Core Message

TRACE-RAF is a supervised, leakage-safe residual analogue forecasting method.
It is not a foundation model. It stores out-of-fold residual trajectories from
the training period, retrieves only historically available analogues, and
applies their residual correction through a validation-selected trust gate.
NOAA event context is an investigated auxiliary signal, not a demonstrated
source of improvement in this holdout.

## Slide Narrative

### 1. Problem and Scope

Forecast Los Angeles County hourly PM2.5 for the next 24 hours from a 168-hour
history. Inputs combine EPA PM2.5, NOAA hourly weather, calendar covariates, and
retrospectively aligned NOAA Storm Events. The study is a one-county
methodological evaluation, not an operational deployment claim.

### 2. Leakage-Safe Data Design

The pipeline produces 56,194 windows: 45,784 train, 7,950 validation, and 2,460
test origins. The test set therefore contains 59,040 horizon-level forecasts.
All preprocessing, model selection, retrieval weights, and gate strength are
fit without test targets.

### 3. Base Forecast and Residual Memory

The base is a validation-selected supervised ensemble. Each training-memory
entry stores an out-of-fold base-model residual rather than a raw future target.
At query time, candidate entries must precede the forecast origin and pass the
configured exclusion gap. This design prevents target leakage and makes the
retrieved evidence auditable.

### 4. Retrieval and Trust Gate

Similarity combines PM2.5 shape, weather, calendar, and event context. Retrieved
residuals are aggregated into a candidate correction. The learned trust gate
uses retrieval confidence and distribution-shift diagnostics to attenuate that
correction. The selected gate strength is 0.25; the observed mean effective gate
is 0.090.

### 5. Mechanism Ladder

Use this ablation sequence to explain what was actually learned:

| Variant | Validation choice / test result | Interpretation |
|---|---|---|
| C06 base ensemble | MSE 40.115 | Starting point |
| A04 base + raw analogue | weight 0; same as base | Raw-target fusion was rejected |
| A05 base + ungated residual | MSE 44.959 | Uncontrolled residual transfer is harmful |
| A06 constant residual | strength 0; same as base | A global correction was rejected |
| M13 gated residual TRACE | MSE 40.062 | Selective gating recovers a small nominal gain |

The full forecast is verified pointwise as base plus gate times retrieved
residual. The raw correction has mean absolute magnitude 1.708 micrograms per
cubic metre; gating reduces it to 0.149.

### 6. Main Results

Ridge is the strongest tested model: MSE 39.144, MAE 4.079, RMSE 6.257, and
R-squared 0.457. TRACE records MSE 40.062, MAE 4.134, RMSE 6.329, and R-squared
0.444. TRACE is competitive but does not outperform all conventional baselines.

### 7. Statistical Interpretation

TRACE improves over C06 by 0.0212 MAE. Its unadjusted bootstrap interval excludes
zero, but the 29-comparison family gives Holm-adjusted bootstrap p = 0.355 and
DM p = 0.381. State this as a nominal improvement that is not statistically
confirmed after multiplicity correction.

### 8. Event and Foundation-Model Findings

TRACE and event-free A03 are nearly identical: their MAE difference is 0.0003,
and only 70 test origins have event context. The event-specific benefit is
therefore unverified. Chronos-Bolt is a separately evaluated frozen TSFM. The
tested event-conditioned output-space fusion worsens Chronos MAE from 4.202 to
4.574; this does not justify a general conclusion about retrieval-augmented
foundation models.

### 9. Evidence and Limitations

Notebook 02 verifies 106 manifest hashes, 43 tables, exact prediction
decomposition, and manuscript claims. Limitations are the retrospective event
availability assumption, one county, spatial aggregation, weak event support,
and no hardware-normalized runtime comparison.

### 10. Contribution and Next Experiment

The contribution is the combination of out-of-fold residual analogue memory,
temporal leakage controls, a selective trust gate, and evidence-preserving
evaluation. The highest-value extension is external validation in a second
county with richer event exposure and true issue timestamps.

## Live Notebook Sequence

1. Run `01_event_timeraf_kaggle_pipeline.ipynb` only when rebuilding the full
   experiment from attached official data.
2. Run `02_results_and_figures.ipynb` live to verify hashes, recompute metrics
   and inference, check decomposition, and regenerate figures.
3. Run Notebook 03 afterward for the concise cross-table claim review.
4. Show the run ID, 106/106 hash check, split reconciliation, model metrics,
   mechanism ladder, paired inference, and exact TRACE decomposition.

## Likely Q&A

**What is novel if Ridge performs better?**

Novelty is methodological rather than leaderboard superiority: residual
analogue transfer is trained out of fold, retrieval is temporally constrained,
the correction is trust-gated, and every final prediction retains its retrieved
evidence and decomposition.

**Does TRACE prove that events improve PM2.5 forecasts?**

No. The event-free ablation is almost identical and the event subset is small.
The experiment tests an event channel and documents why stronger event data are
needed.

**Why does the ungated residual model fail?**

Residual analogues are not uniformly transferable. A05 applies them without
confidence control and increases MSE to 44.959. The gate suppresses uncertain
corrections and reduces their average magnitude by more than an order of
magnitude.

**Is TRACE-RAF a foundation model?**

No. TRACE-RAF is supervised. Chronos-Bolt is the frozen foundation-model
baseline, and only one output-space retrieval configuration was tested.

**Can this be called concept-drift adaptation?**

Not as a general causal claim. The implementation provides a reproducible
distribution-shift diagnostic and model-specific regime label; external
temporal and geographic validation is still required.
