# Event-TimeRAF Review Action Plan

## Execution Status (2026-08-10)

Implemented and tested:

- Dense causal knowledge bases with 192-, 24-, 6-, and 1-hour stride settings.
- Calendar retrieval, hourly-by-month climatology, and regularized linear controls.
- Event-weight and event-stratified retrieval sensitivity with persisted candidate composition.
- A 2,000-resample, 168-origin paired block bootstrap, overlap-aware DM tests, and Holm adjustment.
- RMSE-based diagnostic uncertainty scale, upper-tail drift component diagnostics, origin attrition, and subset target variance.
- Run-specific output directories so revised experiments cannot overwrite archived runs.

Validated in a full-data local smoke run:

- 48,332 valid windows, including 7,199 test origins, were reconstructed from the official local caches.
- All four retrieval-only stride settings completed and every saved selected candidate satisfied the temporal embargo.
- The primary models, controls, explanations, figures, and results notebook completed without mixed-run artifacts.

Completed final run and manuscript synchronization:

- Kaggle run `20260810T103436161252Z` completed with fresh Chronos inference,
  final mode, all four retrieval strides, and all four M08/M09 stride models.
- All 54 manifest entries were verified by byte size and SHA-256.
- The verification notebook, manuscripts, figures, log, presentation source,
  and presenter notes were synchronized to this run only. The older generated
  PPTX/PDF is excluded from the final package pending regeneration and visual
  verification.
- The paper was retitled and reframed as an audited evaluation and bounded
  negative result because the final evidence does not support retrieval-specific
  or event-specific superiority.
- DLinear and PatchTST remain deferred because adding a new sequence-training framework before the corrective run would materially expand the study and reproducibility surface.

The local smoke run is not a publication run: it reused shape-checked archived Chronos predictions and intentionally disabled the per-stride M08/M09 retraining. Its values must not be copied into the manuscript.

## 1. Objective

Revise Event-TimeRAF so that every reported claim is supported by a reproducible
notebook result, the retrieval experiment is sufficiently powered to be
interpretable, and the paper accurately distinguishes forecasting performance,
event-context evidence, drift diagnostics, and provenance traceability.

The revision will remain within the current Los Angeles County, 2019--2024,
24-hour PM2.5 forecasting study. Site-level modelling, 2025 data, full TimeRAF
Channel Prompting, and a large collection of additional deep models are deferred
unless the revised core experiment remains insufficient.

## 2. Preserve the Current Verified Run

Before changing code or configuration:

- Preserve run `20260723T112033170131Z` and its manifest-backed artifacts as the
  original experiment.
- Do not overwrite the current predictions, evidence, tables, or manifest.
- Give the revised experiment a new run identifier and output directory.
- Keep the current notebooks as historical evidence until the revised run has
  passed all verification checks.

## 3. Phase A: Corrections That Do Not Require a Model Rerun

### A1. Align the claims with the present evidence

- State directly that M09 is worse than M04 on the overall, event, and drift
  subsets in the current run.
- Describe the Chronos retrieval result as a small, statistically unresolved
  difference, not as a confirmed gain.
- Use `traceability` or `evidence record` for the structured explanation output.
  Do not claim evaluated explainability or causal explanation.
- Describe the drift mechanism as a distribution-shift diagnostic. It does not
  adaptively select or update a forecaster.
- Retain the retrospective event-availability caveat prominently.

### A2. Resolve reporting inconsistencies

- Remove the remaining M12 reference from both manuscript versions. M12 is not
  part of the verified result set.
- Define A00 and A01 completely, including their feature sets and purpose.
- State the exact test-period dates.
- Explain that three categories pass the configured minimum-record threshold,
  while five mapped categories exist in the source data.
- Add an origin-attrition table: raw hours, aligned hours, valid windows,
  train/validation/test exclusions, and final evaluated origins.
- State every audit threshold beside its pass/fail result.
- Define `epsilon` and replace manually typed equation numbers with references.
- Replace the author placeholder and remove draft-only IEEE page-style overrides
  before submission.

### A3. Improve presentation without changing the study

- Shorten the abstract while preserving all verified headline values.
- Remove repeated contribution and configuration descriptions.
- Explicitly discuss the event- and drift-subset reversals below their table.
- Add PM2.5 units to the forecast-case figure.
- Move plot legends so they do not obscure early forecast horizons.
- Reduce or rotate crowded date labels in retrieval diagnostics.

## 4. Phase B: Essential Experiment Corrections

These changes require a complete new run and new manifest.

### B1. Expand the retrieval knowledge base

The current 192-hour stride leaves only 191 candidates and is the highest-impact
valid criticism in the review.

- Remove the configuration rule requiring knowledge-base candidates to be
  mutually non-overlapping.
- Preserve the existing query--candidate embargo: every candidate target must end
  before the query lookback begins.
- Evaluate knowledge-base strides of 24, 6, and 1 hour.
- Record candidate count, eligible count per query, runtime, and memory use for
  each stride.
- Re-run retrieval-only and feature-level models M05--M09 for each stride.
- Keep `k = {1, 4, 8, 16}` as a secondary sensitivity axis.
- If stride 1 is computationally excessive, report that fact and retain stride 6
  as the densest completed setting; do not silently omit it.

Acceptance checks:

- All selected candidate targets satisfy the embargo.
- Candidate overlap does not create query-future leakage.
- Retrieval is deterministic for a fixed seed and persisted candidate IDs.
- The paper does not generalize beyond the evaluated stride range.

### B2. Add defensible retrieval and fusion controls

- Add an hour-of-day by month climatology baseline.
- Add calendar-only retrieval as a seasonal retrieval control.
- In Chronos fusion, replace the retrieved forecast with climatology and then
  persistence, selecting the fusion weight on validation exactly as for M11.
- Compare retrieval fusion against both placebo fusions on identical test origins.

Interpretation rule:

- Claim retrieval-specific fusion value only if it outperforms the placebo
  fusions under the revised paired comparison.
- Otherwise describe the result as ordinary forecast averaging.

### B3. Test whether the event channel can affect retrieval

- Evaluate event weights `delta = {0, 0.2, 0.5, 0.8, 1.0}` with the remaining
  channel weights renormalized.
- Report how often adding the event channel changes at least one top-k candidate.
- Report event-score sparsity and candidate composition for event and non-event
  queries separately.
- Add event-stratified retrieval for event-flagged queries, with a documented
  fallback when fewer than k eligible event candidates exist.
- Compare hybrid, no-event, and event-stratified retrieval on all origins and on
  the event subset.

Do not claim that events lack signal if the event channel rarely changes the
retrieved set. In that case, conclude only that the tested event representation
or retrieval rule was ineffective.

### B4. Correct statistical inference

- Increase paired bootstrap resamples from 500 to at least 2,000.
- Use blocks of at least 168 forecast origins, or document a data-driven
  stationary-bootstrap block length.
- Predefine the small family of primary model comparisons before the final run.
- Apply Holm correction across that family.
- Add a Diebold--Mariano comparison with an overlap-aware/HAC variance estimate
  if implementation and assumptions can be verified.
- Report effect sizes and intervals without converting a zero-crossing interval
  into a positive or negative performance claim.

### B5. Correct the evidence uncertainty field

- Replace validation MAE with validation RMSE if the field is intended as a
  residual scale.
- Rename the output to `diagnostic_uncertainty_scale` and continue to state that
  it is not a calibrated prediction interval.
- Regenerate all 7,199-or-more explanation records in the revised run.
- Update tests, the schema description, equation, notebook output, and paper
  together.

### B6. Retain or revise the drift diagnostic conservatively

Preferred scoped option:

- Keep the one-sided transform, rename it an `upper-tail shift score`, and state
  explicitly which shifts it cannot detect.
- Report each component's zero rate, distribution, and pairwise correlation.
- Report validation and test flag rates and discuss a test rate below the
  validation-calibrated 10% threshold.

Larger optional option:

- Implement and compare a two-sided robust deviation score.
- Treat this as a new ablation, not a silent correction to the previous run.

Benchmarking ADWIN, KSWIN, Page-Hinkley, or MMD is deferred unless concept-drift
detection remains a central title claim.

## 5. Phase C: Focused Baseline Improvements

Required for a journal-oriented revision:

- Hour-of-day by month climatology.
- One regularized linear model using the same available feature groups.

Deferred after implementation review:

- DLinear and PatchTST. They require an additional sequence-training framework
  and are not necessary to resolve the retrieval, event-channel, inference, or
  traceability criticisms in the current single-series study.

Optional only if resources permit:

- LightGBM.
- A larger Chronos-Bolt checkpoint.
- Probabilistic Chronos evaluation using pinball loss and interval coverage.

Not required for this revision:

- Full TimeRAF learned retriever and Channel Prompting reproduction.
- LSTM, GRU, Informer, TimesNet, iTransformer, and TimeMixer all together.
- Multiple cities, multiple counties, or nationwide validation.

## 6. Phase D: Data and Scope Clarification

Complete using the current dataset where possible:

- Report the number of active monitors contributing to the county median over
  time and acknowledge that the monitor roster changes.
- Explain that a county median suppresses localized extremes and limits the
  event hypothesis.
- Report target mean and variance separately for event, non-event, drift, and
  non-drift subsets before comparing subset MSE values.
- Avoid interpreting lower raw MSE in one subset as greater forecasting skill.
- Remove or narrow wildfire-smoke and stagnation motivation because NOAA Storm
  Events is not a complete smoke-observation source.

Deferred extensions:

- Site-level forecasting for multiple monitors.
- NOAA HMS smoke integration.
- Extension through 2025.
- Multi-station or gridded meteorology.

These are valuable future studies, but they would change the present study's
data contract and should not be mixed into the minimum corrective rerun.

## 7. Revised Execution Order

1. Archive and checksum the current verified run.
2. Implement the dense-KB configuration and leakage tests.
3. Add climatology, calendar-only retrieval, and placebo fusion controls.
4. Add event-weight and event-stratified retrieval experiments.
5. Update bootstrap inference and the diagnostic uncertainty field.
6. Add attrition, subset-variance, event-ranking-change, and drift-component
   diagnostics to the main notebook.
7. Run a small smoke test on a reduced query set.
8. Run the complete experiment once on Kaggle with a new run identifier.
9. Run the results notebook exclusively against the new frozen manifest.
10. Recompute every paper table and figure from the revised artifacts.
11. Update both LaTeX manuscripts only after the revised values are frozen.
12. Compile, inspect, and update the verification log claim by claim.

## 8. Decision Rules After the Rerun

### Outcome A: Dense retrieval produces a verified gain

- Present the original sparse-KB result as a scale limitation.
- Report where retrieval helps and whether the event channel contributes beyond
  numerical and calendar similarity.
- Retain a method-oriented title only if the complete model is supported.

### Outcome B: Retrieval remains null after the corrected experiment

- Reframe the paper as an audited negative result or evaluation protocol.
- Make the sufficiently powered null result the contribution.
- Do not describe Event-TimeRAF as superior to the context baseline.

### Outcome C: Only placebo fusion improves Chronos

- Attribute the improvement to forecast combination rather than retrieval.
- Remove retrieval-specific positive wording from the abstract and conclusion.

### Outcome D: Event-stratified retrieval helps only event origins

- Present the gain as conditional and subset-specific.
- Report subset target variance and corrected uncertainty alongside the result.

## 9. Final Acceptance Checklist

- The final notebook runs top to bottom in the documented Kaggle environment.
- Every number and figure in the paper is generated or verified by a notebook
  cell tied to one run identifier.
- The manifest includes configurations, predictions, metrics, evidence,
  checksums, package versions, and model identifiers.
- No post-manifest artifact is reported.
- All primary comparisons use the revised inference procedure.
- Event and drift claims match their subset results.
- The paper distinguishes traceability from evaluated explainability.
- No title, abstract, conclusion, or figure implies functionality that was not
  executed.
- Author information, repository URL, citations, equations, and figure labels
  are final and compile without unresolved references.

## 10. Explicitly Out of Scope for the Minimum Revision

- Full TimeRAF reproduction.
- Strict real-time event issuance without a source containing issue timestamps.
- External-city validation.
- Site-level and 2025 data expansion.
- A comprehensive benchmark of every deep forecasting architecture.
- A human-subject explainability study.

These items may strengthen a future journal extension but are not required to
correct the current experiment's most consequential weaknesses.
