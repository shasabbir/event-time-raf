# TRACE-RAF Faculty Submission

## Authoritative Evidence

- Run ID: `20260827T043457543402Z`
- Immutable evidence archive:
  `event_timeraf_publication_candidate_20260827T043457543402Z.zip`
- Repository: <https://github.com/shasabbir/event-time-raf>
- Paper source: `paper/main.tex`
- Paper PDF: `paper/main.pdf`
- Journal format: official May 13, 2026 IEEE Access LaTeX template
- AI verification log: `paper/verification_log.pdf`
- Overleaf package: `event_timeraf_verified_overleaf.zip`

`paper/main_humanized.tex` is not part of this synchronized submission.

## Reproducible Notebook Order

1. `notebooks/01_event_timeraf_kaggle_pipeline.ipynb` rebuilds the experiment
   from the attached official datasets.
2. `notebooks/02_results_and_figures.ipynb` verifies all manifest hashes,
   recomputes metrics and statistical tests, checks the TRACE decomposition,
   and regenerates publication figures. This is the primary live-verification
   notebook.
3. `notebooks/03_paper_claim_verification.ipynb` performs a final cross-table
   run-ID and completeness audit and displays the claim-supporting tables.

Notebook 02 checks all 106 manifest-listed artifacts by SHA-256, reconciles all
43 result tables, verifies 56,194 windows and their 45,784/7,950/2,460
chronological split, reconstructs 59,040 test forecast points, and checks the
exact TRACE forecast decomposition.

## Results That May Be Reported

The strongest tested model is C01 Ridge:

- MSE 39.144
- MAE 4.079
- RMSE 6.257
- R-squared 0.457

M13 TRACE records MSE 40.062, MAE 4.134, RMSE 6.329, and R-squared 0.444. It
improves nominally over the C06 base ensemble but the difference is not
significant after Holm correction across the 29-comparison family.

The mechanism ladder is fully measured:

- A04 raw-target analogue fusion selects weight 0.
- A05 ungated residual transfer increases MSE to 44.959.
- A06 constant residual transfer selects strength 0.
- M13 selects trust-gate strength 0.25 and limits the mean absolute applied
  correction to 0.149.

The event-free comparison is nearly identical to TRACE and only 70 test origins
have event context. Event awareness must therefore be described as investigated,
not demonstrated. Chronos-Bolt is a separate frozen TSFM baseline; the negative
result applies only to the tested event-conditioned output-space fusion.

## Claims That Must Not Be Made

- TRACE-RAF is not a foundation model.
- TRACE does not beat all baselines; Ridge is better on every headline metric.
- The TRACE-versus-base gain is not multiplicity-adjusted significant.
- The experiment does not establish an event-specific forecasting benefit.
- NOAA Storm Events does not provide strict machine-readable issue timestamps.
- The distribution-shift score is not proof of general concept-drift adaptation.
- One Los Angeles County aggregate does not establish geographic generalization.
- Runtime is not hardware-normalized.

## Submission Check

Before submitting, confirm that the Overleaf PDF matches `paper/main.pdf`, add
the final Overleaf link, and ensure the presentation uses the run ID and values
above. Do not include an older Event-TimeRAF deck or any July evidence archive.
