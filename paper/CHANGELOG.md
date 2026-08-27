# Paper Changelog

## 27 August 2026 - Final Run Synchronization

- Set run `20260827T043457543402Z` as the sole numerical source of truth.
- Reconciled `main.tex` with 106 manifest-listed artifacts, 43 result tables,
  56,194 windows, and 59,040 test forecast points.
- Updated every headline metric to the executed results: Ridge MSE 39.144 and
  TRACE MSE 40.062, with Ridge identified as the strongest tested model.
- Added the complete mechanism ladder: base ensemble, raw-target analogue
  fusion, ungated residual transfer, constant residual transfer, and gated
  residual TRACE.
- Added exact forecast decomposition results: selected gate strength 0.25,
  mean effective gate 0.090, raw mean absolute residual 1.708, and applied mean
  absolute correction 0.149.
- Corrected statistical language to distinguish unadjusted bootstrap intervals
  from Holm-adjusted bootstrap and Diebold-Mariano p-values across the
  29-comparison family.
- Reframed the event channel as an investigated retrospective component. The
  event-free ablation is nearly identical and only 70 test origins contain
  event context.
- Clarified that TRACE-RAF is a supervised residual-correction model. The
  Chronos-Bolt result is limited to one frozen TSFM and one event-conditioned
  output-space fusion configuration.
- Replaced general concept-drift terminology with distribution-shift diagnostic
  or model-specific regime label where warranted.
- Replaced causal explainability language with auditable evidence and exact
  forecast decomposition.
- Updated the one-page AI verification log, faculty README, submission audit,
  and presenter notes to the current run.
- Added the executed LSER event-weight sensitivity table and explicitly marked
  the primary coefficient as predeclared rather than optimal.
- Added DLinear, LSTM, PatchTST, optimizer, seed, and selected-epoch details to
  the reproducibility table.
- Reduced the abstract from 318 to 248 words without changing reported results.
- Kept `main_humanized.tex` unchanged by explicit instruction.

## Remaining Submission Actions

- Converted the manuscript from generic `IEEEtran` formatting to the official
  May 13, 2026 IEEE Access `ieeeaccess` class and submission assets.
- Rebuild the legacy Event-TimeRAF presentation from the synchronized notes.
- Add an external county only if a new experiment is completed; do not infer
  geographic generalization from the Los Angeles result.
- Add operational event evidence only if an archived source with true issue
  timestamps is obtained.
