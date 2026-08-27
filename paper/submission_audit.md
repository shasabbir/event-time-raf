# TRACE-RAF Submission Audit

Audit date: 27 August 2026
Evidence run: `20260827T043457543402Z`

| Requirement | Status | Where it is | Verification / action taken |
|---|---|---|---|
| Title | Present | `main.tex` | Identifies TRACE-RAF, residual analogue correction, and PM2.5 forecasting. |
| Abstract | Present | `main.tex` | States scope, mechanism, current headline metrics, adjusted inference, and limitations without superiority language. |
| Keywords | Present | `main.tex` | Uses domain, forecasting, retrieval, and auditability terms. |
| Introduction | Present | `main.tex` | Motivates the task and separates methodological novelty from empirical ranking. |
| Contribution statements | Present | Introduction | Names out-of-fold residual memory, leakage controls, trust gating, and evidence-preserving evaluation. |
| Related work | Present | `main.tex` | Organized thematically and distinguishes supervised forecasting, retrieval, TSFMs, events, and drift. |
| Critical gaps and limitations | Present | `main.tex` | Maps leakage, raw-target transfer, weak event timing, shift claims, and auditability to the method. |
| Problem formulation | Present | `main.tex` | Defines inputs, 24-step target, residuals, retrieval, and final forecast. |
| Methodology | Present | `main.tex` | Gives a full forward pass, named modules, algorithm, tensor dimensions, and implementation details. |
| Dataset description | Present | `main.tex` | Documents EPA PM2.5, NOAA GHCNh weather, Storm Events, provenance, coverage, and retrospective event timing. |
| Experimental setup | Present | `main.tex` | Reports chronological splits, model families, selection protocol, hyperparameters, seeds, and runtime scope. |
| Evaluation metrics | Present | `main.tex` | Defines MSE, MAE, RMSE, R-squared, event subsets, bootstrap, DM tests, and Holm correction. |
| Baseline comparison | Present | `main.tex` | Includes conventional supervised, neural, frozen Chronos-Bolt, and retrieval variants. |
| Ablation study | Present | `main.tex` | Includes the measured raw analogue, ungated residual, constant residual, gated residual, and event-free ladder. |
| Explainability / auditability | Present | `main.tex` | Uses evidence records and exact base-plus-gated-residual decomposition; avoids a causal explanation claim. |
| Results and discussion | Present | `main.tex` | Reports Ridge as best and TRACE as competitive; event and TSFM conclusions are explicitly scoped. |
| Limitations | Present | `main.tex` | Covers retrospective events, one county, spatial compression, statistical uncertainty, and runtime limits. |
| Future work | Present | `main.tex` | Prioritizes a second county, true issue timestamps, richer event exposure, and stronger prospective tests. |
| Conclusion | Present | `main.tex` | Matches the verified evidence and does not claim universal superiority. |
| Code availability | Present | `main.tex` | Links the public GitHub repository and identifies the executable notebooks. |
| References | Present | `references.bib` | 44 entries: 39 from 2020--2026; no arXiv/preprint entry detected. |
| Figures and tables | Present | `main.tex`, `figures/` | 9 figures and 20 tables are captioned, labeled, and referenced; method diagrams are vector PDF/SVG. |
| Equation formatting | Present | `main.tex` | 30 equation/align environments use consistent notation and labels. |
| Numeric reproducibility | Present | Notebooks 02 and 03 | 106 hashes and 43 tables verified; all headline values reconcile to the final archive. |
| AI verification log | Present | `verification_log.tex/.pdf` | One-page claim-by-claim log marks confirmed, corrected, and unverified statements. |
| Journal template | Present | `main.tex` and official template assets | Converted to the official May 13, 2026 IEEE Access `ieeeaccess` class; the 19-page paper is within the journal's recommended sub-20-page length, and the Overleaf package includes the required class, style, logo, and font files. |
| Journal declarations | Missing | N/A | Author must supply funding, competing-interest, contribution, acknowledgment, and required AI-use statements. |
| External validation | Missing | N/A | No second county has been run; keep all geographic claims local. |
| Operational event timestamps | Missing | N/A | NOAA Storm Events lacks the required issue-time field; keep the retrospective caveat. |
| Presentation sync | Pending | `presentation/` | Presenter notes are current; the legacy Event-TimeRAF PPTX must not be submitted until rebuilt against this run. |

## Cross-Artifact Consistency

- Headline data counts: 56,194 windows; 45,784/7,950/2,460 split; 59,040
  horizon-level test predictions.
- Headline model values: Ridge MSE 39.144 and TRACE MSE 40.062.
- Inference family: 29 comparisons per metric, 5,000 resamples, 168-hour blocks,
  Holm-adjusted bootstrap and DM p-values.
- TRACE mechanism: selected gate strength 0.25, mean effective gate 0.090,
  mean absolute applied residual correction 0.149.
- Evidence source: only run `20260827T043457543402Z` is authoritative.

No unsupported claim of event-specific improvement, foundation-model identity,
general concept-drift adaptation, geographic generalization, or universal model
superiority remains in `main.tex`.
