# Submission Checklist Audit

Audit date: 5 August 2026

| Requirement | Status | Where it is | What I changed |
|---|---|---|---|
| Title | Present | `main.tex` | Retained the method-forward Event-TimeRAF title. |
| Abstract, 200-250 words | Present | `main.tex` | Names SACB, LSER, and DFEH and reports only manifest-backed M04, M09, M10, and M11 values. |
| Introduction with contribution bullets | Present | `sections/01_introduction.tex` | Added task motivation, scoped objectives, and one mechanism-focused bullet for each named module. |
| Related Work, thematic and comparative | Present | `sections/02_related_work.tex` | Reorganized into thematic subsections and added comparative discussion instead of paper-by-paper summaries. |
| Critical Gaps and Limitations of Previous Studies | Present | `sections/03_critical_gaps_contributions.tex` | Added explicit provenance, temporal leakage, event-context, drift, and evidence gaps. |
| Core Contributions | Present | `sections/03_critical_gaps_contributions.tex` | Added a separate contribution section with one-to-one responses to the stated gaps. |
| Problem formulation | Present | `sections/03_problem_formulation.tex` | Standardized symbols, tensor meanings, chronological split, and explicit equation references. |
| Methodology and pipeline overview | Present | `sections/04_methodology.tex` | Rebuilt around a compact tensor forward pass and three named, code-faithful modules. |
| Custom model design | Weak | `sections/04_methodology.tex` | SACB, LSER, and DFEH form a custom integrated framework, but the forecasters are XGBoost and frozen Chronos-Bolt; there is no learned retriever or internal TimeRAF Channel Prompting. |
| Mathematical formulation | Present | `sections/03_problem_formulation.tex`; `sections/04_methodology.tex` | Preserved prior equations, added tensor shapes and implementation equations, numbered them, and referenced each in prose. |
| Dataset Description | Present | `sections/05_experimental_setup.tex` | Added official sources, study period, station/target construction, counts, coverage, split, preprocessing, and event-availability caveat. |
| Dataset links | Present | `sections/05_experimental_setup.tex`; `references.bib` | Linked official EPA, NOAA Global Hourly, and NOAA Storm Events source pages. |
| Code Availability | Present | `sections/05_experimental_setup.tex` | Added `https://github.com/shasabbir/event-time-raf`, language, package versions, and repository contents. |
| Experimental Setup | Weak | `sections/05_experimental_setup.tex` | Added a complete model and hyperparameter table. Exact CPU, GPU, RAM, and peak-memory identifiers were not captured by the run. |
| Evaluation Metrics | Present | `sections/05_experimental_setup.tex` | Added justified MSE, MAE, RMSE, and R-squared equations plus subset and bootstrap rules. |
| Comparison with Existing Studies | Weak | `sections/04_methodology.tex`; `sections/05_experimental_setup.tex` | Added methodological comparison tables. A numerical cross-paper ranking is intentionally omitted because datasets and protocols differ. |
| Ablation Study | Present | `sections/06_results_discussion.tex` | Retained every measured value and mapped comparisons to SACB, LSER, and DFEH. Results are honestly non-monotonic. |
| Explainability Analysis | Present | `sections/06_results_discussion.tex` | Documented signed XGBoost effects, retrieved IDs, event IDs, drift components, and the uncertainty proxy for 7,199 records. |
| Results and Discussion | Present | `sections/06_results_discussion.tex` | Interprets overall, horizon, event, drift, retrieval, uncertainty, and computational evidence without claiming uniform superiority. |
| Metric consistency | Present | Abstract, Results, Conclusion; verification notebook | Recomputed all values from 7,199 test origins. Corrected 85/151 dimensions and removed unsupported M12 values. |
| Limitations | Present | `sections/07_limitations_future_work.tex` | Documents retrospective event timing, single-region scope, no learned TimeRAF integration, excluded M12 lineage, missing hardware identifiers, and NumPy-sensitive random replay. |
| Future Work | Present | `sections/07_limitations_future_work.tex` | Specifies a complete drift-gate rerun, second-region validation, issue-time event sources, and learned retrieval as future extensions. |
| Conclusion | Present | `sections/07_conclusion.tex` | Synchronized module names, unchanged values, caveats, and supported takeaway. |
| References | Present | `references.bib` | Exactly 40 entries; 28 (70%) are from 2023-2026. Year distribution: 1997: 1, 2014: 1, 2016: 1, 2017: 2, 2020: 2, 2021: 4, 2022: 1, 2023: 9, 2024: 13, 2025: 2, 2026: 4. |
| Figures and tables | Present | `figures/`; all manuscript sections | Four vector methodology figures plus five notebook-generated empirical figures. All 9 figures and 13 tables are numbered, captioned below, and referenced in body text. |
| External validation | Missing | Limitations and Future Work | No second city, county, or external dataset has been evaluated; this cannot be supplied without a new run. |
| Page numbers and IEEE journal format | Present | `main.tex` | Preserved `IEEEtran` journal class and enabled plain page numbering. |
| Writing quality and originality check | Weak | Entire manuscript | Removed classroom/process wording and tightened scientific register. The author must still verify every sentence, citation, and originality requirement before submission. |

## Reference Recency

The bibliography meets the numerical target of 40-60 entries. Of 40 entries, 28 are dated 2023-2026, so recent work forms a 70% majority. The remaining older sources are retained for established methods, datasets, metrics, and foundational forecasting literature.

## Audit Conclusion

The manuscript is internally consistent with the executed verification notebook
and contains no post-manifest M12 result. It remains intentionally scoped to one
county, retrospective Storm Events availability, and non-normalized runtime.
Those are reported limitations rather than unsupported claims. Author metadata,
the authenticated Overleaf link, and final human originality review remain user
actions.
