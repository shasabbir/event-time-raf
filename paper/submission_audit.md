# Submission Checklist Audit

Audit date: 10 August 2026

| Requirement | Status | Where it is | What I changed |
|---|---|---|---|
| Title | Present | `main.tex` | Reframed as an audited evaluation; removed unsupported explainability and concept-drift performance wording. |
| Abstract, 200-250 words | Present | `main.tex` | Reports final M04, M09, M10, M11, climatology-fusion, dense-memory, and event-weight findings. |
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
| Ablation Study | Present | `main.tex` | Reports measured component intervals, four stride runs, event weights, event-stratified retrieval, and fusion placebos. |
| Evidence and Traceability Analysis | Present | `main.tex` | Documented signed XGBoost effects, retrieved IDs, event IDs, drift components, and the diagnostic uncertainty scale for 7,199 records. |
| Results and Discussion | Present | `sections/06_results_discussion.tex` | Interprets overall, horizon, event, drift, retrieval, uncertainty, and computational evidence without claiming uniform superiority. |
| Metric consistency | Present | Abstract, Results, Conclusion; verification notebook | Recomputed final values from 7,199 origins and 172,776 test points; all 54 manifest entries match. |
| Limitations | Present | `main.tex` | Documents retrospective event timing, single-region scope, no learned TimeRAF integration, missing hardware identifiers, and county-median aggregation. |
| Future Work | Present | `main.tex` | Prioritizes issue-time event sources, smoke evidence, site-level and second-county validation, and learned retrieval. |
| Conclusion | Present | `sections/07_conclusion.tex` | Synchronized module names, unchanged values, caveats, and supported takeaway. |
| References | Present | `references.bib` | Exactly 40 entries; 28 (70%) are from 2023-2026. Year distribution: 1997: 1, 2014: 1, 2016: 1, 2017: 2, 2020: 2, 2021: 4, 2022: 1, 2023: 9, 2024: 13, 2025: 2, 2026: 4. |
| Figures and tables | Present | `figures/`; `main.tex` | Four vector methodology figures plus five notebook-generated empirical figures. All figures and final-run tables are numbered, captioned below, and referenced. |
| External validation | Missing | Limitations and Future Work | No second city, county, or external dataset has been evaluated; this cannot be supplied without a new run. |
| Page numbers and IEEE journal format | Present | `main.tex` | Preserved the `IEEEtran` journal class and its default publication page style. |
| Writing quality and originality check | Weak | Entire manuscript | Removed classroom/process wording and tightened scientific register. The author must still verify every sentence, citation, and originality requirement before submission. |

## Reference Recency

The bibliography meets the numerical target of 40-60 entries. Of 40 entries, 28 are dated 2023-2026, so recent work forms a 70% majority. The remaining older sources are retained for established methods, datasets, metrics, and foundational forecasting literature.

## Audit Conclusion

The manuscript is internally consistent with final run
`20260810T103436161252Z` and its executed verification notebook. It reports the
dense-retrieval and control results as a bounded negative finding and remains
scoped to one county, retrospective Storm Events availability, and
non-normalized runtime.
Those are reported limitations rather than unsupported claims. Institutional
email/corresponding-author metadata, the authenticated Overleaf link, and final
human originality review remain user actions.
