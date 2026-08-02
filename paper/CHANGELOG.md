# Manuscript Upgrade Changelog

## Manuscript

- `main.tex`: rewrote the abstract around SACB, LSER, and DFEH; retained every reported value; marked M12 post hoc; enabled page numbers.
- `sections/01_introduction.tex`: clarified motivation, objectives, and mechanism-based contributions.
- `sections/02_related_work.tex`: replaced list-like summaries with thematic, comparative coverage of air-quality forecasting, foundation models, retrieval, event context, drift, and explainability.
- `sections/03_critical_gaps_contributions.tex`: separated mandatory critical gaps from core contributions and made the mapping explicit.
- `sections/03_problem_formulation.tex`: standardized notation, added tensor definitions, labels, and equation references.
- `sections/04_methodology.tex`: rebuilt the section with an end-to-end tensor forward pass, detailed SACB/LSER/DFEH modules, figure walkthroughs, and implementation-level equations. Existing methodology equations were retained in their original relative order.
- `sections/05_experimental_setup.tex`: expanded dataset audit, availability, model catalog, hyperparameters, metrics, statistical checks, and methodological comparison.
- `sections/06_results_discussion.tex`: preserved all result and ablation values; identified M04 as the strongest pre-specified model; recast M12 as exploratory; expanded subset, explainability, diagnostic, and reproducibility interpretation.
- `sections/07_limitations_future_work.tex`: separated limitations and future work and added the event-timing, external-validation, model-scope, hardware, and M12 caveats.
- `sections/07_conclusion.tex`: synchronized the conclusion with the supported results and new module terminology.
- `references.bib`: expanded and normalized the bibliography to 40 entries, including recent forecasting, TSFM, drift, retrieval, and explainability work.

## Figures

- Added `figures/event_timeraf_pipeline_overview.svg` and `.pdf`.
- Added `figures/source_audited_context_builder.svg` and `.pdf`.
- Added `figures/leakage_safe_event_context_retriever.svg` and `.pdf`.
- Added `figures/drift_aware_forecast_evidence_head.svg` and `.pdf`.
- Added `tools/build_methodology_figures.py`, a reusable SVG generator with vector PDF export.
- Kept figure names and captions below each diagram in LaTeX; no figure number or caption is embedded inside the artwork.

## Integrity Corrections

- Corrected the archived M12 characterization after inspecting `m12_gate_selection_posthoc.json`; its values were not changed.
- Removed any implication that Event-TimeRAF beats every baseline or reproduces TimeRAF's learned retriever and Channel Prompting.
- Kept NOAA Storm Events findings explicitly retrospective because publication timestamps are unavailable.
- Did not modify datasets, notebooks, source-model code, result tables, ablation values, or artifact identifiers as part of this paper upgrade.

