# Methodology Structure and Domain Mapping

## Structural Template Extracted from the Sample

The sample methodology opens with one framing paragraph that names the model's major components and states how they cooperate. It then gives a pipeline-overview subsection, a compact tensor-notation forward pass, and a figure-led walkthrough. The remaining methodology is divided into named modules. Each module has a motivation, an internal figure, definitions of input and output tensors, numbered equations, and prose that explicitly explains the figure. The notation follows a consistent pattern: uppercase symbols denote tensors or sets, lowercase symbols denote vectors or scalar quantities, dimensions are stated with `B` for batch size, and every figure caption begins with the module name followed by a concise description.

## Mapping to Event-TimeRAF

| Structural role in the sample | Event-TimeRAF equivalent | Domain-specific interpretation |
|---|---|---|
| Stagewise input encoder and feature preparation | Source-Audited Context Builder (SACB) | Audits and aligns 1-D PM2.5, weather, calendar, and event records into an 85-dimensional origin vector and chronological windows. |
| Adaptive context extraction and hierarchical evidence refinement | Leakage-Safe Event-Context Retriever (LSER) | Applies temporal eligibility before multi-channel similarity ranking, then returns eight historical 24-hour futures and a 51-dimensional retrieval summary. |
| Cross-scale fusion and uncertainty-aware dual output | Drift-Aware Forecast and Evidence Head (DFEH) | Produces verified direct-XGBoost and frozen-TSFM forecast paths plus source-grounded evidence records. The later drift gate is excluded pending a complete rerun. |

This mapping copies only the sample's organizational depth. It does not transfer image convolutions, spatial attention, tumor guidance, or classification heads into the time-series study.

## Preservation Record

The rewrite retains the implemented Event-TimeRAF functional mapping,
knowledge-base definition, temporal eligibility rule, retrieved-set definition,
hybrid similarity score, top-k rule, Chronos forecast fusion, and drift-score
logic in their original relative order. Numerical claims are restricted to the
immutable manifest-backed run. The 85/151 feature dimensions were corrected
from saved feature names, and the post-manifest M12 selector was removed.
