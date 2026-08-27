# Methodology Structure and Domain Mapping

## Structural Template

The methodology opens with a framing paragraph naming the four implemented
components, followed by an end-to-end tensor forward pass. Each component then
receives a motivated subsection, an internal figure, input/output definitions,
numbered equations, and a prose walkthrough. Uppercase symbols denote tensors
or sets, lowercase symbols denote vectors or scalars, and `B` denotes batch
size.

## TRACE-RAF Domain Mapping

| Structural role | TRACE-RAF component | Implemented function |
|---|---|---|
| Source alignment and feature construction | Source-Audited Context Builder (SACB) | Aligns PM2.5, weather, calendar, and retrospective event records into 168-hour windows and an 85-variable origin context while retaining provenance and readiness checks. |
| Historical evidence selection | Leakage-Safe Event-Context Retriever (LSER) | Applies the candidate-target embargo before transparent PM2.5, weather, calendar, and event similarity ranking. |
| Selective analogue correction | Trust-gated Residual Analog Correction (TRACE) | Builds an out-of-fold residual memory, retrieves residual trajectories, and applies them through a validation-trained reliability gate. |
| Forecast output and evidence | Drift-Aware Forecast and Evidence Head (DFEH) | Evaluates supervised, TRACE, and frozen-Chronos paths while storing distribution-shift diagnostics and prediction-level evidence. |

The mapping transfers only the sample paper's organizational depth. It does not
import image-domain operations, learned TimeRAF Channel Prompting, or any
unimplemented neural retriever.

## Preservation Record

The current methodology preserves the implemented knowledge-base construction,
temporal eligibility rule, hybrid score, top-k retrieval, out-of-fold residual
construction, trust gate, Chronos output-fusion control, and distribution-shift
diagnostic in their code order. Dimensions and numerical claims are tied only
to run `20260827T043457543402Z`. M12 is retained as a separately
validation-selected drift-router baseline, not as the proposed model.
