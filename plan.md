# Event-TimeRAF Master Plan

Project title:

**Event-TimeRAF: Event-Aware Retrieval-Augmented Foundation Models for Explainable Air Quality Forecasting Under Concept Drift**

Base paper:

**TimeRAF: Retrieval-Augmented Foundation Model for Zero-Shot Time Series Forecasting**

This document is the corrected high-level plan for the project. It is scoped for a simple Kaggle-friendly implementation first, with advanced TimeRAF-style components added only after a working baseline pipeline exists.

Project status: **implementation scaffold complete; full 2019--2025 Kaggle run and
final result generation pending**.

`structured_plan.md` is the binding implementation contract. It resolves detailed decisions about target shape, data-readiness gates, artifact schemas, leakage controls, model identifiers, the foundation-model publication gate, and phase exit criteria. If the two documents differ on an implementation detail, follow `structured_plan.md` and update this overview afterward.

---

## 1. Core Research Idea

The project extends the TimeRAF idea from numeric time-series retrieval to event-aware air-quality forecasting.

TimeRAF retrieves useful historical time-series candidates from a knowledge base and injects retrieved knowledge into a frozen time-series foundation model. Event-TimeRAF keeps that central retrieval idea but adapts it to urban air quality, where PM2.5 changes are often influenced by weather, holidays, fires, traffic, construction, public events, and other external context.

The target application is:

```text
Forecast future PM2.5 for Los Angeles County using:
- historical PM2.5 time series
- weather variables
- calendar features
- retrieved similar historical windows
- retrieved event/news context
- concept drift indicators
- evidence-based explanations
```

The first implementation must be simple and runnable on Kaggle. The goal is a complete, defensible research pipeline, not a full reproduction of TimeRAF.

---

## 2. Research Objective

Given historical air-quality observations, weather covariates, calendar features, and event context, predict future PM2.5 values for a forecast horizon.

Formal problem:

```text
X_t = [x_{t-L+1}, ..., x_t]
W_t = weather covariates aligned with X_t
C_t = calendar/context features aligned with X_t
E_t = retrieved event documents or structured event features near time t

Goal:
Y_hat_{t+1:t+H} = F(X_t, W_t, C_t, E_t)
```

Symbols:

| Symbol | Meaning |
| --- | --- |
| `L` | Lookback window |
| `H` | Forecast horizon |
| `X_t` | Historical PM2.5 / air-quality sequence |
| `W_t` | Weather features |
| `C_t` | Calendar features |
| `E_t` | Event/news context |
| `Y_hat` | Predicted future PM2.5 |

Initial practical setting:

```text
lookback L = 168 hours
forecast horizon H = 24 hours
target = the full PM2.5 sequence from t+1 through t+24
location = Los Angeles County, California, USA
study period = 2019-01-01 through 2025-12-31, subject to the data audit
validation = 2024-08-25 through 2025-08-24
selection-independent final test = 2025-08-25 onward
January 2025 = validation-only development-stress analysis, not final-holdout evidence
frequency = hourly if data quality allows
```

Later extensions:

```text
H = 48, 72, 96 hours
multiple stations if available
advanced retriever
advanced neural fusion
```

---

## 3. Key Research Gap

TimeRAF focuses mainly on retrieving numerical time-series patterns. That is useful, but air quality is not driven by time-series history alone.

For Los Angeles County PM2.5 forecasting, abnormal changes may be caused by:

```text
rainfall
low wind speed
high humidity
traffic disruption
public holidays
fire incidents
construction and road dust
industrial activity
large public gatherings
policy restrictions
seasonal behavior
```

Event-TimeRAF addresses this gap by combining:

```text
time-series retrieval
+ weather/calendar context
+ event/news retrieval
+ drift detection
+ grounded explanation generation
```

---

## 4. Base Paper Details to Preserve

The base TimeRAF paper adds several details that should shape this project.

TimeRAF core mechanics:

```text
- channel-independent forecasting formulation
- sliding-window time-series knowledge base
- consistent Z-score normalization for input and retrieved sequences
- dual-encoder retriever inspired by DPR
- dot-product similarity between query and candidate embeddings
- retriever supervision using forecaster feedback
- frozen TSFM backbone during training
- Channel Prompting for retrieved-knowledge integration
- uniform averaging across retrieved candidates after MLP extraction
- candidate augmentation during retriever training to avoid narrow retrieval behavior
```

TimeRAF experimental details to mention accurately:

```text
- TTM-Base is the main backbone.
- Input context length is 512 and forecast horizon is 96 in most experiments.
- The default number of retrieved candidates is k = 8.
- MSE is the main metric in the base paper.
- Evaluation datasets include ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Electricity.
- Baselines include TTM, Moirai, MOMENT, Timer, Chronos, and TimesFM.
```

Limitations and extension opportunities from the base paper:

```text
- knowledge bases are built from original time-series data without trend-seasonal decomposition
- knowledge integration is still somewhat heuristic
- channel interdependencies are not deeply modeled
- retrieval is based on time-series data, not event/news context
```

Event-TimeRAF should use these points to position the contribution:

```text
TimeRAF retrieves time-series candidates.
Event-TimeRAF retrieves time-series candidates plus event/weather/calendar context.
TimeRAF uses Channel Prompting for TSFM embeddings.
Event-TimeRAF MVP uses lightweight retrieval-feature fusion.
A frozen TSFM plus retrieval-augmentation experiment is required before the final paper retains "Foundation Models" in its title.
Full TimeRAF-style Channel Prompting remains an advanced extension, not the first Kaggle implementation.
```

---

## 5. Expected Contributions

The paper should claim contributions carefully and only at methodology level until experiments are completed.

Planned contributions:

1. An event-aware retrieval-augmented forecasting framework for Los Angeles County PM2.5 prediction.
2. A hybrid knowledge base combining historical PM2.5 windows, weather summaries, calendar context, and event records.
3. A simple TimeRAF-inspired retrieval baseline for air-quality forecasting.
4. A concept-drift-aware extension using rolling statistics and retrieval similarity indicators.
5. A grounded explanation module based on retrieved cases, event context, weather conditions, and feature importance.
6. An ablation plan comparing basic forecasting, random retrieval, cosine retrieval, event retrieval, drift-aware retrieval, and the full Event-TimeRAF pipeline.

Do not claim state-of-the-art performance unless real experiments prove it.

---

## 6. Paper Scope for First Draft

First paper deliverable:

```text
Title
Abstract draft
Introduction
Related Work
Problem Formulation
Proposed Methodology
Experimental Setup plan
Results placeholder
Conclusion placeholder
References placeholder
```

Do not write fake results.

The results section must say:

```text
Experimental results will be reported after model implementation and evaluation.
```

The first paper draft should be stored later as:

```text
paper/
  main.tex
  references.bib
  sections/
    01_title_abstract.tex
    02_introduction.tex
    03_related_work.tex
    04_problem_statement.tex
    05_methodology.tex
    06_experimental_setup_placeholder.tex
    07_results_placeholder.tex
    08_conclusion_placeholder.tex
```

---

## 7. Paper Writing Requirements

Use IEEE-style academic writing.

Rules:

```text
- Avoid exaggerated claims.
- Do not overuse "novel".
- Use "proposed framework" instead of marketing language.
- Do not claim SOTA without real results.
- Use citation placeholders where references are not finalized.
- Clearly explain TimeRAF as the base method.
- Clearly state how Event-TimeRAF differs from TimeRAF.
- Mention TimeRAF's retriever, Channel Prompting, frozen TSFM backbone, default `k = 8`, context length 512, forecast horizon 96, and MSE evaluation.
- Keep Results, Discussion, and Conclusion as placeholders until experiments are complete.
```

Introduction structure:

1. Importance of urban air-quality forecasting.
2. Limitations of traditional and deep forecasting models during abnormal events.
3. Time-series foundation models and TimeRAF-style retrieval augmentation.
4. Research gap: numeric retrieval alone misses external event context.
5. Contributions of Event-TimeRAF.

Related Work subsections:

```text
2.1 Time-Series Forecasting
2.2 Time-Series Foundation Models
2.3 Retrieval-Augmented Forecasting
2.4 Air-Quality Forecasting
2.5 Event-Aware and Explainable Forecasting
```

Methodology subsections:

```text
4.1 Overview
4.2 Dataset Construction
4.3 Time-Series Knowledge Base
4.4 Event Knowledge Base
4.5 Time-Series Retriever
4.6 Event Retriever
4.7 Hybrid Fusion Module
4.8 Forecasting Backbone
4.9 Concept Drift Detection
4.10 Explanation Generation
```

---

## 8. Kaggle-Friendly Repository Scope

The original plan had a large repository tree. For Kaggle, keep the implementation simple.

Use this lightweight structure later:

```text
ASA/
  plan.md
  structured_plan.md
  README.md
  requirements.txt
  configs/
    default.yaml
  notebooks/
    01_event_timeraf_kaggle_pipeline.ipynb
    02_results_and_figures.ipynb
  src/
    event_timeraf/
      __init__.py
      config.py
      data.py
      features.py
      windows.py
      retrieval.py
      models.py
      drift.py
      explain.py
      evaluation.py
      plots.py
  tests/
    test_windows.py
    test_retrieval.py
    test_evaluation.py
  data/
    raw/
    processed/
    knowledge_base/
  outputs/
    audit/
    tables/
    figures/
    predictions/
    evidence/
    models/
    logs/
  paper/
    main.tex
    references.bib
    sections/
```

Why this structure:

```text
- one main Kaggle notebook for end-to-end execution
- small src package for reusable functions
- focused tests for target alignment, retrieval eligibility, and metrics
- one config file first
- simple outputs folder for results
- paper folder separated from experiments
```

Avoid creating many scripts and config files until the MVP works.

---

## 9. Data Plan

### 9.1 Air Quality

Primary target:

```text
US EPA AirData/AQS hourly parameter 88101
Location: Los Angeles County, California, USA
State/county codes: 06/037
Study period: 2019-01-01 through 2025-12-31
Frequency: hourly if available
Target: PM2.5
```

Processing requirements:

```text
- parse timestamps
- convert to America/Los_Angeles with daylight-saving handling
- remove impossible PM2.5 values
- resample to hourly frequency
- fill only short input gaps using past-available information
- never interpolate target values; discard windows with incomplete targets
- preserve missingness report
- fit normalization statistics on training data only
- select the AQS site with the strongest continuous coverage during the audit;
  if no single site passes the PM2.5 coverage gate, use the formally documented
  Los Angeles County hourly median across official AQS monitors
- save processed data as data/processed/la_pm25_hourly.parquet
```

### 9.2 Weather

Preferred source:

```text
NOAA NCEI Global Hourly/ISD through the official NOAA NODD public buckets first
Open-Meteo optional fallback
```

Weather variables:

```text
temperature_2m
relative_humidity_2m
precipitation
wind_speed_10m
wind_direction_10m
surface_pressure
cloud_cover
visibility if available
```

### 9.3 Calendar

Generate:

```text
hour
day_of_week
is_weekend
month
season
is_us_federal_holiday
is_california_holiday
```

Use the Saturday/Sunday weekend convention and verified US federal and California holiday dates.

### 9.4 Events

Use NOAA Storm Events as the required structured event source. NOAA HMS historical fire/smoke records are an optional enrichment.

NOAA Storm Events does not expose a machine-readable publication timestamp. The
MVP records `published_at = event_start` together with
`availability_assumption = event_start`; all resulting event-aware comparisons
must be labeled retrospective sensitivity experiments. Strict operational event
claims require a source-preserving cache with genuine publication or issuance
timestamps.

MVP structured event features:

```text
event_count_24h
event_count_72h
wildfire_event_count_72h
smoke_event_count_72h
traffic_event_count_72h
heavy_rain_event_count_72h
industrial_event_count_7d
top_event_text
```

If event collection becomes unstable, the fallback is a privately attached
cache of unchanged NOAA annual archives prepared from the official NCEI index.
The preparation notebook records official URLs, retrieval time, file sizes, and
SHA-256 hashes. The main pipeline verifies this manifest before loading any
event. Third-party combined or transformed datasets are not acceptable for the
final study. This is a delivery fallback only: records stay attributed to NOAA
Storm Events, and the audit records the cache path and delivery mode. Events
must never be manually invented.

---

## 10. Feature Engineering

TimeRAF-specific preprocessing lessons:

```text
- build input-output pairs using sliding windows
- keep input window length and retrieved candidate window length aligned
- normalize input windows and retrieved windows consistently
- fit scalers only on the training split to avoid leakage
- store metadata for window start/end time, station, frequency, and split
```

Lag features:

```text
pm25_lag_1h
pm25_lag_3h
pm25_lag_6h
pm25_lag_12h
pm25_lag_24h
pm25_lag_48h
pm25_lag_168h
```

Rolling features:

```text
pm25_roll_mean_3h
pm25_roll_mean_6h
pm25_roll_mean_24h
pm25_roll_mean_168h
pm25_roll_std_24h
pm25_roll_max_24h
pm25_roll_min_24h
```

Weather interaction features:

```text
wind_low_flag
rain_flag
humidity_high_flag
stagnation_index
rain_washout_indicator
```

Event features:

```text
event_count_24h
event_count_72h
wildfire_event_count_72h
smoke_event_count_72h
traffic_event_count_72h
industrial_event_count_7d
holiday_event_flag
```

---

## 11. Model Plan

### 11.1 Baselines

Implement these before Event-TimeRAF:

```text
1. Persistence baseline
2. Daily and weekly seasonal naive baselines
3. Direct XGBoost baseline using one regressor per forecast horizon
4. Optional ARIMA/SARIMA
5. Optional LSTM/GRU if time permits
```

For Kaggle, XGBoost is the locked primary learned model and LightGBM is a fallback. Every model must output the complete 24-hour sequence.

### 11.2 TimeRAF-Inspired Retrieval Baseline

Simplified retrieval baseline:

```text
current PM2.5 window
-> retrieve top-k similar historical windows
-> aggregate retrieved future trajectories
-> add retrieved summary features
-> forecast with the direct XGBoost models
```

Initial settings:

```text
window length = 168 hours
forecast horizon = 24 hours
k = 1, 4, 8, 16, optionally 32
retrieval = cosine similarity or FAISS nearest neighbor
default TimeRAF-inspired setting = k = 8
```

Do not attempt full TimeRAF reproduction first.

Base-paper faithfulness:

```text
- use random retrieval and cosine retrieval as explicit baselines
- aggregate retrieved candidates with uniform weighting first
- test score-weighted aggregation only as an ablation
- optionally add a dual-encoder MLP retriever after the MVP works
- optionally add candidate augmentation for the learnable retriever
```

### 11.3 Event-TimeRAF MVP

MVP architecture:

```text
PM2.5 lag/rolling features
+ weather features
+ calendar features
+ retrieved time-series summary features
+ event count/tone/category features
+ drift indicators
-> direct XGBoost forecast models
-> explanation template
```

Advanced architecture after MVP:

```text
window encoder
+ weather/calendar encoder
+ event embedding
+ retrieved sequence embedding
-> Channel-Prompting-inspired fusion
-> forecast head
```

Channel-Prompting-inspired fusion should follow the base paper conceptually:

```text
input representation
+ retrieved candidate representation
-> concatenate/flatten or concatenate features
-> MLP compression
-> residual connection to preserve input information
-> average uniformly across k retrieved candidates
```

### 11.4 Foundation-Model Publication Gate

The classical MVP can be completed first, but the current paper title requires a real frozen TSFM experiment before submission. Prefer a small TTM checkpoint because the base paper uses TTM-Base, with a small Chronos-family checkpoint as a compatibility fallback. The selected checkpoint must support the same `L=168`, `H=24` task; the evaluation window must not be changed merely to fit a checkpoint.

Required comparison:

```text
frozen zero-shot TSFM
vs frozen TSFM plus a validation-weighted retrieved historical forecast
vs frozen TSFM plus event-conditioned hybrid retrieval
```

This lightweight forecast-level fusion must be described accurately and must not be called TimeRAF Channel Prompting. If the TSFM gate cannot be completed, remove "Foundation Models" from the title and all corresponding claims as specified in `structured_plan.md`.

---

## 12. Retrieval Plan

Time-series knowledge base:

```text
data/knowledge_base/ts_windows.parquet
```

Fields:

```text
window_id
start_time
end_time
input_window_values
future_window_values
weather_summary
calendar_summary
event_summary_id
embedding_vector
split
normalization_id
```

Retrieval methods:

```text
1. Random retrieval
2. Cosine similarity retrieval
3. FAISS nearest-neighbor retrieval if available
4. Optional dual-encoder MLP retriever
```

Leakage controls:

```text
- validation/test queries may retrieve only from training-history windows
- every candidate's complete future window must end before the query lookback begins
- candidate future windows must never overlap the forecast target being evaluated
- self-matches and query/candidate overlap are excluded by requiring each
  candidate target to end before the query lookback begins
- candidate-to-candidate overlap is allowed; evaluate 1-, 6-, and 24-hour
  knowledge-base strides explicitly
- retrieved windows should have the same lookback length as the query
- record selected candidate IDs and use stride sensitivity to quantify redundancy
- events are usable only when published_at is no later than the forecast origin
```

Event knowledge base:

```text
data/knowledge_base/event_kb.parquet
```

Fields:

```text
event_id
event_datetime
location
event_type
source
title
summary
tone
keywords
embedding_vector optional
```

Hybrid retrieval score:

```text
score = alpha * time_series_similarity
      + beta  * weather_similarity
      + gamma * calendar_similarity
      + delta * event_relevance
```

Initial weights:

```text
alpha = 0.5
beta = 0.2
gamma = 0.1
delta = 0.2
```

Tune only after validation results exist.

---

## 13. Concept Drift Plan

The first implementation treats drift as an operational distribution-shift indicator, not as proof that true concept drift has been identified. Robustness claims require separate error analysis on transparently flagged periods.

MVP drift indicators:

```text
recent_mean_shift
recent_variance_shift
retrieval_similarity_drop
weather_pattern_shift
event_burst_indicator
```

Simple methods:

```text
rolling mean/std change
z-score anomaly flag
retrieval similarity threshold
```

Outputs:

```text
drift_flag
drift_score
```

Advanced methods such as ADWIN, Page-Hinkley, KS test, or MMD can be added later if the MVP is stable.

---

## 14. Explanation Plan

Explanations must be evidence-based.

Use:

```text
retrieved similar historical windows
retrieved events
weather conditions
calendar context
SHAP or model feature importance
drift score
forecast trend direction
```

Do not generate explanations from an LLM alone.

Example explanation format:

```text
The model predicts a PM2.5 increase over the next 24 hours. The main evidence is high recent PM2.5 persistence, low wind speed, and no rainfall. Retrieved historical windows with similar stagnant weather showed increasing PM2.5. Recent event retrieval found a source-recorded wildfire or smoke event affecting Los Angeles County. The drift score is moderate, so the forecast should be interpreted with caution.
```

---

## 15. Evaluation Plan

Metrics:

```text
MSE
MAE
RMSE
MAPE
sMAPE
R2
```

Validation MSE is the locked model-selection metric because it matches the TimeRAF base paper. MAE/RMSE/sMAPE remain important for interpretation in air-quality forecasting, and results must also be reported separately for all 24 horizons.

Special analysis:

```text
event-period MAE/RMSE
drift-period MAE/RMSE
normal-period MAE/RMSE
```

Split:

```text
Train: earliest 70%
Validation: next 15%
Test: latest 15%
```

Never use random splitting for the main time-series evaluation.

Backtesting:

```text
rolling-origin evaluation after the first stable baseline is complete
```

TimeRAF-aligned analysis:

```text
- compare random retrieval vs cosine retrieval vs proposed retrieval
- test candidate counts k = 1, 4, 8, 16, optionally 32
- test knowledge-base size sensitivity if runtime allows
- report computational cost or inference time if feasible
- use mean/std over repeated runs for stochastic models if runtime allows
```

---

## 16. Ablation Study

Core ablation table:

| Model | PM2.5 | Weather | Calendar | TS Retrieval | Event Retrieval | Drift | Explanation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | Yes | No | No | No | No | No | No |
| Seasonal naive | Yes | No | Yes | No | No | No | No |
| XGBoost-basic | Yes | No | No | No | No | No | No |
| XGBoost-weather-calendar | Yes | Yes | Yes | No | No | No | No |
| Random-retrieval baseline | Yes | Yes | Yes | Random | No | No | Limited |
| Cosine TimeRAF-inspired | Yes | Yes | Yes | Cosine | No | No | Limited |
| Event-TimeRAF-no-drift | Yes | Yes | Yes | Yes | Yes | No | Yes |
| Event-TimeRAF-full | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Frozen TSFM | Yes | No | No | No | No | No | No |
| Frozen TSFM + hybrid retrieval | Yes | Yes | Yes | Yes | Yes | No | Limited |

This ablation is the main experimental proof of the project.

Additional TimeRAF-derived ablations:

```text
- k sensitivity: 1, 4, 8, 16, optionally 32
- retrieval aggregation: uniform vs similarity-weighted
- knowledge base: train-only domain-specific vs expanded external/multi-station if available
- fusion: simple concatenation vs MLP residual fusion
```

---

## 17. Figures and Tables

Planned figures:

```text
Figure 1: Event-TimeRAF architecture
Figure 2: Time-series and event knowledge-base construction
Figure 3: Hybrid retrieval process
Figure 4: Forecast visualization
Figure 5: Error comparison across models
Figure 6: Ablation study
Figure 7: Explanation case study
Figure 8: Retrieved historical case study
```

Planned tables:

```text
Table 1: Comparison with TimeRAF
Table 2: Dataset description
Table 3: Feature groups
Table 4: Baseline models
Table 5: Main forecasting results
Table 6: Ablation study
Table 7: Drift-period performance
Table 8: Computational cost
Table 9: Candidate-count sensitivity
Table 10: Retrieval/fusion ablation
```

Only tables 1 to 4 can be drafted before experiments. Results tables must remain placeholders until real results exist.

---

## 18. Milestones

### Milestone 1: Paper Draft Up to Methodology

Status: complete as a pre-implementation draft.

Deliverables:

```text
paper/main.tex
paper/references.bib
paper sections up to methodology
architecture diagram draft
dataset description table placeholder
```

### Milestone 2: Kaggle MVP Data Pipeline

Deliverables:

```text
processed hourly PM2.5 dataset
processed weather dataset
calendar features
merged modeling dataset
EDA summary
data-readiness audit and run manifest
```

### Milestone 3: Baselines

Deliverables:

```text
persistence baseline
daily and weekly seasonal naive baselines
24 direct XGBoost horizon models
metrics table
forecast plots
```

### Milestone 4: Time-Series Retrieval

Deliverables:

```text
time-series knowledge base
cosine retrieval
random retrieval baseline
top-k retrieval experiments
retrieved case-study plots
retrieval baseline results
```

### Milestone 5: Event Retrieval

Deliverables:

```text
event knowledge base
event aggregation features
event retrieval examples
event-enhanced modeling dataset
```

### Milestone 6: Event-TimeRAF MVP

Deliverables:

```text
hybrid feature model
drift indicators
forecast output
evidence-based explanation module
```

### Milestone 7: Frozen TSFM Publication Gate

Deliverables:

```text
frozen zero-shot TSFM predictions
retrieval-augmented frozen TSFM predictions
compatibility and runtime record
title-retention or title-fallback decision
```

### Milestone 8: Ablation and Final Results

Deliverables:

```text
main comparison table
ablation table
drift-period analysis
case study figures
```

### Milestone 9: Final Paper

Deliverables:

```text
complete IEEE-style paper
final references
final figures
final tables
presentation material if needed
```

---

## 19. Implementation Guardrails

Do not over-engineer the first version.

Bad first version:

```text
full TimeRAF reproduction
+ full TSFM training
+ LLM reasoning
+ multiple live event/news sources
+ ERA5
+ satellite AOD
+ dashboard
+ paper
```

Good first version:

```text
Step 1: PM2.5 + weather + calendar baseline
Step 2: time-series retrieval baseline
Step 3: event retrieval features
Step 4: Event-TimeRAF feature fusion
Step 5: drift and explanation module
Step 6: frozen TSFM publication gate
Step 7: ablation
Step 8: final paper
```

The first implementation should be small, reproducible, and Kaggle-compatible.

---

## 20. Implementation Execution Contract

The repository scaffold, official-source ingestion, causal feature pipeline,
baselines, retrieval modules, evaluation utilities, tests, and Kaggle notebooks
now implement this contract. Use it together with `structured_plan.md` when
running and extending the experiments:

```text
Build the Kaggle-friendly research repository for:

"Event-TimeRAF: Event-Aware Retrieval-Augmented Foundation Models for Explainable Air Quality Forecasting Under Concept Drift."

Base paper:
"TimeRAF: Retrieval-Augmented Foundation Model for Zero-Shot Time Series Forecasting."

Start with a simple, runnable MVP. Do not fully reproduce TimeRAF first.

Implementation order:
1. Create the repository skeleton, configuration, tests, and run manifest.
2. Audit PM2.5, weather, calendar, and event data against the readiness gates.
3. Build the causal hourly table and 168-hour-to-24-hour windows.
4. Train persistence, daily/weekly seasonal naive, and direct XGBoost baselines.
5. Build leakage-safe random and cosine time-series retrieval.
6. Add source-preserving event features and hybrid retrieval.
7. Add drift indicators and evidence-grounded explanations.
8. Run the frozen-TSFM and retrieval-augmented-TSFM publication gate.
9. Freeze validation choices, evaluate the test set, and run ablations.
10. Complete the paper only from saved, verified result artifacts.

Rules:
- Do not invent results.
- Use time-based train/validation/test split.
- Predict the complete t+1 through t+24 PM2.5 sequence.
- Enforce every leakage and causality rule in structured_plan.md.
- Keep notebooks runnable on Kaggle.
- Keep code modular but simple.
- Save all intermediate processed datasets and outputs.
- Generate all paper tables from saved predictions and metrics.
- Document data limitations clearly.
```
