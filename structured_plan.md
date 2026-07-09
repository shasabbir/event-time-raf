# Event-TimeRAF Structured Execution Plan

This file converts the master plan into a detailed execution checklist. It is for preparation only; no code implementation is included here.

---

## Phase 0: Scope Lock

Goal:

```text
Define a Kaggle-friendly MVP that can support a research paper without overbuilding.
```

Decisions:

| Item | Decision |
| --- | --- |
| Target | Dhaka PM2.5 |
| First horizon | 24 hours |
| First lookback | 168 hours |
| First model family | Persistence, seasonal naive, XGBoost/LightGBM |
| First retrieval | Cosine similarity over historical PM2.5 windows |
| First event module | Aggregated event counts and keyword categories |
| First explanation | Evidence-based templates using features, retrieval, and drift |
| First execution environment | Kaggle notebook |

Out of scope for MVP:

```text
full TimeRAF reproduction
training a time-series foundation model
heavy LLM reasoning
dashboard
satellite AOD
large multi-source data engineering
```

---

## Phase 1: Paper Draft Up to Methodology

Purpose:

```text
Create the academic framing before experiments, without inventing results.
```

Files to create later:

```text
paper/main.tex
paper/references.bib
paper/sections/01_title_abstract.tex
paper/sections/02_introduction.tex
paper/sections/03_related_work.tex
paper/sections/04_problem_statement.tex
paper/sections/05_methodology.tex
paper/sections/06_experimental_setup_placeholder.tex
paper/sections/07_results_placeholder.tex
paper/sections/08_conclusion_placeholder.tex
```

Writing checklist:

| Section | Required content | Result status |
| --- | --- | --- |
| Title/Abstract | Explain Event-TimeRAF at methodology level | No performance claims |
| Introduction | Problem, TimeRAF background, gap, contributions | No fake result claims |
| Related Work | Forecasting, TSFMs, RAG, air quality, explainability | Citation placeholders allowed |
| Problem Statement | Define `X_t`, `W_t`, `C_t`, `E_t`, `Y_hat` | Formal notation |
| Methodology | Architecture and modules | Implementation-oriented |
| Experimental Setup | Planned datasets, baselines, metrics | Placeholder only |
| Results | Placeholder sentence | No numbers |
| Conclusion | Placeholder sentence | No claims |

Acceptance criteria:

```text
- The paper clearly explains TimeRAF.
- The paper clearly explains why event retrieval is needed.
- The paper does not contain invented results.
- The paper is ready to receive real experiment outputs later.
```

---

## Phase 2: Kaggle Repository Skeleton

Purpose:

```text
Keep the repository small enough to run and maintain on Kaggle.
```

Planned structure:

```text
ASA/
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
      retrieval.py
      models.py
      drift.py
      explain.py
      evaluation.py
      plots.py
  data/
    raw/
    processed/
    knowledge_base/
  outputs/
    tables/
    figures/
    predictions/
    logs/
  paper/
    main.tex
    references.bib
    sections/
```

Module responsibilities:

| Module | Responsibility |
| --- | --- |
| `config.py` | Load configuration and constants |
| `data.py` | Download/load/clean PM2.5, weather, event data |
| `features.py` | Lag, rolling, weather, calendar, event features |
| `retrieval.py` | Build time-series KB and retrieve top-k windows |
| `models.py` | Baselines and Event-TimeRAF MVP model wrappers |
| `drift.py` | Drift scores and flags |
| `explain.py` | Template explanations grounded in evidence |
| `evaluation.py` | Metrics, splits, backtesting |
| `plots.py` | Forecast and result visualizations |

Acceptance criteria:

```text
- One main notebook can run the full MVP.
- Source modules stay small and reusable.
- Outputs are saved in predictable folders.
- The structure does not require complex local setup.
```

---

## Phase 3: Data Pipeline

Purpose:

```text
Build one reliable modeling table before training models.
```

Inputs:

```text
air quality data
weather data
calendar features
event/news data
```

Air quality tasks:

```text
1. Load or download Dhaka PM2.5 data.
2. Standardize columns.
3. Convert timestamps to Asia/Dhaka.
4. Remove impossible values.
5. Resample to hourly frequency.
6. Interpolate short gaps only.
7. Save missingness report.
8. Save data/processed/dhaka_pm25_hourly.csv.
```

Weather tasks:

```text
1. Download/load weather for Dhaka coordinates.
2. Align hourly timestamps.
3. Keep key variables.
4. Generate basic weather flags.
5. Save data/processed/dhaka_weather_hourly.csv.
```

Calendar tasks:

```text
1. Generate hour, day_of_week, month, season.
2. Add configurable weekend flag.
3. Add public holiday/Ramadan/Eid flags if available.
4. Save calendar features or merge directly.
```

Event tasks:

```text
1. Query or load event/news records.
2. Filter to Dhaka/Bangladesh context.
3. Assign event categories using keywords.
4. Aggregate event counts by timestamp window.
5. Save data/processed/dhaka_event_features.csv.
```

Merged dataset:

```text
data/processed/modeling_hourly.csv
```

Acceptance criteria:

```text
- Timestamp alignment is verified.
- No random future leakage is introduced.
- Missing values are reported.
- The modeling table has a clear target column.
```

---

## Phase 4: Feature Engineering

Purpose:

```text
Create model-ready features without leakage.
```

Feature groups:

| Group | Examples |
| --- | --- |
| PM2.5 lags | `pm25_lag_1h`, `pm25_lag_24h`, `pm25_lag_168h` |
| PM2.5 rolling | rolling mean/std/max/min |
| Weather | temperature, humidity, wind, rain, pressure |
| Weather interactions | low wind, rain flag, stagnation index |
| Calendar | hour, weekend, season, holiday flags |
| Events | event counts by category and time window |
| Retrieval | retrieved future mean, retrieved similarity, retrieved trend |
| Drift | mean shift, variance shift, similarity drop |

Target creation:

```text
target_pm25_t_plus_24h
```

Optional multi-output target later:

```text
target sequence for t+1 to t+24
```

Leakage rules:

```text
- Use only past and present features for a forecast timestamp.
- Do not use future weather unless the experiment explicitly treats it as forecast weather.
- Do not compute rolling features using future values.
- Build retrieval candidates only from training history during validation/test.
```

Acceptance criteria:

```text
- Features can be generated repeatedly from raw/processed data.
- The target horizon is explicit.
- Leakage checks are documented.
```

---

## Phase 5: Baseline Models

Purpose:

```text
Create a defensible comparison before adding retrieval.
```

Required baselines:

| Model | Purpose |
| --- | --- |
| Persistence | Minimum sanity baseline |
| Seasonal naive | Daily/weekly periodic baseline |
| XGBoost or LightGBM basic | Strong tabular baseline |
| XGBoost/LightGBM weather-calendar | Tests added context |

Optional baselines:

```text
ARIMA/SARIMA
Prophet
LSTM/GRU
PatchTST or other transformer model
```

Evaluation:

```text
time-based 70/15/15 split
MAE
RMSE
MAPE
sMAPE
R2
```

Acceptance criteria:

```text
- At least persistence, seasonal naive, and one tree model run successfully.
- Metrics are saved to outputs/tables.
- Forecast plots are saved to outputs/figures.
```

---

## Phase 6: Time-Series Retrieval Baseline

Purpose:

```text
Implement a simplified TimeRAF-inspired baseline without reproducing full TimeRAF.
```

Knowledge base:

```text
data/knowledge_base/ts_windows.parquet
```

Each record:

```text
window_id
start_time
end_time
input_window_values
future_window_values
weather_summary
calendar_summary
embedding_vector
```

Retrieval methods:

```text
random retrieval
cosine similarity
FAISS nearest neighbor if available
```

Retrieved features:

```text
retrieved_future_mean
retrieved_future_std
retrieved_future_trend
retrieval_similarity_mean
retrieval_similarity_max
retrieved_neighbor_count
```

Experiments:

```text
k = 1, 4, 8, 16
window length = 168 hours
horizon = 24 hours
```

Acceptance criteria:

```text
- Retrieval excludes future/test leakage.
- Top-k examples can be inspected.
- Retrieval features improve or at least provide analyzable ablation results.
```

---

## Phase 7: Event Retrieval

Purpose:

```text
Add external context without making the system dependent on hallucinated reasoning.
```

Event categories:

```text
fire
traffic
rainstorm
construction
industrial
public_gathering
holiday_movement
dust
flood
policy
```

Event knowledge base:

```text
data/knowledge_base/event_kb.parquet
```

Each record:

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
```

Aggregated event features:

```text
event_count_24h
event_count_72h
fire_event_count_72h
traffic_event_count_72h
rain_event_count_72h
industrial_event_count_7d
negative_tone_avg_72h
top_event_text
```

Acceptance criteria:

```text
- Event features align with forecast timestamps.
- Event text is stored for explanation evidence.
- Event module has a fallback if live event collection fails.
```

---

## Phase 8: Event-TimeRAF MVP

Purpose:

```text
Combine baseline features, retrieval features, event features, and drift indicators into one model.
```

MVP input:

```text
PM2.5 lag/rolling features
weather features
calendar features
time-series retrieval summary features
event aggregation features
drift indicators
```

MVP model:

```text
XGBoost or LightGBM
```

Outputs:

```text
forecast
confidence proxy
drift_flag
drift_score
explanation text
retrieved evidence
```

Confidence proxy options:

```text
variation among retrieved future trajectories
model residual quantiles on validation set
tree ensemble prediction interval if available
```

Acceptance criteria:

```text
- The full model runs end to end.
- Predictions are saved.
- Each prediction can be linked to retrieval and event evidence.
```

---

## Phase 9: Drift Detection

Purpose:

```text
Support the "under concept drift" claim with measurable indicators.
```

MVP drift indicators:

```text
rolling mean shift
rolling variance shift
retrieval similarity drop
event burst indicator
weather pattern shift
```

Outputs:

```text
drift_score
drift_flag
drift_reason
```

Evaluation:

```text
compare normal-period error and drift-period error
```

Acceptance criteria:

```text
- Drift periods are identified by transparent rules.
- Drift-period metrics are reported separately.
- The explanation module can mention drift only when drift evidence exists.
```

---

## Phase 10: Explanation Module

Purpose:

```text
Generate explanations grounded in model and retrieval evidence.
```

Evidence inputs:

```text
top feature importance or SHAP values
retrieved historical cases
retrieved event records
weather flags
calendar flags
drift score
forecast direction
```

Template structure:

```text
1. Forecast direction.
2. Main numerical drivers.
3. Weather/calendar context.
4. Retrieved similar historical behavior.
5. Retrieved event context.
6. Drift warning if applicable.
```

Rules:

```text
- Do not invent events.
- Do not mention a driver unless it exists in evidence.
- Do not use a pure LLM explanation.
- Save explanation evidence with each output.
```

Acceptance criteria:

```text
- Explanations are reproducible.
- Explanations cite retrieved evidence fields.
- Case-study examples can be inserted into the paper.
```

---

## Phase 11: Ablation Study

Purpose:

```text
Prove which components help.
```

Ablation matrix:

| Model | PM2.5 | Weather | Calendar | TS Retrieval | Event Retrieval | Drift | Explanation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persistence | Yes | No | No | No | No | No | No |
| Seasonal naive | Yes | No | Yes | No | No | No | No |
| XGBoost-basic | Yes | No | No | No | No | No | No |
| XGBoost-weather-calendar | Yes | Yes | Yes | No | No | No | No |
| TimeRAF-inspired | Yes | Yes | Yes | Yes | No | No | Limited |
| Event-TimeRAF-no-drift | Yes | Yes | Yes | Yes | Yes | No | Yes |
| Event-TimeRAF-full | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

Outputs:

```text
outputs/tables/main_results.csv
outputs/tables/ablation_results.csv
outputs/tables/drift_period_results.csv
outputs/figures/error_comparison.png
outputs/figures/ablation_bar_chart.png
```

Acceptance criteria:

```text
- All models use the same split.
- All metrics are computed consistently.
- Results are real and reproducible.
```

---

## Phase 12: Final Paper Completion

Purpose:

```text
Convert real experiment outputs into the final paper.
```

Add after experiments:

```text
main forecasting results
ablation results
drift-period analysis
event case study
explanation examples
limitations
conclusion
```

Paper figures:

```text
Figure 1: Event-TimeRAF architecture
Figure 2: Knowledge-base construction
Figure 3: Hybrid retrieval process
Figure 4: Forecast visualization
Figure 5: Error comparison
Figure 6: Ablation study
Figure 7: Explanation case study
```

Paper tables:

```text
Table 1: Comparison with TimeRAF
Table 2: Dataset description
Table 3: Feature groups
Table 4: Baseline models
Table 5: Main forecasting results
Table 6: Ablation study
Table 7: Drift-period performance
Table 8: Computational cost
```

Acceptance criteria:

```text
- No placeholder result remains in the final results section.
- Every performance claim is backed by a saved result table.
- Limitations are clearly stated.
- The final paper is internally consistent with the implementation.
```

---

## Execution Order Summary

Use this order when implementation begins:

```text
1. Paper draft up to methodology.
2. Lightweight Kaggle repository skeleton.
3. PM2.5 + weather + calendar data pipeline.
4. Baseline models.
5. Time-series retrieval baseline.
6. Event features and event retrieval.
7. Event-TimeRAF MVP.
8. Drift detection.
9. Explanation module.
10. Ablation experiments.
11. Final paper update with real results.
```

The project should not move to advanced neural fusion or TSFM integration until steps 1 to 7 are complete and reproducible.
