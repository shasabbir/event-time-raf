Below is the **Codex Plus execution plan**. Give this to Codex almost directly. It is strict, implementation-oriented, and paper-oriented.

Your project idea:

**Event-TimeRAF: Event-Aware Retrieval-Augmented Foundation Models for Explainable Air Quality Forecasting Under Concept Drift**

Base paper:

**TimeRAF: Retrieval-Augmented Foundation Model for Zero-Shot Time Series Forecasting**

TimeRAF is a 2025 IEEE TKDE paper. It proposes retrieval augmentation for time-series foundation models, using a learnable retriever, a time-series knowledge base, Channel Prompting, and a frozen TSFM backbone. 

---

# 1. Main instruction for Codex Plus

Use this as the master instruction:

```text
You are Codex Plus acting as a research engineer and academic paper assistant.

Your task is to build a complete research project based on the paper:
"TimeRAF: Retrieval-Augmented Foundation Model for Zero-Shot Time Series Forecasting."

The new proposed work is:
"Event-TimeRAF: Event-Aware Retrieval-Augmented Foundation Models for Explainable Air Quality Forecasting Under Concept Drift."

First, write the paper from Title to Methodology only. Do not write fake results. Results, discussion, and conclusion must remain placeholders until experiments are completed.

Then implement the full experimental pipeline in Python, preferably notebook-compatible for Kaggle/Colab and also runnable as scripts.

The work must include:
1. TimeRAF base-paper explanation.
2. Clear research gap.
3. Proposed Event-TimeRAF architecture.
4. Dataset acquisition and preprocessing.
5. Baseline forecasting models.
6. TimeRAF-inspired retrieval baseline.
7. Event-aware retrieval module.
8. Explainable forecast output.
9. Ablation experiments.
10. Final paper-writing pipeline after results are available.

Do not invent experimental results. Use placeholders until real results are generated.
```

---

# 2. Research objective

## Project title

**Event-TimeRAF: Event-Aware Retrieval-Augmented Foundation Models for Explainable Air Quality Forecasting Under Concept Drift**

## Core objective

Build a forecasting system that predicts future **PM2.5 / AQI for Dhaka** by combining:

```text
historical air-quality time series
+
weather features
+
calendar features
+
retrieved similar time-series patterns
+
retrieved real-world event/news context
+
time-series foundation model / forecasting backbone
+
explanation module
```

## Why this project exists

TimeRAF retrieves relevant **time-series sequences** from a knowledge base and injects them into a frozen time-series foundation model using Channel Prompting. 

But TimeRAF mainly retrieves **numeric time-series patterns**. It does not deeply retrieve or reason over external real-world events such as:

```text
rainfall
fire incidents
traffic disruption
holidays
construction activity
industrial activity
dust events
public gatherings
policy changes
```

Your teacher’s topic list already suggested **RAG-Enhanced Time Series Forecasting with Event Reasoning**, **Adaptive Context Retrieval for TSFMs under Concept Drift**, and **EventRAG: Retrieval-Augmented Forecasting with News-Aware Temporal Reasoning**. The listed EventRAG pipeline is: historical data → event retrieval → LLM reasoning → TSFM → forecast + explanation. 

So the project is not random. It is a direct extension of TimeRAF using your teacher’s suggested novelty direction.

---

# 3. Codex phase 1: write paper up to methodology first

This is the first task. Codex must create:

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

## Paper should not include fake results

Codex must write only:

1. Title
2. Abstract draft
3. Introduction
4. Related Work
5. Problem Statement
6. Proposed Methodology
7. Experimental Setup plan

Results section should say:

```text
Experimental results will be reported after model implementation and evaluation.
```

No fake numbers. No fake improvement percentage. Fake results will kill the paper.

---

# 4. Paper content plan up to methodology

## 4.1 Title

Use:

```text
Event-TimeRAF: Event-Aware Retrieval-Augmented Foundation Models for Explainable Air Quality Forecasting Under Concept Drift
```

Alternative shorter title:

```text
Event-TimeRAF: Event-Aware Retrieval-Augmented Forecasting for Urban Air Quality Prediction
```

Use the first one for research depth.

---

## 4.2 Abstract draft instruction

Codex should write an abstract with this structure:

```text
Air quality forecasting is important for urban health and environmental decision-making.
Recent time-series foundation models show strong zero-shot forecasting capability, but they often depend mainly on historical numerical patterns.
TimeRAF improves zero-shot forecasting by retrieving useful time-series sequences from a knowledge base and injecting them into a frozen foundation model.
However, many air-quality changes are event-driven and influenced by meteorology, holidays, traffic, industrial activity, and abnormal environmental events.
This paper proposes Event-TimeRAF, an event-aware retrieval-augmented forecasting framework that combines historical time-series retrieval with weather, calendar, and event/news retrieval.
The proposed system produces PM2.5 forecasts, confidence estimates, drift indicators, and natural-language explanations.
The methodology is evaluated using Dhaka air-quality data, meteorological variables, calendar features, and event-context retrieval.
```

Do not mention final performance yet.

---

## 4.3 Introduction structure

Codex must write the introduction in 5 paragraphs:

### Paragraph 1: Problem importance

Urban air pollution is a major forecasting problem. PM2.5 forecasting helps health alerts, policy, and city planning.

### Paragraph 2: Existing forecasting weakness

Traditional models such as ARIMA, Prophet, LSTM, GRU, XGBoost, and Transformers learn historical patterns but struggle during abnormal external events.

### Paragraph 3: Time-series foundation models

TSFMs improve generalization and zero-shot forecasting. TimeRAF extends TSFMs with retrieval augmentation by retrieving useful time-series candidates from a curated knowledge base. TimeRAF keeps the backbone frozen and uses Channel Prompting for knowledge integration. 

### Paragraph 4: Research gap

TimeRAF retrieves numerical time-series knowledge, but air quality is strongly affected by external context: weather, rainfall, traffic, public events, holidays, fires, construction, and industrial incidents.

### Paragraph 5: Contribution

List contributions:

```text
1. Event-aware retrieval-augmented forecasting framework.
2. Hybrid knowledge base combining time-series windows, weather, calendar, and event context.
3. Concept-drift-aware retrieval strategy.
4. Explanation module for forecast reasoning.
5. Evaluation on Dhaka air-quality forecasting with TimeRAF-inspired baselines.
```

---

## 4.4 Related work structure

Codex should write Related Work with these subsections:

```text
2.1 Time-Series Forecasting
2.2 Time-Series Foundation Models
2.3 Retrieval-Augmented Generation for Time-Series Forecasting
2.4 Air Quality Forecasting
2.5 Event-Aware and Explainable Forecasting
```

Important: relate TimeRAF properly.

TimeRAF’s experiment setup used ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Electricity for zero-shot evaluation. It used TTM-Base as the backbone, context length 512, forecast horizon 96, and compared against TTM, Moirai, MOMENT, Timer, Chronos, and TimesFM. 

Codex should mention this in related work or methodology background.

---

## 4.5 Problem statement

Codex should define:

```text
Given historical PM2.5 observations X_air,
weather covariates X_weather,
calendar features X_calendar,
and event context E,
predict future PM2.5 values Y_hat for horizon H.
```

Formal version:

```text
X_t = [x_{t-L+1}, ..., x_t]
W_t = weather covariates aligned with X_t
C_t = calendar/context features aligned with X_t
E_t = retrieved event documents or structured event features near time t

Goal:
Y_hat_{t+1:t+H} = F(X_t, W_t, C_t, E_t)
```

Where:

| Symbol  | Meaning                                 |
| ------- | --------------------------------------- |
| `L`     | lookback window                         |
| `H`     | forecast horizon                        |
| `X_t`   | historical PM2.5 / air-quality sequence |
| `W_t`   | weather features                        |
| `C_t`   | calendar features                       |
| `E_t`   | event/news context                      |
| `Y_hat` | predicted future PM2.5                  |

---

# 5. Methodology plan for paper

Codex must write methodology as if it will later be implemented.

## 5.1 Overall architecture

```text
Historical Air Quality Data
        ↓
Weather + Calendar Feature Alignment
        ↓
Sliding Window Construction
        ↓
Time-Series Knowledge Base
        ↓
Time-Series Retriever
        ↓
Event/News Knowledge Base
        ↓
Event Retriever
        ↓
Context Fusion Module
        ↓
Forecasting Backbone
        ↓
Forecast + Confidence + Explanation
```

## 5.2 TimeRAF baseline explanation

Codex should explain TimeRAF as the base method:

```text
TimeRAF retrieves top-k time-series candidates from a knowledge base.
The input sequence is used as the query.
A learnable retriever scores candidate windows.
Retrieved candidates are integrated with the input using Channel Prompting.
The knowledge-enhanced representation is passed into a frozen time-series foundation model.
```

TimeRAF’s inference process retrieves the highest-scoring candidates, applies preprocessing, uses Channel Prompting for integration, and feeds the enhanced embeddings into the TSFM backbone for final prediction. 

## 5.3 Our Event-TimeRAF extension

Codex should define four modules.

### Module A: Time-Series Retriever

Input:

```text
current PM2.5/weather time-series window
```

Retrieves:

```text
top-k similar historical air-quality windows
```

Candidate methods:

```text
FAISS nearest neighbor
cosine similarity
Euclidean distance
DTW distance
learnable MLP retriever
```

MVP should use FAISS/cosine first. Learnable retriever is optional later.

### Module B: Event Retriever

Input:

```text
forecast timestamp
location = Dhaka
event keywords
recent weather/event context
```

Retrieves:

```text
news/event records around Dhaka within last 24h / 72h / 7d
```

Event categories:

```text
fire
traffic
rainstorm
construction
industrial incident
public gathering
holiday movement
dust
flood
policy restriction
```

### Module C: Context Fusion

Combine:

```text
input sequence embedding
retrieved time-series embedding
weather feature embedding
calendar feature embedding
event-context embedding
```

Possible fusion methods:

```text
early concatenation
MLP fusion
cross-attention fusion
gated fusion
```

MVP:

```text
concatenate all structured features + retrieved summary features
feed into XGBoost / LightGBM / MLP / TTM-style model
```

Advanced:

```text
Channel Prompting-inspired embedding fusion
```

### Module D: Explanation Generator

Output explanation using:

```text
retrieved events
top weather drivers
SHAP feature importance
retrieved similar historical cases
forecast trend direction
```

Example explanation:

```text
The PM2.5 forecast increases because similar past windows with low wind speed and high humidity showed rising pollution. No rainfall event was retrieved for the forecast horizon, reducing expected particle washout. Recent traffic-related events may also contribute to the predicted increase.
```

Do not let Codex use a pure LLM explanation without evidence. That becomes hallucination. Explanation must be grounded in retrieved data and feature contribution.

---

# 6. Repository structure Codex should create

```text
event-timeraf/
│
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── default.yaml
│   ├── data_openaq.yaml
│   ├── data_weather.yaml
│   ├── model_baseline.yaml
│   ├── model_event_timeraf.yaml
│   └── experiment_ablation.yaml
│
├── data/
│   ├── raw/
│   │   ├── air_quality/
│   │   ├── weather/
│   │   ├── events/
│   │   └── calendar/
│   ├── interim/
│   ├── processed/
│   └── knowledge_base/
│
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_data_cleaning_eda.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_time_series_retrieval.ipynb
│   ├── 05_event_retrieval.ipynb
│   ├── 06_event_timeraf_training.ipynb
│   ├── 07_ablation_study.ipynb
│   └── 08_results_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── download_openaq.py
│   │   ├── download_weather.py
│   │   ├── download_events_gdelt.py
│   │   ├── build_calendar_features.py
│   │   ├── clean_air_quality.py
│   │   ├── align_datasets.py
│   │   └── make_windows.py
│   │
│   ├── retrieval/
│   │   ├── build_ts_kb.py
│   │   ├── ts_retriever.py
│   │   ├── event_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── retrieval_evaluation.py
│   │
│   ├── features/
│   │   ├── lag_features.py
│   │   ├── rolling_features.py
│   │   ├── weather_features.py
│   │   ├── event_features.py
│   │   └── feature_store.py
│   │
│   ├── models/
│   │   ├── persistence.py
│   │   ├── arima_model.py
│   │   ├── prophet_model.py
│   │   ├── xgboost_model.py
│   │   ├── lstm_model.py
│   │   ├── ttm_wrapper.py
│   │   ├── timeraf_baseline.py
│   │   └── event_timeraf.py
│   │
│   ├── explanation/
│   │   ├── shap_explainer.py
│   │   ├── event_reasoning.py
│   │   ├── explanation_templates.py
│   │   └── generate_report.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── backtesting.py
│   │   ├── drift_detection.py
│   │   ├── ablation.py
│   │   └── plots.py
│   │
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       ├── seed.py
│       └── io.py
│
├── experiments/
│   ├── run_baselines.py
│   ├── run_timeraf_baseline.py
│   ├── run_event_timeraf.py
│   ├── run_ablation.py
│   └── run_all.py
│
├── results/
│   ├── tables/
│   ├── figures/
│   ├── predictions/
│   └── logs/
│
└── paper/
    ├── main.tex
    ├── references.bib
    ├── figures/
    ├── tables/
    └── sections/
```

---

# 7. Dataset implementation plan

## 7.1 Main target dataset

Use:

```text
OpenAQ / air quality data
Target: PM2.5
Location: Dhaka, Bangladesh
Frequency: hourly
```

Features:

```text
datetime
station/location
latitude
longitude
pollutant parameter
PM2.5 value
unit
source/provider
```

Codex tasks:

```text
1. Download Dhaka PM2.5 data.
2. Convert all timestamps to Asia/Dhaka timezone.
3. Resample to hourly frequency.
4. Remove impossible values.
5. Interpolate short gaps.
6. Keep missingness report.
7. Save processed file as data/processed/dhaka_pm25_hourly.csv.
```

## 7.2 Weather dataset

Use:

```text
Open-Meteo first
ERA5 optional later
```

Features:

```text
temperature_2m
relative_humidity_2m
rainfall/precipitation
wind_speed_10m
wind_direction_10m
surface_pressure
cloud_cover
visibility if available
```

Codex tasks:

```text
1. Download weather for Dhaka coordinates.
2. Match weather timestamps with PM2.5 timestamps.
3. Create lagged weather features.
4. Save as data/processed/dhaka_weather_hourly.csv.
```

## 7.3 Calendar features

Codex should generate:

```text
hour
day_of_week
is_weekend
month
season
is_public_holiday
is_ramadan
is_eid_period
```

Bangladesh weekend should be handled carefully. Do not blindly use Western Saturday/Sunday only. Use configurable weekend days.

## 7.4 Event/news data

Use:

```text
GDELT or another accessible news/event source
```

MVP event features:

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

Codex tasks:

```text
1. Query GDELT for Dhaka/Bangladesh event records.
2. Filter by date and location.
3. Filter by event keywords.
4. Aggregate event counts by day/hour.
5. Create event text summaries.
6. Save as data/processed/dhaka_event_features.csv.
```

---

# 8. Feature engineering plan

Codex must create these features.

## 8.1 Lag features

```text
pm25_lag_1h
pm25_lag_3h
pm25_lag_6h
pm25_lag_12h
pm25_lag_24h
pm25_lag_48h
pm25_lag_168h
```

## 8.2 Rolling features

```text
pm25_roll_mean_3h
pm25_roll_mean_6h
pm25_roll_mean_24h
pm25_roll_mean_168h
pm25_roll_std_24h
pm25_roll_max_24h
pm25_roll_min_24h
```

## 8.3 Weather interaction features

```text
wind_low_flag
rain_flag
humidity_high_flag
stagnation_index = low wind + high humidity
rain_washout_indicator
```

## 8.4 Event features

```text
event_count_24h
event_count_72h
fire_event_count_72h
traffic_event_count_72h
industrial_event_count_7d
holiday_event_flag
```

## 8.5 Forecast target

Start with:

```text
H = 24 hours
target = PM2.5 next 24 hours
```

Later add:

```text
H = 48 hours
H = 72 hours
H = 96 hours
```

TimeRAF used forecast horizon 96 in most experiments, but for Dhaka air quality, 24h should be the first practical target. 

---

# 9. Model implementation plan

## 9.1 Baselines

Codex must implement these first:

```text
1. Persistence baseline
2. Seasonal naive baseline
3. ARIMA/SARIMA
4. Prophet or NeuralProphet
5. XGBoost / LightGBM
6. LSTM / GRU
7. Transformer/PatchTST if feasible
```

This is not optional. Without baselines, your paper is weak.

## 9.2 TimeRAF-inspired baseline

Implement a simplified TimeRAF baseline:

```text
Input current PM2.5 window
↓
Retrieve top-k similar historical windows
↓
Aggregate retrieved future trajectories
↓
Fuse retrieved features with current input
↓
Forecast with XGBoost/MLP/TTM backbone
```

Start simple:

```text
FAISS/cosine retrieval
k = 1, 4, 8, 16
window length = 168 hours
forecast horizon = 24 hours
```

Then add:

```text
learnable retriever using MLP encoder
```

TimeRAF uses top-k candidates and reports that too many candidates can introduce redundancy/noise, so Codex should test k values instead of blindly choosing one. 

## 9.3 Event-TimeRAF model

MVP architecture:

```text
Air-quality window encoder
Weather/calendar feature encoder
Time-series retriever
Event retriever
Fusion MLP
Forecast head
Explanation module
```

Advanced architecture:

```text
Channel Prompting-inspired fusion:
input embedding
+
retrieved time-series embedding
+
event embedding
+
weather/calendar embedding
↓
MLP/gated fusion
↓
forecasting backbone
```

Do not try to fully reproduce TimeRAF first. That is too heavy. Build a working TimeRAF-inspired baseline, then improve it.

---

# 10. Retrieval module plan

## 10.1 Time-series knowledge base

Codex must build:

```text
data/knowledge_base/ts_windows.parquet
```

Each record:

```text
window_id
start_time
end_time
station
location
input_window_values
future_window_values
weather_summary
calendar_summary
event_summary_id
embedding_vector
```

## 10.2 Retrieval methods

Implement three retrieval methods:

```text
1. Random retrieval
2. Cosine similarity retrieval
3. FAISS nearest-neighbor retrieval
```

Optional advanced:

```text
4. Learnable MLP retriever
```

TimeRAF compared against random and cosine-style retrieval to show the value of its learnable retriever, so your ablation should also compare random, similarity-based, and event-aware retrieval. 

## 10.3 Event knowledge base

Codex must build:

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
embedding_vector
```

## 10.4 Hybrid retrieval score

Use:

```text
score = α * time_series_similarity
      + β * weather_similarity
      + γ * calendar_similarity
      + δ * event_relevance
```

Start with:

```text
α = 0.5
β = 0.2
γ = 0.1
δ = 0.2
```

Then tune with validation.

---

# 11. Concept drift plan

Codex must implement drift detection. Keep it simple first.

## Drift indicators

```text
recent_mean_shift
recent_variance_shift
distribution_distance
retrieval_similarity_drop
weather_pattern_shift
event_burst_indicator
```

## Methods

MVP:

```text
rolling mean/std change
z-score anomaly
retrieval similarity threshold
```

Advanced:

```text
ADWIN
Page-Hinkley
Kolmogorov-Smirnov test
Maximum Mean Discrepancy
```

Output:

```text
drift_flag = 0/1
drift_score = continuous value
```

This directly supports your paper’s “under concept drift” claim.

---

# 12. Explainability plan

Codex must create explanation output for each forecast.

## Explanation should use evidence from:

```text
1. SHAP feature importance
2. Retrieved similar historical windows
3. Retrieved event records
4. Weather conditions
5. Calendar context
6. Drift score
```

## Example generated explanation

```text
The model predicts a PM2.5 increase in the next 24 hours. The forecast is mainly influenced by high recent PM2.5 persistence, low wind speed, high humidity, and absence of rainfall. Retrieved historical windows with similar stagnant weather patterns showed similar PM2.5 increases. Recent event retrieval also found traffic-related events near Dhaka, which may contribute to emission accumulation. The drift score is moderate, indicating that the current pattern differs from normal weekly behavior.
```

Important rule for Codex:

```text
Do not generate explanation from the LLM alone.
Use evidence-based templates.
```

---

# 13. Evaluation plan

## 13.1 Metrics

Use:

```text
MAE
RMSE
MAPE
sMAPE
R²
```

For high-pollution event periods:

```text
event-period MAE
event-period RMSE
drift-period MAE
normal-period MAE
```

## 13.2 Forecast horizons

Use:

```text
24h
48h
72h
96h
```

## 13.3 Train/validation/test split

Use time-based split:

```text
Train: earliest 70%
Validation: next 15%
Test: latest 15%
```

Never randomly split time series. That is leakage.

## 13.4 Backtesting

Use rolling-origin evaluation:

```text
train on past
validate on future
slide forward
repeat
```

---

# 14. Ablation study plan

Codex must implement this table.

| Model                    | PM2.5 | Weather | Calendar | TS Retrieval | Event Retrieval | Drift | Explanation |
| ------------------------ | ----: | ------: | -------: | -----------: | --------------: | ----: | ----------: |
| Persistence              |   Yes |      No |       No |           No |              No |    No |          No |
| XGBoost-basic            |   Yes |      No |       No |           No |              No |    No |          No |
| XGBoost-weather          |   Yes |     Yes |       No |           No |              No |    No |          No |
| XGBoost-weather-calendar |   Yes |     Yes |      Yes |           No |              No |    No |          No |
| TimeRAF-inspired         |   Yes |     Yes |      Yes |          Yes |              No |    No |     Limited |
| Event-TimeRAF-no-drift   |   Yes |     Yes |      Yes |          Yes |             Yes |    No |         Yes |
| Event-TimeRAF-full       |   Yes |     Yes |      Yes |          Yes |             Yes |   Yes |         Yes |

This is the core proof. If this ablation is missing, the paper is weak.

---

# 15. Experiment commands Codex should create

```bash
# setup
pip install -r requirements.txt

# data
python experiments/run_data_pipeline.py --config configs/default.yaml

# baselines
python experiments/run_baselines.py --config configs/model_baseline.yaml

# retrieval baseline
python experiments/run_timeraf_baseline.py --config configs/model_timeraf_baseline.yaml

# proposed model
python experiments/run_event_timeraf.py --config configs/model_event_timeraf.yaml

# ablation
python experiments/run_ablation.py --config configs/experiment_ablation.yaml

# generate figures and tables
python experiments/generate_results.py

# compile paper
cd paper && latexmk -pdf main.tex
```

---

# 16. Figures Codex must generate

## For paper

```text
Figure 1: Overall Event-TimeRAF architecture
Figure 2: Time-series and event knowledge-base construction
Figure 3: Hybrid retrieval process
Figure 4: Forecasting pipeline
Figure 5: Example forecast visualization
Figure 6: Error comparison across models
Figure 7: Ablation study bar chart
Figure 8: Explanation case study
```

## For teacher presentation

```text
1. Base TimeRAF process
2. Our Event-TimeRAF process
3. Dataset stack
4. Architecture diagram
5. Evaluation plan
```

---

# 17. Tables Codex must generate

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

---

# 18. Literature/reference plan

Codex should create `references.bib` with at least these groups:

## Base paper

```text
TimeRAF, IEEE TKDE 2025
```

## Time-series foundation models

```text
TimesFM
Moirai
Timer
MOMENT
Chronos
TTM
PatchTST
Time-LLM
```

## Retrieval/RAG

```text
RAG
DPR
Retrieval-augmented time-series forecasting
TS-RAG
RAFT
```

## Air-quality forecasting

```text
PM2.5 forecasting with ML/deep learning
meteorological impact on PM2.5
urban air pollution forecasting
```

## Dataset/data source references

```text
OpenAQ
Open-Meteo
ERA5
GDELT
```

---

# 19. Codex writing style for the paper

Tell Codex:

```text
Write in IEEE academic style.
Avoid exaggerated claims.
Do not say "novel" too many times.
Use "proposed framework" instead of marketing language.
Do not claim SOTA unless experiments prove it.
Do not write fake results.
Every claim in Related Work must have a citation placeholder.
```

---

# 20. Final paper structure after experiments

Codex should eventually produce:

```text
Title
Abstract
Keywords

1. Introduction
2. Related Work
   2.1 Time-Series Forecasting
   2.2 Time-Series Foundation Models
   2.3 Retrieval-Augmented Forecasting
   2.4 Air-Quality Forecasting
   2.5 Explainable and Event-Aware Forecasting

3. Problem Formulation

4. Proposed Methodology
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

5. Experimental Setup
   5.1 Dataset
   5.2 Baselines
   5.3 Metrics
   5.4 Implementation Details

6. Results and Analysis
   6.1 Main Forecasting Results
   6.2 Ablation Study
   6.3 Drift Period Analysis
   6.4 Event Case Study
   6.5 Explanation Quality
   6.6 Computational Efficiency

7. Discussion
8. Limitations
9. Conclusion
References
```

---

# 21. Important implementation warning

Do **not** let Codex over-engineer the first version.

Bad plan:

```text
Full TimeRAF reproduction
+
full TSFM training
+
LLM reasoning
+
GDELT
+
ERA5
+
satellite AOD
+
dashboard
+
paper
```

That is too much. It will collapse.

Good plan:

```text
Step 1: PM2.5 + weather + calendar baseline
Step 2: time-series retrieval baseline
Step 3: event retrieval features
Step 4: Event-TimeRAF fusion
Step 5: ablation
Step 6: write final paper
```

---

# 22. Practical milestone plan

## Milestone 1: Paper draft up to methodology

Deliverables:

```text
paper/main.tex
paper/references.bib
paper sections up to methodology
architecture diagram draft
dataset description table
```

## Milestone 2: Dataset pipeline

Deliverables:

```text
processed hourly PM2.5 dataset
processed weather dataset
calendar features
merged modeling dataset
EDA notebook
```

## Milestone 3: Baseline models

Deliverables:

```text
persistence
seasonal naive
XGBoost
LSTM/GRU
metrics table
forecast plots
```

## Milestone 4: TimeRAF-inspired retrieval

Deliverables:

```text
time-series knowledge base
retrieval module
top-k candidate experiments
retrieval baseline results
```

## Milestone 5: Event retrieval

Deliverables:

```text
event knowledge base
event aggregation features
event retrieval examples
event-enhanced dataset
```

## Milestone 6: Event-TimeRAF model

Deliverables:

```text
hybrid fusion model
forecast output
confidence/drift score
explanation module
```

## Milestone 7: Ablation and final results

Deliverables:

```text
full comparison table
ablation table
drift-period analysis
case study
figures
```

## Milestone 8: Final paper

Deliverables:

```text
complete IEEE-format paper
final references
final figures
final tables
presentation update
```

---

# 23. Final Codex prompt

Paste this as the final prompt:

```text
Build a complete research repository for the project:

"Event-TimeRAF: Event-Aware Retrieval-Augmented Foundation Models for Explainable Air Quality Forecasting Under Concept Drift."

Base paper:
"TimeRAF: Retrieval-Augmented Foundation Model for Zero-Shot Time Series Forecasting," IEEE TKDE 2025.

First task:
Write the paper from Title to Methodology only. Do not create fake results. Keep Results, Discussion, and Conclusion as placeholders.

The paper must explain:
1. The TimeRAF base method.
2. Its use of retrieval augmentation for TSFMs.
3. Its learnable retriever and Channel Prompting.
4. Its limitation: retrieval is mainly numerical time-series pattern retrieval.
5. Our extension: retrieve both time-series patterns and external event/weather/calendar context.
6. Our target domain: Dhaka PM2.5 / air-quality forecasting.
7. Our outputs: forecast, confidence, drift indicator, and explanation.

Then create the repository structure and implementation plan.

Implement in phases:
1. Dataset pipeline for Dhaka PM2.5, weather, calendar, and event data.
2. Baselines: persistence, seasonal naive, ARIMA/SARIMA, XGBoost/LightGBM, LSTM/GRU.
3. TimeRAF-inspired retrieval baseline using a time-series knowledge base and top-k similar windows.
4. Event retriever using news/event records and structured event features.
5. Event-TimeRAF hybrid model combining PM2.5 history, weather, calendar, time-series retrieval, and event retrieval.
6. Drift detection.
7. Explanation generation using retrieved evidence and SHAP/template-based reasoning.
8. Evaluation using MAE, RMSE, MAPE, sMAPE, R², event-period error, and drift-period error.
9. Ablation study comparing no retrieval, time-series retrieval only, event retrieval only, and full Event-TimeRAF.
10. Generate final paper tables, figures, and IEEE-style manuscript after real results are produced.

Rules:
- Do not invent results.
- Use time-based train/validation/test split.
- Save all data processing scripts.
- Make notebooks runnable in Kaggle/Colab.
- Keep the first implementation simple and working before adding advanced modules.
- Produce clean, reusable, documented code.
```

This is the plan you need. It is strict enough for Codex, but not so bloated that it turns into a failed fantasy project.
