# Event-TimeRAF Finalized Implementation Plan

Status: **implementation scaffold complete; full 2019--2024 Kaggle experiment
and frozen result artifacts pending**

This document is the binding implementation contract for the project. `plan.md`
remains the research overview, while this file controls implementation order,
interfaces, leakage rules, experiments, and completion criteria. Any later scope
change should be recorded here before code and paper claims are changed.

---

## 1. Locked Scope

### 1.1 Primary forecasting task

| Item | Final decision |
| --- | --- |
| Location | Los Angeles County, California, USA |
| Study period | 2019--2024, subject to the data-readiness audit |
| Target | Hourly PM2.5 in the source unit, expected to be `ug/m3` |
| Forecast output | Full sequence from `t+1` through `t+24` |
| Lookback | `L = 168` hourly observations |
| Horizon | `H = 24` hourly values |
| Forecast origins | Every valid hour; also report horizon-wise metrics |
| Main split | Earliest 70% train, next 15% validation, latest 15% test |
| Model-selection metric | Validation MSE |
| Main reported metrics | MSE, MAE, RMSE, sMAPE, and R2 |
| Retrieval count | `k = 8` by default; test `1, 4, 8, 16` |
| Primary learned model | 24 direct `XGBRegressor` models, one per horizon |
| Fallback learned model | LightGBM only if XGBoost is unavailable or fails |
| Random seed | `42`; record all library and environment versions |
| Execution target | A Kaggle notebook with CPU support; GPU is optional |

The primary target is not a single value at `t+24`. For every forecast origin,
the expected target and prediction shape is `[24]`. Longer horizons and
multi-station modeling are extensions and must not delay the primary task.

### 1.2 Research-critical components

The final study requires all of the following:

1. PM2.5-only and context-aware non-retrieval baselines.
2. A leakage-safe historical time-series knowledge base.
3. Random, cosine, and hybrid retrieval comparisons.
4. Source-preserving event data aligned by information availability.
5. Transparent drift indicators and separate drift-period evaluation.
6. Evidence-grounded explanations linked to saved retrieval records.
7. A frozen time-series foundation-model baseline and a retrieval-augmented
   variant before retaining "Foundation Models" in the final paper title.
8. A reproducible ablation table generated from saved predictions.

### 1.3 Explicitly deferred work

The following are not part of the first implementation pass:

- training a TSFM from scratch;
- reproducing TimeRAF's 320-million-point training setup;
- an end-to-end dual-encoder retriever;
- full Channel Prompting inside a TSFM backbone;
- unrestricted LLM explanations;
- satellite AOD, a dashboard, or real-time deployment;
- multi-city or multi-pollutant forecasting;
- advanced online drift algorithms unless the core pipeline is complete.

### 1.4 Naming gate

The XGBoost retrieval pipeline is a **TimeRAF-inspired MVP**, not a reproduction
of TimeRAF and not by itself a foundation model. Before final submission:

- if the frozen TSFM and retrieval-augmented TSFM experiments run, the current
  title may be retained;
- if they do not run, change the title to **Event-TimeRAF: Event-Aware
  Retrieval-Augmented Forecasting for Explainable Air Quality Prediction Under
  Distribution Shift** and remove foundation-model claims from the abstract,
  methodology, results, and conclusion.

---

## 2. Data Readiness Gate

Implementation begins with an audit, not model training. The audit must save
source URLs or dataset identifiers, retrieval dates, licenses where available,
time coverage, units, duplicate counts, missingness, station counts, and event
coverage.

### 2.1 Source order

| Data | Preferred source | Fallback rule |
| --- | --- | --- |
| PM2.5 | US EPA AirData/AQS hourly parameter `88101` | Use parameter `88502` only as a separately documented sensitivity source; never merge parameter codes silently |
| Weather | NOAA NCEI Global Hourly/ISD observations | Open-Meteo may be used only as a documented weather fallback |
| Calendar | Deterministic US federal and California holiday features | Omit uncertain special-day flags and document the omission |
| Events | NOAA Storm Events; optional NOAA HMS fire/smoke records | Use a previously downloaded source-preserving cache with original fields |

Kaggle internet access cannot be assumed. Raw data should be downloaded once,
validated, and attached as a Kaggle Dataset. The pipeline must accept local input
paths and must not require a live API during the main experiment run.

### 2.2 Minimum data criteria

Proceed with the hourly primary task only if the selected period has:

- at least nine usable months, with twelve or more preferred;
- at least 70% observed hourly PM2.5 values before filling;
- known and consistent PM2.5 units;
- weather coverage sufficient to align at least 95% of usable PM2.5 hours;
- no dependence on interpolating PM2.5 targets;
- at least 180 overlapping event-data days, 30 event-active days, and two event
  categories with at least 20 retained records each for event-effect claims.

Long gaps are excluded through window validity checks. The pipeline must not
silently switch from hourly to daily frequency. If the PM2.5 gate fails, replace
the source or formally revise the task and paper. If only the event gate fails,
run event-free diagnostics but revise the event-aware claims before submission.

### 2.3 Canonical schemas

Air-quality observations:

```text
timestamp_utc, timestamp_local, site_id, latitude, longitude,
pm25_observed, pm25, pm25_filled, monitor_count
```

Weather observations:

```text
timestamp_utc, timestamp_local, temperature_c, dewpoint_c,
relative_humidity, precipitation_mm, wind_speed_ms,
wind_direction_deg, pressure_hpa
```

Event records:

```text
event_id, event_time, event_end, published_at,
availability_assumption, source_coverage_start, source_coverage_end,
coverage_basis, category, source, source_url, title, summary
```

`published_at` is the default information-availability timestamp. If it is not
available, the record must be marked and excluded from strict causal experiments
or handled in a clearly labeled sensitivity analysis.

NOAA Storm Events lacks a machine-readable publication timestamp. The implemented
MVP therefore records `published_at = event_time` and
`availability_assumption = event_start`; event-aware results from this source are
retrospective sensitivity results. The audit separately reports source-coverage
days, event-active days, category counts, and whether strict publication-time
availability is satisfied.

---

## 3. Repository Structure

The implementation will use this small structure:

```text
ASA/
  README.md
  requirements.txt
  plan.md
  structured_plan.md
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
```

The main notebook orchestrates the package and can run top to bottom on Kaggle.
The second notebook reads saved artifacts only and creates final tables and
figures. It must not retrain models.

### 3.1 Module ownership

| Module | Responsibility |
| --- | --- |
| `config.py` | Validate YAML settings, paths, seeds, and run identifiers |
| `data.py` | Load, validate, standardize, align, and audit all sources |
| `features.py` | Create past-only PM2.5, weather, calendar, and event features |
| `windows.py` | Build aligned `[N,168]` inputs, `[N,24]` targets, and metadata |
| `retrieval.py` | Build candidate pools, score candidates, retrieve top-k, and save evidence |
| `models.py` | Naive, XGBoost, retrieval-only, and frozen-TSFM model interfaces |
| `drift.py` | Fit training references and compute past-only drift indicators |
| `explain.py` | Produce deterministic explanations from stored evidence |
| `evaluation.py` | Splits, metrics, bootstrap intervals, ablations, and result tables |
| `plots.py` | EDA, forecast, retrieval, drift, ablation, and case-study figures |

### 3.2 Configuration contract

`configs/default.yaml` will contain at least:

```text
data paths and source metadata
timezone = America/Los_Angeles
study years = 2019..2024
EPA state/county codes = 06/037
EPA PM2.5 parameter = 88101
frequency = 1h
lookback = 168
horizon = 24
split ratios = [0.70, 0.15, 0.15]
maximum past-only fill gap
feature lists
event categories and keyword rules
retrieval method, k, and hybrid weights
drift windows and threshold quantile
XGBoost parameters
TSFM checkpoint and cache path
random seed
output root and run id
```

No experiment-critical constant should exist only in a notebook cell.

---

## 4. Artifact Contracts

The pipeline is complete only when it writes inspectable artifacts with stable
schemas.

| Artifact | Required content |
| --- | --- |
| `outputs/audit/data_audit.json` | Source metadata, units, coverage, gaps, duplicates, and readiness decisions |
| `data/processed/modeling_hourly.parquet` | One row per local hour with observed/filled flags and all causal features |
| `data/processed/window_metadata.parquet` | Window ID, origin, target range, split, validity, station, and missingness |
| `data/processed/window_arrays.npz` | PM2.5 lookbacks `[N,168]` and targets `[N,24]` |
| `data/knowledge_base/ts_kb_metadata.parquet` | Eligible training candidate metadata and normalization statistics |
| `data/knowledge_base/ts_kb_arrays.npz` | Candidate inputs, futures, and retrieval vectors |
| `data/knowledge_base/event_kb.parquet` | Deduplicated source-preserving event records |
| `outputs/evidence/retrieval_evidence.parquet` | Query, rank, candidate, all score components, and aligned candidate future |
| `outputs/predictions/predictions.parquet` | Model, origin, horizon, target time, actual, prediction, seed, and subset flags |
| `outputs/tables/metrics.csv` | Model and subset metrics overall and by horizon |
| `outputs/tables/ablation_results.csv` | Fixed model variants and confidence intervals |
| `outputs/logs/run_manifest.json` | Config hash, seed, dependency versions, data hashes, runtime, and artifact paths |

Predictions use long format: one row per model, forecast origin, and horizon.
All paper tables must be generated from these saved artifacts.

---

## 5. Leakage and Causality Rules

These rules are mandatory and will be tested:

1. Split boundaries are chronological and based on forecast origin.
2. Training-only statistics fit normalization, missing-value fallbacks, drift
   references, feature selection, and model hyperparameters.
3. PM2.5 labels are never interpolated. A window with a missing target value is
   invalid.
4. Input gaps may be filled only from information available at or before the
   forecast origin and only up to the configured short-gap limit. Save a filled
   indicator for every affected row.
5. Rolling and lag features at origin `t` use values no later than `t`.
6. Observed future weather is not used. The MVP uses weather through `t`; a later
   forecast-weather experiment must be separately labeled.
7. An event is usable only when `published_at <= t`.
8. Validation and test queries retrieve from the training knowledge base only.
9. For every query, a candidate is eligible only when its full future segment
   ends before the query origin. Self-matches and overlapping input/target windows
   are excluded.
10. Random, cosine, and hybrid retrieval use the same eligible candidate pool.
11. Validation selects hyperparameters and thresholds. Test data is evaluated
    once after choices are frozen.

The tests must include deliberately invalid timestamps and fail when any rule is
violated.

---

## 6. Feature and Window Design

### 6.1 Causal input features

PM2.5 features:

```text
lags: 1, 3, 6, 12, 24, 48, 168 hours
rolling mean: 3, 6, 12, 24, 72, 168 hours
rolling std/min/max: 24 and 168 hours
first differences: 1 and 24 hours
missing/filled indicators
```

Weather features:

```text
current and past summaries for temperature, humidity, precipitation,
wind speed/direction, pressure, and cloud cover
rain flag, low-wind flag, high-humidity flag, stagnation proxy
```

Calendar features:

```text
origin and known target-hour cyclical encodings, day of week, month, season,
Saturday/Sunday weekend, verified US federal and California holidays
```

Future calendar values are allowed because they are deterministic and known at
the forecast origin. Future observed weather and future event records are not.

Event features:

```text
counts by category over past 24, 72, and 168 hours
total counts, event-burst ratio, mean tone, source count,
and identifiers of the highest-ranked evidence records
```

Event categories begin with wildfire, smoke, high wind, excessive heat, heavy
rain, flood, dust, traffic, industrial activity, and policy. Rules must be
deterministic, versioned, and auditable.

### 6.2 Window construction

For each valid forecast origin `t`:

```text
X[t] = PM2.5 from t-167 through t            shape [168]
Y[t] = PM2.5 from t+1 through t+24           shape [24]
F[t] = causal structured features at t       shape [num_features]
C[t] = known calendar features for t+1:t+24  shape [24, num_calendar]
```

Split labels are assigned after target alignment. Boundary windows are retained
only if their complete target lies in the same split as the forecast origin.

---

## 7. Retrieval Design

### 7.1 Time-series candidate representation

Each candidate stores its 168-hour input, 24-hour observed future, original
input mean and standard deviation, structured context summaries, and temporal
metadata. Similarity uses per-window z-normalized inputs:

```text
x_norm = (x - mean(x)) / max(std(x), epsilon)
```

To transfer a candidate trajectory to the query scale:

```text
y_candidate_norm = (y_candidate - candidate_input_mean) / candidate_input_std
y_aligned = query_input_mean + query_input_std * y_candidate_norm
```

This prevents a candidate's absolute historical pollution level from being
copied blindly while retaining its future pattern.

### 7.2 Retrieval variants

| Variant | Definition |
| --- | --- |
| Random | Uniform sample of `k` eligible candidates using the recorded seed |
| Cosine | Top-k cosine similarity on normalized PM2.5 windows |
| Hybrid | Weighted score using time-series, weather, calendar, and event-context similarities |

The default hybrid score is:

```text
0.5 * time_series_similarity
+ 0.2 * weather_similarity
+ 0.1 * calendar_similarity
+ 0.2 * event_similarity
```

All components must be transformed to comparable `[0,1]` ranges using
training/validation rules. The stated weights remain the default. A small
validation-only weight search is an ablation, not part of the primary result.

Brute-force or scikit-learn nearest-neighbor search is the default. FAISS is
introduced only if measured retrieval time is a bottleneck.

### 7.3 Retrieval outputs

For every query, save:

```text
candidate IDs and timestamps
rank and total score
each component score
aligned 24-hour candidate future
uniform retrieved forecast
similarity-weighted retrieved forecast
mean/max similarity and candidate spread
```

Uniform averaging is primary because it follows the relevant TimeRAF ablation.
Similarity weighting is reported separately.

### 7.4 Event retrieval semantics

Event information has two roles:

1. Past event-count vectors influence hybrid ranking of historical windows.
2. Recent source records known at the forecast origin provide explanation
   evidence, ranked by location, category match, recency, and source metadata.

The MVP does not require text embeddings. No event may be manually invented,
and every quoted or paraphrased event must retain its source record.

---

## 8. Model Matrix

Each model consumes the same split and writes the same prediction schema.

| ID | Model | Required |
| --- | --- | --- |
| `M00` | Persistence: repeat `x_t` for all 24 horizons | Yes |
| `M01` | Daily seasonal naive: use values from the preceding 24-hour cycle | Yes |
| `M02` | Weekly seasonal naive: use values from the preceding 168-hour cycle | Yes |
| `M03` | Direct XGBoost with PM2.5 features | Yes |
| `M04` | Direct XGBoost with PM2.5, weather, and calendar | Yes |
| `M05` | Random-retrieval uniform trajectory | Yes |
| `M06` | Cosine-retrieval uniform trajectory | Yes |
| `M07` | XGBoost plus cosine-retrieval features | Yes |
| `M08` | XGBoost plus hybrid retrieval and event features, no drift | Yes |
| `M09` | Full Event-TimeRAF MVP including drift features | Yes |
| `M10` | Frozen TSFM zero-shot baseline | Publication gate |
| `M11` | Frozen TSFM plus retrieval augmentation | Publication gate |

For direct XGBoost, train one regressor for each horizon `h = 1..24`. Each model
uses origin-time causal features plus deterministic calendar features for its
target hour. Use one fixed, modest parameter set first. Only a small validation
search is allowed; do not launch a large optimization study before the full
pipeline works.

### 8.1 Frozen TSFM publication gate

Prefer a small TTM checkpoint because TimeRAF uses TTM-Base. During
implementation, first run a compatibility spike that verifies checkpoint access,
input length, forecast length, licenses, memory, and offline Kaggle caching. A
small Chronos-family checkpoint is the fallback if TTM cannot support the task.
The selected checkpoint must evaluate the same `L=168`, `H=24` task. Use
padding or masking only when officially supported and document it; do not change
the primary test windows merely to fit a checkpoint.

The TSFM experiment must include:

```text
frozen zero-shot TSFM forecast
uniform retrieved historical forecast
validation-selected convex fusion of those two forecasts
event-conditioned hybrid retrieval variant
```

Choose the convex weight from `{0, 0.25, 0.5, 0.75, 1}` on validation data. This
is a lightweight retrieval-augmented TSFM experiment, not TimeRAF's learned
Channel Prompting. Use that exact distinction in the paper.

---

## 9. Drift Design

The MVP uses an operational distribution-shift score, not a claim of perfect
concept-drift detection. Every component is computed from data available at the
forecast origin:

```text
recent mean shift against the training reference
recent variance shift against the training reference
retrieval-similarity drop
weather-vector shift
event-burst ratio
```

Fit component scaling on training data. Combine the components with fixed equal
weights first, then flag origins above the validation 90th percentile. Save the
component values and the reason with each flag. Evaluate normal and flagged
periods separately; do not claim drift robustness merely because the score was
computed.

Add ADWIN, Page-Hinkley, KS, or MMD only as an optional sensitivity analysis.

---

## 10. Explanation Design

Explanations are deterministic records generated from:

- forecast direction and magnitude;
- top local feature effects or SHAP values;
- weather and calendar flags;
- top retrieved historical cases;
- top recent event records;
- drift components and threshold status;
- uncertainty proxy from retrieved-trajectory spread and validation residuals.

Every explanation output must include machine-readable evidence IDs. A sentence
is omitted when its evidence is absent. The module must not infer causation,
invent an event, or use an unrestricted LLM to fill missing evidence.

---

## 11. Evaluation Protocol

### 11.1 Metric reporting

Report metrics in three views:

1. overall across all origin-horizon pairs;
2. separately for each horizon from 1 to 24;
3. macro average of the 24 horizon-level metrics.

MAPE may be shown as a diagnostic with an explicit epsilon but is not used for
model selection because it is unstable near zero. Report event-period,
drift-flagged, and normal-period metrics only when each subset has at least 50
forecast origins; otherwise present those cases descriptively.

### 11.2 Uncertainty and comparisons

Use paired moving-block bootstrap intervals over forecast origins for the main
test-set MSE and MAE differences. The block length and number of resamples must
be fixed in configuration. Save point estimates and intervals; avoid significance
claims when intervals are inconclusive.

### 11.3 Required ablations

```text
PM2.5 only vs weather/calendar
no retrieval vs random vs cosine retrieval
cosine vs hybrid event-conditioned retrieval
uniform vs similarity-weighted candidate aggregation
k in {1, 4, 8, 16}
full model without events
full model without drift features
full model with all components
frozen TSFM vs retrieval-augmented frozen TSFM
```

Knowledge-base size and learned fusion experiments are optional after these are
complete.

---

## 12. Execution Phases and Exit Gates

### Phase 0: Environment and skeleton

Create the repository structure, config loader, dependency file, logging, and a
Kaggle-compatible smoke notebook.

Exit gate: imports work in a clean session, config validation passes, output
directories are created, and the seed/version manifest is saved.

### Phase 1: Data acquisition and audit

Load or acquire source-preserving PM2.5, weather, calendar, and event data. Run
the readiness checks before feature generation.

Exit gate: `data_audit.json` records a pass/fail decision for every source and
the selected study interval is frozen.

### Phase 2: Preprocessing and windows

Build the hourly modeling table, causal features, aligned windows, chronological
splits, and leakage tests.

Exit gate: all array shapes and timestamps pass tests; missing targets are absent;
split and feature causality checks pass.

### Phase 3: Non-retrieval baselines

Run `M00` through `M04`, save predictions, metrics, and forecast plots.

Exit gate: all baselines use identical test rows; naive formulas pass unit tests;
XGBoost beats or is diagnostically compared with naive baselines without hiding
negative results.

### Phase 4: Time-series retrieval

Build the training KB and run random and cosine retrieval, aligned future
aggregation, `k` sensitivity, and retrieval evidence plots.

Exit gate: candidate eligibility tests pass, manual inspection of sampled queries
shows no future overlap, and `M05` through `M07` artifacts exist.

### Phase 5: Event and hybrid retrieval

Deduplicate/classify events, build past-window event vectors, attach event context
to historical candidates, and implement hybrid ranking.

Exit gate: all used records satisfy `published_at <= origin`, event coverage and
availability assumptions are reported, strict and retrospective-sensitivity
results are not conflated, and `M08` predictions and evidence exist.

### Phase 6: Drift and full MVP

Fit training drift references, choose the validation threshold, create flags,
and run `M09`.

Exit gate: component-level drift evidence is saved and normal/drift subset counts
are reported before their metrics are interpreted.

### Phase 7: Grounded explanations

Generate explanations for test predictions and select case studies using fixed
criteria such as highest event score, highest drift score, and representative
median-error cases.

Exit gate: every explanation sentence maps to stored evidence and no event or
causal claim is unsupported.

### Phase 8: Frozen TSFM gate

Run the compatibility spike, cache the selected checkpoint for Kaggle, then run
`M10` and `M11` or activate the naming fallback in Section 1.4.

Exit gate: TSFM predictions share the primary evaluation schema, or all
foundation-model wording is formally scheduled for removal.

### Phase 9: Final evaluation and ablations

Freeze all choices, run the test set once, compute intervals and ablations, and
generate final figures from saved outputs.

Exit gate: one run manifest reproduces every final table and figure; no table is
manually populated.

### Phase 10: Paper synchronization and packaging

Replace placeholders only with verified outputs, update method details to match
the code, discuss failed or negative results, state limitations, compile LaTeX,
and create the Overleaf zip.

Exit gate: every numerical claim traces to an artifact, references resolve, the
PDF compiles without warnings that affect content, and title claims satisfy the
foundation-model gate.

---

## 13. Verification Checklist

Before declaring implementation complete:

```text
[ ] Fresh-session imports and config validation pass.
[ ] Data audit and source metadata are saved.
[ ] Target arrays have shape [N, 24].
[ ] Chronological split and target boundaries are correct.
[ ] No target interpolation or future feature access occurs.
[ ] Retrieval candidates end before each query origin.
[ ] Event records were published by each query origin, or the run is explicitly
    labeled as a retrospective availability sensitivity.
[ ] Baseline, retrieval, full-MVP, and TSFM-gate predictions exist.
[ ] Metrics are available overall and by horizon.
[ ] Required ablations and bootstrap intervals are saved.
[ ] Explanations link to machine-readable evidence.
[ ] A run manifest records config, data hashes, versions, and seed.
[ ] Paper claims and title match the experiments actually completed.
```

---

## 14. Experiment Execution Order

The code and notebooks cover the following stages. Execute them in this order,
and stop at each checkpoint for the required audit or evidence review before
adding complexity:

```text
1. Create the skeleton, configuration, tests, and run manifest.
2. Implement and run the data audit.
3. Build the causal hourly table and 168-to-24 windows.
4. Run naive and direct-XGBoost baselines.
5. Build leakage-safe random and cosine retrieval.
6. Add event-conditioned hybrid retrieval.
7. Add drift indicators and grounded explanations.
8. Run the frozen-TSFM publication gate.
9. Freeze choices, run ablations, and update the paper from artifacts.
```

Do not begin Phase 5 until the Phase 4 retrieval evidence has been manually
inspected, and do not update the paper with results until Phase 9 is frozen.
