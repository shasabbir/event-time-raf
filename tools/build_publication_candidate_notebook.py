from __future__ import annotations

from pathlib import Path
import re
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_event_timeraf_kaggle_pipeline.ipynb"


def code(value: str) -> str:
    return dedent(value).strip() + "\n"


nb = nbformat.read(NOTEBOOK, as_version=4)
nb.cells = [
    cell for cell in nb.cells
    if "SITE_LEVEL_SENSITIVITY_ARM" not in cell.source
    and not cell.source.startswith("## 6. Site-level target sensitivity")
]
nb.cells[0].source = code(
    """
    # TRACE-RAF: Publication-Candidate Kaggle Pipeline

    This experiment log runs the complete study from official cached sources. It uses a dense,
    leakage-safe historical knowledge base, explicit event-conditioned retrieval, two-sided
    drift evidence, real component ablations, a frozen Chronos baseline, and manifest-backed
    outputs. The proposed TRACE-RAF model adds trust-gated out-of-fold residual analog correction
    to the strongest supervised ensemble. Numerical paper claims must be updated only from the ZIP
    produced by the final cell.
    """
)
nb.cells[1].source = code(
    """
    # Kaggle publication setup. Pin PatchTST to the stable Transformers 4.x API.
    import importlib.metadata
    import importlib.util
    import subprocess
    import sys

    requirements = []
    if importlib.util.find_spec('chronos') is None:
        requirements.append('chronos-forecasting>=2.1,<3')
    transformers_version = None
    try:
        transformers_version = importlib.metadata.version('transformers')
    except importlib.metadata.PackageNotFoundError:
        pass
    if transformers_version is None or int(transformers_version.split('.')[0]) >= 5:
        requirements.append('transformers>=4.49,<5')
    if importlib.util.find_spec('lightgbm') is None:
        requirements.append('lightgbm>=4,<5')
    if requirements:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *requirements])
    """
)

setup = nb.cells[2].source
setup = re.sub(r"(?:DirectRidgeForecaster, ){2,}", "DirectRidgeForecaster, ", setup)
setup = re.sub(
    r"(?:RUN_STRIDE_MODEL_SWEEP = True\n){2,}",
    "RUN_STRIDE_MODEL_SWEEP = True\n",
    setup,
)
setup = setup.replace(
    "from event_timeraf.features import build_modeling_table",
    "from event_timeraf.features import EVENT_CATEGORIES, build_modeling_table",
)
setup = setup.replace(
    "load_storm_events_cache, prepare_epa_pm25,",
    "load_storm_events_cache, prepare_epa_pm25, prepare_epa_site_pm25,",
)
setup = setup.replace(
    "build_data_audit, download_epa_pm25,",
    "build_data_audit, download_epa_pm25, haversine_km,",
)
while "haversine_km, haversine_km," in setup:
    setup = setup.replace("haversine_km, haversine_km,", "haversine_km,")
while "prepare_epa_site_pm25, prepare_epa_site_pm25," in setup:
    setup = setup.replace(
        "prepare_epa_site_pm25, prepare_epa_site_pm25,", "prepare_epa_site_pm25,"
    )
if "DirectRidgeForecaster" not in setup:
    setup = setup.replace(
        "DirectXGBForecaster, choose_fusion_weight, chronos_forecast,",
        "DirectRidgeForecaster, DirectXGBForecaster, choose_fusion_weight, chronos_forecast,",
    )
if "hour_month_climatology_forecast" not in setup:
    setup = setup.replace(
        "daily_seasonal_forecast, fuse_forecasts, origin_feature_matrix,",
        "daily_seasonal_forecast, fuse_forecasts, hour_month_climatology_forecast, origin_feature_matrix,",
    )
if "KSTwoSampleDriftDetector" not in setup:
    setup = setup.replace(
        "from event_timeraf.drift import DriftDetector, drift_evidence_frame",
        "from event_timeraf.drift import DriftDetector, KSTwoSampleDriftDetector, drift_evidence_frame",
    )
if "RUN_STRIDE_MODEL_SWEEP = True" not in setup:
    setup = setup.replace(
        "RETRIEVAL_EVIDENCE_REVIEWED = True",
        "RETRIEVAL_EVIDENCE_REVIEWED = True\nRUN_STRIDE_MODEL_SWEEP = True",
    )
if "RUN_JOURNAL_BASELINES = True" not in setup:
    setup = setup.replace(
        "RUN_STRIDE_MODEL_SWEEP = True",
        "RUN_STRIDE_MODEL_SWEEP = True\nRUN_JOURNAL_BASELINES = True",
    )
if "Clear stale writable outputs before a publication run" not in setup:
    setup = setup.replace(
        "# Final-publication profile.",
        """# Clear stale writable outputs before a publication run. Raw caches are outside this path.
if Path('/kaggle/working').exists() and cfg.paths.outputs.resolve().is_relative_to(Path('/kaggle/working').resolve()):
    reset_directory(cfg.paths.outputs)
    cfg.paths.create()

# Final-publication profile.""",
    )
setup = re.sub(
    r"from event_timeraf\.evaluation import \(.*?\)\n",
    """from event_timeraf.evaluation import (
    build_event_period_flags, diebold_mariano_hac, exceedance_metrics,
    holm_adjust_pvalues, horizon_skill_table, interval_metrics, log_scale_metrics,
    metric_values, metrics_table, quantile_forecast_metrics,
    paired_block_bootstrap_loss_difference, paired_masked_block_bootstrap_loss_difference,
    predictions_long,
)
""",
    setup,
    flags=re.DOTALL,
)
setup = re.sub(
    r"from event_timeraf\.explain import .*\n",
    "from event_timeraf.explain import generate_explanations, grouped_feature_perturbation, xgb_local_contributions\n",
    setup,
)
setup = re.sub(
    r"from event_timeraf\.models import \(.*?\)\n",
    """from event_timeraf.models import (
    ConvexForecastEnsemble, DirectLightGBMForecaster, DirectRidgeForecaster,
    DirectXGBForecaster, NeuralWindowForecaster, SelectiveResidualGate,
    choose_fusion_weight, chronos_quantile_forecast, daily_seasonal_forecast,
    fuse_forecasts, hour_month_climatology_forecast, origin_feature_matrix,
    persistence_forecast, residual_gate_features, weekly_seasonal_forecast,
)
""",
    setup,
    flags=re.DOTALL,
)
setup = re.sub(
    r"from event_timeraf\.retrieval import .*\n",
    "from event_timeraf.retrieval import HistoricalRetriever, build_knowledge_base, residual_correction_from_retrieval\n",
    setup,
)
setup = re.sub(
    r"from event_timeraf\.windows import .*\n",
    "from event_timeraf.windows import build_window_dataset, window_attrition_table\n",
    setup,
)
setup = setup.replace(
    "for directory in ('configs', 'src'):\n            target = writable / directory",
    "for directory in ('configs', 'src', 'notebooks', 'tests'):\n            target = writable / directory",
)
nb.cells[2].source = setup

nb.cells[6].source = code(
    """
    modeling = build_modeling_table(pm25, weather, events, cfg)
    modeling_path = cfg.paths.processed / 'modeling_hourly.parquet'
    modeling.to_parquet(modeling_path, index=False)

    dataset = build_window_dataset(modeling, cfg)
    window_arrays_path = cfg.paths.processed / 'window_arrays.npz'
    window_metadata_path = cfg.paths.processed / 'window_metadata.parquet'
    dataset.save(window_arrays_path, window_metadata_path)
    train = dataset.subset('train')
    validation = dataset.subset('validation')
    test = dataset.subset('test')
    attrition = window_attrition_table(modeling, dataset, cfg, RUN_ID)
    attrition_path = cfg.paths.outputs / 'tables' / 'window_origin_attrition.csv'
    attrition.to_csv(attrition_path, index=False)
    print({'train': len(train.x), 'validation': len(validation.x), 'test': len(test.x)})
    print('X/Y shapes:', dataset.x.shape, dataset.y.shape)
    display(attrition)
    if min(len(train.x), len(validation.x), len(test.x)) == 0:
        raise RuntimeError('At least one chronological split is empty after validity filtering.')
    """
)

nb.cells[7].source = code(
    """
    ## 3. Dense leakage-safe and event-conditioned retrieval

    Candidate windows may overlap one another. For every query, the complete candidate target
    must still end before the query lookback begins. The primary 24-hour stride is evaluated
    against 6-hour and 1-hour alternatives without changing this embargo.
    """
)
nb.cells[8].source = code(
    """
    knowledge_base = build_knowledge_base(dataset, cfg)
    kb_arrays_path = cfg.paths.knowledge_base / 'ts_kb_arrays.npz'
    kb_metadata_path = cfg.paths.knowledge_base / 'ts_kb_metadata.parquet'
    knowledge_base.save(kb_arrays_path, kb_metadata_path)
    retriever = HistoricalRetriever(knowledge_base, cfg)

    def retrieve_splits(method, event_weight=None):
        return {
            split: retriever.retrieve(values, method=method, event_weight=event_weight)
            for split, values in {'train': train, 'validation': validation, 'test': test}.items()
        }

    retrievals = {
        method: retrieve_splits(method)
        for method in ('random', 'cosine', 'calendar', 'hybrid_no_event', 'event_conditioned')
    }
    random_train, random_validation, random_test = retrievals['random'].values()
    cosine_train, cosine_validation, cosine_test = retrievals['cosine'].values()
    calendar_train, calendar_validation, calendar_test = retrievals['calendar'].values()
    no_event_train, no_event_validation, no_event_test = retrievals['hybrid_no_event'].values()
    event_train, event_validation, event_test = retrievals['event_conditioned'].values()

    required_evaluation_results = [
        method[split]
        for method in retrievals.values()
        for split in ('validation', 'test')
    ]
    if not all(result.valid_mask.all() for result in required_evaluation_results):
        raise RuntimeError('A validation or test query has no causally eligible retrieval candidate.')
    training_validity = pd.DataFrame([
        {
            'method': method,
            'valid_training_queries': int(results['train'].valid_mask.sum()),
            'excluded_early_training_queries': int((~results['train'].valid_mask).sum()),
        }
        for method, results in retrievals.items()
    ])
    if training_validity['valid_training_queries'].min() == 0:
        raise RuntimeError('No causally retrievable training queries remain for a retrieval method.')
    evidence = pd.concat(
        [retrievals[method]['test'].evidence for method in retrievals], ignore_index=True
    )
    evidence.insert(0, 'run_id', RUN_ID)
    retrieval_evidence_path = cfg.paths.outputs / 'evidence' / 'retrieval_evidence.parquet'
    evidence.to_parquet(retrieval_evidence_path, index=False)
    review_sample = (
        evidence.groupby('method', group_keys=False)
        .sample(n=min(5, evidence.groupby('method').size().min()), random_state=cfg.seed)
        .sort_values(['method', 'query_origin', 'rank'])
    )
    retrieval_review_path = cfg.paths.outputs / 'evidence' / 'retrieval_review_sample.csv'
    review_sample.to_csv(retrieval_review_path, index=False)
    print({'knowledge_base_candidates': len(knowledge_base.metadata),
           'primary_stride_hours': cfg.retrieval.kb_stride_hours})
    display(training_validity)
    display(review_sample.drop(columns=['aligned_future']).head(20))
    if FINAL_EXPERIMENT and not RETRIEVAL_EVIDENCE_REVIEWED:
        raise RuntimeError('Review retrieval_review_sample.csv before the final run.')
    """
)

nb.cells[9].source = code(
    """
    ## 4. Baselines, Event-TimeRAF variants, and TRACE-RAF

    All learned variants use the same chronological splits. M08 uses the event-conditioned
    retriever, while A00 removes event inputs and event-conditioned retrieval under an otherwise
    comparable direct-model design. TRACE-RAF uses expanding-window out-of-fold residual memory,
    a convex XGBoost-LightGBM base, and a validation-trained gate with a zero-correction fallback.
    """
)
nb.cells[10].source = code(
    """
    predictions = {
        'C00_hour_month_climatology': hour_month_climatology_forecast(train, test, cfg.timezone),
        'C02_calendar_retrieval': calendar_test.prediction,
        'C03_event_conditioned_retrieval': event_test.prediction,
        'M00_persistence': persistence_forecast(test.x, cfg.forecast.horizon),
        'M01_daily_seasonal': daily_seasonal_forecast(test.x, cfg.forecast.horizon),
        'M02_weekly_seasonal': weekly_seasonal_forecast(test.x, cfg.forecast.horizon),
        'M05_random_retrieval': random_test.prediction,
        'M06_cosine_retrieval': cosine_test.prediction,
    }
    validation_predictions = {
        'C00_hour_month_climatology': hour_month_climatology_forecast(train, validation, cfg.timezone),
        'C02_calendar_retrieval': calendar_validation.prediction,
        'C03_event_conditioned_retrieval': event_validation.prediction,
        'M00_persistence': persistence_forecast(validation.x, cfg.forecast.horizon),
        'M01_daily_seasonal': daily_seasonal_forecast(validation.x, cfg.forecast.horizon),
        'M02_weekly_seasonal': weekly_seasonal_forecast(validation.x, cfg.forecast.horizon),
    }

    pm_train, pm_names = origin_feature_matrix(train, ('pm25_',))
    pm_validation, _ = origin_feature_matrix(validation, ('pm25_',))
    pm_test, _ = origin_feature_matrix(test, ('pm25_',))
    m03 = DirectXGBForecaster(cfg, include_future_calendar=False).fit(
        pm_train, train.future_calendar, train.y, pm_names, train.calendar_names
    )
    validation_predictions['M03_xgb_pm25'] = m03.predict(pm_validation, validation.future_calendar)
    predictions['M03_xgb_pm25'] = m03.predict(pm_test, test.future_calendar)

    context_prefixes = ('pm25_', 'weather_', 'cal_')
    context_train, context_names = origin_feature_matrix(train, context_prefixes)
    context_validation, _ = origin_feature_matrix(validation, context_prefixes)
    context_test, _ = origin_feature_matrix(test, context_prefixes)
    m04 = DirectXGBForecaster(cfg).fit(
        context_train, train.future_calendar, train.y, context_names, train.calendar_names
    )
    validation_predictions['M04_xgb_context'] = m04.predict(context_validation, validation.future_calendar)
    predictions['M04_xgb_context'] = m04.predict(context_test, test.future_calendar)
    ridge = DirectRidgeForecaster(cfg).fit(
        context_train, train.future_calendar, train.y, context_names, train.calendar_names
    )
    validation_predictions['C01_ridge_context'] = ridge.predict(context_validation, validation.future_calendar)
    predictions['C01_ridge_context'] = ridge.predict(context_test, test.future_calendar)
    if RUN_JOURNAL_BASELINES:
        lightgbm = DirectLightGBMForecaster(cfg).fit(
            context_train, train.future_calendar, train.y, context_names, train.calendar_names
        )
        validation_predictions['C05_lightgbm_context'] = lightgbm.predict(
            context_validation, validation.future_calendar
        )
        predictions['C05_lightgbm_context'] = lightgbm.predict(context_test, test.future_calendar)

        dlinear = NeuralWindowForecaster(cfg, 'dlinear').fit(train, validation)
        lstm = NeuralWindowForecaster(cfg, 'lstm').fit(train, validation)
        patchtst = NeuralWindowForecaster(cfg, 'patchtst').fit(train, validation)
        validation_predictions['B00_dlinear'] = dlinear.predict(validation)
        validation_predictions['B02_lstm'] = lstm.predict(validation)
        validation_predictions['B01_patchtst'] = patchtst.predict(validation)
        predictions['B00_dlinear'] = dlinear.predict(test)
        predictions['B02_lstm'] = lstm.predict(test)
        predictions['B01_patchtst'] = patchtst.predict(test)
        neural_training_path = cfg.paths.outputs / 'tables' / 'neural_baseline_training.csv'
        pd.concat([
            dlinear.history_frame(RUN_ID), lstm.history_frame(RUN_ID), patchtst.history_frame(RUN_ID)
        ], ignore_index=True).to_csv(neural_training_path, index=False)
        dlinear_path = cfg.paths.outputs / 'models' / 'B00_dlinear.pt'
        lstm_path = cfg.paths.outputs / 'models' / 'B02_lstm.pt'
        patchtst_path = cfg.paths.outputs / 'models' / 'B01_patchtst.pt'
        dlinear.save(dlinear_path)
        lstm.save(lstm_path)
        patchtst.save(patchtst_path)
    elif FINAL_EXPERIMENT:
        raise RuntimeError('Final publication mode requires DLinear, LSTM, PatchTST, and LightGBM baselines.')

    # Expanding-window predictions provide genuinely out-of-fold residuals for
    # TRACE-RAF's historical memory. Every fit block is embargoed before the
    # first query input in the following prediction block.
    oof_xgb = np.full_like(train.y, np.nan, dtype=np.float32)
    oof_lightgbm = np.full_like(train.y, np.nan, dtype=np.float32)
    oof_rows = []
    train_target_end = pd.to_datetime(train.metadata['target_end'], utc=True)
    boundaries = cfg.selective_residual.oof_boundaries
    for start_fraction, end_fraction in zip(boundaries[:-1], boundaries[1:]):
        block_start = int(len(train.x) * start_fraction)
        block_end = int(len(train.x) * end_fraction)
        block_input_start = pd.Timestamp(train.metadata.iloc[block_start]['input_start'])
        fit_mask = (train_target_end < block_input_start).to_numpy()
        fit_indices = np.flatnonzero(fit_mask)
        block_indices = np.arange(block_start, block_end)
        if len(fit_indices) < 100 or len(block_indices) == 0:
            raise RuntimeError('TRACE-RAF OOF block is too small after temporal embargo.')
        fold_xgb = DirectXGBForecaster(cfg).fit(
            context_train[fit_indices], train.future_calendar[fit_indices],
            train.y[fit_indices], context_names, train.calendar_names,
        )
        fold_lightgbm = DirectLightGBMForecaster(cfg).fit(
            context_train[fit_indices], train.future_calendar[fit_indices],
            train.y[fit_indices], context_names, train.calendar_names,
        )
        oof_xgb[block_indices] = fold_xgb.predict(
            context_train[block_indices], train.future_calendar[block_indices]
        )
        oof_lightgbm[block_indices] = fold_lightgbm.predict(
            context_train[block_indices], train.future_calendar[block_indices]
        )
        oof_rows.append({
            'run_id': RUN_ID, 'block_start_fraction': start_fraction,
            'block_end_fraction': end_fraction, 'fit_origins': len(fit_indices),
            'predicted_origins': len(block_indices),
            'fit_target_end_max': train_target_end.iloc[fit_indices].max(),
            'prediction_input_start_min': block_input_start,
            'embargo_passed': bool(train_target_end.iloc[fit_indices].max() < block_input_start),
        })
    oof_valid = np.isfinite(oof_xgb).all(axis=1) & np.isfinite(oof_lightgbm).all(axis=1)
    context_ensemble = ConvexForecastEnsemble(
        'C05_lightgbm_context', 'M04_xgb_context'
    ).fit(train.y[oof_valid], oof_lightgbm[oof_valid], oof_xgb[oof_valid])
    validation_predictions['C06_context_ensemble'] = context_ensemble.predict(
        validation_predictions['C05_lightgbm_context'],
        validation_predictions['M04_xgb_context'],
    )
    predictions['C06_context_ensemble'] = context_ensemble.predict(
        predictions['C05_lightgbm_context'], predictions['M04_xgb_context']
    )
    oof_ensemble = np.full_like(train.y, np.nan, dtype=np.float32)
    oof_ensemble[oof_valid] = context_ensemble.predict(
        oof_lightgbm[oof_valid], oof_xgb[oof_valid]
    )
    trace_oof_audit_path = cfg.paths.outputs / 'tables' / 'trace_raf_oof_audit.csv'
    pd.DataFrame(oof_rows).to_csv(trace_oof_audit_path, index=False)
    if not pd.DataFrame(oof_rows)['embargo_passed'].all():
        raise RuntimeError('TRACE-RAF OOF temporal embargo failed.')

    def fit_retrieval_xgb(train_result, validation_result, test_result, prefixes, label):
        names = train_result.feature_names(label)
        train_matrix, model_names = origin_feature_matrix(train, prefixes, train_result.as_features(), names)
        validation_matrix, _ = origin_feature_matrix(validation, prefixes, validation_result.as_features(), names)
        test_matrix, _ = origin_feature_matrix(test, prefixes, test_result.as_features(), names)
        mask = train_result.valid_mask
        model = DirectXGBForecaster(cfg).fit(
            train_matrix[mask], train.future_calendar[mask], train.y[mask],
            model_names, train.calendar_names,
        )
        return model, train_matrix, validation_matrix, test_matrix, model_names

    m07, m07_train, m07_validation, m07_test, m07_names = fit_retrieval_xgb(
        cosine_train, cosine_validation, cosine_test, context_prefixes, 'cosine_retrieval'
    )
    validation_predictions['M07_xgb_cosine'] = m07.predict(m07_validation, validation.future_calendar)
    predictions['M07_xgb_cosine'] = m07.predict(m07_test, test.future_calendar)

    full_prefixes = ('pm25_', 'weather_', 'cal_', 'event_')
    full_train, full_names = origin_feature_matrix(train, full_prefixes)
    full_validation, _ = origin_feature_matrix(validation, full_prefixes)
    full_test, _ = origin_feature_matrix(test, full_prefixes)
    c04 = DirectXGBForecaster(cfg).fit(
        full_train, train.future_calendar, train.y, full_names, train.calendar_names
    )
    validation_predictions['C04_xgb_context_event'] = c04.predict(
        full_validation, validation.future_calendar
    )
    predictions['C04_xgb_context_event'] = c04.predict(full_test, test.future_calendar)

    event_columns = [index for index, name in enumerate(full_names) if name.startswith('event_')]
    if not event_columns:
        raise RuntimeError('Matched event-feature control requires event columns.')
    def matched_event_placebo(matrix, seed):
        changed = matrix.copy()
        order = np.random.default_rng(seed).permutation(len(changed))
        changed[:, event_columns] = changed[order][:, event_columns]
        return changed

    placebo_train = matched_event_placebo(full_train, cfg.seed + 101)
    placebo_validation = matched_event_placebo(full_validation, cfg.seed + 102)
    placebo_test = matched_event_placebo(full_test, cfg.seed + 103)
    a02 = DirectXGBForecaster(cfg).fit(
        placebo_train, train.future_calendar, train.y, full_names, train.calendar_names
    )
    validation_predictions['A02_xgb_matched_event_placebo'] = a02.predict(
        placebo_validation, validation.future_calendar
    )
    predictions['A02_xgb_matched_event_placebo'] = a02.predict(
        placebo_test, test.future_calendar
    )
    feature_control_design_path = cfg.paths.outputs / 'tables' / 'feature_count_control_design.csv'
    pd.DataFrame([
        {
            'run_id': RUN_ID, 'model': 'M04_xgb_context',
            'origin_feature_count': len(context_names),
            'future_calendar_feature_count': len(train.calendar_names),
            'event_feature_count': 0, 'control': 'lower-dimensional context baseline',
        },
        {
            'run_id': RUN_ID, 'model': 'C04_xgb_context_event',
            'origin_feature_count': len(full_names),
            'future_calendar_feature_count': len(train.calendar_names),
            'event_feature_count': len(event_columns), 'control': 'observed event features',
        },
        {
            'run_id': RUN_ID, 'model': 'A02_xgb_matched_event_placebo',
            'origin_feature_count': len(full_names),
            'future_calendar_feature_count': len(train.calendar_names),
            'event_feature_count': len(event_columns),
            'control': 'split-local joint row permutation of event features',
        },
    ]).to_csv(feature_control_design_path, index=False)

    m08, m08_train, m08_validation, m08_test, m08_names = fit_retrieval_xgb(
        event_train, event_validation, event_test, full_prefixes, 'event_conditioned_retrieval'
    )
    validation_predictions['M08_event_timeraf_no_drift'] = m08.predict(
        m08_validation, validation.future_calendar
    )
    predictions['M08_event_timeraf_no_drift'] = m08.predict(m08_test, test.future_calendar)

    random_model, _, random_validation_matrix, random_test_matrix, _ = fit_retrieval_xgb(
        random_train, random_validation, random_test, context_prefixes, 'random_retrieval'
    )
    validation_predictions['A01_xgb_random_retrieval'] = random_model.predict(
        random_validation_matrix, validation.future_calendar
    )
    predictions['A01_xgb_random_retrieval'] = random_model.predict(
        random_test_matrix, test.future_calendar
    )
    """
)

nb.cells[11].source = code(
    """
    drift_detector = DriftDetector(cfg).fit_reference(train, event_train.mean_similarity)
    drift_detector.calibrate(validation, event_validation.mean_similarity)
    drift_train = drift_detector.transform(train, event_train.mean_similarity)
    drift_validation = drift_detector.transform(validation, event_validation.mean_similarity)
    drift_test = drift_detector.transform(test, event_test.mean_similarity)
    drift_names = [f'drift_{name}' for name in drift_test.component_names] + ['drift_score']
    train_extra = np.column_stack([event_train.as_features(), drift_train.components, drift_train.score])
    validation_extra = np.column_stack([
        event_validation.as_features(), drift_validation.components, drift_validation.score
    ])
    test_extra = np.column_stack([event_test.as_features(), drift_test.components, drift_test.score])
    m09_extra_names = event_train.feature_names('event_conditioned_retrieval') + drift_names
    m09_train, m09_names = origin_feature_matrix(train, full_prefixes, train_extra, m09_extra_names)
    m09_validation, _ = origin_feature_matrix(validation, full_prefixes, validation_extra, m09_extra_names)
    m09_test, _ = origin_feature_matrix(test, full_prefixes, test_extra, m09_extra_names)
    m09 = DirectXGBForecaster(cfg).fit(
        m09_train[event_train.valid_mask], train.future_calendar[event_train.valid_mask],
        train.y[event_train.valid_mask], m09_names, train.calendar_names,
    )
    validation_predictions['M09_event_timeraf_full'] = m09.predict(
        m09_validation, validation.future_calendar
    )
    predictions['M09_event_timeraf_full'] = m09.predict(m09_test, test.future_calendar)

    no_event_detector = DriftDetector(cfg, include_event_component=False).fit_reference(
        train, no_event_train.mean_similarity
    )
    no_event_detector.calibrate(validation, no_event_validation.mean_similarity)
    no_event_drift_train = no_event_detector.transform(train, no_event_train.mean_similarity)
    no_event_drift_validation = no_event_detector.transform(
        validation, no_event_validation.mean_similarity
    )
    no_event_drift_test = no_event_detector.transform(test, no_event_test.mean_similarity)

    kb_position_by_window = pd.Series(
        np.arange(len(train.metadata)), index=train.metadata['window_id'].astype(str)
    )
    kb_train_positions = kb_position_by_window.reindex(
        knowledge_base.metadata['window_id'].astype(str)
    ).to_numpy(dtype=int)
    candidate_oof_prediction = oof_ensemble[kb_train_positions]
    candidate_residuals = knowledge_base.y - candidate_oof_prediction
    residual_candidate_mask = np.isfinite(candidate_residuals).all(axis=1)
    if residual_candidate_mask.sum() < max(100, cfg.retrieval.k):
        raise RuntimeError('TRACE-RAF has too few out-of-fold residual candidates.')
    clip_quantile = cfg.selective_residual.residual_clip_quantile
    residual_clip_lower = np.quantile(
        candidate_residuals[residual_candidate_mask], clip_quantile, axis=0
    )
    residual_clip_upper = np.quantile(
        candidate_residuals[residual_candidate_mask], 1.0 - clip_quantile, axis=0
    )
    candidate_residuals[residual_candidate_mask] = np.clip(
        candidate_residuals[residual_candidate_mask],
        residual_clip_lower, residual_clip_upper,
    )

    event_residual_validation = retriever.retrieve(
        validation, method='event_conditioned', candidate_mask=residual_candidate_mask
    )
    event_residual_test = retriever.retrieve(
        test, method='event_conditioned', candidate_mask=residual_candidate_mask
    )
    no_event_residual_validation = retriever.retrieve(
        validation, method='hybrid_no_event', candidate_mask=residual_candidate_mask
    )
    no_event_residual_test = retriever.retrieve(
        test, method='hybrid_no_event', candidate_mask=residual_candidate_mask
    )
    event_correction_validation = residual_correction_from_retrieval(
        knowledge_base, validation, event_residual_validation, candidate_residuals,
        cfg.retrieval.epsilon,
    )
    event_correction_test = residual_correction_from_retrieval(
        knowledge_base, test, event_residual_test, candidate_residuals,
        cfg.retrieval.epsilon,
    )
    no_event_correction_validation = residual_correction_from_retrieval(
        knowledge_base, validation, no_event_residual_validation, candidate_residuals,
        cfg.retrieval.epsilon,
    )
    no_event_correction_test = residual_correction_from_retrieval(
        knowledge_base, test, no_event_residual_test, candidate_residuals,
        cfg.retrieval.epsilon,
    )
    correction_results = (
        event_correction_validation, event_correction_test,
        no_event_correction_validation, no_event_correction_test,
    )
    if not all(result.valid_mask.all() for result in correction_results):
        raise RuntimeError('TRACE-RAF residual correction is incomplete.')

    validation_base = validation_predictions['C06_context_ensemble']
    test_base = predictions['C06_context_ensemble']
    validation_disagreement = np.abs(
        validation_predictions['C05_lightgbm_context']
        - validation_predictions['M04_xgb_context']
    )
    test_disagreement = np.abs(
        predictions['C05_lightgbm_context'] - predictions['M04_xgb_context']
    )

    def trace_features(base, correction_result, retrieval_result, disagreement, drift):
        return residual_gate_features(
            base, correction_result.correction, correction_result.spread,
            retrieval_result.mean_similarity, retrieval_result.max_similarity,
            correction_result.candidate_count, retrieval_result.eligible_candidate_count,
            retrieval_result.selected_event_fraction,
            retrieval_result.event_conditioning_applied,
            disagreement, drift.score, drift.flag,
        )

    trace_validation_features, trace_feature_names = trace_features(
        validation_base, event_correction_validation, event_residual_validation,
        validation_disagreement, drift_validation,
    )
    trace_test_features, _ = trace_features(
        test_base, event_correction_test, event_residual_test,
        test_disagreement, drift_test,
    )
    trace_gate = SelectiveResidualGate(cfg).fit(
        validation.y, validation_base, event_correction_validation.correction,
        trace_validation_features, trace_feature_names,
    )
    validation_predictions['M13_trace_raf'] = trace_gate.predict(
        validation_base, event_correction_validation.correction, trace_validation_features
    )
    predictions['M13_trace_raf'] = trace_gate.predict(
        test_base, event_correction_test.correction, trace_test_features
    )

    no_event_validation_features, no_event_trace_names = trace_features(
        validation_base, no_event_correction_validation, no_event_residual_validation,
        validation_disagreement, no_event_drift_validation,
    )
    no_event_test_features, _ = trace_features(
        test_base, no_event_correction_test, no_event_residual_test,
        test_disagreement, no_event_drift_test,
    )
    trace_no_event_gate = SelectiveResidualGate(cfg).fit(
        validation.y, validation_base, no_event_correction_validation.correction,
        no_event_validation_features, no_event_trace_names,
    )
    validation_predictions['A03_trace_raf_no_event'] = trace_no_event_gate.predict(
        validation_base, no_event_correction_validation.correction,
        no_event_validation_features,
    )
    predictions['A03_trace_raf_no_event'] = trace_no_event_gate.predict(
        test_base, no_event_correction_test.correction, no_event_test_features
    )

    trace_selection_rows = []
    for model_name, gate in {
        'M13_trace_raf': trace_gate,
        'A03_trace_raf_no_event': trace_no_event_gate,
    }.items():
        for strength, validation_mse in gate.selection_scores.items():
            trace_selection_rows.append({
                'run_id': RUN_ID, 'model': model_name, 'gate_strength': strength,
                'selection_mse': validation_mse,
                'selected_on_validation': strength == gate.selected_strength,
                'residual_candidate_count': int(residual_candidate_mask.sum()),
            })
    trace_gate_selection_path = cfg.paths.outputs / 'tables' / 'trace_raf_gate_selection.csv'
    pd.DataFrame(trace_selection_rows).to_csv(trace_gate_selection_path, index=False)
    trace_design_path = cfg.paths.outputs / 'tables' / 'trace_raf_design.csv'
    pd.DataFrame([{
        'run_id': RUN_ID,
        'model': 'M13_trace_raf',
        'expanded_name': 'Trust-gated Residual Analog Correction for Event-aware Retrieval-Augmented Forecasting',
        'base_weights_json': json.dumps(context_ensemble.weights(), sort_keys=True),
        'oof_boundaries_json': json.dumps(cfg.selective_residual.oof_boundaries),
        'residual_candidate_count': int(residual_candidate_mask.sum()),
        'residual_clip_quantile': clip_quantile,
        'retrieval_k': cfg.retrieval.k,
        'gate_fit_fraction': cfg.selective_residual.gate_fit_fraction,
        'gate_max_depth': cfg.selective_residual.gate_max_depth,
        'gate_max_iter': cfg.selective_residual.gate_max_iter,
        'gate_learning_rate': cfg.selective_residual.gate_learning_rate,
        'gate_l2_regularization': cfg.selective_residual.gate_l2_regularization,
        'selected_gate_strength': trace_gate.selected_strength,
        'test_used_for_selection': False,
    }]).to_csv(trace_design_path, index=False)
    trace_predictions_path = cfg.paths.outputs / 'predictions' / 'trace_raf_predictions.npz'
    np.savez_compressed(
        trace_predictions_path,
        validation_actual=validation.y, test_actual=test.y,
        validation_base=validation_base, test_base=test_base,
        event_validation_correction=event_correction_validation.correction,
        event_test_correction=event_correction_test.correction,
        no_event_validation_correction=no_event_correction_validation.correction,
        no_event_test_correction=no_event_correction_test.correction,
        validation_gate=trace_gate.gate_values(trace_validation_features),
        test_gate=trace_gate.gate_values(trace_test_features),
        M13_validation=validation_predictions['M13_trace_raf'],
        M13_test=predictions['M13_trace_raf'],
        A03_validation=validation_predictions['A03_trace_raf_no_event'],
        A03_test=predictions['A03_trace_raf_no_event'],
        residual_candidate_mask=residual_candidate_mask,
        residual_clip_lower=residual_clip_lower,
        residual_clip_upper=residual_clip_upper,
    )
    trace_retrieval_evidence_path = (
        cfg.paths.outputs / 'evidence' / 'trace_raf_retrieval_evidence.parquet'
    )
    pd.concat([
        event_residual_test.evidence.assign(trace_variant='event_conditioned'),
        no_event_residual_test.evidence.assign(trace_variant='no_event'),
    ], ignore_index=True).assign(run_id=RUN_ID).to_parquet(
        trace_retrieval_evidence_path, index=False
    )

    no_event_names = no_event_train.feature_names('no_event_retrieval')
    no_event_drift_names = [f'no_event_drift_{name}' for name in no_event_drift_test.component_names] + ['no_event_drift_score']
    no_event_train_extra = np.column_stack([
        no_event_train.as_features(), no_event_drift_train.components, no_event_drift_train.score
    ])
    no_event_validation_extra = np.column_stack([
        no_event_validation.as_features(), no_event_drift_validation.components,
        no_event_drift_validation.score
    ])
    no_event_test_extra = np.column_stack([
        no_event_test.as_features(), no_event_drift_test.components, no_event_drift_test.score
    ])
    ablation_train, ablation_names = origin_feature_matrix(
        train, context_prefixes, no_event_train_extra, no_event_names + no_event_drift_names
    )
    ablation_validation, _ = origin_feature_matrix(
        validation, context_prefixes, no_event_validation_extra, no_event_names + no_event_drift_names
    )
    ablation_test, _ = origin_feature_matrix(
        test, context_prefixes, no_event_test_extra, no_event_names + no_event_drift_names
    )
    m09_no_events = DirectXGBForecaster(cfg).fit(
        ablation_train[no_event_train.valid_mask], train.future_calendar[no_event_train.valid_mask],
        train.y[no_event_train.valid_mask], ablation_names, train.calendar_names,
    )
    validation_predictions['A00_full_without_events'] = m09_no_events.predict(
        ablation_validation, validation.future_calendar
    )
    predictions['A00_full_without_events'] = m09_no_events.predict(
        ablation_test, test.future_calendar
    )

    ks_detector = KSTwoSampleDriftDetector(cfg).fit_reference(train)
    ks_detector.calibrate(validation)
    ks_validation = ks_detector.transform(validation)
    ks_test = ks_detector.transform(test)
    drift_evidence = pd.concat([
        drift_evidence_frame(train, drift_train, RUN_ID),
        drift_evidence_frame(validation, drift_validation, RUN_ID),
        drift_evidence_frame(test, drift_test, RUN_ID),
    ], ignore_index=True)
    drift_evidence_path = cfg.paths.outputs / 'evidence' / 'drift_evidence.parquet'
    drift_evidence.to_parquet(drift_evidence_path, index=False)
    drift_detector_path = cfg.paths.outputs / 'models' / 'drift_detector.joblib'
    no_event_detector_path = cfg.paths.outputs / 'models' / 'drift_detector_no_events.joblib'
    ks_detector_path = cfg.paths.outputs / 'models' / 'ks_drift_detector.joblib'
    drift_detector.save(drift_detector_path)
    no_event_detector.save(no_event_detector_path)
    ks_detector.save(ks_detector_path)

    models_to_save = {
        'M03': m03, 'M04': m04, 'M07': m07, 'M08': m08, 'M09': m09,
        'M13_trace_raf': trace_gate,
        'A03_trace_raf_no_event': trace_no_event_gate,
        'C06_context_ensemble': context_ensemble,
        'A00_full_without_events': m09_no_events,
        'A01_xgb_random_retrieval': random_model,
        'A02_xgb_matched_event_placebo': a02,
        'C01_ridge_context': ridge, 'C04_xgb_context_event': c04,
    }
    if RUN_JOURNAL_BASELINES:
        models_to_save['C05_lightgbm_context'] = lightgbm
    for name, model in models_to_save.items():
        model.save(cfg.paths.outputs / 'models' / f'{name}.joblib')
    print({'drift_score_mode': cfg.drift.score_mode, 'event_timeraf_threshold': drift_test.threshold,
           'event_timeraf_flags': int(drift_test.flag.sum()), 'ks_flags': int(ks_test.flag.sum())})
    """
)

nb.cells[12].source = code(
    """
    ## 5. Retrieval, event, drift, and knowledge-base diagnostics

    These analyses are executed before the frozen foundation-model gate. They test whether the
    event channel changes rankings, whether denser knowledge bases improve analogue quality, and
    whether the proposed drift score agrees with a standard two-sample KS comparator.
    """
)
nb.cells[13].source = code(
    """
    event_flags = build_event_period_flags(test.metadata, events)
    subset_masks = {
        'event': event_flags['target_event_flag'].to_numpy(),
        'non_event': ~event_flags['target_event_flag'].to_numpy(),
        'recent_event': event_flags['recent_event_flag'].to_numpy(),
        'active_event': event_flags['active_event_flag'].to_numpy(),
        'event_context': event_test.query_event_context,
        'non_event_context': ~event_test.query_event_context,
        'drift': drift_test.flag,
        'non_drift': ~drift_test.flag,
    }
    subset_counts = pd.DataFrame([
        {'run_id': RUN_ID, 'subset': name, 'n_origins': int(mask.sum()),
         'eligible_for_metrics': int(mask.sum()) >= cfg.evaluation.minimum_subset_origins}
        for name, mask in subset_masks.items()
    ])
    subset_counts_path = cfg.paths.outputs / 'tables' / 'subset_counts.csv'
    subset_counts.to_csv(subset_counts_path, index=False)

    k_rows = []
    k_predictions = {'actual': test.y}
    for method in ('cosine', 'event_conditioned'):
        for candidate_k in cfg.retrieval.k_values:
            result = retriever.retrieve(test, method=method, k=candidate_k)
            k_predictions[f'{method}_k_{candidate_k}'] = result.prediction
            k_rows.append({'run_id': RUN_ID, 'method': method, 'k': candidate_k,
                           **metric_values(test.y, result.prediction)})
    k_sensitivity_path = cfg.paths.outputs / 'tables' / 'k_sensitivity_results.csv'
    k_predictions_path = cfg.paths.outputs / 'predictions' / 'k_sensitivity_predictions.npz'
    pd.DataFrame(k_rows).to_csv(k_sensitivity_path, index=False)
    np.savez_compressed(k_predictions_path, **k_predictions)

    event_weight_rows = []
    composition_rows = []
    weight_results = {}
    for weight in cfg.retrieval.event_weight_values:
        result = retriever.retrieve(test, method='event_conditioned', event_weight=weight)
        weight_results[weight] = result
        for subset, mask in {'all': np.ones(len(test.x), bool),
                             'event_context': result.query_event_context,
                             'non_event_context': ~result.query_event_context}.items():
            event_weight_rows.append({
                'run_id': RUN_ID, 'event_weight': weight, 'subset': subset,
                'n_origins': int(mask.sum()), **metric_values(test.y[mask], result.prediction[mask]),
            })
        selected = result.evidence
        composition_rows.append({
            'run_id': RUN_ID, 'event_weight': weight,
            'event_context_queries': int(result.query_event_context.sum()),
            'conditioned_queries': int(result.event_conditioning_applied.sum()),
            'selected_candidate_event_fraction': float(selected['candidate_has_event_context'].mean()),
            'event_score_zero_fraction': float((selected['event_score'] == 0).mean()),
        })
    baseline_ids = (
        weight_results[0.0].evidence.sort_values(['query_window_id', 'rank'])['candidate_window_id'].to_numpy()
    )
    for row, weight in zip(event_weight_rows[::3], cfg.retrieval.event_weight_values):
        ids = weight_results[weight].evidence.sort_values(['query_window_id', 'rank'])['candidate_window_id'].to_numpy()
        row['selected_candidate_changed_fraction_vs_weight_0'] = float(np.mean(ids != baseline_ids))
    event_weight_path = cfg.paths.outputs / 'tables' / 'event_weight_sensitivity.csv'
    event_composition_path = cfg.paths.outputs / 'tables' / 'event_candidate_composition.csv'
    event_weight_retrieval_predictions_path = (
        cfg.paths.outputs / 'predictions' / 'event_weight_retrieval_predictions.npz'
    )
    pd.DataFrame(event_weight_rows).to_csv(event_weight_path, index=False)
    pd.DataFrame(composition_rows).to_csv(event_composition_path, index=False)
    np.savez_compressed(
        event_weight_retrieval_predictions_path,
        actual=test.y,
        **{
            f"weight_{weight:.2f}".replace('.', 'p'): result.prediction
            for weight, result in weight_results.items()
        },
    )

    category_rows = []
    primary_evidence = weight_results[float(cfg.retrieval.weights['event'])].evidence
    candidate_lookup = pd.Series(
        np.arange(len(knowledge_base.metadata)),
        index=knowledge_base.metadata['window_id'].astype(str),
    )
    for category in EVENT_CATEGORIES:
        feature_name = f'event_{category}_72h'
        if feature_name not in test.feature_names or feature_name not in knowledge_base.feature_names:
            continue
        query_column = test.feature_names.index(feature_name)
        candidate_column = knowledge_base.feature_names.index(feature_name)
        query_has_category = test.features[:, query_column] > cfg.retrieval.epsilon
        query_ids = set(test.metadata.loc[query_has_category, 'window_id'].astype(str))
        selected = primary_evidence.loc[
            primary_evidence['query_window_id'].astype(str).isin(query_ids)
        ]
        selected_indices = candidate_lookup.reindex(selected['candidate_window_id'].astype(str)).dropna().astype(int)
        selected_fraction = (
            float(np.mean(knowledge_base.features[selected_indices, candidate_column] > cfg.retrieval.epsilon))
            if len(selected_indices) else np.nan
        )
        category_rows.append({
            'run_id': RUN_ID, 'category': category,
            'query_origins_with_category': int(query_has_category.sum()),
            'selected_candidates': int(len(selected_indices)),
            'selected_candidate_category_fraction': selected_fraction,
            'knowledge_base_category_fraction': float(np.mean(
                knowledge_base.features[:, candidate_column] > cfg.retrieval.epsilon
            )),
        })
    event_category_composition_path = cfg.paths.outputs / 'tables' / 'event_category_candidate_composition.csv'
    pd.DataFrame(category_rows).to_csv(event_category_composition_path, index=False)

    event_weight_model_rows = []
    event_weight_model_predictions = {'validation_actual': validation.y, 'test_actual': test.y}
    primary_event_weight = float(cfg.retrieval.weights['event'])
    for weight in cfg.retrieval.event_weight_values:
        if np.isclose(weight, primary_event_weight):
            validation_values = validation_predictions['M08_event_timeraf_no_drift']
            test_values = predictions['M08_event_timeraf_no_drift']
        else:
            train_result = retriever.retrieve(train, method='event_conditioned', event_weight=weight)
            validation_result = retriever.retrieve(validation, method='event_conditioned', event_weight=weight)
            test_result = weight_results[weight]
            weight_model, _, validation_matrix, test_matrix, _ = fit_retrieval_xgb(
                train_result, validation_result, test_result, full_prefixes,
                f'event_weight_{weight:.2f}'.replace('.', 'p'),
            )
            validation_values = weight_model.predict(validation_matrix, validation.future_calendar)
            test_values = weight_model.predict(test_matrix, test.future_calendar)
        key = f"weight_{weight:.2f}".replace('.', 'p')
        event_weight_model_predictions[f'{key}_validation'] = validation_values
        event_weight_model_predictions[f'{key}_test'] = test_values
        for split, actual, values in (
            ('validation', validation.y, validation_values), ('test', test.y, test_values)
        ):
            event_weight_model_rows.append({
                'run_id': RUN_ID, 'event_weight': weight, 'split': split,
                'model': 'M08_event_timeraf_no_drift',
                'primary_configuration': bool(np.isclose(weight, primary_event_weight)),
                **metric_values(actual, values),
            })
    event_weight_model_path = cfg.paths.outputs / 'tables' / 'event_weight_model_sensitivity.csv'
    event_weight_predictions_path = cfg.paths.outputs / 'predictions' / 'event_weight_model_predictions.npz'
    pd.DataFrame(event_weight_model_rows).to_csv(event_weight_model_path, index=False)
    np.savez_compressed(event_weight_predictions_path, **event_weight_model_predictions)

    stride_rows = []
    stride_retrieval_predictions = {'actual': test.y}
    stride_cache = {}
    for stride in cfg.retrieval.kb_stride_values:
        started = time.perf_counter()
        if stride == cfg.retrieval.kb_stride_hours:
            stride_kb, stride_retriever = knowledge_base, retriever
            stride_results = {
                'random': random_test, 'cosine': cosine_test,
                'event_conditioned': event_test,
            }
        else:
            stride_kb = build_knowledge_base(dataset, cfg, stride_hours=stride)
            stride_retriever = HistoricalRetriever(stride_kb, cfg)
            stride_results = {
                method: stride_retriever.retrieve(test, method=method)
                for method in ('random', 'cosine', 'event_conditioned')
            }
        stride_cache[stride] = (stride_kb, stride_retriever, stride_results)
        elapsed = time.perf_counter() - started
        for method, result in stride_results.items():
            stride_retrieval_predictions[f'stride_{stride}_{method}'] = result.prediction
            stride_rows.append({
                'run_id': RUN_ID, 'stride_hours': stride, 'method': method,
                'candidate_count': len(stride_kb.metadata), 'runtime_seconds': elapsed,
                'minimum_eligible_candidates': int(result.eligible_candidate_count.min()),
                **metric_values(test.y, result.prediction),
            })
    stride_path = cfg.paths.outputs / 'tables' / 'kb_stride_sensitivity.csv'
    stride_predictions_path = cfg.paths.outputs / 'predictions' / 'kb_stride_retrieval_predictions.npz'
    pd.DataFrame(stride_rows).to_csv(stride_path, index=False)
    np.savez_compressed(stride_predictions_path, **stride_retrieval_predictions)

    stride_model_rows = []
    stride_model_predictions = {'actual': test.y}
    if RUN_STRIDE_MODEL_SWEEP:
        for stride in cfg.retrieval.kb_stride_values:
            if stride == cfg.retrieval.kb_stride_hours:
                for name in ('M07_xgb_cosine', 'M08_event_timeraf_no_drift', 'M09_event_timeraf_full'):
                    stride_model_rows.append({'run_id': RUN_ID, 'stride_hours': stride,
                                              'model': name, **metric_values(test.y, predictions[name])})
                    stride_model_predictions[f'stride_{stride}_{name}'] = predictions[name]
                continue
            stride_kb, stride_retriever, test_results = stride_cache[stride]
            train_cos = stride_retriever.retrieve(train, method='cosine')
            val_cos = stride_retriever.retrieve(validation, method='cosine')
            train_event = stride_retriever.retrieve(train, method='event_conditioned')
            val_event = stride_retriever.retrieve(validation, method='event_conditioned')
            model07, _, _, matrix07, _ = fit_retrieval_xgb(
                train_cos, val_cos, test_results['cosine'], context_prefixes, f'cosine_s{stride}'
            )
            pred07 = model07.predict(matrix07, test.future_calendar)
            model08, _, _, matrix08, _ = fit_retrieval_xgb(
                train_event, val_event, test_results['event_conditioned'], full_prefixes,
                f'event_s{stride}'
            )
            pred08 = model08.predict(matrix08, test.future_calendar)
            detector = DriftDetector(cfg).fit_reference(train, train_event.mean_similarity)
            detector.calibrate(validation, val_event.mean_similarity)
            drift_train_s = detector.transform(train, train_event.mean_similarity)
            drift_test_s = detector.transform(test, test_results['event_conditioned'].mean_similarity)
            extra_train = np.column_stack([train_event.as_features(), drift_train_s.components, drift_train_s.score])
            extra_test = np.column_stack([test_results['event_conditioned'].as_features(), drift_test_s.components, drift_test_s.score])
            extra_names = train_event.feature_names(f'event_s{stride}') + [
                f'drift_s{stride}_{name}' for name in drift_test_s.component_names
            ] + [f'drift_s{stride}_score']
            matrix09_train, names09 = origin_feature_matrix(train, full_prefixes, extra_train, extra_names)
            matrix09_test, _ = origin_feature_matrix(test, full_prefixes, extra_test, extra_names)
            model09 = DirectXGBForecaster(cfg).fit(
                matrix09_train[train_event.valid_mask], train.future_calendar[train_event.valid_mask],
                train.y[train_event.valid_mask], names09, train.calendar_names,
            )
            pred09 = model09.predict(matrix09_test, test.future_calendar)
            for name, values in {'M07_xgb_cosine': pred07,
                                 'M08_event_timeraf_no_drift': pred08,
                                 'M09_event_timeraf_full': pred09}.items():
                stride_model_predictions[f'stride_{stride}_{name}'] = values
                stride_model_rows.append({'run_id': RUN_ID, 'stride_hours': stride,
                                          'model': name, **metric_values(test.y, values)})
        stride_model_frame = pd.DataFrame(stride_model_rows)
        expected_stride_rows = {
            (stride, model)
            for stride in cfg.retrieval.kb_stride_values
            for model in ('M07_xgb_cosine', 'M08_event_timeraf_no_drift', 'M09_event_timeraf_full')
        }
        actual_stride_rows = set(zip(
            stride_model_frame['stride_hours'], stride_model_frame['model']
        ))
        if actual_stride_rows != expected_stride_rows:
            raise RuntimeError(
                f'Incomplete learned stride sweep: {sorted(expected_stride_rows - actual_stride_rows)}'
            )
    else:
        stride_model_frame = pd.DataFrame(stride_model_rows)
    stride_model_path = cfg.paths.outputs / 'tables' / 'kb_stride_model_sensitivity.csv'
    stride_model_predictions_path = cfg.paths.outputs / 'predictions' / 'kb_stride_model_predictions.npz'
    stride_model_frame.to_csv(stride_model_path, index=False)
    np.savez_compressed(stride_model_predictions_path, **stride_model_predictions)

    drift_comparison = pd.DataFrame([
        {'run_id': RUN_ID, 'detector': 'Event-TimeRAF composite', 'threshold': drift_test.threshold,
         'flagged_origins': int(drift_test.flag.sum())},
        {'run_id': RUN_ID, 'detector': 'KS two-sample', 'threshold': ks_test.threshold,
         'flagged_origins': int(ks_test.flag.sum())},
    ])
    union = drift_test.flag | ks_test.flag
    drift_comparison['flag_jaccard'] = (
        float((drift_test.flag & ks_test.flag).sum() / union.sum()) if union.any() else np.nan
    )
    drift_comparison_path = cfg.paths.outputs / 'tables' / 'drift_detector_comparison.csv'
    drift_comparison.to_csv(drift_comparison_path, index=False)
    display(pd.DataFrame(stride_rows))
    display(pd.DataFrame(event_weight_rows))
    display(pd.DataFrame(category_rows))
    """
)

nb.cells[14].source = code(
    """
    ## 6. Frozen-TSFM gate and validation-selected drift router

    Chronos is evaluated without fine-tuning. Retrieval fusion and the exploratory drift router
    are selected on validation data only; the test set is not used to choose their weights or
    component models.
    """
)
nb.cells[15].source = code(
    """
    tsfm_gate_path = cfg.paths.outputs / 'logs' / 'tsfm_gate_status.json'
    if RUN_TSF_MODEL:
        tsfm_validation, tsfm_val_quantiles = chronos_quantile_forecast(
            validation.x, cfg.forecast.horizon, cfg.tsfm.checkpoint,
            cfg.evaluation.probabilistic_quantiles, cfg.tsfm.batch_size
        )
        tsfm_test, tsfm_test_quantiles = chronos_quantile_forecast(
            test.x, cfg.forecast.horizon, cfg.tsfm.checkpoint,
            cfg.evaluation.probabilistic_quantiles, cfg.tsfm.batch_size
        )
        lower_index = cfg.evaluation.probabilistic_quantiles.index(0.1)
        upper_index = cfg.evaluation.probabilistic_quantiles.index(0.9)
        tsfm_test_low = tsfm_test_quantiles[..., lower_index]
        tsfm_test_high = tsfm_test_quantiles[..., upper_index]
        selected_weight, fusion_scores = choose_fusion_weight(
            validation.y, tsfm_validation, event_validation.prediction, cfg.tsfm.fusion_weights
        )
        validation_predictions['M10_frozen_chronos'] = tsfm_validation
        validation_predictions['M11_chronos_event_retrieval'] = fuse_forecasts(
            tsfm_validation, event_validation.prediction, selected_weight
        )
        predictions['M10_frozen_chronos'] = tsfm_test
        predictions['M11_chronos_event_retrieval'] = fuse_forecasts(
            tsfm_test, event_test.prediction, selected_weight
        )
        fused_test_quantiles = (
            selected_weight * tsfm_test_quantiles
            + (1.0 - selected_weight) * event_test.prediction[..., None]
        ).astype(np.float32)
        placebo_sources = {
            'P00_chronos_climatology_fusion': 'C00_hour_month_climatology',
            'P01_chronos_persistence_fusion': 'M00_persistence',
        }
        placebo_fusion_rows = []
        placebo_weights = {}
        for placebo_name, source_name in placebo_sources.items():
            placebo_weight, placebo_scores = choose_fusion_weight(
                validation.y, tsfm_validation, validation_predictions[source_name],
                cfg.tsfm.fusion_weights,
            )
            placebo_weights[placebo_name] = placebo_weight
            validation_predictions[placebo_name] = fuse_forecasts(
                tsfm_validation, validation_predictions[source_name], placebo_weight
            )
            predictions[placebo_name] = fuse_forecasts(
                tsfm_test, predictions[source_name], placebo_weight
            )
            placebo_fusion_rows.extend({
                'run_id': RUN_ID, 'placebo_model': placebo_name, 'source_model': source_name,
                'tsfm_weight': weight, 'validation_mse': score,
                'selected_on_validation': weight == placebo_weight,
            } for weight, score in placebo_scores.items())
        tsfm_predictions_path = cfg.paths.outputs / 'predictions' / 'tsfm_predictions.npz'
        np.savez_compressed(
            tsfm_predictions_path, validation_mean=tsfm_validation,
            validation_quantiles=tsfm_val_quantiles,
            test_mean=tsfm_test, test_quantiles=tsfm_test_quantiles,
            fused_test_mean=predictions['M11_chronos_event_retrieval'],
            fused_test_quantiles=fused_test_quantiles,
            quantile_levels=np.asarray(cfg.evaluation.probabilistic_quantiles),
            fusion_weight=selected_weight,
        )
        fusion_scores_path = cfg.paths.outputs / 'tables' / 'tsfm_fusion_validation.csv'
        pd.DataFrame([{'run_id': RUN_ID, 'tsfm_weight': weight, 'validation_mse': score}
                      for weight, score in fusion_scores.items()]).to_csv(fusion_scores_path, index=False)
        placebo_fusion_path = cfg.paths.outputs / 'tables' / 'tsfm_placebo_fusion_validation.csv'
        pd.DataFrame(placebo_fusion_rows).to_csv(placebo_fusion_path, index=False)
        interval_path = cfg.paths.outputs / 'tables' / 'tsfm_interval_metrics.csv'
        fused_test_low = fused_test_quantiles[..., lower_index]
        fused_test_high = fused_test_quantiles[..., upper_index]
        pd.concat([
            interval_metrics(
                test.y, tsfm_test_low, tsfm_test_high, alpha=0.2,
                model='M10_frozen_chronos', run_id=RUN_ID,
            ),
            interval_metrics(
                test.y, fused_test_low, fused_test_high, alpha=0.2,
                model='M11_chronos_event_retrieval', run_id=RUN_ID,
            ),
        ], ignore_index=True).to_csv(interval_path, index=False)
        quantile_calibration_path = cfg.paths.outputs / 'tables' / 'tsfm_quantile_calibration.csv'
        probabilistic_path = cfg.paths.outputs / 'tables' / 'tsfm_probabilistic_metrics.csv'
        calibration_frames = []
        probabilistic_frames = []
        for name, values in {
            'M10_frozen_chronos': tsfm_test_quantiles,
            'M11_chronos_event_retrieval': fused_test_quantiles,
        }.items():
            calibration, probabilistic = quantile_forecast_metrics(
                test.y, values, cfg.evaluation.probabilistic_quantiles, name, RUN_ID
            )
            calibration_frames.append(calibration)
            probabilistic_frames.append(probabilistic)
        pd.concat(calibration_frames, ignore_index=True).to_csv(
            quantile_calibration_path, index=False
        )
        pd.concat(probabilistic_frames, ignore_index=True).to_csv(
            probabilistic_path, index=False
        )
        tsfm_gate_status = {'run_id': RUN_ID, 'completed': True,
                            'checkpoint': cfg.tsfm.checkpoint,
                            'selected_fusion_weight': selected_weight,
                            'placebo_fusion_weights': placebo_weights}
    else:
        tsfm_gate_status = {'run_id': RUN_ID, 'completed': False,
                            'checkpoint': cfg.tsfm.checkpoint,
                            'reason': 'RUN_TSF_MODEL is False'}
    tsfm_gate_path.write_text(json.dumps(tsfm_gate_status, indent=2), encoding='utf-8')
    if FINAL_EXPERIMENT and not tsfm_gate_status['completed']:
        raise RuntimeError('A final experiment requires the frozen-TSFM gate.')

    def subset_mse(actual, predicted, mask):
        return float(np.mean((actual[mask] - predicted[mask]) ** 2)) if mask.any() else np.inf

    router_candidates = [
        'M04_xgb_context', 'M08_event_timeraf_no_drift', 'M09_event_timeraf_full',
        'M11_chronos_event_retrieval',
    ]
    router_selection = {}
    for state, mask in {'non_drift': ~drift_validation.flag, 'drift': drift_validation.flag}.items():
        router_selection[state] = min(
            router_candidates,
            key=lambda name: subset_mse(validation.y, validation_predictions[name], mask),
        )
    predictions['M12_validation_drift_router'] = np.where(
        drift_test.flag[:, None], predictions[router_selection['drift']],
        predictions[router_selection['non_drift']],
    ).astype(np.float32)
    validation_predictions['M12_validation_drift_router'] = np.where(
        drift_validation.flag[:, None], validation_predictions[router_selection['drift']],
        validation_predictions[router_selection['non_drift']],
    ).astype(np.float32)
    router_path = cfg.paths.outputs / 'logs' / 'm12_gate_selection.json'
    router_path.write_text(json.dumps({
        'run_id': RUN_ID, 'selection_basis': 'validation-only drift subsets',
        'non_drift_source': router_selection['non_drift'],
        'drift_source': router_selection['drift'],
    }, indent=2), encoding='utf-8')
    print({'tsfm_weight': selected_weight, 'placebo_weights': placebo_weights,
           'drift_router': router_selection})
    """
)

nb.cells[16].source = code(
    """
    ## 7. Final evaluation, explanations, figures, and run package

    The final cell writes every number and figure from arrays produced in this run, records all
    artifacts in the manifest, and creates the ZIP to download from Kaggle.
    """
)
nb.cells[17].source = code(
    """
    def model_metric_frames(name, values):
        frames = [metrics_table(test.y, values, name, run_id=RUN_ID,
                                event_availability_mode=EVENT_AVAILABILITY_MODE)]
        for subset, mask in subset_masks.items():
            if int(mask.sum()) >= cfg.evaluation.minimum_subset_origins:
                frames.append(metrics_table(
                    test.y[mask], values[mask], name, subset=subset, run_id=RUN_ID,
                    event_availability_mode=EVENT_AVAILABILITY_MODE,
                ))
        return frames

    metrics = pd.concat([
        frame for name, values in predictions.items()
        for frame in model_metric_frames(name, values)
    ], ignore_index=True)
    prediction_table = pd.concat([
        predictions_long(
            test.y, values, test.metadata, name, cfg.seed, RUN_ID,
            drift_flag=drift_test.flag, drift_score=drift_test.score,
            event_flags=event_flags, event_availability_mode=EVENT_AVAILABILITY_MODE,
        ) for name, values in predictions.items()
    ], ignore_index=True)
    metrics_path = cfg.paths.outputs / 'tables' / 'metrics.csv'
    main_results_path = cfg.paths.outputs / 'tables' / 'main_results.csv'
    predictions_path = cfg.paths.outputs / 'predictions' / 'predictions.parquet'
    metrics.to_csv(metrics_path, index=False)
    metrics.loc[(metrics['horizon'].astype(str) == 'overall') & (metrics['subset'] == 'all')].to_csv(
        main_results_path, index=False
    )
    prediction_table.to_parquet(predictions_path, index=False)

    def period_summary(flag_name, flag_values):
        rows = []
        for state in (False, True):
            mask = np.asarray(flag_values, dtype=bool) == state
            reference_mse = metric_values(
                test.y[mask], predictions['C00_hour_month_climatology'][mask]
            )['mse']
            for name, values in predictions.items():
                scores = metric_values(test.y[mask], values[mask])
                rows.append({
                    'run_id': RUN_ID, 'model': name, flag_name: state,
                    'n_origins': int(mask.sum()), 'n_points': int(mask.sum() * test.y.shape[1]),
                    'eligible_for_metrics': int(mask.sum()) >= cfg.evaluation.minimum_subset_origins,
                    'mse': scores['mse'], 'mae': scores['mae'],
                    'skill_vs_climatology': 1 - scores['mse'] / reference_mse,
                    'event_availability_mode': EVENT_AVAILABILITY_MODE,
                })
        result = pd.DataFrame(rows)
        result.loc[~result['eligible_for_metrics'], ['mse', 'mae', 'skill_vs_climatology']] = np.nan
        return result

    drift_period_path = cfg.paths.outputs / 'tables' / 'drift_period_results.csv'
    event_period_path = cfg.paths.outputs / 'tables' / 'event_period_results.csv'
    period_summary('drift_flag', drift_test.flag).to_csv(drift_period_path, index=False)
    period_summary('target_event_flag', event_flags['target_event_flag']).to_csv(
        event_period_path, index=False
    )

    component_columns = [f'drift_{name}' for name in drift_test.component_names]
    drift_diagnostic_rows = []
    for split, frame in drift_evidence.groupby('split'):
        for component in component_columns:
            values = frame[component].to_numpy(dtype=float)
            drift_diagnostic_rows.append({
                'run_id': RUN_ID, 'split': split, 'component': component,
                'score_mode': cfg.drift.score_mode, 'n_origins': len(values),
                'zero_fraction': float(np.mean(values == 0)), 'mean': float(np.mean(values)),
                'std': float(np.std(values)), 'p50': float(np.quantile(values, 0.5)),
                'p90': float(np.quantile(values, 0.9)), 'p95': float(np.quantile(values, 0.95)),
                'max': float(np.max(values)),
            })
    drift_diagnostics_path = cfg.paths.outputs / 'tables' / 'drift_component_diagnostics.csv'
    pd.DataFrame(drift_diagnostic_rows).to_csv(drift_diagnostics_path, index=False)
    correlations = drift_evidence.loc[drift_evidence['split'] == 'test', component_columns].corr()
    correlations.index.name = 'component'
    drift_correlations_path = cfg.paths.outputs / 'tables' / 'drift_component_correlations.csv'
    correlation_output = correlations.reset_index()
    correlation_output.insert(0, 'run_id', RUN_ID)
    correlation_output.to_csv(drift_correlations_path, index=False)

    climatology = predictions['C00_hour_month_climatology']
    subset_stats = []
    for subset, mask in {'all': np.ones(len(test.x), bool), **subset_masks}.items():
        if not mask.any():
            continue
        reference_mse = metric_values(test.y[mask], climatology[mask])['mse']
        subset_stats.append({
            'run_id': RUN_ID, 'subset': subset, 'n_origins': int(mask.sum()),
            'target_mean': float(test.y[mask].mean()), 'target_variance': float(test.y[mask].var()),
            'climatology_mse': reference_mse,
            'M04_skill_vs_climatology': 1 - metric_values(test.y[mask], predictions['M04_xgb_context'][mask])['mse'] / reference_mse,
            'M09_skill_vs_climatology': 1 - metric_values(test.y[mask], predictions['M09_event_timeraf_full'][mask])['mse'] / reference_mse,
        })
    subset_stats_path = cfg.paths.outputs / 'tables' / 'subset_target_statistics.csv'
    pd.DataFrame(subset_stats).to_csv(subset_stats_path, index=False)

    horizon_skill_path = cfg.paths.outputs / 'tables' / 'horizon_skill_vs_climatology.csv'
    pd.concat([
        horizon_skill_table(test.y, values, climatology, name, RUN_ID)
        for name, values in predictions.items()
    ], ignore_index=True).to_csv(horizon_skill_path, index=False)

    exceedance_path = cfg.paths.outputs / 'tables' / 'aqi_exceedance_metrics.csv'
    pd.concat([
        exceedance_metrics(test.y, values, cfg.evaluation.aqi_thresholds, name, RUN_ID)
        for name, values in predictions.items()
    ], ignore_index=True).to_csv(exceedance_path, index=False)
    log_metrics_path = cfg.paths.outputs / 'tables' / 'log_scale_metrics.csv'
    pd.concat([
        log_scale_metrics(test.y, values, name, RUN_ID)
        for name, values in predictions.items()
    ], ignore_index=True).to_csv(log_metrics_path, index=False)

    trace_subset_rows = []
    trace_subset_comparisons = {
        'M13_minus_C06_trace_residual_gate': (
            predictions['M13_trace_raf'], predictions['C06_context_ensemble']
        ),
        'M13_minus_A03_trace_event_conditioning': (
            predictions['M13_trace_raf'], predictions['A03_trace_raf_no_event']
        ),
    }
    for subset, mask in {
        'event': subset_masks['event'],
        'event_context': subset_masks['event_context'],
        'drift': subset_masks['drift'],
        'non_event': subset_masks['non_event'],
    }.items():
        if int(mask.sum()) < cfg.evaluation.minimum_subset_origins:
            continue
        for comparison_name, (prediction_a, prediction_b) in trace_subset_comparisons.items():
            for metric_name in ('mse', 'mae'):
                interval = paired_masked_block_bootstrap_loss_difference(
                    test.y, prediction_a, prediction_b, mask, metric_name,
                    cfg.evaluation.bootstrap_block_hours,
                    cfg.evaluation.bootstrap_resamples, cfg.seed,
                )
                trace_subset_rows.append({
                    'run_id': RUN_ID, 'subset': subset, 'comparison': comparison_name,
                    'metric': metric_name, **interval,
                })
    trace_subset_inference = pd.DataFrame(trace_subset_rows)
    for metric_name, indices in trace_subset_inference.groupby('metric').groups.items():
        indices = list(indices)
        trace_subset_inference.loc[indices, 'bootstrap_p_value_holm'] = holm_adjust_pvalues(
            trace_subset_inference.loc[indices, 'bootstrap_p_value'].to_numpy()
        )
    trace_subset_inference_path = (
        cfg.paths.outputs / 'tables' / 'trace_raf_subset_inference.csv'
    )
    trace_subset_inference.to_csv(trace_subset_inference_path, index=False)
    trace_subset_masks_path = cfg.paths.outputs / 'predictions' / 'trace_raf_subset_masks.npz'
    np.savez_compressed(
        trace_subset_masks_path,
        event=subset_masks['event'], event_context=subset_masks['event_context'],
        drift=subset_masks['drift'], non_event=subset_masks['non_event'],
    )

    comparisons = {
        'M04_minus_M03_weather_calendar': (predictions['M04_xgb_context'], predictions['M03_xgb_pm25']),
        'M07_minus_M04_cosine_retrieval': (predictions['M07_xgb_cosine'], predictions['M04_xgb_context']),
        'M08_minus_M07_event_conditioning': (predictions['M08_event_timeraf_no_drift'], predictions['M07_xgb_cosine']),
        'M09_minus_M08_drift_features': (predictions['M09_event_timeraf_full'], predictions['M08_event_timeraf_no_drift']),
        'M09_minus_A00_events': (predictions['M09_event_timeraf_full'], predictions['A00_full_without_events']),
        'M09_minus_M04_full': (predictions['M09_event_timeraf_full'], predictions['M04_xgb_context']),
        'A01_minus_M04_random_control': (predictions['A01_xgb_random_retrieval'], predictions['M04_xgb_context']),
        'M09_minus_A01_random_control': (predictions['M09_event_timeraf_full'], predictions['A01_xgb_random_retrieval']),
        'C04_minus_M04_raw_event_features': (predictions['C04_xgb_context_event'], predictions['M04_xgb_context']),
        'A02_minus_M04_matched_feature_count': (predictions['A02_xgb_matched_event_placebo'], predictions['M04_xgb_context']),
        'C04_minus_A02_event_signal': (predictions['C04_xgb_context_event'], predictions['A02_xgb_matched_event_placebo']),
        'C05_minus_M04_lightgbm': (predictions['C05_lightgbm_context'], predictions['M04_xgb_context']),
        'C06_minus_C05_context_ensemble': (predictions['C06_context_ensemble'], predictions['C05_lightgbm_context']),
        'M13_minus_C06_trace_residual_gate': (predictions['M13_trace_raf'], predictions['C06_context_ensemble']),
        'A03_minus_C06_no_event_residual_gate': (predictions['A03_trace_raf_no_event'], predictions['C06_context_ensemble']),
        'M13_minus_A03_trace_event_conditioning': (predictions['M13_trace_raf'], predictions['A03_trace_raf_no_event']),
        'M13_minus_C05_trace_vs_best_baseline': (predictions['M13_trace_raf'], predictions['C05_lightgbm_context']),
        'B00_minus_M04_dlinear': (predictions['B00_dlinear'], predictions['M04_xgb_context']),
        'B01_minus_M04_patchtst': (predictions['B01_patchtst'], predictions['M04_xgb_context']),
        'B02_minus_M04_lstm': (predictions['B02_lstm'], predictions['M04_xgb_context']),
        'M11_minus_M10_retrieval': (predictions['M11_chronos_event_retrieval'], predictions['M10_frozen_chronos']),
        'M11_minus_P00_climatology_placebo': (predictions['M11_chronos_event_retrieval'], predictions['P00_chronos_climatology_fusion']),
        'M11_minus_P01_persistence_placebo': (predictions['M11_chronos_event_retrieval'], predictions['P01_chronos_persistence_fusion']),
        'M12_minus_M04_drift_router': (predictions['M12_validation_drift_router'], predictions['M04_xgb_context']),
    }
    bootstrap_rows = []
    for comparison_name, (prediction_a, prediction_b) in comparisons.items():
        for metric_name in ('mse', 'mae'):
            interval = paired_block_bootstrap_loss_difference(
                test.y, prediction_a, prediction_b, metric_name,
                cfg.evaluation.bootstrap_block_hours, cfg.evaluation.bootstrap_resamples,
                cfg.seed,
            )
            dm = diebold_mariano_hac(
                test.y, prediction_a, prediction_b, metric_name,
                cfg.evaluation.dm_hac_lags,
            )
            bootstrap_rows.append({'run_id': RUN_ID, 'comparison': comparison_name,
                                   'comparison_family': 'component_and_placebo_tests',
                                   'metric': metric_name, **interval, **dm})
    ablation_path = cfg.paths.outputs / 'tables' / 'ablation_results.csv'
    ablation_results = pd.DataFrame(bootstrap_rows)
    for metric_name, indices in ablation_results.groupby('metric').groups.items():
        indices = list(indices)
        ablation_results.loc[indices, 'bootstrap_p_value_holm'] = holm_adjust_pvalues(
            ablation_results.loc[indices, 'bootstrap_p_value'].to_numpy()
        )
        ablation_results.loc[indices, 'dm_p_value_holm'] = holm_adjust_pvalues(
            ablation_results.loc[indices, 'dm_p_value'].to_numpy()
        )
    ablation_results['significant_dm_holm_0_05'] = ablation_results['dm_p_value_holm'] < 0.05
    ablation_results.to_csv(ablation_path, index=False)

    fusion_rows = []
    for method, result in {'cosine': cosine_test, 'event_conditioned': event_test}.items():
        for aggregation, values in {'uniform': result.prediction,
                                    'similarity_weighted': result.weighted_prediction}.items():
            fusion_rows.append({'run_id': RUN_ID, 'method': method,
                                'aggregation': aggregation, **metric_values(test.y, values)})
    retrieval_fusion_path = cfg.paths.outputs / 'tables' / 'retrieval_fusion_ablation.csv'
    pd.DataFrame(fusion_rows).to_csv(retrieval_fusion_path, index=False)

    feature_groups = {
        'pm25': ('pm25_',), 'weather': ('weather_',), 'calendar': ('cal_',),
        'event': ('event_',), 'retrieval': ('event_conditioned_retrieval_',),
        'drift': ('drift_',),
    }
    group_importance = pd.concat([
        grouped_feature_perturbation(
            validation.y, m08_validation, m08_train[event_train.valid_mask], m08_names,
            feature_groups,
            lambda matrix: m08.predict(matrix, validation.future_calendar),
            cfg.seed, RUN_ID, 'M08_event_timeraf_no_drift',
        ),
        grouped_feature_perturbation(
            validation.y, m09_validation, m09_train[event_train.valid_mask], m09_names,
            feature_groups,
            lambda matrix: m09.predict(matrix, validation.future_calendar),
            cfg.seed, RUN_ID, 'M09_event_timeraf_full',
        ),
    ], ignore_index=True)
    group_importance_path = cfg.paths.outputs / 'tables' / 'validation_group_faithfulness.csv'
    group_importance.to_csv(group_importance_path, index=False)

    leaf_rows = []
    for horizon in range(cfg.forecast.horizon):
        matrix = m09._matrix(m09_test, test.future_calendar, horizon)
        leaf_indices = np.asarray(m09.models[horizon].apply(matrix), dtype=np.int64)
        if leaf_indices.ndim == 1:
            leaf_indices = leaf_indices[:, None]
        for state, state_mask in {'non_drift': ~drift_test.flag, 'drift': drift_test.flag}.items():
            other_mask = ~state_mask
            support_sum = 0.0
            exclusive_count = 0
            assignment_count = int(state_mask.sum() * leaf_indices.shape[1])
            occupied_pairs = 0
            for tree in range(leaf_indices.shape[1]):
                leaves = leaf_indices[:, tree]
                unique, inverse = np.unique(leaves, return_inverse=True)
                state_counts = np.bincount(inverse[state_mask], minlength=len(unique))
                other_counts = np.bincount(inverse[other_mask], minlength=len(unique))
                support_sum += float(state_counts[inverse[state_mask]].sum())
                exclusive_count += int(np.sum(other_counts[inverse[state_mask]] == 0))
                occupied_pairs += int(np.count_nonzero(state_counts))
            leaf_rows.append({
                'run_id': RUN_ID, 'model': 'M09_event_timeraf_full',
                'horizon': horizon + 1, 'state': state,
                'n_origins': int(state_mask.sum()), 'n_trees': int(leaf_indices.shape[1]),
                'occupied_tree_leaf_pairs': occupied_pairs,
                'mean_same_state_leaf_support': support_sum / assignment_count if assignment_count else np.nan,
                'exclusive_assignment_fraction': exclusive_count / assignment_count if assignment_count else np.nan,
            })
    drift_leaf_path = cfg.paths.outputs / 'tables' / 'drift_leaf_occupancy.csv'
    pd.DataFrame(leaf_rows).to_csv(drift_leaf_path, index=False)

    horizon_contributions = []
    for horizon in range(cfg.forecast.horizon):
        matrix = m09._matrix(m09_test, test.future_calendar, horizon)
        horizon_contributions.append(xgb_local_contributions(m09.models[horizon], matrix))
    mean_contributions = np.mean(np.stack(horizon_contributions, axis=1), axis=1)
    validation_residual_rmse = np.sqrt(np.mean(
        (validation.y - validation_predictions['M09_event_timeraf_full']) ** 2, axis=0
    ))
    feature_effects_path = cfg.paths.outputs / 'evidence' / 'mean_24h_feature_effects.npz'
    np.savez_compressed(feature_effects_path, contributions=mean_contributions,
                        feature_names=np.asarray(m09.feature_names),
                        window_ids=test.metadata['window_id'].to_numpy())
    explanations = generate_explanations(
        test, predictions['M09_event_timeraf_full'], event_test, drift_test, events,
        mean_contributions, m09.feature_names, validation_residual_rmse,
    )
    explanations.insert(0, 'run_id', RUN_ID)
    explanations['event_availability_mode'] = EVENT_AVAILABILITY_MODE
    explanations_path = cfg.paths.outputs / 'evidence' / 'explanations.parquet'
    explanations.to_parquet(explanations_path, index=False)

    horizon_figure_path = cfg.paths.outputs / 'figures' / 'mae_by_horizon.png'
    mse_figure_path = cfg.paths.outputs / 'figures' / 'mse_by_horizon.png'
    retrieval_figure_path = cfg.paths.outputs / 'figures' / 'retrieval_diagnostics.png'
    drift_figure_path = cfg.paths.outputs / 'figures' / 'drift_scores.png'
    forecast_figure_path = cfg.paths.outputs / 'figures' / 'forecast_case.png'
    plot_horizon_metrics(metrics.loc[metrics['model'].str.startswith('M')], 'mae', horizon_figure_path)
    plot_horizon_metrics(metrics.loc[metrics['model'].str.startswith('M')], 'mse', mse_figure_path)
    plot_retrieval_diagnostics(evidence, retrieval_figure_path)
    plot_drift_scores(drift_evidence.loc[drift_evidence['split'] == 'test'], drift_figure_path)
    case_index = int(np.argsort(np.abs(
        test.y.mean(axis=1) - predictions['M13_trace_raf'].mean(axis=1)
    ))[len(test.y) // 2])
    plot_forecast_case(test.x[case_index], test.y[case_index], {
        'Persistence': predictions['M00_persistence'][case_index],
        'LightGBM context': predictions['C05_lightgbm_context'][case_index],
        'TRACE-RAF': predictions['M13_trace_raf'][case_index],
    }, forecast_figure_path)

    artifact_paths = [
        CONFIG_PATH, cfg.paths.outputs / 'audit' / 'data_audit.json',
        pm25_path, weather_path, event_kb_path, modeling_path,
        window_arrays_path, window_metadata_path, window_arrays_path.with_suffix('.json'),
        kb_arrays_path, kb_metadata_path, kb_arrays_path.with_suffix('.json'),
        retrieval_evidence_path, retrieval_review_path, drift_evidence_path,
        trace_retrieval_evidence_path,
        feature_effects_path, explanations_path, predictions_path, metrics_path,
        main_results_path, attrition_path, feature_control_design_path,
        subset_counts_path, subset_stats_path,
        horizon_skill_path, exceedance_path, log_metrics_path, k_sensitivity_path,
        k_predictions_path,
        drift_period_path, event_period_path, drift_diagnostics_path, drift_correlations_path,
        drift_leaf_path,
        trace_oof_audit_path, trace_design_path, trace_gate_selection_path,
        trace_subset_inference_path,
        trace_predictions_path, trace_subset_masks_path,
        event_weight_path, event_composition_path, event_category_composition_path,
        event_weight_retrieval_predictions_path,
        event_weight_model_path,
        event_weight_predictions_path, stride_path, stride_predictions_path,
        stride_model_path, stride_model_predictions_path,
        drift_comparison_path, retrieval_fusion_path, group_importance_path,
        ablation_path, tsfm_gate_path, fusion_scores_path, placebo_fusion_path,
        interval_path, quantile_calibration_path, probabilistic_path, router_path,
        horizon_figure_path, mse_figure_path, forecast_figure_path,
        retrieval_figure_path, drift_figure_path,
        drift_detector_path, no_event_detector_path, ks_detector_path, tsfm_predictions_path,
        site_level_sensitivity_path, site_level_design_path, site_selection_audit_path,
        site_level_predictions_path,
    ]
    if RUN_JOURNAL_BASELINES:
        artifact_paths.extend([neural_training_path, dlinear_path, lstm_path, patchtst_path])
    artifact_paths.extend(site_model_paths)
    artifact_paths.extend(cfg.paths.outputs / 'models' / f'{name}.joblib' for name in models_to_save)
    manifest = write_run_manifest(
        cfg, artifact_paths, run_id=RUN_ID, config_path=CONFIG_PATH,
        started_at_utc=RUN_STARTED_AT.isoformat(),
        runtime_seconds=time.perf_counter() - RUN_STARTED_PERF,
        run_options={
            'force_download': FORCE_DOWNLOAD, 'require_events': REQUIRE_EVENTS,
            'strict_event_availability': REQUIRE_STRICT_EVENT_AVAILABILITY,
            'event_availability_mode': EVENT_AVAILABILITY_MODE,
            'storm_events_cache': str(storm_cache) if storm_cache is not None else None,
            'event_delivery_modes': audit['event_delivery_modes'],
            'run_tsf_model': RUN_TSF_MODEL, 'final_experiment': FINAL_EXPERIMENT,
            'retrieval_evidence_reviewed': RETRIEVAL_EVIDENCE_REVIEWED,
            'publication_title_allowed': bool(
                tsfm_gate_status['completed']
                and RUN_JOURNAL_BASELINES
                and set(cfg.retrieval.kb_stride_values) == {1, 6, 24}
                and len(stride_model_frame) == 3 * len(cfg.retrieval.kb_stride_values)
                and len(site_level_design) == 3
            ),
            'primary_kb_stride_hours': cfg.retrieval.kb_stride_hours,
            'kb_stride_values': list(cfg.retrieval.kb_stride_values),
            'stride_model_sweep_completed': bool(
                RUN_STRIDE_MODEL_SWEEP
                and len(stride_model_frame) == 3 * len(cfg.retrieval.kb_stride_values)
            ),
            'journal_baselines_completed': RUN_JOURNAL_BASELINES,
            'event_retrieval_method': 'event_conditioned',
            'event_weight_values': list(cfg.retrieval.event_weight_values),
            'event_weight_model_sweep_completed': True,
            'drift_score_mode': cfg.drift.score_mode,
            'bootstrap_block_hours': cfg.evaluation.bootstrap_block_hours,
            'bootstrap_resamples': cfg.evaluation.bootstrap_resamples,
            'dm_hac_lags': cfg.evaluation.dm_hac_lags,
            'holm_adjustment': True,
            'placebo_fusion_controls': ['climatology', 'persistence'],
            'aqi_thresholds': list(cfg.evaluation.aqi_thresholds),
            'probabilistic_quantiles': list(cfg.evaluation.probabilistic_quantiles),
            'chronos_native_quantile_grid': bool(
                min(cfg.evaluation.probabilistic_quantiles) >= 0.1
                and max(cfg.evaluation.probabilistic_quantiles) <= 0.9
            ),
            'trace_raf_completed': True,
            'trace_raf_oof_embargo_passed': bool(pd.DataFrame(oof_rows)['embargo_passed'].all()),
            'trace_raf_residual_candidate_count': int(residual_candidate_mask.sum()),
            'trace_raf_selected_gate_strength': trace_gate.selected_strength,
            'matched_feature_count_control': True,
            'site_level_sensitivity_completed': len(site_level_design) == 3,
        },
    )

    package_path = Path('/kaggle/working') / f'event_timeraf_publication_candidate_{RUN_ID}.zip'
    if not Path('/kaggle/working').exists():
        package_path = PROJECT_ROOT / f'event_timeraf_publication_candidate_{RUN_ID}.zip'
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        package_root = PurePosixPath(f'event_timeraf_{RUN_ID}')
        for directory in ('configs', 'src', 'tests', 'notebooks'):
            for path in (PROJECT_ROOT / directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    bundle.write(path, (package_root / path.relative_to(PROJECT_ROOT).as_posix()).as_posix())
        for data_directory in (cfg.paths.processed, cfg.paths.knowledge_base):
            for path in data_directory.rglob('*'):
                if path.is_file():
                    bundle.write(path, (package_root / path.relative_to(PROJECT_ROOT).as_posix()).as_posix())
        for path in cfg.paths.outputs.rglob('*'):
            if path.is_file():
                relative = PurePosixPath('outputs') / path.relative_to(cfg.paths.outputs).as_posix()
                bundle.write(path, (package_root / relative).as_posix())
        for name in ('README.md', 'requirements.txt', 'requirements-optional.txt'):
            path = PROJECT_ROOT / name
            if path.exists():
                bundle.write(path, (package_root / name).as_posix())
    with zipfile.ZipFile(package_path) as bundle:
        package_names = set(bundle.namelist())
        if any('\\\\' in name for name in package_names):
            raise AssertionError('Backslash path found in final ZIP')
        missing_artifacts = [
            relative for relative in manifest['artifacts']
            if not Path(relative).is_absolute()
            and (package_root / PurePosixPath(relative)).as_posix() not in package_names
        ]
        if missing_artifacts:
            raise AssertionError(f'Manifest artifacts missing from final ZIP: {missing_artifacts}')
    display(metrics.loc[(metrics['horizon'].astype(str) == 'overall') &
                        (metrics['subset'] == 'all')].sort_values('mse'))
    print({'manifest': str(cfg.paths.outputs / 'logs' / 'run_manifest.json'),
           'download_this_zip': str(package_path),
           'size_mb': round(package_path.stat().st_size / 1_000_000, 1)})
    """
)

nb.cells[14].source = nb.cells[14].source.replace(
    "## 6. Frozen-TSFM", "## 7. Frozen-TSFM"
)
nb.cells[16].source = nb.cells[16].source.replace(
    "## 7. Final evaluation", "## 8. Final evaluation"
)
site_markdown = nbformat.v4.new_markdown_cell(code(
    """
    ## 6. Site-level target sensitivity

    The three highest-coverage individual monitors are evaluated separately. This arm keeps
    the shared meteorological covariates fixed to isolate whether county-median aggregation
    masks the event-retrieval effect; station-to-monitor distance is reported as a limitation.
    """
))
site_code = nbformat.v4.new_code_cell(code(
    """
    # SITE_LEVEL_SENSITIVITY_ARM
    site_cfg = replace(
        cfg,
        retrieval=replace(cfg.retrieval, kb_stride_hours=24, kb_stride_values=(24,)),
    )
    site_candidates = (
        site_coverage.loc[site_coverage['aggregation_method'].eq('single_monitor')]
        .drop_duplicates('site_id')
        .sort_values(['observed_hours', 'site_id'], ascending=[False, True])
    )
    screened_sites = []
    site_selection_rows = []
    for _, site_candidate in site_candidates.iterrows():
        candidate_id = str(site_candidate['site_id'])
        try:
            candidate_pm25, candidate_metadata = prepare_epa_site_pm25(
                epa_raw, candidate_id, site_cfg
            )
            candidate_modeling = build_modeling_table(candidate_pm25, weather, events, site_cfg)
            candidate_dataset = build_window_dataset(candidate_modeling, site_cfg)
            candidate_counts = {
                split: len(candidate_dataset.subset(split).x)
                for split in ('train', 'validation', 'test')
            }
            selected = min(candidate_counts.values()) >= site_cfg.evaluation.minimum_subset_origins
            reason = 'selected' if selected else 'fewer than minimum origins in a chronological split'
            if selected:
                screened_sites.append(
                    (site_candidate, candidate_pm25, candidate_metadata, candidate_dataset)
                )
        except Exception as error:
            candidate_counts = {'train': 0, 'validation': 0, 'test': 0}
            selected = False
            reason = f'{type(error).__name__}: {error}'
        site_selection_rows.append({
            'run_id': RUN_ID, 'site_id': candidate_id,
            'overall_observed_hours': int(site_candidate['observed_hours']),
            'overall_coverage': float(site_candidate['coverage']),
            **{f'{split}_origins': count for split, count in candidate_counts.items()},
            'selected': selected, 'reason': reason,
        })
        if len(screened_sites) == 3:
            break
    if len(screened_sites) < 3:
        raise RuntimeError(
            'Fewer than three EPA monitors have usable train/validation/test windows. '
            + '; '.join(f"{row['site_id']}: {row['reason']}" for row in site_selection_rows)
        )

    station_latitude = float(weather_station['LAT'])
    station_longitude = float(weather_station['LON'])
    site_rows = []
    site_design_rows = []
    site_prediction_arrays = {}
    site_model_paths = []
    for site_candidate, site_pm25, site_metadata, site_dataset in screened_sites:
        site_id = str(site_candidate['site_id'])
        site_key = site_id.replace('-', '_')
        site_train = site_dataset.subset('train')
        site_validation = site_dataset.subset('validation')
        site_test = site_dataset.subset('test')
        if min(len(site_train.x), len(site_validation.x), len(site_test.x)) == 0:
            raise RuntimeError(f'Site {site_id} has an empty chronological split.')

        site_kb = build_knowledge_base(site_dataset, site_cfg)
        site_retriever = HistoricalRetriever(site_kb, site_cfg)
        site_event_train = site_retriever.retrieve(site_train, method='event_conditioned')
        site_event_test = site_retriever.retrieve(site_test, method='event_conditioned')
        if not site_event_test.valid_mask.all() or not site_event_train.valid_mask.any():
            raise RuntimeError(f'Site {site_id} fails the causal retrieval eligibility gate.')

        site_context_prefixes = ('pm25_', 'weather_', 'cal_')
        site_context_train, site_context_names = origin_feature_matrix(
            site_train, site_context_prefixes
        )
        site_context_test, _ = origin_feature_matrix(site_test, site_context_prefixes)
        site_m04 = DirectXGBForecaster(site_cfg).fit(
            site_context_train, site_train.future_calendar, site_train.y,
            site_context_names, site_train.calendar_names,
        )
        site_m04_prediction = site_m04.predict(site_context_test, site_test.future_calendar)

        site_full_prefixes = ('pm25_', 'weather_', 'cal_', 'event_')
        site_retrieval_names = site_event_train.feature_names('event_conditioned_retrieval')
        site_event_matrix_train, site_event_names = origin_feature_matrix(
            site_train, site_full_prefixes, site_event_train.as_features(), site_retrieval_names
        )
        site_event_matrix_test, _ = origin_feature_matrix(
            site_test, site_full_prefixes, site_event_test.as_features(), site_retrieval_names
        )
        site_m08 = DirectXGBForecaster(site_cfg).fit(
            site_event_matrix_train[site_event_train.valid_mask],
            site_train.future_calendar[site_event_train.valid_mask],
            site_train.y[site_event_train.valid_mask], site_event_names,
            site_train.calendar_names,
        )
        site_m08_prediction = site_m08.predict(site_event_matrix_test, site_test.future_calendar)
        site_climatology = hour_month_climatology_forecast(
            site_train, site_test, site_cfg.timezone
        )
        site_event_flags = build_event_period_flags(site_test.metadata, events)[
            'target_event_flag'
        ].to_numpy(dtype=bool)
        site_prediction_arrays[f'{site_key}_actual'] = site_test.y
        site_prediction_arrays[f'{site_key}_M04'] = site_m04_prediction
        site_prediction_arrays[f'{site_key}_M08'] = site_m08_prediction
        site_prediction_arrays[f'{site_key}_climatology'] = site_climatology
        site_prediction_arrays[f'{site_key}_event_mask'] = site_event_flags

        for subset_name, subset_mask in {
            'all': np.ones(len(site_test.x), dtype=bool),
            'event': site_event_flags,
            'non_event': ~site_event_flags,
        }.items():
            eligible = int(subset_mask.sum()) >= site_cfg.evaluation.minimum_subset_origins
            climatology_mse = (
                metric_values(site_test.y[subset_mask], site_climatology[subset_mask])['mse']
                if eligible else np.nan
            )
            for model_name, model_prediction in {
                'S_M04_xgb_context': site_m04_prediction,
                'S_M08_event_retrieval': site_m08_prediction,
            }.items():
                scores = (
                    metric_values(site_test.y[subset_mask], model_prediction[subset_mask])
                    if eligible else {'mse': np.nan, 'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}
                )
                site_rows.append({
                    'run_id': RUN_ID, 'site_id': site_id, 'subset': subset_name,
                    'model': model_name, 'n_origins': int(subset_mask.sum()),
                    'eligible_for_metrics': eligible, **scores,
                    'skill_vs_site_climatology': (
                        1 - scores['mse'] / climatology_mse
                        if eligible and climatology_mse > 0 else np.nan
                    ),
                })

        monitor_latitude = float(site_metadata.loc[0, 'latitude'])
        monitor_longitude = float(site_metadata.loc[0, 'longitude'])
        station_distance = float(haversine_km(
            monitor_latitude, monitor_longitude,
            [station_latitude], [station_longitude],
        )[0])
        site_design_rows.append({
            'run_id': RUN_ID, 'site_id': site_id,
            'observed_hour_coverage': float(site_metadata.loc[0, 'coverage']),
            'train_origins': len(site_train.x), 'validation_origins': len(site_validation.x),
            'test_origins': len(site_test.x), 'knowledge_base_windows': len(site_kb.metadata),
            'monitor_latitude': monitor_latitude, 'monitor_longitude': monitor_longitude,
            'weather_station_id': f"{weather_station['USAF']}{weather_station['WBAN']}",
            'weather_station_distance_km': station_distance,
            'weather_covariate_design': 'shared primary station; target-construction sensitivity only',
        })
        for suffix, model in {'M04': site_m04, 'M08': site_m08}.items():
            model_path = cfg.paths.outputs / 'models' / f'site_{site_key}_{suffix}.joblib'
            model.save(model_path)
            site_model_paths.append(model_path)

    site_level_sensitivity = pd.DataFrame(site_rows)
    site_level_design = pd.DataFrame(site_design_rows)
    site_selection_audit = pd.DataFrame(site_selection_rows)
    site_level_sensitivity_path = cfg.paths.outputs / 'tables' / 'site_level_sensitivity.csv'
    site_level_design_path = cfg.paths.outputs / 'tables' / 'site_level_design.csv'
    site_selection_audit_path = cfg.paths.outputs / 'tables' / 'site_selection_audit.csv'
    site_level_predictions_path = cfg.paths.outputs / 'predictions' / 'site_level_predictions.npz'
    site_level_sensitivity.to_csv(site_level_sensitivity_path, index=False)
    site_level_design.to_csv(site_level_design_path, index=False)
    site_selection_audit.to_csv(site_selection_audit_path, index=False)
    np.savez_compressed(site_level_predictions_path, **site_prediction_arrays)
    display(site_level_design)
    display(site_selection_audit)
    display(site_level_sensitivity)
    """
))
nb.cells[14:14] = [site_markdown, site_code]

for index, cell in enumerate(nb.cells):
    cell.id = f"event-timeraf-{index:03d}"
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nbformat.write(nb, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")

RESULTS_NOTEBOOK = ROOT / "notebooks" / "02_results_and_figures.ipynb"
results_nb = nbformat.read(RESULTS_NOTEBOOK, as_version=4)
results_nb.cells[0].source = code(
    """
    # TRACE-RAF Publication-Candidate Results

    This notebook reads one manifest-backed final run. It does not train models or alter
    predictions. It rejects mixed run identifiers and displays the diagnostics required before
    any paper number is updated.
    """
)
results_setup = results_nb.cells[1].source
results_setup = re.sub(
    r"metrics = pd\.read_csv.*?(?=run_ids = \()",
    """metrics = pd.read_csv(cfg.paths.outputs / 'tables' / 'metrics.csv')
predictions = pd.read_parquet(cfg.paths.outputs / 'predictions' / 'predictions.parquet')
explanations = pd.read_parquet(cfg.paths.outputs / 'evidence' / 'explanations.parquet')
ablation = pd.read_csv(cfg.paths.outputs / 'tables' / 'ablation_results.csv')
subset_counts = pd.read_csv(cfg.paths.outputs / 'tables' / 'subset_counts.csv')
drift_results = pd.read_csv(cfg.paths.outputs / 'tables' / 'drift_period_results.csv')
event_results = pd.read_csv(cfg.paths.outputs / 'tables' / 'event_period_results.csv')
stride_results = pd.read_csv(cfg.paths.outputs / 'tables' / 'kb_stride_sensitivity.csv')
stride_models = pd.read_csv(cfg.paths.outputs / 'tables' / 'kb_stride_model_sensitivity.csv')
event_weights = pd.read_csv(cfg.paths.outputs / 'tables' / 'event_weight_sensitivity.csv')
event_composition = pd.read_csv(cfg.paths.outputs / 'tables' / 'event_candidate_composition.csv')
event_category_composition = pd.read_csv(cfg.paths.outputs / 'tables' / 'event_category_candidate_composition.csv')
event_weight_models = pd.read_csv(cfg.paths.outputs / 'tables' / 'event_weight_model_sensitivity.csv')
subset_statistics = pd.read_csv(cfg.paths.outputs / 'tables' / 'subset_target_statistics.csv')
drift_comparison = pd.read_csv(cfg.paths.outputs / 'tables' / 'drift_detector_comparison.csv')
drift_leaf_occupancy = pd.read_csv(cfg.paths.outputs / 'tables' / 'drift_leaf_occupancy.csv')
group_faithfulness = pd.read_csv(cfg.paths.outputs / 'tables' / 'validation_group_faithfulness.csv')
attrition = pd.read_csv(cfg.paths.outputs / 'tables' / 'window_origin_attrition.csv')
horizon_skill = pd.read_csv(cfg.paths.outputs / 'tables' / 'horizon_skill_vs_climatology.csv')
exceedance = pd.read_csv(cfg.paths.outputs / 'tables' / 'aqi_exceedance_metrics.csv')
interval_results = pd.read_csv(cfg.paths.outputs / 'tables' / 'tsfm_interval_metrics.csv')
placebo_validation = pd.read_csv(cfg.paths.outputs / 'tables' / 'tsfm_placebo_fusion_validation.csv')
log_metrics = pd.read_csv(cfg.paths.outputs / 'tables' / 'log_scale_metrics.csv')
quantile_calibration = pd.read_csv(cfg.paths.outputs / 'tables' / 'tsfm_quantile_calibration.csv')
probabilistic_metrics = pd.read_csv(cfg.paths.outputs / 'tables' / 'tsfm_probabilistic_metrics.csv')
neural_training = pd.read_csv(cfg.paths.outputs / 'tables' / 'neural_baseline_training.csv')
feature_control_design = pd.read_csv(cfg.paths.outputs / 'tables' / 'feature_count_control_design.csv')
site_level_sensitivity = pd.read_csv(cfg.paths.outputs / 'tables' / 'site_level_sensitivity.csv')
site_level_design = pd.read_csv(cfg.paths.outputs / 'tables' / 'site_level_design.csv')
site_selection_audit = pd.read_csv(cfg.paths.outputs / 'tables' / 'site_selection_audit.csv')
trace_gate_selection = pd.read_csv(cfg.paths.outputs / 'tables' / 'trace_raf_gate_selection.csv')
trace_design = pd.read_csv(cfg.paths.outputs / 'tables' / 'trace_raf_design.csv')
trace_oof_audit = pd.read_csv(cfg.paths.outputs / 'tables' / 'trace_raf_oof_audit.csv')
trace_subset_inference = pd.read_csv(cfg.paths.outputs / 'tables' / 'trace_raf_subset_inference.csv')
""",
    results_setup,
    flags=re.DOTALL,
)
results_setup = re.sub(
    r"run_ids = \(.*?\n\)",
    """run_ids = (
    set(metrics['run_id']) | set(predictions['run_id']) | set(explanations['run_id'])
    | set(ablation['run_id']) | set(subset_counts['run_id'])
    | set(drift_results['run_id']) | set(event_results['run_id'])
    | set(stride_results['run_id']) | set(stride_models['run_id'])
    | set(event_weights['run_id']) | set(event_composition['run_id'])
    | set(event_category_composition['run_id'])
    | set(event_weight_models['run_id'])
    | set(subset_statistics['run_id']) | set(drift_comparison['run_id'])
    | set(drift_leaf_occupancy['run_id'])
    | set(group_faithfulness['run_id']) | set(attrition['run_id'])
    | set(horizon_skill['run_id']) | set(exceedance['run_id'])
    | set(interval_results['run_id']) | set(placebo_validation['run_id'])
    | set(log_metrics['run_id']) | set(quantile_calibration['run_id'])
    | set(probabilistic_metrics['run_id']) | set(neural_training['run_id'])
    | set(feature_control_design['run_id'])
    | set(site_level_sensitivity['run_id']) | set(site_level_design['run_id'])
    | set(site_selection_audit['run_id'])
    | set(trace_gate_selection['run_id']) | set(trace_oof_audit['run_id'])
    | set(trace_design['run_id'])
    | set(trace_subset_inference['run_id'])
)""",
    results_setup,
    flags=re.DOTALL,
)
results_setup = re.sub(
    r"if run_options\.get\('primary_kb_stride_hours'\).*?if REQUIRE_PUBLICATION_TITLE_ALLOWED",
    """if run_options.get('primary_kb_stride_hours') != 24:
    raise RuntimeError('Publication-candidate run must use the primary 24-hour knowledge-base stride.')
if run_options.get('drift_score_mode') != 'two_sided':
    raise RuntimeError('Publication-candidate run must use two-sided drift scoring.')
if run_options.get('bootstrap_block_hours', 0) < 168:
    raise RuntimeError('Publication-candidate inference must use blocks of at least 168 origins.')
if run_options.get('bootstrap_resamples', 0) < 2000:
    raise RuntimeError('Publication-candidate inference must use at least 2,000 resamples.')
if not run_options.get('holm_adjustment'):
    raise RuntimeError('Publication-candidate inference must include Holm multiplicity adjustment.')
if set(run_options.get('placebo_fusion_controls', [])) != {'climatology', 'persistence'}:
    raise RuntimeError('Both Chronos fusion placebo controls are required.')
if set(run_options.get('kb_stride_values', [])) != {1, 6, 24}:
    raise RuntimeError('Publication-candidate run must complete 1/6/24-hour KB sensitivity.')
if not run_options.get('journal_baselines_completed'):
    raise RuntimeError('DLinear, LSTM, PatchTST, and LightGBM baselines are required.')
if not run_options.get('event_weight_model_sweep_completed'):
    raise RuntimeError('The learned event-weight sweep is required.')
if not run_options.get('matched_feature_count_control'):
    raise RuntimeError('The matched event-feature-count control is required.')
if not run_options.get('site_level_sensitivity_completed'):
    raise RuntimeError('The three-monitor site-level sensitivity arm is required.')
if not run_options.get('trace_raf_completed') or not run_options.get('trace_raf_oof_embargo_passed'):
    raise RuntimeError('TRACE-RAF and its out-of-fold temporal embargo are required.')
if not run_options.get('chronos_native_quantile_grid'):
    raise RuntimeError('Chronos probabilistic evaluation must use only its native quantile grid.')
expected_stride_rows = 3 * len(run_options.get('kb_stride_values', []))
if len(stride_models) != expected_stride_rows:
    raise RuntimeError('The learned 1/6/24-hour stride table is incomplete.')
if REQUIRE_PUBLICATION_TITLE_ALLOWED""",
    results_setup,
    flags=re.DOTALL,
)
results_nb.cells[1].source = results_setup
results_nb.cells[2].source = code(
    """
    overall = metrics.loc[
        (metrics['horizon'].astype(str) == 'overall') & (metrics['subset'] == 'all')
    ].sort_values('mse')
    display(overall)
    primary_metrics = metrics.loc[metrics['model'].str.startswith('M')]
    plot_horizon_metrics(primary_metrics, 'mse')
    plot_horizon_metrics(primary_metrics, 'mae')
    display(subset_counts)
    display(attrition)
    display(subset_statistics)
    display(ablation)
    display(exceedance)
    display(interval_results)
    display(log_metrics.sort_values('mse'))
    display(probabilistic_metrics)
    display(quantile_calibration)
    """
)
results_nb.cells[3].source = code(
    """
    display(stride_results.sort_values(['method', 'stride_hours']))
    display(stride_models.sort_values(['model', 'stride_hours']))
    display(event_weights.sort_values(['subset', 'event_weight']))
    display(event_composition)
    display(event_category_composition)
    display(event_weight_models.sort_values(['split', 'event_weight']))
    display(placebo_validation.loc[placebo_validation['selected_on_validation']])
    display(group_faithfulness.sort_values('mse_increase', ascending=False))
    display(horizon_skill.sort_values(['model', 'horizon']))
    display(neural_training.loc[neural_training['selected_epoch']])
    display(feature_control_design)
    display(site_level_design)
    display(site_selection_audit)
    display(site_level_sensitivity)
    display(trace_oof_audit)
    display(trace_design)
    display(trace_gate_selection.sort_values(['model', 'gate_strength']))
    display(trace_subset_inference.sort_values(['subset', 'comparison', 'metric']))
    display(drift_comparison)
    display(drift_leaf_occupancy)
    display(drift_results)
    display(event_results)
    display(explanations.sort_values('drift_score', ascending=False).head(10))
    """
)
for index, cell in enumerate(results_nb.cells):
    cell.id = f"trace-results-{index:03d}"
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
nbformat.write(results_nb, RESULTS_NOTEBOOK)
print(f"Wrote {RESULTS_NOTEBOOK}")
