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
nb.cells[0].source = code(
    """
    # Event-TimeRAF: Publication-Candidate Kaggle Pipeline

    This experiment log runs the complete study from official cached sources. It uses a dense,
    leakage-safe historical knowledge base, explicit event-conditioned retrieval, two-sided
    drift evidence, real component ablations, a frozen Chronos baseline, and manifest-backed
    outputs. Numerical paper claims must be updated only from the ZIP produced by the final cell.
    """
)

setup = nb.cells[2].source
setup = re.sub(r"(?:DirectRidgeForecaster, ){2,}", "DirectRidgeForecaster, ", setup)
setup = re.sub(
    r"(?:RUN_STRIDE_MODEL_SWEEP = True\n){2,}",
    "RUN_STRIDE_MODEL_SWEEP = True\n",
    setup,
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
if "Clear stale writable outputs before a publication run" not in setup:
    setup = setup.replace(
        "# Final-publication profile.",
        """# Clear stale writable outputs before a publication run. Raw caches are outside this path.
if Path('/kaggle/working').exists() and cfg.paths.outputs.resolve().is_relative_to(Path('/kaggle/working').resolve()):
    reset_directory(cfg.paths.outputs)
    cfg.paths.create()

# Final-publication profile.""",
    )
nb.cells[2].source = setup

nb.cells[7].source = code(
    """
    ## 3. Dense leakage-safe and event-conditioned retrieval

    Candidate windows may overlap one another. For every query, the complete candidate target
    must still end before the query lookback begins. The primary 24-hour stride is evaluated
    against 192-hour and 6-hour alternatives without changing this embargo.
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

    required_results = [result for method in retrievals.values() for result in method.values()]
    if not all(result.valid_mask.all() for result in required_results):
        raise RuntimeError('A query has no causally eligible retrieval candidate.')
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
    display(review_sample.drop(columns=['aligned_future']).head(20))
    if FINAL_EXPERIMENT and not RETRIEVAL_EVIDENCE_REVIEWED:
        raise RuntimeError('Review retrieval_review_sample.csv before the final run.')
    """
)

nb.cells[9].source = code(
    """
    ## 4. Baselines, regularized control, and Event-TimeRAF variants

    All learned variants use the same chronological splits. M08 uses the event-conditioned
    retriever, while A00 removes event inputs and event-conditioned retrieval under an otherwise
    comparable direct-model design.
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
    no_event_drift_validation = no_event_detector.transform(validation, no_event_validation.mean_similarity)
    no_event_drift_test = no_event_detector.transform(test, no_event_test.mean_similarity)
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
        'A00_full_without_events': m09_no_events,
        'A01_xgb_random_retrieval': random_model, 'C01_ridge_context': ridge,
    }
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
    for method in ('cosine', 'event_conditioned'):
        for candidate_k in cfg.retrieval.k_values:
            result = retriever.retrieve(test, method=method, k=candidate_k)
            k_rows.append({'run_id': RUN_ID, 'method': method, 'k': candidate_k,
                           **metric_values(test.y, result.prediction)})
    k_sensitivity_path = cfg.paths.outputs / 'tables' / 'k_sensitivity_results.csv'
    pd.DataFrame(k_rows).to_csv(k_sensitivity_path, index=False)

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
    pd.DataFrame(event_weight_rows).to_csv(event_weight_path, index=False)
    pd.DataFrame(composition_rows).to_csv(event_composition_path, index=False)

    stride_rows = []
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
            stride_rows.append({
                'run_id': RUN_ID, 'stride_hours': stride, 'method': method,
                'candidate_count': len(stride_kb.metadata), 'runtime_seconds': elapsed,
                'minimum_eligible_candidates': int(result.eligible_candidate_count.min()),
                **metric_values(test.y, result.prediction),
            })
    stride_path = cfg.paths.outputs / 'tables' / 'kb_stride_sensitivity.csv'
    pd.DataFrame(stride_rows).to_csv(stride_path, index=False)

    stride_model_rows = []
    if RUN_STRIDE_MODEL_SWEEP:
        for stride in cfg.retrieval.kb_stride_values:
            if stride == cfg.retrieval.kb_stride_hours:
                for name in ('M07_xgb_cosine', 'M08_event_timeraf_no_drift', 'M09_event_timeraf_full'):
                    stride_model_rows.append({'run_id': RUN_ID, 'stride_hours': stride,
                                              'model': name, **metric_values(test.y, predictions[name])})
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
                stride_model_rows.append({'run_id': RUN_ID, 'stride_hours': stride,
                                          'model': name, **metric_values(test.y, values)})
    stride_model_path = cfg.paths.outputs / 'tables' / 'kb_stride_model_sensitivity.csv'
    pd.DataFrame(stride_model_rows).to_csv(stride_model_path, index=False)

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
        tsfm_validation, tsfm_val_low, tsfm_val_high = chronos_forecast(
            validation.x, cfg.forecast.horizon, cfg.tsfm.checkpoint, cfg.tsfm.batch_size
        )
        tsfm_test, tsfm_test_low, tsfm_test_high = chronos_forecast(
            test.x, cfg.forecast.horizon, cfg.tsfm.checkpoint, cfg.tsfm.batch_size
        )
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
        tsfm_predictions_path = cfg.paths.outputs / 'predictions' / 'tsfm_predictions.npz'
        np.savez_compressed(
            tsfm_predictions_path, validation_mean=tsfm_validation,
            validation_lower=tsfm_val_low, validation_upper=tsfm_val_high,
            test_mean=tsfm_test, test_lower=tsfm_test_low, test_upper=tsfm_test_high,
            fused_test_mean=predictions['M11_chronos_event_retrieval'],
            fusion_weight=selected_weight,
        )
        fusion_scores_path = cfg.paths.outputs / 'tables' / 'tsfm_fusion_validation.csv'
        pd.DataFrame([{'run_id': RUN_ID, 'tsfm_weight': weight, 'validation_mse': score}
                      for weight, score in fusion_scores.items()]).to_csv(fusion_scores_path, index=False)
        interval_path = cfg.paths.outputs / 'tables' / 'tsfm_interval_metrics.csv'
        pd.DataFrame([{
            'run_id': RUN_ID, 'nominal_interval': '10--90%',
            'empirical_coverage': float(np.mean((test.y >= tsfm_test_low) & (test.y <= tsfm_test_high))),
            'mean_width': float(np.mean(tsfm_test_high - tsfm_test_low)),
        }]).to_csv(interval_path, index=False)
        tsfm_gate_status = {'run_id': RUN_ID, 'completed': True,
                            'checkpoint': cfg.tsfm.checkpoint,
                            'selected_fusion_weight': selected_weight}
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
    print({'tsfm_weight': selected_weight, 'drift_router': router_selection})
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

    metric_functions = {
        'mse': lambda y, p: float(np.mean((y - p) ** 2)),
        'mae': lambda y, p: float(np.mean(np.abs(y - p))),
    }
    comparisons = {
        'M04_minus_M03_weather_calendar': (predictions['M04_xgb_context'], predictions['M03_xgb_pm25']),
        'M07_minus_M04_cosine_retrieval': (predictions['M07_xgb_cosine'], predictions['M04_xgb_context']),
        'M08_minus_M07_event_conditioning': (predictions['M08_event_timeraf_no_drift'], predictions['M07_xgb_cosine']),
        'M09_minus_M08_drift_features': (predictions['M09_event_timeraf_full'], predictions['M08_event_timeraf_no_drift']),
        'M09_minus_A00_events': (predictions['M09_event_timeraf_full'], predictions['A00_full_without_events']),
        'M09_minus_M04_full': (predictions['M09_event_timeraf_full'], predictions['M04_xgb_context']),
        'M11_minus_M10_retrieval': (predictions['M11_chronos_event_retrieval'], predictions['M10_frozen_chronos']),
        'M12_minus_M04_drift_router': (predictions['M12_validation_drift_router'], predictions['M04_xgb_context']),
    }
    bootstrap_rows = []
    for comparison_name, (prediction_a, prediction_b) in comparisons.items():
        for metric_name, metric_fn in metric_functions.items():
            result = paired_block_bootstrap_difference(
                test.y, prediction_a, prediction_b, metric_fn,
                cfg.evaluation.bootstrap_block_hours,
                cfg.evaluation.bootstrap_resamples, cfg.seed,
            )
            bootstrap_rows.append({'run_id': RUN_ID, 'comparison': comparison_name,
                                   'metric': metric_name, **result})
    ablation_path = cfg.paths.outputs / 'tables' / 'ablation_results.csv'
    pd.DataFrame(bootstrap_rows).to_csv(ablation_path, index=False)

    fusion_rows = []
    for method, result in {'cosine': cosine_test, 'event_conditioned': event_test}.items():
        for aggregation, values in {'uniform': result.prediction,
                                    'similarity_weighted': result.weighted_prediction}.items():
            fusion_rows.append({'run_id': RUN_ID, 'method': method,
                                'aggregation': aggregation, **metric_values(test.y, values)})
    retrieval_fusion_path = cfg.paths.outputs / 'tables' / 'retrieval_fusion_ablation.csv'
    pd.DataFrame(fusion_rows).to_csv(retrieval_fusion_path, index=False)

    rng = np.random.default_rng(cfg.seed)
    baseline_validation_mse = metric_values(
        validation.y, validation_predictions['M08_event_timeraf_no_drift']
    )['mse']
    group_rows = []
    for group, prefix in {'pm25': 'pm25_', 'weather': 'weather_', 'calendar': 'cal_',
                          'event': 'event_', 'retrieval': 'event_conditioned_retrieval_'}.items():
        columns = [i for i, name in enumerate(m08_names) if name.startswith(prefix)]
        if not columns:
            continue
        permuted = m08_validation.copy()
        order = rng.permutation(len(permuted))
        permuted[:, columns] = permuted[order][:, columns]
        permuted_mse = metric_values(
            validation.y, m08.predict(permuted, validation.future_calendar)
        )['mse']
        group_rows.append({'run_id': RUN_ID, 'feature_group': group,
                           'validation_mse': permuted_mse,
                           'mse_increase': permuted_mse - baseline_validation_mse})
    group_importance_path = cfg.paths.outputs / 'tables' / 'validation_group_permutation.csv'
    pd.DataFrame(group_rows).to_csv(group_importance_path, index=False)

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
        test.y.mean(axis=1) - predictions['M09_event_timeraf_full'].mean(axis=1)
    ))[len(test.y) // 2])
    plot_forecast_case(test.x[case_index], test.y[case_index], {
        'Persistence': predictions['M00_persistence'][case_index],
        'XGBoost context': predictions['M04_xgb_context'][case_index],
        'Event-TimeRAF': predictions['M09_event_timeraf_full'][case_index],
    }, forecast_figure_path)

    artifact_paths = [
        CONFIG_PATH, cfg.paths.outputs / 'audit' / 'data_audit.json',
        pm25_path, weather_path, event_kb_path, modeling_path,
        window_arrays_path, window_metadata_path, window_arrays_path.with_suffix('.json'),
        kb_arrays_path, kb_metadata_path, kb_arrays_path.with_suffix('.json'),
        retrieval_evidence_path, retrieval_review_path, drift_evidence_path,
        feature_effects_path, explanations_path, predictions_path, metrics_path,
        main_results_path, subset_counts_path, subset_stats_path, k_sensitivity_path,
        drift_period_path, event_period_path, drift_diagnostics_path, drift_correlations_path,
        event_weight_path, event_composition_path, stride_path, stride_model_path,
        drift_comparison_path, retrieval_fusion_path, group_importance_path,
        ablation_path, tsfm_gate_path, fusion_scores_path, interval_path, router_path,
        horizon_figure_path, mse_figure_path, forecast_figure_path,
        retrieval_figure_path, drift_figure_path,
        drift_detector_path, no_event_detector_path, ks_detector_path, tsfm_predictions_path,
    ]
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
            'publication_title_allowed': bool(tsfm_gate_status['completed']),
            'primary_kb_stride_hours': cfg.retrieval.kb_stride_hours,
            'kb_stride_values': list(cfg.retrieval.kb_stride_values),
            'stride_model_sweep_completed': RUN_STRIDE_MODEL_SWEEP,
            'event_retrieval_method': 'event_conditioned',
            'event_weight_values': list(cfg.retrieval.event_weight_values),
            'drift_score_mode': cfg.drift.score_mode,
            'bootstrap_block_hours': cfg.evaluation.bootstrap_block_hours,
        },
    )

    package_path = Path('/kaggle/working') / f'event_timeraf_publication_candidate_{RUN_ID}.zip'
    if not Path('/kaggle/working').exists():
        package_path = PROJECT_ROOT / f'event_timeraf_publication_candidate_{RUN_ID}.zip'
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        package_root = PurePosixPath(f'event_timeraf_{RUN_ID}')
        for directory in ('configs', 'src'):
            for path in (PROJECT_ROOT / directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    bundle.write(path, (package_root / path.relative_to(PROJECT_ROOT).as_posix()).as_posix())
        for path in cfg.paths.outputs.rglob('*'):
            if path.is_file():
                relative = PurePosixPath('outputs') / path.relative_to(cfg.paths.outputs).as_posix()
                bundle.write(path, (package_root / relative).as_posix())
        for name in ('01_event_timeraf_kaggle_pipeline.ipynb', '02_results_and_figures.ipynb'):
            path = PROJECT_ROOT / 'notebooks' / name
            if path.exists():
                bundle.write(path, (package_root / 'notebooks' / name).as_posix())
    with zipfile.ZipFile(package_path) as bundle:
        if any('\\\\' in name for name in bundle.namelist()):
            raise AssertionError('Backslash path found in final ZIP')
    display(metrics.loc[(metrics['horizon'].astype(str) == 'overall') &
                        (metrics['subset'] == 'all')].sort_values('mse'))
    print({'manifest': str(cfg.paths.outputs / 'logs' / 'run_manifest.json'),
           'download_this_zip': str(package_path),
           'size_mb': round(package_path.stat().st_size / 1_000_000, 1)})
    """
)

for cell in nb.cells:
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
    # Event-TimeRAF Publication-Candidate Results

    This notebook reads one manifest-backed final run. It does not train models or alter
    predictions. It rejects mixed run identifiers and displays the diagnostics required before
    any paper number is updated.
    """
)
results_setup = results_nb.cells[1].source
if "stride_results = pd.read_csv" not in results_setup:
    results_setup = results_setup.replace(
    "event_results = pd.read_csv(cfg.paths.outputs / 'tables' / 'event_period_results.csv')",
    """event_results = pd.read_csv(cfg.paths.outputs / 'tables' / 'event_period_results.csv')
stride_results = pd.read_csv(cfg.paths.outputs / 'tables' / 'kb_stride_sensitivity.csv')
stride_models = pd.read_csv(cfg.paths.outputs / 'tables' / 'kb_stride_model_sensitivity.csv')
event_weights = pd.read_csv(cfg.paths.outputs / 'tables' / 'event_weight_sensitivity.csv')
event_composition = pd.read_csv(cfg.paths.outputs / 'tables' / 'event_candidate_composition.csv')
subset_statistics = pd.read_csv(cfg.paths.outputs / 'tables' / 'subset_target_statistics.csv')
drift_comparison = pd.read_csv(cfg.paths.outputs / 'tables' / 'drift_detector_comparison.csv')
group_permutation = pd.read_csv(cfg.paths.outputs / 'tables' / 'validation_group_permutation.csv')""",
    ).replace(
    "| set(drift_results['run_id']) | set(event_results['run_id'])",
    """| set(drift_results['run_id']) | set(event_results['run_id'])
    | set(stride_results['run_id']) | set(stride_models['run_id'])
    | set(event_weights['run_id']) | set(event_composition['run_id'])
    | set(subset_statistics['run_id']) | set(drift_comparison['run_id'])
    | set(group_permutation['run_id'])""",
    ).replace(
    "'run_tsf_model': run_options.get('run_tsf_model'),",
    """'run_tsf_model': run_options.get('run_tsf_model'),
        'stride_model_sweep_completed': run_options.get('stride_model_sweep_completed'),""",
    ).replace(
    "if REQUIRE_PUBLICATION_TITLE_ALLOWED and not run_options.get('publication_title_allowed'):",
    """if run_options.get('primary_kb_stride_hours') != 24:
    raise RuntimeError('Publication-candidate run must use the primary 24-hour knowledge-base stride.')
if run_options.get('drift_score_mode') != 'two_sided':
    raise RuntimeError('Publication-candidate run must use two-sided drift scoring.')
if run_options.get('bootstrap_block_hours', 0) < 168:
    raise RuntimeError('Publication-candidate inference must use blocks of at least 168 origins.')
if REQUIRE_PUBLICATION_TITLE_ALLOWED and not run_options.get('publication_title_allowed'):""",
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
    display(subset_statistics)
    display(ablation)
    """
)
results_nb.cells[3].source = code(
    """
    display(stride_results.sort_values(['method', 'stride_hours']))
    display(stride_models.sort_values(['model', 'stride_hours']))
    display(event_weights.sort_values(['subset', 'event_weight']))
    display(event_composition)
    display(group_permutation.sort_values('mse_increase', ascending=False))
    display(drift_comparison)
    display(drift_results)
    display(event_results)
    display(explanations.sort_values('drift_score', ascending=False).head(10))
    """
)
for cell in results_nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
nbformat.write(results_nb, RESULTS_NOTEBOOK)
print(f"Wrote {RESULTS_NOTEBOOK}")
