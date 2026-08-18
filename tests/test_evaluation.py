from __future__ import annotations

import numpy as np
import pandas as pd

from event_timeraf.evaluation import (
    build_event_period_flags,
    diebold_mariano_hac,
    exceedance_metrics,
    holm_adjust_pvalues,
    horizon_skill_table,
    interval_metrics,
    log_scale_metrics,
    metric_values,
    metrics_table,
    paired_block_bootstrap_difference,
    paired_block_bootstrap_loss_difference,
    paired_masked_block_bootstrap_loss_difference,
    predictions_long,
    quantile_forecast_metrics,
    select_exceedance_decision_thresholds,
)


def test_metrics_are_zero_for_perfect_prediction():
    actual = np.arange(48, dtype=float).reshape(2, 24) + 1
    values = metric_values(actual, actual)
    assert values["mse"] == 0
    assert values["mae"] == 0
    assert values["rmse"] == 0
    table = metrics_table(actual, actual, "perfect")
    assert len(table) == 26


def test_metric_counts_exclude_invalid_prediction_pairs():
    actual = np.ones((2, 2))
    predicted = actual.copy()
    predicted[0, 0] = np.nan
    table = metrics_table(actual, predicted, "partial")
    assert table.loc[table["horizon"] == "overall", "n_points"].item() == 3


def test_paired_bootstrap_reports_correct_direction():
    actual = np.ones((50, 24))
    better = np.ones_like(actual)
    worse = np.zeros_like(actual)
    result = paired_block_bootstrap_difference(
        actual,
        worse,
        better,
        metric=lambda y, p: float(np.mean((y - p) ** 2)),
        block_length=5,
        resamples=20,
        seed=42,
    )
    assert result["difference"] > 0


def test_efficient_bootstrap_and_dm_report_correct_direction():
    actual = np.ones((200, 24))
    better = np.ones_like(actual)
    worse = np.zeros_like(actual)
    bootstrap = paired_block_bootstrap_loss_difference(
        actual, worse, better, "mse", block_length=24, resamples=200, seed=42
    )
    dm = diebold_mariano_hac(actual, worse, better, "mse", hac_lags=24)
    assert bootstrap["difference"] > 0
    assert bootstrap["ci_low"] > 0
    assert dm["dm_statistic"] > 0
    assert dm["dm_p_value"] < 0.05


def test_holm_adjustment_is_monotone_in_sorted_order():
    raw = np.array([0.03, 0.001, 0.02])
    adjusted = holm_adjust_pvalues(raw)
    assert np.all(adjusted >= raw)
    assert adjusted[np.argmin(raw)] == 0.003


def test_operational_and_probabilistic_metrics():
    actual = np.array([[10.0, 10.0], [40.0, 40.0], [60.0, 60.0]])
    predicted = np.array([[12.0, 12.0], [42.0, 42.0], [30.0, 30.0]])
    exceedance = exceedance_metrics(actual, predicted, [35.4, 55.4], "model")
    assert exceedance.loc[exceedance["threshold_ug_m3"] == 35.4, "recall"].item() == 0.5
    assert exceedance.loc[exceedance["threshold_ug_m3"] == 55.4, "fn"].item() == 1

    intervals = interval_metrics(actual, predicted - 5, predicted + 5, 0.2, "model")
    assert 0 <= intervals.loc[0, "empirical_coverage"] <= 1
    assert intervals.loc[0, "mean_width"] == 10

    skill = horizon_skill_table(actual, actual, predicted, "perfect")
    assert np.allclose(skill["skill_vs_climatology"], 1.0)

    log_values = log_scale_metrics(actual, actual, "perfect")
    assert log_values.loc[0, "mse"] == 0

    levels = (0.1, 0.5, 0.9)
    quantiles = np.stack([actual - 5, actual, actual + 5], axis=-1)
    calibration, probabilistic = quantile_forecast_metrics(
        actual, quantiles, levels, "model"
    )
    assert len(calibration) == len(levels)
    assert probabilistic.loc[0, "crps_quantile_approximation"] >= 0


def test_exceedance_decision_threshold_is_selected_only_from_validation():
    validation_actual = np.array([[10.0], [36.0], [40.0], [12.0]])
    validation_predicted = np.array([[8.0], [30.0], [32.0], [15.0]])
    selection = select_exceedance_decision_thresholds(
        validation_actual, validation_predicted, [35.4], "model"
    )
    decision = selection.loc[0, "selected_decision_threshold_ug_m3"]
    assert decision < 35.4
    test_actual = np.array([[10.0], [38.0]])
    test_predicted = np.array([[12.0], [31.0]])
    calibrated = exceedance_metrics(
        test_actual,
        test_predicted,
        [35.4],
        "model",
        decision_thresholds={35.4: decision},
    )
    assert calibrated.loc[0, "decision_rule"] == "validation_calibrated"
    assert calibrated.loc[0, "recall"] == 1.0


def test_long_predictions_preserve_event_and_drift_labels():
    actual = np.ones((2, 2))
    metadata = pd.DataFrame(
        {
            "window_id": ["w1", "w2"],
            "origin_time": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "split": ["test", "test"],
        }
    )
    event_flags = pd.DataFrame(
        {
            "recent_event_flag": [True, False],
            "active_event_flag": [False, False],
            "target_event_flag": [True, False],
        }
    )
    result = predictions_long(
        actual,
        actual,
        metadata,
        "model",
        seed=42,
        run_id="run-1",
        drift_flag=np.array([False, True]),
        drift_score=np.array([0.1, 0.9]),
        event_flags=event_flags,
        event_availability_mode="retrospective_event_start",
    )
    assert result["event_flag"].tolist() == [True, True, False, False]
    assert result["drift_flag"].tolist() == [False, False, True, True]
    assert result["seed"].eq(42).all()
    assert result["run_id"].eq("run-1").all()


def test_event_period_flags_separate_recent_active_and_target_overlap():
    metadata = pd.DataFrame(
        {
            "origin_time": pd.to_datetime(["2024-01-01 12:00"], utc=True),
            "target_start": pd.to_datetime(["2024-01-01 13:00"], utc=True),
            "target_end": pd.to_datetime(["2024-01-02 12:00"], utc=True),
        }
    )
    events = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2024-01-01 18:00"], utc=True),
            "event_end": pd.to_datetime(["2024-01-01 20:00"], utc=True),
            "published_at": pd.to_datetime(["2024-01-01 18:00"], utc=True),
        }
    )
    flags = build_event_period_flags(metadata, events)
    assert not flags.loc[0, "recent_event_flag"]
    assert not flags.loc[0, "active_event_flag"]
    assert flags.loc[0, "target_event_flag"]


def test_masked_block_bootstrap_preserves_subset_direction():
    actual = np.ones((240, 4), dtype=float)
    better = actual.copy()
    worse = actual.copy()
    mask = np.zeros(240, dtype=bool)
    mask[::6] = True
    worse[mask] = 0.0
    result = paired_masked_block_bootstrap_loss_difference(
        actual, better, worse, mask, "mse", block_length=24,
        resamples=500, seed=42,
    )
    assert result["difference"] < 0
    assert result["subset_origins"] == int(mask.sum())
    assert result["ci_high"] < 0
