from __future__ import annotations

import numpy as np
import pandas as pd

from event_timeraf.evaluation import (
    build_event_period_flags,
    diebold_mariano_test,
    holm_adjust,
    interval_metrics,
    metric_values,
    metrics_table,
    paired_block_bootstrap_difference,
    predictions_long,
    subset_target_summary,
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


def test_diebold_mariano_test_reports_positive_loss_difference():
    actual = np.ones((250, 24))
    worse = np.zeros_like(actual)
    better = np.full_like(actual, 0.9)
    result = diebold_mariano_test(actual, worse, better, loss="mse", hac_lag=23)
    assert result["mean_difference"] > 0
    assert result["dm_statistic"] > 0
    assert result["p_value"] < 0.05


def test_holm_adjustment_is_monotone_and_not_smaller_than_raw_values():
    raw = {"a": 0.01, "b": 0.03, "c": 0.2}
    adjusted = holm_adjust(raw)
    assert all(adjusted[name] >= value for name, value in raw.items())
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_subset_target_summary_reports_variance_and_origin_count():
    actual = np.arange(12, dtype=float).reshape(3, 4)
    summary = subset_target_summary(actual, {"all": np.ones(3, dtype=bool), "last": [False, False, True]})
    assert summary.loc[summary["subset"] == "all", "n_origins"].item() == 3
    assert summary.loc[summary["subset"] == "last", "target_variance"].item() > 0


def test_interval_metrics_reports_complete_coverage():
    actual = np.ones((3, 2))
    result = interval_metrics(actual, np.zeros_like(actual), np.full_like(actual, 2.0))
    assert result["coverage"] == 1.0
    assert result["mean_width"] == 2.0
