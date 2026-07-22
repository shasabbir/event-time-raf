from __future__ import annotations

import numpy as np
import pandas as pd

from event_timeraf.evaluation import (
    build_event_period_flags,
    metric_values,
    metrics_table,
    paired_block_bootstrap_difference,
    predictions_long,
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
