from __future__ import annotations

import numpy as np
import pandas as pd

from event_timeraf.evaluation import (
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
    result = predictions_long(
        actual,
        actual,
        metadata,
        "model",
        drift_flag=np.array([False, True]),
        event_flag=np.array([True, False]),
    )
    assert result["event_flag"].tolist() == [True, True, False, False]
    assert result["drift_flag"].tolist() == [False, False, True, True]
