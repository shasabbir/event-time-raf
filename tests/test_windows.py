from __future__ import annotations

import numpy as np

from event_timeraf.models import daily_seasonal_forecast, persistence_forecast, weekly_seasonal_forecast
from event_timeraf.windows import assert_window_integrity, build_window_dataset, window_attrition_table


def test_window_shapes_and_chronology(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    assert dataset.x.shape[1] == 168
    assert dataset.y.shape[1] == 24
    assert dataset.future_calendar.shape[1] == 24
    assert set(dataset.metadata["split"]) == {"train", "validation", "test"}
    assert_window_integrity(dataset, cfg)


def test_missing_target_is_never_retained(modeling_frame, cfg):
    modeling_frame.loc[900, "pm25_observed"] = np.nan
    dataset = build_window_dataset(modeling_frame, cfg)
    assert not np.isnan(dataset.y).any()
    target_ranges = dataset.metadata[["target_start", "target_end"]]
    missing_time = modeling_frame.loc[900, "timestamp_utc"]
    assert not ((target_ranges["target_start"] <= missing_time) & (target_ranges["target_end"] >= missing_time)).any()


def test_naive_forecasts_have_expected_alignment(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg).subset("test")
    persistence = persistence_forecast(dataset.x, 24)
    daily = daily_seasonal_forecast(dataset.x, 24)
    weekly = weekly_seasonal_forecast(dataset.x, 24)
    assert np.allclose(persistence[:, 0], dataset.x[:, -1])
    assert np.allclose(daily, dataset.x[:, -24:])
    assert np.allclose(weekly, dataset.x[:, :24])


def test_window_attrition_reconciles_retained_windows(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    attrition = window_attrition_table(modeling_frame, dataset, cfg, run_id="test")
    assert attrition.iloc[-1]["stage"] == "retained_windows"
    assert attrition.iloc[-1]["count"] == len(dataset.x)
    assert (attrition["removed_since_previous"] >= 0).all()

