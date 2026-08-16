from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from event_timeraf.config import load_config
from event_timeraf.features import CALENDAR_FEATURES


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(Path(__file__).parents[1] / "configs" / "default.yaml", project_root=tmp_path)


@pytest.fixture
def modeling_frame(cfg):
    rows = 3_000
    timestamp = pd.date_range("2019-01-01", periods=rows, freq="h", tz="UTC")
    phase = np.arange(rows)
    pm25 = 18 + 5 * np.sin(2 * np.pi * phase / 24) + 0.01 * phase
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamp,
            "timestamp_local": timestamp.tz_convert(cfg.timezone),
            "site_id": "06-037-0001",
            "pm25": pm25,
            "pm25_observed": pm25,
            "pm25_filled": False,
            "pm25_current": pm25,
            "pm25_roll_mean_24h": pd.Series(pm25).rolling(24, min_periods=1).mean(),
            "pm25_roll_mean_168h": pd.Series(pm25).rolling(168, min_periods=1).mean(),
            "pm25_roll_std_24h": pd.Series(pm25).rolling(24, min_periods=2).std(ddof=0).fillna(0.1),
            "pm25_roll_std_168h": pd.Series(pm25).rolling(168, min_periods=2).std(ddof=0).fillna(0.1),
            "weather_temperature_c": 20 + np.sin(2 * np.pi * phase / 24),
            "weather_relative_humidity": 55 + np.cos(2 * np.pi * phase / 24),
            "weather_pressure_hpa": 1012.0,
            "weather_wind_speed_ms": 3.0,
            "event_count_24h": 0.0,
            "event_count_72h": 0.0,
            "event_burst_ratio": 0.0,
        }
    )
    local = timestamp.tz_convert(cfg.timezone)
    calendar = {
        "cal_hour_sin": np.sin(2 * np.pi * local.hour / 24),
        "cal_hour_cos": np.cos(2 * np.pi * local.hour / 24),
        "cal_dow_sin": np.sin(2 * np.pi * local.dayofweek / 7),
        "cal_dow_cos": np.cos(2 * np.pi * local.dayofweek / 7),
        "cal_month_sin": np.sin(2 * np.pi * (local.month - 1) / 12),
        "cal_month_cos": np.cos(2 * np.pi * (local.month - 1) / 12),
        "cal_is_weekend": (local.dayofweek >= 5).astype(int),
        "cal_is_us_federal_holiday": 0,
        "cal_is_california_holiday": 0,
    }
    for name in CALENDAR_FEATURES:
        frame[name] = calendar[name]
    return frame
