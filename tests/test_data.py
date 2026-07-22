from __future__ import annotations

import pandas as pd

from event_timeraf.data import _event_coverage_days, prepare_noaa_weather, write_run_manifest


def test_event_coverage_counts_union_of_source_ranges():
    events = pd.DataFrame(
        {
            "source_coverage_start": pd.to_datetime(
                ["2020-01-01", "2020-01-05", "2021-01-01"], utc=True
            ),
            "source_coverage_end": pd.to_datetime(
                ["2020-01-10", "2020-01-15", "2021-01-04"], utc=True
            ),
        }
    )
    result = _event_coverage_days(
        events,
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2021-01-03", tz="UTC"),
    )
    assert result == 16


def test_noaa_precipitation_preserves_missingness(cfg):
    raw = pd.DataFrame(
        {
            "DATE": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            "TMP": ["+0100,1", "+0110,1"],
            "DEW": ["+0050,1", "+0060,1"],
            "SLP": ["10120,1", "10110,1"],
            "WND": ["090,1,N,0030,1", "100,1,N,0040,1"],
            "AA1": [pd.NA, "01,0010,1,1"],
        }
    )
    weather = prepare_noaa_weather(raw, cfg)
    assert bool(weather.loc[0, "precipitation_missing"])
    assert weather.loc[0, "precipitation_mm"] == 0.0
    assert not bool(weather.loc[1, "precipitation_missing"])
    assert weather.loc[1, "precipitation_mm"] == 1.0


def test_run_manifest_records_config_and_artifact_hashes(cfg, tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("reproducible", encoding="utf-8")
    manifest = write_run_manifest(
        cfg,
        [artifact],
        run_id="test-run",
        runtime_seconds=1.25,
        run_options={"tsfm": False},
    )
    assert manifest["run_id"] == "test-run"
    assert len(manifest["config_sha256"]) == 64
    assert manifest["runtime_seconds"] == 1.25
    assert any(record["sha256"] for record in manifest["artifacts"].values())
