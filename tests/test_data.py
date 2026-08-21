from __future__ import annotations

import hashlib
import gzip
import json
from dataclasses import replace

import pandas as pd
import pytest
import requests

from event_timeraf.data import (
    NOAA_GLOBAL_HOURLY_URL,
    NOAA_GHCNH_URL,
    NOAA_ISD_END,
    NOAA_ISD_HISTORY_URL,
    _event_coverage_days,
    _ghcnh_station_id,
    _ghcnh_to_isd_schema,
    _noaa_cache_covers_required_period,
    download_file,
    load_storm_events_cache,
    noaa_station_candidates,
    prepare_epa_pm25,
    prepare_epa_site_pm25,
    prepare_noaa_weather,
    prepare_storm_events,
    write_run_manifest,
)


def test_noaa_weather_uses_official_nodd_endpoints():
    assert NOAA_ISD_HISTORY_URL == "https://noaa-isd-pds.s3.amazonaws.com/isd-history.csv"
    assert NOAA_GLOBAL_HOURLY_URL == "https://noaa-global-hourly-pds.s3.amazonaws.com"
    assert NOAA_GHCNH_URL.endswith("hourly/access/by-year")
    assert NOAA_ISD_END == pd.Timestamp("2025-08-29", tz="UTC")


def test_ghcnh_station_mapping_and_schema_conversion(cfg):
    station = pd.Series({"CTRY": "US", "WBAN": "23152"})
    assert _ghcnh_station_id(station) == "USW00023152"
    ghcnh = pd.DataFrame(
        {
            "Year": [2025], "Month": [9], "Day": [1], "Hour": [0], "Minute": [0],
            "temperature": [21.6], "dew_point_temperature": [14.2],
            "sea_level_pressure": [1012.8], "station_level_pressure": [987.4],
            "wind_direction": [270], "wind_speed": [3.5], "precipitation": [0.2],
        }
    )
    converted = _ghcnh_to_isd_schema(ghcnh)
    weather = prepare_noaa_weather(converted, cfg)
    assert weather.loc[0, "timestamp_utc"] == pd.Timestamp("2025-09-01", tz="UTC")
    assert weather.loc[0, "temperature_c"] == pytest.approx(21.6)
    assert weather.loc[0, "pressure_hpa"] == pytest.approx(1012.8)
    assert weather.loc[0, "wind_direction_deg"] == pytest.approx(270)
    assert weather.loc[0, "wind_speed_ms"] == pytest.approx(3.5)
    assert weather.loc[0, "precipitation_mm"] == pytest.approx(0.2)
    assert weather.loc[0, "weather_source"] == "NOAA GHCNh"


def test_station_inventory_lag_does_not_reject_final_year_overlap(cfg):
    inventory_path = cfg.paths.raw / "noaa_isd" / "isd-history.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "USAF": "722950", "WBAN": "23174", "STATION NAME": "LOS ANGELES AIRPORT",
                "CTRY": "US", "ICAO": "KLAX", "LAT": 33.94, "LON": -118.40,
                "BEGIN": "19440101", "END": "20250827",
            }
        ]
    ).to_csv(inventory_path, index=False)
    candidates = noaa_station_candidates(cfg, 34.05, -118.25)
    assert candidates.iloc[0]["USAF"] == "722950"


def test_download_error_explains_kaggle_internet_or_cache(monkeypatch, tmp_path):
    def fail_request(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", fail_request)
    with pytest.raises(RuntimeError, match="Enable Internet in the Kaggle notebook"):
        download_file("https://example.invalid/data.csv", tmp_path / "data.csv")


def test_noaa_cache_must_reach_configured_study_end(cfg, tmp_path):
    cache = tmp_path / "station_2025.csv"
    pd.DataFrame(
        {"DATE": pd.date_range("2025-01-01", "2025-08-24 23:00", freq="h", tz="UTC")}
    ).to_csv(cache, index=False)
    assert not _noaa_cache_covers_required_period(cache, 2025, cfg)

    pd.DataFrame(
        {"DATE": pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h", tz="UTC")}
    ).to_csv(cache, index=False)
    assert _noaa_cache_covers_required_period(cache, 2025, cfg)


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


def test_attached_storm_cache_filters_study_year_and_los_angeles(cfg, tmp_path):
    source = tmp_path / "StormEvents_details-ftp_v1.0_d2019_c20260323.csv.gz"
    pd.DataFrame(
        {
            "EVENT_ID": [1, 2],
            "YEAR": [2019, 2019],
            "STATE": ["CALIFORNIA", "CALIFORNIA"],
            "CZ_FIPS": [37, 59],
            "CZ_TYPE": ["C", "C"],
            "CZ_NAME": ["LOS ANGELES", "ORANGE"],
            "BEGIN_DATE_TIME": ["01-JAN-19 00:00:00", "01-JAN-19 00:00:00"],
            "END_DATE_TIME": ["01-JAN-19 01:00:00", "01-JAN-19 01:00:00"],
            "EVENT_TYPE": ["High Wind", "Flood"],
            "EVENT_NARRATIVE": ["LA event", "Orange event"],
        }
    ).to_csv(source, index=False, compression="gzip")
    manifest = {
        "source_name": "NOAA NCEI Storm Events",
        "official_index_url": "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
        "files": [
            {
                "name": source.name,
                "url": f"https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/{source.name}",
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    raw = load_storm_events_cache(tmp_path, cfg)
    assert raw["EVENT_ID"].tolist() == [1]
    assert raw["SOURCE_ARCHIVE_YEAR"].tolist() == [2019]
    events = prepare_storm_events(raw, cfg)
    assert events["delivery_mode"].eq("verified_official_noaa_cache").all()


def test_attached_storm_cache_accepts_manifest_verified_decompressed_csv(cfg, tmp_path):
    archive = tmp_path / "StormEvents_details-ftp_v1.0_d2020_c20260323.csv.gz"
    csv_path = tmp_path / "StormEvents_details-ftp_v1.0_d2020_c20260323.csv"
    pd.DataFrame(
        {
            "EVENT_ID": [10],
            "YEAR": [2020],
            "STATE": ["CALIFORNIA"],
            "CZ_FIPS": [37],
            "CZ_TYPE": ["C"],
            "CZ_NAME": ["LOS ANGELES"],
            "BEGIN_DATE_TIME": ["01-JAN-20 00:00:00"],
            "END_DATE_TIME": ["01-JAN-20 01:00:00"],
            "EVENT_TYPE": ["High Wind"],
            "EVENT_NARRATIVE": ["LA event"],
        }
    ).to_csv(archive, index=False, compression="gzip")
    with gzip.open(archive, "rb") as source:
        csv_bytes = source.read()
    csv_path.write_bytes(csv_bytes)
    manifest = {
        "source_name": "NOAA NCEI Storm Events",
        "official_index_url": "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
        "files": [
            {
                "name": archive.name,
                "url": f"https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/{archive.name}",
                "bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "uncompressed_name": csv_path.name,
                "uncompressed_bytes": len(csv_bytes),
                "uncompressed_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            }
        ],
    }
    archive.unlink()
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    raw = load_storm_events_cache(tmp_path, cfg)
    assert raw["EVENT_ID"].tolist() == [10]
    assert raw["SOURCE_ARCHIVE_YEAR"].tolist() == [2020]


def test_attached_storm_cache_rejects_unverified_combined_file(cfg, tmp_path):
    source = tmp_path / "StormEvents_details.csv"
    source.write_text("EVENT_ID,YEAR\n1,2019\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_manifest.json"):
        load_storm_events_cache(tmp_path, cfg)


def test_pm25_falls_back_to_county_hourly_median_when_sites_are_sparse(cfg):
    hours = pd.date_range("2024-01-01", periods=7_000, freq="h", tz="UTC")
    rows = []
    for i, timestamp in enumerate(hours):
        site_num = "0001" if i % 2 == 0 else "0002"
        rows.append(
            {
                "State Code": "06",
                "County Code": "037",
                "Site Num": site_num,
                "Parameter Code": 88101,
                "POC": 1,
                "Latitude": 34.05 if site_num == "0001" else 34.10,
                "Longitude": -118.25 if site_num == "0001" else -118.30,
                "Date GMT": timestamp.strftime("%Y-%m-%d"),
                "Time GMT": timestamp.strftime("%H:%M"),
                "Sample Measurement": 10.0 + (i % 5),
                "Units of Measure": "Micrograms/cubic meter (LC)",
                "Sample Duration": "1 HOUR",
            }
        )
    test_cfg = replace(
        cfg, data=replace(cfg.data, start_year=2024, end_year=2024, study_end_date=None)
    )
    pm25, coverage = prepare_epa_pm25(pd.DataFrame(rows), test_cfg)
    selected = coverage.iloc[0]
    assert selected["site_id"] == "06-037-COUNTY"
    assert selected["aggregation_method"] == "county_hourly_median"
    assert selected["coverage"] >= test_cfg.data.minimum_pm25_coverage
    assert pm25["site_id"].eq("06-037-COUNTY").all()
    assert pm25["pm25_observed"].notna().mean() >= test_cfg.data.minimum_pm25_coverage


def test_prepare_epa_site_pm25_keeps_named_monitor(cfg):
    hours = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    rows = []
    for site_num in ("0001", "0002"):
        for timestamp in hours:
            rows.append(
                {
                    "State Code": "06", "County Code": "037", "Site Num": site_num,
                    "POC": 1, "Latitude": 34.05, "Longitude": -118.25,
                    "Date GMT": timestamp.strftime("%Y-%m-%d"),
                    "Time GMT": timestamp.strftime("%H:%M"),
                    "Sample Measurement": 10.0 if site_num == "0001" else 20.0,
                    "Units of Measure": "Micrograms/cubic meter (LC)",
                    "Sample Duration": "1 HOUR",
                }
            )
    test_cfg = replace(
        cfg, data=replace(cfg.data, start_year=2024, end_year=2024, study_end_date=None)
    )
    hourly, metadata = prepare_epa_site_pm25(pd.DataFrame(rows), "06-037-0002", test_cfg)
    assert metadata.loc[0, "site_id"] == "06-037-0002"
    assert metadata.loc[0, "aggregation_method"] == "single_monitor_sensitivity"
    assert hourly["site_id"].eq("06-037-0002").all()
    assert hourly["pm25_observed"].dropna().eq(20.0).all()


def test_noaa_precipitation_preserves_missingness(cfg):
    raw = pd.DataFrame(
        {
            "DATE": ["2024-01-01T00:00:00", "2024-01-01T02:00:00"],
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
    assert bool(weather.loc[1, "precipitation_missing"])
    assert weather.loc[1, "precipitation_mm"] == 0.0
    assert not bool(weather.loc[2, "precipitation_missing"])
    assert weather.loc[2, "precipitation_mm"] == 1.0
    assert weather["precipitation_reported"].dtype == bool


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
