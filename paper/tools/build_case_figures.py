from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "figures"


def publication_archive() -> Path:
    candidates = sorted(
        ROOT.glob("event_timeraf_publication_candidate_*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No publication-candidate archive found in the repository root")
    return candidates[0]


def archive_reader(path: Path):
    bundle = zipfile.ZipFile(path)
    manifest_member = next(
        PurePosixPath(name)
        for name in bundle.namelist()
        if PurePosixPath(name).parts[-3:] == ("outputs", "logs", "run_manifest.json")
    )
    prefix = PurePosixPath(*manifest_member.parts[:-3])

    def read(relative: str) -> bytes:
        return bundle.read((prefix / relative).as_posix())

    manifest = json.loads(read("outputs/logs/run_manifest.json"))
    for relative in (
        "data/processed/window_arrays.npz",
        "data/processed/window_metadata.parquet",
        "outputs/predictions/predictions.parquet",
    ):
        payload = read(relative)
        expected = manifest["artifacts"][relative]
        if len(payload) != expected["bytes"]:
            raise ValueError(f"Byte-count mismatch for {relative}")
        if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {relative}")
    return bundle, read


def prediction_matrix(
    predictions: pd.DataFrame, model: str, window_ids: pd.Series
) -> np.ndarray:
    frame = predictions.loc[predictions["model"].eq(model)]
    matrix = frame.pivot(index="window_id", columns="horizon", values="prediction")
    return matrix.reindex(window_ids).to_numpy(dtype=float)


def main() -> None:
    archive = publication_archive()
    bundle, read = archive_reader(archive)
    try:
        arrays = np.load(BytesIO(read("data/processed/window_arrays.npz")))
        metadata = pd.read_parquet(BytesIO(read("data/processed/window_metadata.parquet")))
        predictions = pd.read_parquet(BytesIO(read("outputs/predictions/predictions.parquet")))
    finally:
        bundle.close()

    test_mask = metadata["split"].eq("test").to_numpy()
    test_metadata = metadata.loc[test_mask].reset_index(drop=True)
    test_x = arrays["x"][test_mask]
    test_y = arrays["y"][test_mask]
    window_ids = test_metadata["window_id"]

    model_ids = {
        "LightGBM context": "C05_lightgbm_context",
        "Context ensemble": "C06_context_ensemble",
        "Event-feature XGBoost": "M09_event_timeraf_full",
        "TRACE-RAF": "M13_trace_raf",
    }
    model_predictions = {
        label: prediction_matrix(predictions, model, window_ids)
        for label, model in model_ids.items()
    }

    actual = (
        predictions.loc[predictions["model"].eq("M13_trace_raf")]
        .pivot(index="window_id", columns="horizon", values="actual")
        .reindex(window_ids)
        .to_numpy(dtype=float)
    )
    if not np.allclose(actual, test_y):
        raise ValueError("Archived prediction targets do not match the processed test windows")

    sys.path.insert(0, str(ROOT / "src"))
    from event_timeraf.plots import plot_forecast_case

    trace = model_predictions["TRACE-RAF"]
    case_index = int(
        np.argsort(np.abs(test_y.mean(axis=1) - trace.mean(axis=1)))[len(test_y) // 2]
    )

    event_flags = (
        predictions.loc[predictions["model"].eq("M13_trace_raf")]
        .groupby("window_id")["target_event_flag"]
        .first()
        .reindex(window_ids)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    event_candidates = np.flatnonzero(event_flags)
    event_case_index = (
        int(event_candidates[np.argmax(test_y[event_candidates].mean(axis=1))])
        if len(event_candidates)
        else int(np.argmax(test_y.mean(axis=1)))
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for index, destination in (
        (case_index, FIGURE_DIR / "forecast_case.png"),
        (event_case_index, FIGURE_DIR / "forecast_event_case.png"),
    ):
        figure = plot_forecast_case(
            test_x[index],
            test_y[index],
            {label: values[index] for label, values in model_predictions.items()},
            destination,
        )
        plt.close(figure)

    print(
        {
            "archive": archive.name,
            "ordinary_window": window_ids.iloc[case_index],
            "event_window": window_ids.iloc[event_case_index],
        }
    )


if __name__ == "__main__":
    main()
