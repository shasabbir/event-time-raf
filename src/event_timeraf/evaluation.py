from __future__ import annotations

from collections.abc import Callable
from math import erfc, sqrt

import numpy as np
import pandas as pd


EVENT_FLAG_COLUMNS = (
    "recent_event_flag",
    "active_event_flag",
    "target_event_flag",
)


def build_event_period_flags(
    metadata: pd.DataFrame,
    events: pd.DataFrame,
    recent_hours: int = 72,
) -> pd.DataFrame:
    """Build causal context flags and a post-hoc target-overlap label."""
    output = pd.DataFrame(False, index=np.arange(len(metadata)), columns=EVENT_FLAG_COLUMNS)
    if events.empty:
        return output

    event_start = pd.to_datetime(events["event_time"], errors="coerce", utc=True)
    event_end = pd.to_datetime(events.get("event_end", event_start), errors="coerce", utc=True).fillna(event_start)
    published = pd.to_datetime(events["published_at"], errors="coerce", utc=True)
    valid = event_start.notna() & event_end.notna() & published.notna()
    event_start = event_start[valid].to_numpy()
    event_end = event_end[valid].to_numpy()
    published = published[valid].to_numpy()
    if len(event_start) == 0:
        return output

    origins = pd.to_datetime(metadata["origin_time"], utc=True)
    target_starts = pd.to_datetime(metadata["target_start"], utc=True)
    target_ends = pd.to_datetime(metadata["target_end"], utc=True)
    for index, (origin, target_start, target_end) in enumerate(zip(origins, target_starts, target_ends)):
        recent_start = origin - pd.Timedelta(hours=recent_hours)
        output.loc[index, "recent_event_flag"] = bool(((published > recent_start) & (published <= origin)).any())
        output.loc[index, "active_event_flag"] = bool(
            ((event_start <= origin) & (event_end >= origin) & (published <= origin)).any()
        )
        # This is a post-hoc evaluation label. It must never enter model inputs.
        output.loc[index, "target_event_flag"] = bool(
            ((event_start <= target_end) & (event_end >= target_start)).any()
        )
    return output


def _smape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-6) -> float:
    denominator = np.abs(actual) + np.abs(predicted) + epsilon
    return float(200 * np.mean(np.abs(actual - predicted) / denominator))


def _mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-3) -> float:
    return float(100 * np.mean(np.abs(actual - predicted) / np.maximum(np.abs(actual), epsilon)))


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_flat = np.asarray(actual, dtype=float).ravel()
    predicted_flat = np.asarray(predicted, dtype=float).ravel()
    valid = np.isfinite(actual_flat) & np.isfinite(predicted_flat)
    if not valid.any():
        return {name: np.nan for name in ("mse", "mae", "rmse", "mape", "smape", "r2")}
    actual_flat = actual_flat[valid]
    predicted_flat = predicted_flat[valid]
    residual = actual_flat - predicted_flat
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((actual_flat - actual_flat.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else np.nan
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(mse)),
        "mape": _mape(actual_flat, predicted_flat),
        "smape": _smape(actual_flat, predicted_flat),
        "r2": r2,
    }


def metrics_table(
    actual: np.ndarray,
    predicted: np.ndarray,
    model: str,
    subset: str = "all",
    run_id: str = "unassigned",
    event_availability_mode: str = "not_applicable",
) -> pd.DataFrame:
    valid_points = np.isfinite(actual) & np.isfinite(predicted)
    metadata = {
        "run_id": run_id,
        "model": model,
        "subset": subset,
        "event_availability_mode": event_availability_mode,
        "n_origins": int(len(actual)),
        "n_points": int(valid_points.sum()),
    }
    rows = [{**metadata, "horizon": "overall", **metric_values(actual, predicted)}]
    for horizon in range(actual.shape[1]):
        rows.append(
            {
                **metadata,
                "horizon": horizon + 1,
                **metric_values(actual[:, horizon], predicted[:, horizon]),
            }
        )
    horizon_frame = pd.DataFrame(rows[1:])
    macro = {metric: float(horizon_frame[metric].mean()) for metric in ("mse", "mae", "rmse", "mape", "smape", "r2")}
    rows.append({**metadata, "horizon": "macro", **macro})
    return pd.DataFrame(rows)


def predictions_long(
    actual: np.ndarray,
    predicted: np.ndarray,
    metadata: pd.DataFrame,
    model: str,
    seed: int,
    run_id: str,
    drift_flag: np.ndarray | None = None,
    drift_score: np.ndarray | None = None,
    event_flags: pd.DataFrame | None = None,
    event_availability_mode: str = "not_applicable",
) -> pd.DataFrame:
    horizon = actual.shape[1]
    origins = pd.to_datetime(metadata["origin_time"], utc=True).repeat(horizon).reset_index(drop=True)
    steps = np.tile(np.arange(1, horizon + 1), len(metadata))
    target_time = origins + pd.to_timedelta(steps, unit="h")
    if event_flags is None:
        event_flags = pd.DataFrame(False, index=np.arange(len(metadata)), columns=EVENT_FLAG_COLUMNS)
    missing_flags = set(EVENT_FLAG_COLUMNS) - set(event_flags.columns)
    if missing_flags:
        raise ValueError(f"Event flags are missing columns: {sorted(missing_flags)}")
    if len(event_flags) != len(metadata):
        raise ValueError("Event flags and metadata must have the same number of rows")
    result = pd.DataFrame(
        {
            "run_id": run_id,
            "seed": int(seed),
            "model": model,
            "window_id": metadata["window_id"].repeat(horizon).to_numpy(),
            "origin_time": origins,
            "horizon": steps,
            "target_time": target_time,
            "actual": actual.reshape(-1),
            "prediction": predicted.reshape(-1),
            "split": metadata["split"].repeat(horizon).to_numpy(),
            "drift_flag": np.repeat(drift_flag, horizon) if drift_flag is not None else False,
            "drift_score": np.repeat(drift_score, horizon) if drift_score is not None else np.nan,
            "event_availability_mode": event_availability_mode,
        }
    )
    for column in EVENT_FLAG_COLUMNS:
        result[column] = np.repeat(event_flags[column].to_numpy(dtype=bool), horizon)
    # Backward-compatible name used by result tables: event means target overlap.
    result["event_flag"] = result["target_event_flag"]
    return result


def paired_block_bootstrap_difference(
    actual: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if not (actual.shape == prediction_a.shape == prediction_b.shape):
        raise ValueError("Actual and prediction arrays must have identical shapes")
    rng = np.random.default_rng(seed)
    n = len(actual)
    if n == 0:
        raise ValueError("Actual values must contain at least one origin")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    block_length = min(int(block_length), n)
    differences = np.empty(resamples, dtype=float)
    blocks_needed = int(np.ceil(n / block_length))
    for sample in range(resamples):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]
        differences[sample] = metric(actual[indices], prediction_a[indices]) - metric(actual[indices], prediction_b[indices])
    return {
        "difference": float(metric(actual, prediction_a) - metric(actual, prediction_b)),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


def diebold_mariano_test(
    actual: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    loss: str = "mse",
    hac_lag: int = 167,
) -> dict[str, float | int | str]:
    """Paired forecast comparison using a Newey-West/HAC variance estimate.

    Loss is first averaged across horizons for each origin. A positive mean
    difference means prediction A has greater loss than prediction B.
    """
    if not (actual.shape == prediction_a.shape == prediction_b.shape):
        raise ValueError("Actual and prediction arrays must have identical shapes")
    if actual.ndim != 2:
        raise ValueError("Forecast arrays must have shape (origins, horizons)")
    if loss not in {"mse", "mae"}:
        raise ValueError("loss must be mse or mae")
    valid = (
        np.isfinite(actual).all(axis=1)
        & np.isfinite(prediction_a).all(axis=1)
        & np.isfinite(prediction_b).all(axis=1)
    )
    if not valid.any():
        raise ValueError("No complete forecast origins are available")
    error_a = actual[valid] - prediction_a[valid]
    error_b = actual[valid] - prediction_b[valid]
    if loss == "mse":
        difference = np.mean(error_a**2, axis=1) - np.mean(error_b**2, axis=1)
    else:
        difference = np.mean(np.abs(error_a), axis=1) - np.mean(np.abs(error_b), axis=1)
    n = len(difference)
    lag = min(max(int(hac_lag), 0), n - 1)
    centered = difference - difference.mean()
    long_run_variance = float(np.dot(centered, centered) / n)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        long_run_variance += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    variance_of_mean = max(long_run_variance / n, 0.0)
    statistic = float(difference.mean() / np.sqrt(variance_of_mean)) if variance_of_mean > 0 else np.nan
    p_value = float(erfc(abs(statistic) / sqrt(2.0))) if np.isfinite(statistic) else np.nan
    return {
        "loss": loss,
        "n_origins": int(n),
        "hac_lag": int(lag),
        "mean_difference": float(difference.mean()),
        "dm_statistic": statistic,
        "p_value": p_value,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values keyed like the input mapping."""
    valid = [(name, float(value)) for name, value in p_values.items() if np.isfinite(value)]
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted: dict[str, float] = {name: np.nan for name in p_values}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (total - rank) * value)
        adjusted[name] = min(running, 1.0)
    return adjusted


def subset_target_summary(
    actual: np.ndarray,
    subset_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for name, mask in subset_masks.items():
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(actual):
            raise ValueError(f"Subset mask {name} does not match forecast origins")
        values = np.asarray(actual[mask], dtype=float)
        valid = values[np.isfinite(values)]
        rows.append(
            {
                "subset": name,
                "n_origins": int(mask.sum()),
                "n_points": int(len(valid)),
                "target_mean": float(np.mean(valid)) if len(valid) else np.nan,
                "target_std": float(np.std(valid)) if len(valid) else np.nan,
                "target_variance": float(np.var(valid)) if len(valid) else np.nan,
                "target_min": float(np.min(valid)) if len(valid) else np.nan,
                "target_max": float(np.max(valid)) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def interval_metrics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = 0.8,
) -> dict[str, float]:
    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("Actual and interval arrays must have identical shapes")
    valid = np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return {"coverage": np.nan, "mean_width": np.nan, "interval_score": np.nan}
    y = actual[valid]
    lo = lower[valid]
    hi = upper[valid]
    alpha = 1.0 - float(nominal_coverage)
    width = hi - lo
    score = width + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width": float(np.mean(width)),
        "interval_score": float(np.mean(score)),
    }
