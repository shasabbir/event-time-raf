from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .windows import WindowDataset


def normalize_windows(values: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = values.mean(axis=1)
    stds = values.std(axis=1)
    safe_stds = np.maximum(stds, epsilon)
    normalized = (values - means[:, None]) / safe_stds[:, None]
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized / np.maximum(norms, epsilon)
    return normalized.astype(np.float32), means.astype(np.float32), safe_stds.astype(np.float32)


def _standardized_unit(
    candidates: np.ndarray,
    queries: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(candidates, axis=0)
    scale = np.nanstd(candidates, axis=0)
    scale = np.maximum(scale, epsilon)
    candidate_scaled = np.nan_to_num((candidates - center) / scale)
    query_scaled = np.nan_to_num((queries - center) / scale)
    candidate_norm = np.linalg.norm(candidate_scaled, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_scaled, axis=1, keepdims=True)
    candidate_unit = candidate_scaled / np.maximum(candidate_norm, epsilon)
    query_unit = query_scaled / np.maximum(query_norm, epsilon)
    return candidate_unit.astype(np.float32), query_unit.astype(np.float32)


@dataclass
class KnowledgeBase:
    x: np.ndarray
    y: np.ndarray
    vectors: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    features: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]

    def save(self, arrays_path: Path, metadata_path: Path) -> None:
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_path,
            x=self.x,
            y=self.y,
            vectors=self.vectors,
            input_mean=self.input_mean,
            input_std=self.input_std,
            features=self.features,
        )
        self.metadata.to_parquet(metadata_path, index=False)
        arrays_path.with_suffix(".json").write_text(
            json.dumps({"feature_names": self.feature_names, "size": len(self.metadata)}, indent=2),
            encoding="utf-8",
        )


@dataclass
class RetrievalResult:
    prediction: np.ndarray
    weighted_prediction: np.ndarray
    spread: np.ndarray
    mean_similarity: np.ndarray
    max_similarity: np.ndarray
    candidate_count: np.ndarray
    evidence: pd.DataFrame

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.prediction).all(axis=1)

    def as_features(self) -> np.ndarray:
        return np.column_stack(
            [
                self.prediction,
                self.spread,
                self.mean_similarity,
                self.max_similarity,
                self.candidate_count,
            ]
        ).astype(np.float32)

    def feature_names(self, prefix: str = "retrieval") -> list[str]:
        horizon = self.prediction.shape[1]
        return (
            [f"{prefix}_trajectory_h{step:02d}" for step in range(1, horizon + 1)]
            + [f"{prefix}_spread_h{step:02d}" for step in range(1, horizon + 1)]
            + [
                f"{prefix}_mean_similarity",
                f"{prefix}_max_similarity",
                f"{prefix}_candidate_count",
            ]
        )


def build_knowledge_base(
    dataset: WindowDataset,
    cfg: ProjectConfig,
    stride_hours: int | None = None,
) -> KnowledgeBase:
    train_indices = np.flatnonzero(dataset.metadata["split"].to_numpy() == "train")
    if train_indices.size == 0:
        raise ValueError("Training split is empty")
    stride = int(stride_hours or cfg.retrieval.kb_stride_hours)
    if stride <= 0:
        raise ValueError("Knowledge-base stride must be positive")
    origin_rows = dataset.metadata.loc[train_indices, "origin_row"].to_numpy(dtype=int)
    keep = origin_rows % stride == 0
    indices = train_indices[keep]
    if indices.size < cfg.retrieval.k:
        raise ValueError("Knowledge base has fewer candidates than configured k")
    metadata = dataset.metadata.loc[indices].reset_index(drop=True)
    metadata["kb_stride_hours"] = stride
    vectors, means, stds = normalize_windows(dataset.x[indices], cfg.retrieval.epsilon)
    return KnowledgeBase(
        x=dataset.x[indices],
        y=dataset.y[indices],
        vectors=vectors,
        input_mean=means,
        input_std=stds,
        features=dataset.features[indices],
        metadata=metadata,
        feature_names=dataset.feature_names,
    )


class HistoricalRetriever:
    def __init__(self, kb: KnowledgeBase, cfg: ProjectConfig):
        self.kb = kb
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.groups = {
            "weather": [i for i, name in enumerate(kb.feature_names) if name.startswith("weather_")],
            "calendar": [i for i, name in enumerate(kb.feature_names) if name.startswith("cal_")],
            "event": [i for i, name in enumerate(kb.feature_names) if name.startswith("event_")],
        }
        self._context_reference: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for group, columns in self.groups.items():
            if not columns:
                continue
            candidate = self.kb.features[:, columns]
            center = np.nanmean(candidate, axis=0)
            scale = np.maximum(np.nanstd(candidate, axis=0), self.cfg.retrieval.epsilon)
            candidate_scaled = np.nan_to_num((candidate - center) / scale)
            candidate_norm = np.linalg.norm(candidate_scaled, axis=1, keepdims=True)
            candidate_unit = candidate_scaled / np.maximum(candidate_norm, self.cfg.retrieval.epsilon)
            self._context_reference[group] = (
                center.astype(np.float32),
                scale.astype(np.float32),
                candidate_unit.astype(np.float32),
            )
        event_columns = self.groups["event"]
        self._candidate_has_event = (
            np.linalg.norm(self.kb.features[:, event_columns], axis=1) > self.cfg.retrieval.epsilon
            if event_columns
            else np.zeros(len(self.kb.metadata), dtype=bool)
        )

    def _context_similarity(self, query_features: np.ndarray, group: str) -> np.ndarray:
        columns = self.groups[group]
        if not columns:
            return np.zeros((len(query_features), len(self.kb.metadata)), dtype=np.float32)
        candidate = self.kb.features[:, columns]
        query = query_features[:, columns]
        center, scale, candidate_unit = self._context_reference[group]
        query_scaled = np.nan_to_num((query - center) / scale)
        query_norm = np.linalg.norm(query_scaled, axis=1, keepdims=True)
        query_unit = query_scaled / np.maximum(query_norm, self.cfg.retrieval.epsilon)
        similarity = query_unit @ candidate_unit.T
        if group == "event":
            candidate_zero = np.linalg.norm(candidate, axis=1) <= self.cfg.retrieval.epsilon
            query_zero = np.linalg.norm(query, axis=1) <= self.cfg.retrieval.epsilon
            similarity = np.clip(similarity, 0, 1)
            similarity[np.ix_(query_zero, candidate_zero)] = 1.0
            similarity[np.ix_(query_zero, ~candidate_zero)] = 0.0
            similarity[np.ix_(~query_zero, candidate_zero)] = 0.0
            return similarity.astype(np.float32)
        return np.clip((similarity + 1.0) / 2.0, 0, 1).astype(np.float32)

    def retrieve(
        self,
        queries: WindowDataset,
        method: str = "cosine",
        k: int | None = None,
        event_weight: float | None = None,
    ) -> RetrievalResult:
        allowed_methods = {
            "random", "cosine", "calendar", "hybrid", "hybrid_no_event", "event_stratified"
        }
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {sorted(allowed_methods)}")
        k = k or self.cfg.retrieval.k
        if k <= 0:
            raise ValueError("k must be positive")
        weights = event_weighted_channel_weights(self.cfg.retrieval.weights, event_weight)
        rng = np.random.default_rng(self.cfg.seed)
        query_vectors, query_means, query_stds = normalize_windows(queries.x, self.cfg.retrieval.epsilon)
        n_queries = len(queries.metadata)
        horizon = queries.y.shape[1]
        prediction = np.full((n_queries, horizon), np.nan, dtype=np.float32)
        weighted_prediction = np.full_like(prediction, np.nan)
        spread = np.full_like(prediction, np.nan)
        mean_similarity = np.full(n_queries, np.nan, dtype=np.float32)
        max_similarity = np.full(n_queries, np.nan, dtype=np.float32)
        candidate_count = np.zeros(n_queries, dtype=np.int32)
        evidence_rows: list[dict] = []

        kb_target_end = pd.to_datetime(self.kb.metadata["target_end"], utc=True).to_numpy()
        query_input_starts = pd.to_datetime(queries.metadata["input_start"], utc=True).to_numpy()
        event_columns = self.groups["event"]

        for block_start in range(0, n_queries, self.cfg.retrieval.block_size):
            block_end = min(block_start + self.cfg.retrieval.block_size, n_queries)
            block_vectors = query_vectors[block_start:block_end]
            raw_ts = block_vectors @ self.kb.vectors.T
            ts_score = np.clip((raw_ts + 1.0) / 2.0, 0, 1)
            if method in {"calendar", "hybrid", "hybrid_no_event", "event_stratified"}:
                block_features = queries.features[block_start:block_end]
                weather = self._context_similarity(block_features, "weather")
                calendar = self._context_similarity(block_features, "calendar")
                if method == "calendar":
                    event = np.zeros_like(ts_score)
                    scores = calendar
                elif method in {"hybrid", "event_stratified"}:
                    event = self._context_similarity(block_features, "event")
                    scores = (
                        weights["time_series"] * ts_score
                        + weights["weather"] * weather
                        + weights["calendar"] * calendar
                        + weights["event"] * event
                    )
                else:
                    event = np.zeros_like(ts_score)
                    denominator = weights["time_series"] + weights["weather"] + weights["calendar"]
                    scores = (
                        weights["time_series"] * ts_score
                        + weights["weather"] * weather
                        + weights["calendar"] * calendar
                    ) / denominator
            else:
                weather = calendar = event = np.zeros_like(ts_score)
                scores = ts_score.copy()

            for local_index, query_index in enumerate(range(block_start, block_end)):
                # The complete candidate input and target must precede the
                # query lookback. This prevents near-duplicate retrieval from
                # reusing values already present in the query input.
                eligible = np.flatnonzero(kb_target_end < query_input_starts[query_index])
                if eligible.size == 0:
                    continue
                stratified = False
                stratified_fallback = False
                query_has_event = bool(
                    event_columns
                    and np.linalg.norm(queries.features[query_index, event_columns])
                    > self.cfg.retrieval.epsilon
                )
                if method == "event_stratified" and query_has_event:
                    event_eligible = eligible[self._candidate_has_event[eligible]]
                    if event_eligible.size >= k:
                        eligible = event_eligible
                        stratified = True
                    elif self.cfg.retrieval.event_stratified_fallback == "skip":
                        eligible = event_eligible
                        stratified = True
                    else:
                        stratified_fallback = True
                    if eligible.size == 0:
                        continue
                count = min(k, eligible.size)
                if method == "random":
                    selected = rng.choice(eligible, size=count, replace=False)
                    selected_scores = np.zeros(count, dtype=float)
                else:
                    eligible_scores = scores[local_index, eligible]
                    if count == eligible.size:
                        order = np.argsort(eligible_scores)[::-1]
                    else:
                        partition = np.argpartition(eligible_scores, -count)[-count:]
                        order = partition[np.argsort(eligible_scores[partition])[::-1]]
                    selected = eligible[order]
                    selected_scores = scores[local_index, selected]

                normalized_future = (
                    self.kb.y[selected] - self.kb.input_mean[selected, None]
                ) / self.kb.input_std[selected, None]
                aligned = query_means[query_index] + query_stds[query_index] * normalized_future
                prediction[query_index] = aligned.mean(axis=0)
                spread[query_index] = aligned.std(axis=0)
                positive = np.maximum(selected_scores, 0)
                if positive.sum() <= self.cfg.retrieval.epsilon:
                    candidate_weights = np.full(count, 1.0 / count)
                else:
                    candidate_weights = positive / positive.sum()
                weighted_prediction[query_index] = np.average(aligned, axis=0, weights=candidate_weights)
                similarities = ts_score[local_index, selected]
                mean_similarity[query_index] = float(similarities.mean())
                max_similarity[query_index] = float(similarities.max())
                candidate_count[query_index] = count

                query_row = queries.metadata.iloc[query_index]
                for rank, (candidate_index, total_score) in enumerate(zip(selected, selected_scores), start=1):
                    candidate_row = self.kb.metadata.iloc[candidate_index]
                    evidence_rows.append(
                        {
                            "query_window_id": query_row["window_id"],
                            "query_origin": query_row["origin_time"],
                            "query_input_start": query_row["input_start"],
                            "method": method,
                            "rank": rank,
                            "candidate_window_id": candidate_row["window_id"],
                            "candidate_input_start": candidate_row["input_start"],
                            "candidate_origin": candidate_row["origin_time"],
                            "candidate_target_end": candidate_row["target_end"],
                            "total_score": float(total_score),
                            "time_series_score": float(ts_score[local_index, candidate_index]),
                            "weather_score": float(weather[local_index, candidate_index]),
                            "calendar_score": float(calendar[local_index, candidate_index]),
                            "event_score": float(event[local_index, candidate_index]),
                            "event_weight": float(weights["event"]),
                            "query_has_event_context": query_has_event,
                            "candidate_has_event_context": bool(self._candidate_has_event[candidate_index]),
                            "event_stratified": stratified,
                            "event_stratified_fallback": stratified_fallback,
                            "eligible_candidate_count": int(eligible.size),
                            "kb_stride_hours": int(candidate_row["kb_stride_hours"]),
                            "aligned_future": aligned[rank - 1].astype(float).tolist(),
                        }
                    )

        evidence = pd.DataFrame(evidence_rows)
        assert_retrieval_causality(evidence)
        return RetrievalResult(
            prediction=prediction,
            weighted_prediction=weighted_prediction,
            spread=spread,
            mean_similarity=mean_similarity,
            max_similarity=max_similarity,
            candidate_count=candidate_count,
            evidence=evidence,
        )


def assert_retrieval_causality(evidence: pd.DataFrame) -> None:
    if evidence.empty:
        return
    query = pd.to_datetime(evidence["query_origin"], utc=True)
    query_input_start = pd.to_datetime(evidence["query_input_start"], utc=True)
    candidate_end = pd.to_datetime(evidence["candidate_target_end"], utc=True)
    valid = (candidate_end < query_input_start) & (candidate_end < query)
    if not valid.all():
        invalid = evidence.loc[~valid].head()
        raise AssertionError(f"Retrieval leakage detected:\n{invalid}")


def event_weighted_channel_weights(
    base_weights: dict[str, float],
    event_weight: float | None,
) -> dict[str, float]:
    """Set the event weight and proportionally renormalize the other channels."""
    if event_weight is None:
        return {name: float(value) for name, value in base_weights.items()}
    event_weight = float(event_weight)
    if not 0 <= event_weight <= 1:
        raise ValueError("event_weight must be between 0 and 1")
    non_event_names = ("time_series", "weather", "calendar")
    denominator = sum(float(base_weights[name]) for name in non_event_names)
    if denominator <= 0 and event_weight < 1:
        raise ValueError("Non-event retrieval weights must have positive mass")
    scale = (1.0 - event_weight) / denominator if denominator > 0 else 0.0
    return {
        **{name: float(base_weights[name]) * scale for name in non_event_names},
        "event": event_weight,
    }


def topk_change_summary(
    reference: RetrievalResult,
    comparison: RetrievalResult,
) -> dict[str, float | int]:
    """Measure how often two retrieval configurations return different top-k IDs."""
    reference_groups = reference.evidence.groupby("query_window_id")["candidate_window_id"].apply(tuple)
    comparison_groups = comparison.evidence.groupby("query_window_id")["candidate_window_id"].apply(tuple)
    common = reference_groups.index.intersection(comparison_groups.index)
    if len(common) == 0:
        return {"n_queries": 0, "changed_queries": 0, "changed_fraction": np.nan}
    changed = sum(reference_groups.loc[key] != comparison_groups.loc[key] for key in common)
    return {
        "n_queries": int(len(common)),
        "changed_queries": int(changed),
        "changed_fraction": float(changed / len(common)),
    }
