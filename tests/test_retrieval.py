from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from event_timeraf.retrieval import (
    HistoricalRetriever,
    assert_retrieval_causality,
    build_knowledge_base,
    event_weighted_channel_weights,
    topk_change_summary,
)
from event_timeraf.windows import build_window_dataset


def test_retrieval_candidates_are_strictly_historical(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    knowledge_base = build_knowledge_base(dataset, cfg)
    test = dataset.subset("test")
    result = HistoricalRetriever(knowledge_base, cfg).retrieve(test, method="hybrid")
    assert result.valid_mask.all()
    assert not result.evidence.empty
    assert_retrieval_causality(result.evidence)
    assert (
        pd.to_datetime(result.evidence["candidate_target_end"], utc=True)
        < pd.to_datetime(result.evidence["query_input_start"], utc=True)
    ).all()


def test_dense_knowledge_base_allows_candidate_overlap_without_query_leakage(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    metadata = build_knowledge_base(dataset, cfg).metadata
    candidate_end = pd.to_datetime(metadata["target_end"], utc=True).to_numpy()
    next_input_start = pd.to_datetime(metadata["input_start"], utc=True).to_numpy()[1:]
    assert (candidate_end[:-1] >= next_input_start).any()

    result = HistoricalRetriever(build_knowledge_base(dataset, cfg), cfg).retrieve(
        dataset.subset("test"), method="hybrid"
    )
    assert_retrieval_causality(result.evidence)


def test_no_event_retrieval_removes_event_score(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    knowledge_base = build_knowledge_base(dataset, cfg)
    result = HistoricalRetriever(knowledge_base, cfg).retrieve(
        dataset.subset("test"), method="hybrid_no_event"
    )
    assert result.valid_mask.all()
    assert result.evidence["event_score"].eq(0).all()
    assert all(not name.startswith("extra_") for name in result.feature_names("no_event"))


def test_random_retrieval_is_repeatable_per_invocation(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    retriever = HistoricalRetriever(build_knowledge_base(dataset, cfg), cfg)
    first = retriever.retrieve(dataset.subset("test"), method="random")
    second = retriever.retrieve(dataset.subset("test"), method="random")
    assert np.array_equal(first.prediction, second.prediction)
    assert first.evidence["candidate_window_id"].tolist() == second.evidence["candidate_window_id"].tolist()


def test_event_weight_override_preserves_normalized_channel_weights(cfg):
    weights = event_weighted_channel_weights(cfg.retrieval.weights, 0.8)
    assert weights["event"] == pytest.approx(0.8)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_event_stratified_retrieval_uses_event_candidates(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    event_columns = [i for i, name in enumerate(dataset.feature_names) if name.startswith("event_")]
    dataset.features[:, event_columns] = 1.0
    dense_cfg = replace(cfg, retrieval=replace(cfg.retrieval, kb_stride_hours=24))
    result = HistoricalRetriever(build_knowledge_base(dataset, dense_cfg), dense_cfg).retrieve(
        dataset.subset("test"), method="event_stratified"
    )
    assert result.valid_mask.all()
    assert result.evidence["event_stratified"].all()
    assert result.evidence["candidate_has_event_context"].all()
    assert result.evidence["event_stratified_fallback"].eq(False).all()


def test_topk_change_summary_detects_identical_retrieval(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    retriever = HistoricalRetriever(build_knowledge_base(dataset, cfg), cfg)
    result = retriever.retrieve(dataset.subset("test"), method="hybrid")
    summary = topk_change_summary(result, result)
    assert summary["n_queries"] == len(dataset.subset("test").metadata)
    assert summary["changed_fraction"] == 0.0


def test_retrieval_causality_guard_rejects_future_candidate():
    evidence = pd.DataFrame(
        {
            "query_origin": [pd.Timestamp("2024-01-02", tz="UTC")],
            "query_input_start": [pd.Timestamp("2023-12-26", tz="UTC")],
            "candidate_target_end": [pd.Timestamp("2024-01-03", tz="UTC")],
        }
    )
    with pytest.raises(AssertionError):
        assert_retrieval_causality(evidence)


def test_retrieval_guard_rejects_candidate_overlapping_query_input():
    evidence = pd.DataFrame(
        {
            "query_origin": [pd.Timestamp("2024-01-08", tz="UTC")],
            "query_input_start": [pd.Timestamp("2024-01-01", tz="UTC")],
            "candidate_target_end": [pd.Timestamp("2024-01-02", tz="UTC")],
        }
    )
    with pytest.raises(AssertionError):
        assert_retrieval_causality(evidence)
