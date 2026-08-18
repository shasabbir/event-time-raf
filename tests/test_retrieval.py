from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from event_timeraf.retrieval import (
    HistoricalRetriever,
    assert_retrieval_causality,
    build_knowledge_base,
    residual_correction_from_retrieval,
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


def test_early_training_queries_are_excluded_but_evaluation_is_complete(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    retriever = HistoricalRetriever(build_knowledge_base(dataset, cfg), cfg)
    train_result = retriever.retrieve(dataset.subset("train"), method="event_conditioned")
    validation_result = retriever.retrieve(
        dataset.subset("validation"), method="event_conditioned"
    )
    test_result = retriever.retrieve(dataset.subset("test"), method="event_conditioned")
    assert (~train_result.valid_mask).any()
    assert train_result.valid_mask.any()
    assert validation_result.valid_mask.all()
    assert test_result.valid_mask.all()


def test_dense_knowledge_base_allows_candidate_overlap(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    metadata = build_knowledge_base(dataset, cfg).metadata
    candidate_end = pd.to_datetime(metadata["target_end"], utc=True).to_numpy()
    next_input_start = pd.to_datetime(metadata["input_start"], utc=True).to_numpy()[1:]
    origins = pd.to_datetime(metadata["origin_time"], utc=True)
    assert (origins.diff().dropna() == pd.Timedelta(hours=cfg.retrieval.kb_stride_hours)).all()
    assert (candidate_end[:-1] >= next_input_start).any()


def test_no_event_retrieval_removes_event_score(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    knowledge_base = build_knowledge_base(dataset, cfg)
    result = HistoricalRetriever(knowledge_base, cfg).retrieve(
        dataset.subset("test"), method="hybrid_no_event"
    )
    assert result.valid_mask.all()
    assert result.evidence["event_score"].eq(0).all()
    assert all(not name.startswith("extra_") for name in result.feature_names("no_event"))


def test_event_conditioning_selects_event_candidates(modeling_frame, cfg):
    modeling_frame = modeling_frame.copy()
    modeling_frame.loc[modeling_frame.index % 96 < 24, "event_count_72h"] = 1.0
    dataset = build_window_dataset(modeling_frame, cfg)
    knowledge_base = build_knowledge_base(dataset, cfg)
    queries = dataset.subset("test")
    result = HistoricalRetriever(knowledge_base, cfg).retrieve(
        queries, method="event_conditioned"
    )
    conditioned = result.evidence["event_conditioning_applied"]
    assert conditioned.any()
    assert result.evidence.loc[conditioned, "candidate_has_event_context"].all()
    assert result.evidence["event_score"].between(0, 1).all()


def test_random_retrieval_is_independent_of_call_order(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    retriever = HistoricalRetriever(build_knowledge_base(dataset, cfg), cfg)
    queries = dataset.subset("test")
    first = retriever.retrieve(queries, method="random")
    retriever.retrieve(queries, method="cosine")
    second = retriever.retrieve(queries, method="random")
    pd.testing.assert_frame_equal(
        first.evidence[["query_window_id", "rank", "candidate_window_id"]].reset_index(drop=True),
        second.evidence[["query_window_id", "rank", "candidate_window_id"]].reset_index(drop=True),
    )


def test_residual_retrieval_uses_only_oof_candidates(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    knowledge_base = build_knowledge_base(dataset, cfg)
    candidate_mask = np.zeros(len(knowledge_base.metadata), dtype=bool)
    candidate_mask[len(candidate_mask) // 2 :] = True
    candidate_residuals = np.full_like(knowledge_base.y, np.nan)
    candidate_residuals[candidate_mask] = 1.0
    queries = dataset.subset("validation")
    result = HistoricalRetriever(knowledge_base, cfg).retrieve(
        queries, method="event_conditioned", candidate_mask=candidate_mask
    )
    assert result.valid_mask.all()
    assert candidate_mask[result.selected_indices[result.selected_indices >= 0]].all()
    correction = residual_correction_from_retrieval(
        knowledge_base, queries, result, candidate_residuals, cfg.retrieval.epsilon
    )
    assert correction.valid_mask.all()
    assert (correction.candidate_count > 0).all()


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
