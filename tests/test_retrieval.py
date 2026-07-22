from __future__ import annotations

import pandas as pd
import pytest

from event_timeraf.retrieval import HistoricalRetriever, assert_retrieval_causality, build_knowledge_base
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


def test_knowledge_base_windows_do_not_overlap(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    metadata = build_knowledge_base(dataset, cfg).metadata
    candidate_end = pd.to_datetime(metadata["target_end"], utc=True).to_numpy()
    next_input_start = pd.to_datetime(metadata["input_start"], utc=True).to_numpy()[1:]
    assert (candidate_end[:-1] < next_input_start).all()


def test_no_event_retrieval_removes_event_score(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    knowledge_base = build_knowledge_base(dataset, cfg)
    result = HistoricalRetriever(knowledge_base, cfg).retrieve(
        dataset.subset("test"), method="hybrid_no_event"
    )
    assert result.valid_mask.all()
    assert result.evidence["event_score"].eq(0).all()
    assert all(not name.startswith("extra_") for name in result.feature_names("no_event"))


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
