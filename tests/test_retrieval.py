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
    assert (pd.to_datetime(result.evidence["candidate_target_end"], utc=True) < pd.to_datetime(result.evidence["query_origin"], utc=True)).all()


def test_retrieval_causality_guard_rejects_future_candidate():
    evidence = pd.DataFrame(
        {
            "query_origin": [pd.Timestamp("2024-01-02", tz="UTC")],
            "candidate_target_end": [pd.Timestamp("2024-01-03", tz="UTC")],
        }
    )
    with pytest.raises(AssertionError):
        assert_retrieval_causality(evidence)

