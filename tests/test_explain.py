from __future__ import annotations

import json

import numpy as np
import pandas as pd

from event_timeraf.drift import DriftResult
from event_timeraf.explain import generate_explanations, grouped_feature_perturbation
from event_timeraf.retrieval import RetrievalResult
from event_timeraf.windows import WindowDataset


def test_explanations_save_semantic_effects_and_uncertainty():
    metadata = pd.DataFrame(
        {
            "window_id": ["q1"],
            "origin_time": [pd.Timestamp("2024-01-01", tz="UTC")],
        }
    )
    dataset = WindowDataset(
        x=np.ones((1, 168), dtype=np.float32),
        y=np.ones((1, 24), dtype=np.float32),
        features=np.empty((1, 0), dtype=np.float32),
        future_calendar=np.empty((1, 24, 0), dtype=np.float32),
        metadata=metadata,
        feature_names=(),
        calendar_names=(),
    )
    retrieval = RetrievalResult(
        prediction=np.full((1, 24), 2.0, dtype=np.float32),
        weighted_prediction=np.full((1, 24), 2.0, dtype=np.float32),
        spread=np.full((1, 24), 0.5, dtype=np.float32),
        mean_similarity=np.array([0.8], dtype=np.float32),
        max_similarity=np.array([0.9], dtype=np.float32),
        candidate_count=np.array([8]),
        eligible_candidate_count=np.array([100]),
        selected_event_fraction=np.array([0.5], dtype=np.float32),
        query_event_context=np.array([True]),
        event_conditioning_applied=np.array([True]),
        evidence=pd.DataFrame(
            {"query_window_id": ["q1"], "rank": [1], "candidate_window_id": ["c1"]}
        ),
    )
    drift = DriftResult(
        components=np.array([[0.7, 0.1]], dtype=np.float32),
        component_names=("mean_shift", "weather_shift"),
        score=np.array([0.4], dtype=np.float32),
        flag=np.array([True]),
        threshold=0.3,
    )
    result = generate_explanations(
        dataset,
        np.full((1, 24), 2.0, dtype=np.float32),
        retrieval,
        drift,
        pd.DataFrame(),
        feature_contributions=np.array([[0.2, -0.5]], dtype=np.float32),
        contribution_names=["pm25_current", "hybrid_retrieval_mean_similarity"],
        validation_residual_rmse=np.full(24, 1.0),
    )
    effects = json.loads(result.loc[0, "top_feature_effects"])
    assert effects[0]["feature"] == "hybrid_retrieval_mean_similarity"
    assert result.loc[0, "uncertainty_proxy"] > 1.0
    assert "uncertainty proxy" in result.loc[0, "explanation"].lower()
    assert "validation_residual_rmse" in result


def test_grouped_feature_perturbation_uses_validation_predictions():
    matrix = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    actual = matrix[:, :1]
    result = grouped_feature_perturbation(
        actual,
        matrix,
        matrix,
        ["signal_value", "noise_value"],
        {"signal": ("signal_",), "noise": ("noise_",)},
        lambda values: values[:, :1],
        seed=42,
        model="toy",
    )
    signal = result.loc[result["feature_group"] == "signal", "mse_increase"]
    noise = result.loc[result["feature_group"] == "noise", "mse_increase"]
    assert (signal > 0).all()
    assert (noise == 0).all()
