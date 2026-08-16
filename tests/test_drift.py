from __future__ import annotations

import numpy as np

from event_timeraf.drift import DriftDetector, KSTwoSampleDriftDetector, drift_evidence_frame
from event_timeraf.windows import build_window_dataset


def test_drift_evidence_has_semantic_components(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    train = dataset.subset("train")
    validation = dataset.subset("validation")
    detector = DriftDetector(cfg, include_event_component=False).fit_reference(
        train, np.full(len(train.x), 0.8)
    )
    detector.calibrate(validation, np.full(len(validation.x), 0.8))
    result = detector.transform(validation, np.full(len(validation.x), 0.8))
    evidence = drift_evidence_frame(validation, result, "run-1")
    assert "drift_event_burst" not in evidence
    assert "drift_retrieval_similarity_drop" in evidence
    assert evidence["run_id"].eq("run-1").all()


def test_two_sided_drift_scores_both_sides_of_reference(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    train = dataset.subset("train")
    validation = dataset.subset("validation")
    detector = DriftDetector(cfg, include_event_component=False).fit_reference(
        train, np.full(len(train.x), 0.8)
    )
    validation_similarity = np.linspace(0.7, 0.9, len(validation.x))
    detector.calibrate(validation, validation_similarity)
    components = detector.transform(validation, validation_similarity).components
    assert (components > 0).any(axis=0).all()


def test_ks_drift_benchmark_calibrates_on_validation(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    detector = KSTwoSampleDriftDetector(cfg).fit_reference(dataset.subset("train"))
    detector.calibrate(dataset.subset("validation"))
    result = detector.transform(dataset.subset("test"))
    assert result.component_names == ("ks_statistic",)
    assert np.isfinite(result.score).all()
    assert result.threshold > 0
