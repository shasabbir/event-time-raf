from __future__ import annotations

import numpy as np

from event_timeraf.drift import DriftDetector, drift_evidence_frame
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
