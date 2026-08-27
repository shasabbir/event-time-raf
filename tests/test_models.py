from __future__ import annotations

import numpy as np

from event_timeraf.models import (
    apply_residual_correction,
    blend_base_with_analogue,
    choose_analogue_weight,
    choose_residual_strength,
    ConvexForecastEnsemble,
    DirectRidgeForecaster,
    NeuralWindowForecaster,
    SelectiveResidualGate,
    hour_month_climatology_forecast,
    residual_gate_features,
)
from event_timeraf.windows import build_window_dataset


def test_climatology_and_ridge_controls_are_finite(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    train = dataset.subset("train")
    validation = dataset.subset("validation")
    climatology = hour_month_climatology_forecast(train, validation, cfg.timezone)
    assert climatology.shape == validation.y.shape
    assert np.isfinite(climatology).all()

    origin_features = train.features[:, :4]
    model = DirectRidgeForecaster(cfg, alpha=1.0).fit(
        origin_features,
        train.future_calendar,
        train.y,
        feature_names=list(train.feature_names[:4]),
        calendar_names=train.calendar_names,
    )
    prediction = model.predict(validation.features[:, :4], validation.future_calendar)
    assert prediction.shape == validation.y.shape
    assert np.isfinite(prediction).all()


def test_neural_window_normalization_is_origin_local(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg).subset("validation")
    model = NeuralWindowForecaster(cfg, "dlinear")
    x_normalized, y_normalized = model._normalize(dataset.x, dataset.y)
    assert np.allclose(x_normalized.mean(axis=1), 0, atol=1e-5)
    assert x_normalized.shape == dataset.x.shape
    assert y_normalized.shape == dataset.y.shape


def test_neural_window_forecaster_accepts_required_architectures(cfg):
    for architecture in ("dlinear", "lstm", "patchtst"):
        assert NeuralWindowForecaster(cfg, architecture).architecture == architecture


def test_convex_ensemble_recovers_better_component():
    actual = np.arange(48, dtype=np.float32).reshape(2, 24)
    first = actual.copy()
    second = actual + 2.0
    ensemble = ConvexForecastEnsemble("first", "second").fit(actual, first, second)
    assert ensemble.first_weight == 1.0
    assert np.allclose(ensemble.predict(first, second), actual)


def test_trace_raf_gate_has_validation_no_correction_fallback(cfg):
    rows, horizon = 60, 24
    actual = np.zeros((rows, horizon), dtype=np.float32)
    base = np.zeros_like(actual)
    harmful_correction = np.ones_like(actual)
    features = np.column_stack([np.linspace(0, 1, rows), np.ones(rows)]).astype(np.float32)
    gate = SelectiveResidualGate(cfg).fit(
        actual, base, harmful_correction, features, ["signal", "constant"]
    )
    assert gate.selected_strength == 0.0
    assert np.allclose(gate.predict(base, harmful_correction, features), base)


def test_trace_mechanism_ablation_selectors_use_validation_error():
    actual = np.arange(48, dtype=np.float32).reshape(2, 24)
    base = actual + 2.0
    correction = np.full_like(actual, -2.0)
    analogue = actual.copy()
    grid = (0.0, 0.25, 0.5, 0.75, 1.0)

    residual_strength, residual_scores = choose_residual_strength(
        actual, base, correction, grid
    )
    analogue_weight, analogue_scores = choose_analogue_weight(
        actual, base, analogue, grid
    )

    assert residual_strength == 1.0
    assert analogue_weight == 1.0
    assert residual_scores[1.0] == 0.0
    assert analogue_scores[1.0] == 0.0
    assert np.allclose(apply_residual_correction(base, correction, 1.0), actual)
    assert np.allclose(blend_base_with_analogue(base, analogue, 1.0), actual)


def test_trace_mechanism_ablation_helpers_reject_invalid_weights():
    values = np.zeros((2, 24), dtype=np.float32)
    with np.testing.assert_raises(ValueError):
        apply_residual_correction(values, values, 1.1)
    with np.testing.assert_raises(ValueError):
        blend_base_with_analogue(values, values, -0.1)


def test_residual_gate_features_are_finite_and_origin_level():
    rows, horizon = 5, 24
    base = np.full((rows, horizon), 10.0, dtype=np.float32)
    correction = np.full_like(base, 0.5)
    spread = np.full_like(base, 0.2)
    disagreement = np.full_like(base, 0.3)
    matrix, names = residual_gate_features(
        base,
        correction,
        spread,
        np.full(rows, 0.7),
        np.full(rows, 0.8),
        np.full(rows, 8),
        np.full(rows, 100),
        np.full(rows, 0.5),
        np.ones(rows, dtype=bool),
        disagreement,
        np.full(rows, 0.4),
        np.zeros(rows, dtype=bool),
    )
    assert matrix.shape == (rows, len(names))
    assert np.isfinite(matrix).all()
