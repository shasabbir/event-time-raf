from __future__ import annotations

import numpy as np

from event_timeraf.models import DirectRidgeForecaster, HourlyMonthlyClimatology
from event_timeraf.windows import build_window_dataset


def test_hourly_monthly_climatology_produces_finite_forecasts(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    model = HourlyMonthlyClimatology(cfg.timezone).fit(dataset.subset("train"))
    prediction = model.predict(dataset.subset("validation"))
    assert prediction.shape == dataset.subset("validation").y.shape
    assert np.isfinite(prediction).all()


def test_direct_ridge_forecaster_produces_one_model_per_horizon(modeling_frame, cfg):
    dataset = build_window_dataset(modeling_frame, cfg)
    train = dataset.subset("train")
    validation = dataset.subset("validation")
    model = DirectRidgeForecaster(cfg, alpha=1.0).fit(
        train.features,
        train.future_calendar,
        train.y,
        list(train.feature_names),
        train.calendar_names,
    )
    prediction = model.predict(validation.features, validation.future_calendar)
    assert prediction.shape == validation.y.shape
    assert len(model.models) == train.y.shape[1]
    assert np.isfinite(prediction).all()
