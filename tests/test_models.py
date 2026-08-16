from __future__ import annotations

import numpy as np

from event_timeraf.models import DirectRidgeForecaster, hour_month_climatology_forecast
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
