from __future__ import annotations

import pytest

from engine.data.schema import emit_json_schemas, validate_forecast


def test_forecast_schema_roundtrip() -> None:
    emit_json_schemas()
    forecast = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "horizon": 1,
        "mean_return": 0.01,
        "stdev": 0.02,
        "quantiles": {"p05": -0.02, "p50": 0.01, "p95": 0.04},
        "calibration_score": 0.9,
        "provenance": {"model_id": "x"},
    }
    parsed = validate_forecast(forecast)
    assert parsed.mean_return == 0.01


def test_forecast_schema_rejects_invalid() -> None:
    bad = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "horizon": 1,
        "mean_return": 0.01,
        "stdev": -0.02,
        "quantiles": {},
        "calibration_score": 0.9,
        "provenance": {"model_id": "x"},
    }
    with pytest.raises(Exception):
        validate_forecast(bad)
