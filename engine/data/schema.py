from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


class MarketData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    timezone: str
    records: list[dict[str, float | str]]
    adjustments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("records")
    @classmethod
    def validate_records(cls, value: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
        required = {"timestamp", "open", "high", "low", "close", "volume", "return_1"}
        for row in value:
            missing = required - row.keys()
            if missing:
                raise ValueError(f"missing keys: {missing}")
        return value


class FeatureFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    horizon: int
    lags: int
    feature_set_version: str
    records: list[dict[str, float | str]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    horizon: int
    mean_return: float
    stdev: float = Field(ge=0)
    quantiles: dict[str, float]
    calibration_score: float
    provenance: dict[str, Any]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    target_position: float
    risk_budget: float
    expected_return: float
    expected_vol: float
    constraints_hit: list[str]
    transaction_cost: float = Field(ge=0)
    turnover: float = Field(ge=0)
    rationale: str


class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metrics: dict[str, float]
    gross_curve: list[dict[str, float | str]]
    net_curve: list[dict[str, float | str]]
    drawdown_series: list[dict[str, float | str]]
    constraint_hit_counts: dict[str, int]
    logs: list[str]
    artifact_paths: dict[str, str]
    hashes: dict[str, str]


def _load_schema(name: str) -> dict[str, Any]:
    with (SCHEMA_DIR / name).open("r", encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
        return loaded


def _validate(obj: dict[str, Any], schema_file: str) -> None:
    jsonschema.validate(instance=obj, schema=_load_schema(schema_file))


def validate_market_data(obj: dict[str, Any]) -> MarketData:
    _validate(obj, "market_data.schema.json")
    return MarketData.model_validate(obj)


def validate_feature_frame(obj: dict[str, Any]) -> FeatureFrame:
    _validate(obj, "feature_frame.schema.json")
    return FeatureFrame.model_validate(obj)


def validate_forecast(obj: dict[str, Any]) -> Forecast:
    _validate(obj, "forecast.schema.json")
    return Forecast.model_validate(obj)


def validate_decision(obj: dict[str, Any]) -> Decision:
    _validate(obj, "decision.schema.json")
    return Decision.model_validate(obj)


def validate_backtest_result(obj: dict[str, Any]) -> BacktestResult:
    _validate(obj, "backtest_result.schema.json")
    return BacktestResult.model_validate(obj)


def emit_json_schemas() -> None:
    schema_map: dict[str, type[BaseModel]] = {
        "market_data.schema.json": MarketData,
        "feature_frame.schema.json": FeatureFrame,
        "forecast.schema.json": Forecast,
        "decision.schema.json": Decision,
        "backtest_result.schema.json": BacktestResult,
    }
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, model in schema_map.items():
        with (SCHEMA_DIR / filename).open("w", encoding="utf-8") as fh:
            json.dump(model.model_json_schema(), fh, indent=2)
