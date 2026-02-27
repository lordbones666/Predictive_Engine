from __future__ import annotations

from typing import Any

import pandas as pd

from engine.models.base import BaseModel


class ZeroReturnModel(BaseModel):
    model_id = "baseline_zero"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "ZeroReturnModel":
        self.sigma = float(y.std(ddof=0) if len(y) else 0.0)
        return self

    def predict(
        self, x: pd.DataFrame, timestamps: pd.DatetimeIndex, horizon: int
    ) -> list[dict[str, Any]]:
        return [
            {
                "timestamp": ts.isoformat(),
                "horizon": horizon,
                "mean_return": 0.0,
                "stdev": self.sigma,
                "quantiles": self._quantiles(0.0, self.sigma),
                "calibration_score": 0.0,
                "provenance": {"model_id": self.model_id},
            }
            for ts in timestamps
        ]


class RollingMeanModel(BaseModel):
    model_id = "baseline_rolling_mean"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "RollingMeanModel":
        self.train_returns = y.copy()
        self.sigma = float(y.std(ddof=0))
        return self

    def predict(
        self, x: pd.DataFrame, timestamps: pd.DatetimeIndex, horizon: int
    ) -> list[dict[str, Any]]:
        mean = (
            float(self.train_returns.tail(self.window).mean()) if len(self.train_returns) else 0.0
        )
        return [
            {
                "timestamp": ts.isoformat(),
                "horizon": horizon,
                "mean_return": mean,
                "stdev": self.sigma,
                "quantiles": self._quantiles(mean, self.sigma),
                "calibration_score": 0.0,
                "provenance": {"model_id": self.model_id},
            }
            for ts in timestamps
        ]


class RandomWalkReturnModel(ZeroReturnModel):
    model_id = "baseline_random_walk_return"
