from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge

from engine.models.base import BaseModel


class LinearModel(BaseModel):
    def __init__(self, model_type: str = "ridge", random_state: int = 42) -> None:
        self.model_id = f"linear_{model_type}"
        if model_type == "elastic_net":
            self.model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state)
        else:
            self.model = Ridge(alpha=1.0, random_state=random_state)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "LinearModel":
        self.model.fit(x, y)
        residual = y.values - self.model.predict(x)
        self.sigma = float(np.std(residual))
        return self

    def predict(
        self, x: pd.DataFrame, timestamps: pd.DatetimeIndex, horizon: int
    ) -> list[dict[str, Any]]:
        means = self.model.predict(x)
        return [
            {
                "timestamp": ts.isoformat(),
                "horizon": horizon,
                "mean_return": float(mu),
                "stdev": self.sigma,
                "quantiles": self._quantiles(float(mu), self.sigma),
                "calibration_score": 0.0,
                "provenance": {"model_id": self.model_id},
            }
            for ts, mu in zip(timestamps, means)
        ]
