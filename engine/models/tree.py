from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from engine.models.base import BaseModel


class RandomForestModel(BaseModel):
    model_id = "rf_regressor"

    def __init__(self, random_state: int = 42) -> None:
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=random_state,
            min_samples_leaf=5,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.model.fit(x, y)
        return self

    def predict(
        self, x: pd.DataFrame, timestamps: pd.DatetimeIndex, horizon: int
    ) -> list[dict[str, Any]]:
        preds = np.array([t.predict(x) for t in self.model.estimators_])
        means = preds.mean(axis=0)
        stdevs = preds.std(axis=0)
        out: list[dict[str, Any]] = []
        for ts, mean, std in zip(timestamps, means, stdevs):
            out.append(
                {
                    "timestamp": ts.isoformat(),
                    "horizon": horizon,
                    "mean_return": float(mean),
                    "stdev": float(std),
                    "quantiles": {
                        "p05": float(np.quantile(preds[:, len(out)], 0.05)),
                        "p50": float(np.quantile(preds[:, len(out)], 0.50)),
                        "p95": float(np.quantile(preds[:, len(out)], 0.95)),
                    },
                    "calibration_score": 0.0,
                    "provenance": {"model_id": self.model_id},
                }
            )
        return out
