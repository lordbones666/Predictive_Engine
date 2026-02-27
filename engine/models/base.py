from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    model_id: str

    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "BaseModel": ...

    @abstractmethod
    def predict(
        self, x: pd.DataFrame, timestamps: pd.DatetimeIndex, horizon: int
    ) -> list[dict[str, Any]]: ...

    @staticmethod
    def _quantiles(mean: float, stdev: float) -> dict[str, float]:
        z05, z50, z95 = -1.6448536269514722, 0.0, 1.6448536269514722
        return {
            "p05": float(mean + z05 * stdev),
            "p50": float(mean + z50 * stdev),
            "p95": float(mean + z95 * stdev),
        }


def calibration_score(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> float:
    within = np.abs(y_true - y_pred) <= 1.96 * np.maximum(sigma, 1e-9)
    return float(within.mean())
