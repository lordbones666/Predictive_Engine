from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=0)
    return float((returns.mean() / std) * np.sqrt(252)) if std > 0 else 0.0


def sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0].std(ddof=0)
    return float((returns.mean() / downside) * np.sqrt(252)) if downside > 0 else 0.0


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return float(dd.min())
