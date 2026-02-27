from __future__ import annotations

import numpy as np
import pandas as pd


def log_return(series: pd.Series) -> pd.Series:
    return pd.Series(np.log(series).diff().fillna(0.0), index=series.index, dtype=float)


def realized_vol(series: pd.Series, window: int = 20) -> pd.Series:
    return pd.Series(
        series.rolling(window).std(ddof=0).fillna(0.0), index=series.index, dtype=float
    )


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return pd.Series(tr.rolling(window).mean().fillna(0.0), index=df.index, dtype=float)


def ema_slope(series: pd.Series, span: int = 10) -> pd.Series:
    ema = series.ewm(span=span, adjust=False).mean()
    return pd.Series(ema.diff().fillna(0.0), index=series.index, dtype=float)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff().fillna(0.0)
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down + 1e-12)
    return pd.Series((100 - (100 / (1 + rs))).fillna(50.0), index=series.index, dtype=float)


def macd(series: pd.Series) -> pd.Series:
    fast = series.ewm(span=12, adjust=False).mean()
    slow = series.ewm(span=26, adjust=False).mean()
    return pd.Series((fast - slow).fillna(0.0), index=series.index, dtype=float)
