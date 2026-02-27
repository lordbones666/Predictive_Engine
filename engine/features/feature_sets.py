from __future__ import annotations

from typing import Any

import pandas as pd

from engine.features.transforms import atr, ema_slope, log_return, macd, realized_vol, rsi

FEATURE_SET_VERSION = "v1.0.0"


def build_feature_frame(
    market_df: pd.DataFrame,
    symbol: str,
    horizon: int,
    lags: int,
    feature_set_version: str = FEATURE_SET_VERSION,
) -> dict[str, Any]:
    df = market_df.copy().sort_index()
    df["log_return"] = log_return(df["close"])
    df["ret_h3"] = df["close"].pct_change(3).fillna(0.0)
    df["realized_vol"] = realized_vol(df["log_return"], 20)
    df["atr"] = atr(df)
    df["ema_slope"] = ema_slope(df["close"])
    df["rsi"] = rsi(df["close"])
    df["macd"] = macd(df["close"])
    df["volume_z"] = (
        (df["volume"] - df["volume"].rolling(20).mean())
        / (df["volume"].rolling(20).std(ddof=0) + 1e-12)
    ).fillna(0.0)
    df["range_z"] = (
        ((df["high"] - df["low"]) - (df["high"] - df["low"]).rolling(20).mean())
        / ((df["high"] - df["low"]).rolling(20).std(ddof=0) + 1e-12)
    ).fillna(0.0)
    rolling_max = df["close"].rolling(50).max()
    df["drawdown_depth"] = ((df["close"] / (rolling_max + 1e-12)) - 1.0).fillna(0.0)
    df["trend_strength"] = df["ema_slope"].rolling(20).mean().fillna(0.0)

    for col in [
        "log_return",
        "ret_h3",
        "realized_vol",
        "atr",
        "ema_slope",
        "rsi",
        "macd",
        "volume_z",
        "range_z",
        "drawdown_depth",
        "trend_strength",
    ]:
        df[col] = df[col].shift(lags)

    df["target"] = df["close"].pct_change(horizon).shift(-horizon)
    df = df.dropna()

    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        ts = pd.Timestamp(str(idx))
        records.append(
            {
                "timestamp": ts.isoformat(),
                "target": float(row["target"]),
                "log_return": float(row["log_return"]),
                "ret_h3": float(row["ret_h3"]),
                "realized_vol": float(row["realized_vol"]),
                "atr": float(row["atr"]),
                "ema_slope": float(row["ema_slope"]),
                "rsi": float(row["rsi"]),
                "macd": float(row["macd"]),
                "volume_z": float(row["volume_z"]),
                "range_z": float(row["range_z"]),
                "drawdown_depth": float(row["drawdown_depth"]),
                "trend_strength": float(row["trend_strength"]),
            }
        )
    return {
        "symbol": symbol,
        "horizon": horizon,
        "lags": lags,
        "feature_set_version": feature_set_version,
        "records": records,
        "metadata": {"row_count": len(records)},
    }


def to_xy(feature_frame: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    records = feature_frame["records"]
    df = pd.DataFrame(records)
    idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    y = pd.Series(df["target"], dtype=float)
    x = df.drop(columns=["timestamp", "target"]).astype(float)
    return x, y, idx
