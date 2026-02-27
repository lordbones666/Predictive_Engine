from __future__ import annotations

from pathlib import Path

import pandas as pd


def _canonicalize(df: pd.DataFrame, symbol: str, timezone: str = "UTC") -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        out = out.set_index("timestamp")
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    out = out.tz_convert(timezone)
    out["return_1"] = (out["close"]).pct_change().fillna(0.0)
    out["symbol"] = symbol
    return out[["open", "high", "low", "close", "volume", "return_1", "symbol"]]


def load_csv(path: str | Path, symbol: str, timezone: str = "UTC") -> pd.DataFrame:
    return _canonicalize(pd.read_csv(path), symbol=symbol, timezone=timezone)


def load_parquet(path: str | Path, symbol: str, timezone: str = "UTC") -> pd.DataFrame:
    return _canonicalize(pd.read_parquet(path), symbol=symbol, timezone=timezone)


def load_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf  # optional dependency loaded lazily

    data = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    data = data.rename_axis("timestamp").reset_index()
    return _canonicalize(data, symbol=symbol)
