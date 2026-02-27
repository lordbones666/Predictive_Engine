from __future__ import annotations

import pandas as pd

from engine.data.adapters import load_csv, load_parquet


def test_csv_adapter(synthetic_ohlcv: pd.DataFrame, tmp_path) -> None:
    src = synthetic_ohlcv.reset_index().rename(columns={"index": "timestamp"})
    path = tmp_path / "sample.csv"
    src.to_csv(path, index=False)
    out = load_csv(path, symbol="SYN")
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "return_1", "symbol"]
    assert out["symbol"].iloc[-1] == "SYN"


def test_parquet_adapter(synthetic_ohlcv: pd.DataFrame, tmp_path) -> None:
    src = synthetic_ohlcv.reset_index().rename(columns={"index": "timestamp"})
    path = tmp_path / "sample.parquet"
    src.to_parquet(path, index=False)
    out = load_parquet(path, symbol="SYN")
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "return_1", "symbol"]
    assert out["symbol"].iloc[0] == "SYN"
