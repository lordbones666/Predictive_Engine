from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data.quality import run_quality_gates


def test_quality_gates_find_issues(synthetic_ohlcv: pd.DataFrame) -> None:
    df = synthetic_ohlcv.copy()
    dup_row = df.iloc[[5]].copy()
    dup_row.index = [df.index[6]]
    df = pd.concat([df.iloc[:7], dup_row, df.iloc[7:]])
    df = df.drop(df.index[10])
    df.loc[df.index[20], "close"] = df["close"].iloc[19] * 20
    clean, quarantined = run_quality_gates(df)
    reasons = set(quarantined["reason"].tolist())
    assert "duplicate_timestamp" in reasons
    assert "missing_timestamp" in reasons
    assert "spike_detected" in reasons
    assert "return_1" in clean.columns


def test_quality_gate_does_not_flag_weekend_gaps_for_business_data() -> None:
    idx = pd.bdate_range("2024-01-01", periods=30, tz="UTC")
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000},
        index=idx,
    )
    _, quarantined = run_quality_gates(df)
    reasons = set(quarantined["reason"].tolist()) if not quarantined.empty else set()
    assert "missing_timestamp" not in reasons


def test_quality_gate_flags_missing_business_day() -> None:
    idx = pd.bdate_range("2024-01-01", periods=30, tz="UTC")
    idx = idx.delete(10)
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000},
        index=idx,
    )
    _, quarantined = run_quality_gates(df, expected_freq="B")
    reasons = set(quarantined["reason"].tolist()) if not quarantined.empty else set()
    assert "missing_timestamp" in reasons
