from __future__ import annotations

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
