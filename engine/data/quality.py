from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class QuarantineItem:
    timestamp: pd.Timestamp
    reason: str


def _infer_expected_index(index: pd.DatetimeIndex, expected_freq: str | None) -> pd.DatetimeIndex:
    if expected_freq is not None:
        return pd.date_range(index.min(), index.max(), freq=expected_freq, tz="UTC")

    inferred = pd.infer_freq(index)
    if inferred is not None:
        return pd.date_range(index.min(), index.max(), freq=inferred, tz="UTC")

    weekday_share = float((index.dayofweek < 5).mean()) if len(index) else 0.0
    if weekday_share > 0.95:
        return pd.date_range(index.min(), index.max(), freq="B", tz="UTC")
    return pd.date_range(index.min(), index.max(), freq="D", tz="UTC")


def run_quality_gates(
    df: pd.DataFrame,
    stale_threshold: int = 3,
    spike_sigma: float = 8.0,
    expected_freq: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = df.copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).sort_values()
    frame = frame.sort_index()
    quarantined: list[QuarantineItem] = []

    duplicated = frame.index.duplicated(keep="first")
    for ts in frame.index[duplicated]:
        quarantined.append(QuarantineItem(timestamp=pd.Timestamp(ts), reason="duplicate_timestamp"))
    frame = frame[~duplicated]

    expected = _infer_expected_index(pd.DatetimeIndex(frame.index), expected_freq)
    missing = expected.difference(pd.DatetimeIndex(frame.index))
    for ts in missing:
        quarantined.append(QuarantineItem(timestamp=pd.Timestamp(ts), reason="missing_timestamp"))

    returns = frame["close"].pct_change().fillna(0.0)
    z = ((returns - returns.mean()) / (returns.std(ddof=0) + 1e-12)).abs()
    for ts in z[z > spike_sigma].index:
        quarantined.append(QuarantineItem(timestamp=pd.Timestamp(ts), reason="spike_detected"))

    stale_run = (frame["close"].diff().fillna(0).abs() < 1e-12).astype(int)
    stale_run = stale_run.groupby((stale_run != stale_run.shift()).cumsum()).cumsum()
    for ts in stale_run[stale_run >= stale_threshold].index:
        quarantined.append(QuarantineItem(timestamp=pd.Timestamp(ts), reason="stale_price"))

    if (frame["low"] > frame["high"]).any():
        for ts in frame[frame["low"] > frame["high"]].index:
            quarantined.append(QuarantineItem(timestamp=pd.Timestamp(ts), reason="bad_ohlc"))

    reasons = pd.DataFrame(
        [{"timestamp": q.timestamp.isoformat(), "reason": q.reason} for q in quarantined]
    ).drop_duplicates()
    frame["return_1"] = frame["close"].pct_change().fillna(0.0)
    return frame, reasons


def to_market_data(
    clean_df: pd.DataFrame,
    symbol: str,
    timezone: str = "UTC",
    adjustments: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = clean_df.copy()
    data.index = pd.to_datetime(data.index, utc=True).tz_convert(timezone)
    recs: list[dict[str, Any]] = []
    for idx, row in data.iterrows():
        ts = pd.Timestamp(str(idx))
        recs.append(
            {
                "timestamp": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "return_1": float(row["return_1"]),
            }
        )
    return {
        "symbol": symbol,
        "timezone": timezone,
        "records": recs,
        "adjustments": adjustments or {"status": "unadjusted"},
    }
