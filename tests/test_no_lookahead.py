from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features.feature_sets import build_feature_frame


def test_features_no_future_leakage() -> None:
    idx = pd.date_range("2023-01-01", periods=60, freq="D", tz="UTC")
    close = pd.Series(np.arange(1, 61, dtype=float), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        },
        index=idx,
    )
    ff = build_feature_frame(df, symbol="SYN", horizon=1, lags=1)
    rec = ff["records"][5]
    ts = pd.Timestamp(rec["timestamp"])

    expected_target = (df.loc[ts + pd.Timedelta(days=1), "close"] / df.loc[ts, "close"]) - 1
    assert rec["target"] == expected_target

    log_ret_series = np.log(df["close"]).diff().fillna(0.0).shift(1)
    assert rec["log_return"] == float(log_ret_series.loc[ts])
