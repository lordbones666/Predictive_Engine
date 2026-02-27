from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=220, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, size=len(idx)))
    open_ = close + rng.normal(0, 0.2, size=len(idx))
    high = np.maximum(open_, close) + rng.uniform(0, 0.5, size=len(idx))
    low = np.minimum(open_, close) - rng.uniform(0, 0.5, size=len(idx))
    volume = rng.integers(1000, 5000, size=len(idx))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
