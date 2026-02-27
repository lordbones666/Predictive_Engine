from __future__ import annotations

import pandas as pd

from engine.data.schema import validate_feature_frame
from engine.features.feature_sets import build_feature_frame


def test_feature_frame_contract(synthetic_ohlcv: pd.DataFrame) -> None:
    ff = build_feature_frame(synthetic_ohlcv, "SYN", horizon=1, lags=1)
    parsed = validate_feature_frame(ff)
    assert parsed.feature_set_version == "v1.0.0"
    assert parsed.lags == 1
