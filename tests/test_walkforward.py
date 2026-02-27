from __future__ import annotations

import pandas as pd

from engine.backtest.walkforward import run_walkforward
from engine.features.feature_sets import build_feature_frame


def test_walkforward_end_to_end(synthetic_ohlcv: pd.DataFrame) -> None:
    ff = build_feature_frame(synthetic_ohlcv, "SYN", horizon=1, lags=1)
    cfg = {
        "seed": 7,
        "backtest": {"train_window": 50, "step": 1},
        "risk": {
            "target_vol": 0.1,
            "max_leverage": 1.0,
            "max_position": 1.0,
            "turnover_cap": 0.4,
            "drawdown_stop": 0.2,
            "linear_cost_bps": 5.0,
            "spread_bps": 1.0,
        },
        "promotion": {"min_improvement": 0.0, "min_calibration": 0.1, "max_drawdown": 0.9},
    }
    result = run_walkforward(ff, cfg, model_name="linear_ridge")
    assert "metrics" in result
    assert len(result["net_curve"]) > 0
    assert "gross_return" in result["metrics"]
    assert "net_return" in result["metrics"]
