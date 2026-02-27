from __future__ import annotations

import numpy as np
import pandas as pd

from engine.backtest.walkforward import run_walkforward
from engine.features.feature_sets import build_feature_frame


def _config(cost_bps: float, drawdown_stop: float) -> dict[str, object]:
    return {
        "seed": 7,
        "backtest": {"train_window": 50, "step": 1},
        "risk": {
            "target_vol": 0.1,
            "max_leverage": 1.0,
            "max_position": 1.0,
            "turnover_cap": 0.5,
            "drawdown_stop": drawdown_stop,
            "linear_cost_bps": cost_bps,
            "spread_bps": 0.0,
        },
        "promotion": {"min_improvement": 0.0, "min_calibration": 0.1, "max_drawdown": 0.95},
    }


def test_higher_costs_reduce_net_return(synthetic_ohlcv: pd.DataFrame) -> None:
    ff = build_feature_frame(synthetic_ohlcv, "SYN", horizon=1, lags=1)
    low_cost = run_walkforward(
        ff, _config(cost_bps=1.0, drawdown_stop=0.5), model_name="linear_ridge"
    )
    high_cost = run_walkforward(
        ff, _config(cost_bps=100.0, drawdown_stop=0.5), model_name="linear_ridge"
    )
    assert high_cost["metrics"]["net_return"] < low_cost["metrics"]["net_return"]


def test_drawdown_stop_clamps_positions() -> None:
    idx = pd.date_range("2024-01-01", periods=180, freq="D", tz="UTC")
    up = np.linspace(100, 130, 120)
    down = np.linspace(130, 70, 60)
    close = pd.Series(np.concatenate([up, down]), index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1500},
        index=idx,
    )
    ff = build_feature_frame(df, "SYN", horizon=1, lags=1)
    cfg = _config(cost_bps=5.0, drawdown_stop=0.005)
    cfg["risk"]["target_vol"] = 1.0
    result = run_walkforward(ff, cfg, model_name="baseline_roll")
    assert result["constraint_hit_counts"].get("drawdown_stop", 0) > 0
