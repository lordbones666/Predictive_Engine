from __future__ import annotations

from engine.backtest.walkforward import promotion_gate


def test_promotion_gate() -> None:
    candidate = {"metrics": {"net_return": 0.12, "calibration_score": 0.7, "mdd": -0.1}}
    baseline = {"metrics": {"net_return": 0.1, "calibration_score": 0.5, "mdd": -0.1}}
    cfg = {"promotion": {"min_improvement": 0.01, "min_calibration": 0.6, "max_drawdown": 0.2}}
    ok, reasons = promotion_gate(candidate, baseline, cfg)
    assert ok is True
    assert reasons == []
