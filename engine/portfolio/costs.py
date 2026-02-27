from __future__ import annotations


def transaction_cost(turnover: float, linear_bps: float = 5.0, spread_bps: float = 2.0) -> float:
    return turnover * (linear_bps + spread_bps) / 10000.0
