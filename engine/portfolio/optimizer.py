from __future__ import annotations

from typing import Any

from engine.portfolio.costs import transaction_cost
from engine.portfolio.risk import enforce_constraints
from engine.portfolio.sizing import vol_target_position


def make_decision(
    forecast: dict[str, Any],
    current_position: float,
    drawdown: float,
    risk_budget: float,
    config: dict[str, float],
) -> dict[str, Any]:
    mu = float(forecast["mean_return"])
    sigma = max(float(forecast["stdev"]), 1e-8)
    proposed = vol_target_position(mu, sigma, config["target_vol"], config["max_position"])
    constrained, constraints_hit = enforce_constraints(
        proposed,
        current_position,
        config["max_leverage"],
        config["max_position"],
        config["turnover_cap"],
        drawdown,
        config["drawdown_stop"],
    )
    turnover = abs(constrained - current_position)
    cost = transaction_cost(turnover, config["linear_cost_bps"], config["spread_bps"])
    return {
        "timestamp": forecast["timestamp"],
        "target_position": constrained,
        "risk_budget": risk_budget,
        "expected_return": mu - cost,
        "expected_vol": sigma,
        "constraints_hit": constraints_hit,
        "transaction_cost": cost,
        "turnover": turnover,
        "rationale": f"vol_target={config['target_vol']};cost={cost:.6f};turnover={turnover:.4f}",
    }
