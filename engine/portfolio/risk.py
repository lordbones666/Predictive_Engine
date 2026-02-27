from __future__ import annotations


def enforce_constraints(
    proposed_position: float,
    current_position: float,
    max_leverage: float,
    max_position: float,
    turnover_cap: float,
    drawdown: float,
    drawdown_stop: float,
) -> tuple[float, list[str]]:
    constraints_hit: list[str] = []
    position = proposed_position

    if abs(position) > max_position:
        constraints_hit.append("max_position")
        position = max(-max_position, min(max_position, position))

    if abs(position) > max_leverage:
        constraints_hit.append("max_leverage")
        position = max(-max_leverage, min(max_leverage, position))

    turnover = abs(position - current_position)
    if turnover > turnover_cap:
        constraints_hit.append("turnover_cap")
        position = current_position + (
            turnover_cap if position > current_position else -turnover_cap
        )

    if drawdown <= -abs(drawdown_stop):
        constraints_hit.append("drawdown_stop")
        position = 0.0

    return position, constraints_hit
