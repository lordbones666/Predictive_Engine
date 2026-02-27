from __future__ import annotations

from engine.portfolio.risk import enforce_constraints


def test_constraints_enforced() -> None:
    pos, hits = enforce_constraints(
        proposed_position=2.0,
        current_position=0.0,
        max_leverage=1.0,
        max_position=0.8,
        turnover_cap=0.2,
        drawdown=0.0,
        drawdown_stop=0.2,
    )
    assert abs(pos) <= 0.2
    assert "max_position" in hits
    assert "turnover_cap" in hits
