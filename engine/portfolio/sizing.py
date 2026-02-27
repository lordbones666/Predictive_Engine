from __future__ import annotations


def vol_target_position(
    expected_return: float, expected_vol: float, target_vol: float, cap: float
) -> float:
    if expected_vol <= 1e-8:
        return 0.0
    raw = (expected_return / expected_vol) * target_vol
    return max(-cap, min(cap, raw))
