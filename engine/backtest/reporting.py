from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [cast(dict[str, Any], json.loads(line)) for line in lines]


def build_decision_support_report(
    comparison_payload: dict[str, Any],
    external_context: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    candidate = comparison_payload["candidate"]
    run_dir = Path(candidate["artifact_paths"]["run_dir"])
    forecasts = _load_jsonl(run_dir / "forecasts.jsonl")
    decisions = _load_jsonl(run_dir / "decisions.jsonl")

    latest_forecast = forecasts[-1]
    latest_decision = decisions[-1]
    promotion_summary = comparison_payload.get("promotion_vs_baseline", {})

    constraints: dict[str, int] = {}
    for decision in decisions:
        for hit in decision.get("constraints_hit", []):
            constraints[hit] = constraints.get(hit, 0) + 1

    signal = "FLAT"
    if float(latest_decision["target_position"]) > 0.05:
        signal = "LONG"
    elif float(latest_decision["target_position"]) < -0.05:
        signal = "SHORT"

    expected_return = float(latest_decision["expected_return"])
    expected_vol = max(float(latest_decision["expected_vol"]), 1e-9)
    conviction = expected_return / expected_vol

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_of_truth": {
            "comparison_payload": "engine.cli backtest --run-baselines output",
            "candidate_run_dir": str(run_dir),
        },
        "candidate_model": comparison_payload["candidate_model"],
        "promotion_vs_baseline": promotion_summary,
        "latest_signal": {
            "timestamp": latest_decision["timestamp"],
            "signal": signal,
            "target_position": latest_decision["target_position"],
            "expected_return": expected_return,
            "expected_vol": expected_vol,
            "conviction": conviction,
            "uncertainty_quantiles": latest_forecast["quantiles"],
            "rationale": latest_decision["rationale"],
        },
        "constraint_hit_counts": constraints,
        "guardrails": {
            "note": (
                "External context is explanatory only; model forecasts and "
                "decision outputs are unchanged."
            ),
            "no_override": True,
        },
        "external_context": external_context or [],
    }
    return report


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
