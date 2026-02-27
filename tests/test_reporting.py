from __future__ import annotations

import json
from pathlib import Path

from engine.backtest.reporting import build_decision_support_report


def test_reporting_builds_actionable_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "forecasts.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "quantiles": {"p05": -0.01, "p50": 0.001, "p95": 0.02},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "decisions.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "target_position": 0.2,
                "expected_return": 0.003,
                "expected_vol": 0.01,
                "constraints_hit": ["turnover_cap"],
                "rationale": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    comparison = {
        "candidate_model": "linear_ridge",
        "candidate": {"artifact_paths": {"run_dir": str(run_dir)}},
        "promotion_vs_baseline": {"baseline_zero": {"promoted": True, "reasons": []}},
    }

    report = build_decision_support_report(comparison)
    assert report["latest_signal"]["signal"] == "LONG"
    assert report["latest_signal"]["uncertainty_quantiles"]["p95"] == 0.02
    assert report["guardrails"]["no_override"] is True
