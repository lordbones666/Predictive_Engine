from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "engine.cli", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def test_cli_end_to_end_offline(synthetic_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    synthetic_ohlcv.reset_index().rename(columns={"index": "timestamp"}).to_csv(
        csv_path, index=False
    )

    market_path = tmp_path / "market_data.json"
    quarantine_path = tmp_path / "quarantine.json"
    features_path = tmp_path / "feature_frame.json"
    comparison_path = tmp_path / "comparison.json"
    context_path = tmp_path / "context.json"
    support_path = tmp_path / "support.json"

    cfg = yaml.safe_load(Path("engine/config/default.yaml").read_text(encoding="utf-8"))
    assert isinstance(cfg["risk"]["turnover_cap"], float)

    context_path.write_text(
        json.dumps(
            [{"source": "calendar", "timestamp": "2024-01-01T00:00:00Z", "note": "earnings week"}],
            indent=2,
        ),
        encoding="utf-8",
    )

    _run(
        [
            "ingest",
            "--config",
            "engine/config/default.yaml",
            "--input",
            str(csv_path),
            "--output",
            str(market_path),
            "--quarantine",
            str(quarantine_path),
        ],
        cwd=Path.cwd(),
    )
    _run(
        [
            "features",
            "--config",
            "engine/config/default.yaml",
            "--market-data",
            str(market_path),
            "--output",
            str(features_path),
        ],
        cwd=Path.cwd(),
    )
    _run(
        [
            "backtest",
            "--config",
            "engine/config/default.yaml",
            "--features",
            str(features_path),
            "--walkforward",
            "--model",
            "linear_ridge",
            "--run-baselines",
            "--comparison-output",
            str(comparison_path),
        ],
        cwd=Path.cwd(),
    )

    _run(
        [
            "analyze",
            "--comparison",
            str(comparison_path),
            "--external-context",
            str(context_path),
            "--output",
            str(support_path),
        ],
        cwd=Path.cwd(),
    )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert "candidate" in comparison
    assert "baselines" in comparison
    assert "baseline_zero" in comparison["baselines"]
    assert Path(comparison["candidate"]["artifact_paths"]["run_dir"]).exists()
    support = json.loads(support_path.read_text(encoding="utf-8"))
    assert support["latest_signal"]["signal"] in {"LONG", "SHORT", "FLAT"}
    assert len(support["external_context"]) == 1
