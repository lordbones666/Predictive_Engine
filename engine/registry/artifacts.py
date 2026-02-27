from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_run_artifacts(
    config: dict[str, Any],
    metrics: dict[str, float],
    forecasts: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    summary: str,
    root: Path = Path("artifacts"),
) -> dict[str, str]:
    run_dir = root / f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "config_snapshot.yaml"
    metrics_path = run_dir / "metrics.json"
    forecasts_path = run_dir / "forecasts.jsonl"
    decisions_path = run_dir / "decisions.jsonl"
    summary_path = run_dir / "summary.md"

    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    forecasts_path.write_text("\n".join(json.dumps(x) for x in forecasts) + "\n", encoding="utf-8")
    decisions_path.write_text("\n".join(json.dumps(x) for x in decisions) + "\n", encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")

    hashes = {
        "config": _sha256_text(cfg_path.read_text(encoding="utf-8")),
        "metrics": _sha256_text(metrics_path.read_text(encoding="utf-8")),
        "forecasts": _sha256_text(forecasts_path.read_text(encoding="utf-8")),
        "decisions": _sha256_text(decisions_path.read_text(encoding="utf-8")),
    }
    return {
        "run_dir": str(run_dir),
        "config": str(cfg_path),
        "metrics": str(metrics_path),
        "forecasts": str(forecasts_path),
        "decisions": str(decisions_path),
        "summary": str(summary_path),
        "hashes": json.dumps(hashes),
    }
