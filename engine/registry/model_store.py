from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


class ModelStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, model: Any, model_id: str, metadata: dict[str, Any]) -> dict[str, str]:
        model_path = self.root / f"{model_id}.joblib"
        meta_path = self.root / f"{model_id}.metadata.json"
        joblib.dump(model, model_path)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"model": str(model_path), "metadata": str(meta_path)}
