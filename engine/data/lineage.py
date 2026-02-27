from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_json(obj: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True).encode("utf-8"))


def sha256_dataframe(df: pd.DataFrame) -> str:
    serialized = df.sort_index().to_csv(index=True).encode("utf-8")
    return sha256_bytes(serialized)
