from __future__ import annotations

from pathlib import Path


def write_placeholder_plot(path: Path) -> None:
    path.write_text("plot generation not enabled in minimal build\n", encoding="utf-8")
