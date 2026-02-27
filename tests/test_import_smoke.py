from __future__ import annotations


def test_import_smoke() -> None:
    import engine  # noqa: F401
    import engine.data.schema  # noqa: F401
