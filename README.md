# Predictive Engine

A contracts-first, leakage-safe, uncertainty-aware financial prediction engine with walk-forward evaluation.

## Setup

```bash
python -m pip install -e '.[dev]'
```

## Core commands

```bash
python -m engine.cli ingest --config engine/config/default.yaml --input sample.csv
python -m engine.cli features --config engine/config/default.yaml --market-data market_data.json
python -m engine.cli train
python -m engine.cli backtest --config engine/config/default.yaml --features feature_frame.json --walkforward --model linear_ridge --run-baselines
python -m engine.cli promote --config engine/config/default.yaml --candidate candidate.json --baseline baseline.json
python -c "import engine; import engine.data.schema"
```

## Guardrails implemented
- JSON-schema + typed contract validation at boundaries.
- Deterministic feature construction with explicit horizon and lags.
- Walk-forward backtesting only.
- Forecast distributions include mean, stdev, quantiles.
- Decision layer applies costs + hard risk constraints.
- Baseline-vs-candidate comparison outputs and promotion checks.
- Artifact logging under `artifacts/run_<timestamp>/`.

## Tooling

```bash
black .
isort .
ruff check .
mypy engine
pytest --cov=engine --cov-report=term-missing
```
