# AGENTS.md — Financial Prediction Engine (Reproducible, Testable, Uncertainty-Aware)

This repo is built for **honest forecasting** and **risk-constrained decisions**.
All contributions must preserve: **reproducibility, schema contracts, leakage safety, and walk-forward evaluation**.

---

## BUILD (architecture + artifacts)

### North Star
A modular financial prediction engine with:
- typed, schema-validated interfaces between modules
- deterministic feature computation (no future leakage)
- probabilistic forecasts (uncertainty included)
- decision layer with risk constraints
- walk-forward evaluation + promotion gates
- artifact logging + monitoring-ready outputs

### Layered System Architecture

#### Layer 0 — Contracts First (typed + schema-validated)
Core data contracts (canonical schemas):
- **MarketData**: OHLCV + derived returns; indexed; timezone-aware; adjustment metadata
- **FeatureFrame**: aligned features + targets + metadata (feature version, lags, horizon)
- **Forecast**: point + distribution + horizon + calibration stats + provenance
- **Decision**: position sizing + risk limits + rationale + constraints hit
- **BacktestResult**: metrics + curves + logs + artifacts (paths + hashes)

**Rule:** modules communicate ONLY via these contracts.

#### Layer 1 — Data Ingestion + Quality Gates
Adapters may include: yfinance / Polygon / Alpaca / IBKR / CSV-Parquet.
Quality gates (hard fail or quarantine):
- missing bars, duplicates, out-of-order timestamps
- corporate actions adjustments (split/dividend) sanity checks
- abnormal spikes / stale prices
- explicit lookahead / leakage checks (targets strictly future-only)

Output: validated **MarketData**.

#### Layer 2 — Feature Engineering (deterministic, versioned)
Feature sets are versioned and reproducible:
- returns: log returns, multi-horizon returns
- volatility: realized vol, Parkinson/GK, ATR
- momentum: SMA/EMA slopes, RSI, MACD
- microstructure proxies: volume z-score, ranges, gaps
- regime features: vol state, trend strength, drawdown depth

**Hard rule:** feature computation never sees the future.

#### Layer 3 — Modeling (zoo, unified API)
Standard API:
- `fit(train: FeatureFrame) -> ModelArtifact`
- `predict(X: FeatureFrame, horizon) -> Forecast` (must include uncertainty)

Tiers:
- Tier 0 Baselines: zero return, random walk, rolling mean
- Tier 1 Classical: ridge/elastic net, RF, XGBoost, ARIMA/ETS (careful), GARCH (vol)
- Tier 2 Sequence: LSTM/Transformer only with strict eval + calibration; train on returns

**Principle:** forecast returns or distribution params, not naive price.

#### Layer 4 — Regime Detection (interpretable)
- Rule-based regimes (preferred): trend × vol_state (quantiles) → semantic labels
- Probabilistic regimes: HMM on returns + vol features → regime probabilities

If clustering is used, must include **cluster labeling function** mapping stats → semantics.

#### Layer 5 — Portfolio + Risk Engine (Decision Layer)
Convert forecast distribution → expected return + risk:
- vol targeting / capped Kelly / constrained mean-variance
Risk controls:
- max leverage, max drawdown stop, turnover caps
- exposure caps (asset/sector), transaction costs + slippage model

Output: **Decision(position, confidence, risk_budget, constraints_hit)**.

#### Layer 6 — Evaluation (walk-forward or it’s not real)
Walk-forward:
- rolling train window → predict next horizon → advance

Metrics:
- forecasting: MAE/RMSE on returns; optional direction hit-rate
- calibration: reliability curves; calibration score
- trading: Sharpe/Sortino/MDD/turnover; **net of costs**

Promotion gate (required):
- beats Tier 0 baseline by X% **net costs**
- passes leakage tests
- stable across regimes
- calibrated probabilistic forecasts

#### Layer 7 — Deployment (boring, reliable, observable)
- deterministic config + seeds
- model registry with versioning
- artifact store: features, backtest logs, metrics, plots
- monitoring JSON: drift, performance decay, calibration decay, data feed health

---

## Repo Layout (canonical blueprint)


engine/
init.py
cli.py
config/
default.yaml
data/
adapters.py
quality.py
schema.py
features/
feature_sets.py
transforms.py
regimes/
rule_based.py
hmm.py
models/
base.py
baselines.py
linear.py
tree.py
seq.py
volatility.py
portfolio/
sizing.py
optimizer.py
costs.py
risk.py
backtest/
walkforward.py
metrics.py
plots.py
registry/
model_store.py
artifacts.py
schemas/
market_data.schema.json
feature_frame.schema.json
forecast.schema.json
decision.schema.json
backtest_result.schema.json
tests/
test_quality_gates.py
test_no_lookahead.py
test_feature_alignment.py
test_forecast_schema.py
test_walkforward.py
test_promotion_gates.py
notebooks/ (optional; never authoritative) 
---

## Contracts & Validation Rules

### JSON Schema is mandatory at boundaries
All outputs crossing module boundaries must be:
1) **typed (pydantic/dataclasses)** AND
2) **JSON schema-valid** AND
3) include **provenance**: data source(s), feature version, model version, code hash, seed

### Forecast schema minimum
- `timestamp`, `horizon`
- `mean_return`, `stdev`
- `quantiles` (e.g., p05/p50/p95)
- `calibration_score`
- `model_id`, `feature_set_version`, `training_window`

### Decision schema minimum
- `target_position`
- `risk_budget`
- `expected_return`, `expected_vol`
- `constraints_hit` (list)
- `rationale` (short, structured)

---

## Determinism & Reproducibility (non-negotiable)

- Single config entrypoint: `engine/config/default.yaml`
- Every run logs:
  - config snapshot
  - git commit hash (or code hash)
  - random seeds (Python/NumPy/torch)
  - dataset identifiers (symbol list, date range, vendor)
- Feature computation is **pure** and **versioned**
- No silent defaults: every model/feature must declare explicit parameters

---

## Quality Gates (must exist and must run)

### Data quality gates
- missing/duplicate/out-of-order checks
- corporate action adjustment sanity checks
- spike/stale detection
- timestamp timezone normalization

### Leakage gates
- target alignment: `target_t` uses only info ≤ `t`
- lag checks: features must be properly lagged
- walk-forward only (no random shuffles for time series)

### Promotion gates (model cannot be “production eligible” unless)
- beats baselines net-of-costs
- stable across regimes
- calibrated (for probabilistic forecasts)
- produces schema-valid Forecast + Decision outputs

---

## How to Use GPT (allowed vs disallowed)

### Allowed uses (must be verified)
- propose candidate features (then implemented deterministically)
- summarize backtest artifacts and failure modes
- generate human-readable reports from metrics JSON
- Predictions 

### Disallowed uses
- inventing metrics/causal graphs/SHAP without computation
- changing trading logic without tests + gate rerun
- Inventing data.

Control plane:
- GPT outputs must be **schema-valid**
- Suggestions become PRs with:
  - unit tests
  - walk-forward backtest rerun
  - gate pass

---

## Development Workflow

### Required commands (provide these in README later)
- `python -m engine.cli ingest ...`
- `python -m engine.cli features ...`
- `python -m engine.cli train ...`
- `python -m engine.cli backtest --walkforward ...`
- `python -m engine.cli promote --model_id ...`

### PR checklist (must pass)
- ✅ schemas updated (if contract changes)
- ✅ tests added/updated
- ✅ leakage gates pass
- ✅ walk-forward run produces artifacts
- ✅ baselines comparison included
- ✅ deterministic run reproducible from config

---

## AUDIT PASS (Adversarial Review)

### Likely failure modes (finance reality check)
- non-stationarity → performance decays
- costs/slippage assumptions dominate
- subjective regime definitions can be gamed
- overfitting via repeated research loops

### Mitigations this repo enforces
- walk-forward evaluation + strict baselines
- leakage automation
- artifact logging and monitoring hooks
- explicit calibration + risk constraints

---

## RISKS
- Overfitting from iterative experimentation without a true holdout discipline
- Leakage creeping in via feature alignment, corporate actions, or target construction
- Decision optimization amplifying forecast error if constraints are too loose

---

## ASSUMPTIONS
- Goal is real decision support, not a demo
- You prefer reproducibility + evaluation + risk controls over novelty models

---

## NEXT STEPS
1) Implement schemas + typed contracts (pydantic + JSON Schema export)
2) Build ingestion adapters + quality gates + quarantine path
3) Implement deterministic feature sets + versioning
4) Add Tier 0 baselines + walk-forward harness
5) Add promotion gates + artifact logging
6) Add portfolio/risk decision layer + cost model
7) Add monitoring JSON outputs (drift/perf/calibration)


