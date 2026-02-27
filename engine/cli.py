from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from engine.backtest.walkforward import promotion_gate, run_walkforward
from engine.data.adapters import load_csv, load_parquet
from engine.data.quality import run_quality_gates, to_market_data
from engine.data.schema import emit_json_schemas, validate_feature_frame, validate_market_data
from engine.features.feature_sets import build_feature_frame

DEFAULT_BASELINES = ["baseline_zero", "baseline_rw", "baseline_roll"]


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return dict(cfg)


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    df = (
        load_csv(args.input, cfg["ingest"]["symbol"])
        if args.input.endswith(".csv")
        else load_parquet(args.input, cfg["ingest"]["symbol"])
    )
    clean, quarantined = run_quality_gates(df)
    market = to_market_data(
        clean, symbol=cfg["ingest"]["symbol"], timezone=cfg["ingest"]["timezone"]
    )
    validate_market_data(market)
    Path(args.output).write_text(json.dumps(market, indent=2), encoding="utf-8")
    Path(args.quarantine).write_text(
        quarantined.to_json(orient="records", indent=2), encoding="utf-8"
    )


def cmd_features(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    market = json.loads(Path(args.market_data).read_text(encoding="utf-8"))
    mdf = pd.DataFrame(market["records"])
    mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], utc=True)
    mdf = mdf.set_index("timestamp")
    ff = build_feature_frame(
        mdf, market["symbol"], cfg["features"]["horizon"], cfg["features"]["lags"]
    )
    validate_feature_frame(ff)
    Path(args.output).write_text(json.dumps(ff, indent=2), encoding="utf-8")


def cmd_train(_: argparse.Namespace) -> None:
    print("Train command is integrated into walk-forward backtest for this initial build.")


def _parse_baselines(raw: str) -> list[str]:
    baselines = [x.strip() for x in raw.split(",") if x.strip()]
    return baselines if baselines else DEFAULT_BASELINES


def cmd_backtest(args: argparse.Namespace) -> None:
    emit_json_schemas()
    cfg = _load_config(args.config)
    ff = json.loads(Path(args.features).read_text(encoding="utf-8"))

    candidate_result = run_walkforward(ff, cfg, model_name=args.model)
    payload: dict[str, Any] = {"candidate_model": args.model, "candidate": candidate_result}

    if args.run_baselines:
        baseline_models = _parse_baselines(args.baseline_models)
        baselines: dict[str, Any] = {}
        promotion: dict[str, Any] = {}
        for baseline_model in baseline_models:
            baseline_result = run_walkforward(ff, cfg, model_name=baseline_model)
            baselines[baseline_model] = baseline_result
            promoted, reasons = promotion_gate(candidate_result, baseline_result, cfg)
            promotion[baseline_model] = {"promoted": promoted, "reasons": reasons}
        payload["baselines"] = baselines
        payload["promotion_vs_baseline"] = promotion
        Path(args.comparison_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


def cmd_promote(args: argparse.Namespace) -> None:
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    cfg = _load_config(args.config)
    ok, reasons = promotion_gate(candidate, baseline, cfg)
    print(json.dumps({"promoted": ok, "reasons": reasons}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m engine.cli")
    sub = p.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--config", required=True)
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--output", default="market_data.json")
    ingest.add_argument("--quarantine", default="quarantine.json")
    ingest.set_defaults(func=cmd_ingest)

    features = sub.add_parser("features")
    features.add_argument("--config", required=True)
    features.add_argument("--market-data", required=True)
    features.add_argument("--output", default="feature_frame.json")
    features.set_defaults(func=cmd_features)

    train = sub.add_parser("train")
    train.set_defaults(func=cmd_train)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--config", required=True)
    backtest.add_argument("--features", required=True)
    backtest.add_argument("--walkforward", action="store_true")
    backtest.add_argument("--model", default="linear_ridge")
    backtest.add_argument("--run-baselines", action="store_true")
    backtest.add_argument("--baseline-models", default=",".join(DEFAULT_BASELINES))
    backtest.add_argument("--comparison-output", default="comparison_results.json")
    backtest.set_defaults(func=cmd_backtest)

    promote = sub.add_parser("promote")
    promote.add_argument("--config", required=True)
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--baseline", required=True)
    promote.set_defaults(func=cmd_promote)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
