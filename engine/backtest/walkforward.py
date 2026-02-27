from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from engine.backtest.metrics import mae, max_drawdown, rmse, sharpe, sortino
from engine.data.schema import (
    validate_backtest_result,
    validate_decision,
    validate_feature_frame,
    validate_forecast,
)
from engine.features.feature_sets import to_xy
from engine.models.base import calibration_score
from engine.portfolio.optimizer import make_decision
from engine.registry.artifacts import write_run_artifacts


def _instantiate(name: str, seed: int) -> Any:
    if name == "baseline_zero":
        from engine.models.baselines import ZeroReturnModel

        return ZeroReturnModel()
    if name == "baseline_rw":
        from engine.models.baselines import RandomWalkReturnModel

        return RandomWalkReturnModel()
    if name == "baseline_roll":
        from engine.models.baselines import RollingMeanModel

        return RollingMeanModel(window=20)
    if name == "linear_elastic":
        from engine.models.linear import LinearModel

        return LinearModel(model_type="elastic_net", random_state=seed)
    if name == "linear_ridge":
        from engine.models.linear import LinearModel

        return LinearModel(model_type="ridge", random_state=seed)
    from engine.models.tree import RandomForestModel

    return RandomForestModel(random_state=seed)


def _drawdown_from_curve(curve: list[float]) -> float:
    if not curve:
        return 0.0
    series = pd.Series(curve)
    peak = series.cummax()
    return float((series.iloc[-1] / peak.iloc[-1]) - 1.0)


def run_walkforward(
    feature_frame: dict[str, Any],
    config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    ff = validate_feature_frame(feature_frame)
    x, y, ts = to_xy(ff.model_dump())
    train = int(config["backtest"]["train_window"])
    step = int(config["backtest"]["step"])
    risk_cfg = config["risk"]

    forecasts: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    gross_pnl: list[float] = []
    net_pnl: list[float] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    sigma_pred: list[float] = []
    current_position = 0.0
    gross_equity = 1.0
    net_equity = 1.0
    gross_curve: list[dict[str, float | str]] = []
    net_curve: list[dict[str, float | str]] = []
    drawdown_series: list[dict[str, float | str]] = []
    net_curve_values: list[float] = []
    constraint_hit_counts: dict[str, int] = {}

    for i in range(train, len(x), step):
        x_train, y_train = x.iloc[i - train : i], y.iloc[i - train : i]
        x_test, y_test = x.iloc[i : i + step], y.iloc[i : i + step]
        ts_test = ts[i : i + step]
        if len(x_test) == 0:
            continue

        model = _instantiate(model_name, seed=int(config["seed"]))
        model.fit(x_train, y_train)
        batch_fc = model.predict(x_test, ts_test, horizon=int(ff.horizon))
        for fc, yv in zip(batch_fc, y_test.values):
            fc["provenance"].update(
                {
                    "feature_set_version": ff.feature_set_version,
                    "seed": config["seed"],
                    "model_name": model_name,
                    "training_window": train,
                }
            )
            validate_forecast(fc)

            current_drawdown = _drawdown_from_curve(net_curve_values)
            decision = make_decision(
                fc,
                current_position,
                drawdown=current_drawdown,
                risk_budget=max(0.0, 1.0 + current_drawdown),
                config=risk_cfg,
            )
            validate_decision(decision)
            for hit in decision["constraints_hit"]:
                constraint_hit_counts[hit] = constraint_hit_counts.get(hit, 0) + 1

            forecasts.append(fc)
            decisions.append(decision)
            ret = float(yv)
            target_position = float(decision["target_position"])
            gross_realized = target_position * ret
            net_realized = gross_realized - float(decision["transaction_cost"])
            gross_pnl.append(gross_realized)
            net_pnl.append(net_realized)
            y_true.append(ret)
            y_pred.append(float(fc["mean_return"]))
            sigma_pred.append(float(fc["stdev"]))
            current_position = target_position

            gross_equity *= 1 + gross_realized
            net_equity *= 1 + net_realized
            net_curve_values.append(net_equity)
            gross_curve.append({"timestamp": str(fc["timestamp"]), "equity": gross_equity})
            net_curve.append({"timestamp": str(fc["timestamp"]), "equity": net_equity})
            drawdown_series.append(
                {
                    "timestamp": str(fc["timestamp"]),
                    "drawdown": _drawdown_from_curve(net_curve_values),
                }
            )

    gross_pnl_series = pd.Series(gross_pnl)
    net_pnl_series = pd.Series(net_pnl)
    positions = [0.0] + [float(d["target_position"]) for d in decisions]
    net_equity_series = pd.Series([float(x["equity"]) for x in net_curve])
    metrics = {
        "mae": mae(np.array(y_true), np.array(y_pred)),
        "rmse": rmse(np.array(y_true), np.array(y_pred)),
        "gross_sharpe": sharpe(gross_pnl_series),
        "net_sharpe": sharpe(net_pnl_series),
        "net_sortino": sortino(net_pnl_series),
        "mdd": max_drawdown(net_equity_series) if not net_equity_series.empty else 0.0,
        "turnover": float(np.mean(np.abs(np.diff(positions)))) if len(positions) > 1 else 0.0,
        "gross_return": float(gross_pnl_series.sum()),
        "net_return": float(net_pnl_series.sum()),
        "calibration_score": calibration_score(
            np.array(y_true), np.array(y_pred), np.array(sigma_pred)
        ),
    }

    summary = (
        f"# Backtest Summary\n\n"
        f"- Model: {model_name}\n"
        f"- Gross Return: {metrics['gross_return']:.6f}\n"
        f"- Net Return: {metrics['net_return']:.6f}\n"
        f"- Net Sharpe: {metrics['net_sharpe']:.4f}\n"
        f"- Calibration: {metrics['calibration_score']:.4f}\n"
    )
    artifact_paths = write_run_artifacts(config, metrics, forecasts, decisions, summary)

    result = {
        "metrics": metrics,
        "gross_curve": gross_curve,
        "net_curve": net_curve,
        "drawdown_series": drawdown_series,
        "constraint_hit_counts": constraint_hit_counts,
        "logs": ["walkforward_complete"],
        "artifact_paths": {k: v for k, v in artifact_paths.items() if k != "hashes"},
        "hashes": json.loads(artifact_paths["hashes"]),
    }
    validate_backtest_result(result)
    return result


def promotion_gate(
    candidate: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    margin = float(config["promotion"]["min_improvement"])
    if float(candidate["metrics"]["net_return"]) < float(baseline["metrics"]["net_return"]) * (
        1 + margin
    ):
        reasons.append("net_of_cost_underperformance")
    if float(candidate["metrics"]["calibration_score"]) < float(
        config["promotion"]["min_calibration"]
    ):
        reasons.append("calibration_below_threshold")
    if float(candidate["metrics"]["mdd"]) < -abs(float(config["promotion"]["max_drawdown"])):
        reasons.append("drawdown_too_large")
    return len(reasons) == 0, reasons
