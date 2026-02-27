from __future__ import annotations

import pandas as pd


def detect_regimes(df: pd.DataFrame, vol_window: int = 20, vol_q: float = 0.7) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ema = df["close"].ewm(span=20, adjust=False).mean()
    out["trend"] = ema.diff().fillna(0.0)
    out["trend_sign"] = out["trend"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    vol = df["close"].pct_change().rolling(vol_window).std(ddof=0).fillna(0.0)
    threshold = vol.rolling(vol_window).quantile(vol_q).fillna(vol.median())
    out["vol_state"] = (vol > threshold).astype(int)

    def label(row: pd.Series) -> str:
        trend_label = (
            "Bull" if row["trend_sign"] > 0 else "Bear" if row["trend_sign"] < 0 else "Sideways"
        )
        vol_label = "HighVol" if row["vol_state"] == 1 else "LowVol"
        return f"{trend_label}_{vol_label}"

    out["regime_label"] = out.apply(label, axis=1)
    for col in [
        "Bull_HighVol",
        "Bull_LowVol",
        "Bear_HighVol",
        "Bear_LowVol",
        "Sideways_HighVol",
        "Sideways_LowVol",
    ]:
        out[f"prob_{col}"] = (out["regime_label"] == col).astype(float)
    return out
