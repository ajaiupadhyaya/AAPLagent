from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register_feature


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature computation: {sorted(missing)}")


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    x_var = np.sum(x_centered**2)

    def _slope(window_values: np.ndarray) -> float:
        y = np.asarray(window_values, dtype=float)
        y_centered = y - y.mean()
        return float(np.sum(x_centered * y_centered) / x_var)

    return values.rolling(window).apply(_slope, raw=True)


@register_feature("price.multi_horizon_returns", description="Multi-horizon close-to-close returns")
def multi_horizon_returns(df: pd.DataFrame, **_: object) -> pd.DataFrame:
    _require_columns(df, ["Close"])
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    for horizon in [1, 5, 10, 20, 60]:
        out[f"ret_{horizon}"] = close.pct_change(horizon)
    return out


@register_feature("price.rolling_vwap_distance", description="Distance of close from rolling VWAP")
def rolling_vwap_distance(df: pd.DataFrame, **_: object) -> pd.Series:
    _require_columns(df, ["Close", "Volume"])
    volume = df["Volume"].clip(lower=0)
    vwap = (df["Close"] * volume).rolling(20).sum() / volume.rolling(20).sum()
    return ((df["Close"] - vwap) / vwap).rename("vwap_distance_20")


@register_feature("price.realized_volatility", description="Parkinson and Garman-Klass realized volatility")
def realized_volatility(df: pd.DataFrame, **_: object) -> pd.DataFrame:
    _require_columns(df, ["Open", "High", "Low", "Close"])

    log_hl = pd.Series(np.log(df["High"] / df["Low"]), index=df.index).replace(
        [np.inf, -np.inf], np.nan
    )
    parkinson_var = (log_hl**2).rolling(20).mean() / (4.0 * np.log(2.0))
    parkinson = np.sqrt(252.0 * parkinson_var)

    log_co = pd.Series(np.log(df["Close"] / df["Open"]), index=df.index).replace(
        [np.inf, -np.inf], np.nan
    )
    gk_var = (0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)).rolling(20).mean()
    garman_klass = np.sqrt(252.0 * gk_var.clip(lower=0))

    return pd.DataFrame(
        {
            "realized_vol_parkinson_20": parkinson,
            "realized_vol_garman_klass_20": garman_klass,
        },
        index=df.index,
    )


@register_feature("price.atr_variants", description="ATR absolute and percentage variants")
def atr_variants(df: pd.DataFrame, **_: object) -> pd.DataFrame:
    _require_columns(df, ["High", "Low", "Close"])

    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_14 = true_range.rolling(14).mean()
    atr_28 = true_range.rolling(28).mean()

    return pd.DataFrame(
        {
            "atr_14": atr_14,
            "atr_28": atr_28,
            "atr_pct_14": atr_14 / df["Close"],
            "atr_pct_28": atr_28 / df["Close"],
        },
        index=df.index,
    )


@register_feature("price.trend_slope", description="Rolling trend slope via linear regression")
def trend_slope(df: pd.DataFrame, **_: object) -> pd.Series:
    _require_columns(df, ["Close"])
    slope = _rolling_slope(df["Close"], window=20)
    normalized = slope / df["Close"].rolling(20).mean()
    return normalized.rename("trend_slope_20")


@register_feature("price.volume_weighted_momentum", description="Volume-weighted momentum")
def volume_weighted_momentum(df: pd.DataFrame, **_: object) -> pd.Series:
    _require_columns(df, ["Close", "Volume"])
    returns_1 = df["Close"].pct_change(1)
    vol_weight = df["Volume"] / df["Volume"].rolling(20).mean()
    vwm = (returns_1 * vol_weight).rolling(10).sum()
    return vwm.rename("volume_weighted_momentum_10")
