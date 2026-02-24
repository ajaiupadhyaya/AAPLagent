from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register_feature


def _extract_close(market_data: dict[str, pd.DataFrame], symbol: str) -> pd.Series | None:
    if symbol not in market_data:
        return None
    frame = market_data[symbol]
    if "Close" not in frame.columns:
        return None
    return frame["Close"]


@register_feature("vol.vix_term_structure_proxy", description="VIX short-long term proxy")
def vix_term_structure_proxy(df: pd.DataFrame, market_data: dict[str, pd.DataFrame], **_: object) -> pd.Series:
    vix = _extract_close(market_data, "VIX")
    if vix is None:
        return pd.Series(np.nan, index=df.index, name="vix_term_proxy")

    aligned = vix.reindex(df.index).ffill()
    short = aligned.rolling(10).mean()
    long = aligned.rolling(30).mean()
    return (short - long).rename("vix_term_proxy")


@register_feature("vol.realized_implied_spread", description="Spread between realized and implied volatility")
def realized_implied_spread(
    df: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
    **_: object,
) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Column 'Close' is required for realized-implied spread.")

    vix = _extract_close(market_data, "VIX")
    if vix is None:
        return pd.Series(np.nan, index=df.index, name="realized_implied_spread_20")

    realized = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    implied = (vix.reindex(df.index).ffill() / 100.0)
    return (realized - implied).rename("realized_implied_spread_20")


@register_feature("vol.vol_of_vol", description="Volatility of implied volatility")
def vol_of_vol(df: pd.DataFrame, market_data: dict[str, pd.DataFrame], **_: object) -> pd.Series:
    vix = _extract_close(market_data, "VIX")
    if vix is None:
        return pd.Series(np.nan, index=df.index, name="vol_of_vol_20")

    vix_ret = vix.reindex(df.index).ffill().pct_change()
    return (vix_ret.rolling(20).std() * np.sqrt(252)).rename("vol_of_vol_20")


@register_feature("vol.regime_vol_percentile", description="Percentile of current realized vol within history")
def regime_vol_percentile(df: pd.DataFrame, **_: object) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Column 'Close' is required for regime volatility percentile.")

    realized = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    percentile = realized.rolling(252).rank(pct=True)
    return percentile.rename("regime_vol_percentile_252")
