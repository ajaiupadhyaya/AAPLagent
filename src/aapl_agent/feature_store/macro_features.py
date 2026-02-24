from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register_feature


def _aligned_close(market_data: dict[str, pd.DataFrame], symbol: str, index: pd.Index) -> pd.Series:
    frame = market_data.get(symbol)
    if frame is None or "Close" not in frame.columns:
        return pd.Series(np.nan, index=index)
    return frame["Close"].reindex(index).ffill()


@register_feature("macro.rolling_beta_spy", description="Rolling beta of AAPL versus SPY")
def rolling_beta_spy(df: pd.DataFrame, market_data: dict[str, pd.DataFrame], **_: object) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Column 'Close' is required for rolling beta.")

    aapl_ret = df["Close"].pct_change()
    spy_ret = _aligned_close(market_data, "SPY", df.index).pct_change()
    cov = aapl_ret.rolling(60).cov(spy_ret)
    var = spy_ret.rolling(60).var()
    beta = cov / var.replace(0, np.nan)
    return beta.rename("beta_spy_60")


@register_feature("macro.relative_strength_qqq", description="AAPL relative strength vs QQQ")
def relative_strength_qqq(df: pd.DataFrame, market_data: dict[str, pd.DataFrame], **_: object) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Column 'Close' is required for relative strength.")

    aapl_ret = df["Close"].pct_change(20)
    qqq_ret = _aligned_close(market_data, "QQQ", df.index).pct_change(20)
    return (aapl_ret - qqq_ret).rename("relative_strength_qqq_20")


@register_feature("macro.rate_sensitivity_tnx", description="Rolling correlation to TNX changes")
def rate_sensitivity_tnx(df: pd.DataFrame, market_data: dict[str, pd.DataFrame], **_: object) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Column 'Close' is required for rate sensitivity.")

    aapl_ret = df["Close"].pct_change()
    tnx_change = _aligned_close(market_data, "TNX", df.index).diff()
    sensitivity = aapl_ret.rolling(60).corr(tnx_change)
    return sensitivity.rename("rate_sensitivity_tnx_60")


@register_feature("macro.correlation_regime", description="Rolling AAPL-SPY/QQQ correlation regime")
def correlation_regime(df: pd.DataFrame, market_data: dict[str, pd.DataFrame], **_: object) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("Column 'Close' is required for correlation regime features.")

    aapl_ret = df["Close"].pct_change()
    spy_ret = _aligned_close(market_data, "SPY", df.index).pct_change()
    qqq_ret = _aligned_close(market_data, "QQQ", df.index).pct_change()

    return pd.DataFrame(
        {
            "corr_spy_60": aapl_ret.rolling(60).corr(spy_ret),
            "corr_qqq_60": aapl_ret.rolling(60).corr(qqq_ret),
        },
        index=df.index,
    )


@register_feature("macro.cross_asset_momentum", description="Cross-asset momentum blend")
def cross_asset_momentum(df: pd.DataFrame, market_data: dict[str, pd.DataFrame], **_: object) -> pd.Series:
    spy_mom = _aligned_close(market_data, "SPY", df.index).pct_change(20)
    qqq_mom = _aligned_close(market_data, "QQQ", df.index).pct_change(20)
    vix_mom = _aligned_close(market_data, "VIX", df.index).pct_change(20)
    tnx_mom = _aligned_close(market_data, "TNX", df.index).pct_change(20)

    combo = (spy_mom + qqq_mom - vix_mom - tnx_mom) / 4.0
    return combo.rename("cross_asset_momentum_20")
