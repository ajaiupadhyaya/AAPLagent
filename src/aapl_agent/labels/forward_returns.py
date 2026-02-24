from __future__ import annotations

import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def make_multi_horizon_forward_returns(
    df: pd.DataFrame,
    horizons: list[int],
    price_col: str = "Close",
    prefix: str = "fwd_ret",
) -> pd.DataFrame:
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    out = pd.DataFrame(index=df.index)
    for h in horizons:
        out[f"{prefix}_{h}"] = forward_return(df[price_col], h)
    return out
