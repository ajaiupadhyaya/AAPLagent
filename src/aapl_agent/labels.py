from __future__ import annotations

import pandas as pd


def make_forward_return(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    return df["Close"].shift(-horizon_days) / df["Close"] - 1.0


def make_binary_label(df: pd.DataFrame, horizon_days: int, threshold: float = 0.02) -> pd.Series:
    fwd = make_forward_return(df, horizon_days=horizon_days)
    return (fwd > threshold).astype("int8")
