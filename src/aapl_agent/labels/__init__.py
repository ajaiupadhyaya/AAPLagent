from __future__ import annotations

import pandas as pd

from .forward_returns import make_multi_horizon_forward_returns
from .meta_labels import meta_label_from_events
from .triple_barrier import triple_barrier_labels


def make_forward_return(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Missing 'Close' column for forward return labeling.")
    return df["Close"].shift(-horizon_days) / df["Close"] - 1.0


def make_binary_label(df: pd.DataFrame, horizon_days: int, threshold: float = 0.02) -> pd.Series:
    fwd = make_forward_return(df, horizon_days=horizon_days)
    return (fwd > threshold).astype("int8")


__all__ = [
    "make_forward_return",
    "make_binary_label",
    "make_multi_horizon_forward_returns",
    "triple_barrier_labels",
    "meta_label_from_events",
]
