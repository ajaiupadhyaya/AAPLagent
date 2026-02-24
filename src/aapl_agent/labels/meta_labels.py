from __future__ import annotations

import pandas as pd


def meta_label_from_events(
    df: pd.DataFrame,
    event_col: str,
    future_return_col: str,
    threshold: float = 0.0,
) -> pd.Series:
    """Binary meta-label: 1 when event direction was correct beyond threshold."""

    if event_col not in df.columns:
        raise ValueError(f"Missing event column: {event_col}")
    if future_return_col not in df.columns:
        raise ValueError(f"Missing future return column: {future_return_col}")

    signal = df[event_col]
    future_ret = df[future_return_col]

    success = (signal * future_ret) > threshold
    return success.astype("int8").rename("meta_label")
