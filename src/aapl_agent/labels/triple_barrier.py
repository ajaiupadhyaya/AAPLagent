from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    *,
    horizon: int,
    up_mult: float,
    down_mult: float,
    vol_window: int = 20,
    price_col: str = "Close",
) -> pd.Series:
    """Generate +1/-1/0 labels using a simple triple-barrier implementation."""

    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    close = df[price_col]
    vol = close.pct_change().rolling(vol_window).std().clip(lower=1e-8)

    labels = np.zeros(len(df), dtype=np.int8)
    for i in range(len(df)):
        if i + horizon >= len(df) or pd.isna(vol.iloc[i]):
            labels[i] = 0
            continue

        entry = close.iloc[i]
        upper = entry * (1.0 + up_mult * vol.iloc[i])
        lower = entry * (1.0 - down_mult * vol.iloc[i])

        path = close.iloc[i + 1 : i + horizon + 1]
        hit_up = (path >= upper).any()
        hit_down = (path <= lower).any()

        if hit_up and not hit_down:
            labels[i] = 1
        elif hit_down and not hit_up:
            labels[i] = -1
        elif hit_up and hit_down:
            first_up = int(np.argmax((path >= upper).to_numpy()))
            first_down = int(np.argmax((path <= lower).to_numpy()))
            labels[i] = 1 if first_up <= first_down else -1
        else:
            labels[i] = 0

    return pd.Series(labels, index=df.index, name=f"tb_label_h{horizon}")
