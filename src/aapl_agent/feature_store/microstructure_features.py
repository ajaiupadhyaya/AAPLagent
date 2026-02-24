from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register_feature


def _timestamp(df: pd.DataFrame) -> pd.Series:
    if "Datetime" in df.columns:
        return pd.to_datetime(df["Datetime"], errors="coerce")
    if "Date" in df.columns:
        return pd.to_datetime(df["Date"], errors="coerce")
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    raise ValueError("Microstructure features require Datetime or Date column (or DatetimeIndex).")


@register_feature("micro.intraday_seasonality", description="Sin/cos intraday seasonality encoding")
def intraday_seasonality(df: pd.DataFrame, **_: object) -> pd.DataFrame:
    ts = _timestamp(df)
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    session_progress = (minute_of_day - 570) / 390.0

    return pd.DataFrame(
        {
            "minute_of_day": minute_of_day,
            "session_progress": session_progress,
            "minute_sin": np.sin(2 * np.pi * session_progress),
            "minute_cos": np.cos(2 * np.pi * session_progress),
            "time_since_open_min": minute_of_day - 570,
        },
        index=df.index,
    )


@register_feature("micro.opening_range_breakout", description="Opening range breakout signals")
def opening_range_breakout(df: pd.DataFrame, **_: object) -> pd.DataFrame:
    required = {"High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for opening range breakout: {sorted(missing)}")

    ts = _timestamp(df)
    day = ts.dt.floor("D")
    minute = ts.dt.hour * 60 + ts.dt.minute
    in_opening_range = minute.between(570, 600)

    opening_high = df["High"].where(in_opening_range).groupby(day).transform("max")
    opening_low = df["Low"].where(in_opening_range).groupby(day).transform("min")

    breakout_up = (df["Close"] > opening_high).astype(float)
    breakout_down = (df["Close"] < opening_low).astype(float)

    return pd.DataFrame(
        {
            "opening_range_high": opening_high,
            "opening_range_low": opening_low,
            "opening_breakout_up": breakout_up,
            "opening_breakout_down": breakout_down,
        },
        index=df.index,
    )


@register_feature("micro.volume_imbalance_proxy", description="Volume imbalance proxy")
def volume_imbalance_proxy(df: pd.DataFrame, **_: object) -> pd.Series:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for volume imbalance: {sorted(missing)}")

    range_width = (df["High"] - df["Low"]).replace(0, np.nan)
    imbalance = ((df["Close"] - df["Open"]) / range_width) * df["Volume"]
    scaled = imbalance / df["Volume"].rolling(20).mean()
    return scaled.rename("volume_imbalance_proxy")


@register_feature("micro.intraday_volatility_buckets", description="Intraday volatility percentile buckets")
def intraday_volatility_buckets(df: pd.DataFrame, **_: object) -> pd.Series:
    if "Close" not in df.columns:
        raise ValueError("Column 'Close' is required for intraday volatility buckets.")

    rv = df["Close"].pct_change().rolling(20).std()
    pct = rv.rolling(252).rank(pct=True)
    bucket = np.select(
        [pct < 0.33, pct < 0.66, pct >= 0.66],
        [0, 1, 2],
        default=np.nan,
    )
    return pd.Series(bucket, index=df.index, name="intraday_vol_bucket")


@register_feature("micro.lunch_hour_dummy", description="Lunch-hour microstructure dummy")
def lunch_hour_dummy(df: pd.DataFrame, **_: object) -> pd.Series:
    ts = _timestamp(df)
    hour = ts.dt.hour
    minute = ts.dt.minute
    is_lunch = ((hour == 12) | ((hour == 13) & (minute == 0))).astype(float)
    return pd.Series(is_lunch, index=df.index, name="lunch_hour_dummy")
