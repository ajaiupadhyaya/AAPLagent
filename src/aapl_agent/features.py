from __future__ import annotations

import numpy as np
import pandas as pd
import ta


def load_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["returns_1d"] = data["Close"].pct_change(1)
    data["returns_5d"] = data["Close"].pct_change(5)
    data["returns_21d"] = data["Close"].pct_change(21)

    data["volatility_21d"] = data["returns_1d"].rolling(21).std() * np.sqrt(252)
    data["volume_zscore_21d"] = (
        (data["Volume"] - data["Volume"].rolling(21).mean()) / data["Volume"].rolling(21).std()
    )

    data["rsi_14"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()

    macd = ta.trend.MACD(close=data["Close"], window_slow=26, window_fast=12, window_sign=9)
    data["macd"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
    data["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / data["Close"]

    atr = ta.volatility.AverageTrueRange(
        high=data["High"], low=data["Low"], close=data["Close"], window=14
    )
    data["atr_14"] = atr.average_true_range()

    return data


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["day_of_week"] = data["Date"].dt.dayofweek
    data["month"] = data["Date"].dt.month
    return data
