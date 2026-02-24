from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["AAPL", "SPY", "QQQ", "MSFT", "NVDA", "^VIX", "^TNX"]
    for symbol in symbols:
        df = yf.download(symbol, period="12y", interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        out_path = output_dir / f"{symbol.replace('^', '')}_daily.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {symbol} -> {out_path}")

    intraday_symbols = ["AAPL", "SPY", "QQQ"]
    for symbol in intraday_symbols:
        df = yf.download(symbol, period="60d", interval="5m", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        out_path = output_dir / f"{symbol}_intraday_5m.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved {symbol} intraday -> {out_path}")


if __name__ == "__main__":
    main()
