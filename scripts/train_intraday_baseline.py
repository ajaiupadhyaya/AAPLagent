from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from aapl_agent.features import build_technical_features


def load_intraday(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


def merge_context(aapl: pd.DataFrame, spy: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    merged = aapl.merge(
        spy[["Datetime", "Close"]].rename(columns={"Close": "SPY_Close"}),
        on="Datetime",
        how="left",
    ).merge(
        qqq[["Datetime", "Close"]].rename(columns={"Close": "QQQ_Close"}),
        on="Datetime",
        how="left",
    )
    merged[["SPY_Close", "QQQ_Close"]] = merged[["SPY_Close", "QQQ_Close"]].ffill()
    return merged


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    data = build_technical_features(
        df.rename(columns={"Datetime": "Date"}).assign(Date=df["Datetime"])
    ).rename(columns={"Date": "Datetime"})
    data["aapl_vs_spy_6bar"] = data["Close"].pct_change(6) - data["SPY_Close"].pct_change(6)
    data["aapl_vs_qqq_6bar"] = data["Close"].pct_change(6) - data["QQQ_Close"].pct_change(6)
    data["minute_of_day"] = data["Datetime"].dt.hour * 60 + data["Datetime"].dt.minute
    data["session_progress"] = data["minute_of_day"] / 390.0
    return data


def make_long_short_label(df: pd.DataFrame, horizon_bars: int = 12, threshold: float = 0.003) -> pd.Series:
    fwd = df["Close"].shift(-horizon_bars) / df["Close"] - 1.0
    label = np.where(fwd > threshold, 2, np.where(fwd < -threshold, 0, 1))
    return pd.Series(label, index=df.index)


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main() -> None:
    aapl_path = Path("data/AAPL_intraday_5m.csv")
    spy_path = Path("data/SPY_intraday_5m.csv")
    qqq_path = Path("data/QQQ_intraday_5m.csv")
    for path in [aapl_path, spy_path, qqq_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run scripts/fetch_market_data.py first.")

    aapl = load_intraday(aapl_path)
    spy = load_intraday(spy_path)
    qqq = load_intraday(qqq_path)

    df = merge_context(aapl, spy, qqq)
    df = add_intraday_features(df)
    df["label"] = make_long_short_label(df, horizon_bars=12, threshold=0.003)
    df = df.dropna().reset_index(drop=True)

    features = [
        "returns_1d",
        "returns_5d",
        "returns_21d",
        "volatility_21d",
        "volume_zscore_21d",
        "rsi_14",
        "macd",
        "bb_width",
        "atr_14",
        "aapl_vs_spy_6bar",
        "aapl_vs_qqq_6bar",
        "minute_of_day",
        "session_progress",
    ]

    train_df, test_df = time_split(df)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=450,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(train_df[features], train_df["label"])

    probs = model.predict_proba(test_df[features])
    preds = probs.argmax(axis=1)
    auc = roc_auc_score(test_df["label"], probs, multi_class="ovr")
    print(f"Test OVR ROC-AUC: {auc:.4f}")
    print(classification_report(test_df["label"], preds, digits=4))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / "aapl_intraday_xgb.joblib"
    metadata_path = models_dir / "aapl_intraday_xgb_features.txt"
    joblib.dump(model, artifact_path)
    metadata_path.write_text("\n".join(features))
    print(f"Saved model artifact -> {artifact_path}")
    print(f"Saved feature list -> {metadata_path}")


if __name__ == "__main__":
    main()
