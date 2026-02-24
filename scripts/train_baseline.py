from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from aapl_agent.features import add_calendar_features, build_technical_features, load_price_csv
from aapl_agent.labels import make_binary_label


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main() -> None:
    data_path = Path("data/AAPL_daily.csv")
    if not data_path.exists():
        raise FileNotFoundError("Expected data/AAPL_daily.csv. Run scripts/fetch_market_data.py first.")

    df = load_price_csv(str(data_path))
    df = build_technical_features(df)
    df = add_calendar_features(df)
    df["label"] = make_binary_label(df, horizon_days=5, threshold=0.02)
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
        "day_of_week",
        "month",
    ]

    train_df, test_df = time_split(df)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(train_df[features], train_df["label"])

    probs = model.predict_proba(test_df[features])[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(test_df["label"], probs)
    print(f"Test ROC-AUC: {auc:.4f}")
    print(classification_report(test_df["label"], preds, digits=4))


if __name__ == "__main__":
    main()
