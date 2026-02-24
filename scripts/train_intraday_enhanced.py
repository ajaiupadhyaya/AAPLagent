from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

from aapl_agent.features import build_technical_features


def load_intraday(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])
    return df.sort_values("Datetime").reset_index(drop=True)


def merge_context(aapl: pd.DataFrame, spy: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
    data = aapl.merge(
        spy[["Datetime", "Close"]].rename(columns={"Close": "SPY_Close"}),
        on="Datetime",
        how="left",
    ).merge(
        qqq[["Datetime", "Close"]].rename(columns={"Close": "QQQ_Close"}),
        on="Datetime",
        how="left",
    )
    data[["SPY_Close", "QQQ_Close"]] = data[["SPY_Close", "QQQ_Close"]].ffill()
    return data


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    data = build_technical_features(
        df.rename(columns={"Datetime": "Date"}).assign(Date=df["Datetime"])
    ).rename(columns={"Date": "Datetime"})

    data["aapl_vs_spy_6bar"] = data["Close"].pct_change(6) - data["SPY_Close"].pct_change(6)
    data["aapl_vs_qqq_6bar"] = data["Close"].pct_change(6) - data["QQQ_Close"].pct_change(6)
    data["aapl_vs_spy_24bar"] = data["Close"].pct_change(24) - data["SPY_Close"].pct_change(24)
    data["aapl_vs_qqq_24bar"] = data["Close"].pct_change(24) - data["QQQ_Close"].pct_change(24)

    data["minute_of_day"] = data["Datetime"].dt.hour * 60 + data["Datetime"].dt.minute
    data["session_progress"] = data["minute_of_day"] / 390.0
    data["minute_sin"] = np.sin(2 * np.pi * data["session_progress"])
    data["minute_cos"] = np.cos(2 * np.pi * data["session_progress"])

    data["overnight_gap"] = data["Open"] / data["Close"].shift(1) - 1.0
    data["intrabar_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["volatility_10bar"] = data["returns_1d"].rolling(10).std()

    return data


def make_long_short_label(df: pd.DataFrame, horizon_bars: int = 12, threshold: float = 0.0025) -> pd.Series:
    forward_return = df["Close"].shift(-horizon_bars) / df["Close"] - 1.0
    label = np.where(forward_return > threshold, 2, np.where(forward_return < -threshold, 0, 1))
    return pd.Series(label, index=df.index)


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_ratio)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def class_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    total = len(y)
    num_classes = len(counts)
    weights = {cls: total / (num_classes * count) for cls, count in counts.items()}
    return y.map(weights).to_numpy()


def apply_confidence_policy(probabilities: np.ndarray, confidence_threshold: float) -> np.ndarray:
    hard_pred = probabilities.argmax(axis=1)
    max_conf = probabilities.max(axis=1)
    adjusted = np.where(max_conf >= confidence_threshold, hard_pred, 1)
    return adjusted.astype(int)


def evaluate_fold(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
    params: dict,
) -> tuple[float, float]:
    model = XGBClassifier(**params)
    sample_weight = class_weights(train_y)
    model.fit(train_x, train_y, sample_weight=sample_weight)
    valid_prob = model.predict_proba(valid_x)
    valid_pred = valid_prob.argmax(axis=1)
    auc = roc_auc_score(valid_y, valid_prob, multi_class="ovr")
    bacc = balanced_accuracy_score(valid_y, valid_pred)
    return auc, bacc


def walk_forward_select(train_df: pd.DataFrame, features: list[str], target: str) -> dict:
    candidate_params = [
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "mlogloss",
            "random_state": 42,
        },
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 700,
            "max_depth": 3,
            "learning_rate": 0.02,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "eval_metric": "mlogloss",
            "random_state": 42,
        },
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 450,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "eval_metric": "mlogloss",
            "random_state": 42,
        },
    ]

    n = len(train_df)
    fold_boundaries = [int(n * 0.5), int(n * 0.65), int(n * 0.8)]
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for boundary in fold_boundaries:
        sub_train = train_df.iloc[:boundary]
        sub_valid = train_df.iloc[boundary : min(n, boundary + int(n * 0.1))]
        if len(sub_valid) > 100:
            folds.append((sub_train, sub_valid))

    best_score = -np.inf
    best_params = candidate_params[0]
    for params in candidate_params:
        fold_scores = []
        for sub_train, sub_valid in folds:
            auc, bacc = evaluate_fold(
                sub_train[features],
                sub_train[target],
                sub_valid[features],
                sub_valid[target],
                params,
            )
            fold_scores.append(0.7 * auc + 0.3 * bacc)
        score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        if score > best_score:
            best_score = score
            best_params = params
    return best_params


def tune_confidence_threshold(model: XGBClassifier, train_df: pd.DataFrame, features: list[str]) -> float:
    split = int(len(train_df) * 0.8)
    tuning_set = train_df.iloc[split:].copy()
    y_true = tuning_set["label"].to_numpy()
    probabilities = model.predict_proba(tuning_set[features])

    thresholds = np.arange(0.45, 0.71, 0.02)
    best_threshold = 0.55
    best_score = -np.inf
    for threshold in thresholds:
        pred = apply_confidence_policy(probabilities, float(threshold))
        bacc = balanced_accuracy_score(y_true, pred)
        activity = float(np.mean(pred != 1))
        score = bacc + 0.10 * activity
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


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

    data = merge_context(aapl, spy, qqq)
    data = add_intraday_features(data)
    data["label"] = make_long_short_label(data, horizon_bars=12, threshold=0.0025)
    data = data.dropna().reset_index(drop=True)

    features = [
        "returns_1d",
        "returns_5d",
        "returns_21d",
        "volatility_21d",
        "volatility_10bar",
        "volume_zscore_21d",
        "rsi_14",
        "macd",
        "bb_width",
        "atr_14",
        "aapl_vs_spy_6bar",
        "aapl_vs_qqq_6bar",
        "aapl_vs_spy_24bar",
        "aapl_vs_qqq_24bar",
        "minute_of_day",
        "session_progress",
        "minute_sin",
        "minute_cos",
        "overnight_gap",
        "intrabar_range",
    ]

    train_df, test_df = time_split(data)

    best_params = walk_forward_select(train_df, features, "label")
    model = XGBClassifier(**best_params)
    sample_weight = class_weights(train_df["label"])
    model.fit(train_df[features], train_df["label"], sample_weight=sample_weight)

    confidence_threshold = tune_confidence_threshold(model, train_df, features)

    test_prob = model.predict_proba(test_df[features])
    raw_pred = test_prob.argmax(axis=1)
    policy_pred = apply_confidence_policy(test_prob, confidence_threshold)

    auc = roc_auc_score(test_df["label"], test_prob, multi_class="ovr")
    bacc_raw = balanced_accuracy_score(test_df["label"], raw_pred)
    bacc_policy = balanced_accuracy_score(test_df["label"], policy_pred)
    print(f"Enhanced Test OVR ROC-AUC: {auc:.4f}")
    print(f"Enhanced Balanced Accuracy (raw): {bacc_raw:.4f}")
    print(f"Enhanced Balanced Accuracy (policy): {bacc_policy:.4f}")
    print(classification_report(test_df["label"], policy_pred, digits=4))

    models_dir = Path("models")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "aapl_intraday_xgb_enhanced.joblib")
    (models_dir / "aapl_intraday_xgb_enhanced_features.txt").write_text("\n".join(features))

    summary = {
        "model": "aapl_intraday_xgb_enhanced",
        "selected_params": best_params,
        "confidence_threshold": confidence_threshold,
        "test_metrics": {
            "ovr_roc_auc": float(auc),
            "balanced_accuracy_raw": float(bacc_raw),
            "balanced_accuracy_policy": float(bacc_policy),
        },
    }
    (reports_dir / "enhanced_training_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved enhanced artifacts and training summary.")


if __name__ == "__main__":
    main()
