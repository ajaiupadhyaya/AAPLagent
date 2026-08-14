from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = build_technical_features(
        df.rename(columns={"Datetime": "Date"}).assign(Date=df["Datetime"])
    ).rename(columns={"Date": "Datetime"})

    data["aapl_vs_spy_6bar"] = data["Close"].pct_change(6) - data["SPY_Close"].pct_change(6)
    data["aapl_vs_qqq_6bar"] = data["Close"].pct_change(6) - data["QQQ_Close"].pct_change(6)
    data["aapl_vs_spy_24bar"] = data["Close"].pct_change(24) - data["SPY_Close"].pct_change(24)
    data["aapl_vs_qqq_24bar"] = data["Close"].pct_change(24) - data["QQQ_Close"].pct_change(24)
    data["aapl_vs_spy_48bar"] = data["Close"].pct_change(48) - data["SPY_Close"].pct_change(48)
    data["aapl_vs_qqq_48bar"] = data["Close"].pct_change(48) - data["QQQ_Close"].pct_change(48)

    data["minute_of_day"] = data["Datetime"].dt.hour * 60 + data["Datetime"].dt.minute
    data["session_progress"] = data["minute_of_day"] / 390.0
    data["minute_sin"] = np.sin(2 * np.pi * data["session_progress"])
    data["minute_cos"] = np.cos(2 * np.pi * data["session_progress"])
    data["minute_sin_2"] = np.sin(4 * np.pi * data["session_progress"])
    data["minute_cos_2"] = np.cos(4 * np.pi * data["session_progress"])

    data["overnight_gap"] = data["Open"] / data["Close"].shift(1) - 1.0
    data["intrabar_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["body_size"] = (data["Close"] - data["Open"]).abs() / data["Close"]
    data["volatility_10bar"] = data["returns_1d"].rolling(10).std()
    data["volatility_30bar"] = data["returns_1d"].rolling(30).std()
    data["trend_strength"] = (data["Close"] / data["Close"].rolling(20).mean()) - 1.0
    data["volatility_regime_pct"] = data["volatility_30bar"].rolling(252).rank(pct=True)

    rolling_vwap = (data["Close"] * data["Volume"]).rolling(20).sum() / data["Volume"].rolling(20).sum()
    data["vwap_distance_20"] = (data["Close"] - rolling_vwap) / rolling_vwap

    return data


def make_label(df: pd.DataFrame, threshold: float = 0.002) -> pd.Series:
    forward_return = df["Close"].shift(-12) / df["Close"] - 1.0
    label = np.where(forward_return > threshold, 2, np.where(forward_return < -threshold, 0, 1))
    return pd.Series(label, index=df.index)


def split_train_test(df: pd.DataFrame, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = int(len(df) * ratio)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


def class_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    n = len(y)
    k = len(counts)
    mapping = {cls: n / (k * cnt) for cls, cnt in counts.items()}
    return y.map(mapping).to_numpy()


def policy_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    hard = probabilities.argmax(axis=1)
    conf = probabilities.max(axis=1)
    return np.where(conf >= threshold, hard, 1)


def strategy_stats(df: pd.DataFrame, pred: np.ndarray, slippage_bps: float = 1.5) -> dict[str, float]:
    pred_series = pd.Series(pred, index=df.index)
    position = pred_series.map({0: -1.0, 1: 0.0, 2: 1.0}).astype(float)
    next_ret = df["Close"].pct_change().shift(-1).fillna(0.0)
    turnover = position.diff().abs().fillna(0.0)
    costs = turnover * (slippage_bps / 10000)
    pnl = position * next_ret - costs
    cumulative = (1 + pnl).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1.0

    bar_vol = pnl.std(ddof=0)
    sharpe_like = 0.0 if bar_vol == 0 else (pnl.mean() / bar_vol) * np.sqrt(252 * 78)
    return {
        "return": float(cumulative.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe_like": float(sharpe_like),
        "turnover": float(turnover.mean()),
        "activity": float((pred_series != 1).mean()),
    }


def purged_walk_forward_score(
    df: pd.DataFrame,
    features: list[str],
    params: dict,
    purge_bars: int = 12,
    val_fraction: float = 0.1,
) -> float:
    n = len(df)
    boundaries = [int(n * 0.48), int(n * 0.60), int(n * 0.72)]
    fold_scores = []

    for boundary in boundaries:
        valid_start = boundary + purge_bars
        valid_end = min(n, valid_start + int(n * val_fraction))
        train_end = boundary - purge_bars
        if train_end < 400 or (valid_end - valid_start) < 160:
            continue

        fold_train = df.iloc[:train_end]
        fold_valid = df.iloc[valid_start:valid_end]

        model = XGBClassifier(**params)
        model.fit(
            fold_train[features],
            fold_train["label"],
            sample_weight=class_weights(fold_train["label"]),
        )
        valid_prob = model.predict_proba(fold_valid[features])

        candidate_thresholds = [0.34, 0.38, 0.42, 0.46, 0.50]
        best_fold = -np.inf
        for threshold in candidate_thresholds:
            valid_pred = policy_predictions(valid_prob, threshold=threshold)
            auc = roc_auc_score(fold_valid["label"], valid_prob, multi_class="ovr")
            bacc = balanced_accuracy_score(fold_valid["label"], valid_pred)
            strategy = strategy_stats(fold_valid, valid_pred)

            activity_penalty = abs(strategy["activity"] - 0.28)
            turnover_penalty = strategy["turnover"]
            drawdown_penalty = abs(strategy["max_drawdown"])

            fold_score = (
                0.40 * auc
                + 0.30 * bacc
                + 0.35 * strategy["sharpe_like"]
                - 0.60 * activity_penalty
                - 0.35 * turnover_penalty
                - 0.15 * drawdown_penalty
            )
            if fold_score > best_fold:
                best_fold = fold_score

        fold_scores.append(best_fold)

    return float(np.mean(fold_scores)) if fold_scores else -np.inf


def select_hyperparameters(train_df: pd.DataFrame, features: list[str]) -> tuple[dict, list[dict[str, float]]]:
    candidates = [
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 650,
            "max_depth": 3,
            "learning_rate": 0.015,
            "subsample": 0.88,
            "colsample_bytree": 0.88,
            "min_child_weight": 5,
            "reg_alpha": 0.5,
            "reg_lambda": 1.8,
            "gamma": 0.1,
            "eval_metric": "mlogloss",
            "random_state": 42,
        },
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 800,
            "max_depth": 4,
            "learning_rate": 0.012,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "min_child_weight": 6,
            "reg_alpha": 0.65,
            "reg_lambda": 2.0,
            "gamma": 0.15,
            "eval_metric": "mlogloss",
            "random_state": 42,
        },
        {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": 520,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.80,
            "colsample_bytree": 0.80,
            "min_child_weight": 7,
            "reg_alpha": 0.8,
            "reg_lambda": 2.4,
            "gamma": 0.2,
            "eval_metric": "mlogloss",
            "random_state": 42,
        },
    ]

    scored: list[dict[str, float]] = []
    best_params = candidates[0]
    best_score = -np.inf
    for idx, params in enumerate(candidates):
        score = purged_walk_forward_score(train_df, features, params)
        scored.append({"candidate_index": float(idx), "score": float(score)})
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, scored


def choose_threshold(calibrated_model: CalibratedClassifierCV, calib_df: pd.DataFrame, features: list[str]) -> float:
    probs = calibrated_model.predict_proba(calib_df[features])
    candidate_thresholds = np.arange(0.30, 0.56, 0.02)
    best_threshold = 0.40
    best_score = -np.inf

    for threshold in candidate_thresholds:
        pred = policy_predictions(probs, threshold=float(threshold))
        stats = strategy_stats(calib_df, pred)
        bacc = balanced_accuracy_score(calib_df["label"], pred)

        if stats["activity"] < 0.12 or stats["activity"] > 0.55:
            continue

        score = (
            0.50 * stats["sharpe_like"]
            + 0.35 * bacc
            - 0.25 * abs(stats["activity"] - 0.28)
            - 0.20 * stats["turnover"]
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def load_round2_metrics() -> dict[str, float] | None:
    path = Path("reports/round2_training_summary.json")
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    metrics = payload.get("test_metrics", {})
    return {
        "ovr_auc": float(metrics.get("ovr_auc", np.nan)),
        "balanced_accuracy_policy": float(metrics.get("balanced_accuracy_policy", np.nan)),
        "strategy_return": float(metrics.get("strategy_return", np.nan)),
        "strategy_sharpe_like": float(metrics.get("strategy_sharpe_like", np.nan)),
    }


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
    data = add_features(data)
    data["label"] = make_label(data, threshold=0.002)
    data = data.dropna().reset_index(drop=True)

    features = [
        "returns_1d",
        "returns_5d",
        "returns_21d",
        "volatility_21d",
        "volatility_10bar",
        "volatility_30bar",
        "volatility_regime_pct",
        "volume_zscore_21d",
        "rsi_14",
        "macd",
        "bb_width",
        "atr_14",
        "trend_strength",
        "vwap_distance_20",
        "aapl_vs_spy_6bar",
        "aapl_vs_qqq_6bar",
        "aapl_vs_spy_24bar",
        "aapl_vs_qqq_24bar",
        "aapl_vs_spy_48bar",
        "aapl_vs_qqq_48bar",
        "minute_of_day",
        "session_progress",
        "minute_sin",
        "minute_cos",
        "minute_sin_2",
        "minute_cos_2",
        "overnight_gap",
        "intrabar_range",
        "body_size",
    ]

    train_df, test_df = split_train_test(data)
    calib_start = int(len(train_df) * 0.82)
    fit_df = train_df.iloc[:calib_start].copy()
    calib_df = train_df.iloc[calib_start:].copy()

    best_params, candidate_scores = select_hyperparameters(fit_df, features)
    base_model = XGBClassifier(**best_params)
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    calibrated.fit(
        fit_df[features],
        fit_df["label"],
        sample_weight=class_weights(fit_df["label"]),
    )

    threshold = choose_threshold(calibrated, calib_df, features)

    test_prob = calibrated.predict_proba(test_df[features])
    test_pred_raw = test_prob.argmax(axis=1)
    test_pred_policy = policy_predictions(test_prob, threshold=threshold)

    test_auc = roc_auc_score(test_df["label"], test_prob, multi_class="ovr")
    test_bacc_raw = balanced_accuracy_score(test_df["label"], test_pred_raw)
    test_bacc_policy = balanced_accuracy_score(test_df["label"], test_pred_policy)
    test_stats = strategy_stats(test_df, test_pred_policy)

    print(f"Round3 Test OVR ROC-AUC: {test_auc:.4f}")
    print(f"Round3 Balanced Accuracy (raw): {test_bacc_raw:.4f}")
    print(f"Round3 Balanced Accuracy (policy): {test_bacc_policy:.4f}")
    print(f"Round3 Strategy Return: {test_stats['return']:.4%}")
    print(f"Round3 Strategy Max DD: {test_stats['max_drawdown']:.4%}")
    print(classification_report(test_df["label"], test_pred_policy, digits=4, zero_division=0))

    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated, models_dir / "aapl_intraday_round3_calibrated.joblib")
    (models_dir / "aapl_intraday_round3_features.txt").write_text("\n".join(features))

    round2_metrics = load_round2_metrics()
    improvement_vs_round2: dict[str, float] | None = None
    if round2_metrics is not None:
        improvement_vs_round2 = {
            "delta_ovr_auc": float(test_auc - round2_metrics["ovr_auc"]),
            "delta_balanced_accuracy_policy": float(
                test_bacc_policy - round2_metrics["balanced_accuracy_policy"]
            ),
            "delta_strategy_return": float(test_stats["return"] - round2_metrics["strategy_return"]),
            "delta_strategy_sharpe_like": float(
                test_stats["sharpe_like"] - round2_metrics["strategy_sharpe_like"]
            ),
        }

    summary = {
        "model": "aapl_intraday_round3_calibrated",
        "calibration": "sigmoid",
        "selected_params": best_params,
        "candidate_scores": candidate_scores,
        "threshold": threshold,
        "test_metrics": {
            "ovr_auc": float(test_auc),
            "balanced_accuracy_raw": float(test_bacc_raw),
            "balanced_accuracy_policy": float(test_bacc_policy),
            "strategy_return": float(test_stats["return"]),
            "strategy_max_drawdown": float(test_stats["max_drawdown"]),
            "strategy_sharpe_like": float(test_stats["sharpe_like"]),
            "strategy_activity": float(test_stats["activity"]),
            "strategy_turnover": float(test_stats["turnover"]),
        },
        "improvement_vs_round2": improvement_vs_round2,
    }

    (reports_dir / "round3_training_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved round3 model artifacts and summary.")


if __name__ == "__main__":
    main()
