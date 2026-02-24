from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

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
    data["minute_of_day"] = data["Datetime"].dt.hour * 60 + data["Datetime"].dt.minute
    data["session_progress"] = data["minute_of_day"] / 390.0
    return data


def make_long_short_label(df: pd.DataFrame, horizon_bars: int = 12, threshold: float = 0.003) -> pd.Series:
    forward_return = df["Close"].shift(-horizon_bars) / df["Close"] - 1.0
    label = np.where(forward_return > threshold, 2, np.where(forward_return < -threshold, 0, 1))
    return pd.Series(label, index=df.index)


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_ratio)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def ensure_dirs() -> tuple[Path, Path]:
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir, figures_dir


def save_class_distribution(test_df: pd.DataFrame, figures_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    sns.countplot(x=test_df["label"], hue=test_df["label"], palette="viridis", legend=False)
    plt.title("Test Set Label Distribution")
    plt.xlabel("Class (0=Short, 1=Neutral, 2=Long)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close()


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, figures_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Short", "Neutral", "Long"],
        cmap="Blues",
        ax=axis,
        colorbar=False,
    )
    display.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def save_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, figures_dir: Path) -> None:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(7, 5))
    for class_index, class_name in enumerate(["Short", "Neutral", "Long"]):
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true_bin[:, class_index], y_prob[:, class_index])
        class_auc = roc_auc_score(y_true_bin[:, class_index], y_prob[:, class_index])
        plt.plot(false_positive_rate, true_positive_rate, label=f"{class_name} AUC={class_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("One-vs-Rest ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close()


def save_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, figures_dir: Path) -> None:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(7, 5))
    for class_index, class_name in enumerate(["Short", "Neutral", "Long"]):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, class_index], y_prob[:, class_index])
        plt.plot(recall, precision, label=class_name)
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curves.png", dpi=150)
    plt.close()


def save_prediction_confidence(y_prob: np.ndarray, figures_dir: Path) -> None:
    confidence = y_prob.max(axis=1)
    plt.figure(figsize=(7, 4))
    sns.histplot(confidence, bins=30, kde=True)
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Max Class Probability")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(figures_dir / "prediction_confidence.png", dpi=150)
    plt.close()


def save_feature_importance(model: object, features: list[str], figures_dir: Path) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    sorted_indices = np.argsort(importances)
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(features)[sorted_indices], np.array(importances)[sorted_indices], color="teal")
    plt.title("Model Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close()


def save_strategy_curve(test_df: pd.DataFrame, y_pred: np.ndarray, figures_dir: Path) -> tuple[float, float]:
    test_data = test_df.reset_index(drop=True)
    position_map = {0: -1.0, 1: 0.0, 2: 1.0}
    positions = pd.Series(y_pred, index=test_data.index).map(position_map).astype(float)
    next_bar_return = test_data["Close"].pct_change().shift(-1).fillna(0.0)
    turnover = positions.diff().abs().fillna(0.0)
    trading_cost = turnover * (1.5 / 10000)
    strategy_return = positions * next_bar_return - trading_cost
    buy_hold_return = next_bar_return

    strategy_curve = (1 + strategy_return).cumprod()
    buy_hold_curve = (1 + buy_hold_return).cumprod()

    plt.figure(figsize=(8, 5))
    plt.plot(test_data["Datetime"], strategy_curve, label="Predicted Strategy")
    plt.plot(test_data["Datetime"], buy_hold_curve, label="AAPL Buy & Hold")
    plt.title("Cumulative Return Curve (Test Window)")
    plt.xlabel("Datetime")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "strategy_vs_buyhold.png", dpi=150)
    plt.close()

    return float(strategy_curve.iloc[-1] - 1.0), float(buy_hold_curve.iloc[-1] - 1.0)


def save_rolling_accuracy(y_true: np.ndarray, y_pred: np.ndarray, figures_dir: Path) -> None:
    rolling_window = 100
    rolling_series = pd.Series((y_true == y_pred).astype(float)).rolling(rolling_window).mean()
    plt.figure(figsize=(8, 4))
    plt.plot(rolling_series)
    plt.title(f"Rolling Accuracy ({rolling_window} bars)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(figures_dir / "rolling_accuracy.png", dpi=150)
    plt.close()


def save_probability_heatmap(y_prob: np.ndarray, figures_dir: Path) -> None:
    sample_size = min(150, len(y_prob))
    heatmap_data = y_prob[:sample_size]
    plt.figure(figsize=(9, 4))
    sns.heatmap(heatmap_data.T, cmap="mako", cbar=True)
    plt.title("Class Probability Heatmap (First Test Samples)")
    plt.xlabel("Test Sample")
    plt.ylabel("Class Probability (Short/Neutral/Long)")
    plt.tight_layout()
    plt.savefig(figures_dir / "probability_heatmap.png", dpi=150)
    plt.close()


def main() -> None:
    model_path = Path("models/aapl_intraday_xgb.joblib")
    feature_path = Path("models/aapl_intraday_xgb_features.txt")
    aapl_path = Path("data/AAPL_intraday_5m.csv")
    spy_path = Path("data/SPY_intraday_5m.csv")
    qqq_path = Path("data/QQQ_intraday_5m.csv")

    required_paths = [model_path, feature_path, aapl_path, spy_path, qqq_path]
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required file: {required_path}")

    model = joblib.load(model_path)
    features = [line.strip() for line in feature_path.read_text().splitlines() if line.strip()]

    aapl = load_intraday(aapl_path)
    spy = load_intraday(spy_path)
    qqq = load_intraday(qqq_path)
    data = merge_context(aapl, spy, qqq)
    data = add_intraday_features(data)
    data["label"] = make_long_short_label(data, horizon_bars=12, threshold=0.003)
    data = data.dropna().reset_index(drop=True)

    train_df, test_df = time_split(data)
    y_test = test_df["label"].to_numpy()
    y_prob = model.predict_proba(test_df[features])
    y_pred = y_prob.argmax(axis=1)

    reports_dir, figures_dir = ensure_dirs()

    save_class_distribution(test_df, figures_dir)
    save_confusion_matrix(y_test, y_pred, figures_dir)
    save_roc_curves(y_test, y_prob, figures_dir)
    save_pr_curves(y_test, y_prob, figures_dir)
    save_prediction_confidence(y_prob, figures_dir)
    save_feature_importance(model, features, figures_dir)
    strategy_return, buy_hold_return = save_strategy_curve(test_df, y_pred, figures_dir)
    save_rolling_accuracy(y_test, y_pred, figures_dir)
    save_probability_heatmap(y_prob, figures_dir)

    majority_class = int(pd.Series(train_df["label"]).mode().iloc[0])
    majority_pred = np.full_like(y_test, fill_value=majority_class)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "ovr_roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr"),
        "multiclass_log_loss": log_loss(y_test, y_prob),
        "majority_baseline_accuracy": accuracy_score(y_test, majority_pred),
        "strategy_test_return": strategy_return,
        "buyhold_test_return": buy_hold_return,
    }

    report_text = [
        "# AAPL Intraday Model Proof Report",
        "",
        "## Artifact checks",
        f"- Model loaded from: {model_path}",
        f"- Feature count: {len(features)}",
        f"- Test samples: {len(test_df)}",
        "",
        "## Core metrics",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- OVR ROC-AUC: {metrics['ovr_roc_auc']:.4f}",
        f"- Multi-class log loss: {metrics['multiclass_log_loss']:.4f}",
        f"- Majority-class baseline accuracy: {metrics['majority_baseline_accuracy']:.4f}",
        f"- Strategy test return (simple, cost-adjusted): {metrics['strategy_test_return']:.4%}",
        f"- Buy & hold test return (same window): {metrics['buyhold_test_return']:.4%}",
        "",
        "## Classification report",
        "```",
        classification_report(y_test, y_pred, digits=4),
        "```",
        "",
        "## Figures generated",
        "- reports/figures/class_distribution.png",
        "- reports/figures/confusion_matrix.png",
        "- reports/figures/roc_curves.png",
        "- reports/figures/precision_recall_curves.png",
        "- reports/figures/prediction_confidence.png",
        "- reports/figures/feature_importance.png",
        "- reports/figures/strategy_vs_buyhold.png",
        "- reports/figures/rolling_accuracy.png",
        "- reports/figures/probability_heatmap.png",
    ]

    report_path = reports_dir / "model_proof_report.md"
    report_path.write_text("\n".join(report_text))
    print(f"Saved proof report -> {report_path}")
    print(f"Saved figures in -> {figures_dir}")


if __name__ == "__main__":
    main()
