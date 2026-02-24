from __future__ import annotations

import json
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
    roc_auc_score,
)

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


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    data = build_technical_features(
        df.rename(columns={"Datetime": "Date"}).assign(Date=df["Datetime"])
    ).rename(columns={"Date": "Datetime"})
    data["aapl_vs_spy_6bar"] = data["Close"].pct_change(6) - data["SPY_Close"].pct_change(6)
    data["aapl_vs_qqq_6bar"] = data["Close"].pct_change(6) - data["QQQ_Close"].pct_change(6)
    data["minute_of_day"] = data["Datetime"].dt.hour * 60 + data["Datetime"].dt.minute
    data["session_progress"] = data["minute_of_day"] / 390.0
    return data


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    data = add_base_features(df)
    data["aapl_vs_spy_24bar"] = data["Close"].pct_change(24) - data["SPY_Close"].pct_change(24)
    data["aapl_vs_qqq_24bar"] = data["Close"].pct_change(24) - data["QQQ_Close"].pct_change(24)
    data["minute_sin"] = np.sin(2 * np.pi * data["session_progress"])
    data["minute_cos"] = np.cos(2 * np.pi * data["session_progress"])
    data["overnight_gap"] = data["Open"] / data["Close"].shift(1) - 1.0
    data["intrabar_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["volatility_10bar"] = data["returns_1d"].rolling(10).std()
    return data


def make_label(df: pd.DataFrame, threshold: float) -> pd.Series:
    forward_return = df["Close"].shift(-12) / df["Close"] - 1.0
    return pd.Series(np.where(forward_return > threshold, 2, np.where(forward_return < -threshold, 0, 1)))


def time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_ratio)
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


def confidence_policy(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    pred = probabilities.argmax(axis=1)
    conf = probabilities.max(axis=1)
    return np.where(conf >= threshold, pred, 1).astype(int)


def strategy_returns(test_df: pd.DataFrame, pred: np.ndarray) -> pd.Series:
    position = pd.Series(pred, index=test_df.index).map({0: -1.0, 1: 0.0, 2: 1.0}).astype(float)
    next_ret = test_df["Close"].pct_change().shift(-1).fillna(0.0)
    turnover = position.diff().abs().fillna(0.0)
    costs = turnover * (1.5 / 10000)
    return position * next_ret - costs


def ensure_dirs() -> tuple[Path, Path]:
    out_dir = Path("reports/improvement_round_1")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir


def save_metric_bars(metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    melted = metrics_df.melt(id_vars=["model"], var_name="metric", value_name="value")
    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="model")
    plt.xticks(rotation=25, ha="right")
    plt.title("Baseline vs Enhanced Metrics")
    plt.tight_layout()
    plt.savefig(fig_dir / "metric_comparison_bars.png", dpi=150)
    plt.close()


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Short", "Neutral", "Long"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_cumulative_returns(time_index: pd.Series, curves: dict[str, pd.Series], fig_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for name, curve in curves.items():
        plt.plot(time_index, curve, label=name)
    plt.title("Cumulative Return Comparison")
    plt.xlabel("Datetime")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "cumulative_returns_comparison.png", dpi=150)
    plt.close()


def save_rolling_metric(series_map: dict[str, pd.Series], fig_dir: Path, name: str, y_label: str) -> None:
    plt.figure(figsize=(9, 4.8))
    for label, series in series_map.items():
        plt.plot(series.values, label=label)
    plt.title(name)
    plt.ylabel(y_label)
    plt.xlabel("Test Sample Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def save_confidence_hist(conf_baseline: np.ndarray, conf_enhanced: np.ndarray, fig_dir: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    sns.histplot(conf_baseline, bins=30, color="steelblue", alpha=0.5, label="Baseline", kde=True)
    sns.histplot(conf_enhanced, bins=30, color="darkorange", alpha=0.5, label="Enhanced", kde=True)
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Max class probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "confidence_distribution_comparison.png", dpi=150)
    plt.close()


def save_trade_activity(base_pred: np.ndarray, enh_pred: np.ndarray, fig_dir: Path) -> None:
    base_activity = pd.Series(base_pred).value_counts(normalize=True).sort_index()
    enh_activity = pd.Series(enh_pred).value_counts(normalize=True).sort_index()
    activity = pd.DataFrame(
        {
            "class": [0, 1, 2],
            "baseline": [base_activity.get(i, 0.0) for i in [0, 1, 2]],
            "enhanced": [enh_activity.get(i, 0.0) for i in [0, 1, 2]],
        }
    )
    melted = activity.melt(id_vars="class", var_name="model", value_name="share")
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=melted, x="class", y="share", hue="model")
    plt.title("Trade Signal Distribution")
    plt.xlabel("Predicted class (0=Short, 1=Neutral, 2=Long)")
    plt.ylabel("Share")
    plt.tight_layout()
    plt.savefig(fig_dir / "signal_distribution_comparison.png", dpi=150)
    plt.close()


def save_enhanced_feature_importance(model: object, features: list[str], fig_dir: Path) -> None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    order = np.argsort(importances)
    plt.figure(figsize=(8.5, 6.5))
    plt.barh(np.array(features)[order], np.array(importances)[order], color="seagreen")
    plt.title("Enhanced Model Feature Importance")
    plt.tight_layout()
    plt.savefig(fig_dir / "enhanced_feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    baseline_model_path = Path("models/aapl_intraday_xgb.joblib")
    baseline_feat_path = Path("models/aapl_intraday_xgb_features.txt")
    enhanced_model_path = Path("models/aapl_intraday_xgb_enhanced.joblib")
    enhanced_feat_path = Path("models/aapl_intraday_xgb_enhanced_features.txt")
    summary_path = Path("reports/enhanced_training_summary.json")

    required = [
        baseline_model_path,
        baseline_feat_path,
        enhanced_model_path,
        enhanced_feat_path,
        summary_path,
        Path("data/AAPL_intraday_5m.csv"),
        Path("data/SPY_intraday_5m.csv"),
        Path("data/QQQ_intraday_5m.csv"),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    summary = json.loads(summary_path.read_text())
    threshold = float(summary.get("confidence_threshold", 0.56))

    baseline_model = joblib.load(baseline_model_path)
    enhanced_model = joblib.load(enhanced_model_path)
    baseline_features = [x.strip() for x in baseline_feat_path.read_text().splitlines() if x.strip()]
    enhanced_features = [x.strip() for x in enhanced_feat_path.read_text().splitlines() if x.strip()]

    aapl = load_intraday(Path("data/AAPL_intraday_5m.csv"))
    spy = load_intraday(Path("data/SPY_intraday_5m.csv"))
    qqq = load_intraday(Path("data/QQQ_intraday_5m.csv"))
    merged = merge_context(aapl, spy, qqq)

    base_data = add_base_features(merged)
    base_data["label"] = make_label(base_data, threshold=0.003)
    base_data = base_data.dropna().reset_index(drop=True)
    _, base_test = time_split(base_data)

    enhanced_data = add_enhanced_features(merged)
    enhanced_data["label"] = make_label(enhanced_data, threshold=0.0025)
    enhanced_data = enhanced_data.dropna().reset_index(drop=True)
    _, enhanced_test = time_split(enhanced_data)

    compare_len = min(len(base_test), len(enhanced_test))
    base_test = base_test.iloc[-compare_len:].reset_index(drop=True)
    enhanced_test = enhanced_test.iloc[-compare_len:].reset_index(drop=True)

    y_base = base_test["label"].to_numpy()
    y_enh = enhanced_test["label"].to_numpy()

    base_prob = baseline_model.predict_proba(base_test[baseline_features])
    base_pred = base_prob.argmax(axis=1)

    enh_prob = enhanced_model.predict_proba(enhanced_test[enhanced_features])
    enh_pred_raw = enh_prob.argmax(axis=1)
    enh_pred_policy = confidence_policy(enh_prob, threshold)

    base_returns = strategy_returns(base_test, base_pred)
    enh_returns = strategy_returns(enhanced_test, enh_pred_policy)
    buy_hold_returns = enhanced_test["Close"].pct_change().shift(-1).fillna(0.0)

    base_curve = (1 + base_returns).cumprod()
    enh_curve = (1 + enh_returns).cumprod()
    buy_hold_curve = (1 + buy_hold_returns).cumprod()

    metrics_df = pd.DataFrame(
        [
            {
                "model": "baseline",
                "accuracy": accuracy_score(y_base, base_pred),
                "balanced_accuracy": balanced_accuracy_score(y_base, base_pred),
                "ovr_auc": roc_auc_score(y_base, base_prob, multi_class="ovr"),
                "log_loss": log_loss(y_base, base_prob),
                "return": float(base_curve.iloc[-1] - 1.0),
            },
            {
                "model": "enhanced_policy",
                "accuracy": accuracy_score(y_enh, enh_pred_policy),
                "balanced_accuracy": balanced_accuracy_score(y_enh, enh_pred_policy),
                "ovr_auc": roc_auc_score(y_enh, enh_prob, multi_class="ovr"),
                "log_loss": log_loss(y_enh, enh_prob),
                "return": float(enh_curve.iloc[-1] - 1.0),
            },
            {
                "model": "buy_hold",
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "ovr_auc": np.nan,
                "log_loss": np.nan,
                "return": float(buy_hold_curve.iloc[-1] - 1.0),
            },
        ]
    )

    out_dir, fig_dir = ensure_dirs()

    save_metric_bars(metrics_df.query("model != 'buy_hold'"), fig_dir)
    save_confusion(y_base, base_pred, fig_dir / "baseline_confusion_matrix.png", "Baseline Confusion")
    save_confusion(
        y_enh,
        enh_pred_policy,
        fig_dir / "enhanced_confusion_matrix.png",
        "Enhanced Confusion (Policy)",
    )
    save_cumulative_returns(
        enhanced_test["Datetime"],
        {
            "Baseline Strategy": base_curve,
            "Enhanced Strategy": enh_curve,
            "AAPL Buy & Hold": buy_hold_curve,
        },
        fig_dir,
    )

    rolling_window = 120
    rolling_acc = {
        "Baseline": pd.Series((y_base == base_pred).astype(float)).rolling(rolling_window).mean(),
        "Enhanced (Policy)": pd.Series((y_enh == enh_pred_policy).astype(float)).rolling(rolling_window).mean(),
    }
    save_rolling_metric(rolling_acc, fig_dir, "Rolling Accuracy Comparison", "Accuracy")

    rolling_ret = {
        "Baseline": base_returns.rolling(rolling_window).sum(),
        "Enhanced": enh_returns.rolling(rolling_window).sum(),
        "Buy & Hold": buy_hold_returns.rolling(rolling_window).sum(),
    }
    save_rolling_metric(rolling_ret, fig_dir, "Rolling Return Comparison", "Window return")

    save_confidence_hist(base_prob.max(axis=1), enh_prob.max(axis=1), fig_dir)
    save_trade_activity(base_pred, enh_pred_policy, fig_dir)
    save_enhanced_feature_importance(enhanced_model, enhanced_features, fig_dir)

    reports_text = [
        "# Improvement Round 1: Baseline vs Enhanced",
        "",
        "## Summary",
        f"- Enhanced confidence threshold: {threshold:.2f}",
        f"- Baseline OVR AUC: {metrics_df.loc[metrics_df.model == 'baseline', 'ovr_auc'].iloc[0]:.4f}",
        f"- Enhanced OVR AUC: {metrics_df.loc[metrics_df.model == 'enhanced_policy', 'ovr_auc'].iloc[0]:.4f}",
        f"- Baseline balanced accuracy: {metrics_df.loc[metrics_df.model == 'baseline', 'balanced_accuracy'].iloc[0]:.4f}",
        f"- Enhanced balanced accuracy: {metrics_df.loc[metrics_df.model == 'enhanced_policy', 'balanced_accuracy'].iloc[0]:.4f}",
        f"- Baseline strategy return: {metrics_df.loc[metrics_df.model == 'baseline', 'return'].iloc[0]:.4%}",
        f"- Enhanced strategy return: {metrics_df.loc[metrics_df.model == 'enhanced_policy', 'return'].iloc[0]:.4%}",
        f"- Buy & hold return: {metrics_df.loc[metrics_df.model == 'buy_hold', 'return'].iloc[0]:.4%}",
        "",
        "## Classification Reports",
        "### Baseline",
        "```",
        classification_report(y_base, base_pred, digits=4),
        "```",
        "",
        "### Enhanced (Policy)",
        "```",
        classification_report(y_enh, enh_pred_policy, digits=4),
        "```",
        "",
        "## Figures",
        "- reports/improvement_round_1/figures/metric_comparison_bars.png",
        "- reports/improvement_round_1/figures/baseline_confusion_matrix.png",
        "- reports/improvement_round_1/figures/enhanced_confusion_matrix.png",
        "- reports/improvement_round_1/figures/cumulative_returns_comparison.png",
        "- reports/improvement_round_1/figures/rolling_accuracy_comparison.png",
        "- reports/improvement_round_1/figures/rolling_return_comparison.png",
        "- reports/improvement_round_1/figures/confidence_distribution_comparison.png",
        "- reports/improvement_round_1/figures/signal_distribution_comparison.png",
        "- reports/improvement_round_1/figures/enhanced_feature_importance.png",
    ]

    (out_dir / "improvement_report.md").write_text("\n".join(reports_text))
    metrics_df.to_csv(out_dir / "metric_table.csv", index=False)
    print(f"Saved improvement report -> {out_dir / 'improvement_report.md'}")
    print(f"Saved figures in -> {fig_dir}")


if __name__ == "__main__":
    main()
