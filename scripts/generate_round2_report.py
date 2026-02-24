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


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    data = build_technical_features(
        df.rename(columns={"Datetime": "Date"}).assign(Date=df["Datetime"])
    ).rename(columns={"Date": "Datetime"})
    data["aapl_vs_spy_6bar"] = data["Close"].pct_change(6) - data["SPY_Close"].pct_change(6)
    data["aapl_vs_qqq_6bar"] = data["Close"].pct_change(6) - data["QQQ_Close"].pct_change(6)
    data["minute_of_day"] = data["Datetime"].dt.hour * 60 + data["Datetime"].dt.minute
    data["session_progress"] = data["minute_of_day"] / 390.0
    return data


def enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    data = base_features(df)
    data["aapl_vs_spy_24bar"] = data["Close"].pct_change(24) - data["SPY_Close"].pct_change(24)
    data["aapl_vs_qqq_24bar"] = data["Close"].pct_change(24) - data["QQQ_Close"].pct_change(24)
    data["minute_sin"] = np.sin(2 * np.pi * data["session_progress"])
    data["minute_cos"] = np.cos(2 * np.pi * data["session_progress"])
    data["overnight_gap"] = data["Open"] / data["Close"].shift(1) - 1.0
    data["intrabar_range"] = (data["High"] - data["Low"]) / data["Close"]
    data["volatility_10bar"] = data["returns_1d"].rolling(10).std()
    return data


def round2_features(df: pd.DataFrame) -> pd.DataFrame:
    data = enhanced_features(df)
    data["aapl_vs_spy_48bar"] = data["Close"].pct_change(48) - data["SPY_Close"].pct_change(48)
    data["aapl_vs_qqq_48bar"] = data["Close"].pct_change(48) - data["QQQ_Close"].pct_change(48)
    data["minute_sin_2"] = np.sin(4 * np.pi * data["session_progress"])
    data["minute_cos_2"] = np.cos(4 * np.pi * data["session_progress"])
    data["body_size"] = (data["Close"] - data["Open"]).abs() / data["Close"]
    data["volatility_30bar"] = data["returns_1d"].rolling(30).std()
    data["trend_strength"] = (data["Close"] / data["Close"].rolling(20).mean()) - 1.0
    return data


def make_label(df: pd.DataFrame, threshold: float) -> pd.Series:
    forward_return = df["Close"].shift(-12) / df["Close"] - 1.0
    return pd.Series(np.where(forward_return > threshold, 2, np.where(forward_return < -threshold, 0, 1)))


def split(df: pd.DataFrame, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = int(len(df) * ratio)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


def confidence_policy(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    hard = probabilities.argmax(axis=1)
    conf = probabilities.max(axis=1)
    return np.where(conf >= threshold, hard, 1)


def strategy_returns(df: pd.DataFrame, pred: np.ndarray) -> pd.Series:
    pos = pd.Series(pred, index=df.index).map({0: -1.0, 1: 0.0, 2: 1.0}).astype(float)
    next_ret = df["Close"].pct_change().shift(-1).fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    costs = turnover * (1.5 / 10000)
    return pos * next_ret - costs


def ensure_dirs() -> tuple[Path, Path]:
    out_dir = Path("reports/improvement_round_2")
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir


def draw_metric_chart(metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    melted = metrics_df.melt(id_vars=["model"], var_name="metric", value_name="value")
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=melted, x="metric", y="value", hue="model")
    plt.xticks(rotation=25, ha="right")
    plt.title("Round-2 Metric Comparison")
    plt.tight_layout()
    plt.savefig(fig_dir / "round2_metric_comparison.png", dpi=150)
    plt.close()


def draw_cumulative_curves(time_index: pd.Series, curves: dict[str, pd.Series], fig_dir: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    for name, curve in curves.items():
        plt.plot(time_index, curve, label=name)
    plt.title("Cumulative Return Curves")
    plt.xlabel("Datetime")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "round2_cumulative_curves.png", dpi=150)
    plt.close()


def draw_rolling(series_map: dict[str, pd.Series], fig_dir: Path, filename: str, title: str, y_label: str) -> None:
    plt.figure(figsize=(10, 5))
    for label, series in series_map.items():
        plt.plot(series.values, label=label)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("Test Sample Index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / filename, dpi=150)
    plt.close()


def draw_confusion(y_true: np.ndarray, y_pred: np.ndarray, fig_dir: Path, filename: str, title: str) -> None:
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
    plt.savefig(fig_dir / filename, dpi=150)
    plt.close()


def draw_activity(base_pred: np.ndarray, enh_pred: np.ndarray, r2_pred: np.ndarray, fig_dir: Path) -> None:
    rows = []
    for model_name, pred in [
        ("baseline", base_pred),
        ("round1_enhanced", enh_pred),
        ("round2_calibrated", r2_pred),
    ]:
        counts = pd.Series(pred).value_counts(normalize=True)
        for cls in [0, 1, 2]:
            rows.append({"model": model_name, "class": cls, "share": float(counts.get(cls, 0.0))})
    activity_df = pd.DataFrame(rows)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=activity_df, x="class", y="share", hue="model")
    plt.title("Signal Class Distribution Across Rounds")
    plt.xlabel("Class (0=Short, 1=Neutral, 2=Long)")
    plt.tight_layout()
    plt.savefig(fig_dir / "round2_signal_distribution.png", dpi=150)
    plt.close()


def draw_confidence(base_prob: np.ndarray, enh_prob: np.ndarray, r2_prob: np.ndarray, fig_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.kdeplot(base_prob.max(axis=1), label="baseline")
    sns.kdeplot(enh_prob.max(axis=1), label="round1_enhanced")
    sns.kdeplot(r2_prob.max(axis=1), label="round2_calibrated")
    plt.title("Confidence Distribution Across Rounds")
    plt.xlabel("Max class probability")
    plt.tight_layout()
    plt.legend()
    plt.savefig(fig_dir / "round2_confidence_density.png", dpi=150)
    plt.close()


def draw_drawdown(curves: dict[str, pd.Series], fig_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    for name, curve in curves.items():
        dd = curve / curve.cummax() - 1.0
        plt.plot(dd.values, label=name)
    plt.title("Drawdown Curves")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.legend()
    plt.savefig(fig_dir / "round2_drawdown_curves.png", dpi=150)
    plt.close()


def draw_round2_feature_importance(model: object, features: list[str], fig_dir: Path) -> None:
    estimator = None
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        estimator = getattr(model.calibrated_classifiers_[0], "estimator", None)
    if estimator is None:
        estimator = getattr(model, "estimator", None)
    if estimator is None:
        estimator = model
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return
    order = np.argsort(importances)
    plt.figure(figsize=(9, 7))
    plt.barh(np.array(features)[order], np.array(importances)[order], color="darkgreen")
    plt.title("Round-2 Feature Importance")
    plt.tight_layout()
    plt.savefig(fig_dir / "round2_feature_importance.png", dpi=150)
    plt.close()


def main() -> None:
    required_paths = [
        Path("models/aapl_intraday_xgb.joblib"),
        Path("models/aapl_intraday_xgb_features.txt"),
        Path("models/aapl_intraday_xgb_enhanced.joblib"),
        Path("models/aapl_intraday_xgb_enhanced_features.txt"),
        Path("models/aapl_intraday_round2_calibrated.joblib"),
        Path("models/aapl_intraday_round2_features.txt"),
        Path("reports/enhanced_training_summary.json"),
        Path("reports/round2_training_summary.json"),
        Path("data/AAPL_intraday_5m.csv"),
        Path("data/SPY_intraday_5m.csv"),
        Path("data/QQQ_intraday_5m.csv"),
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    r1_summary = json.loads(Path("reports/enhanced_training_summary.json").read_text())
    r2_summary = json.loads(Path("reports/round2_training_summary.json").read_text())
    r1_threshold = float(r1_summary.get("confidence_threshold", 0.45))
    r2_threshold = float(r2_summary.get("threshold", 0.55))

    baseline_model = joblib.load("models/aapl_intraday_xgb.joblib")
    round1_model = joblib.load("models/aapl_intraday_xgb_enhanced.joblib")
    round2_model = joblib.load("models/aapl_intraday_round2_calibrated.joblib")

    baseline_features = [x.strip() for x in Path("models/aapl_intraday_xgb_features.txt").read_text().splitlines() if x.strip()]
    round1_features = [x.strip() for x in Path("models/aapl_intraday_xgb_enhanced_features.txt").read_text().splitlines() if x.strip()]
    round2_features_list = [x.strip() for x in Path("models/aapl_intraday_round2_features.txt").read_text().splitlines() if x.strip()]

    aapl = load_intraday(Path("data/AAPL_intraday_5m.csv"))
    spy = load_intraday(Path("data/SPY_intraday_5m.csv"))
    qqq = load_intraday(Path("data/QQQ_intraday_5m.csv"))
    merged = merge_context(aapl, spy, qqq)

    base_data = base_features(merged)
    base_data["label"] = make_label(base_data, 0.003)
    base_data = base_data.dropna().reset_index(drop=True)
    _, base_test = split(base_data)

    r1_data = enhanced_features(merged)
    r1_data["label"] = make_label(r1_data, 0.0025)
    r1_data = r1_data.dropna().reset_index(drop=True)
    _, r1_test = split(r1_data)

    r2_data = round2_features(merged)
    r2_data["label"] = make_label(r2_data, 0.002)
    r2_data = r2_data.dropna().reset_index(drop=True)
    _, r2_test = split(r2_data)

    compare_len = min(len(base_test), len(r1_test), len(r2_test))
    base_test = base_test.iloc[-compare_len:].reset_index(drop=True)
    r1_test = r1_test.iloc[-compare_len:].reset_index(drop=True)
    r2_test = r2_test.iloc[-compare_len:].reset_index(drop=True)

    y_base = base_test["label"].to_numpy()
    y_r1 = r1_test["label"].to_numpy()
    y_r2 = r2_test["label"].to_numpy()

    base_prob = baseline_model.predict_proba(base_test[baseline_features])
    base_pred = base_prob.argmax(axis=1)

    r1_prob = round1_model.predict_proba(r1_test[round1_features])
    r1_pred = confidence_policy(r1_prob, r1_threshold)

    r2_prob = round2_model.predict_proba(r2_test[round2_features_list])
    r2_pred = confidence_policy(r2_prob, r2_threshold)

    base_ret = strategy_returns(base_test, base_pred)
    r1_ret = strategy_returns(r1_test, r1_pred)
    r2_ret = strategy_returns(r2_test, r2_pred)
    bh_ret = r2_test["Close"].pct_change().shift(-1).fillna(0.0)

    base_curve = (1 + base_ret).cumprod()
    r1_curve = (1 + r1_ret).cumprod()
    r2_curve = (1 + r2_ret).cumprod()
    bh_curve = (1 + bh_ret).cumprod()

    metric_rows = [
        {
            "model": "baseline",
            "accuracy": accuracy_score(y_base, base_pred),
            "balanced_accuracy": balanced_accuracy_score(y_base, base_pred),
            "ovr_auc": roc_auc_score(y_base, base_prob, multi_class="ovr"),
            "log_loss": log_loss(y_base, base_prob),
            "return": float(base_curve.iloc[-1] - 1.0),
        },
        {
            "model": "round1_enhanced",
            "accuracy": accuracy_score(y_r1, r1_pred),
            "balanced_accuracy": balanced_accuracy_score(y_r1, r1_pred),
            "ovr_auc": roc_auc_score(y_r1, r1_prob, multi_class="ovr"),
            "log_loss": log_loss(y_r1, r1_prob),
            "return": float(r1_curve.iloc[-1] - 1.0),
        },
        {
            "model": "round2_calibrated",
            "accuracy": accuracy_score(y_r2, r2_pred),
            "balanced_accuracy": balanced_accuracy_score(y_r2, r2_pred),
            "ovr_auc": roc_auc_score(y_r2, r2_prob, multi_class="ovr"),
            "log_loss": log_loss(y_r2, r2_prob),
            "return": float(r2_curve.iloc[-1] - 1.0),
        },
    ]
    metrics_df = pd.DataFrame(metric_rows)

    out_dir, fig_dir = ensure_dirs()
    draw_metric_chart(metrics_df, fig_dir)
    draw_cumulative_curves(
        r2_test["Datetime"],
        {
            "Baseline": base_curve,
            "Round1 Enhanced": r1_curve,
            "Round2 Calibrated": r2_curve,
            "Buy & Hold": bh_curve,
        },
        fig_dir,
    )

    rolling_window = 120
    draw_rolling(
        {
            "Baseline": pd.Series((y_base == base_pred).astype(float)).rolling(rolling_window).mean(),
            "Round1": pd.Series((y_r1 == r1_pred).astype(float)).rolling(rolling_window).mean(),
            "Round2": pd.Series((y_r2 == r2_pred).astype(float)).rolling(rolling_window).mean(),
        },
        fig_dir,
        "round2_rolling_accuracy.png",
        "Rolling Accuracy Across Rounds",
        "Accuracy",
    )

    draw_rolling(
        {
            "Baseline": base_ret.rolling(rolling_window).sum(),
            "Round1": r1_ret.rolling(rolling_window).sum(),
            "Round2": r2_ret.rolling(rolling_window).sum(),
            "BuyHold": bh_ret.rolling(rolling_window).sum(),
        },
        fig_dir,
        "round2_rolling_returns.png",
        "Rolling Window Return Across Rounds",
        "Window Return",
    )

    draw_confusion(y_base, base_pred, fig_dir, "round2_baseline_confusion.png", "Baseline Confusion")
    draw_confusion(y_r1, r1_pred, fig_dir, "round2_round1_confusion.png", "Round1 Confusion")
    draw_confusion(y_r2, r2_pred, fig_dir, "round2_round2_confusion.png", "Round2 Confusion")
    draw_activity(base_pred, r1_pred, r2_pred, fig_dir)
    draw_confidence(base_prob, r1_prob, r2_prob, fig_dir)
    draw_drawdown(
        {
            "Baseline": base_curve,
            "Round1": r1_curve,
            "Round2": r2_curve,
            "BuyHold": bh_curve,
        },
        fig_dir,
    )
    draw_round2_feature_importance(round2_model, round2_features_list, fig_dir)

    report_lines = [
        "# Improvement Round 2 Report",
        "",
        "## Core results",
        f"- Round1 threshold: {r1_threshold:.2f}",
        f"- Round2 threshold: {r2_threshold:.2f}",
    ]
    for _, row in metrics_df.iterrows():
        report_lines.extend(
            [
                f"- {row['model']} -> AUC {row['ovr_auc']:.4f}, BalAcc {row['balanced_accuracy']:.4f}, Return {row['return']:.4%}",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Classification reports",
            "### Baseline",
            "```",
            classification_report(y_base, base_pred, digits=4, zero_division=0),
            "```",
            "### Round1",
            "```",
            classification_report(y_r1, r1_pred, digits=4, zero_division=0),
            "```",
            "### Round2",
            "```",
            classification_report(y_r2, r2_pred, digits=4, zero_division=0),
            "```",
            "",
            "## Figures",
            "- reports/improvement_round_2/figures/round2_metric_comparison.png",
            "- reports/improvement_round_2/figures/round2_cumulative_curves.png",
            "- reports/improvement_round_2/figures/round2_rolling_accuracy.png",
            "- reports/improvement_round_2/figures/round2_rolling_returns.png",
            "- reports/improvement_round_2/figures/round2_baseline_confusion.png",
            "- reports/improvement_round_2/figures/round2_round1_confusion.png",
            "- reports/improvement_round_2/figures/round2_round2_confusion.png",
            "- reports/improvement_round_2/figures/round2_signal_distribution.png",
            "- reports/improvement_round_2/figures/round2_confidence_density.png",
            "- reports/improvement_round_2/figures/round2_drawdown_curves.png",
            "- reports/improvement_round_2/figures/round2_feature_importance.png",
        ]
    )

    (out_dir / "round2_report.md").write_text("\n".join(report_lines))
    metrics_df.to_csv(out_dir / "round2_metric_table.csv", index=False)
    print(f"Saved round2 report -> {out_dir / 'round2_report.md'}")
    print(f"Saved round2 figures -> {fig_dir}")


if __name__ == "__main__":
    main()
