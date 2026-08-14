from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _extract_metric(payload: dict | None, key_path: list[str], default: float | None = None) -> float | None:
    if payload is None:
        return default
    node: object = payload
    for key in key_path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    if node is None:
        return default
    if isinstance(node, (int, float, str)):
        try:
            return float(node)
        except ValueError:
            return default
    return default


def main() -> None:
    baseline_metrics_path = Path("reports/improvement_round_1/metric_table.csv")
    round2_summary = _safe_read_json(Path("reports/round2_training_summary.json"))
    round3_summary = _safe_read_json(Path("reports/round3_training_summary.json"))

    baseline_auc = None
    baseline_bacc = None
    baseline_return = None

    if baseline_metrics_path.exists():
        baseline_df = pd.read_csv(baseline_metrics_path)
        baseline_row = baseline_df.loc[baseline_df["model"] == "baseline"]
        if len(baseline_row):
            baseline_auc = float(baseline_row.iloc[0]["ovr_auc"])
            baseline_bacc = float(baseline_row.iloc[0]["balanced_accuracy"])
            baseline_return = float(baseline_row.iloc[0]["return"])

    round1 = _safe_read_json(Path("reports/enhanced_training_summary.json"))

    rows = [
        {
            "round": "baseline",
            "model": "aapl_intraday_xgb",
            "ovr_auc": baseline_auc,
            "balanced_accuracy_policy": baseline_bacc,
            "strategy_return": baseline_return,
        },
        {
            "round": "round1",
            "model": "aapl_intraday_xgb_enhanced",
            "ovr_auc": _extract_metric(round1, ["test_metrics", "ovr_roc_auc"]),
            "balanced_accuracy_policy": _extract_metric(
                round1, ["test_metrics", "balanced_accuracy_policy"]
            ),
            "strategy_return": None,
        },
        {
            "round": "round2",
            "model": "aapl_intraday_round2_calibrated",
            "ovr_auc": _extract_metric(round2_summary, ["test_metrics", "ovr_auc"]),
            "balanced_accuracy_policy": _extract_metric(
                round2_summary, ["test_metrics", "balanced_accuracy_policy"]
            ),
            "strategy_return": _extract_metric(round2_summary, ["test_metrics", "strategy_return"]),
        },
        {
            "round": "round3",
            "model": "aapl_intraday_round3_calibrated",
            "ovr_auc": _extract_metric(round3_summary, ["test_metrics", "ovr_auc"]),
            "balanced_accuracy_policy": _extract_metric(
                round3_summary, ["test_metrics", "balanced_accuracy_policy"]
            ),
            "strategy_return": _extract_metric(round3_summary, ["test_metrics", "strategy_return"]),
        },
    ]

    audit_df = pd.DataFrame(rows)
    audit_df["delta_auc_vs_prev"] = audit_df["ovr_auc"].diff()
    audit_df["delta_bacc_vs_prev"] = audit_df["balanced_accuracy_policy"].diff()
    audit_df["delta_return_vs_prev"] = audit_df["strategy_return"].diff()

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "iteration_audit.csv"
    out_md = reports_dir / "iteration_audit.md"
    audit_df.to_csv(out_csv, index=False)

    lines = [
        "# Iteration Audit",
        "",
        "This file tracks whether each training round improved vs the previous one.",
        "",
        "## Summary Table",
        "",
        audit_df.to_markdown(index=False),
        "",
    ]

    if round3_summary and round3_summary.get("improvement_vs_round2"):
        lines.extend(
            [
                "## Round3 vs Round2 (direct deltas)",
                "```json",
                json.dumps(round3_summary["improvement_vs_round2"], indent=2),
                "```",
            ]
        )

    out_md.write_text("\n".join(lines))
    print(f"Saved iteration audit -> {out_md}")
    print(f"Saved iteration audit csv -> {out_csv}")


if __name__ == "__main__":
    main()
