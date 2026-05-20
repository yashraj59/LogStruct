"""Build Norman perturbation pilot outputs from logged results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "norman_2019_perturb_identity" / "results.json"
TABLE_DIR = ROOT / "paper" / "tables"


METHOD_LABELS = {
    "dummy_most_frequent": "Most frequent",
    "logistic_l2": "L2 logistic",
    "logstruct_identity": "LogStruct identity",
    "logstruct_string_700": "LogStruct STRING-700",
}


def _fmt(metric):
    return f"{metric['mean']:.3f} ({metric['ci_low']:.3f}, {metric['ci_high']:.3f})"


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    result = json.loads(RESULTS.read_text())
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, metrics in result["aggregate_metrics"].items():
        rows.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Balanced accuracy": _fmt(metrics["balanced_accuracy"]),
                "Top-5 accuracy": _fmt(metrics["top5_accuracy"]),
                "Macro-F1": _fmt(metrics["macro_f1"]),
                "MCC": _fmt(metrics["mcc"]),
                "AUROC OvR": _fmt(metrics["ovr_auroc"]),
                "AUPRC OvR": _fmt(metrics["ovr_auprc"]),
                "balanced_accuracy_mean": metrics["balanced_accuracy"]["mean"],
            }
        )
    df = pd.DataFrame(rows).sort_values("balanced_accuracy_mean", ascending=False)
    display_cols = [
        "Method",
        "Balanced accuracy",
        "Top-5 accuracy",
        "Macro-F1",
        "MCC",
        "AUROC OvR",
        "AUPRC OvR",
    ]
    df[display_cols].to_csv(TABLE_DIR / "table_norman_500g_pilot.csv", index=False)
    (TABLE_DIR / "table_norman_500g_pilot.md").write_text(_to_markdown(df[display_cols]))


if __name__ == "__main__":
    main()
