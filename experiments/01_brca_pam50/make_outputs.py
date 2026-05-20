"""Build METABRIC paper tables and figures from logged results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "metabric_pam50" / "results.json"
TABLE_DIR = ROOT / "paper" / "tables"
FIG_DIR = ROOT / "paper" / "figs"


METHOD_LABELS = {
    "dummy_most_frequent": "Most frequent",
    "logistic_l2": "L2 logistic",
    "nslr_string_700": "Fixed STRING-700",
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
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, metrics in result["aggregate_metrics"].items():
        rows.append(
            {
                "method": method,
                "Method": METHOD_LABELS.get(method, method),
                "Balanced accuracy": _fmt(metrics["balanced_accuracy"]),
                "Macro-F1": _fmt(metrics["macro_f1"]),
                "MCC": _fmt(metrics["mcc"]),
                "AUROC OvR": _fmt(metrics["ovr_auroc"]),
                "AUPRC OvR": _fmt(metrics["ovr_auprc"]),
                "balanced_accuracy_mean": metrics["balanced_accuracy"]["mean"],
                "balanced_accuracy_low": metrics["balanced_accuracy"]["ci_low"],
                "balanced_accuracy_high": metrics["balanced_accuracy"]["ci_high"],
            }
        )

    df = pd.DataFrame(rows).sort_values("balanced_accuracy_mean", ascending=False)
    display_cols = ["Method", "Balanced accuracy", "Macro-F1", "MCC", "AUROC OvR", "AUPRC OvR"]
    df[display_cols].to_csv(TABLE_DIR / "table1_metabric_1000g.csv", index=False)
    (TABLE_DIR / "table1_metabric_1000g.md").write_text(_to_markdown(df[display_cols]))

    fig_df = df.iloc[::-1].copy()
    y = range(len(fig_df))
    x = fig_df["balanced_accuracy_mean"].to_numpy()
    xerr = [
        x - fig_df["balanced_accuracy_low"].to_numpy(),
        fig_df["balanced_accuracy_high"].to_numpy() - x,
    ]

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ax.barh(y, x, color=["#999999", "#0072B2", "#009E73", "#D55E00"][: len(fig_df)])
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="black", capsize=3, linewidth=0.8)
    ax.set_yticks(list(y), fig_df["Method"].tolist())
    ax.set_xlabel("Balanced accuracy (bootstrap 95% CI)")
    ax.set_xlim(0.0, 0.8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("METABRIC PAM50, 1,000 train-selected genes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_metabric_1000g_balanced_accuracy.pdf")
    fig.savefig(FIG_DIR / "fig_metabric_1000g_balanced_accuracy.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
