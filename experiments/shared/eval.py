"""Evaluation utilities shared by LogStruct paper experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def bootstrap_ci(
    values: np.ndarray,
    metric_fn: Callable[[np.ndarray], float] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap a scalar statistic over observed values."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    metric_fn = metric_fn or (lambda x: float(np.mean(x)))

    point = metric_fn(values)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        boot[i] = metric_fn(values[idx])

    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float | None]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, float | None] = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "top5_accuracy": None,
        "ovr_auroc": None,
        "ovr_auprc": None,
    }

    if y_proba is not None:
        classes = np.unique(y_true)
        try:
            proba = np.asarray(y_proba)
            if proba.ndim == 2 and proba.shape[1] > 5 and proba.shape[1] == len(classes):
                top5 = np.argsort(proba, axis=1)[:, -5:]
                true_idx = np.searchsorted(classes, y_true)
                out["top5_accuracy"] = float(np.mean([t in row for t, row in zip(true_idx, top5)]))
            if len(classes) == 2:
                scores = y_proba[:, 1] if np.asarray(y_proba).ndim == 2 else y_proba
                out["ovr_auroc"] = float(roc_auc_score(y_true, scores))
                out["ovr_auprc"] = float(average_precision_score(y_true, scores))
            else:
                out["ovr_auroc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
                out["ovr_auprc"] = float(
                    average_precision_score(
                        np.eye(len(classes))[np.searchsorted(classes, y_true)],
                        y_proba,
                        average="macro",
                    )
                )
        except ValueError:
            pass

    return out


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "pearson_r": float(stats.pearsonr(y_true.ravel(), y_pred.ravel()).statistic),
        "spearman_rho": float(stats.spearmanr(y_true.ravel(), y_pred.ravel()).statistic),
    }


def summarize_outer_folds(
    fold_metrics: list[dict[str, float | None]], *, n_boot: int = 1000, seed: int = 0
) -> dict[str, dict[str, float] | None]:
    keys = sorted({k for row in fold_metrics for k in row})
    summary = {}
    for key in keys:
        vals = [row[key] for row in fold_metrics if row.get(key) is not None]
        if not vals:
            summary[key] = None
        else:
            summary[key] = bootstrap_ci(np.asarray(vals, dtype=float), n_boot=n_boot, seed=seed)
    return summary


def paired_bootstrap_delta(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 10000, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap on per-fold metric differences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap_delta requires aligned arrays")
    return bootstrap_ci(a - b, n_boot=n_boot, seed=seed)


@dataclass
class RunRecord:
    dataset_name: str
    method: str
    seed: int
    train_size: int
    val_size: int
    test_size: int
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    wall_clock_sec: float
    device: str
    git_sha: str
    environment: dict[str, str]


def environment_record() -> dict[str, str]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = str(torch.cuda.is_available())
    except Exception:
        torch_version = "unavailable"
        cuda_available = "unknown"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda_available": cuda_available,
    }


def write_run_record(record: RunRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n")


class Timer:
    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self.elapsed = time.perf_counter() - self.started
