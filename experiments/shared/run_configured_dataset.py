"""Strict entry point for configured real-data experiments."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.shared.data_loader import (  # noqa: E402
    DATA_SOURCES,
    PRIORS,
    load_dataset,
    prepare_gtex_tissue,
    prepare_metabric_pam50,
    prepare_norman_perturb_identity,
    prepare_uci_pancancer,
    require_raw,
    select_top_variable_genes,
)
from experiments.shared.eval import (  # noqa: E402
    classification_metrics,
    environment_record,
    git_sha,
    summarize_outer_folds,
)
from experiments.shared.priors import build_common_priors  # noqa: E402
from logstruct import LogStructClassifier  # noqa: E402


PREPARE_DATASET = {
    "gtex_tissue": prepare_gtex_tissue,
    "metabric_pam50": prepare_metabric_pam50,
    "norman_2019_perturb_identity": prepare_norman_perturb_identity,
    "uci_pancancer": prepare_uci_pancancer,
}

DEFAULT_METHODS = [
    "dummy_most_frequent",
    "logistic_l2",
    "elastic_net",
    "nslr_string_700",
    "logstruct_string_700",
    "logstruct_identity",
]


def _grid(params: dict[str, list[Any]]):
    if not params:
        yield {}
        return
    keys = list(params)
    for values in itertools.product(*(params[k] for k in keys)):
        yield dict(zip(keys, values))


def _set_pipeline_param(estimator, key: str, value: Any) -> None:
    if hasattr(estimator, "named_steps"):
        final = list(estimator.named_steps)[-1]
        estimator.set_params(**{f"{final}__{key}": value})
    else:
        estimator.set_params(**{key: value})


def _clone_with_params(estimator, params: dict[str, Any]):
    from sklearn.base import clone

    model = clone(estimator)
    for key, value in params.items():
        _set_pipeline_param(model, key, value)
    return model


def _tune_sklearn(
    estimator,
    grid: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    inner_splits: list[dict[str, list[int]]],
) -> tuple[dict[str, Any], float]:
    best_params: dict[str, Any] = {}
    best_score = -np.inf
    for params in _grid(grid):
        scores = []
        for split in inner_splits:
            train_idx = np.asarray(split["train_idx"])
            val_idx = np.asarray(split["val_idx"])
            model = _clone_with_params(estimator, params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(balanced_accuracy_score(y[val_idx], pred))
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score


def _load_prior(path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    arr = np.load(path, allow_pickle=True)
    metadata = json.loads(str(arr["metadata"])) if "metadata" in arr else {}
    return arr["adjacency"], arr["gene_names"].astype(str).tolist(), metadata


def _required_prior_keys(methods: list[str]) -> set[str]:
    keys = set()
    for method in methods:
        if method.startswith("logstruct_") or method.startswith("nslr_"):
            keys.add(_prior_key_for_method(method))
    return keys


def _ensure_fold_priors(
    dataset_name: str,
    fold_id: int,
    gene_names: list[str],
    *,
    seed: int,
    required_keys: set[str],
) -> dict[str, Path]:
    if not required_keys:
        return {}
    out_dir = PRIORS / dataset_name / f"fold_{fold_id:02d}_{len(gene_names)}g"
    expected = {
        "identity": out_dir / "identity.npz",
        "random": out_dir / "random.npz",
        "string_400": out_dir / "string_400.npz",
        "string_700": out_dir / "string_700.npz",
    }
    if required_keys and all(expected[key].exists() for key in required_keys if key in expected):
        return {k: v for k, v in expected.items() if v.exists()}

    return build_common_priors(
        gene_names,
        out_dir,
        string_links=DATA_SOURCES["string_links_9606_v12"].raw_path,
        string_aliases=DATA_SOURCES["string_aliases_9606_v12"].raw_path,
        seed=seed,
        include=required_keys,
    )


def _sklearn_specs(seed: int):
    return {
        "dummy_most_frequent": (
            DummyClassifier(strategy="most_frequent"),
            {},
        ),
        "logistic_l2": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
            {"C": [0.03, 0.1, 0.3, 1.0, 3.0]},
        ),
        "elastic_net": (
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=5000,
                    solver="saga",
                    penalty="elasticnet",
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
            {"C": [0.03, 0.1, 0.3, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
        ),
    }


def _logstruct_params(cfg: dict[str, Any], method: str, seed: int, device: str) -> dict[str, Any]:
    base = {
        "alpha": 0.5,
        "lambda_en": 1e-3,
        "lambda_smooth": 1e-3,
        "lambda_kl": 1.0,
        "kl_reduction": "offdiag_mean",
        "temperature": 0.8,
        "self_loop_weight": 1.0,
        "neighbor_weight": 1.0,
        "learning_rate": 1e-2,
        "max_iter": 120,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 12,
        "sparsity_threshold": 0.05,
        "random_state": seed,
        "device": device,
        "verbose": False,
    }
    base.update(cfg.get("logstruct", {}))
    if method.startswith("nslr_"):
        base["freeze_adjacency"] = True
        base["lambda_kl"] = 0.0
    else:
        base["freeze_adjacency"] = False
    return base


def _prior_key_for_method(method: str) -> str:
    if method.endswith("identity"):
        return "identity"
    if method.endswith("random"):
        return "random"
    if method.endswith("string_400"):
        return "string_400"
    if method.endswith("string_700"):
        return "string_700"
    raise ValueError(f"Method {method!r} does not encode a known prior")


def _predict_proba_or_none(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def _save_predictions(path: Path, y_true, y_pred, y_proba, classes, gene_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "y_true": np.asarray(y_true, dtype=str),
        "y_pred": np.asarray(y_pred, dtype=str),
        "classes": np.asarray(classes, dtype=str),
        "gene_names": np.asarray(gene_names, dtype=str),
    }
    if y_proba is not None:
        payload["y_proba"] = np.asarray(y_proba, dtype=np.float32)
    np.savez_compressed(path, **payload)


def _save_logstruct_artifact(path: Path, model: LogStructClassifier, gene_names, prior_metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gene_names=np.asarray(gene_names, dtype=str),
        classes=np.asarray(model.classes_, dtype=str),
        coef=np.asarray(model.coef_, dtype=np.float32),
        effective_coef=np.asarray(model.effective_coef_, dtype=np.float32),
        adjacency=np.asarray(model.adjacency_, dtype=np.float32),
        adjacency_raw=np.asarray(model.adjacency_raw_, dtype=np.float32),
        prior_metadata=json.dumps(prior_metadata, sort_keys=True),
    )


def run_experiment(
    cfg: dict[str, Any],
    *,
    methods: list[str] | None = None,
    max_outer: int | None = None,
    max_genes: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    dataset_name = cfg["dataset_name"]
    required_raw = [ROOT / p for p in cfg.get("required_raw", [])]
    if required_raw:
        require_raw(required_raw)

    if dataset_name in PREPARE_DATASET:
        PREPARE_DATASET[dataset_name](
            outer_folds=int(cfg.get("outer_folds", 5)),
            inner_folds=int(cfg.get("inner_folds", 5)),
            seed=int(cfg.get("seed", 0)),
        )

    X, y, gene_names, splits = load_dataset(dataset_name)
    gene_names = list(gene_names)
    seed = int(cfg.get("seed", 0))
    methods = methods or cfg.get("methods") or DEFAULT_METHODS
    n_genes = int(max_genes or cfg.get("gene_filter", {}).get("max_genes", 5000))
    device = device or cfg.get("device", "cuda")
    out_dir = ROOT / "results" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    fold_metrics_by_method: dict[str, list[dict[str, float | None]]] = {m: [] for m in methods}
    sklearn_specs = _sklearn_specs(seed)
    selected_splits = splits[: max_outer or len(splits)]
    required_prior_keys = _required_prior_keys(methods)

    for split in selected_splits:
        fold_id = int(split["outer_id"])
        train_val_idx = np.asarray(split["train_val_idx"])
        test_idx = np.asarray(split["test_idx"])
        keep, selected_genes = select_top_variable_genes(
            X[train_val_idx], gene_names, n_genes=n_genes
        )
        X_fold = X[:, keep]
        priors = _ensure_fold_priors(
            dataset_name,
            fold_id,
            selected_genes,
            seed=seed + fold_id,
            required_keys=required_prior_keys,
        )

        for method in methods:
            method_started = time.perf_counter()
            hyperparameters: dict[str, Any] = {"max_genes": len(selected_genes)}
            if method in sklearn_specs:
                estimator, grid = sklearn_specs[method]
                best_params, inner_score = _tune_sklearn(
                    estimator,
                    grid,
                    X_fold,
                    y,
                    split["inner"],
                )
                model = _clone_with_params(estimator, best_params)
                model.fit(X_fold[train_val_idx], y[train_val_idx])
                hyperparameters.update(
                    {
                        "grid": grid,
                        "selected": best_params,
                        "inner_balanced_accuracy": inner_score,
                    }
                )
            elif method.startswith("logstruct_") or method.startswith("nslr_"):
                prior_key = _prior_key_for_method(method)
                prior, prior_genes, prior_metadata = _load_prior(priors[prior_key])
                if prior_genes != selected_genes:
                    raise ValueError(f"Prior gene order mismatch for {method} fold {fold_id}")
                params = _logstruct_params(cfg, method, seed + fold_id, device)
                model = LogStructClassifier(prior_adjacency=prior, **params)
                model.fit(X_fold[train_val_idx], y[train_val_idx])
                hyperparameters.update(params)
                hyperparameters["prior"] = prior_key
                hyperparameters["prior_metadata"] = prior_metadata
                _save_logstruct_artifact(
                    out_dir / "model_artifacts" / f"{method}_fold{fold_id}.npz",
                    model,
                    selected_genes,
                    prior_metadata,
                )
            else:
                raise ValueError(f"Unknown method {method!r}")

            y_pred = model.predict(X_fold[test_idx])
            y_proba = _predict_proba_or_none(model, X_fold[test_idx])
            metrics = classification_metrics(y[test_idx], y_pred, y_proba)
            fold_metrics_by_method[method].append(metrics)
            elapsed = time.perf_counter() - method_started
            classes = getattr(model, "classes_", np.unique(y))
            _save_predictions(
                out_dir / "predictions" / f"{method}_fold{fold_id}.npz",
                y[test_idx],
                y_pred,
                y_proba,
                classes,
                selected_genes,
            )
            records.append(
                {
                    "dataset_name": dataset_name,
                    "method": method,
                    "fold": fold_id,
                    "seed": seed + fold_id,
                    "train_size": int(len(train_val_idx)),
                    "val_size": 0,
                    "test_size": int(len(test_idx)),
                    "hyperparameters": hyperparameters,
                    "metrics": metrics,
                    "wall_clock_sec": float(elapsed),
                    "device": device if method.startswith(("logstruct_", "nslr_")) else "cpu",
                    "git_sha": git_sha(),
                    "environment": environment_record(),
                }
            )
            print(
                f"{dataset_name} fold={fold_id} method={method} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    aggregate = {
        method: summarize_outer_folds(metrics, n_boot=1000, seed=seed)
        for method, metrics in fold_metrics_by_method.items()
        if metrics
    }
    result = {
        "dataset_name": dataset_name,
        "task": cfg.get("task"),
        "n_samples": int(X.shape[0]),
        "n_input_genes": int(X.shape[1]),
        "n_selected_genes": n_genes,
        "methods": methods,
        "outer_folds_requested": int(cfg.get("outer_folds", 5)),
        "outer_folds_completed": len(selected_splits),
        "seed": seed,
        "git_sha": git_sha(),
        "environment": environment_record(),
        "wall_clock_sec": float(time.perf_counter() - started),
        "aggregate_metrics": aggregate,
        "fold_records": records,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(config_path: str | Path, argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--max-outer", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    try:
        return run_experiment(
            cfg,
            methods=args.methods,
            max_outer=args.max_outer,
            max_genes=args.max_genes,
            device=args.device,
        )
    except FileNotFoundError as exc:
        blocker = {
            "dataset": cfg["dataset_name"],
            "status": "blocked",
            "reason": str(exc),
            "config": str(config_path),
        }
        out = ROOT / "results" / cfg["dataset_name"] / "BLOCKED.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(blocker, indent=2, sort_keys=True) + "\n")
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("config.yaml"), sys.argv[2:])
