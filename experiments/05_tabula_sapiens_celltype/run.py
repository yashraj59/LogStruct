"""Tabula Sapiens v2 immune-compartment cell-type classification.

This runner keeps the full 60k-gene CELLxGENE matrix sparse until each outer
fold selects genes by variance on training donors only. It is separate from the
dense tabular runner for that reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.shared.data_loader import DATA_SOURCES, PRIORS, require_raw  # noqa: E402
from experiments.shared.eval import (  # noqa: E402
    classification_metrics,
    environment_record,
    git_sha,
    summarize_outer_folds,
)
from experiments.shared.priors import build_common_priors  # noqa: E402
from logstruct import LogStructClassifier  # noqa: E402


METHODS = ["dummy_most_frequent", "logistic_l2", "logstruct_identity", "logstruct_string_700"]


def _load_raw_obs_and_matrix():
    paths = [
        DATA_SOURCES["tabula_sapiens_v2_blood"].raw_path,
        DATA_SOURCES["tabula_sapiens_v2_spleen"].raw_path,
        DATA_SOURCES["tabula_sapiens_v2_lymph_node"].raw_path,
    ]
    require_raw(paths)
    obs_frames = []
    matrices = []
    gene_names = None
    for path in paths:
        adata = ad.read_h5ad(path)
        if "feature_name" in adata.var:
            current_genes = adata.var["feature_name"].astype(str).to_numpy()
        else:
            current_genes = adata.var_names.astype(str).to_numpy()
        if gene_names is None:
            gene_names = current_genes
        elif not np.array_equal(gene_names, current_genes):
            raise ValueError(f"Gene order differs in {path}")
        obs = adata.obs[["donor_id", "cell_type", "broad_cell_class"]].copy()
        obs["source_file"] = path.name
        obs_frames.append(obs)
        X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
        matrices.append(X.astype(np.float32))
    return sparse.vstack(matrices, format="csr"), pd.concat(obs_frames, ignore_index=True), gene_names


def _subsample_by_class(obs, *, min_cells: int, max_cells_per_class: int | None, seed: int):
    rng = np.random.default_rng(seed)
    counts = obs["cell_type"].value_counts()
    keep_classes = counts[counts >= min_cells].index
    keep_indices = []
    for label in keep_classes:
        idx = np.flatnonzero((obs["cell_type"] == label).to_numpy())
        if max_cells_per_class is not None and len(idx) > max_cells_per_class:
            idx = rng.choice(idx, size=max_cells_per_class, replace=False)
        keep_indices.extend(idx.tolist())
    keep_indices = np.asarray(sorted(keep_indices), dtype=int)
    return keep_indices


def _sparse_variance(X):
    X = X.tocsr()
    mean = np.asarray(X.mean(axis=0)).ravel()
    mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    return mean_sq - mean * mean


def _select_top_variable_train_genes(X, train_idx, gene_names, n_genes):
    var = _sparse_variance(X[train_idx])
    keep = np.argsort(var)[-min(n_genes, X.shape[1]) :]
    keep = np.sort(keep)
    return keep, np.asarray(gene_names, dtype=str)[keep].tolist()


def _dense_selected(X, rows, cols):
    return X[rows][:, cols].toarray().astype(np.float32)


def _tune_logistic(X, y, groups, train_idx, *, seed: int):
    unique_groups = np.unique(groups[train_idx])
    n_splits = min(5, len(unique_groups))
    grid = [0.03, 0.1, 0.3, 1.0, 3.0]
    best_c = grid[0]
    best_score = -np.inf
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for c in grid:
        scores = []
        for tr_rel, va_rel in splitter.split(X[train_idx], y[train_idx], groups[train_idx]):
            tr = train_idx[tr_rel]
            va = train_idx[va_rel]
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c,
                    max_iter=3000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
            model.fit(X[tr], y[tr])
            scores.append(balanced_accuracy_score(y[va], model.predict(X[va])))
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_c = c
    return best_c, best_score


def _load_prior(path: Path):
    arr = np.load(path, allow_pickle=True)
    meta = json.loads(str(arr["metadata"])) if "metadata" in arr else {}
    return arr["adjacency"], arr["gene_names"].astype(str).tolist(), meta


def _fold_priors(fold_id: int, genes: list[str], seed: int, required: set[str]):
    out_dir = PRIORS / "tabula_sapiens_immune" / f"fold_{fold_id:02d}_{len(genes)}g"
    paths = {key: out_dir / f"{key}.npz" for key in required}
    if all(path.exists() for path in paths.values()):
        return paths
    return build_common_priors(
        genes,
        out_dir,
        string_links=DATA_SOURCES["string_links_9606_v12"].raw_path,
        string_aliases=DATA_SOURCES["string_aliases_9606_v12"].raw_path,
        seed=seed,
        include=required,
    )


def _save_predictions(path, y_true, y_pred, y_proba, classes, genes, donors):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "y_true": np.asarray(y_true, dtype=str),
        "y_pred": np.asarray(y_pred, dtype=str),
        "classes": np.asarray(classes, dtype=str),
        "gene_names": np.asarray(genes, dtype=str),
        "donor_id": np.asarray(donors, dtype=str),
    }
    if y_proba is not None:
        payload["y_proba"] = np.asarray(y_proba, dtype=np.float32)
    np.savez_compressed(path, **payload)


def _save_logstruct(path, model, genes, prior_meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gene_names=np.asarray(genes, dtype=str),
        classes=np.asarray(model.classes_, dtype=str),
        effective_coef=np.asarray(model.effective_coef_, dtype=np.float32),
        adjacency_raw=np.asarray(model.adjacency_raw_, dtype=np.float32),
        prior_metadata=json.dumps(prior_meta, sort_keys=True),
    )


def run(args):
    cfg = yaml.safe_load((Path(__file__).with_name("config.yaml")).read_text())
    seed = int(cfg.get("seed", 0))
    started = time.perf_counter()
    X_sparse, obs, gene_names = _load_raw_obs_and_matrix()
    keep_cells = _subsample_by_class(
        obs,
        min_cells=args.min_cells,
        max_cells_per_class=args.max_cells_per_class,
        seed=seed,
    )
    X_sparse = X_sparse[keep_cells].tocsr()
    obs = obs.iloc[keep_cells].reset_index(drop=True)
    y = obs["cell_type"].astype(str).to_numpy()
    groups = obs["donor_id"].astype(str).to_numpy()

    out_dir = ROOT / "results" / "tabula_sapiens_immune"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset_name": "tabula_sapiens_immune",
        "raw_shape": [int(X_sparse.shape[0]), int(X_sparse.shape[1])],
        "n_classes": int(len(np.unique(y))),
        "n_donors": int(len(np.unique(groups))),
        "min_cells_per_class": args.min_cells,
        "max_cells_per_class": args.max_cells_per_class,
        "preprocessing_note": "CELLxGENE X used as supplied; genes selected by train-donor variance per fold.",
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        "donor_counts": {str(k): int(v) for k, v in pd.Series(groups).value_counts().items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_records = []
    by_method = {method: [] for method in args.methods}
    required_priors = set()
    for method in args.methods:
        if method.endswith("identity"):
            required_priors.add("identity")
        if method.endswith("string_700"):
            required_priors.add("string_700")

    for fold_id, (train_idx, test_idx) in enumerate(outer.split(X_sparse, y, groups)):
        if args.max_outer is not None and fold_id >= args.max_outer:
            break
        keep_genes, selected_genes = _select_top_variable_train_genes(
            X_sparse, train_idx, gene_names, args.max_genes
        )
        X_selected = _dense_selected(X_sparse, np.arange(X_sparse.shape[0]), keep_genes)
        priors = _fold_priors(fold_id, selected_genes, seed + fold_id, required_priors)
        for method in args.methods:
            t0 = time.perf_counter()
            hyper = {"max_genes": len(selected_genes)}
            if method == "dummy_most_frequent":
                model = DummyClassifier(strategy="most_frequent")
                model.fit(X_selected[train_idx], y[train_idx])
            elif method == "logistic_l2":
                best_c, inner_score = _tune_logistic(
                    X_selected, y, groups, train_idx, seed=seed + fold_id
                )
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=best_c,
                        max_iter=3000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=seed + fold_id,
                    ),
                )
                model.fit(X_selected[train_idx], y[train_idx])
                hyper.update({"C": best_c, "inner_balanced_accuracy": inner_score})
            elif method.startswith("logstruct_"):
                prior_key = "identity" if method.endswith("identity") else "string_700"
                prior, prior_genes, prior_meta = _load_prior(priors[prior_key])
                if prior_genes != selected_genes:
                    raise ValueError("Prior gene order mismatch")
                model = LogStructClassifier(
                    prior_adjacency=prior,
                    alpha=0.5,
                    lambda_en=1e-3,
                    lambda_smooth=1e-3,
                    lambda_kl=1.0,
                    kl_reduction="offdiag_mean",
                    learning_rate=1e-2,
                    max_iter=120,
                    validation_fraction=0.15,
                    n_iter_no_change=12,
                    random_state=seed + fold_id,
                    device=args.device,
                    verbose=False,
                )
                model.fit(X_selected[train_idx], y[train_idx])
                hyper.update({"prior": prior_key, "prior_metadata": prior_meta})
                _save_logstruct(
                    out_dir / "model_artifacts" / f"{method}_fold{fold_id}.npz",
                    model,
                    selected_genes,
                    prior_meta,
                )
            else:
                raise ValueError(f"Unknown method {method}")

            y_pred = model.predict(X_selected[test_idx])
            y_proba = model.predict_proba(X_selected[test_idx]) if hasattr(model, "predict_proba") else None
            metrics = classification_metrics(y[test_idx], y_pred, y_proba)
            by_method[method].append(metrics)
            elapsed = time.perf_counter() - t0
            classes = getattr(model, "classes_", np.unique(y))
            _save_predictions(
                out_dir / "predictions" / f"{method}_fold{fold_id}.npz",
                y[test_idx],
                y_pred,
                y_proba,
                classes,
                selected_genes,
                groups[test_idx],
            )
            fold_records.append(
                {
                    "dataset_name": "tabula_sapiens_immune",
                    "method": method,
                    "fold": fold_id,
                    "seed": seed + fold_id,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "train_donors": sorted(np.unique(groups[train_idx]).tolist()),
                    "test_donors": sorted(np.unique(groups[test_idx]).tolist()),
                    "hyperparameters": hyper,
                    "metrics": metrics,
                    "wall_clock_sec": float(elapsed),
                    "device": args.device if method.startswith("logstruct_") else "cpu",
                    "git_sha": git_sha(),
                    "environment": environment_record(),
                }
            )
            print(
                f"tabula_sapiens_immune fold={fold_id} method={method} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    result = {
        "dataset_name": "tabula_sapiens_immune",
        "task": "single_cell_celltype_classification",
        "metadata": metadata,
        "methods": args.methods,
        "outer_folds_completed": max((r["fold"] for r in fold_records), default=-1) + 1,
        "seed": seed,
        "git_sha": git_sha(),
        "environment": environment_record(),
        "wall_clock_sec": float(time.perf_counter() - started),
        "aggregate_metrics": {
            method: summarize_outer_folds(rows, n_boot=1000, seed=seed)
            for method, rows in by_method.items()
            if rows
        },
        "fold_records": fold_records,
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-outer", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=500)
    parser.add_argument("--min-cells", type=int, default=500)
    parser.add_argument("--max-cells-per-class", type=int, default=3000)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
