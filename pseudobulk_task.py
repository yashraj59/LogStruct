"""
Donor-level phenotype prediction from pseudobulk, with a prior ablation.

This is the setting where a network prior should help most: few samples
(about 100 donors), thousands of genes, and signal spread over pathways.

Data: Stephenson et al. 2021, COVID-19 PBMC (CELLxGENE). Raw counts are
summed per donor and cell type, then log-CPM normalized.

Tasks:
    covid     COVID-19 vs healthy donors
    severity  severe/critical vs asymptomatic/mild/moderate, COVID-19 donors only

For each cell type the same donor-level cross-validation is run for:

    real prior      STRING PPI
    shuffled prior  degree-preserving rewiring of STRING
    no prior        no edges (plain elastic-net logistic regression)
    sklearn LR      reference

Hyperparameters are picked inside each training fold (inner CV), with the
same grid for the real and the shuffled prior. Genes are selected on the
training donors only.

Usage:
    python pseudobulk_task.py --task covid
    python pseudobulk_task.py --task severity --cell_types CD14 CD4 all
"""

from __future__ import annotations

import argparse
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ablation_prior import DATA_DIR, _download, load_string_edges, string_prior
from logstruct import LogStructClassifier, priors

warnings.filterwarnings("ignore")

COVID_URL = "https://datasets.cellxgene.cziscience.com/fe2e847c-1602-4f1b-86a4-112e4dc7a8e3.h5ad"
CELL_TYPE_COL = "initial_clustering"
STATUS_COL = "Status_on_day_collection_summary"
DEFAULT_CELL_TYPES = ["all", "CD14", "CD4", "CD8", "NK_16hi", "B_cell"]
CONDITIONS = ["real prior", "shuffled prior", "no prior", "sklearn LR"]

TASKS = {
    "covid": {
        "positive": ["Asymptomatic", "Mild", "Moderate", "Severe", "Critical"],
        "negative": ["Healthy"],
    },
    "severity": {
        "positive": ["Severe", "Critical"],
        "negative": ["Asymptomatic", "Mild", "Moderate"],
    },
}


# --------------------------------------------------------------------------
# Pseudobulk
# --------------------------------------------------------------------------

def build_pseudobulk(min_cells: int = 20, chunk: int = 50000):
    """
    Sum raw counts per (sample, cell type). Cached on disk.

    Returns
    -------
    counts : float32 array (n_groups, n_genes)
    meta : DataFrame with one row per group (sample, donor, cell type, status, n_cells)
    genes : list of gene symbols
    """
    cache_counts = DATA_DIR / "covid_pseudobulk_counts.npy"
    cache_meta = DATA_DIR / "covid_pseudobulk_meta.csv"
    cache_genes = DATA_DIR / "covid_pseudobulk_genes.txt"
    if cache_counts.exists() and cache_meta.exists() and cache_genes.exists():
        return (np.load(cache_counts), pd.read_csv(cache_meta),
                cache_genes.read_text().split("\n"))

    import scanpy as sc
    from scipy import sparse as sp

    path = _download(COVID_URL, DATA_DIR / "covid_stephenson.h5ad")
    adata = sc.read_h5ad(path, backed="r")
    source = adata.raw if adata.raw is not None else adata
    genes = adata.var["feature_name"].astype(str).tolist()
    obs = adata.obs[["sample_id", "donor_id", CELL_TYPE_COL, STATUS_COL, "Site", "sex"]].astype(str)
    print(f"  {adata.n_obs:,} cells, {obs['sample_id'].nunique()} samples, "
          f"{obs['donor_id'].nunique()} donors")

    # Two groupings: per cell type, and all cells of a sample together.
    keys = pd.concat([
        obs["sample_id"] + "|" + obs[CELL_TYPE_COL],
        obs["sample_id"] + "|all",
    ])
    codes, uniques = pd.factorize(keys)
    codes_ct, codes_all = codes[: len(obs)], codes[len(obs):]
    n_groups = len(uniques)

    counts = np.zeros((n_groups, len(genes)), dtype=np.float64)
    for start in range(0, adata.n_obs, chunk):
        stop = min(start + chunk, adata.n_obs)
        X = sp.csr_matrix(source.X[start:stop])
        rows = np.arange(stop - start)
        for group_codes in (codes_ct[start:stop], codes_all[start:stop]):
            indicator = sp.csr_matrix(
                (np.ones(stop - start), (group_codes, rows)), shape=(n_groups, stop - start)
            )
            counts += (indicator @ X).toarray()
        print(f"    pseudobulk {stop:,}/{adata.n_obs:,}")

    n_cells = np.bincount(codes, minlength=n_groups)
    meta = pd.DataFrame({
        "sample_id": [u.split("|")[0] for u in uniques],
        "cell_type": [u.split("|")[1] for u in uniques],
        "n_cells": n_cells,
    })
    per_sample = obs.drop_duplicates("sample_id").set_index("sample_id")
    for col in ["donor_id", STATUS_COL, "Site", "sex"]:
        meta[col] = meta["sample_id"].map(per_sample[col])

    keep = (meta["n_cells"] >= min_cells).values
    counts, meta = counts[keep].astype(np.float32), meta[keep].reset_index(drop=True)

    np.save(cache_counts, counts)
    meta.to_csv(cache_meta, index=False)
    cache_genes.write_text("\n".join(genes))
    return counts, meta, genes


def task_data(counts, meta, genes, task: str, cell_type: str):
    """log-CPM matrix and labels for one task and cell type, one sample per donor."""
    spec = TASKS[task]
    sel = meta["cell_type"].eq(cell_type) & meta[STATUS_COL].isin(spec["positive"] + spec["negative"])
    m = meta[sel]
    # One sample per donor (the one with most cells), so donors are independent.
    m = m.sort_values("n_cells", ascending=False).drop_duplicates("donor_id").sort_values("donor_id")
    C = counts[m.index.values]

    cpm = C / C.sum(axis=1, keepdims=True) * 1e6
    X = np.log1p(cpm).astype(np.float32)
    y = m[STATUS_COL].isin(spec["positive"]).astype(int).values

    # Drop genes that are almost never detected.
    expressed = (C > 0).mean(axis=0) >= 0.5
    return X[:, expressed], y, [g for g, e in zip(genes, expressed) if e], m.reset_index(drop=True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def make_logstruct(prior, params: dict, args, seed: int) -> LogStructClassifier:
    return LogStructClassifier(
        prior_adjacency=prior,
        alpha=args.alpha,
        lambda_kl=args.lambda_kl,
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=seed,
        device="cpu",
        **params,
    )


def tune_and_fit(prior, grid: list[dict], X, y, args, seed: int):
    """Pick the best grid point by inner-CV AUROC, then refit on all training donors."""
    if len(grid) > 1:
        inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=seed)
        scores = np.zeros(len(grid))
        for tr, va in inner.split(X, y):
            for k, params in enumerate(grid):
                clf = make_logstruct(prior, params, args, seed).fit(X[tr], y[tr])
                scores[k] += roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1])
        best = grid[int(np.argmax(scores))]
    else:
        best = grid[0]
    return make_logstruct(prior, best, args, seed).fit(X, y), best


def select_genes(X_train: np.ndarray, n_genes: int) -> np.ndarray:
    """Most variable genes on the training donors (no labels used)."""
    return np.sort(np.argsort(X_train.var(axis=0))[::-1][:n_genes])


def run_fold(fold_id, train, test, X, y, genes, string_edges, args):
    torch.set_num_threads(1)
    seed = fold_id
    keep = select_genes(X[train], args.n_genes)
    X_train, X_test = X[train][:, keep], X[test][:, keep]
    y_train, y_test = y[train], y[test]
    fold_genes = [genes[i] for i in keep]

    prior = string_prior(string_edges, fold_genes)
    n_edges = int((np.triu(prior, 1) > 0.1).sum())

    plain_grid = [{"lambda_en": en} for en in args.lambda_en]
    graph_grid = [
        {"lambda_en": en, "lambda_smooth": sm, "neighbor_weight": nw}
        for en, sm, nw in product(args.lambda_en, args.lambda_smooth, args.neighbor_weight)
    ]

    rows = []

    def add(condition, proba, extra=None, shuffle_id=0):
        rows.append({
            "fold": fold_id, "condition": condition, "shuffle_id": shuffle_id,
            "auroc": roc_auc_score(y_test, proba),
            "balanced_accuracy": balanced_accuracy_score(y_test, proba > 0.5),
            "n_edges": n_edges, **(extra or {}),
        })

    def logstruct(condition, p, grid, shuffle_id=0):
        clf, best = tune_and_fit(p, grid, X_train, y_train, args, seed)
        add(condition, clf.predict_proba(X_test)[:, 1],
            {**best, "frac_zero_coef": float(np.mean(clf.coef_ == 0))}, shuffle_id)

    logstruct("real prior", prior, graph_grid)
    for k in range(args.n_shuffles):
        shuffled = priors.shuffle_degree_preserving(prior, random_state=1000 * seed + k)
        logstruct("shuffled prior", shuffled, graph_grid, k)
    logstruct("no prior", priors.from_identity(len(keep)), plain_grid)

    # sklearn reference with the same inner CV over C = 1 / lambda_en
    inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=seed)
    best_c, best_score = None, -np.inf
    for en in args.lambda_en:
        s = 0.0
        for tr, va in inner.split(X_train, y_train):
            sk = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0 / en, class_weight="balanced", max_iter=2000))
            sk.fit(X_train[tr], y_train[tr])
            s += roc_auc_score(y_train[va], sk.predict_proba(X_train[va])[:, 1])
        if s > best_score:
            best_c, best_score = 1.0 / en, s
    sk = make_pipeline(StandardScaler(), LogisticRegression(
        C=best_c, class_weight="balanced", max_iter=2000)).fit(X_train, y_train)
    add("sklearn LR", sk.predict_proba(X_test)[:, 1], {"lambda_en": 1.0 / best_c})
    return rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def summarize(runs: pd.DataFrame, output_dir: Path, tag: str, n_folds: int) -> None:
    metrics = ["auroc", "balanced_accuracy"]
    runs = runs.copy()
    runs["repeat"] = runs["fold"] // n_folds
    # Average shuffles inside a fold, then folds inside a repeat: one value per repeat.
    per_fold = runs.groupby(["cell_type", "repeat", "fold", "condition"])[metrics].mean()
    per_repeat = per_fold.groupby(["cell_type", "repeat", "condition"]).mean().reset_index()

    summary = (
        per_repeat.groupby(["cell_type", "condition"])[metrics].agg(["mean", "std"])
        .reindex(CONDITIONS, level="condition")
    )
    summary.to_csv(output_dir / f"{tag}_summary.csv")
    print("\n" + "=" * 78)
    print("SUMMARY (mean +- std over CV repeats)")
    print("=" * 78)
    print(summary.round(3).to_string())

    wide = per_repeat.pivot_table(index=["cell_type", "repeat"], columns="condition", values="auroc")
    rows = []
    for other in ["shuffled prior", "no prior"]:
        diff = wide["real prior"] - wide[other]
        for cell_type, d in diff.groupby(level="cell_type"):
            rows.append({
                "cell_type": cell_type, "comparison": f"real - {other}",
                "mean_auroc_diff": d.mean(), "std_diff": d.std(ddof=1),
                "repeats_real_wins": int((d > 0).sum()), "n_repeats": len(d),
            })
        per_rep = diff.groupby(level="repeat").mean()
        rows.append({
            "cell_type": "MEAN OVER CELL TYPES", "comparison": f"real - {other}",
            "mean_auroc_diff": per_rep.mean(), "std_diff": per_rep.std(ddof=1),
            "repeats_real_wins": int((per_rep > 0).sum()), "n_repeats": len(per_rep),
        })
    paired = pd.DataFrame(rows)
    paired.to_csv(output_dir / f"{tag}_paired.csv", index=False)
    print("\n" + "=" * 78)
    print("PAIRED AUROC DIFFERENCES (same donors in every fold)")
    print("=" * 78)
    print(paired.round(4).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cell_types = list(dict.fromkeys(runs["cell_type"]))
        fig, ax = plt.subplots(figsize=(1.6 * len(cell_types) + 3, 4))
        width = 0.2
        for k, condition in enumerate(CONDITIONS):
            d = per_repeat[per_repeat["condition"] == condition].groupby("cell_type")["auroc"]
            mean, std = d.mean().reindex(cell_types), d.std().reindex(cell_types)
            ax.bar(np.arange(len(cell_types)) + (k - 1.5) * width, mean, width,
                   yerr=std, capsize=2, label=condition)
        ax.set_xticks(np.arange(len(cell_types)))
        ax.set_xticklabels(cell_types)
        ax.set_ylabel("AUROC (held-out donors)")
        ax.set_ylim(max(0.0, per_repeat["auroc"].min() - 0.1), 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{tag}_auroc.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed, skipping plot")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Donor-level pseudobulk prior ablation")
    parser.add_argument("--task", default="covid", choices=list(TASKS))
    parser.add_argument("--cell_types", nargs="+", default=DEFAULT_CELL_TYPES)
    parser.add_argument("--n_genes", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    parser.add_argument("--n_shuffles", type=int, default=2)
    parser.add_argument("--string_score", type=int, default=700)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_en", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    parser.add_argument("--lambda_smooth", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--neighbor_weight", type=float, nargs="+", default=[1.0])
    parser.add_argument("--lambda_kl", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--output_dir", default="./pseudobulk_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"Donor-level pseudobulk task: {args.task}")
    print("=" * 78)
    counts, meta, genes = build_pseudobulk()
    string_edges = load_string_edges(args.string_score)

    all_rows = []
    for cell_type in args.cell_types:
        X, y, ct_genes, m = task_data(counts, meta, genes, args.task, cell_type)
        print(f"\n{cell_type}: {len(y)} donors ({y.sum()} positive, {len(y) - y.sum()} negative), "
              f"{len(ct_genes):,} expressed genes")
        cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=0)
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_fold)(k, tr, te, X, y, ct_genes, string_edges, args)
            for k, (tr, te) in enumerate(cv.split(X, y))
        )
        rows = [dict(r, cell_type=cell_type, n_donors=len(y)) for fold in results for r in fold]
        all_rows += rows
        d = pd.DataFrame(rows).groupby("condition")["auroc"].mean().reindex(CONDITIONS)
        print("  AUROC  " + "  ".join(f"{c}={v:.3f}" for c, v in d.items()))
        pd.DataFrame(all_rows).to_csv(output_dir / f"{args.task}_runs.csv", index=False)

    summarize(pd.DataFrame(all_rows), output_dir, args.task, args.folds)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
