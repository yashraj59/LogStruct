"""
Prior ablation for LogStruct.

Question: does the biological prior carry real signal, or does any graph
with the same shape do the same job?

Each run trains the same model with the same settings and only changes the graph:

    real prior      the prior as given (STRING PPI, or the true modules in synthetic mode)
    shuffled prior  degree-preserving rewiring of the real prior (hubs stay hubs)
    no prior        no edges, so the model is a plain elastic-net linear model

A plain sklearn LogisticRegression is added as a reference.

If "real" is not better than "shuffled", the biology in the prior is not
what helps. If "real" is not better than "no prior", the graph does not help at all.

Usage:
    python ablation_prior.py --dataset synthetic
    python ablation_prior.py --dataset blood --max_cells 20000 --n_genes 2000 \
        --train_sizes 1000 5000 0 --seeds 3

Real data is split by donor when a donor column exists, so test cells come
from donors the model has never seen.
"""

from __future__ import annotations

import argparse
import gzip
import urllib.request
import warnings
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from logstruct import LogStructClassifier, priors

warnings.filterwarnings("ignore")

DATA_DIR = Path("./benchmark_data")

DATASET_URLS = {
    "blood": "https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Blood/Blood.h5ad",
    "liver": "https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Liver/Liver.h5ad",
    "lung": "https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Lung/Lung.h5ad",
    "kidney": "https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Kidney/Kidney.h5ad",
}
STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL = (
    "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
)

CONDITIONS = ["real prior", "shuffled prior", "no prior", "sklearn LR"]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_string_edges(score_threshold: int) -> pd.DataFrame:
    """STRING human PPI as a table of (gene_a, gene_b, weight) with gene symbols."""
    links_path = _download(STRING_LINKS_URL, DATA_DIR / "9606.protein.links.v12.0.txt.gz")
    info_path = _download(STRING_INFO_URL, DATA_DIR / "9606.protein.info.v12.0.txt.gz")

    info = pd.read_csv(info_path, sep="\t", usecols=[0, 1])
    info.columns = ["protein", "symbol"]
    id_to_symbol = dict(zip(info["protein"], info["symbol"].str.upper()))

    links = pd.read_csv(links_path, sep=" ")
    links = links[links["combined_score"] >= score_threshold]
    return pd.DataFrame({
        "a": links["protein1"].map(id_to_symbol),
        "b": links["protein2"].map(id_to_symbol),
        "w": links["combined_score"] / 1000.0,
    }).dropna()


def string_prior(string_edges: pd.DataFrame, gene_names: list[str]) -> np.ndarray:
    idx = {g.upper(): i for i, g in enumerate(gene_names)}
    i = string_edges["a"].map(idx)
    j = string_edges["b"].map(idx)
    keep = i.notna() & j.notna()
    i, j = i[keep].astype(int).values, j[keep].astype(int).values
    w = string_edges["w"][keep].values.astype(np.float32)

    n = len(gene_names)
    adj = np.full((n, n), 0.05, dtype=np.float32)
    np.maximum.at(adj, (i, j), w)
    np.maximum.at(adj, (j, i), w)
    np.fill_diagonal(adj, 0.0)
    return adj


def load_atlas(name: str, max_cells: int, label_col: str, seed: int):
    """Load a CellTypist organ atlas (already log1p-normalized to 10k counts)."""
    import scanpy as sc

    path = _download(DATASET_URLS[name], DATA_DIR / f"{name}.h5ad")
    adata = sc.read_h5ad(path, backed="r")
    print(f"  {name}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    rng = np.random.default_rng(seed)
    if max_cells and adata.n_obs > max_cells:
        keep = np.sort(rng.choice(adata.n_obs, max_cells, replace=False))
        adata = adata[keep].to_memory()
    else:
        adata = adata.to_memory()

    counts = adata.obs[label_col].value_counts()
    adata = adata[adata.obs[label_col].isin(counts[counts >= 50].index)].copy()
    print(f"  Using {adata.n_obs:,} cells, {adata.obs[label_col].nunique()} cell types")
    return adata


def atlas_split(adata, label_col: str, donor_col: str, n_genes: int, seed: int):
    """Donor-held-out split. HVGs are picked on the training cells only."""
    import scanpy as sc

    y = adata.obs[label_col].astype(str).values
    if donor_col in adata.obs:
        groups = adata.obs[donor_col].astype(str).values
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, test_idx = next(splitter.split(y, y, groups))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.25, stratify=y, random_state=seed
        )

    # Keep classes with enough training cells, and test only on those classes.
    train_counts = pd.Series(y[train_idx]).value_counts()
    ok = set(train_counts[train_counts >= 20].index)
    train_idx = train_idx[np.isin(y[train_idx], list(ok))]
    test_idx = test_idx[np.isin(y[test_idx], list(ok))]

    train = adata[train_idx].copy()
    sc.pp.highly_variable_genes(train, n_top_genes=n_genes, flavor="seurat")
    hvg = train.var.highly_variable.values
    genes = adata.var_names[hvg].tolist()

    def dense(a):
        X = a[:, hvg].X
        return np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)

    return dense(train), y[train_idx], dense(adata[test_idx]), y[test_idx], genes


def make_synthetic(seed: int, n_train: int = 400, n_test: int = 3000, n_modules: int = 50,
                   module_size: int = 12, n_classes: int = 5, dropout: float = 0.5):
    """
    Synthetic single-cell-like data where the true graph is known.

    Genes come in modules. Each class shifts the activity of a few modules.
    Single genes are noisy and often dropped out, so averaging over the right
    module recovers the signal and averaging over the wrong genes does not.
    """
    rng = np.random.default_rng(seed)
    p = n_modules * module_size
    module = np.repeat(np.arange(n_modules), module_size)

    shifts = np.zeros((n_classes, n_modules), dtype=np.float32)
    for c in range(n_classes):
        on = rng.choice(n_modules, 3, replace=False)
        shifts[c, on] = rng.choice([-1.0, 1.0], 3) * 1.0

    def sample(n):
        y = rng.integers(0, n_classes, n)
        activity = shifts[y] + rng.standard_normal((n, n_modules)).astype(np.float32)
        X = activity[:, module] + 2.0 * rng.standard_normal((n, p)).astype(np.float32)
        X *= rng.random((n, p)) > dropout
        return X.astype(np.float32), y

    X_train, y_train = sample(n_train)
    X_test, y_test = sample(n_test)

    # Prior: 70% of the true within-module edges plus the same number of false edges.
    same = module[:, None] == module[None, :]
    iu, ju = np.triu_indices(p, k=1)
    true_pairs = np.flatnonzero(same[iu, ju])
    false_pairs = np.flatnonzero(~same[iu, ju])
    kept = rng.choice(true_pairs, int(0.7 * len(true_pairs)), replace=False)
    wrong = rng.choice(false_pairs, len(kept), replace=False)
    pairs = np.concatenate([kept, wrong])
    prior = priors.from_edge_list(p, list(zip(iu[pairs].tolist(), ju[pairs].tolist())))

    genes = [f"g{i}" for i in range(p)]
    return X_train, y_train, X_test, y_test, genes, prior


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def score(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def fit_logstruct(prior, X_train, y_train, X_test, y_test, args, seed) -> dict:
    clf = LogStructClassifier(
        prior_adjacency=prior,
        alpha=args.alpha,
        lambda_en=args.lambda_en,
        lambda_smooth=args.lambda_smooth,
        lambda_kl=args.lambda_kl,
        neighbor_weight=args.neighbor_weight,
        freeze_adjacency=args.freeze_adjacency,
        class_weight="balanced",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        n_iter_no_change=args.patience,
        random_state=seed,
    )
    start = time()
    clf.fit(X_train, y_train)
    out = score(y_test, clf.predict(X_test))
    out.update({
        "train_time": time() - start,
        "n_iter": clf.n_iter_,
        "n_edges": len(clf.edges_),
        "edge_drift": float(np.abs(clf.edge_weights_ - clf.edge_prior_).mean())
        if len(clf.edges_) else 0.0,
        "frac_zero_coef": float(np.mean(clf.coef_ == 0)),
    })
    return out


def fit_sklearn(X_train, y_train, X_test, y_test, args) -> dict:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0 / max(args.lambda_en, 1e-8), class_weight="balanced",
                           max_iter=300),
    )
    start = time()
    clf.fit(X_train, y_train)
    out = score(y_test, clf.predict(X_test))
    out["train_time"] = time() - start
    return out


def subsample_train(X, y, size: int, seed: int):
    if not size or size >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    # Keep at least 5 cells per class so every class can be learned and validated.
    keep = np.concatenate([
        rng.choice(np.flatnonzero(y == c), min(5, np.sum(y == c)), replace=False)
        for c in np.unique(y)
    ])
    rest = np.setdiff1d(np.arange(len(y)), keep)
    extra = rng.choice(rest, max(size - len(keep), 0), replace=False)
    idx = np.concatenate([keep, extra])
    return X[idx], y[idx]


def run_conditions(X_train, y_train, X_test, y_test, prior, args, seed, train_size) -> list[dict]:
    rows = []

    def add(condition, result, shuffle_id=0):
        rows.append({"seed": seed, "train_size": len(y_train), "condition": condition,
                     "shuffle_id": shuffle_id, **result})
        print(f"    {condition:<15} acc={result['accuracy']:.4f} "
              f"f1={result['f1_macro']:.4f} ({result['train_time']:.0f}s)")

    add("real prior", fit_logstruct(prior, X_train, y_train, X_test, y_test, args, seed))
    for k in range(args.n_shuffles):
        shuffled = priors.shuffle_degree_preserving(prior, random_state=1000 * seed + k)
        add("shuffled prior",
            fit_logstruct(shuffled, X_train, y_train, X_test, y_test, args, seed), k)
    no_prior = priors.from_identity(X_train.shape[1])
    add("no prior", fit_logstruct(no_prior, X_train, y_train, X_test, y_test, args, seed))
    if not args.skip_sklearn:
        add("sklearn LR", fit_sklearn(X_train, y_train, X_test, y_test, args))
    return rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def summarize(runs: pd.DataFrame, output_dir: Path, tag: str) -> None:
    metrics = ["accuracy", "f1_macro", "balanced_accuracy"]
    # Average the shuffles inside each seed first, so every condition has one value per seed.
    per_seed = runs.groupby(["train_size", "seed", "condition"])[metrics].mean().reset_index()

    summary = (
        per_seed.groupby(["train_size", "condition"])[metrics]
        .agg(["mean", "std"])
        .reindex(CONDITIONS, level="condition")
    )
    summary.to_csv(output_dir / f"{tag}_summary.csv")
    print("\n" + "=" * 78)
    print("SUMMARY (mean +- std over seeds)")
    print("=" * 78)
    print(summary.round(4).to_string())

    wide = per_seed.pivot_table(index=["train_size", "seed"], columns="condition", values=metrics)
    rows = []
    for metric in metrics:
        for other in ["shuffled prior", "no prior"]:
            if other not in wide[metric]:
                continue
            diff = wide[metric]["real prior"] - wide[metric][other]
            for size, d in diff.groupby(level="train_size"):
                rows.append({
                    "train_size": size, "metric": metric, "comparison": f"real - {other}",
                    "mean_diff": d.mean(), "std_diff": d.std(ddof=1) if len(d) > 1 else np.nan,
                    "seeds_real_wins": int((d > 0).sum()), "n_seeds": len(d),
                })
    paired = pd.DataFrame(rows)
    paired.to_csv(output_dir / f"{tag}_paired.csv", index=False)
    print("\n" + "=" * 78)
    print("PAIRED DIFFERENCES (same seed, same split, same cells)")
    print("=" * 78)
    print(paired.round(4).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes = sorted(per_seed["train_size"].unique())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ["accuracy", "f1_macro"]):
            for condition in CONDITIONS:
                d = per_seed[per_seed["condition"] == condition]
                if d.empty:
                    continue
                g = d.groupby("train_size")[metric].agg(["mean", "std"]).reindex(sizes)
                ax.errorbar(range(len(sizes)), g["mean"], yerr=g["std"].fillna(0),
                            marker="o", capsize=3, label=condition)
            ax.set_xticks(range(len(sizes)))
            ax.set_xticklabels([f"{s:,}" for s in sizes])
            ax.set_xlabel("training cells")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle(f"Prior ablation ({tag})")
        fig.tight_layout()
        fig.savefig(output_dir / f"{tag}_ablation.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed, skipping plot")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LogStruct prior ablation")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic"] + list(DATASET_URLS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n_shuffles", type=int, default=2,
                        help="Shuffled priors per seed (averaged)")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[0],
                        help="Training set sizes, 0 = all training cells")
    parser.add_argument("--max_cells", type=int, default=20000)
    parser.add_argument("--n_genes", type=int, default=2000)
    parser.add_argument("--label_col", default="cell_type")
    parser.add_argument("--donor_col", default="donor_id")
    parser.add_argument("--string_score", type=int, default=700)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambda_en", type=float, default=1.0)
    parser.add_argument("--lambda_smooth", type=float, default=1.0)
    parser.add_argument("--lambda_kl", type=float, default=1.0)
    parser.add_argument("--neighbor_weight", type=float, default=1.0)
    parser.add_argument("--freeze_adjacency", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--skip_sklearn", action="store_true")
    parser.add_argument("--tag", default=None, help="Output file prefix")
    parser.add_argument("--output_dir", default="./ablation_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    tag = args.tag or args.dataset

    print("=" * 78)
    print(f"LogStruct prior ablation: {args.dataset}")
    print("=" * 78)

    adata, string_edges = None, None
    if args.dataset != "synthetic":
        adata = load_atlas(args.dataset, args.max_cells, args.label_col, seed=0)
        print("  Loading STRING...")
        string_edges = load_string_edges(args.string_score)

    rows = []
    for seed in range(args.seeds):
        if args.dataset == "synthetic":
            X_train, y_train, X_test, y_test, genes, prior = make_synthetic(seed)
        else:
            X_train, y_train, X_test, y_test, genes = atlas_split(
                adata, args.label_col, args.donor_col, args.n_genes, seed
            )
            prior = string_prior(string_edges, genes)

        stats = priors.summarize(np.where(prior > 0.1, prior, 0.0))
        print(f"\nSeed {seed}: {len(y_train):,} train / {len(y_test):,} test cells, "
              f"{len(genes)} genes, {stats['n_edges']:,} prior edges")

        for size in args.train_sizes:
            X_sub, y_sub = subsample_train(X_train, y_train, size, seed)
            print(f"  train size {len(y_sub):,}")
            rows += run_conditions(X_sub, y_sub, X_test, y_test, prior, args, seed, size)
            pd.DataFrame(rows).to_csv(output_dir / f"{tag}_runs.csv", index=False)

    summarize(pd.DataFrame(rows), output_dir, tag)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
