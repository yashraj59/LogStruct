"""Dataset registry, download helpers, preprocessing, and split utilities."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
PRIORS = DATA / "priors"


@dataclass(frozen=True)
class DataSource:
    name: str
    url: str
    raw_path: Path
    notes: str = ""


DATA_SOURCES: dict[str, DataSource] = {
    "metabric_cbioportal": DataSource(
        "metabric_cbioportal",
        "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz",
        RAW / "brca_metabric.tar.gz",
        "METABRIC PAM50 from cBioPortal study brca_metabric.",
    ),
    "uci_pancancer": DataSource(
        "uci_pancancer",
        "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip",
        RAW / "uci_gene_expression_cancer_rna_seq.zip",
        "Fast 801-sample TCGA RNA-seq subset for pipeline shakedown.",
    ),
    "string_links_9606_v12": DataSource(
        "string_links_9606_v12",
        "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        RAW / "9606.protein.links.v12.0.txt.gz",
        "STRING v12 human links.",
    ),
    "string_aliases_9606_v12": DataSource(
        "string_aliases_9606_v12",
        "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
        RAW / "9606.protein.aliases.v12.0.txt.gz",
        "STRING v12 human aliases, required for gene-symbol priors.",
    ),
    "pbmc3k_10x": DataSource(
        "pbmc3k_10x",
        "http://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz",
        RAW / "pbmc3k_filtered_gene_bc_matrices.tar.gz",
        "Small single-cell shakedown dataset; not a substitute for Tabula Sapiens.",
    ),
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_source(name: str, *, overwrite: bool = False) -> Path:
    source = DATA_SOURCES[name]
    source.raw_path.parent.mkdir(parents=True, exist_ok=True)
    if source.raw_path.exists() and not overwrite:
        return source.raw_path
    urllib.request.urlretrieve(source.url, source.raw_path)
    return source.raw_path


def write_checksums(paths: Iterable[Path], out_path: str | Path = RAW / "CHECKSUMS.txt") -> None:
    lines = []
    for path in sorted(Path(p) for p in paths):
        if path.exists() and path.is_file():
            lines.append(f"{sha256_file(path)}  {path.relative_to(RAW)}")
    Path(out_path).write_text("\n".join(lines) + ("\n" if lines else ""))


def extract_archive(path: str | Path, dest: str | Path) -> Path:
    path = Path(path)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
    elif path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path) as tf:
            tf.extractall(dest)
    elif path.suffix == ".gz":
        out = dest / path.with_suffix("").name
        with gzip.open(path, "rb") as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        raise ValueError(f"Unsupported archive format: {path}")
    return dest


def create_nested_stratified_splits(
    y,
    *,
    outer_folds: int = 5,
    inner_folds: int = 5,
    seed: int = 0,
    save_path: str | Path | None = None,
):
    y = np.asarray(y)
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    splits = []
    for outer_id, (train_val_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), y)):
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + outer_id)
        inner_splits = []
        y_train_val = y[train_val_idx]
        for train_rel, val_rel in inner.split(np.zeros(len(y_train_val)), y_train_val):
            inner_splits.append(
                {
                    "train_idx": train_val_idx[train_rel].tolist(),
                    "val_idx": train_val_idx[val_rel].tolist(),
                }
            )
        splits.append(
            {
                "outer_id": outer_id,
                "train_val_idx": train_val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "inner": inner_splits,
            }
        )

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(json.dumps(splits, indent=2) + "\n")
    return splits


def standardize_train_only(X, train_idx, *others):
    scaler = StandardScaler().fit(X[train_idx])
    transformed = [scaler.transform(X)]
    transformed.extend(scaler.transform(x) for x in others)
    return (scaler, *transformed)


def select_top_variable_genes(X_train, gene_names, n_genes: int = 5000):
    gene_names = np.asarray(gene_names)
    variances = np.var(X_train, axis=0)
    keep = np.argsort(variances)[-min(n_genes, X_train.shape[1]) :]
    keep = np.sort(keep)
    return keep, gene_names[keep].tolist()


def require_raw(paths: Iterable[str | Path]) -> None:
    missing = [str(Path(p)) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Required raw data are missing. Run the matching downloader first. Missing: "
            + ", ".join(missing)
        )


def prepare_uci_pancancer(*, outer_folds: int = 5, inner_folds: int = 5, seed: int = 0):
    """Prepare the UCI TCGA pan-cancer shakedown subset.

    This dataset uses anonymized `gene_0` style feature names, so it is useful
    for pipeline testing but not for biological validation.
    """
    raw_zip = DATA_SOURCES["uci_pancancer"].raw_path
    require_raw([raw_zip])
    out_dir = PROCESSED / "uci_pancancer"
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(raw_zip) as zf:
        payload = zf.read("TCGA-PANCAN-HiSeq-801x20531.tar.gz")
    import io

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
        data_f = tf.extractfile("TCGA-PANCAN-HiSeq-801x20531/data.csv")
        labels_f = tf.extractfile("TCGA-PANCAN-HiSeq-801x20531/labels.csv")
        if data_f is None or labels_f is None:
            raise FileNotFoundError("UCI archive does not contain expected CSV files")
        data = pd.read_csv(data_f)
        labels = pd.read_csv(labels_f)

    X = data.drop(columns=[data.columns[0]]).to_numpy(dtype=np.float32)
    gene_names = data.columns[1:].to_numpy(dtype=str)
    y = labels["Class"].to_numpy(dtype=str)
    np.savez_compressed(out_dir / "dataset.npz", X=X, y=y, gene_names=gene_names)
    create_nested_stratified_splits(
        y,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        save_path=out_dir / "splits.json",
    )
    return out_dir


def load_dataset(name: str):
    """Load a processed dataset as (X, y, gene_names, splits).

    Full dataset-specific parsers are intentionally strict: they raise until the
    expected raw files are present and a parser has been implemented for that
    exact source. This prevents silent substitution of easier datasets.
    """
    processed = PROCESSED / name / "dataset.npz"
    splits = PROCESSED / name / "splits.json"
    if not processed.exists() or not splits.exists():
        raise FileNotFoundError(
            f"Processed dataset {name!r} is unavailable. Expected {processed} and {splits}."
        )
    arr = np.load(processed, allow_pickle=True)
    return arr["X"], arr["y"], arr["gene_names"].tolist(), json.loads(splits.read_text())
