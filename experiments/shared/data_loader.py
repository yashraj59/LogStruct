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
from scipy import sparse

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
        "https://datahub.assets.cbioportal.org/brca_metabric.tar.gz",
        RAW / "brca_metabric.tar.gz",
        "METABRIC PAM50 from cBioPortal study brca_metabric. "
        "The older S3 URL returned HTTP 403 on 2026-05-20.",
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
    "tabula_sapiens_v2_blood": DataSource(
        "tabula_sapiens_v2_blood",
        "https://datasets.cellxgene.cziscience.com/b225ee37-5e06-4e49-9c25-c3d7b5008dab.h5ad",
        RAW / "tabula_sapiens_v2" / "tabula_sapiens___blood.h5ad",
        "Tabula Sapiens v2 Blood H5AD discovered through CELLxGENE collection metadata.",
    ),
    "tabula_sapiens_v2_spleen": DataSource(
        "tabula_sapiens_v2_spleen",
        "https://datasets.cellxgene.cziscience.com/d1966cc6-4082-43ec-a633-72e56f7c8a9a.h5ad",
        RAW / "tabula_sapiens_v2" / "tabula_sapiens___spleen.h5ad",
        "Tabula Sapiens v2 Spleen H5AD discovered through CELLxGENE collection metadata.",
    ),
    "tabula_sapiens_v2_lymph_node": DataSource(
        "tabula_sapiens_v2_lymph_node",
        "https://datasets.cellxgene.cziscience.com/6ec56d10-a543-45ff-aca0-c70efe8decdf.h5ad",
        RAW / "tabula_sapiens_v2" / "tabula_sapiens___lymph_node.h5ad",
        "Tabula Sapiens v2 Lymph Node H5AD discovered through CELLxGENE collection metadata.",
    ),
    "norman_2019_figshare_h5ad": DataSource(
        "norman_2019_figshare_h5ad",
        "https://ndownloader.figshare.com/files/43390776",
        RAW / "norman_2019_adata.h5ad",
        "Norman et al. 2019 labeled Perturb-seq H5AD, Figshare article 24688110.",
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
    out_path = Path(out_path)
    lines = []
    for path in sorted(Path(p) for p in paths):
        if (
            path.exists()
            and path.is_file()
            and path.resolve() != out_path.resolve()
            and not path.name.endswith(".part")
            and not path.name.startswith(".")
            and ".git" not in path.parts
        ):
            lines.append(f"{sha256_file(path)}  {path.relative_to(RAW)}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


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
    if (out_dir / "dataset.npz").exists() and (out_dir / "splits.json").exists():
        return out_dir

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


def _read_metabric_member(tar_path: Path, member: str, **read_csv_kwargs) -> pd.DataFrame:
    with tarfile.open(tar_path) as tf:
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(f"{member} not found in {tar_path}")
        return pd.read_csv(f, sep="\t", **read_csv_kwargs)


def _collapse_duplicate_genes_by_variance(expr: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-variance probe row for duplicated gene symbols."""
    expr = expr.copy()
    expr["Hugo_Symbol"] = expr["Hugo_Symbol"].astype(str).str.strip()
    expr = expr[(expr["Hugo_Symbol"] != "") & (expr["Hugo_Symbol"].str.upper() != "NAN")]
    sample_cols = [c for c in expr.columns if c not in {"Hugo_Symbol", "Entrez_Gene_Id"}]
    values = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
    variances = values.var(axis=1, skipna=True)
    keep_idx = variances.groupby(expr["Hugo_Symbol"], sort=False).idxmax()
    collapsed = expr.loc[keep_idx].copy()
    collapsed[sample_cols] = values.loc[keep_idx].fillna(0.0)
    collapsed = collapsed.sort_values("Hugo_Symbol")
    return collapsed[["Hugo_Symbol", *sample_cols]]


def prepare_metabric_pam50(*, outer_folds: int = 5, inner_folds: int = 5, seed: int = 0):
    """Prepare METABRIC PAM50/claudin subtype classification from cBioPortal.

    The cBioPortal METABRIC archive exposes the label as `CLAUDIN_SUBTYPE`.
    For the requested 5-class PAM50-like task, this parser keeps LumA, LumB,
    Her2, Basal, and Normal, and excludes claudin-low, NC, and missing labels.
    """
    raw_tar = DATA_SOURCES["metabric_cbioportal"].raw_path
    require_raw([raw_tar])
    out_dir = PROCESSED / "metabric_pam50"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (
        (out_dir / "dataset.npz").exists()
        and (out_dir / "splits.json").exists()
        and (out_dir / "metadata.json").exists()
    ):
        return out_dir

    clinical = _read_metabric_member(
        raw_tar, "brca_metabric/data_clinical_patient.txt", comment="#"
    )
    labels = clinical[["PATIENT_ID", "CLAUDIN_SUBTYPE"]].dropna()
    label_order = ["LumA", "LumB", "Her2", "Basal", "Normal"]
    labels = labels[labels["CLAUDIN_SUBTYPE"].isin(label_order)].copy()

    expr = _read_metabric_member(
        raw_tar,
        "brca_metabric/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",
        low_memory=False,
    )
    expr = _collapse_duplicate_genes_by_variance(expr)

    sample_cols = [c for c in expr.columns if c != "Hugo_Symbol"]
    label_map = labels.set_index("PATIENT_ID")["CLAUDIN_SUBTYPE"]
    sample_ids = [s for s in sample_cols if s in label_map.index]
    if not sample_ids:
        raise ValueError("No overlapping METABRIC expression samples and clinical labels")

    X = expr[sample_ids].T.to_numpy(dtype=np.float32)
    gene_names = expr["Hugo_Symbol"].to_numpy(dtype=str)
    y = label_map.loc[sample_ids].to_numpy(dtype=str)

    # METABRIC z-scores are already normalized. Persist a strict metadata record
    # so downstream result logs can state exactly what was included/excluded.
    metadata = {
        "source": DATA_SOURCES["metabric_cbioportal"].url,
        "expression_member": "brca_metabric/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",
        "label_member": "brca_metabric/data_clinical_patient.txt",
        "label_column": "CLAUDIN_SUBTYPE",
        "kept_classes": label_order,
        "excluded_label_counts": {
            str(k): int(v)
            for k, v in clinical["CLAUDIN_SUBTYPE"]
            .fillna("missing")
            .value_counts()
            .items()
            if k not in label_order
        },
        "n_samples": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
    }

    np.savez_compressed(
        out_dir / "dataset.npz",
        X=X,
        y=y,
        gene_names=gene_names,
        sample_ids=np.asarray(sample_ids, dtype=str),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    create_nested_stratified_splits(
        y,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        save_path=out_dir / "splits.json",
    )
    return out_dir


def _single_perturbation_label(value: str) -> str | None:
    value = str(value)
    if value == "ctrl":
        return "ctrl"
    parts = value.split("+")
    if len(parts) != 2:
        return None
    non_ctrl = [p for p in parts if p.lower() != "ctrl"]
    if len(non_ctrl) == 1:
        return non_ctrl[0]
    return None


def prepare_norman_perturb_identity(
    *,
    outer_folds: int = 5,
    inner_folds: int = 5,
    seed: int = 0,
    min_cells_per_class: int = 50,
):
    """Prepare Norman 2019 single-gene perturbation identity classification."""
    raw_h5ad = DATA_SOURCES["norman_2019_figshare_h5ad"].raw_path
    require_raw([raw_h5ad])
    out_dir = PROCESSED / "norman_2019_perturb_identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (
        (out_dir / "dataset.npz").exists()
        and (out_dir / "splits.json").exists()
        and (out_dir / "metadata.json").exists()
    ):
        return out_dir

    import anndata as ad

    adata = ad.read_h5ad(raw_h5ad)
    if "guide_merged" not in adata.obs:
        raise ValueError("Expected Norman H5AD obs column 'guide_merged'")

    labels = adata.obs["guide_merged"].map(_single_perturbation_label)
    keep = labels.notna().to_numpy()
    labels = labels[keep].astype(str)
    counts = labels.value_counts()
    kept_classes = counts[counts >= min_cells_per_class].index
    keep2 = labels.isin(kept_classes).to_numpy()
    row_idx = np.flatnonzero(keep)[keep2]
    y = labels[keep2].to_numpy(dtype=str)

    X_raw = adata.X[row_idx]
    if sparse.issparse(X_raw):
        X = X_raw.toarray().astype(np.float32)
    else:
        X = np.asarray(X_raw, dtype=np.float32)
    X = np.nan_to_num(X, copy=False)

    if "gene_name" in adata.var:
        gene_names = adata.var["gene_name"].astype(str).to_numpy()
    else:
        gene_names = adata.var_names.astype(str).to_numpy()

    metadata = {
        "source": DATA_SOURCES["norman_2019_figshare_h5ad"].url,
        "figshare_article": "24688110",
        "label_column": "guide_merged",
        "task": "control and single-gene perturbation identity classification",
        "min_cells_per_class": min_cells_per_class,
        "n_samples": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "excluded_double_or_unparsed_cells": int((~keep).sum()),
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().items()},
    }
    np.savez_compressed(
        out_dir / "dataset.npz",
        X=X,
        y=y,
        gene_names=np.asarray(gene_names, dtype=str),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
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
