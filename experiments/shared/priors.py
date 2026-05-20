"""Prior construction utilities for experiments."""

from __future__ import annotations

import json
from pathlib import Path

import gseapy as gp
import numpy as np

from logstruct import priors as logstruct_priors


ENRICHR_LIBRARIES = {
    "kegg": "KEGG_2021_Human",
    "reactome": "Reactome_2022",
    "hallmark": "MSigDB_Hallmark_2020",
}


def save_prior_npz(path: str | Path, adjacency: np.ndarray, gene_names, metadata: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        adjacency=np.asarray(adjacency, dtype=np.float32),
        gene_names=np.asarray(gene_names, dtype=str),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    return path


def pathway_prior_from_enrichr(
    library_name: str,
    gene_names,
    *,
    within_pathway_weight: float = 0.7,
    baseline: float = 0.01,
):
    gene_sets = gp.get_library(name=library_name, organism="Human")
    n = len(gene_names)
    gene_to_idx = {str(g).upper(): i for i, g in enumerate(gene_names)}
    adj = np.full((n, n), baseline, dtype=np.float32)
    for genes in gene_sets.values():
        idx = [gene_to_idx[g.upper()] for g in genes if g.upper() in gene_to_idx]
        for pos, i in enumerate(idx):
            for j in idx[pos + 1 :]:
                adj[i, j] = max(adj[i, j], within_pathway_weight)
                adj[j, i] = max(adj[j, i], within_pathway_weight)
    np.fill_diagonal(adj, 0.0)
    return adj


def build_common_priors(
    gene_names,
    out_dir: str | Path,
    *,
    string_links: str | Path | None = None,
    string_aliases: str | Path | None = None,
    max_random_density: float = 0.02,
    seed: int = 0,
    include: set[str] | list[str] | tuple[str, ...] | None = None,
):
    out_dir = Path(out_dir)
    made = {}
    gene_names = list(gene_names)
    include = set(include) if include is not None else None

    if include is None or "identity" in include:
        identity = logstruct_priors.from_identity(len(gene_names), baseline=0.01)
        made["identity"] = save_prior_npz(
            out_dir / "identity.npz",
            identity,
            gene_names,
            {"source": "identity", "baseline": 0.01},
        )

    if include is None or "random" in include:
        random = logstruct_priors.from_random(
            len(gene_names), density=max_random_density, random_state=seed
        )
        made["random"] = save_prior_npz(
            out_dir / "random.npz",
            random,
            gene_names,
            {"source": "random", "density": max_random_density, "seed": seed},
        )

    if string_links is not None and Path(string_links).exists():
        for threshold in [400, 700]:
            key = f"string_{threshold}"
            if include is not None and key not in include:
                continue
            adj = logstruct_priors.from_string_db(
                string_links,
                gene_names,
                score_threshold=threshold,
                baseline=0.01,
                alias_path=string_aliases,
            )
            made[key] = save_prior_npz(
                out_dir / f"string_{threshold}.npz",
                adj,
                gene_names,
                {"source": "STRING v12", "score_threshold": threshold},
            )

    for short, library in ENRICHR_LIBRARIES.items():
        if include is not None and short not in include:
            continue
        try:
            adj = pathway_prior_from_enrichr(library, gene_names, baseline=0.01)
        except Exception as exc:
            (out_dir / f"{short}.blocked.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            continue
        made[short] = save_prior_npz(
            out_dir / f"{short}.npz",
            adj,
            gene_names,
            {"source": "Enrichr", "library": library},
        )

    return made
