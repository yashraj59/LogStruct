"""
Prior adjacency matrix builders for LogStruct.

Functions to construct biologically meaningful priors from various sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix


def from_random(
    n_features: int,
    density: float = 0.1,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Create a random symmetric adjacency matrix.
    
    Useful for testing or as a baseline prior.
    
    Parameters
    ----------
    n_features : int
        Number of features (genes).
    density : float
        Approximate fraction of non-zero edges.
    random_state : int, optional
        Random seed.
    
    Returns
    -------
    adj : ndarray of shape (n_features, n_features)
        Symmetric adjacency with values in [0, 1].
    """
    rng = np.random.default_rng(random_state)
    
    # Generate upper triangle
    mask = rng.random((n_features, n_features)) < density
    values = rng.random((n_features, n_features)) * 0.5 + 0.3  # [0.3, 0.8]
    
    adj = np.where(mask, values, 0.05)  # small baseline for non-edges
    
    # Symmetrize
    adj = np.triu(adj, k=1)
    adj = adj + adj.T
    
    # No self-loops
    np.fill_diagonal(adj, 0.0)
    
    return adj.astype(np.float32)


def from_identity(n_features: int, baseline: float = 0.01) -> np.ndarray:
    """
    Create a baseline prior with no assumed structure.
    
    All off-diagonal entries are set to a small baseline value,
    effectively letting the model learn structure from scratch.
    
    Parameters
    ----------
    n_features : int
        Number of features.
    baseline : float
        Prior probability for all edges.
    
    Returns
    -------
    adj : ndarray of shape (n_features, n_features)
    """
    adj = np.full((n_features, n_features), baseline, dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def from_correlation(
    X: np.ndarray,
    threshold: float = 0.3,
    absolute: bool = True,
) -> np.ndarray:
    """
    Build prior from feature correlation matrix.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.
    threshold : float
        Minimum correlation to include edge.
    absolute : bool
        Use absolute correlation (captures negative relationships).
    
    Returns
    -------
    adj : ndarray of shape (n_features, n_features)
        Correlation-based adjacency.
    """
    corr = np.corrcoef(X, rowvar=False)
    
    if absolute:
        corr = np.abs(corr)
    else:
        corr = (corr + 1) / 2  # map [-1, 1] to [0, 1]
    
    # Threshold
    adj = np.where(corr >= threshold, corr, 0.05)
    np.fill_diagonal(adj, 0.0)
    
    return adj.astype(np.float32)


def from_edge_list(
    n_features: int,
    edges: Sequence[tuple[int, int]] | Sequence[tuple[int, int, float]],
    default_weight: float = 0.7,
    baseline: float = 0.05,
) -> np.ndarray:
    """
    Build prior from a list of edges.
    
    Parameters
    ----------
    n_features : int
        Number of features.
    edges : sequence of tuples
        Either (i, j) pairs or (i, j, weight) triples.
    default_weight : float
        Weight for edges without explicit weight.
    baseline : float
        Prior probability for non-edges.
    
    Returns
    -------
    adj : ndarray of shape (n_features, n_features)
    """
    adj = np.full((n_features, n_features), baseline, dtype=np.float32)
    
    for edge in edges:
        if len(edge) == 2:
            i, j = edge
            w = default_weight
        else:
            i, j, w = edge
        
        if 0 <= i < n_features and 0 <= j < n_features:
            adj[i, j] = w
            adj[j, i] = w
    
    np.fill_diagonal(adj, 0.0)
    return adj


def from_pathway_gmt(
    gmt_path: str | Path,
    gene_names: Sequence[str],
    within_pathway_weight: float = 0.7,
    baseline: float = 0.05,
) -> np.ndarray:
    """
    Build prior from GMT pathway file.
    
    Genes in the same pathway get connected edges.
    
    Parameters
    ----------
    gmt_path : str or Path
        Path to GMT file (MSigDB format).
    gene_names : sequence of str
        Ordered gene names matching feature indices.
    within_pathway_weight : float
        Edge weight for genes in same pathway.
    baseline : float
        Prior for genes not in same pathway.
    
    Returns
    -------
    adj : ndarray of shape (n_features, n_features)
    
    Notes
    -----
    GMT format: pathway_name \\t description \\t gene1 \\t gene2 \\t ...
    """
    n = len(gene_names)
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    
    adj = np.full((n, n), baseline, dtype=np.float32)
    
    gmt_path = Path(gmt_path)
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            # Get gene indices for this pathway
            genes_in_pathway = [
                gene_to_idx[g.upper()]
                for g in parts[2:]
                if g.upper() in gene_to_idx
            ]
            
            # Connect all pairs within pathway
            for i, gi in enumerate(genes_in_pathway):
                for gj in genes_in_pathway[i + 1:]:
                    adj[gi, gj] = max(adj[gi, gj], within_pathway_weight)
                    adj[gj, gi] = max(adj[gj, gi], within_pathway_weight)
    
    np.fill_diagonal(adj, 0.0)
    return adj


def from_string_db(
    string_path: str | Path,
    gene_names: Sequence[str],
    score_threshold: int = 400,
    baseline: float = 0.05,
) -> np.ndarray:
    """
    Build prior from STRING database protein links.
    
    Parameters
    ----------
    string_path : str or Path
        Path to STRING protein links file (e.g., 9606.protein.links.v12.0.txt).
    gene_names : sequence of str
        Ordered gene/protein names matching feature indices.
    score_threshold : int
        Minimum combined score (0-1000) to include edge.
    baseline : float
        Prior for non-interacting proteins.
    
    Returns
    -------
    adj : ndarray of shape (n_features, n_features)
    
    Notes
    -----
    STRING scores are 0-1000. Common thresholds:
    - 150: low confidence
    - 400: medium confidence
    - 700: high confidence
    - 900: highest confidence
    """
    n = len(gene_names)
    # Handle both gene symbols and ENSP IDs
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    
    adj = np.full((n, n), baseline, dtype=np.float32)
    
    string_path = Path(string_path)
    with open(string_path) as f:
        header = next(f)  # skip header
        
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            p1, p2, score = parts[0], parts[1], int(parts[2])
            
            if score < score_threshold:
                continue
            
            # Extract gene name (STRING format: taxid.ENSP...)
            g1 = p1.split('.')[-1].upper() if '.' in p1 else p1.upper()
            g2 = p2.split('.')[-1].upper() if '.' in p2 else p2.upper()
            
            if g1 in gene_to_idx and g2 in gene_to_idx:
                i, j = gene_to_idx[g1], gene_to_idx[g2]
                weight = score / 1000.0  # normalize to [0, 1]
                adj[i, j] = max(adj[i, j], weight)
                adj[j, i] = max(adj[j, i], weight)
    
    np.fill_diagonal(adj, 0.0)
    return adj


def from_grn(
    tf_target_pairs: Sequence[tuple[str, str]] | Sequence[tuple[str, str, float]],
    gene_names: Sequence[str],
    default_weight: float = 0.7,
    baseline: float = 0.05,
    symmetric: bool = True,
) -> np.ndarray:
    """
    Build prior from gene regulatory network (TF-target pairs).
    
    Parameters
    ----------
    tf_target_pairs : sequence of tuples
        Either (TF, target) or (TF, target, weight).
    gene_names : sequence of str
        Ordered gene names.
    default_weight : float
        Weight for edges without explicit weight.
    baseline : float
        Prior for non-regulatory pairs.
    symmetric : bool
        Whether to symmetrize (for undirected graph models).
    
    Returns
    -------
    adj : ndarray of shape (n_features, n_features)
    """
    n = len(gene_names)
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    
    adj = np.full((n, n), baseline, dtype=np.float32)
    
    for pair in tf_target_pairs:
        if len(pair) == 2:
            tf, target = pair
            w = default_weight
        else:
            tf, target, w = pair
        
        tf_upper, target_upper = tf.upper(), target.upper()
        
        if tf_upper in gene_to_idx and target_upper in gene_to_idx:
            i, j = gene_to_idx[tf_upper], gene_to_idx[target_upper]
            adj[i, j] = max(adj[i, j], w)
            if symmetric:
                adj[j, i] = max(adj[j, i], w)
    
    np.fill_diagonal(adj, 0.0)
    return adj


def to_sparse(adj: np.ndarray, threshold: float = 0.0) -> csr_matrix:
    """
    Convert dense adjacency to sparse format.
    
    Parameters
    ----------
    adj : ndarray
        Dense adjacency matrix.
    threshold : float
        Values below this become zeros.
    
    Returns
    -------
    sparse : csr_matrix
    """
    adj_copy = adj.copy()
    adj_copy[adj_copy < threshold] = 0.0
    return csr_matrix(adj_copy)


def summarize(adj: np.ndarray) -> dict:
    """
    Get summary statistics for an adjacency matrix.
    
    Parameters
    ----------
    adj : ndarray
        Adjacency matrix.
    
    Returns
    -------
    dict with keys:
        - n_nodes: number of nodes
        - n_edges: number of edges (upper triangle non-zero)
        - density: edge density
        - mean_weight: mean edge weight
        - max_weight: maximum edge weight
    """
    n = adj.shape[0]
    upper = np.triu(adj, k=1)
    nonzero = upper[upper > 0]
    
    max_edges = n * (n - 1) / 2
    
    return {
        "n_nodes": n,
        "n_edges": len(nonzero),
        "density": len(nonzero) / max_edges if max_edges > 0 else 0,
        "mean_weight": float(nonzero.mean()) if len(nonzero) > 0 else 0,
        "max_weight": float(nonzero.max()) if len(nonzero) > 0 else 0,
    }
