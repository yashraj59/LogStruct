"""
Visualization utilities for LogStruct.

Functions for visualizing learned networks, adjacency matrices, and coefficients.
"""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization. Run: pip install matplotlib")


def plot_adjacency_comparison(
    prior: np.ndarray,
    learned: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    max_features: int = 50,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
):
    """
    Side-by-side comparison of prior and learned adjacency matrices.
    
    Parameters
    ----------
    prior : ndarray of shape (n_features, n_features)
        Prior adjacency matrix.
    learned : ndarray of shape (n_features, n_features)
        Learned adjacency matrix.
    feature_names : sequence of str, optional
        Names for features (genes).
    max_features : int
        Maximum features to show (for readability).
    figsize : tuple
        Figure size.
    save_path : str, optional
        If provided, save figure to this path.
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    _check_matplotlib()
    import seaborn as sns
    
    n = min(prior.shape[0], max_features)
    prior_sub = prior[:n, :n]
    learned_sub = learned[:n, :n]
    
    if feature_names is not None:
        labels = [str(f)[:10] for f in feature_names[:n]]
    else:
        labels = False
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    sns.heatmap(prior_sub, ax=axes[0], cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Edge Weight'})
    axes[0].set_title('Prior Adjacency')
    
    sns.heatmap(learned_sub, ax=axes[1], cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Edge Weight'})
    axes[1].set_title('Learned Adjacency')
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


def plot_network_graph(
    adjacency: np.ndarray,
    feature_names: Sequence[str],
    top_n: int = 30,
    threshold: float = 0.1,
    figsize: tuple = (12, 12),
    node_size_scale: float = 300,
    save_path: Optional[str] = None,
):
    """
    Visualize top edges as a network graph.
    
    Parameters
    ----------
    adjacency : ndarray of shape (n_features, n_features)
        Adjacency matrix.
    feature_names : sequence of str
        Names for features (genes).
    top_n : int
        Number of top edges to include.
    threshold : float
        Minimum edge weight to include.
    figsize : tuple
        Figure size.
    node_size_scale : float
        Base size for nodes.
    save_path : str, optional
        If provided, save figure to this path.
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    G : networkx Graph
    """
    _check_matplotlib()
    
    if not HAS_NETWORKX:
        raise ImportError("networkx required for network visualization. Run: pip install networkx")
    
    # Extract top edges
    n = adjacency.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] > threshold:
                edges.append((i, j, adjacency[i, j]))
    
    # Sort and take top
    edges.sort(key=lambda x: x[2], reverse=True)
    edges = edges[:top_n]
    
    if not edges:
        print("No edges above threshold")
        return None, None, None
    
    # Build graph
    G = nx.Graph()
    
    for i, j, w in edges:
        G.add_edge(feature_names[i], feature_names[j], weight=w)
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Node sizes based on degree
    degrees = dict(G.degree())
    node_sizes = [degrees[n] * node_size_scale for n in G.nodes()]
    
    # Edge widths based on weight
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                           node_color='steelblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, 
                           alpha=0.6, edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
    
    ax.set_title(f'Learned Gene Network (Top {len(edges)} Edges)')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax, G


def plot_coefficients(
    coef: np.ndarray,
    feature_names: Sequence[str],
    top_n: int = 20,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Bar plot of top positive and negative coefficients.
    
    Parameters
    ----------
    coef : ndarray of shape (n_features,) or (n_classes, n_features)
        Coefficient vector(s).
    feature_names : sequence of str
        Names for features.
    top_n : int
        Number of top coefficients to show (from each end).
    figsize : tuple
        Figure size.
    save_path : str, optional
        If provided, save figure to this path.
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    _check_matplotlib()
    
    # Handle multi-class
    if coef.ndim > 1:
        coef = coef.mean(axis=0)
    
    # Sort
    sorted_idx = np.argsort(coef)
    
    n_show = min(top_n // 2, len(coef) // 2)
    top_neg = sorted_idx[:n_show]
    top_pos = sorted_idx[-n_show:]
    
    combined_idx = list(top_neg) + list(top_pos)
    combined_coef = [coef[i] for i in combined_idx]
    combined_names = [str(feature_names[i])[:15] for i in combined_idx]
    colors = ['salmon'] * n_show + ['steelblue'] * n_show
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(range(len(combined_names)), combined_coef, color=colors)
    ax.set_yticks(range(len(combined_names)))
    ax.set_yticklabels(combined_names, fontsize=9)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Coefficient')
    ax.set_title('Top Positive & Negative Coefficients')
    
    # Legend
    neg_patch = mpatches.Patch(color='salmon', label='Negative')
    pos_patch = mpatches.Patch(color='steelblue', label='Positive')
    ax.legend(handles=[neg_patch, pos_patch], loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_training_history(
    history: dict,
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Plot training curves from model history.
    
    Parameters
    ----------
    history : dict
        Training history with 'train_loss', 'val_loss', etc.
    figsize : tuple
        Figure size.
    save_path : str, optional
        If provided, save figure to this path.
    
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    
    # Accuracy (if available)
    if 'train_acc' in history or 'val_acc' in history:
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train')
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No accuracy data', ha='center', va='center')
        axes[1].set_title('Accuracy Curves')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes
