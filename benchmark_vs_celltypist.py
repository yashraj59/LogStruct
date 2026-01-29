"""
LogStruct vs CellTypist Fair Benchmark

Compares LogStruct against CellTypist (trained from scratch) and sklearn 
LogisticRegression on real organ atlas data from CellTypist.

Usage:
    python benchmark_vs_celltypist.py [--dataset blood|liver] [--n_genes 2000] [--max_cells 50000]
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from time import time

# Single-cell tools
import scanpy as sc

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# LogStruct
from logstruct import LogStructClassifier
from logstruct import priors

# CellTypist
import celltypist
from celltypist import models


# Dataset URLs from CellTypist organ atlas
DATASET_URLS = {
    'blood': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Blood/Blood.h5ad',
    'liver': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Liver/Liver.h5ad',
    'lung': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Lung/Lung.h5ad',
    'kidney': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Kidney/Kidney.h5ad',
}

DATA_DIR = Path('./benchmark_data')


def download_dataset(dataset_name: str) -> sc.AnnData:
    """Download organ atlas dataset from CellTypist."""
    DATA_DIR.mkdir(exist_ok=True)
    
    url = DATASET_URLS.get(dataset_name.lower())
    if url is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Options: {list(DATASET_URLS.keys())}")
    
    local_path = DATA_DIR / f"{dataset_name}.h5ad"
    
    print(f"\n📥 Loading {dataset_name} dataset...")
    if local_path.exists():
        print(f"  Found cached file: {local_path}")
        adata = sc.read_h5ad(local_path)
    else:
        print(f"  Downloading from: {url}")
        print("  (This may take several minutes for large files...)")
        adata = sc.read(local_path, backup_url=url)
    
    print(f"  Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    return adata


def preprocess(adata: sc.AnnData, n_top_genes: int = 3000, max_cells: int = None) -> tuple:
    """
    Preprocess data for benchmark.
    
    Returns both full adata (for CellTypist) and HVG subset (for LogStruct).
    CellTypist requires: log1p normalized expression to 10,000 counts per cell.
    """
    print("\n🔬 Preprocessing...")
    
    # Subsample if needed
    if max_cells is not None and adata.n_obs > max_cells:
        print(f"  Subsampling to {max_cells:,} cells...")
        sc.pp.subsample(adata, n_obs=max_cells, random_state=42)

    # Use raw counts if available
    if adata.raw is not None:
        print("  Resetting X to raw counts from adata.raw...")
        adata = adata.raw.to_adata()
    
    # Find cell type column
    cell_type_col = None
    for col in ['cell_type', 'celltype', 'cell_ontology_class', 'CellType', 'annotation']:
        if col in adata.obs:
            cell_type_col = col
            break
    
    # Also check for harmonized cell type column
    if cell_type_col is None:
        for col in adata.obs.columns:
            if 'type' in col.lower() or 'annot' in col.lower():
                cell_type_col = col
                break
    
    if cell_type_col is None:
        print(f"  Available columns: {adata.obs.columns.tolist()}")
        raise ValueError("No cell type column found!")
    
    print(f"  Using cell type column: {cell_type_col}")
    print(f"  Cell types found: {adata.obs[cell_type_col].nunique()}")
    
    # Check if already normalized (CellTypist data should be)
    if adata.X.max() > 50:  # Likely raw counts
        print("  Normalizing (log1p to 10,000 counts)...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("  Data appears already normalized")
    
    # Filter rare cell types (need enough for train/test split)
    min_cells = 50
    cell_counts = adata.obs[cell_type_col].value_counts()
    valid_types = cell_counts[cell_counts >= min_cells].index
    adata = adata[adata.obs[cell_type_col].isin(valid_types)].copy()
    
    print(f"  Full data: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"  Cell types: {len(valid_types)}")
    
    # Create HVG subset for LogStruct (CellTypist keeps full data)
    print(f"  Selecting top {n_top_genes:,} HVGs for LogStruct...")
    adata_hvg = adata.copy()
    try:
        sc.pp.highly_variable_genes(adata_hvg, n_top_genes=n_top_genes, flavor='seurat_v3', 
                                     layer=None, subset=False)
        adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable].copy()
    except Exception as e:
        print(f"  Warning: HVG selection failed ({e}), using variance-based selection")
        gene_vars = np.var(adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else adata_hvg.X, axis=0)
        top_genes = np.argsort(gene_vars)[-n_top_genes:]
        adata_hvg = adata_hvg[:, top_genes].copy()
    
    print(f"  HVG subset: {adata_hvg.n_obs:,} cells × {adata_hvg.n_vars:,} genes")
    
    return adata, adata_hvg, cell_type_col


def train_celltypist(adata_train: sc.AnnData, cell_type_col: str) -> models.Model:
    """Train CellTypist model from scratch."""
    print("\n🏋️ Training CellTypist...")
    start_time = time()
    
    # CellTypist training
    # Note: check_expression=False because we might be using HVG subset
    ct_model = celltypist.train(
        adata_train,
        labels=cell_type_col,
        n_jobs=-1,
        feature_selection=True,  # Let CellTypist select features
        use_SGD=True,            # Use SGD for large datasets
        check_expression=False,  # Skip normalization check
    )
    
    elapsed = time() - start_time
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Features used: {len(ct_model.features)}")
    
    return ct_model, elapsed


def fetch_string_ppi(gene_names: list, species: int = 9606, score_threshold: int = 700) -> np.ndarray:
    """
    Fetch protein-protein interactions from STRING-DB API.
    
    Parameters
    ----------
    gene_names : list of str
        Gene symbols to look up.
    species : int
        NCBI species ID (9606 = human, 10090 = mouse).
    score_threshold : int
        Minimum combined score (0-1000). 700 = high confidence.
    
    Returns
    -------
    adj : ndarray of shape (n_genes, n_genes)
        PPI adjacency matrix.
    """
    import requests
    
    print(f"  Fetching STRING-DB interactions for {len(gene_names)} genes...")
    
    # STRING-DB API endpoint
    string_api_url = "https://string-db.org/api/json/network"
    
    # Query in batches (STRING has limits)
    batch_size = 500
    all_interactions = []
    
    # Build gene name to index mapping
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    n = len(gene_names)
    
    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i:i+batch_size]
        
        params = {
            "identifiers": "%0d".join(batch),
            "species": species,
            "required_score": score_threshold,
            "caller_identity": "logstruct_benchmark"
        }
        
        try:
            response = requests.post(string_api_url, data=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                all_interactions.extend(data)
        except Exception as e:
            print(f"    Warning: STRING API call failed: {e}")
    
    print(f"    Retrieved {len(all_interactions)} interactions")
    
    # Build adjacency matrix
    adj = np.full((n, n), 0.05, dtype=np.float32)  # baseline
    
    for interaction in all_interactions:
        gene_a = interaction.get('preferredName_A', '').upper()
        gene_b = interaction.get('preferredName_B', '').upper()
        score = interaction.get('score', 0)
        
        if gene_a in gene_to_idx and gene_b in gene_to_idx:
            i, j = gene_to_idx[gene_a], gene_to_idx[gene_b]
            weight = score  # Already 0-1
            adj[i, j] = max(adj[i, j], weight)
            adj[j, i] = max(adj[j, i], weight)
    
    np.fill_diagonal(adj, 0.0)
    
    return adj


def select_features(X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, k: int = 500) -> tuple:
    """
    Select top k features using ANOVA F-test.
    
    Returns
    -------
    X_train_sel, X_test_sel, selected_indices
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    
    print(f"\n🔍 Selecting top {k} features...")
    
    selector = SelectKBest(f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    selected_indices = selector.get_support(indices=True)
    print(f"  Selected {len(selected_indices)} features")
    
    return X_train_sel, X_test_sel, selected_indices


def train_logstruct(X_train: np.ndarray, y_train: np.ndarray, n_features: int,
                    gene_names: list = None, use_ppi: bool = True) -> tuple:
    """
    Train LogStruct classifier with biological prior and optimized training.
    
    Parameters
    ----------
    X_train : array
        Training features.
    y_train : array
        Training labels.
    n_features : int
        Number of features.
    gene_names : list, optional
        Gene names for STRING-DB lookup.
    use_ppi : bool
        Whether to use STRING-DB PPI prior.
    """
    print("\n🧠 Training LogStruct (fully optimized)...")
    start_time = time()
    
    # Build prior
    if use_ppi and gene_names is not None:
        print("  Building STRING-DB PPI prior...")
        try:
            prior = fetch_string_ppi(gene_names, species=9606, score_threshold=700)
            prior_stats = priors.summarize(prior)
            print(f"  PPI Prior: {prior_stats['n_edges']:,} edges above threshold, density={prior_stats['density']:.4f}")
        except Exception as e:
            print(f"  Warning: STRING-DB failed ({e}), falling back to identity prior")
            prior = priors.from_identity(n_features, baseline=0.01)
    else:
        print("  Using identity prior (no network)")
        prior = priors.from_identity(n_features, baseline=0.01)
    
    clf = LogStructClassifier(
        prior_adjacency=prior,
        lambda_en=0.001,          # Optimized: Very light regularization
        lambda_smooth=0.01,       # Optimized: Light smoothing helps!
        lambda_kl=0.0,            # Optimized: No KL needed
        alpha=0.5,
        max_iter=1000,
        learning_rate=0.02,       # Optimized: Higher LR
        early_stopping=True,
        n_iter_no_change=50,
        tol=1e-5,
        verbose=True,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    
    elapsed = time() - start_time
    print(f"  Training time: {elapsed:.1f}s")
    print(f"  Final accuracy: {clf.history_['train_acc'][-1]:.4f}")
    if clf.history_['val_acc']:
        print(f"  Final val accuracy: {clf.history_['val_acc'][-1]:.4f}")
    
    return clf, elapsed


def train_sklearn(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Train sklearn LogisticRegression as baseline."""
    print("\n📊 Training sklearn LogisticRegression...")
    start_time = time()
    
    clf = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    
    elapsed = time() - start_time
    print(f"  Training time: {elapsed:.1f}s")
    
    return clf, elapsed


def evaluate_celltypist(ct_model: models.Model, adata_test: sc.AnnData, 
                        y_test: np.ndarray, le: LabelEncoder) -> dict:
    """Evaluate CellTypist predictions."""
    print("\n📋 Evaluating CellTypist...")
    
    # Predict
    predictions = celltypist.annotate(
        adata_test, 
        model=ct_model, 
        majority_voting=False  # Faster, per-cell prediction
    )
    
    # Get predicted labels and convert to encoded format
    y_pred_labels = predictions.predicted_labels['predicted_labels'].values
    
    # Handle label mapping (some predictions may be new labels not in training)
    y_pred = []
    for label in y_pred_labels:
        if label in le.classes_:
            y_pred.append(le.transform([label])[0])
        else:
            # Assign to most common class if unknown label
            y_pred.append(0)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1:.4f}")
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'y_pred': y_pred,
    }


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Evaluate a classifier (LogStruct or sklearn)."""
    print(f"\n📋 Evaluating {name}...")
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Macro: {f1:.4f}")
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'y_pred': y_pred,
    }


def visualize_results(results: dict, class_names: np.ndarray, output_dir: Path):
    """Create comprehensive visualizations."""
    output_dir.mkdir(exist_ok=True)
    
    print("\n📊 Creating visualizations...")
    
    # 1. Bar chart comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    f1_scores = [results[m]['f1_macro'] for m in model_names]
    train_times = [results[m].get('train_time', 0) for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1 Macro', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('LogStruct vs CellTypist: Cell Type Classification (Organ Atlas)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_results.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'benchmark_results.png'}")
    plt.close()
    
    # 2. Training time comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue', 'coral', 'seagreen']
    ax.bar(model_names, train_times, color=colors[:len(model_names)])
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    for i, t in enumerate(train_times):
        ax.annotate(f'{t:.1f}s', xy=(i, t), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'training_time.png'}")
    plt.close()


def visualize_network(clf, gene_names: list, output_dir: Path):
    """Visualize LogStruct's learned network."""
    print("\n🕸️ Visualizing learned network...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Learned adjacency heatmap (top genes)
    ax = axes[0]
    top_edges = clf.get_top_edges(n=100)
    top_gene_indices = list(set([e[0] for e in top_edges] + [e[1] for e in top_edges]))[:40]
    
    if len(top_gene_indices) > 0:
        sub_adj = clf.adjacency_[np.ix_(top_gene_indices, top_gene_indices)]
        sub_names = [gene_names[i][:10] if i < len(gene_names) else str(i) for i in top_gene_indices]
        
        sns.heatmap(sub_adj, ax=ax, cmap='Blues', 
                    xticklabels=sub_names, yticklabels=sub_names,
                    cbar_kws={'label': 'Edge Weight'})
        ax.set_title('Learned Adjacency (Top 40 Genes)')
        ax.tick_params(axis='x', rotation=90, labelsize=6)
        ax.tick_params(axis='y', rotation=0, labelsize=6)
    
    # 2. Top edges bar chart
    ax = axes[1]
    edge_labels = [f"{gene_names[e[0]][:8]}-{gene_names[e[1]][:8]}" 
                   if e[0] < len(gene_names) and e[1] < len(gene_names) 
                   else f"{e[0]}-{e[1]}" 
                   for e in top_edges[:20]]
    edge_weights = [e[2] for e in top_edges[:20]]
    
    ax.barh(range(len(edge_labels)), edge_weights, color='steelblue')
    ax.set_yticks(range(len(edge_labels)))
    ax.set_yticklabels(edge_labels, fontsize=7)
    ax.set_xlabel('Learned Edge Weight')
    ax.set_title('Top 20 Learned Gene-Gene Edges')
    ax.invert_yaxis()
    
    # 3. Coefficient distribution
    ax = axes[2]
    coef = clf.coef_ if clf.coef_.ndim == 1 else np.abs(clf.coef_).mean(axis=0)
    
    sorted_idx = np.argsort(coef)
    n_show = min(15, len(coef) // 2)
    top_neg = sorted_idx[:n_show]
    top_pos = sorted_idx[-n_show:]
    
    combined_idx = list(top_neg) + list(top_pos)
    combined_coef = [coef[i] for i in combined_idx]
    combined_names = [gene_names[i][:12] if i < len(gene_names) else str(i) for i in combined_idx]
    colors = ['salmon'] * n_show + ['steelblue'] * n_show
    
    ax.barh(range(len(combined_names)), combined_coef, color=colors)
    ax.set_yticks(range(len(combined_names)))
    ax.set_yticklabels(combined_names, fontsize=8)
    ax.set_xlabel('|Coefficient| (mean across classes)')
    ax.set_title('Top Gene Coefficients')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learned_network.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'learned_network.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='LogStruct vs CellTypist Benchmark')
    parser.add_argument('--dataset', type=str, default='blood', 
                        choices=['blood', 'liver', 'lung', 'kidney'],
                        help='Organ atlas dataset to use')
    parser.add_argument('--n_genes', type=int, default=2000,
                        help='Number of highly variable genes to use')
    parser.add_argument('--max_cells', type=int, default=50000,
                        help='Maximum cells to use (for memory constraints)')
    parser.add_argument('--output_dir', type=str, default='./benchmark_output',
                        help='Output directory for results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("LogStruct vs CellTypist Fair Benchmark")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"HVGs: {args.n_genes}")
    print(f"Max cells: {args.max_cells}")
    
    # 1. Load and preprocess data
    adata_raw = download_dataset(args.dataset)
    adata_full, adata_hvg, cell_type_col = preprocess(adata_raw, n_top_genes=args.n_genes, max_cells=args.max_cells)
    
    # 2. Prepare data - use HVG subset for LogStruct
    gene_names_hvg = adata_hvg.var_names.tolist()
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(adata_hvg.obs[cell_type_col].values)
    class_names = le.classes_
    
    # Get expression matrices
    X_hvg = adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else np.array(adata_hvg.X)
    
    print(f"\n📊 Dataset summary:")
    print(f"  Cells: {X_hvg.shape[0]:,}")
    print(f"  HVG genes (LogStruct): {X_hvg.shape[1]:,}")
    print(f"  Full genes (CellTypist): {adata_full.n_vars:,}")
    print(f"  Cell types: {len(class_names)}")
    
    # 3. Split data (80/20)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=42
    )
    
    X_train_hvg, X_test_hvg = X_hvg[train_idx], X_hvg[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Create AnnData subsets
    # Use HVG data for CellTypist too (as requested for speed)
    adata_train = adata_hvg[train_idx].copy()
    adata_test = adata_hvg[test_idx].copy()
    
    print(f"\n  Train: {len(train_idx):,} cells")
    print(f"  Test: {len(test_idx):,} cells")
    
    results = {}
    
    # 4. Feature selection for LogStruct (use 3000 genes for better representation)
    n_select = min(3000, X_hvg.shape[1])
    X_train_sel, X_test_sel, selected_indices = select_features(X_train_hvg, y_train, X_test_hvg, k=n_select)
    selected_gene_names = [gene_names_hvg[i] for i in selected_indices]
    
    # 5. Train and evaluate sklearn (baseline) - on selected features for fair comparison
    sklearn_clf, sklearn_time = train_sklearn(X_train_sel, y_train)
    sklearn_results = evaluate_model(sklearn_clf, X_test_sel, y_test, "sklearn LR")
    sklearn_results['train_time'] = sklearn_time
    results['sklearn LR'] = sklearn_results
    
    # 6. Train and evaluate LogStruct - on selected features with PPI prior
    ls_clf, ls_time = train_logstruct(X_train_sel, y_train, n_select, 
                                       gene_names=selected_gene_names, use_ppi=True)
    ls_results = evaluate_model(ls_clf, X_test_sel, y_test, "LogStruct")
    ls_results['train_time'] = ls_time
    results['LogStruct'] = ls_results
    
    # 7. Train and evaluate CellTypist (uses HVG subset now)
    ct_model, ct_time = train_celltypist(adata_train, cell_type_col)
    ct_results = evaluate_celltypist(ct_model, adata_test, y_test, le)
    ct_results['train_time'] = ct_time
    results['CellTypist'] = ct_results
    
    # 7. Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Accuracy':<12} {'F1 Macro':<12} {'Train Time':<12}")
    print("-" * 56)
    for model_name, metrics in results.items():
        acc = f"{metrics['accuracy']:.4f}"
        f1 = f"{metrics['f1_macro']:.4f}"
        train_t = f"{metrics['train_time']:.1f}s"
        print(f"{model_name:<20} {acc:<12} {f1:<12} {train_t:<12}")
    
    # 8. Visualizations
    visualize_results(results, class_names, output_dir)
    visualize_network(ls_clf, selected_gene_names, output_dir)
    
    # 9. Detailed report
    print("\n" + "=" * 70)
    print("LogStruct Classification Report")
    print("=" * 70)
    print(classification_report(y_test, ls_results['y_pred'], 
                                target_names=class_names, zero_division=0))
    
    print("\n" + "=" * 70)
    print("CellTypist Classification Report")
    print("=" * 70)
    print(classification_report(y_test, ct_results['y_pred'], 
                                target_names=class_names, zero_division=0))
    
    # Save results
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'F1_Macro': [r['f1_macro'] for r in results.values()],
        'Train_Time': [r['train_time'] for r in results.values()],
    })
    results_df.to_csv(output_dir / 'benchmark_results.csv', index=False)
    print(f"\n  Results saved to: {output_dir / 'benchmark_results.csv'}")
    
    print("\n✅ Benchmark complete!")
    print(f"Output files in: {output_dir}")


if __name__ == "__main__":
    main()
