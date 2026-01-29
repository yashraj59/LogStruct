import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import sys
import os
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Add current directory to path for LogStruct import
sys.path.append(os.getcwd())
try:
    from logstruct import LogStructClassifier
    from logstruct import priors
except ImportError:
    print("⚠️  Warning: LogStruct import failed. Ensure you are in the logstruct repository.")

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ... (Previous imports)

# ==========================================
# 4. VISUALIZATION FUNCTIONS
# ==========================================

def visualize_network(top_edges_df, output_dir):
    """Visualize learned gene-gene interaction network."""
    print("    Generatng network plot...")
    G = nx.Graph()
    
    # Add edges
    for _, row in top_edges_df.iterrows():
        G.add_edge(row['Gene A'], row['Gene B'], weight=row['Weight'], type='Prior' if row['In Prior'] else 'New')
        
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    plt.figure(figsize=(12, 12))
    
    # Draw edges based on validation status
    edges_prior = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'Prior']
    edges_new = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'New']
    
    nx.draw_networkx_edges(G, pos, edgelist=edges_prior, width=2, edge_color='green', alpha=0.6, label='In StringDB')
    nx.draw_networkx_edges(G, pos, edgelist=edges_new, width=2, edge_color='gray', style='dashed', alpha=0.5, label='Novel')
    
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', edgecolors='white')
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title("Top Learned Gene Interactions (Green = Validated)", fontsize=16)
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'learned_network_graph.png', dpi=300)
    plt.close()

def visualize_markers(marker_df, output_dir):
    """Visualize top markers for a subset of classes."""
    print("    Generating marker plots...")
    # Pick top 6 largest classes to show
    top_classes = marker_df['Class'].unique()[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, cls in enumerate(top_classes):
        if i >= 6: break
        ax = axes[i]
        data = marker_df[marker_df['Class'] == cls].head(5)
        
        sns.barplot(x='Coefficient', y='Gene', data=data, ax=ax, palette='viridis', hue='Gene', legend=False)
        ax.set_title(cls[:30] + '...' if len(cls)>30 else cls, fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    plt.suptitle("Top Learned Positive Markers per Cell Type", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'learned_markers_barchart.png')
    plt.close()

def visualize_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Visualize confusion matrix."""
    print("    Generating confusion matrix...")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Normalized Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()

def visualize_performance_curves(y_test, y_score, class_names, output_dir):
    """Generate ROC and Precision-Recall curves."""
    print("    Generating ROC and PR curves...")
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    
    # Binarize labels
    y_test_bin = label_binarize(y_test, classes=range(len(class_names)))
    n_classes = y_test_bin.shape[1]
    
    # --- ROC Curves ---
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
             
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    # Plot top 5 classes by AUC for clarity if many classes
    sorted_indices = np.argsort([roc_auc[i] for i in range(n_classes)])[::-1]
    
    for i, color in zip(sorted_indices[:7], colors): # Show top 7
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC {0} (area = {1:0.2f})'.format(class_names[i][:15], roc_auc[i]))
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Top Classes)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300)
    plt.close()
    
    # --- Precision-Recall Curves ---
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
        
    plt.figure(figsize=(10, 8))
    
    # Plot top 7 by AP
    sorted_ap = np.argsort([average_precision[i] for i in range(n_classes)])[::-1]
    for i, color in zip(sorted_ap[:7], colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='PR {0} (AP = {1:0.2f})'.format(class_names[i][:15], average_precision[i]))
                 
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (Top Classes)')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curves.png', dpi=300)
    plt.close()

# Update analyze functions to return dataframes instead of just saving
def analyze_network(clf, gene_names, prior, output_dir, top_n=50):
    """Extract and save top learned gene-gene interactions."""
    print("\n🕸️  Analyzing Learned Network...")
    
    # Extract adjacency
    if hasattr(clf, 'adjacency_'):
        adj_matrix = clf.adjacency_
    elif hasattr(clf, 'parameter_adjacency_'):
        adj_matrix = torch.sigmoid(clf.parameter_adjacency_).detach().cpu().numpy()
    else:
        print("    Error: Could not access adjacency matrix from model.")
        return None

    np.fill_diagonal(adj_matrix, 0)
    
    # Get indices of top edges
    flat_indices = np.argsort(adj_matrix.flatten())[::-1]
    top_edges = []
    
    # Track added pairs to avoid duplicates
    added_pairs = set()
    
    for idx in flat_indices:
        if len(top_edges) >= top_n: break
        
        i, j = divmod(idx, len(gene_names))
        if i >= j: continue 
        
        weight = adj_matrix[i, j]
        gene_a = gene_names[i]
        gene_b = gene_names[j]
        
        top_edges.append({
            'Gene A': gene_a,
            'Gene B': gene_b,
            'Weight': weight,
            'In Prior': prior[i, j] > 0.05
        })
        
    df = pd.DataFrame(top_edges)
    df.to_csv(output_dir / 'learned_network_edges.csv', index=False)
    print(f"    Saved top {top_n} edges to {output_dir}/learned_network_edges.csv")
    print("    Top 5 Interactions:")
    print(df.head(5).to_string(index=False))
    return df

def analyze_coefficients(clf, gene_names, class_names, output_dir):
    """Extract and save top predicted marker genes per class."""
    print("\n🧪 Analyzing Learned Coefficients (Marker Genes)...")
    
    top_genes_per_class = []
    
    for i, class_name in enumerate(class_names):
        coefs = clf.coef_[i]
        top_idx = np.argsort(coefs)[-10:][::-1]
        
        for rank, idx in enumerate(top_idx, 1):
            top_genes_per_class.append({
                'Class': class_name,
                'Rank': rank,
                'Gene': gene_names[idx],
                'Coefficient': coefs[idx]
            })
            
    df = pd.DataFrame(top_genes_per_class)
    df.to_csv(output_dir / 'learned_marker_genes.csv', index=False)
    
    print(f"    Saved markers to {output_dir}/learned_marker_genes.csv")
    return df


def fetch_string_ppi(gene_names, species=9606, score_threshold=700):
    """Fetch PPI from STRING-DB to build prior adjacency."""
    print(f"  Fetching STRING-DB interactions for {len(gene_names)} genes...")
    string_api_url = "https://string-db.org/api/json/network"
    batch_size = 400
    all_interactions = []
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    
    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i:i+batch_size]
        params = {
            "identifiers": "%0d".join(batch),
            "species": species,
            "required_score": score_threshold,
            "caller_identity": "logstruct_analysis"
        }
        try:
            r = requests.post(string_api_url, data=params, timeout=60)
            if r.status_code == 200:
                all_interactions.extend(r.json())
        except Exception as e:
            print(f"    Warning: PPI fetch failed for batch {i}: {e}")
            
    n = len(gene_names)
    adj = np.full((n, n), 0.01, dtype=np.float32) # Weak prior baseline
    
    count = 0
    for interaction in all_interactions:
        a = interaction.get('preferredName_A', '').upper()
        b = interaction.get('preferredName_B', '').upper()
        score = interaction.get('score', 0)
        
        if a in gene_to_idx and b in gene_to_idx:
            i, j = gene_to_idx[a], gene_to_idx[b]
            # Normalize STRING score (0-1000) to 0-1
            w = score 
            adj[i, j] = max(adj[i, j], w)
            adj[j, i] = max(adj[j, i], w)
            count += 1
            
    np.fill_diagonal(adj, 0.0)
    print(f"    Retrieved {count} interactions")
    return adj

# ==========================================
# 3. MAIN WORKFLOW
# ==========================================

def run_analysis(args):
    # Setup Paths
    data_path = Path(args.data_dir) / f"{args.dataset}.h5ad"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"❌ Error: Dataset not found at {data_path}")
        return

    # 1. LOAD DATA
    print(f"\n📂 Loading {args.dataset}...")
    adata = sc.read_h5ad(data_path)
    
    # Subsample
    if args.max_cells and adata.n_obs > args.max_cells:
        print(f"   Subsampling to {args.max_cells} cells...")
        sc.pp.subsample(adata, n_obs=args.max_cells, random_state=42)
    
    # Raw Counts Handling (Critical for proper normalization)
    if adata.raw is not None:
        print("   Using raw counts from adata.raw...")
        adata = adata.raw.to_adata()
        
    # Filter cells
    cell_type_col = args.label_col
    counts = adata.obs[cell_type_col].value_counts()
    valid_types = counts[counts >= 50].index
    adata = adata[adata.obs[cell_type_col].isin(valid_types)].copy()
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # HVGs
    print(f"   Selecting {args.n_genes} HVGs...")
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_genes, flavor='seurat_v3', subset=True)
    except:
        # Fallback for normalized data
        sc.pp.highly_variable_genes(adata, n_top_genes=args.n_genes, subset=True)

    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    le = LabelEncoder()
    y = le.fit_transform(adata.obs[cell_type_col])
    class_names = le.classes_
    gene_names = np.array(adata.var_names)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Feature Select (Univariate filter)
    print("   Applying univariate feature selection...")
    selector = SelectKBest(f_classif, k=args.n_genes)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)
    selected_genes = gene_names[selector.get_support()]
    
    # Prior
    print("\n🧠 Building Biological Prior...")
    prior = fetch_string_ppi(selected_genes)
    
    # 2. TRAIN MODEL
    print("\n🚀 Training LogStruct (Optimized)...")
    clf = LogStructClassifier(
        prior_adjacency=prior,
        lambda_en=0.001,          # Optimized
        lambda_smooth=0.01,       # Optimized
        lambda_kl=0.0,            # Optimized
        learning_rate=0.02,       # Optimized
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=50,
        tol=1e-5,
        verbose=True,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    # 3. EVALUATE
    print("\n📊 Evaluating Performance...")
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)
    else:
        # Fallback if neither exists (unlikely for a classifier)
        print("    Warning: Model does not support probability prediction. improved plots skipped.")
        y_score = None
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 Macro: {f1:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / 'classification_report.csv')
    print(f"   Saved report to {output_dir}/classification_report.csv")
    
    # 4. INTERPRET
    network_df = analyze_network(clf, selected_genes, prior, output_dir)
    if network_df is not None:
        visualize_network(network_df, output_dir)
        
    markers_df = analyze_coefficients(clf, selected_genes, class_names, output_dir)
    if markers_df is not None:
        visualize_markers(markers_df, output_dir)
        
    visualize_confusion_matrix(y_test, y_pred, class_names, output_dir)
    if y_score is not None:
        visualize_performance_curves(y_test, y_score, class_names, output_dir)
    
    print("\n✅ Analysis Complete! Results located in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full LogStruct Model Analysis")
    parser.add_argument('--dataset', type=str, default='blood', help='Dataset name (without .h5ad)')
    parser.add_argument('--data_dir', type=str, default='./benchmark_data', help='Directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='./logstruct_analysis_results', help='Output directory')
    parser.add_argument('--n_genes', type=int, default=3000, help='Number of features')
    parser.add_argument('--max_cells', type=int, default=20000, help='Subsample size for analysis')
    parser.add_argument('--label_col', type=str, default='cell_type', help='Column name for cell types')
    
    args = parser.parse_args()
    run_analysis(args)
