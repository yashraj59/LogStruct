#!/usr/bin/env python3
"""
Multi-Dataset Benchmark: LogStruct vs CellTypist vs Sklearn LR
Runs benchmarks across all CellTypist Organ Atlas datasets.
"""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import sys
import os
import requests
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.getcwd())
try:
    from logstruct import LogStructClassifier
except ImportError:
    print("⚠️  LogStruct import failed.")

try:
    import celltypist
    from celltypist import models
    CELLTYPIST_AVAILABLE = True
except ImportError:
    print("⚠️  CellTypist not installed. Skipping CellTypist benchmarks.")
    CELLTYPIST_AVAILABLE = False

# ==========================================
# DATASET URLS (CellTypist Organ Atlas)
# ==========================================

DATASET_URLS = {
    'blood': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Blood/Blood.h5ad',
    'bone_marrow': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Bone_marrow/Bone_marrow.h5ad',
    'heart': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Heart/Heart.h5ad',
    'hippocampus': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Hippocampus/Hippocampus.h5ad',
    'intestine': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Intestine/Intestine.h5ad',
    'kidney': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Kidney/Kidney.h5ad',
    'liver': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Liver/Liver.h5ad',
    'lung': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Lung/Lung.h5ad',
    'lymph_node': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Lymph_node/Lymph_node.h5ad',
    'pancreas': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Pancreas/Pancreas.h5ad',
    'skeletal_muscle': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Skeletal_muscle/Skeletal_muscle.h5ad',
    'spleen': 'https://celltypist.cog.sanger.ac.uk/Resources/Organ_atlas/Spleen/Spleen.h5ad',
}

# ==========================================
# UTILITIES
# ==========================================

def download_dataset(name, url, data_dir):
    """Download dataset if not present."""
    filepath = data_dir / f"{name}.h5ad"
    if filepath.exists():
        print(f"  ✅ {name} already exists")
        return filepath
    
    print(f"  ⬇️  Downloading {name}...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  ✅ Downloaded {name}")
        return filepath
    except Exception as e:
        print(f"  ❌ Failed to download {name}: {e}")
        return None

def fetch_string_ppi(gene_names):
    """Fetch PPI from STRING-DB."""
    string_api_url = "https://string-db.org/api/json/network"
    batch_size = 400
    all_interactions = []
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    
    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i:i+batch_size]
        params = {"identifiers": "%0d".join(batch), "species": 9606, "required_score": 700, "caller_identity": "logstruct_bench"}
        try:
            r = requests.post(string_api_url, data=params, timeout=30)
            if r.status_code == 200: all_interactions.extend(r.json())
        except: pass
            
    n = len(gene_names)
    adj = np.full((n, n), 0.01, dtype=np.float32)
    for interaction in all_interactions:
        a, b = interaction.get('preferredName_A', '').upper(), interaction.get('preferredName_B', '').upper()
        if a in gene_to_idx and b in gene_to_idx:
            i, j = gene_to_idx[a], gene_to_idx[b]
            w = interaction.get('score', 0)
            adj[i, j] = max(adj[i, j], w)
            adj[j, i] = max(adj[j, i], w)
    np.fill_diagonal(adj, 0.0)
    return adj

# ==========================================
# BENCHMARK FUNCTION
# ==========================================

def benchmark_dataset(name, data_dir, max_cells=300000, n_genes=3000):
    """Run benchmark on a single dataset."""
    filepath = data_dir / f"{name}.h5ad"
    if not filepath.exists():
        return None
    
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARKING: {name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Load
        adata = sc.read_h5ad(filepath)
        original_cells = adata.n_obs
        
        # Subsample first
        if adata.n_obs > max_cells:
            print(f"  Subsampling to {max_cells:,} cells...")
            sc.pp.subsample(adata, n_obs=max_cells, random_state=42)
        
        # Use raw counts
        if adata.raw is not None:
            print("  Resetting X to raw counts from adata.raw...")
            adata.X = adata.raw.X.copy()
        
        # Find cell type column
        cell_type_col = None
        for col in ['cell_type', 'celltype', 'Celltype', 'Cell_type', 'annotation', 'Annotation', 'cell_ontology_class']:
            if col in adata.obs.columns:
                cell_type_col = col
                break
        if cell_type_col is None:
            for col in adata.obs.columns:
                if 'type' in col.lower() or 'annot' in col.lower():
                    cell_type_col = col
                    break
        if cell_type_col is None:
            print(f"  ❌ No cell type column found in {name}")
            return None
        
        # Filter rare types
        vc = adata.obs[cell_type_col].value_counts()
        valid_types = vc[vc >= 30].index
        adata = adata[adata.obs[cell_type_col].isin(valid_types)].copy()
        n_classes = len(valid_types)
        
        print(f"  Cells: {original_cells:,} → {adata.n_obs:,} (subsampled)")
        print(f"  Classes: {n_classes}")
        
        # Normalize FIRST (required for HVG selection)
        if adata.X.max() > 50:  # Likely raw counts
            print("  Normalizing (log1p to 10,000 counts)...")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            print("  Data appears already normalized")
        
        # Keep full adata for CellTypist (it expects all genes)
        adata_full = adata.copy()
        
        # Create HVG subset for LogStruct/sklearn
        print(f"  Selecting top {n_genes:,} HVGs...")
        adata_hvg = adata.copy()
        try:
            sc.pp.highly_variable_genes(adata_hvg, n_top_genes=n_genes, flavor='seurat_v3', 
                                         layer=None, subset=False)
            adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable].copy()
        except Exception as e:
            print(f"  Warning: HVG selection failed ({e}), using variance-based selection")
            gene_vars = np.var(adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else adata_hvg.X, axis=0)
            top_genes = np.argsort(gene_vars)[-n_genes:]
            adata_hvg = adata_hvg[:, top_genes].copy()
        
        print(f"  HVG subset: {adata_hvg.n_obs:,} cells × {adata_hvg.n_vars:,} genes")
        
        X = adata_hvg.X.toarray() if hasattr(adata_hvg.X, 'toarray') else adata_hvg.X
        le = LabelEncoder()
        y = le.fit_transform(adata_hvg.obs[cell_type_col])
        gene_names = np.array(adata_hvg.var_names)
        
        # Split - get indices for proper train/test separation
        from sklearn.model_selection import train_test_split as tts_idx
        indices = np.arange(len(y))
        train_indices, test_indices = tts_idx(indices, test_size=0.2, stratify=y, random_state=42)
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        # Get corresponding observation names for CellTypist
        train_obs_names = adata_hvg.obs.index[train_indices]
        test_obs_names = adata_hvg.obs.index[test_indices]
        
        # Feature selection for LogStruct
        sel = SelectKBest(f_classif, k=min(n_genes, X_train.shape[1]))
        X_train_sel = sel.fit_transform(X_train, y_train)
        X_test_sel = sel.transform(X_test)
        selected_genes = gene_names[sel.get_support()]
        
        # Prior
        print(f"  Fetching PPI...")
        prior = fetch_string_ppi(selected_genes)
        
        results = {'dataset': name, 'n_cells': adata.n_obs, 'n_classes': n_classes}

        
        # --- LogStruct (Optimized) ---
        print(f"  Training LogStruct...")
        t0 = time.time()

        clf_ls = LogStructClassifier(
        prior_adjacency=prior,
        lambda_en=0.001,          # Optimized: Very light regularization
        lambda_smooth=0.01,       # Optimized: Light smoothing helps!
        lambda_kl=0.0,            # Optimized: No KL needed
        alpha=0.5,
        max_iter=1000,
        learning_rate=0.02,       # Optimized: Higher LR
        early_stopping=False,
        n_iter_no_change=50,
        tol=1e-5,
        verbose=True,
        random_state=42,
    )
        clf_ls.fit(X_train_sel, y_train)
        y_pred_ls = clf_ls.predict(X_test_sel)
        t_ls = time.time() - t0
        
        results['logstruct_acc'] = accuracy_score(y_test, y_pred_ls)
        results['logstruct_f1'] = f1_score(y_test, y_pred_ls, average='macro')
        results['logstruct_time'] = t_ls
        
        # --- Sklearn LR ---
        print(f"  Training Sklearn LR...")
        t0 = time.time()
        clf_lr = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1, random_state=42)
        clf_lr.fit(X_train_sel, y_train)
        y_pred_lr = clf_lr.predict(X_test_sel)
        t_lr = time.time() - t0
        
        results['sklearn_acc'] = accuracy_score(y_test, y_pred_lr)
        results['sklearn_f1'] = f1_score(y_test, y_pred_lr, average='macro')
        results['sklearn_time'] = t_lr
        
        # --- CellTypist (uses FULL gene set, but ONLY training data) ---
        if CELLTYPIST_AVAILABLE:
            print(f"  Training CellTypist...")
            try:
                # IMPORTANT: Train CellTypist ONLY on training cells (not test cells!)
                adata_train = adata_full[train_obs_names].copy()
                adata_test_ct = adata_full[test_obs_names].copy()
                
                adata_train.obs['labels'] = adata_train.obs[cell_type_col]
                
                t0 = time.time()
                ct_model = celltypist.train(
                    adata_train, 
                    labels='labels',
                    n_jobs=4,
                    feature_selection=True,  # Let CellTypist do feature selection
                    check_expression=False,
                    use_SGD=True,
                    mini_batch=True
                )
                
                predictions = celltypist.annotate(adata_test_ct, model=ct_model)
                y_pred_ct = predictions.predicted_labels['predicted_labels'].values
                
                # Map predictions to indices
                ct_le = LabelEncoder()
                ct_le.fit(adata_train.obs['labels'])
                
                # Handle unknown labels
                y_pred_ct_enc = []
                for label in y_pred_ct:
                    if label in ct_le.classes_:
                        y_pred_ct_enc.append(ct_le.transform([label])[0])
                    else:
                        y_pred_ct_enc.append(0)  # Assign to most common class
                y_pred_ct_enc = np.array(y_pred_ct_enc)
                
                y_test_ct = ct_le.transform(adata_test_ct.obs[cell_type_col])
                
                t_ct = time.time() - t0
                
                results['celltypist_acc'] = accuracy_score(y_test_ct, y_pred_ct_enc)
                results['celltypist_f1'] = f1_score(y_test_ct, y_pred_ct_enc, average='macro')
                results['celltypist_time'] = t_ct
            except Exception as e:
                print(f"    ⚠️ CellTypist failed: {e}")
                import traceback
                traceback.print_exc()
                results['celltypist_acc'] = None
                results['celltypist_f1'] = None
                results['celltypist_time'] = None
        else:
            results['celltypist_acc'] = None
            results['celltypist_f1'] = None
            results['celltypist_time'] = None


        
        # Summary
        print(f"\n  📈 Results for {name}:")
        print(f"     LogStruct:  {results['logstruct_acc']:.4f} acc, {results['logstruct_f1']:.4f} F1, {t_ls:.1f}s")
        print(f"     Sklearn LR: {results['sklearn_acc']:.4f} acc, {results['sklearn_f1']:.4f} F1, {t_lr:.1f}s")
        if results.get('celltypist_acc') is not None:
            print(f"     CellTypist: {results['celltypist_acc']:.4f} acc, {results['celltypist_f1']:.4f} F1, {results['celltypist_time']:.1f}s")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Error benchmarking {name}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Benchmark")
    parser.add_argument('--data_dir', default='./benchmark_data', help='Data directory')
    parser.add_argument('--output_dir', default='./multi_dataset_results', help='Output directory')
    parser.add_argument('--max_cells', type=int, default=300000, help='Max cells per dataset')
    parser.add_argument('--datasets', nargs='+', default=None, help='Specific datasets to run (default: all)')
    parser.add_argument('--download_only', action='store_true', help='Only download, do not benchmark')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_run = args.datasets if args.datasets else list(DATASET_URLS.keys())
    
    # Download
    print("\n📥 DOWNLOADING DATASETS...")
    for name in datasets_to_run:
        if name in DATASET_URLS:
            download_dataset(name, DATASET_URLS[name], data_dir)
    
    if args.download_only:
        print("\n✅ Download complete.")
        return
    
    # Benchmark
    all_results = []
    for name in datasets_to_run:
        result = benchmark_dataset(name, data_dir, max_cells=args.max_cells)
        if result:
            all_results.append(result)
    
    # Compile Results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / 'multi_dataset_benchmark.csv', index=False)
        
        print("\n" + "="*80)
        print("📊 FINAL RESULTS SUMMARY")
        print("="*80)
        
        # Calculate wins
        ls_wins_vs_sk = sum(1 for r in all_results if r['logstruct_acc'] > r['sklearn_acc'])
        ls_wins_vs_ct = sum(1 for r in all_results if r.get('celltypist_acc') and r['logstruct_acc'] > r['celltypist_acc'])
        total = len(all_results)
        ct_total = sum(1 for r in all_results if r.get('celltypist_acc') is not None)
        
        print(f"\n{'Dataset':<18} {'LogStruct':<12} {'Sklearn LR':<12} {'CellTypist':<12} {'Winner'}")
        print("-"*75)
        for r in all_results:
            ct_acc = f"{r['celltypist_acc']:.4f}" if r.get('celltypist_acc') else "N/A"
            
            # Determine winner
            accs = [('LogStruct', r['logstruct_acc']), ('Sklearn', r['sklearn_acc'])]
            if r.get('celltypist_acc'): accs.append(('CellTypist', r['celltypist_acc']))
            winner = max(accs, key=lambda x: x[1])[0]
            if winner == 'LogStruct': winner = 'LogStruct ✓'
            
            print(f"{r['dataset']:<18} {r['logstruct_acc']:.4f}       {r['sklearn_acc']:.4f}       {ct_acc:<12} {winner}")
        
        print("-"*75)
        print(f"\n🏆 LogStruct vs Sklearn LR:  {ls_wins_vs_sk}/{total} wins")
        if ct_total > 0:
            print(f"🏆 LogStruct vs CellTypist:  {ls_wins_vs_ct}/{ct_total} wins")
        print(f"\n   Mean LogStruct Acc:  {np.mean([r['logstruct_acc'] for r in all_results]):.4f}")
        print(f"   Mean Sklearn Acc:    {np.mean([r['sklearn_acc'] for r in all_results]):.4f}")
        ct_accs = [r['celltypist_acc'] for r in all_results if r.get('celltypist_acc')]
        if ct_accs:
            print(f"   Mean CellTypist Acc: {np.mean(ct_accs):.4f}")
        
        print(f"\n✅ Results saved to {output_dir}/multi_dataset_benchmark.csv")

if __name__ == "__main__":
    main()
