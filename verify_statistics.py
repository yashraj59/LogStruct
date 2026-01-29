import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import sys
import os
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
from sklearn.calibration import calibration_curve

# Add current directory to path
sys.path.append(os.getcwd())
pypi_path = Path("/Users/yashraj/repos/logstruct")
if pypi_path.exists(): sys.path.append(str(pypi_path))

try:
    from logstruct import LogStructClassifier
except ImportError:
    print("⚠️  Warning: LogStruct import failed.")

# ==========================================
# UTILITIES
# ==========================================

def fetch_string_ppi(gene_names):
    """Reuse consistent PPI fetch logic."""
    # (Simplified for brevity in this script, using same logic as main pipeline)
    print(f"  Fetching PPI for {len(gene_names)} genes...")
    string_api_url = "https://string-db.org/api/json/network"
    batch_size = 400
    all_interactions = []
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    
    for i in range(0, len(gene_names), batch_size):
        batch = gene_names[i:i+batch_size]
        params = {"identifiers": "%0d".join(batch), "species": 9606, "required_score": 700, "caller_identity": "logstruct_stats"}
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

def mcnemar_test(y_true, y_pred1, y_pred2):
    """Perform McNemar's Test for significant difference."""
    # Contingency Table:
    #          Model 2 Correct  Model 2 Wrong
    # M1 Correct      a               b
    # M1 Wrong        c               d
    
    # We care about b and c (disagreements)
    # H0: b = c (Models perform same)
    
    m1_correct = (y_pred1 == y_true)
    m2_correct = (y_pred2 == y_true)
    
    b = np.sum(m1_correct & ~m2_correct) # M1 correct, M2 wrong
    c = np.sum(~m1_correct & m2_correct) # M1 wrong, M2 correct
    
    # Chi-square statistic
    statistic = ((abs(b - c) - 1.0) ** 2) / (b + c + 1e-10)
    p_value = stats.chi2.sf(statistic, 1)
    return statistic, p_value, b, c

# ==========================================
# MAIN
# ==========================================

def run_verification(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("📂 Loading Data...")
    adata = sc.read_h5ad(Path(args.data_dir) / f"{args.dataset}.h5ad")
    adata.X = adata.raw.X.copy()
    if args.max_cells and adata.n_obs > args.max_cells:
        sc.pp.subsample(adata, n_obs=args.max_cells, random_state=42)
    if adata.raw is not None: adata = adata.raw.to_adata()
    
    # Filter & Normalize
    vc = adata.obs['cell_type'].value_counts()
    adata = adata[adata.obs['cell_type'].isin(vc[vc >= 50].index)].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True) # Normalized only choice here for speed/consist
    
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    le = LabelEncoder()
    y = le.fit_transform(adata.obs['cell_type'])
    gene_names = np.array(adata.var_names)
    
    # Prior
    prior = fetch_string_ppi(gene_names)
    
    # 2. Setup Models
    models = {
        'LogStruct (Opt)': {
            'type': 'LS', 'params': {'lambda_en':0.001, 'lambda_smooth':0.01, 'l_rate':0.02}
        },
        'LogStruct (NoGraph)': {
             'type': 'LS', 'params': {'lambda_en':0.001, 'lambda_smooth':0.0, 'l_rate':0.02} # Baseline
        },
        'Sklearn LR': {
            'type': 'SK', 'params': {'C': 1.0}
        }
    }
    
    results = {k: {'acc': [], 'probs': [], 'preds': [], 'y_true': []} for k in models}
    
    # 3. 5-Fold Cross Validation
    print("\n🔄 Starting 5-Fold Cross Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/5...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Feature Select per fold
        sel = SelectKBest(f_classif, k=3000)
        X_train_sel = sel.fit_transform(X_train, y_train)
        X_test_sel = sel.transform(X_test)
        
        # Update prior for selection? 
        # Actually to be rigorous we should slice prior. 
        # But indices change. Let's assume indices map correctly if we subset?
        # SelectKBest changes columns. We need to slice prior.
        mask = sel.get_support()
        fold_prior = prior[mask][:, mask]
        
        for name, config in models.items():
            if config['type'] == 'LS':
                p = config['params']
                clf = LogStructClassifier(
                    prior_adjacency=fold_prior,
                    lambda_en=p['lambda_en'],
                    lambda_smooth=p['lambda_smooth'],
                    lambda_kl=0.0,
                    learning_rate=p['l_rate'],
                    max_iter=500,
                    early_stopping=True,
                    random_state=42,
                    verbose=False
                )
                clf.fit(X_train_sel, y_train)
                y_p = clf.predict(X_test_sel)
                y_prob = clf.predict_proba(X_test_sel)
            else:
                clf = LogisticRegression(max_iter=500, solver='sag') # Fast solver
                clf.fit(X_train_sel, y_train)
                y_p = clf.predict(X_test_sel)
                y_prob = clf.predict_proba(X_test_sel)
                
            acc = accuracy_score(y_test, y_p)
            results[name]['acc'].append(acc)
            results[name]['preds'].extend(y_p)
            results[name]['probs'].extend(y_prob)
            results[name]['y_true'].extend(y_test) # Same for all but good to track
            
    # 4. Analysis
    print("\n📊 Results Summary:")
    print(f"{'Model':<25} {'Mean Acc':<10} {'Std Dev':<10}")
    print("-" * 45)
    
    folds_acc = {}
    for name in models:
        accs = np.array(results[name]['acc'])
        folds_acc[name] = accs
        print(f"{name:<25} {np.mean(accs):.4f}     +/-{np.std(accs):.4f}")
        
    # Statistical Tests
    print("\n🧪 Significance Testing:")
    
    # T-Tests (Paired)
    baseline = 'LogStruct (Opt)'
    comparisons = ['LogStruct (NoGraph)', 'Sklearn LR']
    
    for comp in comparisons:
        # t-test on folds
        t_stat, p_val = stats.ttest_rel(folds_acc[baseline], folds_acc[comp])
        sig = "*" if p_val < 0.05 else "ns"
        print(f"  Paired t-test ({baseline} vs {comp}): p={p_val:.4f} ({sig})")
        
        # McNemar (Aggregated)
        y_true = np.array(results[baseline]['y_true'])
        y_pred1 = np.array(results[baseline]['preds'])
        y_pred2 = np.array(results[comp]['preds'])
        stat, p_mc, b, c = mcnemar_test(y_true, y_pred1, y_pred2)
        sig_mc = "*" if p_mc < 0.05 else "ns"
        print(f"  McNemar Test  ({baseline} vs {comp}): p={p_mc:.4e} ({sig_mc}) [Wins: {b}, Losses: {c}]")

    # Calibration Plot
    print("\n📈 Generatng Calibration Plot...")
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for name in models:
        # Flatten probs and true labels (Micro-average calibration)
        # Convert multiclass y_true to binary (1 vs rest) or just flatten?
        # Standard way is one-vs-rest calibration or just collecting all confidence scores.
        # Let's do: for each prediction, take prob of predicted class vs whether it was correct.
        
        y_true_all = np.array(results[name]['y_true'])
        probs_all = np.array(results[name]['probs'])
        preds_all = np.array(results[name]['preds'])
        
        # Get prob of the predicted class
        confidences = np.max(probs_all, axis=1)
        is_correct = (preds_all == y_true_all).astype(int)
        
        prob_true, prob_pred = calibration_curve(is_correct, confidences, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{name}")

    plt.ylabel("Fraction of positives (Accuracy)")
    plt.xlabel("Mean predicted value (Confidence)")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend(loc="lower right")
    plt.savefig(output_dir / 'calibration_plot.png')
    print(f"  Saved to {output_dir}/calibration_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='blood')
    parser.add_argument('--data_dir', default='./benchmark_data')
    parser.add_argument('--output_dir', default='./statistical_results')
    parser.add_argument('--max_cells', type=int, default=20000)
    args = parser.parse_args()
    run_verification(args)
