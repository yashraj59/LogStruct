
import numpy as np

def compare_to_prior(model, gene_names=None, prior_threshold=0.5, learned_threshold=0.1):
    """
    Compare learned adjacency to prior adjacency.
    
    Args:
        model: Trained LogStructClassifier or LogStructRegressor
        gene_names: List of gene names (optional)
        prior_threshold: Threshold for edge existence in prior
        learned_threshold: Threshold for edge existence in learned model
        
    Returns:
        Dictionary containing:
        - gained: List of (i, j, weight) for edges NOT in prior usually
        - lost: List of (i, j, prior_weight, learned_weight)
        - retained: List of (i, j, weight)
        - stats: Summary counts
        - top_edges: Top edges global
    """
    if not hasattr(model, 'adjacency_') or model.adjacency_ is None:
        raise ValueError("Model has not learned an adjacency matrix yet.")
        
    adj_learned = model.adjacency_
    
    # Check if model has prior stored
    if hasattr(model, 'prior_adjacency'):
        adj_prior = model.prior_adjacency
    else:
        # Fallback if prior is not stored in model (though it should be)
        raise ValueError("Model does not have 'prior_adjacency' attribute.")

    n_features = adj_learned.shape[0]
    if gene_names is None:
        gene_names = [f"Feature_{i}" for i in range(n_features)]
        
    # Masks
    prior_mask = adj_prior > prior_threshold
    learned_mask = adj_learned > learned_threshold
    
    # 1. Gained
    gained_mask = learned_mask & (~prior_mask)
    gained_indices = np.where(gained_mask)
    gained_edges = []
    for i, j in zip(*gained_indices):
        if i < j:
            gained_edges.append((i, j, adj_learned[i, j]))
    gained_edges.sort(key=lambda x: x[2], reverse=True)
    
    # 2. Lost (Strict pruning check)
    lost_mask = prior_mask & (adj_learned < 0.01)
    lost_indices = np.where(lost_mask)
    lost_edges = []
    for i, j in zip(*lost_indices):
        if i < j:
            lost_edges.append((i, j, adj_prior[i, j], adj_learned[i, j]))
    lost_edges.sort(key=lambda x: x[2], reverse=True) # Sort by prior weight
    
    # 3. Retained (Strong)
    retained_mask = prior_mask & (adj_learned > 0.9)
    retained_indices = np.where(retained_mask)
    retained_edges = []
    for i, j in zip(*retained_indices):
        if i < j:
            retained_edges.append((i, j, adj_learned[i, j]))
    retained_edges.sort(key=lambda x: x[2], reverse=True)
    
    # 4. Global Top Edges
    top_edges_list = model.get_top_edges(n=30)
    
    result = {
        'gained': gained_edges,
        'lost': lost_edges,
        'retained': retained_edges,
        'top_edges': top_edges_list,
        'stats': {
            'n_gained': len(gained_edges),
            'n_lost': len(lost_edges),
            'n_retained': len(retained_edges)
        },
        'gene_names': gene_names,
        'prior_threshold': prior_threshold
    }
    return result

def print_network_report(analysis_result, n_top=10):
    """Print a formatted report of the network analysis."""
    res = analysis_result
    genes = res['gene_names']
    
    print("\n" + "="*50)
    print("NETWORK EVOLUTION ANALYSIS")
    print("="*50)
    
    # Gained
    print(f"\n[+] GAINED CONNECTIONS (Learned high weight, NOT in Prior)")
    print(f"    Total gained: {res['stats']['n_gained']}")
    if res['stats']['n_gained'] > 0:
        print(f"    Top {n_top} new interactions:")
        print(f"    {'Gene A':<15} {'Gene B':<15} {'Weight':<10}")
        print("    " + "-" * 40)
        for i, j, w in res['gained'][:n_top]:
             print(f"    {genes[i]:<15} {genes[j]:<15} {w:.4f}")
             
    # Lost
    print(f"\n[-] LOST CONNECTIONS (In Prior, but PRUNED by model)")
    print(f"    Total pruned: {res['stats']['n_lost']}")
    if res['stats']['n_lost'] > 0:
        print(f"    Top {n_top} pruned interactions:")
        print(f"    {'Gene A':<15} {'Gene B':<15} {'Prior':<10} {'Learned':<10}")
        print("    " + "-" * 50)
        for i, j, wp, wl in res['lost'][:n_top]:
             print(f"    {genes[i]:<15} {genes[j]:<15} {wp:.4f}     {wl:.4f}")
             
    # Retained
    print(f"\n[=] RETAINED & STRENGTHENED")
    print(f"    Total: {res['stats']['n_retained']} edges kept strong (>0.9)")
    
    # Global Top
    print("\n" + "="*50)
    print(f"TOP {n_top} LEARNED EDGES (GLOBAL)")
    print("="*50)
    print(f"{'Gene A':<15} {'Gene B':<15} {'Weight':<10} {'In Prior?'}")
    print("-" * 55)
    
    # Need to access prior to check existence for global list
    # Passed result doesn't have raw prior, but we have thresholds
    # We can infer from 'gained' list membership or just check manually if adjacency was passed
    # For simplicity, we assume if it's in top edges but NOT in gained, it was in prior.
    
    gained_set = set([(i,j) for i,j,w in res['gained']])
    
    for i, j, w in res['top_edges'][:20]: # Show 20
        pair = (min(i,j), max(i,j))
        in_prior_str = "NO (New)" if pair in gained_set else "Yes"
        print(f"{genes[i]:<15} {genes[j]:<15} {w:.4f}     {in_prior_str}")
