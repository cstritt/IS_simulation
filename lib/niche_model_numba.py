#!/usr/bin/env python3
"""
Niche constraint model for IS6110 with numba JIT acceleration.

Core mechanism:
  - p_essential: probability a genomic site is essential (lethal for insertions)
  - p_tolerated: among non-essential sites, probability of being "tolerated" 
                 (purging allowed; absorbs mutations)
  - r_birth: transposition rate per occupied site per unit branch length
  - r_purge: purging rate for tolerated sites per unit branch length
  - Terminal branches: no purging (singletons persist)

Site types:
  -1: essential (cannot host insertions)
   0: non-essential, non-tolerated (insertions allowed, no purging)
   1: non-essential, tolerated (insertions allowed, purging occurs)
"""

import argparse
import random
import numpy as np
import pandas as pd
from numba import njit
from ete3 import Tree


# ============================================================================
# Numba-accelerated branch simulation
# ============================================================================

@njit
def _simulate_branch_numba(occ, branch_length, r_birth, site_type, r_purge, is_terminal):
    """
    Fast branch simulation with Gillespie algorithm.
    
    Args:
        occ: occupancy vector (0=empty, 1=occupied)
        branch_length: time interval
        r_birth: transposition rate per occupied site
        site_type: array of site types (-1=essential, 0=neutral, 1=tolerated)
        r_purge: purging rate for tolerated sites (ignored if is_terminal=True)
        is_terminal: if True, no purging occurs
    
    Returns:
        Updated occupancy vector
    """
    L = len(occ)
    occ = occ.copy()
    
    if branch_length <= 0:
        return occ
    
    t = 0.0
    
    while t < branch_length:
        # Count occupied sites
        n_occ = 0
        for i in range(L):
            if occ[i] == 1:
                n_occ += 1
        
        if n_occ == 0:
            break
        
        # Count empty compatible sites
        n_empty = 0
        for i in range(L):
            if occ[i] == 0 and site_type[i] >= 0:
                n_empty += 1
        
        # Count tolerated occupied sites (for purging)
        n_tol_occ = 0
        if not is_terminal:
            for i in range(L):
                if occ[i] == 1 and site_type[i] == 1:
                    n_tol_occ += 1
        
        # Rates
        birth_rate = r_birth * float(n_occ) if n_empty > 0 else 0.0
        purge_rate = 0.0 if is_terminal else r_purge * float(n_tol_occ)
        total_rate = birth_rate + purge_rate
        
        if total_rate <= 0.0:
            break
        
        # Poisson shortcut for very short remaining time
        #remaining = branch_length - t
        """"
        if total_rate * remaining < 0.01:
            n_events = np.random.poisson(total_rate * remaining)
            for _ in range(n_events):
                event_prob = np.random.random()
                if event_prob < birth_rate / total_rate and n_empty > 0:
                    # Birth event: pick random empty compatible site
                    idx = 0
                    count = 0
                    r_idx = np.random.randint(0, n_empty)
                    for i in range(L):
                        if occ[i] == 0 and site_type[i] >= 0:
                            if count == r_idx:
                                occ[i] = 1
                                break
                            count += 1
                elif n_tol_occ > 0:
                    # Purge event: pick random tolerated occupied site
                    idx = 0
                    count = 0
                    r_idx = np.random.randint(0, n_tol_occ)
                    for i in range(L):
                        if occ[i] == 1 and site_type[i] == 1:
                            if count == r_idx:
                                occ[i] = 0
                                break
                            count += 1
            break
        """
        
        # Gillespie: exponential wait time
        dt = np.random.exponential(1.0 / total_rate)
        if t + dt > branch_length:
            break
        t += dt
        
        # Execute event
        event_prob = np.random.random()
        if event_prob < birth_rate / total_rate:
            # Birth event
            if n_empty > 0:
                idx = 0
                count = 0
                r_idx = np.random.randint(0, n_empty)
                for i in range(L):
                    if occ[i] == 0 and site_type[i] >= 0:
                        if count == r_idx:
                            occ[i] = 1
                            break
                        count += 1
        else:
            # Purge event
            if n_tol_occ > 0:
                idx = 0
                count = 0
                r_idx = np.random.randint(0, n_tol_occ)
                for i in range(L):
                    if occ[i] == 1 and site_type[i] == 1:
                        if count == r_idx:
                            occ[i] = 0
                            break
                        count += 1
    
    return occ


# ============================================================================
# Niche space initialization
# ============================================================================

def create_niche_space(L, p_essential, p_tolerated, initial_copies=1):
    """
    Create a niche space with heterogeneous tolerance.
    
    Args:
        L: length of genome (number of sites)
        p_essential: fraction of sites that are essential
        p_tolerated: among non-essential, fraction that are tolerated
        initial_copies: number of occupied sites at root
    
    Returns:
        occ (ndarray): occupancy (0/1)
        site_type (ndarray): site classification (-1/0/1)
    """
    site_type = np.full(L, -1, dtype=np.int8)  # default: essential
    
    # Mark non-essential sites
    non_ess = np.where(np.random.rand(L) >= p_essential)[0]
    site_type[non_ess] = 0  # non-essential, non-tolerated
    
    # Among non-essential, mark some as tolerated
    n_tol = int(min(len(non_ess) * p_tolerated, max(0, len(non_ess) - initial_copies)))
    if n_tol > 0:
        tol = np.random.choice(non_ess, n_tol, replace=False)
        site_type[tol] = 1
    
    # Seed initial copies in non-essential sites
    neutral = np.where(site_type == 0)[0]
    init_sites = np.random.choice(neutral, size=min(initial_copies, len(neutral)), replace=False)
    occ = np.zeros(L, dtype=np.int8)
    occ[init_sites] = 1
    
    return occ, site_type


# ============================================================================
# Tree traversal
# ============================================================================

def traverse_and_simulate(node, occ, site_type, params, tip_results):
    """
    Recursive depth-first traversal with branch simulation.
    
    Args:
        node: ete3 TreeNode
        occ: occupancy array
        site_type: site type array
        params: dict with 'r_birth', 'r_purge'
        tip_results: dict to accumulate tip occupancy states
    
    Returns:
        tip_results (modified in place)
    """
    branch_length = node.dist if node.dist is not None else 0.0
    
    # Simulate this branch
    end_occ = _simulate_branch_numba(
        occ,
        float(branch_length),
        float(params["r_birth"]),
        site_type,
        float(params.get("r_purge", 0.0)),
        node.is_leaf()
    )
    
    if node.is_leaf():
        tip_results[node.name] = end_occ
    else:
        for child in node.children:
            traverse_and_simulate(child, end_occ.copy(), site_type, params, tip_results)
    
    return tip_results


# ============================================================================
# Summary statistics
# ============================================================================

def _gini(arr):
    """Gini coefficient of occupancy."""
    arr = np.sort(arr.astype(float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2.0 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1.0) / n


def get_site_frequency_spectrum(mat):
    """
    Compute site frequency spectrum and related statistics.
    
    Args:
        mat: tip_occupancy matrix (tips × sites)
    
    Returns:
        dict with SFS metrics
    """
    occupancy = (mat == 1).sum(axis=0)
    compat = (mat >= 0).any(axis=0)
    occ = occupancy[compat]
    
    n_tips = mat.shape[0]
    
    if len(occ) == 0:
        return {
            "n_singletons": 0,
            "n_doubletons": 0,
            "n_rare": 0,
            "n_common": 0,
            "singleton_prop": 0.0,
            "tajimas_d": 0.0
        }
    
    n_singletons = int((occ == 1).sum())
    singleton_prop = n_singletons / len(occ)
    
    # Tajima's D proxy: neutral expectation of singleton fraction is 1/H_{n-1}
    expected_singleton = (
        1.0 / np.sum(1.0 / np.arange(1, n_tips))
        if n_tips > 1 else 0.5
    )
    tajimas_d = singleton_prop - expected_singleton
    
    return {
        "n_singletons": n_singletons,
        "n_doubletons": int((occ == 2).sum()),
        "n_rare": int((occ <= 5).sum()),
        "n_common": int((occ > 0.1 * n_tips).sum()),
        "singleton_prop": float(singleton_prop),
        "tajimas_d": float(tajimas_d)
    }


def get_summary_stats(tip_niches, tip_names, lineage_idx, unique_lineages):
    """
    Compute summary statistics for ABC.
    
    Args:
        tip_niches: dict mapping tip names to occupancy arrays
        tip_names: list of tip names in consistent order
        lineage_idx: array of lineage indices for each tip
        unique_lineages: list of unique lineages
    
    Returns:
        dict with summary statistics
    """
    # Copy number per tip
    cn = np.array([tip_niches[name].sum() for name in tip_names], dtype=float)
    
    # Basic copy number stats
    stats = {
        "mean_cn": float(cn.mean()),
        "std_cn": float(cn.std()),
        "min_cn": float(cn.min()),
        "q25_cn": float(np.percentile(cn, 25)),
        "median_cn": float(np.percentile(cn, 50)),
        "q75_cn": float(np.percentile(cn, 75)),
        "max_cn": float(cn.max())
    }
    
    # Lineage stats
    lineage_df = pd.DataFrame({
        "lineage": [unique_lineages[idx] for idx in lineage_idx],
        "copy_number": cn
    })

    lineage_summary = lineage_df.groupby("lineage")["copy_number"].agg(["mean", "median", "std", "min", "max","var"])
    stats["lineage_var"] = float(np.var(lineage_summary["mean"], ddof=1))
    stats["lineage_sd"] = float(np.std(lineage_summary["mean"], ddof=1))
    
    
    # Occupancy and site frequency spectrum
    mat = np.vstack([tip_niches[name] for name in tip_names])
    occupancy = (mat == 1).sum(axis=0)
    compat = (mat >= 0).any(axis=0)
    occ = occupancy[compat]
    
    if len(occ) > 0:
        freq = occ / len(tip_names)
        stats["gini_occupancy"] = float(_gini(occ))
        stats["n_sites_freq_05"] = int((freq >= 0.05).sum())
        stats["n_sites_freq_10"] = int((freq >= 0.10).sum())
        stats["max_occupancy"] = int(occ.max())
        stats["mean_occupancy"] = float(occ.mean())
    else:
        stats["gini_occupancy"] = 0.0
        stats["n_sites_freq_05"] = 0
        stats["n_sites_freq_10"] = 0
        stats["max_occupancy"] = 0
        stats["mean_occupancy"] = 0.0
    
    # SFS
    sfs = get_site_frequency_spectrum(mat)
    stats.update({
        "n_singletons": sfs["n_singletons"],
        "n_doubletons": sfs["n_doubletons"],
        "n_rare": sfs["n_rare"],
        "n_common": sfs["n_common"],
        "singleton_prop": sfs["singleton_prop"],
        "tajimas_d": sfs["tajimas_d"]
    })
    
    return stats


# ============================================================================
# Standalone simulation
# ============================================================================

def run_simulation(tree, L, p_essential, p_tolerated, r_birth, r_purge,
                   tip_names, lineage_idx, unique_lineages, seed=None, get_stats=True):
    """
    Run a single niche model simulation.
    
    Returns:
        params, cn, stats dicts
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Create niche space
    occ, site_type = create_niche_space(L, p_essential, p_tolerated, initial_copies=1)
    
    # Simulate
    tip_results = {}
    traverse_and_simulate(tree, occ, site_type, 
                         {"r_birth": r_birth, "r_purge": r_purge},
                         tip_results)
    
    # Summarize
    cn = np.array([tip_results[name].sum() for name in tip_names], dtype=float)
    
    params = {
        "p_essential": p_essential,
        "p_tolerated": p_tolerated,
        "r_birth": r_birth,
        "r_purge": r_purge
    }
    
    if get_stats:
        stats = get_summary_stats(tip_results, tip_names, lineage_idx, unique_lineages)
        return params, cn, stats
    else:
        return params, cn


if __name__ == "__main__":
    import argparse
    
    # Parse command line
    args = argparse.ArgumentParser()
    args.add_argument("--tree", type=str)
    args.add_argument("--niches", type=int)
    args.add_argument("--p_essential", type=float)
    args.add_argument("--p_tolerated", type=float)
    args.add_argument("--r_birth", type=float)
    args.add_argument("--r_purge", type=float)
    args.add_argument("--metadata", type=str)
    args.add_argument("--seed", type=int, default=42)
    args = args.parse_args()
    
    # Parse tree
    tree = Tree(args.tree, format=1)
    tip_names = sorted(tree.get_leaf_names())
    
    # Create lineage index
    metadata = pd.read_csv(args.metadata, sep='\t')
    lineage_map = dict(zip(metadata['GNUMBER'], metadata['LINEAGE_x']))
    lineage_idx = np.array([lineage_map[name] for name in tip_names])
    unique_lineages = np.unique(lineage_idx)
    
    # Example: simulate with fixed parameters
    params, cn = run_simulation(
        tree, args.niches,
        p_essential=args.p_essential,
        p_tolerated=args.p_tolerated,
        r_birth=args.r_birth,
        r_purge=args.r_purge,
        tip_names=tip_names,
        lineage_idx=lineage_idx,
        unique_lineages=unique_lineages,
        seed=args.seed,
        get_stats=False
    )
    
    # Write to stdout: parameters, then strain name and copy number    
    #print(f"{params['p_essential']}\t{params['p_tolerated']}\t{params['r_birth']}\t{params['r_purge']}")
    print("strain\tcopy_number\n")
    for i, name in enumerate(tip_names):
        print(f"{name}\t{cn[i]}")
