#!/usr/bin/env python3
"""
Niche constraint model for IS6110 with distribution of fitness effects (DFE).

Core mechanism:
  - Genomic sites have fitness effects drawn from gamma DFE
  - p_neutral: fraction of sites with fitness = 0 (neutral)
  - gamma_shape, gamma_scale: DFE parameters for deleterious sites
  - r_birth: transposition rate per occupied site per unit branch length
  - r_loss: purging rate coefficient per unit fitness effect per branch length
  - Birth events target sites weighted by targetability preferences
  - Loss events occur with probability proportional to fitness effect

Parameters for ABC:
  p_neutral: prob(fitness=0) ~ U(0.1, 0.99)
  gamma_shape: shape parameter of gamma DFE ~ U(0.5, 3.0)
  gamma_scale: scale parameter ~ U(0.1, 1.0)
  r_birth: transposition rate ~ U(1000, 10000)
  r_loss: purging strength ~ U(0, 5000)
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
def _simulate_branch_numba(occ, branch_length, r_birth, fitness, r_loss, targetability):
    """
    Gillespie simulation of IS dynamics on a single branch.
    
    Args:
        occ: occupancy array (0=empty, 1=occupied), shape (L,)
        branch_length: duration of branch (time units)
        r_birth: transposition rate per occupied site (units: 1/time)
        fitness: fitness effects of each site (shape L,), where fitness[i]=0 => neutral
        r_loss: purging rate coefficient (units: 1/time/fitness_unit)
        targetability: insertion preference at each site (shape L,), normalized weights
    
    Returns:
        occ: updated occupancy array
    """
    occ = occ.copy()
    L = len(occ)
    t = 0.0

    while t < branch_length:
        # Count occupied sites and total loss rate
        n_occ = 0
        total_loss_rate = 0.0
        total_target_weight = 0.0

        for i in range(L):
            if occ[i] == 1:
                n_occ += 1
                total_loss_rate += r_loss * fitness[i]
            else:
                total_target_weight += targetability[i]

        if n_occ == 0:
            break

        # Total event rate (birth + loss)
        birth_rate = r_birth * n_occ if total_target_weight > 0.0 else 0.0
        total_rate = birth_rate + total_loss_rate

        if total_rate <= 0.0:
            break

        # Gillespie waiting time
        dt = np.random.exponential(1.0 / total_rate)
        if t + dt > branch_length:
            break

        t += dt

        # Choose event type and execute
        u = np.random.random()

        if u < (birth_rate / total_rate):
            # --- Birth event: transposition to random site weighted by targetability ---
            u_birth = np.random.random() * total_target_weight
            cum = 0.0
            for i in range(L):
                if occ[i] == 0:
                    cum += targetability[i]
                    if cum >= u_birth:
                        occ[i] = 1
                        break

        else:
            # --- Loss event: deletion weighted by fitness ---
            u_loss = np.random.random() * total_loss_rate
            cum = 0.0
            for i in range(L):
                if occ[i] == 1 and fitness[i] > 0.0:
                    cum += r_loss * fitness[i]
                    if cum >= u_loss:
                        occ[i] = 0
                        break

    return occ


# ============================================================================
# Niche space initialization
# ============================================================================

def create_niche_space(L, p_neutral, gamma_shape, gamma_scale, initial_copies=1):
    """
    Create a genomic niche space with fitness effects drawn from DFE.
    
    Args:
        L: genomic size (number of sites)
        p_neutral: fraction of sites with fitness=0 (neutral)
        gamma_shape: shape parameter of gamma distribution for deleterious sites
        gamma_scale: scale parameter of gamma distribution
        initial_copies: number of initial occupied sites (randomly chosen)
    
    Returns:
        occ: occupancy array (0=empty, 1=occupied), shape (L,)
        fitness: fitness effect at each site, shape (L,)
    """
    # Draw fitness effects from gamma for all sites
    fitness = np.random.gamma(gamma_shape, gamma_scale, size=L).astype(np.float32)

    # Set a fraction to be neutral (fitness=0)
    neutral_mask = np.random.rand(L) < p_neutral
    fitness[neutral_mask] = 0.0

    # Initialize occupancy: place initial_copies at random compatible sites
    occ = np.zeros(L, dtype=np.int8)
    init_sites = np.random.choice(L, size=min(initial_copies, L), replace=False)
    occ[init_sites] = 1

    return occ, fitness


def create_targetability(L, shape=2.0, scale=None):
    """
    Create target site preference array (e.g., insertion hotspots).
    
    Args:
        L: size of genome
        shape: gamma shape parameter
        scale: gamma scale parameter (default: 1/shape for mean=1)
    
    Returns:
        targetability: weights normalized to sum to 1, shape (L,)
    """
    if scale is None:
        scale = 1.0 / shape

    t = np.random.gamma(shape, scale, size=L).astype(np.float32)
    t = t / t.sum()  # normalize to sum=1
    return t


# ============================================================================
# Tree traversal with branch simulation
# ============================================================================

def traverse_and_simulate(node, occ, fitness, targetability, params, tip_results):
    """
    Recursive depth-first tree traversal with branch simulation.
    
    Args:
        node: ete3 TreeNode
        occ: occupancy array at this node
        fitness: fitness landscape (constant across tree)
        targetability: insertion preference landscape (constant across tree)
        params: dict with 'r_birth', 'r_loss'
        tip_results: dict to accumulate final tip states (modified in place)
    """
    branch_length = node.dist if node.dist is not None else 0.0

    # Simulate the branch
    end_occ = _simulate_branch_numba(
        occ,
        float(branch_length),
        float(params["r_birth"]),
        fitness,
        float(params.get("r_loss", 0.0)),
        targetability
    )

    # Recursively process descendants
    if node.is_leaf():
        tip_results[node.name] = end_occ.copy()
    else:
        for child in node.children:
            traverse_and_simulate(child, end_occ.copy(), fitness, targetability, params, tip_results)


# ============================================================================
# Summary statistics for ABC
# ============================================================================

def _gini(arr):
    """Gini coefficient of array (0=perfect equality, 1=perfect inequality)."""
    arr = np.sort(arr.astype(float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2.0 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1.0) / n


def get_site_frequency_spectrum(mat):
    """
    Compute site frequency spectrum metrics from tip occupancy matrix.
    
    Args:
        mat: array of shape (n_tips, n_sites) with occupancy {0, 1}
    
    Returns:
        dict with SFS summary statistics
    """
    occupancy = (mat == 1).sum(axis=0)
    n_tips = mat.shape[0]

    if len(occupancy) == 0 or occupancy.sum() == 0:
        return {
            "n_singletons": 0,
            "n_doubletons": 0,
            "n_rare": 0,
            "n_common": 0,
            "singleton_prop": 0.0,
            "tajimas_d": 0.0
        }

    n_singletons = int((occupancy == 1).sum())
    singleton_prop = n_singletons / len(occupancy)

    # Tajima's D proxy: expected singleton fraction under neutrality
    H_n = np.sum(1.0 / np.arange(1, n_tips)) if n_tips > 1 else 1.0
    expected_singleton = 1.0 / H_n
    tajimas_d = singleton_prop - expected_singleton

    return {
        "n_singletons": n_singletons,
        "n_doubletons": int((occupancy == 2).sum()),
        "n_rare": int((occupancy <= 5).sum()),
        "n_common": int((occupancy > 0.1 * n_tips).sum()),
        "singleton_prop": float(singleton_prop),
        "tajimas_d": float(tajimas_d)
    }


def get_summary_stats(tip_results, tip_names, lineage_map):
    """
    Compute ABC summary statistics from final tip states.
    
    Args:
        tip_results: dict mapping tip name -> occupancy array
        tip_names: list of tip names in consistent order
        lineage_map: dict mapping tip name -> lineage ID
    
    Returns:
        dict of summary statistics
    """
    # Compute copy number per tip
    cn = np.array([tip_results[name].sum() for name in tip_names], dtype=float)

    # Basic copy number statistics
    stats = {
        "mean_cn": float(cn.mean()),
        "std_cn": float(cn.std(ddof=1)) if len(cn) > 1 else 0.0,
        "min_cn": float(cn.min()),
        "q25_cn": float(np.percentile(cn, 25)),
        "median_cn": float(np.percentile(cn, 50)),
        "q75_cn": float(np.percentile(cn, 75)),
        "max_cn": float(cn.max())
    }

    # Lineage-level statistics
    lineage_cn = {}
    for name, copy_n in zip(tip_names, cn):
        lin = lineage_map.get(name, "unknown")
        if lin not in lineage_cn:
            lineage_cn[lin] = []
        lineage_cn[lin].append(copy_n)

    # Get unique lineages from lineage map
    unique_lineages = []
    for lin in lineage_map.values():
        if lin not in unique_lineages:
            unique_lineages.append(lin)

    # Lineage variance
    lineage_means = np.array([np.mean(lineage_cn[lin]) for lin in unique_lineages], dtype=float)
    stats["lineage_var"] = float(np.var(lineage_means, ddof=1)) if len(lineage_means) > 1 else 0.0
    stats["lineage_sd"] = float(np.std(lineage_means, ddof=1)) if len(lineage_means) > 1 else 0.0

    # Site occupancy and SFS
    mat = np.vstack([tip_results[name] for name in tip_names])
    occupancy = (mat == 1).sum(axis=0)

    if len(occupancy) > 0 and occupancy.sum() > 0:
        freq = occupancy / len(tip_names)
        stats["gini_occupancy"] = float(_gini(occupancy))
        stats["n_sites_freq_05"] = int((freq >= 0.05).sum())
        stats["n_sites_freq_10"] = int((freq >= 0.10).sum())
        stats["max_occupancy"] = int(occupancy.max())
        stats["mean_occupancy"] = float(occupancy.mean())
    else:
        stats["gini_occupancy"] = 0.0
        stats["n_sites_freq_05"] = 0
        stats["n_sites_freq_10"] = 0
        stats["max_occupancy"] = 0
        stats["mean_occupancy"] = 0.0

    # Site frequency spectrum
    sfs = get_site_frequency_spectrum(mat)
    stats.update(sfs)

    return stats


# ============================================================================
# Main simulation routine
# ============================================================================

def run_simulation(tree, L, p_neutral, gamma_shape, gamma_scale, r_birth, r_loss,
                   tip_names, lineage_map, seed=None, get_stats=True):
    """
    Run a single niche model simulation.
    
    Args:
        tree: ete3 Tree object
        L: number of genomic sites
        p_neutral: fraction neutral sites
        gamma_shape, gamma_scale: DFE parameters
        r_birth: transposition rate
        r_loss: purging strength
        tip_names: list of tip names
        lineage_map: dict tip -> lineage
        seed: random seed
        get_stats: whether to compute summary statistics
    
    Returns:
        (params, cn) if get_stats=False, else (params, cn, stats)
    """
    import time
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    # Add runtime to output
    start_time = time.time()

    # Create niche space
    occ, fitness = create_niche_space(L, p_neutral, gamma_shape, gamma_scale, initial_copies=1)
    targetability = create_targetability(L, shape=2.0)

    # Simulate tree
    root = tree.get_tree_root()
    tip_results = {}
    traverse_and_simulate(
        root, occ, fitness, targetability,
        {"r_birth": r_birth, "r_loss": r_loss},
        tip_results
    )

    # Extract copy numbers
    cn = np.array([tip_results[name].sum() for name in tip_names], dtype=float)

    # Package parameters
    params = {
        "p_neutral": p_neutral,
        "gamma_shape": gamma_shape,
        "gamma_scale": gamma_scale,
        "r_birth": r_birth,
        "r_loss": r_loss
    }
    
    # Add runtime to output
    end_time = time.time()
    params["runtime"] = end_time - start_time

    if get_stats:
        stats = get_summary_stats(tip_results, tip_names, lineage_map)
        return params, cn, stats
    else:
        return params, cn


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Niche constraint model for IS6110 with DFE"
    )
    parser.add_argument("--tree", type=str, required=True, help="Newick tree file")
    parser.add_argument("--niches", type=int, required=True, help="Genomic size (number of sites)")
    parser.add_argument("--p_neutral", type=float, required=True, help="Fraction of neutral sites")
    parser.add_argument("--gamma_shape", type=float, required=True, help="DFE shape parameter")
    parser.add_argument("--gamma_scale", type=float, required=True, help="DFE scale parameter")
    parser.add_argument("--r_birth", type=float, required=True, help="Transposition rate")
    parser.add_argument("--r_loss", type=float, required=True, help="Purging strength")
    parser.add_argument("--metadata", type=str, required=True, help="Sample metadata (TSV)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load tree and metadata
    tree = Tree(args.tree, format=1)
    tip_names = sorted(tree.get_leaf_names())

    metadata = pd.read_csv(args.metadata, sep="\t")
    lineage_map = dict(zip(metadata["GNUMBER"], metadata["LINEAGE_x"]))

    # Run simulation
    params, cn = run_simulation(
        tree, args.niches,
        p_neutral=args.p_neutral,
        gamma_shape=args.gamma_shape,
        gamma_scale=args.gamma_scale,
        r_birth=args.r_birth,
        r_loss=args.r_loss,
        tip_names=tip_names,
        lineage_map=lineage_map,
        seed=args.seed,
        get_stats=False
    )

    # Output copy numbers
    print("strain\tcopy_number")
    for name, copy_n in zip(tip_names, cn):
        print(f"{name}\t{int(copy_n)}")
