#!/usr/bin/env python3

"""
Get observed summary statistics from detettore6110 output/ MTBC_IS6110 workflow files.

The script can aggregate a site-level presence/absence matrix into a region-level
presence/absence matrix using the grouping information in the metadata table.
"""

#%%
import sys
import yaml
import numpy as np
import pandas as pd

from ete3 import Tree

#%%

def aggregate_sites_to_tip_results(matrix, matrix_meta, tip_names=None, side=5):
    """Collapse site-level presence/absence data to region-level tip results.

    The input matrix is expected to have strains as rows and insertion sites as
    columns. The metadata table must contain the site identifiers in a
    ``site_id`` column and their grouping in a ``context`` column (region name).
    The output is a dict mapping each tip name to a binary region occupancy array
    that matches the style of ``tip_results`` used by ``niche_model``.
    """

    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)
    if not isinstance(matrix_meta, pd.DataFrame):
        matrix_meta = pd.DataFrame(matrix_meta)

    required_columns = {"site_id", "context", "side"}
    missing_columns = required_columns.difference(matrix_meta.columns)
    if missing_columns:
        raise ValueError(f"matrix_meta is missing required columns: {sorted(missing_columns)}")

    metadata = matrix_meta.loc[matrix_meta["side"] == side].copy()
    metadata = metadata.loc[metadata["site_id"].isin(matrix.columns)]

    if metadata.empty:
        return {name: np.array([], dtype=int) for name in (tip_names or matrix.index)}

    region_names = [region for region in metadata["context"].dropna().unique() if region is not None]
    tip_results = {}

    if tip_names is None:
        tip_names = list(matrix.index)

    for tip in tip_names:
        if tip not in matrix.index:
            continue
        tip_vector = []
        for region in region_names:
            region_sites = metadata.loc[metadata["context"] == region, "site_id"]
            region_sites = [site for site in region_sites if site in matrix.columns]
            if region_sites:
                present = int((matrix.loc[tip, region_sites] > 0).any())
            else:
                present = 0
            tip_vector.append(present)
        tip_results[tip] = np.array(tip_vector, dtype=int)

    return tip_results

def add_empty_sites(tip_results, n):
    """ Add n empty sites to each tip

    Args:
        tip_results (dict): Output of aggregate_sites_to_tip_results()
        n (int): Number of empty sites to add
    """

    for tip in tip_results:
        tip_results[tip] = np.append(tip_results[tip], np.zeros(n, dtype=int))
    
    return tip_results


def _gini(arr):
    """Gini coefficient of array (0=perfect equality, 1=perfect inequality)."""
    arr = np.sort(arr.astype(float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2.0 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1.0) / n


def get_site_frequency_spectrum_binned(mat):
    """
    Compute normalized binned site frequency spectrum statistics.

    The returned values are proportions of occupied insertion sites in each
    frequency class (they sum to 1).

    Bins:
        1
        2
        3
        4
        5
        6-10
        11-5% of samples
        5-10% of samples
        10-25% of samples
        >25% of samples

    Args:
        mat: array of shape (n_tips, n_sites) with occupancy {0,1}

    Returns:
        dict of normalized SFS summary statistics.
    """

    occupancy = (mat == 1).sum(axis=0)
    occupancy = occupancy[occupancy > 0]      # occupied sites only
    n_tips = mat.shape[0]

    # Adaptive bin boundaries
    b3 = max(20, int(np.ceil(0.05 * n_tips)))
    b4 = max(b3 + 1, int(np.ceil(0.10 * n_tips)))
    b5 = max(b4 + 1, int(np.ceil(0.25 * n_tips)))

    stats = {
        "sfs_1": 0.0,
        "sfs_2": 0.0,
        "sfs_3": 0.0,
        "sfs_4": 0.0,
        "sfs_5": 0.0,
        "sfs_6_10": 0.0,
        "sfs_11_5pct": 0.0,
        "sfs_5_10pct": 0.0,
        "sfs_10_25pct": 0.0,
        "sfs_gt25pct": 0.0,
    }

    if occupancy.size == 0:
        return stats

    n_sites = float(occupancy.size)

    stats["sfs_1"] = np.sum(occupancy == 1) / n_sites
    stats["sfs_2"] = np.sum(occupancy == 2) / n_sites
    stats["sfs_3"] = np.sum(occupancy == 3) / n_sites
    stats["sfs_4"] = np.sum(occupancy == 4) / n_sites
    stats["sfs_5"] = np.sum(occupancy == 5) / n_sites

    stats["sfs_6_10"] = np.sum((occupancy >= 6) & (occupancy <= 10)) / n_sites
    stats["sfs_11_5pct"] = np.sum((occupancy >= 11) & (occupancy <= b3)) / n_sites
    stats["sfs_5_10pct"] = np.sum((occupancy > b3) & (occupancy <= b4)) / n_sites
    stats["sfs_10_25pct"] = np.sum((occupancy > b4) & (occupancy <= b5)) / n_sites
    stats["sfs_gt25pct"] = np.sum(occupancy > b5) / n_sites

    return stats


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
        "total_cn": float(cn.sum()),
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
    
    
    L = occupancy.size
    n_occupied = np.sum(occupancy > 0)
    n_empty = L - n_occupied
    stats["n_occupied_sites"] = int(n_occupied)
    stats["occupied_fraction"] = n_occupied / L
    stats["n_empty_sites"] = int(np.sum(occupancy == 0))
    
    
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
    sfs = get_site_frequency_spectrum_binned(mat)
    stats.update(sfs)

    return stats


def main():
    """Aggregate a site-level matrix to region-level tip results and optionally write it to disk."""
    
    try:
        treepath = sys.argv[1]
        meta = sys.argv[2]
        matrix = sys.argv[3]
        matrix_meta = sys.argv[4]
    except IndexError:
        sys.exit('Usage: get_observed_stats.py <tree> <metadata> <matrix> <matrix_meta>')
        
    if len(sys.argv) > 5:
        output_path = sys.argv[5]
    else:
        output_path = None

    # Load tree
    tree = Tree(treepath, format=1)
    tip_names = tree.get_leaf_names()

    # Create lineage index
    metadata = pd.read_csv(meta, sep='\t')
    lineage_map = dict(zip(metadata['GNUMBER'], metadata['LINEAGE_x']))
    lineage_map = {k: v for k, v in lineage_map.items() if k in tip_names}
        
    # Load matrix
    matrix = pd.read_csv(matrix, sep='\t', index_col=0)
    matrix_meta = pd.read_csv(matrix_meta, sep='\t')

    # Remove strains that are not in the tree
    if tip_names is not None:
        matrix = matrix.loc[matrix.index.isin(tip_names)]
    # Remove columns that sum to zero
    matrix = matrix.loc[:, (matrix.sum(axis=0) > 0)]

    tip_results = aggregate_sites_to_tip_results(matrix, matrix_meta, tip_names=tip_names, side=5)
    
    # Add empty sites 
    k0 = list(tip_results.keys())[0]
    n_empty = 7360 - len(tip_results[k0])
    tip_results = add_empty_sites(tip_results, n_empty)
    
    stats = get_summary_stats(tip_results, tip_names, lineage_map)
    
    # Convert numpy objects to floats
    for k in stats:
        if not isinstance(stats[k], int):
            stats[k] = float(stats[k])
    
    # Write to yaml
    if output_path is not None:
        with open(output_path, "w") as f:
            yaml.dump(stats, f, sort_keys=False)
    else:
        print(yaml.dump(stats, sort_keys=False))
    

if __name__ == '__main__':
    main()
