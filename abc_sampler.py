#!/usr/bin/env python3
"""
Approximate Bayesian Computation for niche model with parallel execution.

Priors:
  - p_essential: Uniform[0.5, 0.95]  (most sites are essential)
  - p_tolerated: Uniform[0.05, 0.5]  (minority of non-essential are tolerated)
  - r_birth: Uniform[0.001, 2.0]     (per-copy transposition rate)
  - r_purge: Uniform[0.001, 10.0]    (purging rate on tolerated sites)
"""

import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from ete3 import Tree

import niche_model as niche_model


def load_priors(yaml_path):
    """Load ABC priors from a YAML file."""
    import yaml

    with open(yaml_path) as f:
        priors = yaml.safe_load(f)

    return priors


def run_one_simulation(args):
    """Worker function for parallel ABC."""
    sim_id, tree_str, L, tip_names, lineage_map, priors = args

    # Seed each worker independently based on PID and sim_id
    worker_seed = (os.getpid() * 1000003 + sim_id * 97) % (2**31)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # Sample from priors
    p_neutral = random.uniform(priors['p_neutral']['lower'], priors['p_neutral']['upper'])
    gamma_shape = random.uniform(priors['gamma_shape']['lower'], priors['gamma_shape']['upper'])
    gamma_scale = random.uniform(priors['gamma_scale']['lower'], priors['gamma_scale']['upper'])

    # Log-uniform for r_birth and r_loss for better prior coverage
    r_birth = np.exp(random.uniform(np.log(priors['r_birth']['lower']), np.log(priors['r_birth']['upper'])))
    r_loss = np.exp(random.uniform(np.log(priors['r_loss']['lower']), np.log(priors['r_loss']['upper'])))

    # Parse tree (reconstruct from Newick string)
    tree = Tree(tree_str, format=1)

    # Run simulation
    try:
        _, _, stats = niche_model.run_simulation(
            tree=tree,
            L=L,
            p_neutral=p_neutral,
            gamma_shape=gamma_shape,
            gamma_scale=gamma_scale,
            r_birth=r_birth,
            r_loss=r_loss,
            tip_names=tip_names,
            lineage_map=lineage_map,
            seed=worker_seed
        )

        params_list = [p_neutral, gamma_shape, gamma_scale, r_birth, r_loss]
        return (sim_id, params_list, stats)

    except Exception as e:
        print(f"Simulation {sim_id} failed: {e}", file=sys.stderr)
        return (sim_id, [np.nan] * 5, {})


def run_abc_sampler(treepath, nsim, priors_path, metadata_path, outdir, n_workers=None):
    """Run a short ABC sampler workflow and write tab-separated outputs."""
    os.makedirs(outdir, exist_ok=True)

    priors = load_priors(priors_path)

    tree = Tree(treepath, format=1)
    tree_str = tree.write(format=1)
    tip_names = sorted(tree.get_leaf_names())

    metadata = pd.read_csv(metadata_path, sep='\t')
    lineage_map = dict(zip(metadata['GNUMBER'], metadata['LINEAGE_x']))
    lineage_map = {gnumber: lineage for gnumber, lineage in lineage_map.items() if gnumber in tip_names}

    L = 7178

    if n_workers is None:
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        slurm_ntasks = os.environ.get('SLURM_NTASKS')
        if slurm_cpus:
            n_workers = int(slurm_cpus)
        elif slurm_ntasks:
            n_workers = int(slurm_ntasks)
        else:
            n_workers = cpu_count()
    else:
        n_workers = int(n_workers)

    tasks = [(i, tree_str, L, tip_names, lineage_map, priors) for i in range(nsim)]

    params_list = []
    stats_list = []

    with Pool(processes=n_workers) as pool:
        for _, params, stats in pool.imap_unordered(run_one_simulation, tasks):
            params_list.append(params)
            stats_list.append(stats)

    params_df = pd.DataFrame(
        params_list,
        columns=['p_neutral', 'gamma_shape', 'gamma_scale', 'r_birth', 'r_loss']
    )

    if stats_list and stats_list[0]:
        stats_df = pd.DataFrame(stats_list)
    else:
        stats_df = pd.DataFrame()

    params_path = os.path.join(outdir, 'abc_params.tsv')
    stats_path = os.path.join(outdir, 'abc_summaries.tsv')

    params_df.to_csv(params_path, sep='\t', index=False)
    stats_df.to_csv(stats_path, sep='\t', index=False)

    return params_df, stats_df, params_path, stats_path


def main():
    """Main ABC pipeline."""
    parser = argparse.ArgumentParser(
        description="Run parallel ABC simulations for the niche model"
    )
    parser.add_argument("treepath", help="Path to the input tree in Newick format")
    parser.add_argument("nsim", type=int, help="Number of simulations to run")
    parser.add_argument("priors", help="Path to the YAML priors file")
    parser.add_argument("metadata_path", help="Path to the metadata TSV file")
    parser.add_argument("outdir", help="Directory where outputs will be written")
    args = parser.parse_args()

    params_df, stats_df, params_path, stats_path = run_abc_sampler(
        args.treepath,
        args.nsim,
        args.priors,
        args.metadata_path,
        args.outdir,
    )

    print(f"Saved {params_path}")
    print(f"Saved {stats_path}")
    print("\nParameter summary:")
    print(params_df.describe())
    print("\nStatistics summary:")
    print(stats_df.describe())


if __name__ == "__main__":
    main()
