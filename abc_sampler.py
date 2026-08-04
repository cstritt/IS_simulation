#!/usr/bin/env python3
"""
ABC sampler for the niche model.

Simulates nsim particles from the prior, computes summary statistics,
and saves abc_params.tsv + abc_summaries.tsv for downstream inference
with R's abc package.
"""

import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from ete3 import Tree

import niche_model


# Model 'basic'            – niche constraints only, uniform targetability
# Model 'target_site_prefs' – niche constraints + inferred target site heterogeneity
PARAM_COLS = {
    'basic':             ['p_neutral', 'alpha_fitness', 'r_birth', 'r_loss'],
    'target_site_prefs': ['p_neutral', 'alpha_fitness', 'alpha_targetability', 'r_birth', 'r_loss'],
}
LOG_PARAMS = {'r_birth', 'r_loss', 'alpha_targetability'}


def load_priors(yaml_path):
    import yaml
    with open(yaml_path) as f:
        priors = yaml.safe_load(f)
    for key in ['r_birth', 'r_loss', 'alpha_targetability']:
        priors[key]['lower'] = float(priors[key]['lower'])
        priors[key]['upper'] = float(priors[key]['upper'])
    return priors


def sample_priors(priors, model):
    """
    Sample one particle i.i.d. from the prior box.

    For the 'basic' model, alpha_targetability is set to None so that
    create_targetability() returns uniform weights (no hotspot structure).
    For 'target_site_prefs', alpha_targetability is drawn from its prior
    and recorded as an inferred parameter.
    """
    particle = {
        'p_neutral':   random.uniform(
            priors['p_neutral']['lower'],   priors['p_neutral']['upper']
        ),
        'alpha_fitness': random.uniform(
            priors['alpha_fitness']['lower'], priors['alpha_fitness']['upper']
        ),
        'r_birth': np.exp(random.uniform(
            np.log(priors['r_birth']['lower']), np.log(priors['r_birth']['upper']))
        ),
        'r_loss':  np.exp(random.uniform(
            np.log(priors['r_loss']['lower']), np.log(priors['r_loss']['upper']))
        ),
    }
    if model == 'target_site_prefs':
        particle['alpha_targetability'] = np.exp(random.uniform(
            np.log(priors['alpha_targetability']['lower']), np.log(priors['alpha_targetability']['upper']))
        )
    else:
        particle['alpha_targetability'] = None   # → uniform targetability in niche_model
    return particle


def _worker_seed(sim_id):
    return (os.getpid() * 1000003 + sim_id * 97) % (2**31)


def _run_simulation(args):
    sim_id, particle, tree_str, L, tip_names, lineage_map, initial_copies = args

    seed = _worker_seed(sim_id)
    np.random.seed(seed)
    random.seed(seed)

    tree = Tree(tree_str, format=1)

    try:
        tip_results = niche_model.run_simulation(
            tree=tree,
            L=L,
            p_neutral=particle['p_neutral'],
            alpha_fitness=particle['alpha_fitness'],
            alpha_targetability=particle['alpha_targetability'],
            r_birth=particle['r_birth'],
            r_loss=particle['r_loss'],
            initial_copies=initial_copies,
            seed=seed,
        )
        stats = niche_model.get_summary_stats(
            tip_results, tip_names, lineage_map, mode='simulated'
        )
        mean_cn = np.mean([occ.sum() for occ in tip_results.values()])
        return (sim_id, particle, stats, float(mean_cn))

    except Exception as e:
        print(f"sim {sim_id} failed: {e}", file=sys.stderr)
        return (sim_id, particle, None, np.nan)


def _resolve_workers(n_workers):
    if n_workers is not None:
        return int(n_workers)
    for var in ('SLURM_CPUS_PER_TASK', 'SLURM_NTASKS'):
        val = os.environ.get(var)
        if val:
            return int(val)
    return cpu_count()


def run_abc_sampler(treepath, nsim, priors_path, model, metadata_path, outdir, n_workers=None):
    """
    Sample nsim particles from the prior, simulate, compute summary stats,
    write abc_params.tsv and abc_summaries.tsv for R's abc package.
    """
    os.makedirs(outdir, exist_ok=True)

    priors    = load_priors(priors_path)
    tree      = Tree(treepath, format=1)
    tree_str  = tree.write(format=1)
    tip_names = sorted(tree.get_leaf_names())

    metadata    = pd.read_csv(metadata_path, sep='\t')
    lineage_map = {g: l for g, l in zip(metadata['GNUMBER'], metadata['LINEAGE_x'])
                   if g in tip_names}

    L              = int(priors['L']['value'])
    initial_copies = int(priors['initial_copies']['value'])
    n_workers      = _resolve_workers(n_workers)

    param_cols = PARAM_COLS[model]
    proposals  = [sample_priors(priors, model) for _ in range(nsim)]
    tasks = [
        (i, proposals[i], tree_str, L, tip_names, lineage_map, initial_copies)
        for i in range(nsim)
    ]

    params_list, stats_list, mean_cns = [], [], []

    with Pool(processes=n_workers) as pool:
        for _, particle, stats, mean_cn in pool.imap_unordered(_run_simulation, tasks):
            mean_cns.append(mean_cn)
            if stats is not None:
                params_list.append([particle[k] for k in param_cols])
                stats_list.append(stats)

    # Diagnostic: CN distribution across all simulations
    mean_cns = np.array(mean_cns)
    valid    = mean_cns[~np.isnan(mean_cns)]
    print(f"[done] {len(params_list)}/{nsim} simulations completed", file=sys.stderr)
    print(f"  mean CN across tips: "
          f"median={np.median(valid):.1f}, "
          f"range=[{valid.min():.1f}, {valid.max():.1f}]", file=sys.stderr)

    params_df = pd.DataFrame(params_list, columns=param_cols)
    stats_df  = pd.DataFrame(stats_list)

    params_path = os.path.join(outdir, 'abc_params.tsv')
    stats_path  = os.path.join(outdir, 'abc_summaries.tsv')
    params_df.to_csv(params_path, sep='\t', index=False)
    stats_df.to_csv(stats_path,  sep='\t', index=False)

    return params_df, stats_df, params_path, stats_path


def main():
    parser = argparse.ArgumentParser(description="Run ABC simulations for the niche model")
    parser.add_argument("treepath")
    parser.add_argument("priors", help="YAML priors file")
    parser.add_argument("metadata_path", help="TSV with GNUMBER / LINEAGE_x columns")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("--nsim", type=int, required=True, help="Number of simulations")
    parser.add_argument("--model", choices=['basic', 'target_site_prefs'], default='basic')
    parser.add_argument("--n_workers", type=int, default=None)
    args = parser.parse_args()

    params_df, stats_df, params_path, stats_path = run_abc_sampler(
        args.treepath, args.nsim, args.priors, args.model, args.metadata_path, 
        args.outdir, n_workers=args.n_workers,
    )

    print(f"Saved {params_path}")
    print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()