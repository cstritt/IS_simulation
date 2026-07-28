#!/usr/bin/env python3
"""
Approximate Bayesian Computation for niche model with parallel execution.

Two functions:
  - run_abc_sampler:        single-stage, samples from priors, saves
                             params + summary stats. Feed output to R's
                             abc package.
  - run_abc_sampler_staged: two stages.
                             Stage 1: broad priors, filter on simulated
                             copy number range to drop clearly unrealistic
                             parameter draws. Use surviving particles to
                             narrow the prior box with narrow_bounds().
                             Stage 2: simulate from narrowed priors, save
                             all params + summary stats. Feed output to
                             R's abc package.

niche_model.run_simulation(...) -> tip_results (dict tip_name -> occ array).
niche_model.get_summary_stats(tip_results, tip_names, lineage_map,
    mode="simulated") -> stats dict, used by stage 2 and run_abc_sampler.
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


PARAM_COLS = ['p_neutral', 'gamma_shape', 'gamma_scale', 'r_birth', 'r_loss']
LOG_PARAMS  = {'r_birth', 'r_loss'}


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def load_priors(yaml_path):
    import yaml
    with open(yaml_path) as f:
        priors = yaml.safe_load(f)
    for key in ['r_birth', 'r_loss']:
        priors[key]['lower'] = float(priors[key]['lower'])
        priors[key]['upper'] = float(priors[key]['upper'])
    return priors


def load_observed_stats(yaml_path):
    import yaml
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def get_max_occ(priors):
    """Early-exit occupancy cap, read from priors['max_occ']['value'] if present."""
    return priors.get('max_occ', {}).get('value')


def sample_priors(priors):
    """Sample one particle i.i.d. from the prior box."""
    return {
        'p_neutral':   random.uniform(priors['p_neutral']['lower'],   priors['p_neutral']['upper']),
        'gamma_shape': random.uniform(priors['gamma_shape']['lower'], priors['gamma_shape']['upper']),
        'gamma_scale': random.uniform(priors['gamma_scale']['lower'], priors['gamma_scale']['upper']),
        'r_birth': np.exp(random.uniform(np.log(priors['r_birth']['lower']),
                                          np.log(priors['r_birth']['upper']))),
        'r_loss':  np.exp(random.uniform(np.log(priors['r_loss']['lower']),
                                          np.log(priors['r_loss']['upper']))),
    }


def narrow_bounds(kept_particles, orig_priors, pad_frac=0.3, percentile=95):
    """
    Build a new bounds dict (same shape as orig_priors) by taking the
    [100-percentile, percentile] range of kept_particles, padding by
    pad_frac on each side, and clipping to the original bounds.
    r_birth / r_loss are handled in log space.

    The returned dict can be passed directly to sample_priors() as
    `priors`, since all other keys (L, initial_copies, etc.) are
    inherited from orig_priors via {**orig_priors, **new_bounds}.
    """
    lo_q = (100 - percentile) / 2
    hi_q = 100 - lo_q
    new_bounds = {}

    for key in PARAM_COLS:
        x = np.array([p[key] for p in kept_particles], dtype=float)
        if key in LOG_PARAMS:
            x = np.log(x)

        lo, hi = np.percentile(x, [lo_q, hi_q])
        width = hi - lo
        lo -= pad_frac * width
        hi += pad_frac * width

        if key in LOG_PARAMS:
            lo, hi = np.exp(lo), np.exp(hi)

        lo = max(lo, orig_priors[key]['lower'])
        hi = min(hi, orig_priors[key]['upper'])
        new_bounds[key] = {'lower': lo, 'upper': hi}

    return new_bounds


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def _worker_seed(sim_id):
    return (os.getpid() * 1000003 + sim_id * 97) % (2**31)


def _run_stage1(args):
    """
    Stage 1 worker: simulate and return the mean tip copy number only.
    No summary stats needed -- we're just checking plausibility.
    """
    sim_id, particle, tree_str, L, initial_copies, max_occ = args

    seed = _worker_seed(sim_id)
    np.random.seed(seed)
    random.seed(seed)

    tree = Tree(tree_str, format=1)

    try:
        tip_results = niche_model.run_simulation(
            tree=tree,
            L=L,
            p_neutral=particle['p_neutral'],
            gamma_shape=particle['gamma_shape'],
            gamma_scale=particle['gamma_scale'],
            r_birth=particle['r_birth'],
            r_loss=particle['r_loss'],
            initial_copies=initial_copies,
            max_occ=max_occ,
            seed=seed,
        )
        cn = np.fromiter((occ.sum() for occ in tip_results.values()),dtype=float)

        stats = {
            "median_cn": float(np.median(cn)),
            "max_cn": float(cn.max())
        }
        
        return (sim_id, particle, stats)

    except Exception as e:
        print(f"[stage1] sim {sim_id} failed: {e}", file=sys.stderr)
        return (sim_id, particle, None)


def _run_stage2(args):
    """
    Stage 2 worker: simulate and return full summary stats for R's abc().
    """
    sim_id, particle, tree_str, L, tip_names, lineage_map, initial_copies, max_occ = args

    seed = _worker_seed(sim_id)
    np.random.seed(seed)
    random.seed(seed)

    tree = Tree(tree_str, format=1)

    try:
        tip_results = niche_model.run_simulation(
            tree=tree,
            L=L,
            p_neutral=particle['p_neutral'],
            gamma_shape=particle['gamma_shape'],
            gamma_scale=particle['gamma_scale'],
            r_birth=particle['r_birth'],
            r_loss=particle['r_loss'],
            initial_copies=initial_copies,
            max_occ=max_occ,
            seed=seed,
        )
        stats = niche_model.get_summary_stats(
            tip_results, tip_names, lineage_map, mode='simulated'
        )
        return (sim_id, particle, stats)

    except Exception as e:
        print(f"[stage2] sim {sim_id} failed: {e}", file=sys.stderr)
        return (sim_id, particle, None)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def run_abc_sampler(treepath, nsim, priors_path, metadata_path, outdir, n_workers=None):
    """
    Single-stage ABC: sample from priors, simulate, save params + summary
    stats. Feed abc_params.tsv / abc_summaries.tsv to R's abc package.
    """
    os.makedirs(outdir, exist_ok=True)

    priors  = load_priors(priors_path)
    tree    = Tree(treepath, format=1)
    tree_str = tree.write(format=1)
    tip_names = sorted(tree.get_leaf_names())

    metadata = pd.read_csv(metadata_path, sep='\t')
    lineage_map = {g: l for g, l in zip(metadata['GNUMBER'], metadata['LINEAGE_x'])
                   if g in tip_names}

    L             = int(priors['L']['value'])
    initial_copies = int(priors['initial_copies']['value'])
    max_occ       = get_max_occ(priors)
    n_workers     = _resolve_workers(n_workers)

    proposals = [sample_priors(priors) for _ in range(nsim)]
    tasks = [
        (i, proposals[i], tree_str, L, tip_names, lineage_map, initial_copies, max_occ)
        for i in range(nsim)
    ]

    params_list, stats_list = [], []
    with Pool(processes=n_workers) as pool:
        for _, particle, stats in pool.imap_unordered(_run_stage2, tasks):
            if stats is not None:
                params_list.append([particle[k] for k in PARAM_COLS])
                stats_list.append(stats)

    return _save(params_list, stats_list, outdir)


def run_abc_sampler_staged(treepath, priors_path, observed_stats_path,
                            metadata_path, outdir,
                            n_workers=None, pad_frac=0.3, box_percentile=95):
    """
    Two-stage ABC:

    Stage 1 -- broad priors, plausibility filter
        Draw nsim particles from the broad prior, simulate, keep those whose
        mean tip copy number falls in [cn_min, cn_max] (set in priors YAML).
        Use surviving particles to narrow the prior box with narrow_bounds().

    Stage 2 -- narrowed priors, feed to R's abc()
        Draw nsim particles from the narrowed prior, compute full summary
        stats, save everything. Pass abc_params.tsv / abc_summaries.tsv to
        R's abc package along with the observed summary stats.

    Expected priors YAML structure:
        stage1:
          nsim: 100000
          cn_min: 1
          cn_max: 25
        stage2:
          nsim: 500000

    pad_frac, box_percentile: forwarded to narrow_bounds(). Defaults (0.3,
    95) trim the 1st/99th percentile of the stage 1 survivors and pad 20%
    on each side before clipping to the original prior bounds.
    """
    os.makedirs(outdir, exist_ok=True)

    priors    = load_priors(priors_path)
    tree      = Tree(treepath, format=1)
    tree_str  = tree.write(format=1)
    tip_names = sorted(tree.get_leaf_names())

    metadata = pd.read_csv(metadata_path, sep='\t')
    lineage_map = {g: l for g, l in zip(metadata['GNUMBER'], metadata['LINEAGE_x'])
                   if g in tip_names}

    L              = int(priors['L']['value'])
    initial_copies = int(priors['initial_copies']['value'])
    max_occ        = get_max_occ(priors)
    n_workers      = _resolve_workers(n_workers)

    s1      = priors['stage1']
    nsim1   = int(s1['n_simulations'])
    bounds = s1['bounds']
    
    cn_median_min  = float(bounds['median_cn']['min'])
    cn_median_max = float(bounds['median_cn']['max'])
    cn_max_min  = float(bounds['max_cn']['min'])
    cn_max_max = float(bounds['max_cn']['max'])

    nsim2   = int(priors['stage2']['nsim'])

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    print(f"[stage 1] simulating {nsim1} particles from broad priors ...",
          file=sys.stderr)

    proposals1 = [sample_priors(priors) for _ in range(nsim1)]
    tasks1 = [
        (i, proposals1[i], tree_str, L, initial_copies, max_occ)
        for i in range(nsim1)
    ]

    survivors, stage1_stats = [], []
    with Pool(processes=n_workers) as pool:
        for _, particle, stats in pool.imap_unordered(_run_stage1, tasks1):
            
            if stats is None:
                continue
            
            median_cn = stats['median_cn']
            max_cn = stats['max_cn']
            
            accepted = (
                cn_median_min <= median_cn <= cn_median_max and
                cn_max_min <= max_cn <= cn_max_max
            )   

            stage1_stats.append({
                "median_cn": median_cn,
                "max_cn": max_cn,
                "accepted": accepted
            })
            
            if accepted:
                survivors.append(particle)
            

    stage1_df = pd.DataFrame(stage1_stats)
    acc_rate = 100 * len(survivors) / len(stage1_df)

    print(
        f"[stage 1] {len(survivors)}/{nsim1} survivors "
        f"({acc_rate:.1f}%)",
        file=sys.stderr
    )

    print(
        f"  Median CN: "
        f"simulated [{stage1_df['median_cn'].min():.1f}, "
        f"{stage1_df['median_cn'].max():.1f}] "
        f"kept [{cn_median_min}, {cn_median_max}]",
        file=sys.stderr
    )

    print(
        f"  Max CN: "
        f"simulated [{stage1_df['max_cn'].min():.1f}, "
        f"{stage1_df['max_cn'].max():.1f}] "
        f"kept [{cn_max_min}, {cn_max_max}]",
        file=sys.stderr
    )

    if len(survivors) < 10:
        sys.exit("[stage 1] fewer than 10 survivors -- widen cn_min/cn_max "
                 "or revisit priors")

    # Save stage 1 survivors so you can inspect the narrowed box
    s1_df = pd.DataFrame(survivors, columns=PARAM_COLS)
    s1_df.to_csv(os.path.join(outdir, 'abc_params.stage1.tsv'), sep='\t', index=False)
    
    # Save stage 1 stats
    stage1_df.to_csv(os.path.join(outdir, "abc_stage1_stats.tsv"), sep="\t", index=False)
    
    # Narrow the prior
    new_bounds    = narrow_bounds(survivors, priors, pad_frac, box_percentile)
    narrow_priors = {**priors, **new_bounds}

    for key in PARAM_COLS:
        print(f"  {key}: [{priors[key]['lower']:.4g}, {priors[key]['upper']:.4g}]"
              f" -> [{narrow_priors[key]['lower']:.4g}, {narrow_priors[key]['upper']:.4g}]",
              file=sys.stderr)

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    print(f"[stage 2] simulating {nsim2} particles from narrowed priors ...",
          file=sys.stderr)

    proposals2 = [sample_priors(narrow_priors) for _ in range(nsim2)]
    tasks2 = [
        (nsim1 + i, proposals2[i], tree_str, L, tip_names, lineage_map,
         initial_copies, max_occ)
        for i in range(nsim2)
    ]

    params_list, stats_list = [], []
    with Pool(processes=n_workers) as pool:
        for _, particle, stats in pool.imap_unordered(_run_stage2, tasks2):
            if stats is not None:
                params_list.append([particle[k] for k in PARAM_COLS])
                stats_list.append(stats)

    print(f"[stage 2] {len(params_list)}/{nsim2} simulations completed",
          file=sys.stderr)

    return _save(params_list, stats_list, outdir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_workers(n_workers):
    if n_workers is not None:
        return int(n_workers)
    for var in ('SLURM_CPUS_PER_TASK', 'SLURM_NTASKS'):
        val = os.environ.get(var)
        if val:
            return int(val)
    return cpu_count()


def _save(params_list, stats_list, outdir):
    params_df = pd.DataFrame(params_list, columns=PARAM_COLS)
    stats_df  = pd.DataFrame(stats_list)

    params_path = os.path.join(outdir, 'abc_params.tsv')
    stats_path  = os.path.join(outdir, 'abc_summaries.tsv')
    params_df.to_csv(params_path, sep='\t', index=False)
    stats_df.to_csv(stats_path,  sep='\t', index=False)

    return params_df, stats_df, params_path, stats_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run ABC (single-stage or staged) for the niche model"
    )
    parser.add_argument("treepath")
    parser.add_argument("priors",         help="YAML priors file")
    parser.add_argument("observed_stats", help="YAML observed summary statistics")
    parser.add_argument("metadata_path",  help="TSV with GNUMBER / LINEAGE_x columns")
    parser.add_argument("outdir")
    parser.add_argument("--method", choices=["rejection", "staged"], default="staged")
    parser.add_argument("--nsim",  type=int, default=None,
                        help="Simulations for rejection method only")
    parser.add_argument("--n_workers", type=int, default=None)
    args = parser.parse_args()

    if args.method == "rejection":
        if args.nsim is None:
            sys.exit("--nsim required for --method rejection")
        params_df, stats_df, params_path, stats_path = run_abc_sampler(
            args.treepath, args.nsim, args.priors, args.metadata_path,
            args.outdir, n_workers=args.n_workers,
        )
    else:
        params_df, stats_df, params_path, stats_path = run_abc_sampler_staged(
            args.treepath, args.priors, args.observed_stats, args.metadata_path,
            args.outdir, n_workers=args.n_workers,
        )

    print(f"Saved {params_path}")
    print(f"Saved {stats_path}")


if __name__ == "__main__":
    main()