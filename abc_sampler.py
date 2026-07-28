#!/usr/bin/env python3
"""
Approximate Bayesian Computation for niche model with parallel execution.

Three samplers are provided:
  - run_abc_sampler:        plain rejection ABC, single generation.
  - run_abc_sampler_staged: a few generations of simulate-then-keep-the-best-K,
                             with the *prior itself* narrowed to a smaller box
                             each round (see narrow_bounds()) rather than
                             perturbing individual particles. NOT weighted
                             SMC-ABC -- no importance weighting, so treat the
                             narrowing as a computational pre-filter. Every
                             generation, including the last, is an honest
                             i.i.d. sample from a genuine (if progressively
                             narrower) prior -- which is what R's abc package
                             requires of its input. The final generation is
                             meant to keep everything unfiltered and be
                             handed to abc() for the actual inference.
  - run_abc_sampler_smc:    proper importance-weighted SMC-ABC (Toni et al.
                             2009 / Beaumont et al. 2009 "ABC-PMC" scheme).
                             Each generation's accepted particles carry an
                             importance weight that corrects for the fact
                             they weren't drawn straight from the prior (they
                             come from perturbing individual particles from
                             the previous generation, weighted by
                             param_covariance/perturb_particle), so the
                             weighted population at every stage remains a
                             valid (self-)weighted sample from the prior
                             restricted to the accepted region -- and, unlike
                             run_abc_sampler_staged's box-narrowing, this
                             correction is what makes it valid to compare
                             against the same observed data across multiple
                             stages without double-counting it.

niche_model.run_simulation(tree, L, p_neutral, gamma_shape, gamma_scale,
r_birth, r_loss, initial_copies=1, seed=None) -> tip_results
(dict of tip_name -> occupancy array) only; it does not compute summary
statistics. Workers below call niche_model.get_summary_stats(tip_results,
tip_names, lineage_map, mode="simulated") separately to get the stats dict
used for the Mahalanobis distance / ranking metric.
"""

import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from ete3 import Tree
from scipy.stats import norm

import niche_model as niche_model


PARAM_COLS = ['p_neutral', 'gamma_shape', 'gamma_scale', 'r_birth', 'r_loss']
LOG_PARAMS = ('r_birth', 'r_loss')


def load_priors(yaml_path):
    """Load ABC priors from a YAML file."""
    import yaml

    with open(yaml_path) as f:
        priors = yaml.safe_load(f)

    # Make sure that the r_birth and r_loss priors are floats
    for key in ['r_birth', 'r_loss']:
        priors[key]['lower'] = float(priors[key]['lower'])
        priors[key]['upper'] = float(priors[key]['upper'])

    return priors


def load_observed_stats(yaml_path):
    """Load observed summary statistics from a YAML file (flat name: value map)."""
    import yaml

    with open(yaml_path) as f:
        obs_stats = yaml.safe_load(f)

    return obs_stats


def sample_priors(priors):
    """Sample a single particle from the ABC priors."""
    return {
        'p_neutral': random.uniform(priors['p_neutral']['lower'], priors['p_neutral']['upper']),
        'gamma_shape': random.uniform(priors['gamma_shape']['lower'], priors['gamma_shape']['upper']),
        'gamma_scale': random.uniform(priors['gamma_scale']['lower'], priors['gamma_scale']['upper']),

        # Log-uniform for r_birth and r_loss for better prior coverage
        'r_birth': np.exp(random.uniform(np.log(priors['r_birth']['lower']), np.log(priors['r_birth']['upper']))),
        'r_loss': np.exp(random.uniform(np.log(priors['r_loss']['lower']), np.log(priors['r_loss']['upper'])))
    }


def reflect(x, lower, upper):

    while x < lower or x > upper:

        if x < lower:
            x = lower + (lower - x)

        if x > upper:
            x = upper - (x - upper)

    return x


def perturb_particle(parent, cov, priors):
    """
    Perturb a particle with an independent-per-dimension Gaussian random walk
    (variance = diagonal of `cov`), reflecting at the prior bounds. r_birth
    and r_loss are perturbed in log space to match their log-uniform prior.

    cov is a 5x5 matrix ordered like PARAM_COLS (see param_covariance()).
    Only the diagonal is used -- this is a diagonal random-walk kernel.
    """

    child = {}

    child["p_neutral"] = reflect(
        parent["p_neutral"] + np.random.normal(0, np.sqrt(cov[0, 0])),
        priors["p_neutral"]["lower"], priors["p_neutral"]["upper"]
    )

    child["gamma_shape"] = reflect(
        parent["gamma_shape"] + np.random.normal(0, np.sqrt(cov[1, 1])),
        priors["gamma_shape"]["lower"], priors["gamma_shape"]["upper"]
    )

    child["gamma_scale"] = reflect(
        parent["gamma_scale"] + np.random.normal(0, np.sqrt(cov[2, 2])),
        priors["gamma_scale"]["lower"], priors["gamma_scale"]["upper"]
    )

    log_r_birth = np.log(parent["r_birth"]) + np.random.normal(0, np.sqrt(cov[3, 3]))
    child["r_birth"] = np.exp(reflect(
        log_r_birth,
        np.log(priors["r_birth"]["lower"]), np.log(priors["r_birth"]["upper"])
    ))

    log_r_loss = np.log(parent["r_loss"]) + np.random.normal(0, np.sqrt(cov[4, 4]))
    child["r_loss"] = np.exp(reflect(
        log_r_loss,
        np.log(priors["r_loss"]["lower"]), np.log(priors["r_loss"]["upper"])
    ))

    return child


def param_covariance(params_df, weights=None, scale=2.0):
    """
    Covariance of a particle population, used as the perturbation-kernel
    bandwidth for the *next* generation (r_birth/r_loss log-transformed
    first, since they're perturbed in log space). `scale` widens the spread
    a bit past the raw sample covariance so the next generation doesn't
    collapse too tightly around the current population.

    weights=None: plain (unweighted) covariance -- used by
        run_abc_sampler_staged, which doesn't track importance weights.
    weights=<array>: importance-weighted covariance (Beaumont et al. 2009:
        twice the weighted empirical covariance) -- used by
        run_abc_sampler_smc.
    """
    X = params_df[PARAM_COLS].to_numpy(dtype=float).copy()
    for j, col in enumerate(PARAM_COLS):
        if col in LOG_PARAMS:
            X[:, j] = np.log(X[:, j])

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        mean = np.average(X, axis=0, weights=w)
        Xc = X - mean
        cov = (Xc.T * w) @ Xc
    else:
        cov = np.cov(X, rowvar=False)

    cov *= scale
    cov += np.eye(cov.shape[0]) * 1e-8  # regularization

    return cov


def prior_density(particle, priors):
    """Joint prior density of a particle (uniform box x3, log-uniform x2)."""
    d = 1.0
    for key in ('p_neutral', 'gamma_shape', 'gamma_scale'):
        lo, hi = priors[key]['lower'], priors[key]['upper']
        if not (lo <= particle[key] <= hi):
            return 0.0
        d *= 1.0 / (hi - lo)
    for key in LOG_PARAMS:
        lo, hi = priors[key]['lower'], priors[key]['upper']
        x = particle[key]
        if not (lo <= x <= hi):
            return 0.0
        d *= 1.0 / (x * np.log(hi / lo))
    return d


def kernel_density(child, parent, cov, priors):
    """
    Transition kernel density K(child | parent, cov): independent Gaussian
    per linear parameter, log-Gaussian per log parameter -- matching the
    diagonal random walk in perturb_particle(). Boundary reflection is
    ignored in the density (standard simplification, same as the rest of
    this literature).
    """
    dens = 1.0
    for i, key in enumerate(('p_neutral', 'gamma_shape', 'gamma_scale')):
        sigma = np.sqrt(cov[i, i])
        if sigma <= 0:
            continue
        dens *= norm.pdf(child[key], loc=parent[key], scale=sigma)

    for i, key in zip((3, 4), LOG_PARAMS):
        sigma = np.sqrt(cov[i, i])
        if sigma <= 0:
            continue
        # log-normal kernel since the perturbation happens in log space
        dens *= norm.pdf(np.log(child[key]), loc=np.log(parent[key]), scale=sigma) / child[key]

    return dens


def compute_weights(new_particles, prev_particles, prev_weights, cov, priors):
    """
    Importance weights for a new SMC generation (Toni et al. 2009, eq. 2 /
    Beaumont et al. 2009's ABC-PMC): w_i ~ prior(theta_i) / sum_j w_j *
    K(theta_i | theta_j, cov). This is what corrects an accepted particle's
    weight for how easy/hard the perturbation kernel made it to reach that
    point, so the weighted population stays a valid representation of the
    prior restricted to the accepted region rather than drifting toward
    wherever the previous generation happened to cluster.
    """
    weights = np.zeros(len(new_particles))

    for i, theta in enumerate(new_particles):
        numer = prior_density(theta, priors)
        denom = sum(
            w * kernel_density(theta, prev_theta, cov, priors)
            for w, prev_theta in zip(prev_weights, prev_particles)
        )
        weights[i] = numer / denom if denom > 0 else 0.0

    total = weights.sum()
    if total <= 0:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights = weights / total

    return weights


def propose_particle(priors, prev_pop=None, prev_weights=None, prev_cov=None):
    """
    Sample a particle from the prior (prev_pop is None), or resample a
    parent from prev_pop and perturb it.

    prev_weights=None: parent drawn uniformly (run_abc_sampler_staged, which
        doesn't track importance weights).
    prev_weights=<array>: parent drawn according to those weights
        (importance-weighted resampling, run_abc_sampler_smc).
    """
    if prev_pop is None:
        return sample_priors(priors)

    if prev_weights is not None:
        idx = np.random.choice(len(prev_pop), p=prev_weights)
    else:
        idx = np.random.randint(len(prev_pop))

    parent = prev_pop[idx]
    return perturb_particle(parent, prev_cov, priors)


def mahalanobis_distance(sim_stats,
                         obs_stats,
                         stat_names,
                         cov_inv):
    """
    Mahalanobis distance between simulated and observed
    summary statistics.
    """

    sim = np.array(
        [sim_stats[s] for s in stat_names],
        dtype=float
    )

    obs = np.array(
        [obs_stats[s] for s in stat_names],
        dtype=float
    )

    diff = sim - obs

    d2 = diff @ cov_inv @ diff

    return np.sqrt(d2)


def estimate_covariance(summary_df, obs_stats):
    """
    Estimate covariance matrix of summary statistics from
    prior simulations.

    Parameters
    ----------
    summary_df : DataFrame
        Simulated summary statistics
    obs_stats : dict
        Observed summary statistics

    Returns
    -------
    stat_names
    mean
    cov_inv
    """

    stat_names = list(obs_stats.keys())

    X = summary_df[stat_names].to_numpy(dtype=float)

    mean = X.mean(axis=0)

    cov = np.cov(X, rowvar=False)

    # Regularization to avoid singular matrices
    cov += np.eye(cov.shape[0]) * 1e-6 * np.trace(cov) / cov.shape[0]

    cov_inv = np.linalg.inv(cov)

    return stat_names, mean, cov_inv


def rank_and_keep(particles, stats_list, obs_stats, stat_names, cov_inv, n_keep):
    """
    Rank particles by Mahalanobis distance to the observed stats and return
    the n_keep closest (particles, stats, distances), sorted best-first.
    n_keep=None keeps everything (still sorted).
    """
    dists = np.array([
        mahalanobis_distance(s, obs_stats, stat_names, cov_inv) for s in stats_list
    ])
    order = np.argsort(dists)
    if n_keep is not None:
        order = order[:n_keep]

    kept_particles = [particles[i] for i in order]
    kept_stats = [stats_list[i] for i in order]
    kept_dists = dists[order]

    return kept_particles, kept_stats, kept_dists


def narrow_bounds(kept_particles, orig_priors, pad_frac=0.2, percentile=98):
    """
    Build a new bounds dict, same shape as orig_priors (i.e. per-key
    {'lower':, 'upper':}), for drawing the *next* generation's proposals
    i.i.d. from a genuinely narrower prior box -- as opposed to perturbing
    around individual kept particles (see run_abc_sampler_smc, which does
    that properly with an importance-weight correction instead).

    For each parameter: take the [percentile, 100-percentile] range of the
    kept particles (log-scale for r_birth/r_loss), pad it by pad_frac of its
    width on each side, then clip back to orig_priors' bounds so narrowing
    can never extrapolate past what was originally declared as the prior.

    percentile < 100 trims outliers before computing the range (default 98:
    keep the 1st-99th percentile range rather than the literal min/max, so a
    single stray accepted particle can't reopen the whole box). pad_frac
    controls how generously the box is widened past that trimmed range --
    err on the wide side, since this box is what stands in for "the prior"
    in every later round, including whatever gets handed to R's abc().
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

        # never extrapolate past the originally declared prior
        lo = max(lo, orig_priors[key]['lower'])
        hi = min(hi, orig_priors[key]['upper'])

        new_bounds[key] = {'lower': lo, 'upper': hi}

    return new_bounds


def run_one_simulation(args):
    """Worker function for the plain rejection sampler (run_abc_sampler).
    Samples its own particle from the prior -- kept for backward compatibility."""
    sim_id, tree_str, L, tip_names, lineage_map, priors, initial_copies = args

    # Seed each worker independently based on PID and sim_id
    worker_seed = (os.getpid() * 1000003 + sim_id * 97) % (2**31)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    tree = Tree(tree_str, format=1)

    # Sample from priors
    particle = sample_priors(priors)

    try:
        tip_results = niche_model.run_simulation(
            tree=tree,
            L=L,
            p_neutral=particle["p_neutral"],
            gamma_shape=particle["gamma_shape"],
            gamma_scale=particle["gamma_scale"],
            r_birth=particle["r_birth"],
            r_loss=particle["r_loss"],
            initial_copies=initial_copies,
            seed=worker_seed
        )
        stats = niche_model.get_summary_stats(tip_results, tip_names, lineage_map, mode="simulated")

        params_list = [particle[k] for k in PARAM_COLS]

        return (sim_id, params_list, stats)

    except Exception as e:
        print(f"Simulation {sim_id} failed: {e}", file=sys.stderr)
        return (sim_id, [np.nan] * 5, {})


def run_one_particle(args):
    """
    Worker function used by the staged sampler: simulate one already-sampled
    or already-perturbed particle (as opposed to run_one_simulation, which
    samples its own particle internally).
    """
    sim_id, particle, tree_str, L, tip_names, lineage_map, initial_copies = args

    worker_seed = (os.getpid() * 1000003 + sim_id * 97) % (2**31)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    tree = Tree(tree_str, format=1)

    try:
        tip_results = niche_model.run_simulation(
            tree=tree,
            L=L,
            p_neutral=particle["p_neutral"],
            gamma_shape=particle["gamma_shape"],
            gamma_scale=particle["gamma_scale"],
            r_birth=particle["r_birth"],
            r_loss=particle["r_loss"],
            initial_copies=initial_copies,
            seed=worker_seed,
        )
        stats = niche_model.get_summary_stats(tip_results, tip_names, lineage_map, mode="simulated")
        return (sim_id, particle, stats)
    except Exception as e:
        print(f"Simulation {sim_id} failed: {e}", file=sys.stderr)
        return (sim_id, particle, None)


def _simulate_batch(pool, proposals, tree_str, L, tip_names, lineage_map, initial_copies, sim_id_start):
    """Run one parallel batch of proposed particles, dropping failures."""
    tasks = [
        (sim_id_start + i, proposals[i], tree_str, L, tip_names, lineage_map, initial_copies)
        for i in range(len(proposals))
    ]
    particles, stats = [], []
    for _, particle, s in pool.imap_unordered(run_one_particle, tasks):
        if s is not None:
            particles.append(particle)
            stats.append(s)
    return particles, stats, sim_id_start + len(proposals)


def run_abc_sampler(treepath, nsim, priors_path, metadata_path, outdir, n_workers=None):
    """Run a single-generation rejection ABC workflow and write tab-separated outputs."""
    os.makedirs(outdir, exist_ok=True)

    priors = load_priors(priors_path)

    tree = Tree(treepath, format=1)
    tree_str = tree.write(format=1)
    tip_names = sorted(tree.get_leaf_names())

    metadata = pd.read_csv(metadata_path, sep='\t')
    lineage_map = dict(zip(metadata['GNUMBER'], metadata['LINEAGE_x']))
    lineage_map = {gnumber: lineage for gnumber, lineage in lineage_map.items() if gnumber in tip_names}

    L = priors['L']['value']
    initial_copies = int(priors['initial_copies']['value'])

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

    tasks = [(i, tree_str, L, tip_names, lineage_map, priors, initial_copies) for i in range(nsim)]

    params_list = []
    stats_list = []

    with Pool(processes=n_workers) as pool:
        for _, params, stats in pool.imap_unordered(run_one_simulation, tasks):
            params_list.append(params)
            stats_list.append(stats)

    params_df = pd.DataFrame(params_list, columns=PARAM_COLS)

    if stats_list and stats_list[0]:
        stats_df = pd.DataFrame(stats_list)
    else:
        stats_df = pd.DataFrame()

    params_path = os.path.join(outdir, 'abc_params.tsv')
    stats_path = os.path.join(outdir, 'abc_summaries.tsv')

    params_df.to_csv(params_path, sep='\t', index=False)
    stats_df.to_csv(stats_path, sep='\t', index=False)

    return params_df, stats_df, params_path, stats_path


def run_abc_sampler_staged(treepath, priors_path, observed_stats_path, metadata_path, outdir,
                            n_workers=None, pad_frac=0.2, box_percentile=98):
    """
    Staged ABC: a handful of generations of simulate-everything-then-keep-the-
    best-K, with the *prior itself* narrowed each round to a smaller box
    around the previous generation's kept particles (see narrow_bounds()).
    Every generation -- including the last -- is therefore an honest i.i.d.
    sample from a genuine (if progressively narrower) prior, which is what
    R's abc package requires of its input. This is NOT importance-weighted
    SMC-ABC; there's no weight correction, so treat the narrowing as a
    computational pre-filter, not as itself producing a posterior sample --
    see run_abc_sampler_smc for that. The final generation is meant to keep
    everything (n_keep: null) and be handed off to abc() for the actual
    inference.

    Because each round's box is chosen by comparing simulations to the same
    observed data the final abc() call will also condition on, this is a
    mild form of reusing the data twice -- generous pad_frac/box_percentile
    (the defaults trim outliers and pad the range by 20%) keeps this to
    "avoid wasting compute on clearly-implausible regions" rather than an
    actual shrinkage of belief, but it's worth a sentence in the methods
    either way. run_abc_sampler_smc's importance weights are the principled
    way to reuse the same data across rounds without this caveat.

    priors['generations'] is a list of {nsim: int, n_keep: int or null}, e.g.:
        generations:
          - {nsim: 100000, n_keep: 1000}
          - {nsim: 200000, n_keep: 1000}
          - {nsim: 500000, n_keep: null}

    Generation 0 samples from the originally declared prior. Each later
    generation samples i.i.d. from a box built from the previous generation's
    kept particles (narrow_bounds), clipped so it can never extrapolate past
    the original prior.

    Every generation's kept params/stats/distances are also written to
    abc_params.gen<i>.tsv / abc_summaries.gen<i>.tsv so intermediate rounds
    are inspectable.
    """
    os.makedirs(outdir, exist_ok=True)

    priors = load_priors(priors_path)
    obs_stats = load_observed_stats(observed_stats_path)

    tree = Tree(treepath, format=1)
    tree_str = tree.write(format=1)
    tip_names = sorted(tree.get_leaf_names())

    metadata = pd.read_csv(metadata_path, sep='\t')
    lineage_map = dict(zip(metadata['GNUMBER'], metadata['LINEAGE_x']))
    lineage_map = {g: l for g, l in lineage_map.items() if g in tip_names}

    L = int(priors['L']['value'])
    initial_copies = int(priors['initial_copies']['value'])

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

    generations = priors['generations']
    if not generations:
        sys.exit("priors['generations'] is empty")

    current_priors = priors
    stat_names, cov_inv = None, None
    sim_id = 0
    final_params_df = final_stats_df = None

    with Pool(processes=n_workers) as pool:

        for gen_idx, gen_cfg in enumerate(generations):
            nsim = int(gen_cfg['nsim'])
            n_keep = gen_cfg.get('n_keep')
            n_keep = int(n_keep) if n_keep else None
            is_last = (gen_idx == len(generations) - 1)

            bounds_desc = ", ".join(
                f"{k}=[{current_priors[k]['lower']:.3g}, {current_priors[k]['upper']:.3g}]"
                for k in PARAM_COLS
            )
            print(f"[gen {gen_idx}] simulating {nsim} particles "
                  f"(keep={'all' if n_keep is None else n_keep}); "
                  f"prior box: {bounds_desc}", file=sys.stderr)

            proposals = [sample_priors(current_priors) for _ in range(nsim)]

            particles, stats_list, sim_id = _simulate_batch(
                pool, proposals, tree_str, L, tip_names, lineage_map,
                initial_copies, sim_id
            )

            if stat_names is None:
                # Build the summary-stat weighting matrix once, from the
                # broadest (first) generation, and reuse it for every
                # later round's ranking.
                stat_names, _, cov_inv = estimate_covariance(pd.DataFrame(stats_list), obs_stats)

            kept_particles, kept_stats, kept_dists = rank_and_keep(
                particles, stats_list, obs_stats, stat_names, cov_inv, n_keep
            )

            gen_params_df = pd.DataFrame(kept_particles, columns=PARAM_COLS)
            gen_stats_df = pd.DataFrame(kept_stats)
            gen_params_df['distance'] = kept_dists

            gen_params_df.to_csv(os.path.join(outdir, f'abc_params.gen{gen_idx}.tsv'), sep='\t', index=False)
            gen_stats_df.to_csv(os.path.join(outdir, f'abc_summaries.gen{gen_idx}.tsv'), sep='\t', index=False)

            print(f"[gen {gen_idx}] kept {len(kept_particles)}/{nsim}, "
                  f"distance range [{kept_dists.min():.3f}, {kept_dists.max():.3f}]",
                  file=sys.stderr)

            final_params_df, final_stats_df = gen_params_df, gen_stats_df

            if not is_last:
                # Narrow the *prior* for the next generation -- an explicit,
                # i.i.d.-sampleable box, not a perturbation around specific
                # particles (see run_abc_sampler_smc for that, done properly
                # with a weight correction).
                new_bounds = narrow_bounds(kept_particles, priors, pad_frac, box_percentile)
                current_priors = {**priors, **new_bounds}

    final_params_path = os.path.join(outdir, 'abc_params.tsv')
    final_stats_path = os.path.join(outdir, 'abc_summaries.tsv')
    final_params_df.to_csv(final_params_path, sep='\t', index=False)
    final_stats_df.to_csv(final_stats_path, sep='\t', index=False)

    print(f"[done] final generation: {len(final_params_df)} simulations written "
          f"to {final_params_path} / {final_stats_path} for downstream ABC "
          f"(e.g. R's abc package)", file=sys.stderr)

    return final_params_df, final_stats_df, final_params_path, final_stats_path


def run_abc_sampler_smc(treepath, priors_path, observed_stats_path, metadata_path, outdir,
                         n_workers=None):
    """
    Sequential Monte Carlo ABC with the importance-weight + kernel-density
    correction (Toni et al. 2009 / Beaumont et al. 2009's ABC-PMC scheme).

    Structurally this is run_abc_sampler_staged (fixed simulation budget per
    stage, quantile-based keep threshold, narrowing proposal) plus the
    importance weight correction: each stage's kept particles carry a weight
    w_i ~ prior(theta_i) / sum_j w_j * K(theta_i | theta_j), which corrects
    for the fact they were proposed by perturbing the previous stage rather
    than drawn straight from the prior. This keeps every stage -- not just a
    designated final unfiltered one, unlike run_abc_sampler_staged -- a valid
    (self-)weighted sample from the prior restricted to the accepted region.
    See compute_weights()/kernel_density()/prior_density().

    Stage schedule: priors['smc_stages'], a mapping of
        stage_name -> {nsim: <fixed simulation budget>, keep_frac: <fraction to keep>}
    read in YAML/dict insertion order, e.g.:
        smc_stages:
          stage0: {nsim: 100000, keep_frac: 0.01}
          stage1: {nsim: 200000, keep_frac: 0.01}
          stage2: {nsim: 500000, keep_frac: 0.002}
    Each stage runs exactly nsim simulations (bounded, predictable compute --
    no acceptance loop, no unpredictable total simulation count). The
    tolerance isn't a number you pick: it's derived per stage as whatever
    distance the keep_frac*nsim closest simulations happen to fall under
    (see rank_and_keep()), so it naturally adapts as the population narrows.
    keep_frac: null keeps everything for that stage (rare for SMC, since the
    weighted particle set at every stage is the thing you're actually using,
    but supported for consistency with rank_and_keep()).

    Stage 0 samples from the prior; cov_inv (the Mahalanobis weighting
    matrix for summary stats) is estimated once from stage 0's full nsim
    pool and reused unchanged in every later stage. Each later stage samples
    from a weighted mixture of the previous stage's kept particles, perturbs
    with a Gaussian/log-Gaussian kernel (bandwidth = 2x the importance-
    weighted covariance of the previous stage), runs its full nsim budget,
    keeps the closest keep_frac fraction, then recomputes weights.

    Because the output is already a valid weighted posterior sample, this
    doesn't need a downstream tool like R's abc package the way
    run_abc_sampler_staged's final generation does -- though you can still
    feed the last stage's params/stats into abc() if you want the additional
    regression adjustment, just make sure to pass the 'weight' column
    through rather than treating the rows as equally-weighted prior draws.
    """
    os.makedirs(outdir, exist_ok=True)

    priors = load_priors(priors_path)
    obs_stats = load_observed_stats(observed_stats_path)

    tree = Tree(treepath, format=1)
    tree_str = tree.write(format=1)
    tip_names = sorted(tree.get_leaf_names())

    metadata = pd.read_csv(metadata_path, sep='\t')
    lineage_map = dict(zip(metadata['GNUMBER'], metadata['LINEAGE_x']))
    lineage_map = {g: l for g, l in lineage_map.items() if g in tip_names}

    L = int(priors['L']['value'])
    initial_copies = int(priors['initial_copies']['value'])

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

    stages = list(priors['smc_stages'].items())
    if not stages:
        sys.exit("priors['smc_stages'] is empty")

    prev_pop, prev_weights, prev_cov = None, None, None
    stat_names, cov_inv = None, None
    sim_id = 0
    final_params_df = final_stats_df = None

    with Pool(processes=n_workers) as pool:

        for stage_idx, (stage_name, stage_cfg) in enumerate(stages):
            nsim = int(stage_cfg['nsim'])
            keep_frac = stage_cfg.get('keep_frac')
            n_keep = int(np.ceil(keep_frac * nsim)) if keep_frac else None

            print(f"[SMC] stage {stage_idx + 1}/{len(stages)} '{stage_name}': "
                  f"simulating {nsim} particles "
                  f"(keep={'all' if n_keep is None else n_keep})", file=sys.stderr)

            if prev_pop is None:
                proposals = [sample_priors(priors) for _ in range(nsim)]
            else:
                proposals = [
                    propose_particle(priors, prev_pop, prev_weights, prev_cov)
                    for _ in range(nsim)
                ]

            particles, stats_list, sim_id = _simulate_batch(
                pool, proposals, tree_str, L, tip_names, lineage_map,
                initial_copies, sim_id
            )

            if stat_names is None:
                # Build the summary-stat weighting matrix once, from stage
                # 0's full simulation pool, and reuse it for every later
                # stage's ranking.
                stat_names, _, cov_inv = estimate_covariance(pd.DataFrame(stats_list), obs_stats)

            kept_particles, kept_stats, kept_dists = rank_and_keep(
                particles, stats_list, obs_stats, stat_names, cov_inv, n_keep
            )

            params_df = pd.DataFrame(kept_particles, columns=PARAM_COLS)
            stats_df = pd.DataFrame(kept_stats)

            if prev_pop is None:
                weights = np.ones(len(kept_particles)) / len(kept_particles)
            else:
                weights = compute_weights(kept_particles, prev_pop, prev_weights, prev_cov, priors)

            ess = 1.0 / np.sum(weights ** 2)
            print(f"[SMC] stage '{stage_name}': kept {len(kept_particles)}/{nsim}, "
                  f"distance range [{kept_dists.min():.3f}, {kept_dists.max():.3f}], "
                  f"ESS={ess:.1f}", file=sys.stderr)

            params_df_out = params_df.copy()
            params_df_out['weight'] = weights
            params_df_out['distance'] = kept_dists
            params_df_out.to_csv(os.path.join(outdir, f'abc_params.{stage_name}.tsv'), sep='\t', index=False)
            stats_df.to_csv(os.path.join(outdir, f'abc_summaries.{stage_name}.tsv'), sep='\t', index=False)

            prev_pop = kept_particles
            prev_weights = weights
            prev_cov = param_covariance(params_df, weights=weights)

            final_params_df, final_stats_df = params_df_out, stats_df

    final_params_path = os.path.join(outdir, 'abc_params.tsv')
    final_stats_path = os.path.join(outdir, 'abc_summaries.tsv')
    final_params_df.to_csv(final_params_path, sep='\t', index=False)
    final_stats_df.to_csv(final_stats_path, sep='\t', index=False)

    print(f"[done] final stage: {len(final_params_df)} weighted particles "
          f"written to {final_params_path} (see 'weight' column) / "
          f"{final_stats_path}", file=sys.stderr)

    return final_params_df, final_stats_df, final_params_path, final_stats_path


def main():
    """Main ABC pipeline."""
    parser = argparse.ArgumentParser(
        description="Run ABC (single-generation rejection, staged, or SMC) for the niche model"
    )
    parser.add_argument("treepath", help="Path to the input tree in Newick format")
    parser.add_argument("priors", help="Path to the YAML priors file")
    parser.add_argument("observed_stats", help="Observed summary statistics in YAML format")
    parser.add_argument("metadata_path", help="Path to the metadata TSV file")
    parser.add_argument("outdir", help="Directory where outputs will be written")
    parser.add_argument("--method", choices=["rejection", "staged", "smc"], default="staged",
                         help="ABC method to use (default: staged)")
    parser.add_argument("--nsim", type=int, default=None,
                         help="Number of simulations (rejection method only; "
                              "staged/smc generation sizes come from priors['generations'] "
                              "or priors['smc_stages'] respectively)")
    parser.add_argument("--n_workers", type=int, default=None)
    args = parser.parse_args()

    if args.method == "rejection":
        if args.nsim is None:
            sys.exit("--nsim is required for --method rejection")
        params_df, stats_df, params_path, stats_path = run_abc_sampler(
            args.treepath, args.nsim, args.priors, args.metadata_path, args.outdir,
            n_workers=args.n_workers,
        )
    elif args.method == "smc":
        params_df, stats_df, params_path, stats_path = run_abc_sampler_smc(
            args.treepath, args.priors, args.observed_stats, args.metadata_path, args.outdir,
            n_workers=args.n_workers,
        )
    else:
        params_df, stats_df, params_path, stats_path = run_abc_sampler_staged(
            args.treepath, args.priors, args.observed_stats, args.metadata_path, args.outdir,
            n_workers=args.n_workers,
        )

    print(f"Saved {params_path}")
    print(f"Saved {stats_path}")
    print("\nParameter summary:")
    print(params_df.describe())
    print("\nStatistics summary:")
    print(stats_df.describe())


if __name__ == "__main__":
    main()