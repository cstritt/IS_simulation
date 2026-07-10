# Simulation of insertion sequence dynamics along a phylogeny

This repository contains a Numba-accelerated simulation model for IS6110 copy-number evolution under a genomic niche constraint. The implementation is currently centered on two scripts:

- `niche_model.py`: a single simulation engine and CLI for running one realization of the model
- `abc_sampler.py`: a parallel Approximate Bayesian Computation workflow that samples from priors and writes summary outputs
- `abc_priors.yaml`: prior ranges for the fitted parameters

## Model overview

The model represents genomic sites as having different fitness effects:

- neutral sites (`fitness = 0`) are effectively tolerated
- deleterious sites have fitness costs drawn from a gamma distribution
- transposition events can create new copies at targetable sites
- loss events remove occupied sites with probability proportional to their fitness cost

The main parameters are:

- `p_neutral`: fraction of sites with zero fitness effect
- `gamma_shape`: shape of the gamma distribution for deleterious fitness effects
- `gamma_scale`: scale of that distribution
- `r_birth`: transposition rate per occupied site per branch length unit
- `r_loss`: purging rate coefficient scaled by fitness effect

## Repository files

| File | Purpose |
|------|---------|
| `niche_model.py` | Core simulator, summary statistics, and CLI |
| `abc_sampler.py` | Parallel ABC sampler using multiprocessing |
| `abc_priors.yaml` | Prior definitions for the ABC workflow |

## Installation

```bash
pip install numpy pandas numba ete3 pyyaml
```

## Running a single simulation

### From the command line

```bash
python niche_model.py \
  --tree data/subsampled_tree.rooted.nex \
  --niches 7178 \
  --p_neutral 0.2 \
  --gamma_shape 1.0 \
  --gamma_scale 0.5 \
  --r_birth 1e-4 \
  --r_loss 1e-4 \
  --metadata data/metadata_reduced.tsv \
  --seed 42
```

This prints a two-column table with tip names and simulated copy numbers.

### From Python

```python
from ete3 import Tree
from niche_model import run_simulation

tree = Tree("data/subsampled_tree.rooted.nex", format=1)
tip_names = sorted(tree.get_leaf_names())

params, cn, stats = run_simulation(
    tree=tree,
    L=7178,
    p_neutral=0.2,
    gamma_shape=1.0,
    gamma_scale=0.5,
    r_birth=1e-4,
    r_loss=1e-4,
    tip_names=tip_names,
    lineage_map={name: "all" for name in tip_names},
    seed=42,
)

print(params)
print(stats)
```

## Running the ABC sampler

The ABC workflow uses the priors in `abc_priors.yaml` and writes two tab-separated outputs.

```bash
python abc_sampler.py \
  data/subsampled_tree.rooted.nex \
  1000 \
  abc_priors.yaml \
  data/metadata_reduced.tsv \
  output/abc_run
```

The script writes:

- `abc_params.tsv`: sampled parameter values for each simulation
- `abc_summaries.tsv`: summary statistics for each simulation

## Priors

The current prior configuration is defined in `abc_priors.yaml`:

| Parameter | Prior | Notes |
|-----------|-------|-------|
| `p_neutral` | Uniform(0.05, 0.5) | Fraction of neutral sites |
| `gamma_shape` | Uniform(0.5, 2.5) | Shape of deleterious-site fitness distribution |
| `gamma_scale` | Uniform(0.1, 1.0) | Scale of that distribution |
| `r_birth` | Log-uniform(1.36e-05, 3.40e-04) | Transposition rate |
| `r_loss` | Log-uniform(1e-04, 5e-03) | Purging strength |

## Summary statistics

The simulator computes a set of summary statistics for ABC including:

- copy-number summaries: `mean_cn`, `std_cn`, `median_cn`, `max_cn`
- lineage variance: `lineage_var`, `lineage_sd`
- occupancy summaries: `gini_occupancy`, `max_occupancy`, `mean_occupancy`
- site-frequency-spectrum summaries: `n_singletons`, `n_doubletons`, `n_rare`, `n_common`, `singleton_prop`, `tajimas_d`

## Notes

- The ABC sampler expects the metadata file to contain at least `GNUMBER` and `LINEAGE_x` columns.
- The tree should be provided in Newick format.
- The default genome length in the sampler is set to 7178, matching the MTBC0 reference used in the repository examples.
