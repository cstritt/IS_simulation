# Simulation of IS6110 insertion sequence dynamics along a phylogeny

This repository contains a Numba-accelerated Gillespie simulation model for IS6110 copy-number evolution under a genomic niche constraint, together with a parallel Approximate Bayesian Computation (ABC) workflow for parameter inference.

## Files

| File              | Purpose                                                  |
|-------------------|----------------------------------------------------------|
| `niche_model.py`  | Core simulator, summary statistics, and CLI              |
| `abc_sampler.py`  | Parallel ABC sampler using `multiprocessing`             |
| `abc_priors.yaml` | Prior ranges for the five fitted parameters              |

## Model overview

Each genomic site is assigned a fitness cost for IS6110 insertion, drawn from a distribution of fitness effects (DFE):

- a fraction `p_neutral` of sites are **neutral** — they have a small positive fitness floor (rather than exactly zero) so insertion there is tolerated and turnover remains reversible
- the remaining sites are **deleterious**, with fitness costs drawn from a gamma distribution parameterised by `gamma_shape` and `gamma_scale`

At every Gillespie step along a branch:

- a **birth event** transposes a copy to an empty site, weighted by site targetability
- a **loss event** removes a copy, with probability proportional to its site's fitness cost

The founder insertion at the root is always placed on a neutral site, so trajectories are not decided by the coin-flip of whether an unlucky deleterious placement goes extinct before the first transposition.

### Parameters

| Parameter     | Prior                            | Description                                              |
|---------------|----------------------------------|----------------------------------------------------------|
| `p_neutral`   | Uniform(0.05, 0.5)               | Fraction of neutral (tolerated) sites                    |
| `gamma_shape` | Uniform(0.5, 2.5)                | Shape of the deleterious-site fitness distribution       |
| `gamma_scale` | Uniform(0.1, 1.0)                | Scale of that distribution                               |
| `r_birth`     | Log-uniform(1.36e-05, 3.40e-04)  | Transposition rate per occupied site per branch unit     |
| `r_loss`      | Log-uniform(1.00e-04, 5.00e-03)  | Purging-rate coefficient (scaled by site fitness cost)   |

Fixed inputs read from `abc_priors.yaml`:

- `L`: number of genomic sites (default 7178, matching the MTBC0 reference)
- `initial_copies`: number of founder insertions placed at the root (default 1)

## Installation

```bash
pip install numpy pandas numba ete3 pyyaml
```

## Running a single simulation

### Command line

```bash
python niche_model.py \
  --tree      data/subsampled_tree.rooted.nex \
  --niches    7178 \
  --p_neutral 0.2 \
  --gamma_shape 1.0 \
  --gamma_scale 0.5 \
  --r_birth   1e-4 \
  --r_loss    1e-4 \
  --metadata  data/metadata_reduced.tsv \
  --seed      42
```

Prints a two-column table (`strain`, `copy_number`) to stdout.

### Python API

`run_simulation` returns a dict mapping tip names to occupancy arrays.
Summary statistics are computed separately with `get_summary_stats`.

```python
from ete3 import Tree
import pandas as pd
from niche_model import run_simulation, get_summary_stats

tree      = Tree("data/subsampled_tree.rooted.nex", format=1)
tip_names = sorted(tree.get_leaf_names())

metadata    = pd.read_csv("data/metadata_reduced.tsv", sep="\t")
lineage_map = dict(zip(metadata["GNUMBER"], metadata["LINEAGE_x"]))

tip_results = run_simulation(
    tree=tree,
    L=7178,
    p_neutral=0.2,
    gamma_shape=1.0,
    gamma_scale=0.5,
    r_birth=1e-4,
    r_loss=1e-4,
    initial_copies=1,
    seed=42,
)

# tip_results: {tip_name: occupancy_array, ...}
# copy number for one tip:
cn = tip_results["sample_A"].sum()

# summary statistics for ABC:
stats = get_summary_stats(tip_results, tip_names, lineage_map, mode="simulated")
```

## Running the ABC sampler

The sampler draws `nsim` particles i.i.d. from the priors defined in `abc_priors.yaml`, simulates each one in parallel, computes summary statistics, and writes two tab-separated output files for downstream inference with R's `abc` package.

### Command line

```bash
python abc_sampler.py \
  data/subsampled_tree.rooted.nex \
  abc_priors.yaml \
  data/metadata_reduced.tsv \
  output/abc_run \
  --nsim 500000 \
  --n_workers 32
```

`--n_workers` is optional; the script reads `SLURM_CPUS_PER_TASK` / `SLURM_NTASKS` automatically when running on a cluster.

### Outputs

| File                 | Contents                                               |
|----------------------|--------------------------------------------------------|
| `abc_params.tsv`     | One row per simulation, five parameter columns         |
| `abc_summaries.tsv`  | Matching summary statistics for each simulation        |

### Downstream inference in R

```r
library(abc)

obs   <- read.table("observed_stats.tsv", header=TRUE)
param <- read.table("abc_params.tsv",     header=TRUE, sep="\t")
sumst <- read.table("abc_summaries.tsv",  header=TRUE, sep="\t")

stat_cols <- intersect(names(sumst), names(obs))

res <- abc(
  target   = obs[, stat_cols],
  param    = param,
  sumstat  = sumst[, stat_cols],
  tol      = 0.01,
  method   = "loclinear"
)
summary(res)
```

## Summary statistics

`get_summary_stats` returns the following statistics, used as input to R's `abc()`:

**Copy-number distribution across tips**

- `mean_cn`, `std_cn`, `median_cn`, `max_cn`

**Among-lineage variation**

- `lineage_var`, `lineage_sd` — variance and SD of per-lineage mean copy number

**Occupancy across sites**

- `mean_occupancy`, `max_occupancy`, `gini_occupancy`

**Binned site-frequency spectrum** (proportion of occupied sites in each frequency class, summing to 1)

- `sfs_1` through `sfs_5`: singletons, doubletons, ..., 5-tons
- `sfs_6_10`: sites present in 6–10 strains
- `sfs_11_5pct`, `sfs_5_10pct`, `sfs_10_25pct`, `sfs_gt25pct`: adaptive bins relative to sample size

## Notes

- Metadata must contain at least `GNUMBER` and `LINEAGE_x` columns.
- Trees should be in Newick format with branch lengths in units of substitutions per site.
- A diagnostic line is printed to stderr after each run showing the mean copy-number distribution across all simulations — useful for checking that the priors produce realistic copy numbers before running a full inference.