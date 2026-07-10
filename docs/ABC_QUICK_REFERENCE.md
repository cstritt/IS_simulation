# ABC Prior Calibration: Quick Reference

## Summary Table

| Parameter | Lower | Upper | Notes |
|-----------|-------|-------|-------|
| **p_neutral** | 0.30 | 0.95 | Fraction of insertion-compatible sites |
| **gamma_shape** | 0.50 | 2.50 | DFE concentration; moderate range |
| **gamma_scale** | 0.10 | 1.00 | DFE severity; ~1–5% cost/insertion |
| **r_birth** | 1.36e-05 | 3.40e-04 | Transposition rate; log-uniform ±5× |
| **r_loss** | 1.00e-07 | 3.40e-04 | Purging strength; log-uniform wide |

## Key Numbers

**From your empirical data:**
- Observed births (ASR): ~2,800
- Total tree length: 2,748,125 (per-genome substitutions)
- Median copy number: 15
- Tree span: median terminal 3,390, median internal 657

**Empirical birth rate:** 6.79e-05 per substitution per copy
- This is the **central estimate for r_birth**
- Prior spans ±5-fold: [1.36e-05, 3.40e-04]

**Expected births per median terminal branch:** ~3.5
- This is what ASR actually inferred
- Prior range gives 0.7–17 births/branch (covers uncertainty)

## Interpretation

### r_birth
Transposition activates at rate r_birth per occupied copy per per-genome-wide substitution.

- Lower priors (< 1e-05): Copy numbers won't build up; need stronger p_neutral
- Central (6.79e-05): Matches empirical observation
- Higher priors (> 3e-04): Copy numbers drift high even with purging

### r_loss  
Deletion rate = r_loss × fitness[i]. Stronger selection at deleterious sites.

- r_loss → 0: Nearly neutral; no constraint on CN
- r_loss ≈ r_birth × <fitness>: Rough equilibrium at observed CN
- r_loss > r_birth: Strong constraint; CN stays low

For equilibrium around CN=15 with moderate DFE:
```
Expected r_loss ≈ 0.1–1.0 × r_birth × <fitness>
                ≈ 0.1–1.0 × 6.79e-05 × 0.5  [if gamma(1, 0.5)]
                ≈ 3.4e-6 to 6.79e-5
```

But prior is **wide** [1e-7, 3.4e-4] to let data inform strength.

### p_neutral
High p_neutral means insertion-compatible sites are abundant; purifying selection gates the dynamics.

- p_neutral = 0.3: Only 30% of genome can host insertions → more constrained
- p_neutral = 0.95: Nearly all sites compatible → selection is main constraint
- Motif data suggest p_neutral should be **high** (0.7–0.95)

### gamma_shape & gamma_scale
Define distribution of fitness costs among deleterious sites.

Mean fitness effect = gamma_shape × gamma_scale.

Examples:
- gamma(0.5, 0.5) → mean ≈ 0.25 (mild costs, high variance)
- gamma(1.0, 0.5) → mean ≈ 0.5 (moderate costs)
- gamma(2.0, 0.5) → mean ≈ 1.0 (severe; concentrated burden)

Empirically, expect **moderate DFE** (0.5–1.0), so gamma_shape ∈ [0.5, 2.5] and gamma_scale ∈ [0.1, 1.0] is reasonable.

## ABC Workflow

1. **Compute observed summary stats** from your empirical copy number distribution
   - mean_cn, std_cn, median_cn, max_cn
   - gini_occupancy (hotspot concentration)
   - lineage_var (between-lineage variance)
   - SFS metrics (n_singletons, etc.)

2. **Run ABC rejection sampling** with priors above
   - Start with n_sims = 100,000
   - Accept top 1% (tolerance = 0.01)
   - Check acceptance rate; if <0.01%, widen priors 2–5×

3. **Posterior predictive check**
   - Simulate from posterior
   - Verify copy number distribution matches observations
   - Check that lineage patterns are captured

4. **Refine (optional)**
   - Tolerance sequence ABC for higher precision
   - Stage 1: accept top 1% (n_sims=100k)
   - Stage 2: accept top 0.5% (n_sims=200k, seeded from stage 1)
   - Stage 3: accept top 0.1% (n_sims=500k, seeded from stage 2)

## Diagnostics

**If ABC acceptance is very low (<0.001%):**
- Priors too tight
- Model may not explain data
- Check: are observed stats computed correctly?
- Action: widen p_neutral, gamma_shape, or r_birth by 2×

**If ABC acceptance is high (>1%):**
- Priors are conservative (good for initial run)
- Can tighten around posterior for next stage
- Expected: acceptance ≈ 0.1–1%

**If posterior doesn't reproduce copy number distribution:**
- DFE parameters (gamma_shape, gamma_scale, p_neutral) may not be identifiable
- Consider: add more summary stats (e.g., SFS)
- Or: use lineage-specific rates instead of global

## Files Provided

1. **niche_model_DFE_polished.py** — Main simulation code (ready to use)
2. **abc_priors.yaml** — Detailed prior specification and rationale
3. **abc_sampler.py** — ABC rejection sampler wrapper (integrate with pipeline)
4. This file — Quick reference

## Integration with Snakemake

Suggested workflow rule:

```python
rule abc_niche_model:
    input:
        tree = "results/phylogeny/tree.nwk",
        metadata = "data/metadata.tsv",
        observed_stats = "results/copy_number_stats.tsv"
    params:
        n_sims = 100000,
        tolerance = 0.01,
        seed = 42
    output:
        posterior = "results/niche_model/posterior.tsv"
    shell:
        """
        python abc_sampler.py \
          --tree {input.tree} \
          --metadata {input.metadata} \
          --observed {input.observed_stats} \
          --n_sims {params.n_sims} \
          --tolerance {params.tolerance} \
          --output {output.posterior} \
          --seed {params.seed}
        """
```

## References

- **DFE model**: Gamma distribution for fitness effects (Eyre-Walker & Keightley, 2007)
- **ABC rejection**: Standard tolerance-based particle filter
- **Gillespie simulation**: Exact stochastic algorithm for birth-death processes
- **Summary statistics**: Gini coefficient (inequality), site frequency spectrum, lineage variance

---

**Contact**: For questions on calibration or parameter interpretation, refer to abc_priors.yaml or the docstrings in niche_model_DFE_polished.py.
