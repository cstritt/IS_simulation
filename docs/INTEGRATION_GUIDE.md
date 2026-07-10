#!/usr/bin/env python3
"""
Integration guide: Adding niche_model_numba to the MTBC IS6110 workflow

This document walks through the steps to:
  1. Copy optimized niche model into your Snakemake workflow
  2. Add ABC rule to Snakefile
  3. Compare ABC posteriors to empirical hotspot statistics
  4. Validate model fit via posterior predictive checks
"""

# ============================================================================
# STEP 1: Install & Copy Files
# ============================================================================

"""
1.1. Install dependencies:

  pip install numba ete3 numpy pandas scikit-learn

1.2. Copy files to your project:

  cp niche_model_numba.py /path/to/MTBC_IS6110-poly/scripts/
  cp niche_model_abc_parallel_numba.py /path/to/MTBC_IS6110-poly/scripts/

1.3. Add to Snakemake rules/niche_model.smk:

  rule abc_niche_model:
      input:
          tree = "workflow/results/phylogeny/treedata.tree",
          metadata = "workflow/results/metadata.tsv"
      output:
          params = "workflow/results/niche_model/abc_params.tsv",
          summaries = "workflow/results/niche_model/abc_summaries.tsv"
      params:
          nsim = 10000,
          outdir = "workflow/results/niche_model"
      shell:
          \"\"\"
          python scripts/niche_model_abc_parallel_numba.py \\
            {input.tree} {params.nsim} {input.metadata} {params.outdir}
          \"\"\"

1.4. Add to Snakefile main rule:

  rule main:
      input:
          ...
          "workflow/results/niche_model/abc_params.tsv",
          "workflow/results/niche_model/abc_summaries.tsv"
"""

# ============================================================================
# STEP 2: Post-ABC Analysis in R
# ============================================================================

"""
2.1. Load ABC results and perform acceptance in R:

  library(tidyverse)
  library(abc)
  
  # Load observed summary statistics
  obs_stats <- read.csv('workflow/results/hotspot_summary.tsv', sep='\\t')
  
  # For genic insertions, compute observed statistics
  obs <- c(
    mean_cn = mean(obs_stats$nbirths),
    gini_occupancy = ineq::Gini(obs_stats$nbirths),
    n_rare = sum(obs_stats$nbirths <= 5)
  )
  
  # Load ABC results
  abc_params <- read.csv('workflow/results/niche_model/abc_params.tsv', sep='\\t')
  abc_summaries <- read.csv('workflow/results/niche_model/abc_summaries.tsv', sep='\\t')
  
  # Extract relevant statistics columns
  abc_stats <- abc_summaries %>%
    select(mean_cn, gini_occupancy, n_rare, n_singletons, singleton_prop)
  
  # Run ABC rejection with tolerance = 1% quantile
  tolerance <- 0.01
  distances <- mahalanobis(abc_stats, center=obs, cov=cov(abc_stats))
  accepted <- distances <= quantile(distances, tolerance)
  
  # Posterior
  posterior <- abc_params[accepted, ]
  
  print(paste("Accepted:", sum(accepted), "/", nrow(abc_params)))
  print("Posterior summary:")
  print(summary(posterior))
  
  # Visualize posteriors
  pairs(posterior)
  
  # Compare posterior predictive to observed
  pred_summaries <- abc_summaries[accepted, ]
  
  ggplot(pred_summaries, aes(mean_cn)) +
    geom_density(fill='blue', alpha=0.3) +
    geom_vline(xintercept = obs['mean_cn'], col='red', linetype='dashed') +
    labs(x='Mean births per hotspot', y='Density') +
    theme_bw()

2.2. Full Bayesian workflow (R):

  # Load and process
  source('scripts/abc_analysis.R')  # see template below
  
  # Outputs:
  #   posterior.csv - accepted parameter samples
  #   posterior_predictive.csv - posterior predictive distributions
  #   fig_posterior.pdf - posterior plots
  #   fig_ppc.pdf - posterior predictive check plots
"""

# ============================================================================
# STEP 3: R Template for ABC Analysis
# ============================================================================

r_template = """
#!/usr/bin/env Rscript
# scripts/abc_analysis.R

library(tidyverse)
library(abc)
library(cowplot)
library(ineq)

# Configuration
TOLERANCE <- 0.01  # 1% acceptance rate
OUTDIR <- 'workflow/results/niche_model'

# ==================== OBSERVED SUMMARY STATISTICS ====================

# Load hotspot summary (empirical data)
hotspots <- read.csv('workflow/results/hotspot_summary.tsv', sep='\\t')

# Focus on genes with births
hotspots_genic <- subset(hotspots, !grepl(';', region) & nbirths > 0)

# Compute observed statistics
obs <- c(
  mean_births = mean(hotspots_genic$nbirths),
  sd_births = sd(hotspots_genic$nbirths),
  n_hotspots = nrow(hotspots_genic),
  gini_births = Gini(hotspots_genic$nbirths),
  n_rare = sum(hotspots_genic$nbirths <= 5),
  max_births = max(hotspots_genic$nbirths)
)

cat("\\nObserved summary statistics:\\n")
print(obs)

# ==================== ABC RESULTS ====================

# Load ABC outputs
abc_params <- read.csv(file.path(OUTDIR, 'abc_params.tsv'), sep='\\t')
abc_summaries <- read.csv(file.path(OUTDIR, 'abc_summaries.tsv'), sep='\\t')

cat("\\nABC simulations loaded: ", nrow(abc_params), " × ", ncol(abc_params), "\\n")

# ==================== ABC ACCEPTANCE ====================

# Select informative statistics
abc_stats <- abc_summaries %>%
  select(
    mean_cn, std_cn,
    gini_occupancy,
    n_rare, max_occupancy,
    singleton_prop
  ) %>%
  as.matrix()

# Standardize
abc_stats_std <- scale(abc_stats)
obs_std <- scale(rbind(c(obs['mean_births'], obs['sd_births'],
                         obs['gini_births'], obs['n_rare'],
                         obs['max_births'], obs['mean_births']/2)))[1,]

# Euclidean distance in standardized space
distances <- sqrt(rowSums((abc_stats_std - rep(obs_std, each=nrow(abc_stats_std)))^2))

# Acceptance threshold
threshold <- quantile(distances, TOLERANCE)
accepted <- distances <= threshold

cat("\\nABC Rejection:\\n")
cat("  Threshold: ", round(threshold, 3), "\\n")
cat("  Accepted: ", sum(accepted), " / ", nrow(abc_params), 
    " (", round(100*mean(accepted), 2), "%)\\n\\n")

# ==================== POSTERIOR ====================

posterior <- abc_params[accepted, ]
posterior_stats <- abc_summaries[accepted, ]

cat("Posterior summary:\\n")
print(summary(posterior))

# Save posterior
write.csv(posterior, file.path(OUTDIR, 'posterior.csv'), row.names=FALSE)
write.csv(posterior_stats, file.path(OUTDIR, 'posterior_predictive.csv'), row.names=FALSE)

# ==================== POSTERIOR PREDICTIVE CHECKS ====================

# Compare posterior predictive to observed
cat("\\nPosterior predictive checks:\\n")

fig_list <- list()

# 1. Mean copy number
fig_list[[1]] <- ggplot(posterior_stats, aes(mean_cn)) +
  geom_density(fill='skyblue', alpha=0.5) +
  geom_vline(xintercept = obs['mean_births'], col='red', linetype='dashed', size=1) +
  labs(title='Mean hotspot births', x='Statistic', y='Density') +
  theme_bw()

# 2. Gini coefficient
fig_list[[2]] <- ggplot(posterior_stats, aes(gini_occupancy)) +
  geom_density(fill='lightgreen', alpha=0.5) +
  geom_vline(xintercept = obs['gini_births'], col='red', linetype='dashed', size=1) +
  labs(title='Gini (occupancy inequality)', x='Gini', y='Density') +
  theme_bw()

# 3. Max occupancy
fig_list[[3]] <- ggplot(posterior_stats, aes(max_occupancy)) +
  geom_density(fill='lightcoral', alpha=0.5) +
  geom_vline(xintercept = obs['max_births'], col='red', linetype='dashed', size=1) +
  labs(title='Max per-site occupancy', x='Count', y='Density') +
  theme_bw()

# 4. Singleton proportion
fig_list[[4]] <- ggplot(posterior_stats, aes(singleton_prop)) +
  geom_density(fill='gold', alpha=0.5) +
  labs(title='Singleton enrichment', x='Proportion', y='Density') +
  theme_bw()

ppc_fig <- plot_grid(plotlist = fig_list, ncol = 2, labels = c('A', 'B', 'C', 'D'))
ggsave(file.path(OUTDIR, 'fig_ppc.pdf'), ppc_fig, width = 12, height = 8)
cat("  Saved: fig_ppc.pdf\\n")

# ==================== POSTERIOR MARGINALS ====================

# Pairwise scatter plots with posterior density contours
posterior_long <- posterior %>%
  pivot_longer(everything(), names_to = 'param', values_to = 'value')

fig_posterior <- ggplot(posterior, aes(p_essential, r_birth)) +
  geom_point(alpha=0.3, size=0.5) +
  geom_density_2d(col='red', alpha=0.5) +
  labs(x='p_essential', y='r_birth') +
  theme_bw()

ggsave(file.path(OUTDIR, 'fig_posterior.pdf'), fig_posterior, width = 8, height = 6)
cat("  Saved: fig_posterior.pdf\\n")

# ==================== MODEL ASSESSMENT ====================

cat("\\n=== Model Fit Assessment ===\\n")

# Compute coverage: proportion of ABC sims with stat closer to obs than observed is
coverage <- colMeans(
  abs(abc_summaries - rep(as.numeric(obs[1:6]), each=nrow(abc_summaries))) <
  abs(rep(as.numeric(obs[1:6]), nrow(abc_summaries)) - rep(as.numeric(obs[1:6]), each=nrow(abc_summaries)))
)

cat("\\nCoverage (prop ABC < obs distance from prior mean):\\n")
print(coverage)

cat("\\nInterpretation:\\n")
cat("  High coverage (>0.5): summary stat is informative\\n")
cat("  Low coverage (<0.3): model may be misspecified\\n")

# Model rejection criterion
if (mean(coverage) < 0.3) {
  cat("\\n⚠ WARNING: Model coverage is low. Consider:\\n")
  cat("   - Revising prior distributions\\n")
  cat("   - Changing tolerance level\\n")
  cat("   - Adding more informative statistics\\n")
}

cat("\\n=== Complete! ===\\n")
cat("\\nOutput files:\\n")
cat("  - posterior.csv: accepted parameters\\n")
cat("  - posterior_predictive.csv: posterior predictive summaries\\n")
cat("  - fig_ppc.pdf: posterior predictive checks\\n")
cat("  - fig_posterior.pdf: posterior marginals\\n")
"""

# ============================================================================
# STEP 4: Diagnostic Plots
# ============================================================================

"""
4.1. Diagnose poor ABC acceptance:

  - Acceptance rate <<1%: parameters are far from observed
    → Try:
      • Broader priors
      • Tighter tolerance
      • More informative statistics
  
  - All statistics close to prior means: model not learning
    → Try:
      • Different summary statistics
      • Check tree & metadata alignment
      • Verify genome length (L)
  
  - Posterior concentrated but far from observed
    → Model captures wrong signal
      • Check prior ranges
      • Visualize observed vs posterior predictive

4.2. Hotspot-specific posterior checks:

  Posterior_predictive_hotspots <- posterior_stats %>%
    mutate(
      n_hotspots_pred = max_occupancy / mean_occupancy,
      distribution = "posterior"
    ) %>%
    select(n_hotspots_pred, distribution) %>%
    bind_rows(
      data.frame(
        n_hotspots_pred = nrow(hotspots_genic),
        distribution = "observed"
      )
    )
  
  ggplot(posterior_predictive_hotspots, aes(n_hotspots_pred, fill=distribution)) +
    geom_density(alpha=0.5) +
    theme_bw()
"""

# ============================================================================
# STEP 5: Further Model Extensions
# ============================================================================

"""
5.1. Add motif preference (future enhancement):

  Modify niche_model_numba.py:
  
    def select_insertion_site(empty_sites, motif_scores):
      \"\"\"Weighted random selection based on motif enrichment.\"\"\"
      weights = np.exp(motif_scores[empty_sites])
      weights /= weights.sum()
      return np.random.choice(empty_sites, p=weights)
  
  Then in _simulate_branch_numba:
    - Pre-compute motif scores for all sites (before loop)
    - Call weighted selection in birth event

5.2. Add lineage-specific rates:

  Modify traverse_and_simulate to pass lineage info to each branch.
  Store r_birth, r_purge in a per-lineage dictionary.
  Look up lineage during traversal.

5.3. Implement ABC-SMC for iterative refinement:

  Use pyabc library:
    from pyabc import ABCSMC, Sampler, RV
    
    abc_smc = ABCSMC(
      models=[None],  # niche_model wrapper
      parameter_priors=[priors],
      distance_function=...,
      population_size=500
    )

5.4. Diagnostic: compare observed hotspot properties to posterior predictions:

  - Hotspot overlap (same sites in different lineages)
  - Lineage-specific hotspots
  - Correlation with gene essentiality
  - Correlation with genomic GC content
"""

print(__doc__)
