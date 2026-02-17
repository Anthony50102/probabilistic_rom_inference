# Bayesian Modeling Pitfalls: A Practical Guide

This document covers the most common issues encountered in Bayesian inference, why they happen, and what you can do about them. These apply to any probabilistic model, but examples are framed in the context of operator inference and the models in this repository.

---

## Table of Contents

1. [Correlated Posterior](#1-correlated-posterior)
2. [Low Effective Sample Size (ESS)](#2-low-effective-sample-size-ess)
3. [Poor Convergence (R-hat)](#3-poor-convergence-r-hat)
4. [Divergent Transitions](#4-divergent-transitions)
5. [Prior Sensitivity](#5-prior-sensitivity)
6. [Non-Identifiability](#6-non-identifiability)
7. [Multimodality](#7-multimodality)
8. [Poor Mixing](#8-poor-mixing)
9. [Diagnosing with This Codebase](#9-diagnosing-with-this-codebase)

---

## 1. Correlated Posterior

### What it looks like
The posterior samples of two (or more) parameters are highly correlated — when one goes up, the other consistently goes up or down. The correlation matrix shows entries with |r| > 0.9.

### Why it happens
- **Redundant parameterization**: Two parameters play similar roles in the model, so the data can't distinguish them independently. For example, if an operator has rows that produce similar dynamics, their entries may be highly correlated.
- **Funnel geometry**: In hierarchical models, the scale and location parameters can form a "funnel" shape in the joint posterior that creates strong correlations.
- **Insufficient data**: When the likelihood surface is flat in some directions, many parameter combinations explain the data equally well.

### Why it matters
- Correlated posteriors are harder for samplers to explore efficiently, leading to slow mixing and low ESS.
- Point estimates (like the posterior mean) may be misleading — the marginals look wide but the joint distribution is actually narrow along a ridge.
- Predictions may still be fine, but parameter-level interpretation becomes unreliable.

### What to do
- **Reparameterize**: Use a non-centered parameterization (e.g., write `x = mu + sigma * z` where `z ~ Normal(0,1)` instead of `x ~ Normal(mu, sigma)`).
- **Stronger priors**: Tighter priors on individual parameters can break correlations by anchoring the posterior.
- **Dimensionality reduction**: If many operator elements are correlated, consider whether the operator has more degrees of freedom than the data can constrain.
- **Use a guide that captures correlations**: In SVI, use `AutoMultivariateNormal` instead of `AutoNormal` to let the variational approximation model correlations.

---

## 2. Low Effective Sample Size (ESS)

### What it looks like
ESS values much smaller than the actual number of samples drawn (e.g., ESS = 20 from 1000 samples). The diagnostic report flags any parameters with ESS < 100.

### Why it happens
- **Autocorrelation**: Each MCMC sample depends on the previous one. If steps are small (e.g., due to difficult geometry), consecutive samples are nearly identical, so you effectively have far fewer independent samples than you drew.
- **Poor tuning**: The sampler's step size or mass matrix is poorly calibrated for the posterior geometry.
- **High dimensionality**: In high-dimensional spaces, it takes more steps to explore the full posterior.

### Why it matters
- Low ESS means your posterior summaries (means, credible intervals) are noisy and may change significantly if you rerun the chain.
- Tail quantiles are especially unreliable — you need ESS > 400 for reliable 95% intervals.

### What to do
- **Run longer chains**: The simplest fix. More samples → higher ESS.
- **Thin the chain**: Keep every k-th sample, though this is less efficient than just running longer.
- **Reparameterize**: If the geometry is the problem, changing coordinates can dramatically improve ESS.
- **Increase warmup**: Give the sampler more time to tune its step size and mass matrix.
- **Note for SVI**: ESS is not meaningful for SVI since samples are drawn independently from the variational distribution. If using SVI, ESS should approximately equal the number of samples drawn.

---

## 3. Poor Convergence (R-hat)

### What it looks like
R-hat values greater than 1.01 (often written as R̂). Values near 1.0 indicate convergence; values above 1.1 are a serious red flag.

### Why it happens
- **Chains haven't run long enough**: Different chains started in different regions and haven't yet settled into the same stationary distribution.
- **Multimodal posterior**: Different chains found different modes and are stuck there.
- **Stiff dynamics**: The posterior has regions that are hard to traverse, keeping chains separated.

### Why it matters
- If R-hat is high, your posterior estimates are unreliable — they depend on which chain you look at.
- Any downstream predictions (ROM solves, uncertainty bounds) will be inconsistent across chains.

### What to do
- **Run longer**: More warmup and more samples often solve the problem.
- **Better initialization**: Initialize chains near reasonable parameter values (e.g., from SVI or the deterministic solution).
- **Check for multimodality**: If the posterior is genuinely multimodal, you may need specialized samplers or to constrain the model.
- **Require multiple chains**: Always run at least 2 chains so you can compute R-hat. Single-chain MCMC gives no convergence guarantee.

---

## 4. Divergent Transitions

### What it looks like
NumPyro/Stan prints warnings like "There were X divergent transitions after warmup." The diagnostic report counts these explicitly.

### Why it happens
Divergences occur when the Hamiltonian Monte Carlo (HMC/NUTS) integrator encounters regions of the posterior where the curvature changes too rapidly for the current step size. The numerical integration goes off-track, producing unreliable samples.

Common causes:
- **Funnel geometry**: Hierarchical models often have regions where the posterior is extremely narrow (the bottom of the funnel), and the sampler can't navigate these with a fixed step size.
- **Multimodality with narrow bridges**: The posterior has modes connected by very thin ridges.
- **Stiff likelihood**: The ODE constraint likelihood in our models can create stiff gradients if `gamma2` is too small (tight constraint) relative to the GP covariance.

### Why it matters
- Divergences mean the sampler has systematically missed parts of the posterior — your estimates are biased.
- Even a small number of divergences (say 1% of transitions) can indicate that an important region of parameter space is being ignored.

### What to do
- **Increase `target_accept_prob`**: Setting it to 0.95 or 0.99 forces smaller step sizes that can navigate tight curvature. This is slower but more reliable.
- **Reparameterize**: Non-centered parameterizations eliminate funnels in hierarchical models.
- **Adjust model stiffness**: In our context, increasing `gamma2` (loosening the ODE constraint) can smooth out the posterior geometry.
- **Increase `max_tree_depth`**: Allows the sampler to take more leapfrog steps per transition, helping it traverse tricky regions.

---

## 5. Prior Sensitivity

### What it looks like
The posterior changes dramatically when you change the prior, even with the same data. The prior-posterior overlap diagnostic helps detect this:
- **Overlap > 95%**: The data barely moved the prior — either the prior is too tight, or the data is uninformative for this parameter.
- **Overlap < 5%**: The posterior is far from the prior — either the data is very informative (good) or the prior was badly misspecified (check your assumptions).

### Why it happens
- **Weak likelihood**: When the data provides little information about a parameter (flat likelihood), the prior dominates the posterior.
- **Small datasets**: With limited data, Bayesian inference naturally relies more on the prior.
- **Prior misspecification**: Priors centered far from the true parameter value or with inappropriate spread.

### Why it matters
- If your results are prior-sensitive, your conclusions depend on assumptions rather than data — this undermines the whole point of inference.
- For predictions, prior-dominated parameters contribute prior-driven uncertainty that may be unrealistically large or small.

### What to do
- **Prior predictive checks**: Before fitting, sample from the prior and simulate data. Does the simulated data look anything like your real data? If not, your prior is putting mass in implausible regions.
- **Sensitivity analysis**: Run the model with several different reasonable priors and check if conclusions change. If they do, report this honestly.
- **Use weakly informative priors**: Priors that rule out clearly absurd values but are otherwise broad (e.g., `gamma=50` rather than `gamma=0.1` for operator entries that could span a wide range).
- **Get more data**: The best cure for prior sensitivity.

---

## 6. Non-Identifiability

### What it looks like
Multiple very different parameter values produce nearly identical model predictions. The posterior is spread along a ridge or manifold rather than concentrated at a point.

### Why it happens
- **Structural non-identifiability**: The model equations genuinely can't distinguish certain parameter combinations. For example, if the operator appears in the model as a product `a * b`, you can only identify the product, not `a` and `b` separately.
- **Practical non-identifiability**: In principle the parameters should be identifiable, but the available data doesn't constrain them well enough.
- **Overparameterization**: More operator entries than the dynamics require, common when the number of POD modes is large relative to the training data.

### Why it matters
- Non-identifiable parameters will show wide posteriors, high correlations, and low ESS.
- Parameter-level inference is meaningless for non-identifiable quantities.
- However, if the *predictions* are still well-constrained (the operator combinations that matter are identified), the model may still be useful.

### What to do
- **Reduce model complexity**: Use fewer POD modes, a simpler operator structure, or stronger regularization.
- **Add informative priors**: External knowledge about plausible operator values can break degeneracies.
- **Profile the posterior**: Check if the prediction quality varies along the ridge. If not, the non-identifiability is benign for prediction purposes.
- **Check your operator structure**: Make sure all operator terms (`c`, `A`, `H`, etc.) are actually needed for the dynamics.

---

## 7. Multimodality

### What it looks like
The posterior has multiple distinct peaks (modes). Different MCMC chains settle into different modes, giving very different answers. R-hat may be high if chains found different modes.

### Why it happens
- **Symmetries**: The model may be invariant to sign flips or permutations of certain parameters.
- **Multiple local optima**: The likelihood surface has several peaks, and the prior isn't strong enough to prefer one.
- **Label switching**: In mixture-like models, swapping component labels gives identical likelihoods.

### Why it matters
- Standard MCMC summaries (posterior mean) become meaningless — the mean of two modes is typically in a low-probability region between them.
- Prediction uncertainty will be underestimated if the sampler only found one mode, or overestimated if it's trying to average across modes.

### What to do
- **Break symmetries with priors**: Add ordering constraints or informative priors that prefer one mode.
- **Use multiple random initializations**: Run many short chains from different starting points to map out the modes.
- **Consider SVI first**: Variational inference tends to find a single mode, which can be used to initialize MCMC in the right region.
- **Post-process by mode**: If multimodality is real and meaningful, analyze each mode separately.

---

## 8. Poor Mixing

### What it looks like
Trace plots show the chain "stuck" in one region for long stretches, punctuated by rare jumps. The samples look like they're exploring a tiny neighborhood rather than the full posterior.

### Why it happens
- **Difficult geometry**: Narrow ridges, strong correlations, or high curvature make it hard for the sampler to propose moves that are accepted.
- **High dimensionality**: In very high-dimensional spaces, nearly all random directions are orthogonal to the interesting structure.
- **Bad tuning**: The step size is too small (slow exploration) or too large (high rejection rate).

### Why it matters
- Poor mixing means your posterior samples are not representative — you're seeing a biased view of the posterior.
- ESS will be low, and R-hat may not catch the problem if all chains are mixing poorly in the same way.

### What to do
- **Check trace plots**: The single most informative diagnostic. Good mixing looks like a "fuzzy caterpillar."
- **Reparameterize**: This is almost always the right first step for mixing problems.
- **Increase warmup**: Give the sampler more time to adapt.
- **Use a better sampler**: NUTS (the default in NumPyro) is usually good, but you can increase `max_tree_depth` for very complex posteriors.
- **Try SVI**: If MCMC can't mix, a variational approximation gives you *something* — it won't be exact, but it may be good enough for your application.

---

## 9. Diagnosing with This Codebase

The `core/diagnostics.py` module provides automated detection of all the above issues. Here's the typical workflow:

### Quick Start

```python
from core.diagnostics import run_diagnostics

report = run_diagnostics(
    samples=samples,              # Posterior samples dict
    param_name="O",               # Operator parameter name
    prior_mean=prior_operator,    # Prior mean (from grid search)
    prior_std=GAMMA,              # Prior std (gamma hyperparameter)
    mcmc_result=mcmc_result,      # Pass MCMC result for divergence checks
    samples_by_chain=None,        # Per-chain dicts for R-hat (MCMC only)
    verbose=True,                 # Print summary report
    plot=True,                    # Generate diagnostic plots
)
```

### What You Get

| Diagnostic | Plot | What to look for |
|-----------|------|-----------------|
| Correlation matrix | Heatmap | Off-diagonal entries near ±1 |
| ESS | Bar chart | Bars below the threshold line |
| Trace plots | Time series + histogram | "Fuzzy caterpillar" = good; stuck patches = bad |
| Prior vs posterior | Overlaid densities | Complete overlap = data not informative; no overlap = check prior |
| Rank plots (MCMC) | Histograms per chain | Uniform = good mixing; U-shaped = poor mixing |
| Divergences (MCMC) | Count in report | Any divergences warrant investigation |
| R-hat (MCMC) | Value in report | Should be < 1.01 |

### Interpretation Priorities

1. **Start with trace plots** — they give the most intuitive sense of whether something is wrong.
2. **Check ESS** — if ESS is low, your summaries are noisy regardless of everything else.
3. **Look at correlations** — high correlations explain *why* ESS is low and point to what to fix.
4. **Check divergences** (MCMC) — these are the most serious red flag.
5. **Review prior-posterior overlap** — this tells you whether the data is actually informing the model.

### SVI vs MCMC Diagnostics

| Diagnostic | SVI | MCMC |
|-----------|-----|------|
| Correlation | Meaningful (shows variational posterior correlation) | Meaningful |
| ESS | ≈ n_samples (independent draws from guide) | Critical to check |
| R-hat | N/A (no chains) | Critical to check |
| Divergences | N/A | Critical to check |
| Prior-posterior | Meaningful | Meaningful |
| Trace plots | Less useful (independent draws) | Very useful |

For SVI, the most important diagnostics are **correlation** and **prior-posterior overlap**. The ELBO loss curve (plotted during inference) is also important — it should plateau, not still be decreasing.

---

## Further Reading

- Gelman et al., *Bayesian Data Analysis* (3rd ed.), Chapter 11 — convergence diagnostics
- Betancourt, "A Conceptual Introduction to Hamiltonian Monte Carlo" (2017) — intuition for divergences and geometry
- [Stan User's Guide: Diagnostics](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html) — practical advice applicable to any probabilistic programming framework
- Vehtari et al., "Rank-Normalization, Folding, and Localization: An Improved R̂" (2021) — modern R-hat
