# Experiment Log: Model Alternatives

Running summary of experiments exploring different implementations of Bayesian Operator Inference.

---

## Goal

Find an implementation that:
1. Preserves full uncertainty flow: observations → GP → derivatives → operator
2. Achieves low training/prediction error
3. Is 100% stable (all posterior ROM samples integrate without blowup)
4. Has calibrated confidence intervals (90% CI covers 90% of truth)

## Problem Setup (Euler)

- 1D Euler equations, spatial domain 201 points
- Training span: [0, 0.08], Prediction span: [0, 0.15]
- 250 noisy snapshots, 6 POD modes
- Operator structure: "cAH" (constant + linear + quadratic)
- Operator learned via SVI with AutoNormal guide

---

## Experiment 1: Latent-X + Integral Constraint

**File:** `run_latent_integral.py`
**Date:** 2026-03-10
**Status:** FAILED

### Idea
Sample latent GP states explicitly using non-centered parameterization:
- X_raw ~ N(0, I), X = L @ X_raw where L = cholesky(K(ℓ, σ²))
- Observation: y | X ~ N(X, √ν) (pointwise)
- Physics: derivative + integral constraints on f(X)O^T

### Results
- **σ² collapses by 90-96%** regardless of noise mode (fixed/free/tight ν)
- ν inflates to compensate (3500%+ increase when free)
- Training error: 72-80% — model doesn't fit data
- Operator doesn't collapse (norm stays near LS) — integral constraint works for that
- The non-centered parameterization creates a σ²/ν degeneracy: model prefers to attribute all variation to noise rather than structured GP signal

### Why It Fails
The non-centered parameterization X = L @ X_raw has a fundamental degeneracy. When σ² shrinks, L shrinks, X → 0. The guide compensates by inflating X_raw, but the prior penalty on X_raw eventually loses to the ELBO benefit of simpler GP dynamics. Even fixing ν at MLE doesn't help — σ² still collapses because the SVI finds it easier to have near-zero states with trivially satisfied physics constraints.

---

## Experiment 2: Conditional GP + Integral Constraint

**File:** `run_conditional_integral.py`
**Date:** 2026-03-10
**Status:** BEST MODEL

### Idea
Don't sample latent states at all. Instead, compute GP posterior states as **deterministic functions** of sampled GP hyperparameters:

```
θ_GP = (ℓ, σ², ν) ~ LogNormal priors centered at MLE
X_eval = K_star @ K^{-1} @ y_obs    ← deterministic given θ_GP and data
μ_z = K'_star @ K^{-1} @ y_obs      ← GP derivative mean
σ²_z = diag(K'' - K' K^{-1} K'^T)   ← GP derivative variance
O ~ N(O_ls, γ·|O_ls|)               ← operator with informative prior

Physics constraints:
  Derivative: f(X_eval) O^T ≈ μ_z  (with uncertainty σ²_z + γ₂)
  Integral:   ∫ f(X)O^T ds ≈ X(t_b) - X(t_a)  (prevents null basin)
```

The marginal log-likelihood `log p(y | θ_GP)` is included as a factor (weighted by mll_weight=0.1) to provide data fidelity signal for learning GP hyperparameters.

### Role of the Two Physics Constraints

**The integral form constraint is doing most of the heavy lifting.** The derivative constraint adds some refinement but is not essential. We tested integral-only (deriv_weight=0, integral_weight=2) and it works — slightly worse than dual constraint but still functional. This is an important finding.

Why the derivative constraint is weak on its own: the GP derivative posterior variance can be very large (50–2700 in our experiments), which makes the derivative matching extremely loose. The operator barely has to do anything to satisfy it. In our earlier fixed-GP experiments, the derivative-only model was essentially vacuous.

Why the integral form is strong: it compares **state differences** (X(t_b) - X(t_a)) to the **integrated operator dynamics** (∫ f(X)O^T ds). If O ≈ 0, the integral is ≈ 0 but the state differences are NOT zero — this structurally forces a non-trivial operator. Integration is also a smoothing operation that averages out noise, unlike differentiation which amplifies it. The integral constraint provides a cleaner, more stable signal about what the operator should be.

Together: the integral form provides global consistency and prevents the null basin; the derivative constraint adds local pointwise accuracy. But if you had to pick one, pick the integral form.

### Why This Preserves UQ Flow
Different GP hyperparameter samples → different GP posterior states → different derivatives → different operator posteriors. The posterior samples contain **(θ_GP, O) pairs** with correlated uncertainty. This is NOT the 2-stage approach because:
1. GP hypers have priors and are **sampled** (not point MLE)
2. Physics constraints feed back to GP hypers through the joint ELBO
3. The full joint posterior P(O, θ_GP | y, physics) is approximated

### Why This Works (vs Latent-X)
- Eliminates ~1500 X_raw parameters → simpler optimization landscape
- No σ²/ν competition (observations enter through GP conditioning, not a separate likelihood)
- GP states are always consistent with the data (they ARE the GP posterior mean)
- Much fewer parameters: only ~18 GP hypers + ~30 operator entries

### Critical Finding: Learning Rate Matters
- **lr=1e-3 gets stuck in bad local minima** → 24% train error regardless of GP prior scale
- **lr=3e-3 finds good solutions** → 2-3% train error
- This was the single most important optimization detail

### Best Settings
```python
gamma=2.0        # operator prior scale
gamma2=2.0       # ODE constraint slack
gp_prior_scale=0.1  # LogNormal scale for GP hyper priors
mll_weight=0.1   # weight on GP marginal log-likelihood
lr=3e-3          # learning rate (CRITICAL)
num_steps=10000  # SVI iterations
num_eval_points=400  # dense evaluation grid
```

### Noise Robustness (Euler, 6 modes, 250 samples)

| Noise | POD Energy | Train Err | Pred Err | Stability | σ² drift | CV(σ²) | CI Coverage |
|-------|-----------|-----------|----------|-----------|----------|--------|-------------|
| 1%    | 97.7%     | 1.16%     | 7.02%    | 100%      | 28.8%    | ~8%    | 75.9%       |
| 3%    | 88.3%     | 2.21%     | 10.08%   | 100%      | 21.7%    | ~7%    | 67.6%       |
| 5%    | 74.2%     | 4.00%     | 16.04%   | 100%      | 39.0%    | ~5%    | 55.0%       |
| 10%   | 43.2%     | 8.19%     | 23.94%   | 100%      | 50.9%    | ~3%    | 38.3%       |
| 15%   | 26.6%     | 14.76%    | 31.30%   | 100%      | 46.8%    | ~2%    | 30.2%       |

### Comparison with Fixed-GP Model (no UQ flow)

| Noise | Fixed-GP Train | Conditional Train | Fixed-GP Pred | Conditional Pred |
|-------|---------------|------------------|--------------|-----------------|
| 1%    | 1.26%         | **1.16%**        | 7.77%        | **7.02%**       |
| 3%    | **2.52%**     | 2.21%            | **9.77%**    | 10.08%          |
| 5%    | **3.66%**     | 4.00%            | **15.3%**    | 16.04%          |
| 10%   | **6.44%**     | 8.19%            | **23.2%**    | 23.94%          |
| 15%   | **10.0%**     | 14.76%           | **27.1%**    | 31.30%          |

At low noise the conditional model matches or beats fixed-GP. At higher noise the GP hyper drift causes some degradation.

### GP Prior Scale Sweep (noise=0.03, lr=3e-3)

| GP Scale | Train Err | σ² drift | CV(σ²) | CI Coverage |
|----------|-----------|----------|--------|-------------|
| 0.001    | 2.56%     | 0.1%     | 0.1%   | 56.0%       |
| 0.01     | 2.84%     | 0.3%     | 1.0%   | 46.3%       |
| 0.05     | 2.84%     | 10.6%    | 4.3%   | 46.3%       |
| 0.10     | 2.73%     | 35.7%    | 6.4%   | 44.8%       |
| 0.20     | 2.54%     | 95.1%    | 5.7%   | 43.8%       |

Training error is robust across GP scales (all 2.5-2.9%). Larger scales give more GP uncertainty (higher CV) but also more drift. γ₂=2.0 dramatically improves CI coverage.

---

## Experiment 3: Full Robustness Suite

**File:** `run_robustness.py`
**Date:** 2026-03-10
**Status:** COMPLETE (5/6 scenarios)

### Goal
Test the conditional GP + integral constraint model (Experiment 2) across the 3 paper schemas from `generate_paper.py` plus stress tests. Tuned settings: γ=2.0, γ₂=2.0, gp_prior_scale=0.1, mll_weight=0.1, lr=3e-3. Sparse tests use 200 eval points + 15k steps; dense tests use 400 eval points + 15k steps.

### Results

| Schema | Samp | Noise | POD | Stab | Train | Pred | σ²drift | CV(σ²) | CI Cov | CI Width | Time |
|--------|------|-------|-----|------|-------|------|---------|--------|--------|----------|------|
| Paper 1: Dense, low noise | 250 | 3% | 88.3% | 100% | 2.21% | 10.08% | 21.7% | 0.070 | 67.6% | — | 146s |
| Paper 2: Sparse, med noise | 55 | 5% | 76.1% | 100% | 5.64% | 27.9% | 31.2% | 0.042 | 55.1% | 22.9% | 49s |
| Paper 3: Dense, high noise | 250 | 15% | 26.6% | 100% | 14.8% | 31.5% | 48.7% | 0.018 | 30.0% | 20.1% | 383s |
| Stress: Extreme noise | 250 | 25% | 14.0% | 100% | 43.3% | 80.0% | 96.8% | 0.009 | 9.6% | 26.6% | 383s |
| Stress: Sparse+noisy | 55 | 15% | 32.8% | 0% | ∞ | ∞ | 127% | 0.016 | 0% | 0% | 95s |
| Stress: Very sparse | 30 | 5% | — | — | — | — | — | — | — | — | JIT hang |

### Analysis

**100% stability for all paper schemas.** The model is rock-solid for the three main test cases (Paper 1-3). Even the extreme noise stress test (25% noise) maintains 100% stability.

**Paper Schema 1 (Dense, low noise)** is the clear strength: 2.21% train, 10.08% pred, 67.6% CI coverage. This is our flagship result.

**Paper Schema 2 (Sparse, medium noise)** improved significantly with tuning:
- Default (165 eval pts, 10k steps): 15.2% train → Tuned (200 eval pts, 15k steps): **5.64% train**
- 100% stability, 55.1% CI coverage
- Key insight: sparse data needs more SVI steps to converge and shouldn't have too many eval points (creates under-determined conditioning)

**Paper Schema 3 (Dense, high noise)** is fundamentally limited by POD quality (26.6% energy). With only 26.6% of variance captured by 6 modes, the ROM basis is too lossy. The 14.8% train error is actually reasonable given the basis quality.

**Stress tests reveal the limit:** sparse+noisy (55 samples, 15% noise) fails completely — POD captures only 33% of variance and ROM blows up. The very sparse case (30 samples) causes JAX JIT compilation to hang due to new array size shapes. These are expected failure modes at the boundary of what's physically possible.

**Key pattern:** Performance degrades gracefully with noise. The model stays stable even when errors are large. The dominant factor is POD energy — when noise corrupts the basis, everything downstream suffers.

### Sparse Data Tuning Notes
- For 55 samples: use `num_eval_points=200` (not auto-scaled 165), `num_steps=15000`
- The auto-scaling formula `min(400, max(100, num_samples * 3))` gives 165 for 55 samples, which is suboptimal
- Too many eval points relative to data creates ill-conditioning; too few loses resolution
- Rule of thumb: ~3-4x the number of samples, capped at 400

---

## Open Issues

1. **CI coverage still below 90% target** (best: 67.6% at noise=0.03)
   - γ₂=2.0 helps a lot (67.6% vs 45% at noise=0.03)
   - Could try learning γ₂ per mode, or more flexible guide (AutoMultivariateNormal, AutoLowRankMultivariateNormal)
   - Could also try temperature scaling the posterior
2. **High-noise degradation** — at noise=0.15+, the model struggles
   - Root cause is POD basis quality (26.6% at noise=0.15, 14% at noise=0.25)
   - GP hyper drift increases to 49-97%
   - Could try adaptive number of modes (fewer modes at higher noise)
3. **Heat experiment** not yet adapted (multi-trajectory, cAHBN operators)
4. **Lengthscale drift** — ℓ consistently increases 20-40%, even with tight priors
   - This may be physics-driven (smoother states are easier to model)
5. **Very sparse data (30 samples)** causes JAX JIT hang — different array shapes trigger recompilation
   - Workaround: precompile with dummy data at target shape, or use fixed-size padding

---

## Failed Approaches (from earlier work, pre-conversation)

- **Joint GP (marginal likelihood)**: GP marginal likelihood (~120K) overwhelms physics (~9K). GP σ² always drifts massively.
- **KL annealing**: Learns huge γ₂ as escape valve. 8.5% stability.
- **Observation weight tuning (0.001-1.0)**: Doesn't help balance marginal likelihood vs physics.
- **Warm-start joint GP**: σ² collapses from 0.33 → 0.02, noise inflates 0.005 → 0.019.
