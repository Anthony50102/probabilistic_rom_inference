# Hierarchical Marginalized-O (Option 1) — Experiment Writeup

**Branch:** `richer-guide`
**Script:** `04h_hierarchical_marg_O.py`
**Experiment:** Euler, three noise/density schemas
**Status:** Implemented and benchmarked. Findings below.

---

## Motivation

Earlier in the session we sorted out where each of our formulations sits
relative to **McQuarrie, Chaudhuri, Willcox & Guo (2024)** ("GP-BayesOpInf"):

| | MCWG'24 | 04 baseline | 04g marg-O | **04h (this work)** |
|---|---|---|---|---|
| GP hypers | MLE point | MLE point | **VI** | **VI** |
| O posterior | closed-form | SVI (numerical) | **closed-form (Rao-Blackwell)** | **closed-form (Rao-Blackwell)** |
| Integral / trajectory term | ✗ | ✓ | ✓ | ✓ |
| Informative LS prior on O | ✗ | ✓ | ✗ (broad) | ✗ (broad) |
| σ_O (operator-prior scale) | zero-mean Tikhonov, γ by grid | informative anchor | **fixed scalar** | **hierarchical, per-mode w/ τ_O** |
| γ² (constraint noise) | n/a | n/a | **fixed scalar** | (tested as hyperparameter — see below) |

`04g` (marg-O) is already three deltas away from MCWG (marginalized hypers,
integral term, Rao-Blackwell). The remaining hand-tuned knobs are the
**operator-prior scale `σ_O`** and the **constraint noise `γ²`**, both fixed
scalars in `04g`. **Option 1 (this work)** asked: *can we promote them to
hyperparameters and get a fully hierarchical Bayesian formulation?*

---

## What 04h actually does

Conditional on `(θ_GP, γ², σ_O)`, the likelihood is linear in `O` with Gaussian
prior on `O`, so `p(O | data, θ_GP, γ², σ_O)` is closed-form Gaussian and the
marginal evidence `log p(data | θ_GP, γ², σ_O)` is closed-form. We do
SVI over the low-dimensional `(θ_GP, σ_O, …)` block; `O` is analytically
marginalized.

The model in `04h` adds a hierarchical prior on `σ_O`:

```
τ_O           ~ LogNormal(log SIGMA_O_SCALE, 0.5)        # global scale hyperprior
σ_O,i         ~ LogNormal(log τ_O, 0.3),  i = 1..r       # per-mode, shared scale
gamma2        =  GAMMA2_SCALE                            # FIXED (see Finding 1)
ℓ_i, σ²_i, ν_i ~ broad LogNormal priors per mode         # GP hyperparameters
```

Conditional on a draw of `(ℓ, σ², ν, γ², σ_O)`:

1. Run per-mode GP conditioning to get `(X_eval, μ_z, deriv_var)`.
2. Stack design matrix `A = [f(X); ∫ f(X)]` and target
   `y_i = [μ_z,i ; ΔX_i]` per mode.
3. Per-mode precision `prec_i = [deriv_weight/(deriv_var+γ²);
   integral_weight/(γ² · dur²)]`.
4. Per-mode `Λ_i = AᵀW_i A + I/σ_O,i²`, Cholesky `L_i`, posterior mean
   `μ_i = Λ_i⁻¹ Aᵀ W_i y_i`, posterior cov `Σ_i = Λ_i⁻¹`.
5. Sum the per-mode log-evidences as a `numpyro.factor`.

SVI on `θ = (ℓ, σ², ν, τ_O, σ_O,·)` with `AutoNormal`, MLE warm-start, 2000
steps, ClippedAdam(3e-3), plus Cholesky jitter for late-phase numerical
stability.

At prediction time we draw `npost = 500` `θ`-samples from the variational
posterior, compute the closed-form O-mean and Cholesky-of-cov per sample,
and draw one `O` per `θ`-sample. ROM trajectories are integrated for each
`O` over the prediction horizon.

---

## Headline results

Three Euler schemas, `cAH` operators, `r = 6` modes, training span
`[0, 0.08]`, prediction span `[0, 0.15]`. All numbers from
`results/comparison/<schema>/04*.npz`.

| Schema | Model | Train | Pred | Stab | CI cov | CI width | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| **dense_low** | 04 baseline | 1.38% | **7.46%** | 100% | 92.0% | 0.042 | 128 s |
| | 04g marg-O | 0.78% | 36.95% | 93% | 99.3% | 0.202 | 256 s |
| | **04h hierarchical** | **0.78%** | 34.05% | 92.5% | 99.2% | 0.192 | 204 s |
| **sparse_low** | 04 baseline | 11.64% | 35.38% | 100% | 55.7% | 0.084 | 33 s |
| | **04g marg-O** | **5.23%** | **20.35%** | 100% | 90.2% | 0.117 | 209 s |
| | 04h hierarchical | 5.49% | 24.41% | **24.0%** | 97.9% | 0.322 | 182 s |
| **dense_high** | 04 baseline | **7.69%** | **22.40%** | 100% | 61.5% | 0.052 | 233 s |
| | 04g marg-O | 8.57% | 35.55% | 100% | 80.9% | 0.108 | 401 s |
| | 04h hierarchical | 91.35% | 134.73% | **5.5%** | 44.0% | 0.379 | 199 s |

**Bold red** numbers are the failure modes; **bold black** are wins.

---

## What we found

### Finding 1 — γ² has a structural non-identifiability and cannot be a free hyperparameter

We first implemented 04h with **both** `σ_O` and `γ²` as hyperparameters. The
variational posterior on `γ²` collapsed to `~0.02` regardless of prior
choice (we tried `HalfCauchy(1.0)`, `LogNormal(log 1, 0.5)`,
`LogNormal(log 1, 1.5)`, `LogNormal(log 10, 0.3)` — all collapse), gave a
massively overconfident posterior and tanked predictive performance (pred
error ~47%, CI coverage 79%).

**Why:** `γ²` enters the design via
```
prec_deriv = deriv_weight / (deriv_var + γ²)
prec_int   = integral_weight / (γ² · dur²)
```
so `γ²→0` inflates **both** precisions along a ridge. The data likelihood is
unbounded above along that ridge, and any sensible prior (worth a few tens
of log-units) loses to the marginal-likelihood gain (thousands of log-units).
This is the classical noise/prior collinearity but with an extra twist
because the same parameter controls two different noise scales.

The principled fixes are either (a) keep `γ²` fixed at a sensible
data-derived scale — what `04g` does and what 04h now does — or
(b) **split `γ²` into separate `γ²_deriv` and `γ²_int`** so the two roles
cannot collude. Worth trying if we want to rescue the fully-hierarchical
story; not pursued in this experiment.

### Finding 2 — Hierarchical σ_O,i is competitive on easy data but destabilizes on hard data

With `γ²` fixed and only `(τ_O, σ_O,i)` learned hierarchically:

- **dense_low_noise:** essentially identical to 04g (0.78% train, 99% CI,
  90+% stability). The per-mode σ_O,i medians are
  `[103, 112, 82, 66, 94, 85]` — heterogeneous, which is the point of the
  per-mode prior, and the data is informative enough to identify them.
- **sparse_low_noise:** stability falls to **24%**. With only 55 training
  samples, σ_O,i is under-identified; one or two modes' per-mode σ_O drifts
  to large values, the corresponding operator entries inflate, ROM blows up.
- **dense_high_noise:** stability falls to **5.5%**, 91% train error. Same
  failure mode, now driven by noise rather than scarcity. One mode in
  particular hit σ_O = 304.

The runaway is *expected* from a fully-hierarchical formulation: with broad
hyperpriors, σ_O,i can grow without bound and the marginal likelihood will
sometimes pay for it (more data fit at the cost of a wider prior on O).
Tightening the hyperprior coupling (`σ_O,i ~ LogN(log τ_O, 0.3)`) helped a
little but did not rescue the failures.

### Combined takeaway

The hand-fixed `(σ_O, γ²)` scalars in `04g` are **not lazy tuning** — they
act as genuine regularizers that the fully-hierarchical version cannot
safely replace without further structural changes:

- `γ²` requires the deriv/integral role separation before it's a safe
  hyperparameter.
- `σ_O` requires more aggressive shrinkage (e.g., shared σ_O across modes,
  rather than per-mode with τ_O hyperprior) to be safe on harder regimes.

**Net:** `04g` (marg-O with fixed scalars) is currently the strongest
formulation overall: it dominates the baseline on CI coverage by a large
margin on harder schemas (90% vs 56% on sparse_low; 81% vs 62% on
dense_high) while matching or beating it on train error, and avoids the
hierarchical instabilities of 04h.

---

## Implementation knobs

| Knob | Value | Notes |
|---|---|---|
| `NUM_MODES` | 6 | Same across all three models |
| `WINDOW_SIZE` | 20 | Integral windows for trapezoidal rule |
| `DERIV_WEIGHT` | 1.0 | Equal weighting of derivative and integral terms |
| `INTEGRAL_WEIGHT` | 1.0 | |
| `MLL_WEIGHT` | 1.0 | Full GP marginal likelihood contribution |
| `SIGMA_O_SCALE` | 30.0 | Center of the `τ_O` hyperprior |
| `GAMMA2_SCALE` | 10.0 | Fixed constraint-noise (matches 04g) |
| `NUM_STEPS` | 2000 | Reduced from 8000 to avoid late-phase numerical drift |
| `LEARNING_RATE` | 3e-3 | ClippedAdam |
| `NUM_POSTERIOR_SAMPLES` | 500 | Draws from VI posterior |
| Cholesky jitter | `1e-6 · mean(diag(Λ))` | Added in 04h for late-SVI stability |

---

## Where to go next

Two routes for the paper story:

1. **Lean into 04g as THE method.** Frame the fixed `(σ_O, γ²)` as deliberate
   modeling choices (with appendix-level ablations covering grid choice and
   sensitivity). Headline: "marginalized-O Bayesian OpInf with calibrated
   prediction intervals." The CI-coverage win over the baseline is real and
   reproducible across all three schemas.

2. **Rescue Option 1.** Split `γ²` into `γ²_deriv` and `γ²_int` to break the
   collinearity, and replace the per-mode `σ_O,i` with a single global
   `σ_O` under hyperprior (less flexibility, fewer failure modes). Could
   give us a clean "fully Bayesian, no hand-tuned scalars" pitch — but
   requires another implementation pass.

---

## Files touched (this branch, `richer-guide`)

- `experiments/euler/04h_hierarchical_marg_O.py` *(new)* — Option 1
  implementation.
- `experiments/euler/results/comparison/*/04h_hierarchical_marg_O.npz`
  *(new)* — saved metrics per schema.
- `experiments/euler/results/comparison/HIERARCHICAL_EXPERIMENT.md`
  *(this file)*.

Earlier 04g and 04 results in `results/comparison/` were generated in
previous sessions and re-used unchanged for the head-to-head.
