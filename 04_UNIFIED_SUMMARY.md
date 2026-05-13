# 04 Unified — Marginalised-O × Weak-Form Bayesian OpInf

A consolidation of three feature branches into a single 04-method family:

- `tests` (main working branch) — base infrastructure, tumor variants, plotting
- `richer-guide` — analytical marginalisation of O (closed-form posterior)
- `weak-form-revision` — WSINDy-style smooth bump test functions

All three composed cleanly because **both data terms are linear in O**, so 04g's
marginal-O conjugacy applies to the concatenated derivative + weak-form system.

## Method

For each ROM mode $i$:

$$
\begin{bmatrix} f(X) \\ \int\psi_k\,f(X)\,dt \end{bmatrix}\,O_i \;\approx\; \begin{bmatrix} \mu_z \\ -\int\psi_k'\,X\,dt \end{bmatrix} \quad\text{with}\quad O_i \sim \mathcal{N}(0,\sigma_O^2 I)
$$

Stacking gives one design matrix $A$ per mode with per-row precisions from
GP-derivative variance (top block) and from $\gamma^2\int\psi_k^2$ (bottom
block). The closed-form per-mode evidence

$$
\log p(y_i\mid\theta) = -\tfrac12\big(y_i^\top P_i y_i - \mu_i^\top b_i + m\log\sigma_O^2 + \log|\Lambda_i| + \log|\Sigma_i| + N\log 2\pi\big)
$$

is what SVI optimises; $\theta$ is only the $3r$-dim GP hyperparameter vector.
Posterior $O$ is drawn from its closed-form conditional $\mathcal{N}(\mu_i, \Lambda_i^{-1})$
per $\theta$-sample.

For multi-IC experiments (heat), rows from each IC stack vertically — $O$ is
shared, so this is just a bigger linear system; one Cholesky per mode.

## Per-experiment results

| Experiment    | Operators | Regime              | Stab | Train | Pred  | CI    | Time |
|---------------|-----------|---------------------|------|-------|-------|-------|------|
| Euler         | cAH       | dense low-noise     | 100% |  0.9% | 17.6% | 99.4% |  99s |
| Euler         | cAH       | sparse low-noise    | 100% |  5.1% | 20.1% | 92.2% |  26s |
| Euler         | cAH       | dense high-noise    | 100% |  8.6% | 32.3% | 85.7% |  94s |
| Burgers-2D    | cAH       | dense medium-noise  |  49% |  1.6% |  4.6% | 93.4% |  18s |
| Heat (5 ICs)  | cAHBN     | sparse medium-noise |  66% |  1.8% | 17.7% | 98.0% |  72s |
| Tumor         | cA        | dense low-noise     | 100% |  2.3% |  5.7% | 96.4% |  34s |
| Tumor         | cA        | dense medium-noise  | 100% |  3.2% |  3.9% | 60.8% |  25s |
| Tumor         | cA        | dense high-noise    | 100% |  6.5% | 17.2% | 59.2% |  14s |

(All on richer-guide branch / thesis_claude_richer_guide worktree.)

## Per-experiment adaptations

All experiments share the same `build_model`/`run_experiment` skeleton; the
template lives at `experiments/burgers_2d/04_unified.py`. Per-experiment
differences:

- **Euler** (`experiments/euler/04_unified.py`): broad GP priors (no MLE
  anchoring) work because state variances are O(1).
- **Burgers-2D**: σ²~10³ requires **MLE-anchored LogNormal priors**
  (`gp_prior_scale=0.1`), **trace-based ridge** on $\Lambda_i$
  (`ridge = inv_σO² + 1e-6·max(tr(M)/m, 1)`), and **stronger jitter**
  `max(1e-5, σ²·1e-3)` for numerically stable Cholesky.
- **Heat** (multi-IC): operator shared across ICs; per-IC kernel matrices and
  $\psi_k$ design rows precomputed, then vertically concatenated. Inputs passed
  per IC into `_assemble_data_matrix`. **Bug fix**: `input_func` must return
  `np.ndarray` (not jnp) for opinf 0.6's ROM predict; wrapped at eval time.
- **Tumor**: autonomous `cA` operators, **adaptive POD** (probe with
  `NUM_MODES + 4` modes, keep only those with SNR $\sigma^2/\nu>10$ to discard
  noise-dominated modes), `t_pred` constructed from `config.PREDICTION_DAYS`.
- **FitzHugh-Nagumo**: skipped — exploratory run showed the marg-O posterior is
  too wide for FN's stiff state×input coupling (26% stab, 66% train). The
  2-stage `04_conditional_integral.py` remains the recommended path for FN.

## File map

```
experiments/euler/04_unified.py            # reference: broad priors
experiments/burgers_2d/04_unified.py       # reference: MLE-anchored + ridge
experiments/heat/04_unified.py             # multi-IC variant
experiments/tumor/04_unified.py            # adaptive-POD autonomous variant
experiments/euler/compare_04_variants.py   # head-to-head against 04/04b/04g
```

## Commit history (richer-guide)

```
4e05f41 Propagate 04_unified.py to heat experiment (multi-IC)
215fe14 Propagate 04_unified.py to tumor experiment
cbbab82 Propagate 04_unified to burgers_2d
74a31bf Add 04_unified.py for euler + compare script
684f91e Merge weak-form-revision → richer-guide
5be6a98 Merge tests → richer-guide
```

## Known limitations / future work

1. **Burgers stability (49%)** — for sparse data the marginal-O tails are
   wider than the 2-stage posterior. Either tighter `σ_O` or a hierarchical
   `σ_O ~ HalfCauchy` prior (already plumbed via `HIER_SIGMA_O=1` env var)
   could help.
2. **Heat CI over-coverage (98% vs 90%)** — posterior slightly broad; same
   `σ_O` tuning lever applies.
3. **Tumor CI under-coverage (59-60% at higher noise)** — opposite issue:
   posterior too narrow once SNR-truncation discards modes the GP MLL would
   otherwise penalise. Hierarchical noise prior or downweighting MLL might
   widen.
4. **FN deferred** — input×state coupling needs revisiting. Either a stronger
   informative O prior centred on least-squares, or stay on
   `04_conditional_integral.py`.
