# Experimental input-aware GP for chemotherapy

## Status and scope

**This is an opt-in statistical-model extension, not a rollback or a pure
numerical bug fix.** The default `historical` profile retains the original
time-only GP. The input-aware implementation and both sets of results remain
available for research, but its improved results must not be attributed to the
unchanged original model.

Both profiles use the same matched experiment: 80 noisy observations on days
5-70, four POD modes fitted to clean nominal training snapshots, and 400
prediction times through day 110. Each noise regime is fitted only at nominal
dose and evaluated at 0.8x, 1x, and 1.2x without dose-specific refitting.
The neural baseline's saved weights and predictions are unchanged.

## Statistical-model change

The reduced ODE is unchanged:

```text
dq/dt = c + A q + B alpha(t) + N [alpha(t) q]
```

All of its operators remain inferred. No chemotherapy coefficient is fixed to
a known physical value.

The original GP has a zero mean and an RBF covariance in time. The experimental
GP adds three Gaussian random coefficients per mode. In code-style notation:

```text
exposure(t) = integral of alpha(s) from day 5 to t
h(t) = [1, standardized_time(t), standardized_exposure(t)]

q_i(t) = dot(beta_i, h(t)) + residual_gp_i(t)
beta_i ~ Normal(mean=zeros(3), covariance=mode_energy_i * identity(3))

kernel_i(t, s) = rbf_kernel_i(t, s)
                 + mode_energy_i * dot(h(t), h(s))
```

`mode_energy_i` is the variance of the noisy training coefficients of mode i,
plus a small numerical floor. The residual GP variance and lengthscale remain
inferred. The trend coefficients are analytically marginalized, so the
marginal GP still has zero prior mean; this is not a deterministic mean curve
fitted before inference.

Centering and standardization use training times only. Exposure is the exact
integral of the existing piecewise-linear input table. Its derivative is the
same tabulated input. Accordingly, the feature derivatives are zero for the
constant, inverse training-time standard deviation for time, and input divided
by training-exposure standard deviation for exposure. These enter both GP
derivative means and covariances.

This introduces a substantive inductive bias: treatment responses can be
connected through cumulative exposure, not only through temporal proximity.
It does not enforce a negative response, monotonicity, or an exact decay law.
Forecasts still integrate the fitted ODE rather than extrapolating the GP.

## Other differences from the original profile

| Setting | Original `historical` | Experimental `input-aware` |
|---|---|---|
| GP kernel | Time-only RBF | RBF plus marginalized time/exposure trend |
| Observation log-likelihood multiplier | 0.1 | 1.0 |
| Noise-prior median | 1% of each mode's signal variance | Declared voxel-noise covariance projected into the basis |
| GP kernel nugget | Maximum of 1e-5 and 0.001 times GP variance | Dtype-aware roundoff scale |
| Operator inference | Historical jittered normal equations | Whitened augmented QR |
| Precision | Original/default JAX setting | Scoped float64 |
| GP-based IC uncertainty | Original RBF-only calculation | Augmented-kernel posterior |

Measurement-noise variances are derived from the declared generator's scale
and active-voxel mask at the clean training snapshots, excluding the exact
initial observation. They are prior locations: noise is still inferred.
This retains the benchmark's idealized clean-training-data preprocessing and
is not a claim about available noise information in clinical measurements.

Operator priors remain zero-mean and hierarchical. Rank, constraint weights,
model-error slack, training steps, posterior sample counts, and the nominal
training dose are unchanged. The configuration is explicit in
`04_unified_chemo.make_config`.

## What the original-model investigation established

1. The pre-centralization implementation at `0531428` also fails on the
   matched data: 2101.53% nominal full-field forecast error versus 2098.03%
   for the centralized implementation. A centralization rollback does not
   remove this failure.
2. The older 5-60/90 experiment still reproduces its 6.62% training and 9.59%
   forecast **reduced-coordinate** errors. Those are different data, windows,
   and error normalization from the matched full-field comparison.
3. Extending the input table while retaining the old observations and basis
   does not collapse the dominant GP. Changing the observations while keeping
   the old basis does. Replacing only the last observation with day 70 can
   collapse other modes. The trigger is therefore not just forecast duration
   or a basis change.
4. The rescaled observation locations leave a gap around the day-40 treatment.
   The original matched fit has a dominant GP lengthscale near 685 days and
   a noise variance near 124, with roughly 95% observation reconstruction
   error. This is an inference collapse during training, before forecasting.
5. Correcting noise scales without the new trend can fit observations closely
   yet reconstruct the intervening training trajectory poorly: approximately
   16% dominant-mode state error and 43% weak-target error on the dense
   diagnostic grid. Dense clean states were used to diagnose this, not supplied
   to the experimental inference.

### Original-objective initialization audit

Four starts were registered before running on the same matched 1% noise data:
the original initialization key, two deterministic folded keys, and a
data-faithful warm guide from the earlier untempered-likelihood diagnostic.
The warm guide was moment-matched from 500 saved draws in AutoNormal's
unconstrained coordinates; only its initial guide parameters were transferred.
All four runs then used the **unchanged original objective**, including the
0.1 observation-likelihood multiplier, original noise priors, RBF kernel,
normal equations, float32, and 12,000 updates at learning rate 0.003.

Selection maximized training ELBO on 256 common, independent fixed evaluation
keys. Higher is better; uncertainties below are Monte Carlo standard errors,
not uncertainty across optimization runs. GP fidelity was diagnostic only.

| Start | Final original-target ELBO | Dominant GP observation error |
|---|---|---|
| Original | 321.126 +/- 0.202 | 94.84% |
| Fold-in 1 | 321.291 +/- 0.156 | 94.28% |
| Fold-in 2 | 321.177 +/- 0.182 | 95.07% |
| Data-faithful warm guide | 321.091 +/- 0.202 | 94.85% |

The strongest evidence is the warm guide's trajectory under the original
objective:

| Quantity | Before original-objective training | After |
|---|---|---|
| Original-target ELBO | -258.615 +/- 0.272 | 321.091 +/- 0.202 |
| Dominant GP observation error | 3.00% | 94.85% |
| Dominant GP lengthscale, days | 2.46 | 708.03 |
| Dominant GP noise variance | 5.16 | 125.11 |

The paired ELBO gain was **579.707 +/- 0.311**, while data reconstruction
deteriorated. GP errors here use reconstruction at posterior-mean
hyperparameters, consistently before and after training.

Thus, among the tested guides, the original objective strongly favors
collapsed solutions over the supplied data-faithful guide. This is evidence
against merely an unlucky original initialization: even a faithful starting
point is driven toward collapse. It does **not** prove global optimality,
exclude another faithful optimum, or isolate which objective term is
responsible. The downweighted observation likelihood and dynamics evidence
remain a trade-off to investigate without assuming a new GP prior is required.

Fold-in 1 was the empirical ELBO winner, but its lead over the runner-up was
only 0.114 (paired Monte Carlo standard error: 0.100), so that ranking is not resolved.
After freezing this training-only selection, its nominal full-field forecast
error was 1728.35% (reduced-coordinate: 1003.41%); all 200 integrations were
finite. Forecasts did not influence selection, and this candidate did not
replace either saved benchmark.

The session's `chemo-regression-diagnostics/original_objective_audit/`
preserves `objective_audit.py`, `preregistration.json`, `selection.json`,
`conclusion.json`, fixed keys, source hashes, all guide parameters, ELBO
samples, and GP diagnostics. The data fingerprint is
`7c2b80ca3ebec9f33ace8d3d1c6b148a77da8f2f8906669d256e995edd1b2315`.

## Recorded experimental results

Full-field relative L2 forecast errors on days 70-110:

| Noise | Dose | Experimental input-aware Bayesian | Unchanged neural baseline |
|---|---|---|---|
| 1% | 0.8x | 6.84% | 15.63% |
| 1% | 1x | 12.28% | 42.43% |
| 1% | 1.2x | 41.10% | 126.32% |
| 3% | 0.8x | 8.98% | 15.13% |
| 3% | 1x | 12.07% | 43.35% |
| 3% | 1.2x | 47.56% | 126.58% |
| 5% | 0.8x | 12.56% | 14.73% |
| 5% | 1x | 17.69% | 42.47% |
| 5% | 1.2x | 82.85% | 125.79% |

These demonstrate the experimental variant's performance on this benchmark,
not recovery of the unchanged original statistical model. Higher-dose errors
remain large. At 1.2x, reduced-coordinate coverage is about 67-72% for nominal
90% intervals; volume coverage is a separate diagnostic. This is not evidence
of universal calibration, clinical validity, or performance on other patients.

## Reproduction and preserved artifacts

From `experiments/tumor` in the `prob_rom` environment:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Original model on the matched data; also the default.
python 06_compare_chemo.py --method report --bayes-profile historical

# Explicitly opt into the experimental model.
python 06_compare_chemo.py --method bayes --bayes-profile input-aware
python 06_compare_chemo.py --method report --bayes-profile input-aware
```

Original matched results:
`results/chemo_matched_80_5_70_110_v1/`

Experimental results:
`results/chemo_matched_80_5_70_110_v1_input_aware_v1/`

The experimental directory includes CSV/JSON summaries, per-dose artifacts,
figures, GP hyperparameter draws, and diagnostic audit summaries. Incompatible
profile checkpoints are rejected rather than reused. The `historical` profile
does not switch the acquisition protocol back to 5-60/90.

Implementation references: `core/weakform_opinf/features.py`, `gp.py`,
`evidence.py`, `model.py`, and `pipeline.py`; profile selection and data handling
are in `04_unified_chemo.py` and `chemo_protocol.py`.
