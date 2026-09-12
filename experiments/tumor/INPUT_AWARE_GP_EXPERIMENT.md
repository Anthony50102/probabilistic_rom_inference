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

### Bounded observation-design robustness audit

Eight configurations were registered before running: the original observation
design, small time perturbations, and two replacement uniform designs (seeds
43 and 44), each on days 5-60 and 5-70. Every fit used the original model,
12,000 SVI updates, 500 guide draws, and the same initialization and optimizer
random stream. The original day-60 basis and day-110 input/prediction table
were frozen across all eight cases.

Within each span, the original reduced measurement residual at each column
was held fixed while the clean trajectory was evaluated at the new times.
This isolates placement from a fresh noise realization or basis refit; it is
a conditional reduced-data diagnostic, not a new voxel-noise benchmark.
Empirical prior locations were recomputed by the unchanged original rules.
The two control fits reproduced the corresponding earlier fixed-basis
ablations. No candidate was selected as a winner.

Time perturbations preserved both endpoints and sample ordering. Their largest
actual displacement was 0.2155 days, approximately 5.2 hours.

| Training span | Observation design | GP mode 1 observation error | GP mode 2 observation error | Field forecast error, days 70-110 |
|---|---|---|---|---|
| 5-60 | Original control | 1.88% | 1.62% | 39.55% |
| 5-60 | At most 5.2-hour perturbations | 3.01% | 97.41% | 45.63% |
| 5-60 | Uniform replacement, seed 43 | 96.75% | 74.84% | 282.52% |
| 5-60 | Uniform replacement, seed 44 | 97.06% | 90.66% | 346.96% |
| 5-70 | Original control, frozen old basis | 89.47% | 3.19% | 1209.85% |
| 5-70 | At most 5.2-hour perturbations | 89.16% | 3.87% | 1945.40% |
| 5-70 | Uniform replacement, seed 43 | 98.32% | 96.23% | 334.98% |
| 5-70 | Uniform replacement, seed 44 | 97.92% | 97.77% | 336.47% |

Forecasts here are **40-draw screening estimates**, with the original IC
uncertainty rule, not the full 200-draw benchmark. All 320 requested
integrations were finite. Every forecast uses the same day-70-to-110 scoring
interval; the old control is therefore not the earlier 9.59% reduced-coordinate
forecast through day 90. Finite-ensemble uncertainty is substantial:
200 bootstrap resamples give a 90% Monte Carlo range of 32.21-54.02% for the
old control's field error and 38.58-71.01% for its perturbed counterpart.
That small forecast difference is not resolved; the large GP mode-2 training
collapse is the clearer perturbation result. Bootstrap ranges describe
finite-ensemble error estimation, not predictive coverage.

The important result is that **failure occurs without changing either
endpoint**. Both replacement designs on the original 5-60 span collapse the
dominant mode, and small perturbations collapse its second mode while leaving
the dominant mode relatively faithful. In the latter case, full-field
training error rises from 2.83% to 14.97% in the screening forecasts.
Thus the known-good run is demonstrably fragile under this controlled
observation-design audit. These few coupled cases do not estimate a population
failure probability or establish that every configuration must fail.

The session's `chemo-regression-diagnostics/observation_robustness/` contains
the preregistration, frozen reduced datasets, all four-mode GP diagnostics,
guide parameters, losses, forecast draws, CSV/JSON summaries, bootstrap ranges,
and `jitter_gp_reconstruction.png`. The companion scripts are
`robustness_audit.py` and `summarize_robustness.py`. No inference implementation,
default setting, or saved benchmark was replaced.

### Which objective component rewards collapse?

A separate, no-refit audit decomposed the warm guide's before/after ELBO on the
same matched data using 64 common independent keys. All AutoNormal sample
densities, including transformations/Jacobians, were included. Component sums
agreed with native same-key `Trace_ELBO` to less than 0.0001 in float32.
These keys differ from the earlier 256-key evaluation, so the Monte Carlo
estimates are not expected to be identical.

| Component | After-minus-before ELBO contribution | Paired Monte Carlo standard error |
|---|---|---|
| Observation marginal likelihood, weighted 0.1 | -19.86 | 0.22 |
| Marginal operator/dynamics evidence | +599.56 | 0.57 |
| Hyperparameter priors | -20.44 | 0.20 |
| Negative guide log density | +21.36 | 0.08 |
| Total ELBO | +580.61 | 0.56 |

The unweighted observation term loses 198.63. Thus downweighting observations
reduces the penalty for losing data fidelity, but the dominant positive reward
comes from the dynamics evidence, not the priors or guide density.

At arithmetic posterior-mean hyperparameters (a plug-in diagnostic, **not an
ELBO decomposition**), the derivative-covariance log-determinant contribution
improves by 423.39, while the marginal quadratic contribution improves by
144.87. Flattening the GP makes its inferred derivatives small and certain,
and the constraint density rewards that concentration. These are
GP-conditioned constraint summaries, not additional independent observations.
This identifies an incentive in the implemented objective; it does not mean
Gaussian normalization terms can simply be deleted as a principled fix.

The same concern is present on the old observations. With the old day-60
data/basis and the fixed input110 table, a moment-matched known-good guide has
ELBO -262.78 +/- 0.28 and dominant GP observation error 1.88%. A collapsed guide
transferred without refitting from the matched-data audit has ELBO
+89.98 +/- 4.90 and observation error 96.13%. Its paired advantage is
352.76 +/- 4.80. Both guides were scored against exactly the same old-data
target. The transferred guide is a valid candidate distribution, not a claim
to be that target's optimized posterior.

This supports an explanation in terms of attraction to different variational
solutions: the successful observation design reaches a data-faithful solution,
but the objective also gives higher scores to badly collapsed candidates.
The new day-70 observations are not required for this incentive to exist.
The old-data comparison uses input110, not the literal historical input90
table; the fixed-basis control separately reproduces the prior input110 replay.
No claim of global optimality, a population failure rate, or a proven
replacement likelihood follows from these finite comparisons.

The session's `chemo-regression-diagnostics/objective_terms_audit/` retains
the scripts, per-key component arrays, source hashes, plug-in terms, and
`old_target/` comparison. The next methodological issue is the construction
and weighting of GP-conditioned dynamics evidence, including its covariance
approximations, while preserving the original GP family.

### Breadth-first screen of objective alternatives

All five proposed directions were screened before deeper tuning: calibrated
measurement likelihood, explicit GP fidelity constraints, GP-first inference
without dynamics feedback, shared latent trajectories, and direct ODE rollout
likelihood. Two original-objective controls were also run.

Both datasets have 80 observations on days 5-70, the same frozen canonical
matched POD basis, and the same known input through day 110. Dataset A is the
original matched 1% data. Dataset B replaces the observation times using
uniform seed 43 while retaining the same reduced measurement residual at each
column. This is a paired observation-design diagnostic, not an independent
noise/patient sample.

Every reported score below is an **individual ODE point rollout**, starting
from the exact observed initial coordinate. Predictions were independently
recomputed with input-onset-split DOP853 integration. These are not posterior
ensemble medians, calibrated uncertainty results, dose-generalization results,
or comparisons against the saved neural ensemble.

| Approach | A field fit | A field forecast | B field fit | B field forecast |
|---|---|---|---|---|
| Original-objective control | 75.40% | 306.20% | 395.37% | 9606.57% |
| QR/float64/roundoff-nugget control | 294.64% | 16023.74% | 355.45% | 9243.13% |
| Noise-calibrated likelihood | 29.38% | 56.75% | 3.32% | 36.00% |
| Explicit GP fidelity constraint | 29.18% | 55.80% | 2.94% | 36.75% |
| GP-first, feedback cut | 12.92% | 31.71% | 6.50% | 15.31% |
| Shared latent trajectory prototype | 9.91% | 26.10% | 9.41% | 27.72% |
| Direct ODE rollout likelihood | 0.60% | 20.13% | 0.53% | 16.32% |

Fit and forecast intervals are days 5-70 and 70-110. The original-objective
controls and calibrated likelihood used 4,000 SVI updates, not the full
12,000-step benchmark; their operator points are conditional means at mean
hyperparameters. Control magnitudes therefore must not replace or be confused
with the earlier 200-draw benchmark statistics.

#### What was actually changed

The three GP-side alternatives use the original RBF and cABN structures, with
float64, QR, a roundoff-scale GP nugget, full observation likelihood weight,
and declared projected-noise prior medians. Noise remains inferred.

- **Noise calibration:** the original approximate GP/dynamics inference with
  those settings; all 4,000 SVI losses were finite, but convergence was not
  certified.
- **Fidelity constraint:** constrained MAP, at most 200 SLSQP iterations,
  requiring every mode's GP conditional-mean observation RMS to be at most
  three measurement-noise standard deviations. The exact first column is
  excluded from that RMS. Both optimizations converged and were feasible.
  The constraint was inactive on A and active on B. This constrains a GP
  conditional mean, not every draw from a fully Bayesian posterior.
- **Feedback cut:** data-only GP MAP first, then hierarchical operator-scale
  MAP with GP hyperparameters frozen, followed by the analytic conditional
  operator posterior mean. Both stages converged. This is a plug-in cut
  prototype, without GP hyperparameter uncertainty propagation.

The **shared-latent prototype** jointly adjusts 40 whitened GP states per
mode and profiles the conditional Gaussian operator MAP. It includes the
full state/derivative cross-covariance and conditions derivatives on the
shared latent states. However, its GP hyperparameters are fixed at data-only
MAP estimates with different log-hyperparameter priors; fixed, training-scaled
zero-mean operator priors replace the original hierarchy; discrepancy SD is
10% of data-only GP derivative RMS; and weak constraints are omitted.
It is therefore a feasibility prototype, not a one-change ablation.
The latent MAP converged on both cases; two preliminary GP fits on A ended
with abnormal line-search termination and remain a caveat.

The **direct-rollout prototype** fits all 40 original cABN coefficients to
observations using the declared-noise likelihood and original zero-mean
hierarchical operator prior. It uses no GP or extra quadratic/input-trend
terms. Zero and training-only spline-derivative least-squares initializations
were each allowed 200 L-BFGS iterations; the latter won on training posterior
score for both datasets. Those estimates hit the iteration limit and are
**not converged**. Initializing with least squares did not change the prior
mean. Broad bounds on log operator scales were numerical safeguards; neither
selected solution hit them. This is joint MAP, without analytic operator
marginalization or posterior calibration.

#### What the screen establishes, and what it does not

Protecting GP observation fidelity is effective at preventing collapse but
is not sufficient for accurate ODE rollouts. Maximum per-mode GP RMS in noise
units was 0.94/4.42 for calibration, 0.84/3.00 for the constraint, and
1.26/1.93 for the cut on A/B. For example, the constrained A GP fits within
one noise standard deviation while its ODE has about 29% field training error.

The shared-latent GP also remains faithful at measured times after the latent
update: its maximum RMS is 0.82/0.94 noise standard deviations. Those are
measurement residuals, not the much larger latent-state movements at
unobserved collocation points.

After all directions had results, a cheap **no-refit** check isolated latent
movement within that prototype: retaining its identical GP hyperparameters,
priors, slack and collocation, but freezing states at their GP means, gives
field fit/forecast errors 19.00%/33.12% on A and 180.24%/7416.91% on B.
Allowing latent updates materially helps under those same prototype
assumptions; this does not establish that its other changes are necessary.

Direct rollout fitting produces the most consistently accurate training
trajectories in this screen. It demonstrates that the unchanged cABN family
can fit these observations much better than the original GP-derived operator
estimates. Forecast error remains 16-20%, and observation RMS remains
4-13 measurement-noise standard deviations, so model discrepancy and
optimization limits still matter. No claim of solved calibration or a
globally optimal operator posterior is supported.

The evidence prioritizes direct rollout likelihood as an accuracy reference,
and feedback-cut inference as the simpler candidate retaining conditional
Gaussian operator inference. Joint latent inference merits a more controlled
comparison of priors, slack and covariance assumptions before adoption.
Neither the fidelity constraint nor noise calibration alone resolved the
original rollout problem.

All per-method results, convergence records, operators, datasets, numerical
checks and independent predictions are preserved under the session's
`chemo-regression-diagnostics/direction_screen/`, including
`final_report.json` and `final_report.csv`. Scripts include `screen_common.py`,
`screen_controls.py`, `direction_screen/gp_screens.py`,
`joint_latent_screen.py`, `direct_rollout_screen.py`,
`screen_score_verified.py`, and the report/validation helpers. No production
model, default setting, or benchmark artifact was changed.

### Shared-latent regression spots and remaining rollout gap

Two fast existing experiments were evaluated using identical frozen data for
the original method and the generalized shared-latent prototype. Original
baselines retained 8,000 SVI steps; latent fits used the prototype's 40-point,
200-iteration MAP budget. All scores here are point rollouts, not posterior
ensemble comparisons.

| Experiment | Original reduced fit / forecast | Shared-latent reduced fit / forecast |
|---|---|---|
| Diffusion-reaction, rank 3, 60 observations, 3% noise | 0.89% / 7.39% | 0.30% / 10.36% |
| Euler, rank 6, 55 observations, 3% noise | 4.54% / 16.63% | Diverged during the training interval |

The diffusion-reaction experiment lives under `experiments/burgers_2d`.
Its fixed-latent control, using the same prototype assumptions but no latent
movement, scored 0.32% / 9.36%. Thus some degradation is present before latent
updates, with an additional forecast degradation after them.

Euler exposed a genuine portability issue: the prototype's absolute GP
lengthscale floor of 0.01 was inappropriate for training on 0-0.08.
A single replay changed only the numerical bounds to the same
span-relative range used for chemotherapy, [0.0000123077, 8].
All GP bound hits disappeared; learned lengthscales were 0.00370-0.00437,
with observation RMS 0.56-0.77 measurement-noise standard deviations.
Nevertheless, fixed-latent and shared-latent rollouts still reached the
divergence cap at t=0.00782 and t=0.00513 respectively, before training ends.
The latent MAP converged; one preliminary GP fit reported abnormal
termination. Diffusion-reaction had no lengthscale floor hits.

These are **not a clean non-regression pass**. The prototype still differs
from the original in GP fitting, operator scales, discrepancy assumptions,
covariance treatment and omission of weak constraints. Euler failure before
latent movement rules out attributing it solely to latent updates. Correct
compact-quadratic ordering, feature/gradient calculations and Euler's
physical-variable scaling were checked. Production inference was not changed.

#### Where chemotherapy loses accuracy

The continuous GP trajectory was reconstructed conditional on the actual
optimized latent states, with its analytic derivative, rather than by
interpolating the saved collocation values.

| Diagnostic | Dataset A | Dataset B |
|---|---|---|
| Latent trajectory observation error | 0.056% | 0.051% |
| Shared-latent ODE observation error | 27.34% | 25.38% |
| Direct-fit ODE observation error | 0.743% | 0.661% |

On A, the latent trajectory has 22.77% reduced error against training truth
between observations. Approximately 98.43% of its squared training-truth
error lies in the observation gap from day 20.206 to day 23.290.
Thus fitting observations tightly has not recovered a reliable intervening
trajectory.

For the affine, input-driven cABN dynamics, the rollout error relative to the
latent trajectory obeys the exact relation

```text
r(t) = f(x_lat(t), alpha(t); O) - dx_lat(t)/dt
e'(t) = [A + alpha(t) N] e(t) + r(t)
e(5) = observed_initial_state - x_lat(5)
```

Post-fit integration of this decomposition reproduces the baseline error to
about 5.1e-11. The initial-condition component's norm is only 0.127%/0.062%
of the total error norm on A/B; these are norm ratios, not additive variance
fractions. Accumulated derivative defects, including their dynamical
amplification, dominate.

Whitened collocation residual RMS is only 0.26-0.56 across modes, while dense
defects in leading modes reach 74-93% of latent derivative RMS. These use
different normalizations: the point is that small uncertainty-weighted
collocation errors can coexist with large physical defects between nodes.

With saved GP hyperparameters, operator scales and slack held fixed,
80 uniform collocation points gave field fit/forecast 16.28%/33.62% on A
and 10.53%/10.89% on B. Thus forecasting improved on B but training did not,
and all these followups hit the 200-iteration cap. A 55-point input-refined
grid produced severely inaccurate rollouts, with A exceeding the independent
scoring budget. More collocation is not a demonstrated universal fix;
changing grid size also changes the number of physics constraints.

A separate small prior/slack sensitivity check retained the same 40-point
latent method and GP fits. On A, replacing the prototype's scaled operator
prior by fixed SD 5 changed forecast error from 26.10% to 145.79%;
using uniform discrepancy variance 0.035 changed it to 66.76%;
changing both gave 91.20%. On B the corresponding forecasts were
30.46%, 24.42% and 19.56%, versus 27.72% originally, with worse training
fits. Fixed SD 5 is a nominal-scale control, **not** the original inferred
operator-scale hierarchy. These results establish sensitivity, not a
preferred transferable setting.

The regression artifacts are under `chemo-regression-diagnostics/shared_latent_spots/`;
continuous-path, error-decomposition and grid diagnostics are under
`latent_gap_audit/`; prior/slack controls are under `latent_assumption_audit/`.
The prototype is not ready to replace other experiments' defaults.

## Shared-latent weak-only followup

The subsequent strong-plus-weak, optimization and interpolation-uncertainty
investigation is recorded separately in
[Shared latent states: constraints, optimization and interpolation](SHARED_LATENT_CONSTRAINT_STUDY.md).

After the regression and rollout-gap studies, a bounded eight-fit screen
replaced the prototype's strong derivative likelihood with state-only
integration-by-parts weak constraints. No ODE integrations were used during
fitting; integrations below are post-fit evaluation only. These remain
joint-MAP prototypes, not the original operator-marginalized Bayesian model
or a posterior/dose-generalization benchmark.

The two frozen low-noise datasets are the same matched acquisition A and
replacement observation times B (seed 43) used above. Saved data-only GP
hyperparameters, measurement noise, operator prior scales and discrepancy
SDs were held fixed from the 40-state prototype. Latent nodal states retain
their GP prior conditional on measurements; conditional operators are
analytically profiled at their MAP. The weak objective does not use GP
derivative means or derivative covariance.

### Definition and quadrature controls

The initial construction used ten compact polynomial bumps,
`psi = (1-z^2)^6` inside each support, vanishing at support endpoints.
Their radius is `65 * 20/199`, approximately 6.53 days. For trapezoidal
matrices W (weighted psi) and V (weighted psi derivative), the weak target
is `-V X` and the feature matrix is `W F(X, alpha)`. The complete overlap
covariance is `slack_i^2 W W^T`; it is not diagonalized. This is the
pushforward of independent nodal discrepancy, not the original continuous
white-noise-style weak variance. GP state uncertainty is represented by X,
not counted again as an independent weak observation error.

An independent quadrature audit found relative known-input integral errors
of 12.18% and 10.68% at 40 and 80 nodes. Increasing resolution did not
monotonically remove onset-sampling error. Therefore raw failures alone
cannot establish a weakness of weak-form inference.

Two corrected 80-node fits integrated piecewise-linear nodal hat functions
exactly to roundoff: partition at latent knots, input knots and bump support
boundaries, then use eight-point Gauss quadrature. With nodal hats phi,
the matrices are M = integral(psi phi), D = integral(psi' phi),
C = integral(psi), B = integral(psi alpha), and N = integral(psi alpha phi).
The target is `-D X`, the design is `[C, M X, B, N X]`, and discrepancy
covariance is `slack_i^2 M M^T`. Eight- and twelve-point quadrature agree
within 1.78e-15 for the ten-test fits.

This corrects quadrature for a **piecewise-linear nodal path**, not the full
continuous GP path between nodes. In particular, the GP measurement-fit
diagnostic evaluates the conditional GP, not this piecewise-linear path.
Changing node count also changes discrepancy covariance: mean diagonal
approximately halves from 40 to 80 nodes with fixed nodal discrepancy SD.
Thus grid refinement is not purely a numerical-accuracy comparison.

### Results and overdetermination check

Full-field relative L2 percentages; fit is through day 70, forecast is
days 70-110. Strong-only rows are the already completed controls.

| Formulation | Dataset | Field fit | Field forecast |
|---|---|---:|---:|
| Strong-only, 40 states | A | 9.91% | 26.10% |
| Strong-only, 40 states | B | 9.41% | 27.72% |
| Strong-only, 80 states | A | 16.28% | 33.62% |
| Strong-only, 80 states | B | 10.53% | 10.89% |
| Weak-only, raw 40 nodes, 10 tests | A | 51,555.79% | 23,802,206.39% |
| Weak-only, raw 40 nodes, 10 tests | B | 217.70% | 3,110.74% |
| Weak-only, raw 80 nodes, 10 tests | A | 1,766.65% | 49,732.92% |
| Weak-only, raw 80 nodes, 10 tests | B | 13.43% | 62.99% |
| Weak-only, corrected 80 nodes, 10 tests | A | 154.45% | 764.89% |
| Weak-only, corrected 80 nodes, 10 tests | B | 29.81% | 48.25% |
| Weak-only, corrected 80 nodes, 20 tests | A | 1,702.12% | 1,487,312.73% |
| Weak-only, corrected 80 nodes, 20 tests | B | 6.28% | 11.18% |

All six ten-test fits reported gradient-tolerance convergence in 5-24
iterations. Raw whitened weak residuals are only 0.001-0.049 discrepancy SD;
corrected ten-test residuals are 0.004-0.040 SD. There are ten weak equations
per output and ten cABN coefficients, with design rank ten. Near-interpolation
of these moments is therefore a plausible limitation, not evidence of a
faithful ODE trajectory.

The final two fits were one predeclared overdetermination check, not a sweep:
twenty uniformly spaced test centers, unchanged radius and all other
corrected-80 settings. Both reported function-tolerance convergence at
120/149 iterations; gradient infinity norms were 2.61e-4/3.64e-5, above
the requested 1e-5 gradient tolerance. The weak covariance has rank 20,
condition number 39.98 and no added nugget; design rank remains ten.
Whitened residual RMS spans 0.086-0.590 SD. Overdetermination substantially
improves B but catastrophically worsens A. On B its forecast is comparable
to, not better than, the same-grid strong-only control.

Across all eight weak fits, conditional-GP observation errors remain
approximately 0.047-0.057% in reduced coordinates. This is not the earlier
flat-GP failure: a measurement-fitting GP and small weak residuals can still
coexist with an inaccurate or rapidly growing learned ODE. These observations
do not isolate one universal failure mechanism or prove weak forms fail
intrinsically. Correcting quadrature and adding more equations are not a
consistent rescue of this particular finite-grid, fixed-prior prototype.

Saved arrays independently reproduce the weak residuals and field metrics.
Augmented least-squares operator solves agree with the fitting normal
equations to at worst 4.16e-11 relative difference, ruling out that solve's
roundoff as an explanation for these gross errors.

Artifacts are in `chemo-regression-diagnostics/latent_weak_only/`, including
`summary.json`, `corrected_quadrature_summary.json`, `tests20_summary.json`,
and per-case operators, latent states, covariance matrices and predictions.
Implementations are `latent_weak_only_screen.py`, `latent_weak_corrected.py`
and `latent_weak_tests20.py`; the independent audit is
`weak_quadrature_audit.py`. These are in the same preserved session artifact
directory as the earlier screens. No production inference code, defaults,
canonical benchmark outputs or neural checkpoints were changed.

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
