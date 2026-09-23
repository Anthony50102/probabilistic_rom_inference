# Half-exposure chemotherapy: production versus experimental shared-latent models

## Verdict

**Keep the production algorithm for this half-exposure benchmark. Neither of
the two existing experimental recipes matched its accuracy and completion.**

Production retained 5.31-5.51% full-field forecast error on the three
acquisitions. The primary shared-latent prototype produced 38.94%, an unavailable
safety-censored forecast, and 7.71%. Its existing covariance-corrected version
produced 25.70%, 30,922.29%, and 10.66%. The enormous finite error is retained,
not hidden as a missing value or renamed a mathematical blow-up.

This is a comparison of the **current recipes and prescribed budgets**, not
proof that every possible experimental variant must be inferior. Seventeen
of eighteen joint optimizations reached the 2,000-iteration cap. The remaining
run stopped on relative objective change, not the gradient tolerance, and
still generated a safety-censored forecast.

The earlier [cross-experiment survey](CROSS_EXPERIMENT_SHARED_LATENT_SURVEY.md)
still contains an untreated-tumor point advantage for the prototype. This
new result does not rewrite that exception into universal production
dominance.

## Exact matched task

We reused the same half-exposure acquisitions from the
[production versus Neural ODE confirmation](TUMOR_CHEMO_TASK_COMPARISON.md):
seeds 45, 46 and 47; 120 observations with 1% physical-generator noise over
days 5-70; the same archived clean four-mode decoder, observed day-5 IC,
4001-knot forcing table and 400-point prediction grid.

The primary window remains **(70,110]**, all 152 retained forecast queries.
The treatment strength and schedule, observations, noise realization, source
simulation, basis and forecast horizon were not changed. There were no new
production fits, Neural ODE fits, FOM solves or POD fits.

The comparison was frozen before any new data-GP or joint MAP fit.
Experimental initializations and terminal outputs were retained, including
failures and optimizer warnings. All six within-cell training-objective
selections were frozen before any experimental forecast.

## Primary results

Entries are full-field relative RMS percentages, including the affine decoder
shift and unrepresented residual, with equal retained-query weighting.
The production estimator is the saved conditional-operator hyperpoint,
not a posterior predictive mean. NODE is the saved original
training-loss-filtered trajectory median, a secondary reference only.

| Seed | Production | Shared, target-only | Shared, covariance-corrected | NODE, saved |
| --- | --- | --- | --- | --- |
| 45 | 5.39 | 38.94 | 25.70 | 9.90 |
| 46 | 5.51 | U | 30922.29 | 10.04 |
| 47 | 5.31 | 7.71 | 10.66 | 9.93 |

U means an unavailable complete-window score, not zero or a score restricted
to the surviving prefix. The selected target-only seed-46 forecast reached
the unchanged large-state safety threshold at **day 102.302351**.

Production's all-three median is 5.39%; the covariance-corrected prototype's
is 25.70%, with production better in 3/3 matched pairs. The target-only
prototype has **no all-three median**: its two complete forecasts are both
worse than production, and the third is censored. We do not turn that missing
score into a numerical paired win or compute a survivors-only cohort median.

Neither experimental variant meets the existing reasonable-accuracy rule
(median <=15%, every seed <=25%, all three complete). No alternative start
rescued the observed ranking: every one of the 15 complete experimental
endpoints was worse than its corresponding production point; the other three
were safety-censored. This is a post-fit robustness observation, not a new
forecast-selected estimator.

### Finer-reference sensitivity

The same saved predictions were rescored against the previously frozen
0.125-day source; nothing was refitted or reselected.

| Seed | Production, refined | Target-only, refined | Covariance-corrected, refined |
| --- | --- | --- | --- |
| 45 | 5.93 | 37.57 | 24.49 |
| 46 | 5.46 | U | 30570.73 |
| 47 | 5.59 | 7.10 | 9.69 |

The practical verdict is unchanged. The refined-reference all-three medians
are 5.59% for production and 24.49% for the covariance-corrected prototype.

## Which experimental models were tested

The primary recipe is the original combined strong-plus-weak, target-only
block-diagonal shared-latent MAP model used in the cross-experiment survey.
The separately reported control is its existing **full-feature
block-diagonal covariance correction**, including cross-output weak
covariance and the complete half-log-determinant of the operator-dependent
residual covariance. It does not add all strong/weak cross-covariance blocks.
Neither variant was chosen after seeing forecast performance.

Both retained four states, the cABN family (40 operator coefficients), 40
latent/strong nodes, 20 p6 weak tests with radius 0.1 of the training span,
and the same continuous conditional GP path for strong and weak features.
The latent state is profiled exactly; the optimizer acts on the 40 active,
prior-standardized operator coefficients. This is joint MAP, not Gaussian
state marginalization: no state-integral log-determinant was added.

Data-only zero-mean RBF GPs retain fixed noise, unit-SD log-hyperparameter
priors centered at `log(span/20)` and `log(var(y))`, and lengthscale bounds
`span/6500` through `100*span`. The original data-GP optimizer retains
`maxiter=120, ftol=1e-12, gtol=1e-7, maxls=30`.

Operator-prior SD remains
`10 * pooled_data_GP_derivative_RMS / pooled_data_GP_feature_RMS`;
discrepancy SD remains `0.1 * pooled_data_GP_derivative_RMS`. The same
prepared GP, prior and discrepancy are used for both covariance variants
within an acquisition. No prior-strength, noise-estimator or constraint-form
search was performed.

The three starts are the original zero latent perturbation and 0.1-SD
whitened perturbations with seeds 101 and 202. Their target-only conditional
operators initialize both variants identically. L-BFGS-B retains
`maxiter=2000, ftol=1e-12, gtol=1e-6, maxls=40`. The eligible minimum
**own training objective** selects each endpoint. As in the earlier recipe,
finite derivative-qualified capped endpoints remain eligible and visibly
flagged. Objectives are not compared across variants or acquisitions to pick
a winner.

**No ODE solve appears in either fitting objective.** Independent forward
integration is evaluation only.

### Important information and convergence differences

The prototype retains its **privileged synthetic instrument-noise
calibration**: the exact diagonal physical-noise covariance projected
through the fixed compressor, averaged over the 119 noisy observations.
The exact first observation is excluded from that average. The resulting
positive variance is used at all observation times, as in the old recipe.

These already audited diagnostic variances were explicitly authorized as
experimental fitting inputs in this new study. Production did not consume
them. The owned training files contain observations, IC, forcing and these
noise variances, but no clean state trajectory or future state truth.
The calibration itself was derived from the clean training activity mask and
generator scale; only its resulting variances reach this optimizer. The
protocol's no-clean-trajectory wording concerns fitting arrays/targets, not
ground-truth-free calibration; this distinction is archived explicitly.
This is not an equal-information causal ablation. Fixed versus learned GP
hyperparameters/noise, priors, covariance approximations, precision and
constraint grids also differ. Both methods share the same privileged clean
decoder; neither experiment establishes deployment-realistic preprocessing.

There were 18 joint fits: 17 iteration caps and one objective-change
termination. All selected gradient infinity norms remain above `1e-6`;
we do not claim globally optimal or gradient-converged MAP solutions.
The twelve data-GP fits also retained two `ABNORMAL` line-search flags:
seed 45, mode index 0, gradient infinity norm 0.00111739; and seed 47,
mode index 2, 0.00000139081 (indices are zero-based).

## All starts and the actual selection

| Seed | Variant | Start | Selected | Joint termination | Forecast field% |
| --- | --- | --- | --- | --- | --- |
| 45 | Target-only | zero | Yes | Capped | 38.94 |
| 45 | Target-only | random101 | No | Capped | 39.24 |
| 45 | Target-only | random202 | No | Capped | 36.84 |
| 45 | Covariance-corrected | zero | No | Capped | 21.59 |
| 45 | Covariance-corrected | random101 | No | Capped | 25.91 |
| 45 | Covariance-corrected | random202 | Yes | Capped | 25.70 |
| 46 | Target-only | zero | No | Capped | U |
| 46 | Target-only | random101 | Yes | Objective change | U |
| 46 | Target-only | random202 | No | Capped | U |
| 46 | Covariance-corrected | zero | No | Capped | 32342.96 |
| 46 | Covariance-corrected | random101 | No | Capped | 31410.09 |
| 46 | Covariance-corrected | random202 | Yes | Capped | 30922.29 |
| 47 | Target-only | zero | No | Capped | 8.65 |
| 47 | Target-only | random101 | Yes | Capped | 7.71 |
| 47 | Target-only | random202 | No | Capped | 7.72 |
| 47 | Covariance-corrected | zero | Yes | Capped | 10.66 |
| 47 | Covariance-corrected | random101 | No | Capped | 10.41 |
| 47 | Covariance-corrected | random202 | No | Capped | 10.54 |

The covariance-corrected seed-45 zero start has a lower forecast error than
the selected start, but was not substituted. Selection remained based on
training objective. No new starts, extra iterations, new noise setting or
best-checkpoint selection were introduced after viewing forecasts.

## A good latent curve is not a good ODE rollout

The following are errors against the same clean training reference on the
248 retained training queries, not the noisy observation objective.

| Seed | Variant | Selected GP training field% | Selected ODE training field% |
| --- | --- | --- | --- |
| 45 | Target-only | 1.61 | 2.93 |
| 45 | Covariance-corrected | 1.50 | 2.68 |
| 46 | Target-only | 1.77 | 46328.84 |
| 46 | Covariance-corrected | 1.83 | 927.89 |
| 47 | Target-only | 1.33 | 2.11 |
| 47 | Covariance-corrected | 1.34 | 2.43 |

Seed 46 is an especially strong warning: its latent reconstruction looks
reasonable, but its learned ODE is already grossly inaccurate during training.
The target-only fit that reported objective-change convergence is also the
one whose forward forecast is censored. Thus a solver success flag, a good GP
curve or a lower fitting objective would not establish useful dynamics.
This does not isolate a unique cause among the differing inference recipes.

## Decoded physical fields

These diagnostics concern normalized cellularity, not signs of centered POD
coordinates. Material-negative values are below `-1e-6`; above-capacity values
exceed `1+1e-6`. Fractions use available voxel-query pairs, and unavailable
queries remain explicit. These are not clipping or additional selection rules.

| Seed | Variant | Minimum | Maximum | Material-negative pairs% | Above-capacity pairs% | Available forecast queries |
| --- | --- | --- | --- | --- | --- | --- |
| 45 | Target-only | -0.00000 | 0.72576 | 0.0000 | 0.0000 | 152/152 |
| 45 | Covariance-corrected | -0.00000 | 0.61413 | 0.0000 | 0.0000 | 152/152 |
| 46 | Target-only | -79996.14332 | 43.41038 | 10.2049 | 1.1780 | 122/152 |
| 46 | Covariance-corrected | -305.34835 | 0.09861 | 10.6099 | 0.0000 | 152/152 |
| 47 | Target-only | -0.00000 | 0.50734 | 0.0000 | 0.0000 | 152/152 |
| 47 | Covariance-corrected | -0.00000 | 0.53049 | 0.0000 | 0.0000 | 152/152 |

The seed-46 violations are physically meaningful, not merely negative reduced
coordinates. The displayed negative zeros elsewhere are small roundoff-scale
values. Production's previously checked points have no material negative or
above-capacity values on the corresponding query grids.

No new experimental posterior ensemble or calibration claim was constructed.
Multistart spread is not Bayesian uncertainty, and a covariance-frozen
conditional operator root is not the joint posterior. The production
uncertainty limitations in the prior report remain unchanged.

## Numerical checks, accounting and preservation

The frozen recipes use the original spectral RBF implementation with 8,192
maximum features and Gauss-8 integration verified against Gauss-12. Every
input, latent and test-support knot is retained. The existing approved
100,000-point quadrature allocation ceiling was reused; this changes no
quadrature rule, node, GP lengthscale, prior, noise or tolerance. Initial and
terminal directional derivative checks used the original two directions and
three step sizes.

Every one of the 18 endpoints was independently integrated with the same
NumPy float64 DOP853 evaluator used for production and checked against Radau.
**All 18 numerical comparisons passed**, including matched censoring and
query availability. The safety policy remains 1e6 times the frozen training
RMS, 200,000 RHS calls and 120 seconds. A safety censor is not proof of
mathematical finite-time blow-up.

A separate audit reconstructed the prepared GP and original kernel algebra
without rerunning an optimizer or ODE. It verified input-array identity,
paired starts, prior/slack formulas, all terminal objectives/gradients/operators,
the whitened latent-state optimum and covariance roots, every saved checkpoint
prefix, all original/refined forecast metrics and all saved numerical pairs.
The state-profile objective was independently recomposed from its quadratics
and residual covariance determinant.

The new work comprises three preparations, twelve data-GP fits, eighteen
joint MAP starts, six training-objective selections, eighteen primary
rollouts and eighteen Radau checks. There is one anatomy and three acquisition
seeds, not three patients or a population-superiority test. No production,
NODE, source simulation or basis was refitted.

Recovery commit `3600015bb1397cdfdd03b7ca9d5b2a05f8346eb3` is retained as
`backup/half-exposure-shared-before-comparison-3600015`. Only this new research
record is added to the repository. Production defaults, previous reports,
canonical caches and manuscript remain unchanged; nothing is pushed or
promoted to production.

Artifact root:
`/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics/half_exposure_shared_comparison`.

| Evidence | SHA256 |
| --- | --- |
| Frozen protocol | `204fd98a49433469f8b8bd6f22b4f324d1b9050800836804c7122d2697f48cde` |
| Training release | `5b5c9bcc8f740fdce5feb5397250111f075482b668fa59b93964a0793d992eaa` |
| Independent fit/metric audit | `0ebcb21f8d13350d2d33265999965cb9498051ae2b065a85cd230c327b7ec9da` |
| Prior production/NODE study release | `383fa5855380a174d1b92f2f36fc4f8f1c0b41e758a5bb6edb2e840b7403ab0a` |

Machine-readable results retain every start and selected endpoint under
`evaluation/`, `comparison/summary.json` and `independent_audit/summary.json`.
The numerical dataset is persistent local session data, not a portable dataset
bundled with the Git research record.
