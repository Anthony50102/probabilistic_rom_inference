# Cross-experiment shared-latent landscape

## Main finding

**Chemotherapy is an outlier for the existing Bayesian method, but the
experimental shared-latent recipe also has substantial non-chemotherapy
regressions. More observations help interpolation, not necessarily the
learned ODE.** It would be premature to specialize or promote the general
method on the strength of the earlier chemotherapy successes.

This survey deliberately paused treatment-specific restrictions and
input-aware GP changes. It covers all five active experiment adapters,
with one base acquisition per problem and a nested double-observation
control. It is an exploratory landscape, not a repeated-seed reliability
study or a controlled ablation of every difference between the methods.

Recovery commit `013f8da` was preserved as
`backup/cross-experiment-before-survey-013f8da` before this work.
Production code, inference defaults, matched-80 chemotherapy guards and
canonical benchmark artifacts were not changed. No ODE solve was added
to an inference objective.

### Main forecast comparison

Entries are **full-field relative forecast errors in percent**. Lower is
better. The native column is the declared representative conditional
operator, not a posterior predictive mean. Shared entries use their own
training-objective-selected strong-plus-weak, target-only block-diagonal
endpoint.

| Problem | Native, base | Native, more observations | Shared, base | Shared, more observations |
| --- | ---: | ---: | ---: | ---: |
| Cubic heat, all five training forcings | 1.976 | 1.918 | 10.072 | C, one of five targets |
| Diffusion-reaction | 8.910 | 7.425 | C | C |
| Euler | 1.351 | 1.271 | C | C |
| Autonomous tumor | 6.253 | 4.696 | 5.680 | 4.630 |
| Chemotherapy tumor | 324.915 | 618.208 | C | 71.062 |

**C** means the ODE reached the common large-state safety threshold:
`max(abs(q_i) / base_training_RMS_i) = 1e6`. Its full-horizon error is
unavailable, not zero, and it was not omitted from a pooled score. This is
not a proof of finite-time mathematical blow-up. Conversely, numerical
completion does not establish accuracy or physical plausibility.

Heat's separate held-out forcing remains visible: native forecast error
is 1.785 -> 1.656%, versus shared 2.524 -> 2.513%. A reasonable held-out
result does not erase the failed forecast of a training-forcing target.

These are not interchangeable with reduced-coordinate errors. For example,
native Euler has 16.606 -> 8.637% reduced forecast error despite its
1.351 -> 1.271% physical error. Physical pressure units dominate the total;
velocity, pressure and inverse-density scores are all retained separately.

## Frozen experiment matrix

| Case | Base -> more observations | Injected noise | Modes / operators | Training window | Prediction window |
| --- | --- | ---: | --- | --- | --- |
| Heat | 80 -> 160 per training forcing | 5% | 5 / cAHBN, two inputs | 0-1 | 0-2 |
| Diffusion-reaction | 60 -> 120 | 5% | 3 / cAH | 0-1 | 0-3 |
| Euler | 55 -> 110 | 3% | 6 / cAH | 0-0.08 | 0-0.15 |
| Tumor | 40 -> 80 | 1% | 4 / cA | days 5-60 | days 5-90 |
| Chemotherapy | 40 -> 80 | 1% | 4 / cABN, one input | days 5-70 | days 5-110 |

The `burgers_2d` adapter actually runs diffusion-reaction, not a Burgers
advection equation. Heat's historical "IC" labels describe five forcing
parameter pairs with a common physical initial condition:
`(-2,0), (-1,-2), (0,1), (1,-1), (2,2)`. The held-out pair is `(1.5,0.5)`.
All five training trajectories share one inferred heat operator.

Dense/high-noise regimes were chosen for the cheaper examples and sparse
regimes for Euler and the large 3D examples. Some requested combinations
are not active presets: heat's sparse-high count was increased from 20
to 80, diffusion-reaction's dense-medium noise from 3% to 5%, and each
tumor count reduced from 80 to 40. These were predeclared study-only
overrides, not adapter changes or reuse of incompatible historical caches.

Every original observation time and noisy reduced column is embedded
exactly in the more-data cohort. Additional interior times use seed
`1042 + trajectory_index`; independent added noise uses
`2042 + trajectory_index`. Noise scales and masks that depend on signal
range stay fixed from base. The first added point is not made noiseless.
The base seed is 42.

Basis, lifting, centering, physical scaling, input functions, observed ICs,
training/prediction windows and 400-point scoring grids are fixed across
methods and density controls. The new observations do not improve the
basis or add new training forcings/initial conditions. PDE data generation
starts at the original initial time, not at the first added sample.

The tumor basis uses noisy base snapshots; chemotherapy retains its
clean-base-training basis convention. The new sparse chemotherapy basis
and acquisition are not the old matched-80 A/B fixtures. Consequently,
the previous B80 6.2% covariance-correction result is not a reusable
benchmark for this cohort.

## Methods and controls

### Native Bayesian reference

The central model, spectrum/cadence-dependent priors, hierarchical
zero-mean operator prior, normal-equation backend, float32 arithmetic,
ClippedAdam and configured learning rates were retained.

| Case | Configured SVI steps | Learning rate | gamma2 variance | Derivative / weak weights | GP MLL weight | sigma_O |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Heat | 10,000 | 0.003 | 0.5 | 1 / 2 | 0.1 | 0.5 |
| Diffusion-reaction | 8,000 | 0.003 | 10 | 1 / 1 | 1 | 10 |
| Euler | 8,000 | 0.005 | 10 | 1 / 1 | 1 | 30 |
| Tumor | 12,000 | 0.003 | 0.035 | 1 / 8 | 0.1 | 5 |
| Chemotherapy | 12,000 | 0.003 | 0.035 | 1 / 8 | 0.1 | 5 |

Both densities received their configured fit. Each base SVI state was
also continued for half its original budget, retaining optimizer moments,
counter and RNG state. The configured-step result remains primary.
This gives 15 stages and 125,000 recorded SVI updates.

Each stage exports 500 hyperparameter/operator draws. Its point operator
is the conditional mean at the arithmetic means of the positive GP and
block-scale draws. Ensemble context uses the same 32 predeclared indices
from 0 through 499, with native O/IC pairing and every target retained.
There is no forecast filtering or calibrated-coverage claim.

Native more-data priors and hyperparameters follow the unchanged
data-dependent production rules; they are not frozen numerically.

### Experimental shared-latent recipe

This is a **fixed-hyperparameter MAP diagnostic**, not the production
hierarchical Bayesian model expressed in different notation.

Data-only zero-mean RBF GPs use positive, training-only synthetic
instrument calibration. Log-lengthscale and log-variance have unit-SD
Gaussian priors centered at `log(span/20)` and `log(Var(y))`.
Lengthscale bounds are `span/6500` through `100*span`, avoiding the
earlier nonportable absolute lower bound.

There are 40 latent nodes per training trajectory and 20 p6 weak bumps
of radius 0.1 times the training span. Strong and weak means use the
same continuous conditional GP path; weak quadratic features are
integrated as products of that path, not products of averaged states.
Nodal hats are used for discrepancy integration, not state interpolation.

Operator-prior SD is `10 * pooled_GP_derivative_RMS / pooled_feature_RMS`;
discrepancy SD is `0.1 * pooled_GP_derivative_RMS`. Pooling includes all
five heat training trajectories. Base GP hyperparameters, calibration,
priors, discrepancy and latent/test grids are frozen for more data.

Base fits compare strong-only, weak-only and combined block-diagonal
constraints. More data uses combined constraints. Three prescribed starts
are zero and 0.1-SD whitened-state perturbations with seeds 101 and 202.
L-BFGS-B uses maxiter 2000, ftol 1e-12 and gtol 1e-6. Selection is solely
within each form's own training objective, never across forms by forecast.

Quadratic cases profile Gaussian O at fixed states and optimize nonlinear
states. Both affine cases use the same exact-state profile and active
prior-standardized O coordinates for target-only and full-feature fits.
Paired affine starts are identical and derived from the prescribed states.
The full correction retains cross-output weak covariance and
`0.5 * logdet R(O)`, not a state-marginalization log determinant.
It is non-Gaussian in O; covariance-frozen operator roots are not its
conditional posterior.

There are 60 target-only and 12 affine full-feature fits: 72 total.
Forty fixed-curve controls fit O to the data-only GP or a training-clean
reference spline with the same prior/covariance recipe. Eight additional
strict training-source-spline controls address reference interpolation.
No oracle is a candidate or initialization.

Native/shared comparisons also differ in GP feedback, prior learning,
discrepancy scales, covariance approximations and physics grids. Native
evaluation grids are 100/200 points with diagonal covariance blocks;
the shared recipe uses dense covariance within its strong/weak blocks.
Cross-recipe forecast differences do not isolate one causal change.

## Strong, weak and full-feature outcomes

The table below gives base-cohort **physical forecast percentages** for
the own-objective-selected endpoint. Heat pools all five training forcings.

| Case | Strong only | Weak only | Combined | Full-feature combined |
| --- | ---: | ---: | ---: | ---: |
| Heat | 8.663 | C, two of five | 10.072 | Not fitted for quadratic features |
| Diffusion-reaction | C | C | C | Not fitted for quadratic features |
| Euler | C | 4.563 | C | Not fitted for quadratic features |
| Tumor | 5.003 | C | 5.680 | 70.980 |
| Chemotherapy | 772.836 | 1.075e6 | C | 1.860e7 |

Euler weak-only is not a convincing success: its reduced training/forecast
errors are 118.63/170.71%. Its physical forecast errors are velocity
2.716%, pressure 4.563% and inverse density 9.618%.

For more data, the affine full correction gives tumor 3.791% versus
target-only 4.630%, and chemotherapy 44.885% versus 71.062%.
Thus the correction helps these denser cases but strongly harms sparse
autonomous tumor. It is not a generally beneficial chemotherapy fix.

Censor timing matters. Combined diffusion base and both Euler cohorts
cross the threshold within training. Diffusion more completes training
at 1.304% physical error but is censored in forecasting. All heat-more
training targets complete, but one forecast does not. Sparse chemotherapy
combined already has approximately 677,118% physical training error before
its forecast censor. Its full-feature counterpart is finite but grossly
inaccurate; completion is not a competing success.

All 72 endpoints, 40 original controls and eight corrected controls were
scored, totaling 220 target trajectories: 158 completed and 62 censored.
No primary shared/native RHS-budget failure required a solver rescue.
All 30 native point and 960 native draw-target trajectories completed.

## What additional observations change

The following are **training-window reduced GP-curve errors**, not the
physical ODE forecast errors above. Heat is pooled over its five GPs.
The tumor cases use the current-basis source-knot evaluation reference.

| Case | Data-only GP, base -> more | Joint combined latent curve, base -> more |
| --- | --- | --- |
| Heat | 0.984 -> 0.721% | 0.951 -> 0.693% |
| Diffusion-reaction | 0.915 -> 0.435% | 0.852 -> 0.407% |
| Euler | 3.173 -> 2.107% | 3.050 -> 2.068% |
| Tumor | 1.833 -> 1.461% | 1.941 -> 1.575% |
| Chemotherapy | 15.133 -> 5.265% | 18.236 -> 5.155% |

More time samples improve every data-only GP curve, even with base
hyperparameters frozen. This interpolation benefit is not specific to
dosing. But improved curves
do not guarantee improved ODEs: heat more loses a forecast, Euler remains
unusable, and diffusion more still fails outside its training window.

Native point forecasts improve in four cases. Chemotherapy is mixed even
within the native posterior context: its physical forecast 5/50/95%
quantiles are approximately 268.0/620.9/15,620.5% at base, versus
46.60/298.0/38,759.6% with more data. The median improves while the
representative point and upper quantile worsen. These 32-draw summaries
are descriptive, not a precise population-tail or calibration result.

This density control does not test new initial conditions, new treatment
strengths, longer training windows, better POD bases or richer excitation.
The shared physics grid and hyperparameters also stay fixed. Failure here
does not show that all forms of additional data would be ineffective.

## Fixed curves: interpolation is not the only issue

These combined-form controls retain observed ICs and the same O-prior and
constraint weights. The tumor clean controls below use the corrected
training-source spline; other cases use their frozen clean reference.

| Case | Data-GP curve O, base -> more | Clean training-curve O, base -> more |
| --- | --- | --- |
| Heat, training forcings | C -> C | 2.099 -> 2.099% |
| Diffusion-reaction | C -> 13.008% | 39.257 -> 39.254% |
| Euler | C -> C | C -> C |
| Tumor | 5.838 -> 4.733% | 3.382 -> 3.377% |
| Chemotherapy | 81.705 -> 312.691% | 14.860 -> 14.957% |

Heat's clean-curve held-out forecasts are approximately 1.556% in both
cohorts. Joint adjustment helps heat relative to the GP-cut failure, but
does not reach this clean-curve result. It also helps denser chemotherapy
relative to its 312.7% GP-cut forecast, while harming the sparse case.
There is no single "always cut feedback" or "always adjust the GP" answer.

Euler's clean-curve control still fails, and diffusion's is worse than
its native reference. GP interpolation error alone cannot explain those
outcomes. These are still regularized fits with a fixed operator family,
grid and covariance recipe, not proofs that no stable ROM can represent
the dynamics.

A separately declared, privileged clean-IC-only diffusion control keeps
the selected combined O fixed. Base then completes at 13.378% forecast,
but more remains censored. Initial measurement error contributes to the
base failure, without being a universal explanation. Primary observed-IC
scores were not replaced.

## Optimization, information and error amplification

No shared fit reaches 2000 iterations. Sixty-nine stop on relative
objective reduction and three satisfy the actual 1e-6 gradient criterion.
Terminal gradient infinity norms span 4.96e-9 to 0.0393; the largest
within-group start-objective spread is 3.56e-6. Independent Gaussian
conditional profiles agree to about 4.55e-13 in cost.

The 42 data-only GP hyperparameter fits have separate termination records:
40 report success, while autonomous tumor modes 1 and 2 report
`ABNORMAL` line-search termination at gradients 1.85e-5 and 1.12e-5.
Those finite endpoints remain explicitly flagged and frozen for every
tumor comparison; they were not silently relabeled as converged or refit
using downstream outcomes. No GP fit reaches its 120-iteration cap.

Native half-budget continuations change forecasts little; matched-key
ELBO comparisons do not show a clear improvement at their Monte Carlo
resolution. These observations do not certify stationarity/global
optimality, exhaustive basin exploration, or universal optimizer adequacy.
The original archived B80 good/bad-basin contrast was not reproduced by
this new acquisition and the three nearby prescribed starts.

Conditional design information and local sensitivity are different:

| Selected combined case | Minimum likelihood singular value in its own prior coordinates, base -> more | Maximum training GP-path local gain, base -> more |
| --- | --- | --- |
| Shared diffusion-reaction | 0.0241 -> 0.0376 | 48.27 -> 1.859 |
| Shared Euler | 4.554 -> 3.575 | 2.135e22 -> 1.198e36 |
| Shared tumor | 56.00 -> 68.27 | 3.495 -> 2.737 |
| Shared chemotherapy | 9.735 -> 5.103 | 5.005e6 -> 134.94 |
| Native chemotherapy | 8.66e-5 -> 0.0641 | 1.035 -> 50.68 |

Shared Euler has all likelihood directions above the prior scale but an
extremely sensitive learned vector field. Conditioning on an accurate
trajectory does not ensure a usable rollout. Conversely, low training
gain does not ensure future accuracy: diffusion more has modest local
gain but later leaves the training regime and is censored.

Sparse native chemotherapy has approximately 99.56% GP-curve error on
the common source-reference grid and all 40 likelihood singular values
below one. Its quiet, nearly flat GP is unlike the shared model's much
more observation-faithful but amplifying trajectory/operator pair.
Native more-data GP error falls to approximately 9.78%, yet the ODE
forecast remains poor. These are not one identical failure mechanism.

The full-feature tumor base gain is 36.99 versus 3.495 target-only,
alongside forecast error 70.98 versus 5.68%. For chemotherapy more, the
correction reduces gain from 134.94 to 52.73 and forecast error from
71.06 to 44.89%. Gains are diagnostics, not forecast-selection criteria
or probabilities. Prior-coordinate information is not comparable across
different learned/fixed priors as if it were an absolute data quantity.

For nonlinear models, local transport along the GP is not exact transport
of a large ODE error. The finite-error checks use the degree-two identity:

`f(y,u) - f(x,u) = J_f((y+x)/2,u) * (y-x)`.

Writing `e = y_ODE - x_GP`, the derivative defect is
`r = f(x_GP,u) - x_GP'`, and `e' = J_midpoint * e + r`.
Initial-error and defect-forced vectors are retained separately and may
cancel; their norms are not variance fractions. All 40 local gains and
matrices agree under refinement. All 37 eligible finite-error
reconstructions agree; the three training-censored shared paths were not
rescued or assigned a full-window decomposition.

## Noise scales and numerical regularization

The native GP observation diagonal is
`nu + max(1e-5, signal_variance * gp_jitter_rel)`.
The relative nugget is 1e-3 except Euler's 1e-4. Looking at nu alone can
miss much of the effective diagonal.

| Case | Base mean-hyperpoint effective variance / instrument variance, mode range | More-data range |
| --- | --- | --- |
| Heat | 0.552-3.322 | 0.715-2.705 |
| Diffusion-reaction | 3.251-44.572 | 1.998-23.136 |
| Euler | 0.452-0.835 | 0.796-1.040 |
| Tumor | 24.92-7,222 | 84.40-3,834 |
| Chemotherapy | 10.46-3.066e6 | 6.462-23,694 |

These are variance ratios, not SD ratios. Large prior-location ratios
alone do not diagnose the posterior: heat's inferred scales are much
closer to calibration and its native forecasts are useful. Diffusion's
leading mode is largely nugget-dominated, while its third mode is mainly
learned-noise-dominated. Sparse chemotherapy's largest ratio is chiefly
learned nu. There is no single universal nugget explanation.

All 500 scale draws and fixed-32 GP-mean fidelity comparisons are retained.
Chemotherapy's poor mean-hyperpoint GP fidelity persists across those
draws. Tumor has some mean-hyperpoint skew, not a complete explanation.
Noise inflation is not uniquely dosing-related, but these comparisons do
not establish its causal contribution to forecast error. Synthetic
instrument calibration remains diagnostic information, not a native prior
override or a claim that all model discrepancy is measurement noise.

### A noise-conditioning assumption does not transfer cleanly

A bounded, read-only audit after the sweep measures actual
`y - clean_sampled`, not GP prediction error. In diffusion-reaction mode 3,
the original reduced noise has RMS 5.84 times the fixed-basis instrument SD,
versus 0.90 for the independently added columns. Tumor modes 2/3/4 have
original RMS ratios 7.67/24.16/49.98, versus 0.99/1.07/0.88 for added
columns. The more-data cohort still includes every original noisy column;
these added-column figures are not its overall noise level.

Those bases were fitted to the original noisy snapshots. Conditioning
instrument calibration on a fixed basis need not describe the same
observations that selected that basis. The empirical contrast is
consistent with such basis-selection effects. In chemotherapy, whose
basis uses clean training snapshots, original ratios are 0.98-1.05 and
added ratios 0.94-1.15. Euler's original/added ratios are also much closer
to one.

This is a concrete limitation of transporting the shared recipe's
fixed-calibration assumption from chemotherapy. It also means that
`nu / instrument_variance > 1` alone does not show the native model is
overestimating the actual reduced training noise. No recalibration or
refit followed this audit, and it is not a causal explanation for all
failures: Euler's clean-curve control still fails. Basis dependence,
time-varying masks, nonlinear transforms and kernel/model mismatch remain
distinct issues.

The native normal backend also adds physical-coefficient precision jitter
`1e-6 * max(trace(M)/number_of_features, 1)`. This is distinct from its
declared prior and can be substantial relative to weak prior diagonals:
up to 168.8 times a declared-prior diagonal for more-data chemotherapy.
Native information diagnostics retain this term; archived O covariance is
not mislabeled as the inverse of bare `I + D.T*D`. It was not retuned or
declared the sole cause.

## Source-reference and nonlinear-feature qualifications

Chemotherapy's native reference first interpolates source FOM knots to
200 points and then interpolates again. At base observation times it
differs from directly generated clean observations by 0.213685% overall
in reduced coordinates; the worst mode-1 discrepancy is 0.176027 near
day 41.048, approximately 24.97 measurement SD. Dense-grid excursions
can be larger. This is regridding, not shifted ICs or reused noise.

The original source-knot spline in the new basis reproduces clean
observations to about 3e-12. It is an evaluation reference, not an exact
analytic FOM solution. Native 400-query physical scores remain untouched.
The base-observation discrepancy is not a uniform forecast bound or a
correction to subtract from percentages.

Source-reference GP errors remain substantial. The calibrated data-only
chemotherapy GP has 15.133 -> 5.265% training error. Its pointwise
fixed-hyperparameter 95% bands include the source curve at approximately
79.99 -> 49.44% of mode/time entries: more data also contracts the bands.
Tumor inclusion is approximately 46.07 -> 51.00%. These are descriptive
inclusions for single fixed curves, not posterior/repeated-sampling
calibration or uncertainty bands for the fitted MAP paths.

Eight corrected oracle fits use only interior training source knots and
saved clean endpoints, with no outside knots or full-source endpoint
derivatives. The training-spline derivative is not an exact PDE derivative.
For chemotherapy, strong/combined O changes about 27-29% in coordinate
norm versus the coarse oracle; weak-only changes about 0.46%.
Coordinate norms mix coefficient units, so prior-standardized differences
are also archived. Corrected combined forecast is 14.860% rather than
21.985% at base, and 14.957% rather than 21.889% with more data.
Weak-only changes from 14.915% to 14.755% at base. This reference
sensitivity matters for oracle interpretation, not for original fits.

For quadratic models, conditional GP coefficient draws were propagated
through the actual polynomial weak features: 2,048 draws in 16 batches
for each of 14 trajectories/cohorts, including all heat paths.

| Case | Largest omitted weak-feature mean, target-whitened norm, base -> more | Largest pure-quadratic fraction of weak-residual variance |
| --- | --- | --- |
| Heat | 9.64e-11 -> 9.22e-11 | Below 9.4e-22 |
| Diffusion-reaction | 0.0185 -> 0.00302 | 9.24e-5 base; 2.11e-6 more |
| Euler | 6.63e-6 -> 1.72e-5 | Below 1.31e-10 |

The isolated quadratic term is non-Gaussian but small at these endpoints.
Linear random-feature terms also change covariance. Some paired covariance
changes are comparable to their Monte Carlo errors; raw empirical maxima
around 1.08-1.12 are not established variance-inflation factors.
Near-null control-variate covariance estimates retain small negative
eigenvalues, rather than being clipped or used as a likelihood.
No exact affine Gaussian correction was silently applied to quadratic
models, and no Gaussian moment-matched refit was introduced.

## Numerical integrity and boundaries

Data nesting, source hashes, native state continuity and all target/IC
contracts were independently checked. Physical metrics use actual
shifted/scaled decoders: heat's physical first lifted half and Euler's
three disjoint variables. Neither is replaced by an orthonormal-POD
shortcut. Independent native whole-field probe energies agree with the
adapted evaluator to about 1.88e-13 relative.

Shared cached objectives, operators and states replay for all 72 fits.
Independent SVD conditional solves, direct functional covariance
construction, polynomial quadrature and directional derivatives agree.
The largest auxiliary-state roundoff nugget is 5.01e-6 of calibrated
measurement variance. These declared roundoff terms are not empirical
discrepancy inflation.

Parent checks reconstruct every native/shared/control error and preserve
all failed-target pools. The Euler base local gain was also reproduced
with independent Radau integration, agreeing in its variational matrix
to 8.7e-11 relative. Exact autonomous affine matrix exponentials reproduce
native tumor and all four selected shared combined/full tumor curves.
Thus the large sensitivity and sparse full-feature tumor regression are
not merely reported optimizer/solver success flags.

Two execution events are preserved. Chemotherapy preparation initially
hit a 20,000-point allocation guard; its prescribed quadrature needs
21,016 points. A recorded resource-only ceiling increase to 100,000
permitted the same rule and missing predeclared fits, without changing
hyperparameters, nodes, priors or observations.

The generic evaluator initially called dense output with an empty query
inside a short input-knot interval. Its isolated fix skips only that empty
evaluation, retaining state advancement, every input interval, events,
tolerances and budgets. The failed attempt, before/after sources, 14
controls and unchanged prior outputs remain archived. This was an
evaluation exception, not evidence of a model failure.

Native byte-level replay additionally requires the original
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`. Default CPU execution
changed heat point coefficients by up to 4.47e-6; original settings restore
exact point/root equality. Producer O-draw replays cover tested indices
0, 248 and 499 under that setting, not an all-500/platform guarantee.

The existing method already obtains useful derivative/weak-form forecasts
outside chemotherapy without integrating ODEs in the training objective.
This survey does not refute that approach or the shared-state idea. It
does show that the present fixed-hyperparameter MAP recipe is not a safe
general replacement, and that more samples, weak constraints and covariance
correction cannot each be treated as a universal repair.

## Persistent artifacts

All numerical artifacts are under session
`270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics/`.
The repository record does not overwrite canonical experiment outputs.

`broad_survey_protocol.json` and
`broad_survey_method_clarifications.json` declare the scope and paired
affine conventions. `broad_data.py`, `broad_native.py`, `broad_features.py`
and `broad_survey/data_API.txt` define the frozen data/native contracts.
`broad_survey/data/`, `data_guard_audit/`, `native/`,
`evaluation_contract/` and `native_physical_probes/` retain the sources,
calibration, data/basis hashes, checkpoints and native draws.

`broad_shared.py`, `broad_shared_runs.py`, `broad_shared_audit.py` and
`broad_survey/shared/` retain all preparation arrays, starts, accepted
optimization traces, objectives, 40 original controls and uncertainty
draws. Its `source_exact_reference/corrected_summary.json` supplies
evaluation-only source comparisons. Final oracle fitting uses
`source_knot_training_oracles/oracle_handoff.json`, not the superseded
full-source fitting-spline controls. An earlier report-only double
subtraction in a relative-O-change field is explicitly corrected and
archived; no operator or GP metric changed.

The authoritative evaluator handoff is
`broad_survey/evaluation/completed_API.txt` and
`evaluation/final_readout/`: native and all/selected shared CSVs,
corrected-oracle pairs, common 401-grid GP comparisons, linked noise
and prior records, and source/artifact hashes. Per-target trajectories
are in `evaluation/native/` and `evaluation/shared_points/`.
Identification, local matrices and finite-error components are in
`shared_diagnostics/`, `shared_refinement/`, `native_fixed_guides/` and
`native_selected_diagnostics/`, under `evaluation/`.

Parent reproduction scripts are `broad_verify_frozen.py`,
`broad_verify_shared.py`, `broad_verify_uncertainty.py`,
`broad_verify_source.py`, `broad_verify_evaluation.py`,
`broad_verify_shared_predictions.py` and `broad_verify_diagnostics.py`;
their durable results are in `broad_survey/parent_verification/`.
`broad_noise_conditioning_audit.py` and
`broad_survey/parent_verification/noise_conditioning.json` preserve the separate
original-versus-added reduced-noise audit without changing any calibration.
`broad_verify_clean_ic.py` checks the two explicitly privileged IC controls.
The final evaluator summary hash is
`8590ec37dd193ee1407b840f814bc155fd4de405eb4fadfabe8bc60047c01f9c`.
