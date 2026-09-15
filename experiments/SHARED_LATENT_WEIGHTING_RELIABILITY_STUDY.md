# Shared-latent weighting and conditional reliability

## Main findings

**The reduced-data errors are not always ordinary white noise, the choice
of operator terms to regularize matters, and an accurate MAP point does
not establish reliable prediction.**

This follow-up answers three questions from the
[mechanism study](SHARED_LATENT_MECHANISM_STUDY.md):

1. A simple training-observation-based noise estimate does **not** recover
   the earlier truth-informed tumor benefit. It misses smooth or concentrated
   reduced-data errors, and can mistake genuine dynamics for noise.
2. Preserving c/A/B prior widths while shrinking H/N prior widths improves
   the strongest-prior heat and Euler point predictions substantially. It is not a universal fix;
   the moderate heat setting still fails, and chemotherapy interaction
   fits remain iteration-capped.
3. Conditional operator draws expose substantial fragility. The earlier
   tumor point with 2.475% reduced forecast error has seven censored draws
   out of 32, and a **13.10% median among the 25 completed draws**.
   The point improvement is not a corresponding reliability result.

No production code or defaults were changed. Recovery commit `1bceaa9`
was preserved as
`backup/shared-latent-before-weighting-reliability-1bceaa9`.
The previous sealed study and its documentation remain unchanged; this
record links back rather than rewriting their frozen hashes.

## Frozen design

The same five base acquisitions, bases, observation times/noise realizations,
training/forecast windows, inputs and original observed initial states are
used. Heat retains all five training forcings with one shared operator and
a separate heldout forcing. No new FOM data, POD fit, native SVI or native
posterior ensemble was run.

All fits retain combined strong-plus-weak target-only block-diagonal
constraints, 40 latent/strong nodes, 20 p6 tests of radius 0.1 training span,
and the original operator-prior reference and discrepancy arrays.
The same fixed-hyperparameter MAP prototype and exact Gaussian profiling
are used, not a replacement for the native hierarchical Bayesian model.
There is no ODE solve in a fitting objective.

Two new branches were predeclared and are **not combined**:

| Branch | Noise setting | Operator-prior SD rule |
| --- | --- | --- |
| MAD | Frozen training-observation-based estimate below | All columns multiplied by 1, 0.1 or 0.01 |
| H/N-only | Original instrument-noise preparation | H/N columns multiplied by 0.1 or 0.01; c/A/B remain at 1 |

Here H is quadratic in state; N is an input-state interaction, linear in
state when the input is fixed. This grouping is not a stability constraint,
an exact treatment restriction, or a coordinate-invariant prior.
The frozen POD coordinates are unchanged throughout.

Tumor has no H/N columns. Its two H/N-only settings are exact negative
controls that alias the old instrument/SD1 cell, rather than new fits.
There are **25 nominal cells, 23 genuinely new cells, 69 new joint MAP fits,
46 new fixed-curve regressions and 10 exact alias records**.

Each cell keeps all three prescribed starts (zero, 0.1-SD random101,
0.1-SD random202), a fixed-data-GP operator fit and an approved
clean-training-curve operator fit. A joint start is selected only by its
own cell's eligible objective minimum; no family, noise level or strength
is selected by forecast performance.

The clean controls reuse exactly the archived training-only functionals.
Tumor/chemo use the corrected strict source-knot training splines, not
future knots or full-source endpoint derivatives. Original observed ICs,
the ROM family and regularization remain imposed, so these controls are
not error lower bounds.

### Training-observation-based noise proxy

For each eligible irregular-time triple, define
`a = (t_next - t)/(t_next - t_prev)` and `b = 1 - a`, then
`e = (y - a*y_prev - b*y_next)/sqrt(1 + a*a + b*b)`.
The raw SD estimate is
`median(abs(e - median(e)))/Phi_inverse(0.75)`.
The fitting variance is the maximum of its square and the original
instrument variance.

All three observations must satisfy the original calibration exclusions.
The stencil annihilates an affine signal and has unit marginal variance
under independent unit Gaussian observation noise. Its assumptions need
not hold for these reduced data.

The estimator reads noisy training observations, their times, exclusions
and the inherited instrument floor only. **It does not read clean training
trajectories.** The inherited floor itself is the previous approximate
privileged fixed-basis calibration, so this removes extra truth access;
it does not establish fully deployable real-data noise calibration.

All values were frozen before fitting. Only the 23 GP modes whose fitting
variance changed refit lengthscale and signal variance under the original
data-only GP priors, bounds and optimizer. Unchanged hyperparameters and
flags are reused exactly. Neither the operator-prior reference nor
discrepancy was recalculated from the new GP.

## Primary forecast results

Entries are **physical / reduced relative forecast error in percent**,
with the original observed ICs. Heat's column pools every training-forcing
target, never only successful ones.

**C** means the original numerical large-state safety rule was reached:
`max(abs(q_i)/base_training_RMS_i) = 1e6`. Its pooled forecast score is
unavailable, not zero. This is not proof of mathematical finite-time
blow-up; conversely, a finite rollout is not necessarily accurate.

| Case | Branch | SD factor | Selected joint | Fixed data GP | Clean training curve |
| --- | --- | ---: | ---: | ---: | ---: |
| Heat | MAD | 1 | 11.76 / 36.59 | C | 2.099 / 4.246 |
| Heat | MAD | 0.1 | C | C | 2.021 / 4.450 |
| Heat | MAD | 0.01 | 5.965 / 18.78 | 5.486 / 16.92 | 4.958 / 15.49 |
| Heat | H/N-only | 0.1 | C | C | 2.064 / 3.950 |
| Heat | H/N-only | 0.01 | 2.287 / 9.946 | 2.560 / 11.24 | 1.929 / 7.927 |
| Diffusion-reaction | MAD | 1 | 20.16 / 19.54 | 11.67 / 10.52 | 39.17 / 38.89 |
| Diffusion-reaction | MAD | 0.1 | C | 22.90 / 22.36 | 38.90 / 38.62 |
| Diffusion-reaction | MAD | 0.01 | C | 60.80 / 60.67 | 39.73 / 39.45 |
| Diffusion-reaction | H/N-only | 0.1 | 43.79 / 43.55 | 25.89 / 25.42 | 8.885 / 7.308 |
| Diffusion-reaction | H/N-only | 0.01 | C | C | 5.666 / 2.538 |
| Euler | MAD | 1 | C | C | C |
| Euler | MAD | 0.1 | C | C | 1.438 / 25.22 |
| Euler | MAD | 0.01 | 3.323 / 103.8 | 3.227 / 104.4 | 2.466 / 74.43 |
| Euler | H/N-only | 0.1 | C | C | 2.266 / 57.81 |
| Euler | H/N-only | 0.01 | 1.617 / 32.27 | 2.301 / 59.93 | 1.271 / 8.407 |
| Tumor | MAD | 1 | 5.565 / 12.61 | 5.674 / 12.99 | 3.382 / 1.722 |
| Tumor | MAD | 0.1 | 5.563 / 12.60 | 5.669 / 12.97 | 3.334 / 0.6174 |
| Tumor | MAD | 0.01 | 5.165 / 11.17 | 5.331 / 11.77 | 3.462 / 2.713 |
| Tumor | H/N-only | 0.1 | 5.680 / 13.01 | 5.838 / 13.56 | 3.382 / 1.720 |
| Tumor | H/N-only | 0.01 | 5.680 / 13.01 | 5.838 / 13.56 | 3.382 / 1.720 |
| Chemo | MAD | 1 | C | 3894 / 2269 | 14.84 / 8.617 |
| Chemo | MAD | 0.1 | 985.5 / 574.2 | 3441 / 2005 | 15.13 / 8.783 |
| Chemo | MAD | 0.01 | 74.83 / 43.59 | 65.62 / 38.22 | 16.54 / 9.610 |
| Chemo | H/N-only | 0.1 | 647.9 / 377.5 | 76.02 / 44.28 | 15.42 / 8.955 |
| Chemo | H/N-only | 0.01 | 32.67 / 19.02 | 39.42 / 22.96 | 16.17 / 9.391 |

Both tumor H/N-only rows are exact aliases. All chemo H/N-only starts,
including the selected ones, reach the iteration cap; their apparent
differences must not be presented as converged prior comparisons.

### Which terms are penalized matters

At factor 0.01, changing from uniform shrinkage to H/N-only shrinkage
improves heat's reduced forecast error **18.51% -> 9.946%** and Euler's
**97.89% -> 32.27%**, with instrument noise unchanged. The corresponding
physical errors are 5.858% -> 2.287% and 3.127% -> 1.617%.
This is evidence against treating one global prior scale as sufficient.
It does not establish accurate or robust dynamics, and moderate heat
shrinkage remains censored in both families.

Diffusion's clean-curve control improves from 39.64% to 2.538% reduced
forecast error under the same change, but its selected joint fit is
still censored. Thus the operator family/clean-curve fit can behave
very differently from the noisy-data joint problem. A failure of the
joint prototype is not evidence that no useful ROM exists.

Physical and reduced metrics remain distinct. For Euler H/N-only/0.01,
velocity, pressure and inverse-density forecast errors are respectively
0.966%, 1.617% and 1.333%, while reduced error is 32.27%.
Pressure units dominate the aggregate physical energy. The archived
native base point has 16.606% reduced error; the new prototype is not
being promoted over it.

### Heldout heat remains separate

Each entry is **physical / reduced percent**:

| Branch | SD factor | Heldout forecast |
| --- | ---: | ---: |
| MAD | 1 | 2.647 / 10.98 |
| MAD | 0.1 | 2.611 / 10.78 |
| MAD | 0.01 | 3.661 / 16.20 |
| H/N-only | 0.1 | 2.591 / 10.61 |
| H/N-only | 0.01 | 1.847 / 6.164 |

H/N-only/0.01 improves this heldout point relative to uniform/0.01
(3.624% physical / 16.26% reduced). No GP was fitted to the heldout
target, and its result does not erase any failed training-forcing target.

## Why the simple noise estimate misses the earlier benefit

The following are **SD ratios to the inherited instrument calibration**,
not forecast-error percentages. Actual error is `y - clean_y` in the
identical frozen basis, using calibration-eligible observations.
Clean/noise-only diagnostics are computed only after fitting values are
frozen and never feed back into the estimator.

| Reduced component | Actual error RMS / calibration SD | MAD fitting SD / calibration SD |
| --- | ---: | ---: |
| Diffusion mode 3 | 5.838 | 1.284 |
| Tumor mode 2 | 7.674 | 1.048 |
| Tumor mode 3 | 24.16 | 2.300 |
| Tumor mode 4 | 49.98 | 6.147 |

The discrepancy is not simply small-sample MAD bias:

- Tumor modes 2 and 3 have realized ordered lag-one error correlations
  of approximately 0.957 and 0.928. Local linear differencing attenuates
  their error RMS to about 10.3% and 8.8% of its original size. Smooth
  reduced-data error can look like signal to this estimator.
- About 98.7% of tumor mode 4's error energy is concentrated in three
  observations near the end of training. A robust median misses much of
  that large, concentrated error.
- Diffusion mode 3 has about 81.4% of its error energy in three early
  observations, including 56.8% in its noisy initial observation.
- For Euler, the raw data estimator reports SD approximately 2.1-3.2
  times calibration even though actual noise RMS is near one. Applying
  the same estimator to the clean signal alone gives about 1.5-2.7:
  genuine curvature is being counted as noise.

These are descriptive properties of these realized acquisitions, not
a proof of a stationary correlated noise law or that noisy POD selection
is the sole cause. The independent added observations from the old
more-data cohort have a different apparent noise pattern in the same
frozen basis. They are audited separately, never pooled into these fits.

The stencil/implementation also passes exact affine annihilation,
unit-noise marginal-variance, affine-shift, scale and excluded-observation
controls. Across 1,000 Gaussian-noise/affine-signal replicates on each of
nine original time grids, median raw estimated SD is close to one.
No Monte Carlo correction, alternative estimator or tuning was added.

For tumor, the data-based estimate therefore produces only a small
forecast change at global SD1, 13.01% -> 12.61% reduced error, rather
than the truth-floor result of 2.475%. This tests one transparent
estimator, not the impossibility of learning noise without truth.

There are also mixed effects rather than a universal failure. Diffusion
MAD/SD1 becomes finite at 19.54% reduced error, while its fixed-GP control
is better at 10.52%; the other joint MAD strengths remain censored.
In chemo, the data-GP source error worsens 15.13% -> 17.24%.
At SD0.01 the jointly adjusted curve is more accurate (8.670% -> 7.990%),
yet ODE reduced forecast error worsens **20.46% -> 43.59%**.
Improving the latent curve alone is still not sufficient.

## Conditional operator reliability experiment

The panel contains **all 30 old mechanism cells and all 23 genuinely new
cells**, not a forecast-selected subset. Exact tumor aliases reuse their
old context. For each of these 53 contexts:

1. Keep its saved joint latent state, GP hyperparameters, fitting noise,
   operator prior and discrepancy fixed.
2. Compute the exact target-only conditional Gaussian operator mean and
   covariance by augmented QR in that cell's prior coordinates.
3. Evaluate the conditional mean separately from the original MAP point.
   Use a positive-R-diagonal QR orientation and the frozen case-wise
   standard-normal array to draw 32 operators.
4. Hold original observed ICs fixed. One same operator draw serves all
   five heat training forcings; heldout heat remains separate.

No normal-equation jitter, eigenvalue rescue, sampled IC, GP extrapolation
or joint latent/hyperparameter posterior sampling is introduced. This is
**conditional operator uncertainty at a fixed fitted trajectory**, not
the full Bayesian posterior or calibrated coverage. Adding latent/GP
uncertainty need not change nonlinear forecast reliability monotonically.

The conditional mean is not silently substituted for the MAP point.
The largest MAP-to-conditional-center precision distance over all
53 contexts is below 0.006, small relative to the conditional Gaussian
scale. This recentring is not the explanation for the large draw spread,
nor is it a global joint-optimization certificate.

### Illustrative contrasts from the complete 53-context panel

Errors in this table are **reduced forecast percent**. Completion means
the complete training-target set for a draw, including all five heat
training targets. The `<=25%` column counts the compound event
**complete and reduced error <=25% out of all 32 draws**, rather than
discarding failures.

| Case | Setting | MAP error % | Complete / 32 | Complete and <=25% / 32 | Median among completed % |
| --- | --- | ---: | ---: | ---: | ---: |
| Tumor | Truth floor, uniform SD1 | 2.475 | 25 | 16 | 13.10 |
| Tumor | Truth floor, uniform SD0.1 | 2.767 | 26 | 24 | 8.833 |
| Diffusion-reaction | Truth floor, uniform SD0.01 | 8.830 | 28 | 13 | 27.01 |
| Heat | Instrument, uniform SD0.01 | 18.51 | 31 | 8 | 30.29 |
| Heat | Instrument, H/N-only SD0.01 | 9.946 | 31 | 20 | 22.77 |
| Euler | Instrument, uniform SD0.01 | 97.89 | 4 | 0 | 112.8 |
| Euler | Instrument, H/N-only SD0.01 | 32.27 | 10 | 0 | 95.01 |
| Chemo | Instrument, uniform SD0.01 | 20.46 | 32 | 21 | 18.41 |
| Chemo | Instrument, H/N-only SD0.01 | 19.02 | 32 | 23 | 17.17 |
| Chemo | MAD, uniform SD0.01 | 43.59 | 32 | 0 | 42.35 |

The chemo H/N-only row uses an iteration-capped selected fit.
These rows illustrate hypotheses, not selected deployment settings;
every remaining setting is retained in the complete CSV/JSON panel.

Heat's structured prior improves the conditional <=25% count from
8/32 to 20/32 at the tightest scale, not only its MAP point.
But its median completed error remains 22.77%, versus a 9.946% point,
and one draw is censored. Euler's improved point remains particularly
fragile: 22/32 H/N-only draws are censored, and none meets the declared
<=25% compound event.

The earlier tumor truth-floor/SD1 point is also misleading as a
standalone reliability summary: seven draws are censored, and the
95th percentile **among the 25 completed draws** is about 24,720%
reduced error. The floor/SD0.1 point is similar, but its completed
median and conditional tail are substantially better; six draws
still remain censored.

Finite-only quantiles are never all-draw quantiles. With 32 draws,
these counts and tails are exploratory, not precise posterior failure
probabilities. No coverage claim is made. The full target-level panel
retains every heat target, every Euler variable and each completion
denominator.

## Optimization, validation and limits

All 125 nominal new records are retained. Of the 69 genuinely new
joint fits, 63 stop on relative objective reduction and six hit 2,000
iterations. All six capped runs are chemo H/N-only, at both strengths.
Neither new nor aliased joint endpoints meet the strict 1e-6 gradient
tolerance. The largest terminal infinity norm is 0.948; the largest
within-cell objective spread is 0.07228.

The selected chemo H/N-only/0.1 and /0.01 fits have gradients about
0.007428 and 0.06389 respectively and are both capped. Thus this branch's
chemo outcomes do not separate optimization limitations from a poor
objective/prior choice. No extensions or rescue starts were added.

One of 23 GP refits, heat trajectory 2/mode 0 (zero-based), returns
`ABNORMAL` after nine iterations, with gradient norm 5.05e-6.
Its finite parameters, flag and trace are retained without a retry.
Unchanged original GP warning records remain unchanged.

Primary point accounting is 250 target records, including 10 reused
tumor-alias target records: 200 complete and 50 censored.
The uncertainty experiment contains 108 conditional-center targets and
3,456 draw-target integrations: 2,371 complete and 1,085 censored.
These aggregate counts are accounting across different cases/settings,
not a pooled reliability probability. There are **3,804 genuinely new
primary/center/draw target integrations**, excluding numerical reproductions.

Independent checks cover every model/noise/prior-mask invariant, exact
alias, profile/conditional regression, all primary and draw scores,
Gaussian moments and recovered standard-normal draws, every target/role
completion denominator, and 390 target-variable quantile comparisons.
Twenty-five point and 18 fixed-index draw reproductions use stricter
Radau with an independent polynomial RHS. Maximum relative differences
are 9.60e-9 and 8.31e-7 respectively; censor times also agree.

The original DOP853 policy remains rtol 1e-8, atol 1e-10, horizon/400
maximum step, 200,000 RHS calls, 120 seconds per target, and the same
input-knot/training-boundary splitting. There is no clipping or
survivor-only pooling. Float64/thread/runtime/source provenance is
recorded. No new prior/noise family was selected, combined or expanded
after viewing outcomes.

## Persistent artifacts

The existing session artifact root is:

```text
BASE = /Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics
```

| Path relative to BASE | Content |
| --- | --- |
| `weighting_protocol.json` | Predeclared questions, estimator, masks, draw panel and scope |
| `weighting_reliability/frozen_interventions.json` | Every fitting variance/mask and case-wise Gaussian draw source |
| `weighting_reliability/fitting_release.json` | 125-record fitting handoff and immutable producer dependencies |
| `weighting_reliability/fitting_API.txt` | Exact cache, alias and prior/noise contracts |
| `weighting_reliability/parent_noise_controls.json` | Direct estimator, synthetic, clean/noise-only and independent-added audits |
| `weighting_reliability/parent_noise_structure.json` | Post-freeze error concentration and temporal-structure diagnostics |
| `weighting_reliability/point_evaluation/summary.json` | All new nominal primary point records and target pools |
| `weighting_reliability/curve_diagnostics/summary.json` | Source-aligned GP/latent errors, including identity controls |
| `weighting_reliability/conditional/` | All 53 moment arrays, fixed-index draws, centers, target scores and protocols |
| `weighting_reliability/readout/all_point_role_scores.csv` | All 275 old/new nominal point records and applicable roles |
| `weighting_reliability/readout/selected_and_control_role_scores.csv` | Complete primary comparison, including heldout heat |
| `weighting_reliability/readout/conditional_role_scores.csv` | All 53 unique conditional contexts and 64 applicable role rows |
| `weighting_reliability/readout/conditional_contexts.json` | Complete target/variable/role uncertainty summaries and draw links |
| `weighting_reliability/conditional_verification/` | Independent every-draw Gaussian-law/metric checks |
| `weighting_reliability/numerical_primary/` | 25 independent point reproductions |
| `weighting_reliability/numerical_draws/` | 18 fixed-index draw reproductions |
| `weighting_reliability/parent_*_verification.json` | Independent model, point, summary and report checks |
| `weighting_reliability/study_release.json` | Final accounting, source/artifact integrity and record link |

Use the existing `prob_rom` environment and one BLAS/OpenMP thread.
`weighting_models.load_prepared(case, family, SD_multiplier)` loads exact
saved maps; it never silently fits or rebuilds missing caches.
The parent and fitter `weighting_*.py` scripts are preserved under BASE.
Completed output directories are protected and should not be overwritten.
These are persistent exploratory session artifacts, not promoted
canonical benchmark outputs.
