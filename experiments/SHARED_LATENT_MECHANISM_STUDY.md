# Shared-latent mechanism tests across five problems

## Main finding

**Data weighting matters, but there is no universal noise, regularization,
or initial-state switch that fixes this prototype. There is nevertheless
positive evidence that jointly adjusting the latent trajectory and ODE
can help, rather than merely fitting a prettier interpolation.**

The clearest example is autonomous tumor: changing the assumed reduced
noise while retaining the original operator prior improves reduced
forecast error from **13.01% to 2.475%**. Holding that improved data-GP
curve fixed and fitting only the operator instead produces a censored
rollout. Diffusion-reaction also needs an interaction: only the
noise-floor plus tightest-prior joint cell completes its original-IC
forecast, at **8.830% reduced / 10.17% physical error**; its fixed-GP
operator control remains censored.

These noise changes use clean training truth as a **privileged diagnostic**.
They are not a deployable noise estimator or evidence of a fair predictive
advantage over the existing Bayesian method. Stronger operator shrinkage
also helps some failures, but smaller coefficients are not a stability
constraint. Merely starting the ODE on its own fitted GP usually changes
little or makes the forecast worse.

This is a bounded follow-up to
[the cross-experiment survey](CROSS_EXPERIMENT_SHARED_LATENT_SURVEY.md).
Recovery commit `3c0f569` was preserved as
`backup/shared-latent-before-mechanism-tests-3c0f569` before the study.
No production code, default, raw acquisition, basis, or historical artifact
was changed. Nothing was pushed.

## Questions and predeclared experiment

1. Are the fixed reduced-noise assumptions making us over-trust noisy
   compressed observations?
2. Is the operator prior too permissive, allowing unreliable dynamics?
3. Is an initial-state mismatch between the fitted GP and the ODE rollout
   the main trigger of failure?

The protocol and every per-mode noise intervention were frozen before
new fits. The same **base acquisitions** from the survey were reused:

| Case | Observations | Injected noise | Modes / operators | Training | Prediction |
| --- | --- | ---: | --- | --- | --- |
| Heat | 80 each, five training forcings | 5% | 5 / cAHBN, two inputs | 0-1 | 0-2 |
| Diffusion-reaction | 60 | 5% | 3 / cAH | 0-1 | 0-3 |
| Euler | 55 | 3% | 6 / cAH | 0-0.08 | 0-0.15 |
| Autonomous tumor | 40 | 1% | 4 / cA | days 5-60 | days 5-90 |
| Chemotherapy | 40 | 1% | 4 / cABN, one input | days 5-70 | days 5-110 |

There are two noise conditions and three operator-prior SD multipliers:

- **Instrument:** the survey's original fixed-basis instrument-noise
  proxy, itself an approximate privileged calibration rather than the
  conditional noise law after learning a noisy basis.
- **Empirical floor:** for each training trajectory and mode,
  `fitting_variance = max(instrument_variance, mean((y - clean_y)^2))`.
  The original calibration exclusions are retained. Only training
  observation residuals enter this calculation; the floor never decreases
  noise. Bias, cross-time/cross-mode dependence and basis uncertainty are
  not thereby modeled.
- **Operator SD multipliers 1, 0.1, 0.01:** multiply every active entry of
  the original zero-mean operator-prior SD array by that number. At a fixed
  physical operator, the corresponding quadratic prior penalties are
  multiplied by 1, 100 and 10,000. This is not a change between strong and
  weak constraints.

All cells retain the same combined strong-plus-weak, target-only
block-diagonal likelihood: 40 latent/strong nodes, 20 p6 weak tests with
radius 0.1 of the training span, all inputs and all training trajectories.
Within-block correlations remain; the original omitted strong/weak and
cross-output covariance blocks remain omitted.

The original operator-prior reference and discrepancy SD are frozen even
when GP hyperparameters change. Only the 27 of 42 GP mode contexts whose
noise variance actually increases refit lengthscale and signal variance,
under the original data-only hyperpriors, bounds and optimizer.
Unchanged hyperparameter dictionaries and termination flags are reused
exactly. Original instrument calibration arrays remain the metric
denominators, not the larger fitting variances. Numerical nugget formulas
are unchanged; a changed GP signal variance can change its formula-derived
auxiliary-node nugget, not the frozen discrepancy.

The same certified quadrature rule may change subdivisions when a GP
lengthscale changes. Latent nodes, test supports and inputs do not change.
The resource guards remain 8,192 spectral features and 100,000 quadrature
points; no numerical rescue variances or outcome-driven variants are added.

Each of the 30 cells contains three prescribed joint MAP starts, one
fixed-data-GP operator regression, and one clean-training-curve operator
regression. This gives **150 records: 25 exact reuses, 75 new joint fits,
and 50 new fixed-curve regressions**. The joint starts are zero and the
original 0.1-SD perturbations with seeds 101 and 202. Selection uses only
the lowest eligible training objective within its own cell. No noise
level or prior multiplier is selected by objective or forecast performance.

Affine tumor/chemo fits profile the latent state exactly; quadratic cases
profile the operator exactly. This is the fixed-hyperparameter shared-MAP
prototype, not the native hierarchical operator-marginal SVI model.
No ODE is evaluated inside any fitting objective.

The clean controls retain exactly the archived training-only functionals.
Tumor/chemo use the corrected strict source-knot training splines, never
future knots or full-source boundary derivatives. They retain the same
cell covariance, regularization, ROM family and original observed IC:
**a clean-curve control is not an achievable-error lower bound.**

## Complete primary forecast matrix

Each entry is **physical / reduced relative forecast error in percent**,
using the original observed initial states. Heat pools every one of its
five training-forcing targets; heldout heat is separate below.

**C** means the common numerical large-state safety threshold was reached:
`max(abs(q_i) / base_training_RMS_i) = 1e6`. The corresponding pooled
segment has no finite score. It is neither zero error nor proof of
finite-time mathematical blow-up. Numerical completion alone is not success.

| Case | Noise | O-prior SD multiplier | Joint, own-objective selected | Fixed data GP | Clean training curve |
| --- | --- | ---: | ---: | ---: | ---: |
| Heat | Instrument | 1 | 10.07 / 32.02 | C | 2.099 / 4.246 |
| Heat | Instrument | 0.1 | C | C | 2.021 / 4.450 |
| Heat | Instrument | 0.01 | 5.858 / 18.51 | 5.468 / 16.83 | 4.958 / 15.49 |
| Heat | Floor | 1 | 13.35 / 41.68 | C | 2.099 / 4.246 |
| Heat | Floor | 0.1 | C | C | 2.021 / 4.450 |
| Heat | Floor | 0.01 | 6.063 / 18.98 | 5.546 / 16.93 | 4.958 / 15.49 |
| Diffusion-reaction | Instrument | 1 | C | C | 39.26 / 38.98 |
| Diffusion-reaction | Instrument | 0.1 | C | C | 38.92 / 38.64 |
| Diffusion-reaction | Instrument | 0.01 | C | C | 39.91 / 39.64 |
| Diffusion-reaction | Floor | 1 | C | C | 39.12 / 38.84 |
| Diffusion-reaction | Floor | 0.1 | C | C | 38.87 / 38.59 |
| Diffusion-reaction | Floor | 0.01 | 10.17 / 8.830 | C | 39.47 / 39.20 |
| Euler | Instrument | 1 | C | C | C |
| Euler | Instrument | 0.1 | C | C | 1.440 / 25.36 |
| Euler | Instrument | 0.01 | 3.127 / 97.89 | 4.232 / 142.7 | 2.471 / 74.72 |
| Euler | Floor | 1 | C | C | C |
| Euler | Floor | 0.1 | C | C | 1.441 / 25.38 |
| Euler | Floor | 0.01 | 3.029 / 93.93 | 3.938 / 131.9 | 2.469 / 74.66 |
| Tumor | Instrument | 1 | 5.680 / 13.01 | 5.838 / 13.56 | 3.382 / 1.720 |
| Tumor | Instrument | 0.1 | 5.674 / 12.99 | 5.832 / 13.54 | 3.334 / 0.6182 |
| Tumor | Instrument | 0.01 | 5.083 / 10.86 | 5.387 / 11.97 | 3.462 / 2.713 |
| Tumor | Floor | 1 | 3.440 / 2.475 | C | 3.330 / 0.4289 |
| Tumor | Floor | 0.1 | 3.468 / 2.767 | C | 3.337 / 0.7493 |
| Tumor | Floor | 0.01 | 3.973 / 6.140 | 117.2 / 331.1 | 3.462 / 2.713 |
| Chemo | Instrument | 1 | C | 81.71 / 47.60 | 14.86 / 8.628 |
| Chemo | Instrument | 0.1 | 486.5 / 283.4 | 75.73 / 44.12 | 15.33 / 8.901 |
| Chemo | Instrument | 0.01 | 35.13 / 20.46 | 40.13 / 23.37 | 16.42 / 9.540 |
| Chemo | Floor | 1 | C | 83.11 / 48.41 | 14.85 / 8.622 |
| Chemo | Floor | 0.1 | 492.1 / 286.7 | 77.23 / 44.99 | 15.31 / 8.891 |
| Chemo | Floor | 0.01 | 36.11 / 21.02 | 39.29 / 22.88 | 16.40 / 9.530 |

All three starts, training/full-window errors, every target and every
Euler variable remain in the machine-readable records. The table is not
a forecast-selected leaderboard.

### Native reference context, not a new native comparison run

The untouched native representative-point base forecast errors are:

| Case | Physical % | Reduced % |
| --- | ---: | ---: |
| Heat, pooled | 1.976 | 3.124 |
| Diffusion-reaction | 8.910 | 7.339 |
| Euler | 1.351 | 16.606 |
| Tumor | 6.253 | 14.964 |
| Chemo | 324.915 | 189.301 |

These are archived point predictions, not posterior predictive means.
No native SVI or native32-draw evaluation was repeated, and the floor
conditions have extra privileged information.

## What the noise intervention tells us

Data-only GP interpolation improves as follows on the common 401-point
training grids. Tumor/chemo use the source-knot evaluation reference;
the PDE cases retain the native training reference.

| Case | Instrument GP reduced error % | Floor GP reduced error % |
| --- | ---: | ---: |
| Heat | 0.9839 | 0.9822 |
| Diffusion-reaction | 0.9148 | 0.5429 |
| Euler | 3.1729 | 3.0844 |
| Tumor | 1.8327 | 1.5093 |
| Chemo | 15.1333 | 15.1328 |

The previous audit found original tumor noise RMS approximately
7.7, 24.2 and 50.0 times the independent fixed-basis calibration in its
last three modes, and approximately 5.8 times calibration in diffusion's
third mode. The current floor directly de-weights those observations.
Chemo's clean-basis acquisition has ratios near one, so its intervention
is correspondingly small.

For tumor at SD multiplier 1, joint latent error improves
1.941% -> 1.418%, and reduced ODE forecast error improves
13.01% -> 2.475%. But the floor data-GP cut becomes censored despite
the better interpolation. The room to **jointly adjust the trajectory**
is doing something useful here; better independent smoothing alone is
not sufficient.

Diffusion shows a noise/prior interaction rather than either isolated
fix: the floor plus SD multiplier 0.01 produces a joint latent error of
0.397% and reduced forecast error of 8.830%. Its fixed-GP control is
censored. Its clean-curve control is still around 39%, but also starts
from the original noisy IC and uses the same imposed regularization and
model class. This does not show that noisy data are intrinsically more
informative than truth.

These interventions support effective reduced-noise misspecification as
one contributor. They do not prove noisy-POD selection is the sole cause,
and they do not identify a deployable way to learn the correct noise law.
Heat, Euler and chemo remain counterexamples to a universal noise fix.

## What operator shrinkage tells us

On instrument-noise chemo, SD multiplier 1 -> 0.1 -> 0.01 reduces
maximum training local variational gain from about
5.01 million -> 674 -> 9.19. Joint latent interpolation error improves
18.24% -> 12.39% -> 8.67%. The rollout changes from censored to
486.5% -> 35.13% physical forecast error. This is substantial progress
without any treatment-specific restriction, but still not accurate
treatment prediction.

Euler's corresponding local gains fall from approximately
2.13e22 -> 1.32e6 -> 24.1. Its tightest-prior rollout completes, yet
retains **97.89% reduced forecast error**. With the floor that number
is still **93.93%**, despite only 3.029% total physical error.
The physical and reduced metrics answer different questions.

| Euler point | Velocity % | Pressure % | Inverse density % | Reduced % |
| --- | ---: | ---: | ---: | ---: |
| Native, archived | 0.822 | 1.351 | 1.051 | 16.606 |
| Joint instrument, SD 0.01 | 1.784 | 3.127 | 4.484 | 97.89 |
| Joint floor, SD 0.01 | 1.722 | 3.029 | 4.290 | 93.93 |

Pressure units dominate the aggregate physical energy. A small total
field percentage must not be presented as accurate reduced dynamics.

Heat supplies a nonmonotonic counterexample. Instrument SD 1 -> 0.1
shrinks the operator norm in the **same original prior coordinates**
from 0.864 to 0.285, yet increases local gain from 1.81 to 21.9 and
changes its pooled forecast from finite to censored. The clean-curve
control does not share that failure. SD 0.01 becomes finite again but
still has 18.51% reduced forecast error, compared with 3.124% for the
archived native point.

Shrinking all coefficients can remove stabilizing combinations as well
as unreliable ones. Moreover, local gain along the training GP is not
a forecast stability certificate: diffusion with instrument SD 0.01
has local gain only 1.346 but its forecast is censored. Conditional
information spectra are also expressed in each cell's own prior
coordinates; tightening the prior mechanically reduces those singular
values and must not be described as worse data.

## Initial-state consistency controls

Only after all primary cells were scored, each of the 30 selected joint
operators was held fixed and rerun from its own continuous GP initial
state. No clean IC was substituted. Linked training targets use their
own GP; heldout heat retains its original IC because it has no fitted GP.

Entries below are **observed-IC -> fitted-GP-IC physical forecast %**:

| Case | Instrument SD 1 | Instrument SD 0.1 | Instrument SD 0.01 | Floor SD 1 | Floor SD 0.1 | Floor SD 0.01 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Heat, pooled | 10.07 -> 10.13 | C -> C | 5.858 -> 5.862 | 13.35 -> 13.66 | C -> C | 6.063 -> 6.066 |
| Diffusion-reaction | C -> C | C -> C | C -> C | C -> 39.44 | C -> C | 10.17 -> 15.57 |
| Euler | C -> C | C -> C | 3.127 -> 3.301 | C -> C | C -> C | 3.029 -> 3.201 |
| Tumor | 5.680 -> 5.651 | 5.674 -> 5.645 | 5.083 -> 5.057 | 3.440 -> 3.439 | 3.468 -> 3.466 | 3.973 -> 6.399 |
| Chemo | C -> C | 486.5 -> 570.7 | 35.13 -> 35.22 | C -> C | 492.1 -> 569.3 | 36.11 -> 36.20 |

Only one censored selected cell becomes complete: diffusion floor/SD 1,
and its forecast remains poor. Most changes are small or detrimental.
This does not support initial-state mismatch as the common dominant cause.

Only diffusion's original IC is noisy. There, floor/SD 0.01 actually
moves the initial state closer to clean truth in reduced L2 distance,
1.115 -> 0.413, while worsening the physical forecast 10.17% -> 15.57%.
Correcting one ingredient need not improve a misspecified fitted ODE;
errors can compensate. For the other cases, moving from the original
noiseless IC is not automatically denoising.

### Heldout heat is never hidden by pooling

All six heldout selected joint forecasts complete. These are
**physical / reduced percent**, identical under the IC control:

| Noise | SD 1 | SD 0.1 | SD 0.01 |
| --- | ---: | ---: | ---: |
| Instrument | 2.524 / 10.28 | 2.653 / 11.00 | 3.624 / 16.26 |
| Floor | 2.748 / 11.52 | 2.614 / 11.13 | 3.656 / 16.20 |

An acceptable heldout result does not erase a failed training-forcing
forecast. The tighter prior that improves heat's pooled primary score
also worsens this heldout score.

## Optimization and numerical limits

All 150 records are retained. Of the 90 joint records, 88 terminate on
relative objective reduction and two reach 2,000 iterations:
chemo/floor/SD 1, starts zero and random101. The selected start in that
cell is random202, which terminates on relative reduction. No fits
were extended and no additional starts were added.

No joint endpoint meets the declared gradient tolerance of 1e-6.
Terminal infinity norms range from 1.54e-6 to 0.239; every prescribed
terminal directional check passes. Maximum within-cell objective
spread is 3.151e-5. Nearby-start agreement and small relative objective
changes are not global-optimum certificates or a reason to rule out
optimization limitations.

All 27 refitted GP modes report successful termination. The two
original tumor GP warnings remain in the instrument condition; the
unchanged first-mode warning also remains in the floor condition.
Neither old warnings nor iteration caps were silently converted into
strict convergence.

There are 300 primary target scores, including 50 reused target scores:
223 complete and 77 threshold-censored. The GP-IC phase contains 60
target runs: 48 complete and 12 censored. Thus 310 new primary/control
target integrations were performed; every incomplete target remains
visible and invalidates its applicable pooled segment.

Parent checks cover all 150 model records, every profile/conditional
regression, noise/prior/discrepancy invariants, all 360 point-target
metrics and IC routes, and all 30 information / 54 training-path
diagnostics. All 36 autonomous tumor point trajectories and six tumor
variational paths agree with independent matrix exponentials.
Eighteen additional fixed-operator numerical reproductions use an
independent manual polynomial RHS and stricter Radau integration;
maximum relative prediction difference is 2.90e-8.

Primary DOP853 uses rtol 1e-8, atol 1e-10, horizon/400 maximum step,
200,000 RHS calls and 120 seconds per target, with every input knot
and training endpoint split. No state clipping, survivor-only averages
or altered solver policy rescues a result. Shared arithmetic is float64;
actual thread/runtime settings are recorded. Conditional Gaussian
operator uncertainty at fixed state/hyperparameters is not a full
joint posterior; nonlinear local gains are not exact finite-error
transport or posterior probabilities.

## Artifacts and reproduction

Artifacts live under the existing session directory:

```text
BASE = /Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics
```

They are persistent session artifacts, not canonical benchmark outputs.
No old study was overwritten.

| Artifact relative to BASE | Content |
| --- | --- |
| `mechanism_protocol.json` | Predeclared factorial, optimizer and scope |
| `mechanism_tests/frozen_interventions.json` | All per-mode noise values and original prior/slack references |
| `mechanism_tests/fitting_release.json` | Producer release, 150 records and exact source hashes |
| `mechanism_tests/fitting_API.txt` | Cache, parameter, noise and fixed-curve contracts |
| `mechanism_tests/parent_jobs.json` | Hash-locked parent jobs and within-cell selections |
| `mechanism_tests/evaluation/primary/summary.json` | Every original-IC record, target and role pool |
| `mechanism_tests/evaluation/GP_initial/summary.json` | All 30 selected IC controls |
| `mechanism_tests/evaluation/diagnostics/summary.json` | Conditional information, source errors, defects and local gains |
| `mechanism_tests/readout/all_role_scores.csv` | Every start/control and applicable role, including IC controls |
| `mechanism_tests/readout/selected_and_control_scores.csv` | Complete reported factorial, including heldout heat |
| `mechanism_tests/readout/all_target_scores.json` | Every target's training/forecast/full/observation and variable scores |
| `mechanism_tests/readout/selected_curve_and_gain.csv` | All 30 selected curve/gain/termination rows |
| `mechanism_tests/parent_*_verification.json` | Independent freeze, model, prediction and diagnostic records |
| `mechanism_tests/independent_Radau/` | All 18 alternate-integrator reproductions |
| `mechanism_tests/study_release.json` | Final study accounting and integrity links |

Use the existing `prob_rom` environment with BLAS/OpenMP thread counts
set to one. `mechanism_models.load_prepared(case, noise, multiplier)`
loads exact saved maps without refitting. `mechanism_execute.py` separates
`primary`, `initial` and `diagnostics`; output directories are protected,
and these completed phases should not be rerun into existing paths.
The `mechanism_verify_*.py` scripts and `mechanism_model_checks.py`
contain the independent numerical controls. The final artifact manifest
records the actual scripts and results used.

The study stops at the declared matrix. It adds no treatment restriction,
input-aware kernel, weak-only sweep, full-feature covariance variant,
new data/POD fit, or production promotion. These are single-acquisition
mechanism probes, not repeated-seed reliability or uncertainty-coverage
claims.
