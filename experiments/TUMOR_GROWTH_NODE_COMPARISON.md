# Untreated tumor growth: matched Neural ODE comparison

## Main result

**One of the ten biology/acquisition-condition groups met both the provisional
next-month accuracy requirement and the matched Neural ODE superiority
requirement: faster-spreading growth with 40 observations and 1% noise.**

For that group, median first-month full-field error was **8.14% for the native
Bayesian method versus 14.97% for Neural ODE**, with native winning all three
matched acquisitions. Over the full two-month forecast, the corresponding
medians were 22.93% and 31.63%; the favorable short-horizon result should not be
presented as equally accurate long-horizon prediction.

The result is not universal. With the same good-data acquisition and nominal
growth, NODE won all three pairs: first-month medians were 4.15% for NODE versus
6.66% for native. No sparse/noisy group met both requirements. Some showed
relative native improvements but unacceptable absolute errors or severe
acquisition-dependent failures.

An important limitation of the sparse/noisy task is now explicit: **seven of
its eight groups have a median first-month POD projection floor already above
15%**. Even an oracle trajectory in those frozen four-dimensional subspaces
cannot meet the chosen absolute-accuracy target. This is partly a noisy-basis
representation problem shared by both methods, not exclusively a dynamics
learning problem. It does not explain away native censors or the extreme
finite outlier in the remaining results.

This comparison uses every one of the 30 existing acquisitions in the
[native tumor-application study](TUMOR_APPLICATION_STUDIES.md); no favorable
seed or new horizon was selected. The two synthetic biologies share one anatomy.
Three acquisition seeds are not three patients, and this is not a statistical
or clinical superiority claim.

## Provisional next-month point criteria

Training ends at day 60. For days (60,90], native median full-field relative RMS
error across all three acquisitions must be <=15%, every acquisition <=25%,
and all points available. The native median must be lower than NODE's and
native must win at least 2/3 matched pairs. Numerical flags prevent a qualified
superiority claim. These exploratory thresholds were frozen before NODE
comparisons; uncertainty is reported separately.

| Biology | Observations/noise | Native median% | Native maximum% | NODE median% | Native wins | Accurate | Better than NODE | Both |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nominal | 40 obs / 1% | 6.66 | 6.78 | 4.15 | 0/3 | Yes | No | No |
| Nominal | 16 obs / 10% | U | U | 19.12 | U | No | Unresolved | No |
| Nominal | 16 obs / 20% | U | U | 30.36 | U | No | Unresolved | No |
| Nominal | 8 obs / 10% | 19.58 | 20.75 | 21.58 | 3/3 | No | Yes | No |
| Nominal | 8 obs / 20% | 48.87 | 48.92 | 34.09 | 0/3 | No | No | No |
| Faster spreading | 40 obs / 1% | 8.14 | 8.64 | 14.97 | 3/3 | Yes | Yes | Yes |
| Faster spreading | 16 obs / 10% | 23.17 | 36889.33 | 27.06 | 2/3 | No | Yes | No |
| Faster spreading | 16 obs / 20% | U | U | U | U | No | Unresolved | No |
| Faster spreading | 8 obs / 10% | 25.33 | 26.18 | 40.93 | 3/3 | No | Yes | No |
| Faster spreading | 8 obs / 20% | 39.21 | 41.01 | U | U | No | Unresolved | No |

U means an unavailable complete-window comparison, not zero and not a score of
a surviving prefix. Missing NODE output never counts as an automatic native
win. An incomplete cohort has no all-three median.

## All acquisitions and forecast horizons

Errors are full-field percentages, including the affine decoder shift and
unrepresented field residual. Each retained query has equal weight; these are
not quadrature-weighted integrals. The 402-point prediction grid includes exact
60/90-day boundaries. Training is [5,60], first month (60,90], second month
(90,120], full forecast (60,120].

| Biology | Obs/noise | Seed | Native 30d% | NODE 30d% | Native second30d% | NODE second30d% | Native full60d% | NODE full60d% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nominal | 40 obs / 1% | 42 | 6.26 | 4.17 | 20.70 | 12.74 | 16.40 | 10.14 |
| Nominal | 16 obs / 10% | 42 | 15.82 | 19.12 | 25.33 | 28.34 | 21.94 | 24.98 |
| Nominal | 16 obs / 20% | 42 | 29.67 | 32.95 | 40.10 | 42.66 | 36.19 | 38.98 |
| Nominal | 8 obs / 10% | 42 | 20.75 | 27.64 | 30.07 | 35.85 | 26.65 | 32.74 |
| Nominal | 8 obs / 20% | 42 | 48.87 | 40.72 | 60.24 | 45.10 | 55.87 | 43.36 |
| Nominal | 40 obs / 1% | 43 | 6.78 | 4.02 | 23.12 | 12.35 | 18.29 | 9.83 |
| Nominal | 16 obs / 10% | 43 | U | 20.91 | U | 29.40 | U | 26.26 |
| Nominal | 16 obs / 20% | 43 | U | 29.28 | U | 42.02 | U | 37.33 |
| Nominal | 8 obs / 10% | 43 | 19.26 | 20.56 | 28.47 | 28.66 | 25.11 | 25.65 |
| Nominal | 8 obs / 20% | 43 | 48.92 | 34.09 | 60.21 | 40.01 | 55.87 | 37.70 |
| Nominal | 40 obs / 1% | 44 | 6.66 | 4.15 | 23.92 | 12.73 | 18.87 | 10.14 |
| Nominal | 16 obs / 10% | 44 | 15.04 | 17.19 | 24.53 | 26.28 | 21.17 | 23.00 |
| Nominal | 16 obs / 20% | 44 | 28.35 | 30.36 | 38.71 | 42.46 | 34.84 | 37.98 |
| Nominal | 8 obs / 10% | 44 | 19.58 | 21.58 | 28.83 | 29.35 | 25.45 | 26.45 |
| Nominal | 8 obs / 20% | 44 | 48.35 | 33.56 | 59.75 | 42.57 | 55.36 | 39.13 |
| Faster spreading | 40 obs / 1% | 42 | 7.88 | 14.97 | 27.37 | 38.28 | 22.18 | 31.63 |
| Faster spreading | 16 obs / 10% | 42 | 23.17 | 25.34 | 45.17 | 44.55 | 38.44 | 38.50 |
| Faster spreading | 16 obs / 20% | 42 | 32.75 | 36.67 | 52.35 | 50.63 | 46.00 | 45.91 |
| Faster spreading | 8 obs / 10% | 42 | 26.18 | 45.84 | 46.67 | 67.64 | 40.24 | 60.41 |
| Faster spreading | 8 obs / 20% | 42 | 41.01 | 41.15 | 58.35 | 54.78 | 52.54 | 50.11 |
| Faster spreading | 40 obs / 1% | 43 | 8.14 | 12.95 | 28.30 | 35.98 | 22.93 | 29.54 |
| Faster spreading | 16 obs / 10% | 43 | 36889.33 | 27.06 | 530882.45 | 46.43 | 420546.75 | 40.29 |
| Faster spreading | 16 obs / 20% | 43 | U | 34.47 | U | 49.35 | U | 44.37 |
| Faster spreading | 8 obs / 10% | 43 | 24.88 | 40.93 | 45.74 | 67.82 | 39.25 | 59.21 |
| Faster spreading | 8 obs / 20% | 43 | 38.66 | U | 56.54 | U | 50.59 | U |
| Faster spreading | 40 obs / 1% | 44 | 8.64 | 15.14 | 30.89 | 39.03 | 25.00 | 32.23 |
| Faster spreading | 16 obs / 10% | 44 | 22.27 | 28.56 | 44.36 | 52.70 | 37.65 | 45.20 |
| Faster spreading | 16 obs / 20% | 44 | U | U | U | U | U | U |
| Faster spreading | 8 obs / 10% | 44 | 25.33 | 33.06 | 45.98 | 51.25 | 39.53 | 45.30 |
| Faster spreading | 8 obs / 20% | 44 | 39.21 | 42.68 | 56.89 | 64.94 | 50.99 | 57.62 |

Full-window unavailable primary points: native 4/30; NODE 2/30. Extreme finite
errors remain numeric in the table. The native faster-growth
16-observation/10%-noise seed-43 outlier is not discarded or relabeled as a
solver failure.

## Representation and point-estimator diagnostics

The native primary is the conditional operator mean at arithmetic means of
positive GP hyperparameters and block scales, not a posterior predictive mean.
The strict-64 predictive median below is a fixed secondary estimator, not a
replacement chosen after seeing outcomes. NODE's primary is the strict
coordinatewise median of all 20 trajectories. Burden integrates normalized
cellularity over voxel volume, not a clinical cell count.

| Biology | Obs/noise | Seed | Native64 30d% | Native64 60d% | POD floor30d% | POD floor60d% | Native burden60d% | NODE burden60d% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nominal | 40 obs / 1% | 42 | 5.76 | 13.96 | 3.33 | 7.54 | 14.41 | 10.34 |
| Nominal | 16 obs / 10% | 42 | 15.77 | 21.88 | 15.21 | 21.26 | 10.25 | 14.24 |
| Nominal | 16 obs / 20% | 42 | 29.50 | 36.01 | 24.74 | 30.90 | 10.75 | 13.92 |
| Nominal | 8 obs / 10% | 42 | 20.60 | 26.53 | 19.42 | 25.29 | 11.95 | 15.52 |
| Nominal | 8 obs / 20% | 42 | U | U | 30.84 | 37.12 | 60.69 | 30.47 |
| Nominal | 40 obs / 1% | 43 | 6.25 | 15.89 | 3.29 | 7.61 | 14.78 | 9.88 |
| Nominal | 16 obs / 10% | 43 | U | U | 14.47 | 20.55 | U | 13.87 |
| Nominal | 16 obs / 20% | 43 | U | U | 23.58 | 29.69 | U | 13.49 |
| Nominal | 8 obs / 10% | 43 | 19.13 | 24.98 | 18.15 | 23.97 | 11.31 | 17.28 |
| Nominal | 8 obs / 20% | 43 | U | U | 29.28 | 35.44 | 60.65 | 27.95 |
| Nominal | 40 obs / 1% | 44 | 6.21 | 16.98 | 3.29 | 7.55 | 14.27 | 10.45 |
| Nominal | 16 obs / 10% | 44 | 14.99 | 21.10 | 14.50 | 20.54 | 9.93 | 14.86 |
| Nominal | 16 obs / 20% | 44 | 28.19 | 34.64 | 23.70 | 29.88 | 10.21 | 12.48 |
| Nominal | 8 obs / 10% | 44 | 19.45 | 25.33 | 18.48 | 24.33 | 11.76 | 13.97 |
| Nominal | 8 obs / 20% | 44 | U | U | 29.72 | 35.99 | 60.12 | 21.66 |
| Faster spreading | 40 obs / 1% | 42 | 7.77 | 21.80 | 6.66 | 18.69 | 15.76 | 36.07 |
| Faster spreading | 16 obs / 10% | 42 | 23.12 | 38.36 | 22.39 | 36.26 | 23.16 | 35.13 |
| Faster spreading | 16 obs / 20% | 42 | 32.65 | 45.95 | 30.81 | 43.46 | 23.62 | 39.82 |
| Faster spreading | 8 obs / 10% | 42 | 26.17 | 40.24 | 25.85 | 39.13 | 24.53 | 50.12 |
| Faster spreading | 8 obs / 20% | 42 | 41.09 | 52.69 | 37.10 | 48.43 | 27.34 | 37.01 |
| Faster spreading | 40 obs / 1% | 43 | 8.10 | 22.31 | 6.78 | 18.88 | 12.14 | 33.34 |
| Faster spreading | 16 obs / 10% | 43 | U | U | 22.08 | 36.40 | 49507.18 | 39.20 |
| Faster spreading | 16 obs / 20% | 43 | U | U | 29.64 | 42.64 | U | 36.86 |
| Faster spreading | 8 obs / 10% | 43 | 24.85 | 39.21 | 24.58 | 38.17 | 23.98 | 58.65 |
| Faster spreading | 8 obs / 20% | 43 | 38.63 | 50.62 | 35.15 | 46.76 | 25.59 | U |
| Faster spreading | 40 obs / 1% | 44 | 8.55 | 24.45 | 6.71 | 18.70 | 13.91 | 37.22 |
| Faster spreading | 16 obs / 10% | 44 | 22.25 | 37.61 | 21.86 | 36.02 | 22.76 | 47.99 |
| Faster spreading | 16 obs / 20% | 44 | U | U | 29.66 | 42.58 | U | U |
| Faster spreading | 8 obs / 10% | 44 | 25.30 | 39.51 | 25.02 | 38.52 | 24.28 | 44.66 |
| Faster spreading | 8 obs / 20% | 44 | 39.27 | 51.11 | 35.70 | 47.26 | 26.83 | 56.39 |

Independent voxel-level noise can be large in the field but nearly disappear
in total burden by spatial averaging. In the native acquisition audit, 20%
physical-generator noise produced at most 0.3571% relative RMS burden noise.
No claim of 20% burden-measurement noise is intended.

Median first-month projection floors for sparse/noisy groups are:

| Biology | 16 obs / 10% | 16 obs / 20% | 8 obs / 10% | 8 obs / 20% |
| --- | --- | --- | --- | --- |
| Nominal | 14.50% | 23.70% | 18.48% | 29.72% |
| Faster spreading | 22.08% | 29.66% | 25.02% | 35.70% |

Projection floors are computed directly from frozen target geometry,
independently of model availability. An initial report-rendering attempt
incorrectly assumed every censored point-score record contained that metadata;
the renderer was corrected without changing forecasts, errors, failures or
comparison criteria.

## Reliability and uncertainty, without a hidden point-performance gate

All 64 fixed native draw indices and all 20 NODE initializations remain in
denominators. Native draws include hyperparameter, conditional-operator and
original initial-condition uncertainty; NODE spread is initialization
variability. Empirical 5th/95th percentile burden bands are not demonstrated
calibrated posteriors. The inclusion columns require every expected member to
be finite at that query, so surviving-member bands cannot conceal missing tails.

| Biology | Obs/noise | Seed | Native complete60d | NODE complete60d | Native all-finite + burden truth in band | NODE all-finite + burden truth in band |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal | 40 obs / 1% | 42 | 64/64 | 20/20 | 25/210 | 16/210 |
| Nominal | 16 obs / 10% | 42 | 64/64 | 20/20 | 5/210 | 112/210 |
| Nominal | 16 obs / 20% | 42 | 64/64 | 20/20 | 36/210 | 158/210 |
| Nominal | 8 obs / 10% | 42 | 64/64 | 20/20 | 1/210 | 210/210 |
| Nominal | 8 obs / 20% | 42 | 63/64 | 20/20 | 0/210 | 210/210 |
| Nominal | 40 obs / 1% | 43 | 64/64 | 20/20 | 55/210 | 2/210 |
| Nominal | 16 obs / 10% | 43 | 7/64 | 20/20 | 0/210 | 36/210 |
| Nominal | 16 obs / 20% | 43 | 11/64 | 20/20 | 0/210 | 210/210 |
| Nominal | 8 obs / 10% | 43 | 64/64 | 20/20 | 8/210 | 154/210 |
| Nominal | 8 obs / 20% | 43 | 63/64 | 20/20 | 0/210 | 203/210 |
| Nominal | 40 obs / 1% | 44 | 64/64 | 20/20 | 41/210 | 4/210 |
| Nominal | 16 obs / 10% | 44 | 64/64 | 20/20 | 11/210 | 22/210 |
| Nominal | 16 obs / 20% | 44 | 64/64 | 20/20 | 46/210 | 210/210 |
| Nominal | 8 obs / 10% | 44 | 64/64 | 20/20 | 6/210 | 186/210 |
| Nominal | 8 obs / 20% | 44 | 63/64 | 20/20 | 0/210 | 210/210 |
| Faster spreading | 40 obs / 1% | 42 | 64/64 | 20/20 | 16/210 | 0/210 |
| Faster spreading | 16 obs / 10% | 42 | 64/64 | 20/20 | 0/210 | 5/210 |
| Faster spreading | 16 obs / 20% | 42 | 64/64 | 20/20 | 0/210 | 32/210 |
| Faster spreading | 8 obs / 10% | 42 | 64/64 | 20/20 | 0/210 | 154/210 |
| Faster spreading | 8 obs / 20% | 42 | 64/64 | 20/20 | 0/210 | 129/210 |
| Faster spreading | 40 obs / 1% | 43 | 64/64 | 20/20 | 18/210 | 2/210 |
| Faster spreading | 16 obs / 10% | 43 | 30/64 | 20/20 | 0/210 | 5/210 |
| Faster spreading | 16 obs / 20% | 43 | 0/64 | 20/20 | 4/210 | 6/210 |
| Faster spreading | 8 obs / 10% | 43 | 64/64 | 20/20 | 0/210 | 35/210 |
| Faster spreading | 8 obs / 20% | 43 | 64/64 | 19/20 | 0/210 | 0/210 |
| Faster spreading | 40 obs / 1% | 44 | 64/64 | 20/20 | 20/210 | 0/210 |
| Faster spreading | 16 obs / 10% | 44 | 64/64 | 20/20 | 0/210 | 6/210 |
| Faster spreading | 16 obs / 20% | 44 | 1/64 | 19/20 | 0/210 | 0/210 |
| Faster spreading | 8 obs / 10% | 44 | 64/64 | 20/20 | 0/210 | 128/210 |
| Faster spreading | 8 obs / 20% | 44 | 64/64 | 20/20 | 0/210 | 45/210 |

Finite-only quantiles and band counts are preserved in machine-readable tables
with availability counts. A numerical safety censor does not establish
mathematical finite-time blow-up. Negative reduced coordinates alone are not
negative tumor cellularity; neither trajectories nor extreme predictions were
clipped. Point accuracy should not be conflated with reliable uncertainty.

## Matched inputs and fixed numerical recipes

Each biology/seed/condition has the exact previously frozen noisy reduced
observations, four-mode training-only POD decoder, day-5 observed initial state
and forecast grid. No full-order solve, basis refit or native refit was added.
Nominal biology uses k=.025,d=.05,theta=1; faster spreading uses
k=.05,d=.1,theta=1. Observation/noise conditions are
40/1%,16/10%,16/20%,8/10%,8/20%, each at seeds 42,43,44. This is untreated
extrapolation, not treatment-response identification.

Native cA inference retains 20 operator coefficients, 200 GP/constraint nodes,
combined derivative/IBP weights 1/8, diagonal covariance, `gamma2=.035`,
`mll_weight=.1`, spectrum/cadence priors, zero-mean block hierarchy with
`sigma_O=5`, the jittered normal-equation backend, AutoNormal/ClippedAdam at
.003, 12,000 SVI updates and seed 42. **There is no ODE solve in the inference
objective.** Of 500 saved positive-hyperparameter/operator draws,
`linspace(0,499,64,dtype=int)` is evaluated with original paired IC draws.

NODE retains the original 4-128-128-128-4 tanh architecture (34,180 parameters),
all 20 original seed-42 keys, 3,000 Adam updates at .001, no normalization,
regularization changes, early stopping, best-checkpoint choice or
outcome-based restarts. The model capacities and uncertainty constructions
differ; this is not an isolated Bayesian-versus-non-Bayesian treatment effect.

Adaptive Tsit5 with `rtol=1e-5,atol=1e-7,dt0=min(first_gap,55/200)`,
`max_steps=16384` and the original checkpoint adjoint replaces the legacy
constant first-gap integration. This is an explicitly qualified numerical
adaptation, not literally unchanged legacy optimization. A proposed vmap update
was rejected after float32 Adam differences; the scalar-map/scan kernel passed
the frozen update controls.

Parent evaluation independently implements the MLP and Jacobian in NumPy
float64. DOP853 uses `rtol=1e-8,atol=1e-10,max_step=115/400` and splits at 60/90
days. Fixed seed-42 members 0 and 19 across all ten conditions are checked
against Radau at `1e-9/1e-11,max_step=115/800`, with relative trajectory
discrepancy <=2e-6. NODE trainer float32 versus independent float64 training
trajectories must agree within 1e-3 relative norm. Safety limits are 1e6 times
frozen training RMS, 200,000 RHS calls and 120 seconds; unavailable tails remain
unavailable.

Final accounting: 30 ensembles, **598/600 completed members**, 1,799,512
successful of 1,800,000 nominal updates; 20/20 fixed NODE Radau controls passed;
two parent training-qualification flags. The separate export audit checks all
2,400 checkpoints and 1,800,000 saved loss slots.

The two NODE failures are retained. Faster spreading, seed 43, 8 observations/
20% noise, member 3 stopped at failed update 2,639 with a nonfinite post-update
state. Faster spreading, seed 44, 16 observations/20% noise, member 8 stopped
at failed update 2,875 with a nonfinite gradient/post-update state. Their
strict-20 medians remain unavailable. The two parent flags represent these
unavailable fits, not discrepant independent trajectories from completed fits.
The separate NODE trainer float32 scalar-replay flag is the second member's
failed final solve.

## Execution interruption and provenance

Execution stopped after 21 completed ensembles and during the next ensemble.
Completed fits and evaluations were never overwritten. The interrupted
ensemble restored all 20 model/Adam/counter states and the full loss/status
prefix at step 2,000. Recomputing steps 1,000-2,000 from the prior checkpoint
matched every saved state and trace array bitwise before continuation. Those
20,000 repeated hardware-control updates are not new scientific optimization
steps. The eight untouched datasets used the original trainer. Unknown unlogged
work after the last durable checkpoint is not silently counted or interpreted
as a model failure.

The repository retains recovery commit
`0f5e4ba0b358469656efd18489e78bd3c0dee4f1` and backup branch
`backup/growth-node-before-comparison-0f5e4ba`. Production inference, canonical
caches, the prior report and manuscript remain unchanged. This is a local
research record, not a push or model promotion. Numerical artifacts are
persistent local session data, not a portable dataset included here.

Artifact root:
`/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics/node_growth_comparison`.

Machine-readable tables: `reporting/report_tables.json`; exact all-acquisition
readout: `readouts/all_thirty_ensembles.json`.

| Evidence | SHA256 |
| --- | --- |
| Frozen protocol | `927af636c78236cc0d3dd5cfc68bc99bf4cf1ea88a6a3e7308ac60663bf849ec` |
| Training release | `7a168e413c2f5b67e535b71e3e688d8d35c83ee002e1251d53375c005d0fe788` |
| Parent evaluation release | `6c6d5c3575241880437d88ce90b8dc99b0815a54931d6f8c46dc636ffd8db819` |
| Complete readout | `96ac40515f23ff04b3d25e565d06d69992a3511cddf7b3ef77e43a30761848de` |
| Full training audit, `20260918T043719_full_release.json` | `c09901353c7c85ec6c650e80b51e622c20d716503e486060a1acd40bdf2e476c` |
| Machine-readable report tables | `fc68570e72c6ff783da85e486b4db385d188b63e53bdc5f04bc0d401a4a6b580` |
