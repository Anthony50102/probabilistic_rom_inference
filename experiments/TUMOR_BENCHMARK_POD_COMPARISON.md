# Three tumor benchmarks and bounded chemotherapy POD selection

## Status and scope

**Complete: the development-selected representation passed every frozen
fresh-acquisition screen.** It uses four uncentered POD modes fitted only to
clean half-exposure training states. On fresh acquisitions 48-50, production
had median full-field forecast errors of 6.17-6.64% at all four future
strengths, versus 8.66-37.24% for the two prescribed Neural ODE (NODE)
medians. It won every seed at every strength against both medians, for both
source references. Its incremental treatment effects had median errors of
12.63-13.98% and median day-110 gains of 0.90-0.95. NODE predicted almost no
response to the changed future strength: its effect errors were 89-98%, with
filtered-median gains between -0.016 and 0.020.

The Bayesian production algorithm is unchanged. This study changes the shared
spatial representation, not the operator family, inference objective, solver,
precision, optimizer, or scientific training budget. It does not promote the
experimental shared-latent algorithm.

These are fresh acquisitions of one synthetic anatomy and one set of true dose
trajectories, not new patients. The clean-training basis is privileged
benchmark information and was chosen using production development
performance. Production draw bands are not calibrated uncertainty: on
acquisition 50 they missed truth at most burden queries for three of the four
strengths.

## The three reporting cases

| Case | Task and data | Representation | Existing evidence |
|---|---|---|---|
| `untreated-growth` | Faster-spreading untreated tumor (`k=.05,d=.1`); 40 observations with 1% noise on days 5-60; headline forecast days 60-90. | Four mean-centered modes from noisy training observations. | Production median field error 8.14%, NODE 14.97%; production wins all three acquisitions. |
| `single-dose-chemo` | Continue one fixed half-strength regimen; 120 observations with 1% noise on days 5-70; forecast days 70-110. | Preserve the successful nominal-training, mean-centered rank-4 basis. | Production median 5.39%, NODE filtered median 9.93%; production wins all three acquisitions. |
| `multi-dose-chemo` | Preserve the half-strength history; change only the two future pulse strengths; same chemo acquisition and forecast window. | Matched-training, uncentered rank 4, selected on development acquisition 45 and then frozen. | Fresh acquisitions 48-50: production medians 6.17-6.64% versus NODE 8.66-37.24%; production wins every seed at all four strengths. |

Here **single-dose means one fixed strength, not one administration**. Both
chemotherapy cases retain pulses on days 20, 40, 60, 80, and 100. Untreated
growth means no treatment, not removal of the logistic carrying-capacity term.
The public recipe inspector is `experiments/tumor/benchmark_cases.py`; it does
not train models or change historical entry-point defaults. In this release its
defaults remain the preserved controls, keeping the audited protocol source
byte-identical. The confirmed multi-dose representation is requested with
`--pod-rank 4 --pod-source matched_training --pod-centering none`.

The [untreated-growth record](TUMOR_GROWTH_NODE_COMPARISON.md) retains all
contrary evidence: nominal good-data growth favors NODE (4.15% versus 6.66%);
no sparse/noisy group met both goals; and the favorable faster-growth
two-month errors rise to 22.93% versus 31.63%. The
[fixed-strength record](TUMOR_CHEMO_TASK_COMPARISON.md) retains the inaccurate
original-exposure control. These are selected synthetic examples, not universal
production superiority. No earlier experimental-model advantage is erased.

## Dose-transfer task and shared data

All chemo runs use the same 646,812-voxel demo anatomy, `k=.025,d=.05`, and
carrying capacity 1. Half exposure is effective sensitivity 0.25 instead of
the original 0.5, with unchanged pulse dates and decay. It is not half as many
observations, half the forecast horizon, or half the resulting shrinkage.
It is a modeled benchmark intervention, not a clinical recommendation.

Past pulses on days 20, 40, and 60 have effective coefficient 0.25. Only the
day-80/day-100 pulses change:

| Future strength relative to original exposure | Effective coefficient | Interpretation |
|---|---:|---|
| 0.5x | 0.25 | Unchanged half-exposure control |
| 0.25x | 0.125 | Lower future pulse strength |
| 0.75x | 0.375 | Future pulse strength above the training amplitude |
| 1x | 0.5 | Original-exposure-strength future pulses |

Earlier drug tails are preserved, not globally rescaled after day 70.
There are no new full-order solves: the study reuses every primary and refined
branch in the [previous dose-switch study](HALF_EXPOSURE_DOSE_SWITCH.md).
The original same-regimen restart controls reproduced archived voxels bitwise.
The primary RK4 step is 0.25 day, with 0.125-day source-step sensitivity
references; neither is claimed to be continuum truth. The native RK4 endpoint
treatment at hard dose onsets is preserved, not silently corrected.

Each acquisition has 120 sorted uniform observation times on days 5-70, with
endpoints forced. Noise is 1% of the clean training range, applied to the
original active-voxel mask, with the first observation exact and no clipping.
All candidates receive the same underlying physical noise innovations,
observation times, input table, and history. Production and NODE share the
same reduced observations, initial state, affine decoder, and forcing.

Matched PODs use the 120 clean half-exposure snapshots at development
acquisition 45's training times. Cubic interpolation uses source knots only
through day 70. No future fields, alternate treatment histories, or fresh
confirmation observations enter the POD fit. Method-of-snapshots eigenvectors,
a fixed physical sign convention, and QR orthonormalization define the basis.
The selected basis is frozen and reused, not refitted for each fresh seed.

Clean training states are privileged simulation-benchmark information. This
is not a demonstration that a clinical noisy-POD pipeline can learn the same
subspace. Choosing a common representation using production performance also
does not compare against an independently best-representation-tuned NODE.

## Bounded development comparison

Acquisition 45 and all four dose outcomes had already been investigated.
They are explicitly **development data**, never described as an untouched test.
The five candidates, ranking rule, numerical controls, budgets, and fresh
acquisition seeds 48/49/50 were frozen before the new fits. No additional
candidate, favorable horizon, dose omission, or rescue budget is allowed.

Every candidate first receives the same production fit and four hyperpoint
forecasts. The two lowest mean full-field errors are shortlisted, requiring
complete and numerically qualified forecasts at all four strengths.
Only those two representations receive matched NODE comparisons. The old
nominal native fit and NODE ensemble are reused rather than retrained.

### All five production candidates

Full-field relative RMS percentages over days (70,110], including the decoder
shift and unrepresented spatial residual:

| POD candidate | 0.5x control | 0.25x | 0.75x | 1x | Mean across four arms |
|---|---:|---:|---:|---:|---:|
| nominal_mean_r4 | 5.39 | 6.27 | 8.76 | 19.73 | 10.04 |
| matched_mean_r2 | 10.22 | 8.94 | 26.33 | 59.57 | 26.26 |
| matched_mean_r4 | 28.35 | 13.44 | 31.73 | 18.52 | 23.01 |
| matched_mean_r6 | 25.03 | 12.81 | 23.68 | 21.98 | 20.88 |
| matched_none_r4 | 7.35 | 6.18 | 8.95 | 10.21 | 8.17 |

The corresponding oracle projection floors are:

| POD candidate | 0.5x control floor | 0.25x floor | 0.75x floor | 1x floor |
|---|---:|---:|---:|---:|
| nominal_mean_r4 | 4.77 | 5.30 | 4.37 | 4.08 |
| matched_mean_r2 | 6.91 | 8.62 | 5.52 | 4.75 |
| matched_mean_r4 | 1.54 | 1.83 | 1.31 | 1.13 |
| matched_mean_r6 | 0.64 | 0.84 | 0.51 | 0.43 |
| matched_none_r4 | 1.55 | 1.79 | 1.37 | 1.23 |

Better projection alone did not produce better learned dynamics. Centered
rank 6 had the smallest floors but substantially worse production forecasts.
The uncentered four-mode representation had the lowest aggregate production
error, not the smallest spatial projection residual.

### Both shortlisted NODE comparisons

The original training-loss-filtered median and strict all-20 median are the
two prescribed comparison policies. The all-20 mean is retained as a
secondary sensitivity result.

| POD candidate | Future strength | Production | NODE filtered median | NODE all20 median | NODE all20 mean |
|---|---|---:|---:|---:|---:|
| matched_none_r4 | 0.5x | 7.35 | 11.02 | 11.02 | 10.26 |
| matched_none_r4 | 0.25x | 6.18 | 22.72 | 22.72 | 21.97 |
| matched_none_r4 | 0.75x | 8.95 | 16.56 | 16.56 | 15.85 |
| matched_none_r4 | 1x | 10.21 | 32.00 | 32.00 | 30.68 |
| nominal_mean_r4 | 0.5x | 5.39 | 9.90 | 9.96 | 9.53 |
| nominal_mean_r4 | 0.25x | 6.27 | 16.15 | 16.17 | 14.21 |
| nominal_mean_r4 | 0.75x | 8.76 | 23.00 | 22.99 | 16.17 |
| nominal_mean_r4 | 1x | 19.73 | 23.99 | 23.82 | 20.01 |

The selected development NODE ensemble kept all 20 members under the original
filter; the reused nominal ensemble kept 18. No member is removed for a poor
future forecast.

A shortlist candidate gets the requested production-over-NODE preference only
if its production mean is within 10% of the best, every production error is
<=25%, and it beats **both** NODE medians by a mean paired error ratio <=0.90
with at least three wins in four dose cells. Remaining ties prefer lower
production mean, lower rank, then the declared order. Without such a candidate,
the rule selects the lowest production mean and discloses no confirmed
advantage; it does not expand the search.

**Frozen selection: `matched_none_r4`.** It wins all four development pairs
against both NODE medians, with mean paired ratio 0.449711. The nominal control
is outside the 10%-of-best production band. The selected candidate also wins
on the primary production mean without the NODE preference.

Uncentering does not remove the constant or additive input operator. The
original `cABN` family is retained. Representation changes can alter
spectrum-derived priors and neural-network conditioning; this is not a causal
isolation of centering alone.

## Unchanged training and prospective confirmation rules

Production retains 200 GP/constraint nodes, derivative and integration-by-parts
weights 1 and 8, `gamma2=.035`, `mll_weight=.1`, zero-mean hierarchical
block priors, `sigma_O=5`, spectrum/cadence-derived priors, diagonal covariance
blocks, and the original normal-equation backend. AutoNormal/ClippedAdam uses
learning rate .003, seed 42, default float32, and 12,000 updates. The 500
operator/hyperparameter draws and original GP initial-condition rule are
unchanged. No ODE solve is added to Bayesian fitting.

The production point is the conditional operator mean evaluated at arithmetic
means of positive hyperparameters/block scales, not a posterior predictive mean.
The selected 64 operator/IC draw identities are fixed and paired across doses.

NODE retains `(r+1)->128->128->128->r`, with tanh after the first three layers;
only input/output dimensions follow the POD rank. All 20 original keys receive
6,000 scalar Adam updates, learning rate .0005, clipping 1, and float32.
The original adaptive Tsit5 tolerances, nine force-aware cuts, .25-day maximum
step, and 16,384-step budget are unchanged. The last recorded **pre-update**
loss defines the original three-times-median filter before any new forecast.
Full model/optimizer/trace checkpoints support exact resume, not a scientific
restart or best-checkpoint selection.

Confirmation uses fresh observation-time/noise seeds 48/49/50 after the
selection freeze. These are new acquisitions of the **same anatomy, physics,
and true dose trajectories**, not new patients or new biological realizations.
The basis is not reselected from their results.

Per future strength, production must have median full-field error <=15%, each
seed <=25%, and all three complete. Against both NODE medians it must have
median paired error ratio <=0.90 and at least two wins. Undefined, zero-error,
or numerically unqualified NODE scores are not automatic production wins.
Unlike development, confirmation uses the **median**, not mean, paired ratio.

Treatment effects subtract the unchanged 0.5x control over days (80,110].
The native field-effect error must have median <=50%, each seed <=75%;
all three endpoint directions must be correct; endpoint gain must have median
in [0.5,1.5] and every seed in [0.25,1.75]. Both NODE superiority requirements
also apply. Zero predicted response has 100% effect error, not perfect
stability. Effects of point policies are differences of policy centers.

Field scores equally weight the 152 retained forecast queries of the
400-point days-5-to-110 grid. Effect scores use 114 post-day-80 queries.
No failed trajectory is scored only on its surviving prefix. NumPy64 DOP853
forecasts retain all input-knot cuts and strict Radau controls, with unchanged
state, RHS-call, and wall-time budgets. Missing slots and failures remain in
the denominator.

The reliability guard (at least 58/64 complete draws and 32/64 with field error
<=25%) is separate from the point criteria. Empirical 5-95% burden/effect band
inclusion is not calibrated uncertainty. Negative reduced coordinates are not
physical negativity; actual decoded fields require separate checks.

## Fresh-acquisition confirmation

Every result below was produced after the selection freeze, with no rescue,
refit, omitted dose, or reselection. `primary` scores use the 0.25-day RK4
source references; `refined` scores rescore the identical forecasts against
the 0.125-day source-step sensitivity references. Screen columns apply the
frozen per-strength rules above; the 0.5x control has no effect screen.

| Reference | Future strength | Production median | NODE filtered median | NODE all20 median | NODE all20 mean median | Point screen | Effect screen |
|---|---|---:|---:|---:|---:|---|---|
| primary | 0.5x | 6.17 | 9.53 | 8.66 | 8.44 | Pass | n/a |
| primary | 0.25x | 6.64 | 22.08 | 17.35 | 18.52 | Pass | Pass |
| primary | 0.75x | 6.62 | 14.10 | 20.86 | 18.06 | Pass | Pass |
| primary | 1x | 6.37 | 29.39 | 37.24 | 33.67 | Pass | Pass |
| refined | 0.5x | 7.18 | 10.35 | 8.66 | 8.49 | Pass | n/a |
| refined | 0.25x | 6.73 | 22.76 | 18.07 | 19.23 | Pass | Pass |
| refined | 0.75x | 7.76 | 13.60 | 19.81 | 17.09 | Pass | Pass |
| refined | 1x | 7.56 | 28.05 | 35.76 | 32.25 | Pass | Pass |

On the primary reference, median paired production/NODE field-error ratios
against the filtered/all20 medians were 0.529/0.533 (0.5x), 0.301/0.409
(0.25x), 0.305/0.317 (0.75x), and 0.165/0.171 (1x); effect ratios were
0.129-0.147. Refined-reference ratios were 0.204-0.582 for fields and
0.132-0.149 for effects. The frozen cutoff is 0.90.

### Every fresh primary point result

Full-field relative RMS percentages over days (70,110]. The original
training-loss filter kept 17, 18, and 19 of 20 NODE members for acquisitions
48, 49, and 50, respectively.

| Future strength | Acquisition | Production | NODE filtered median | NODE all20 median | NODE all20 mean |
|---|---|---:|---:|---:|---:|
| 0.5x | 48 | 4.19 | 9.53 | 8.66 | 7.91 |
| 0.5x | 49 | 6.17 | 11.67 | 11.59 | 10.60 |
| 0.5x | 50 | 6.95 | 8.54 | 8.41 | 8.44 |
| 0.25x | 48 | 6.64 | 22.08 | 16.24 | 18.52 |
| 0.25x | 49 | 6.01 | 23.19 | 23.15 | 22.85 |
| 0.25x | 50 | 7.41 | 17.16 | 17.35 | 16.04 |
| 0.75x | 48 | 3.42 | 13.87 | 21.44 | 18.06 |
| 0.75x | 49 | 6.70 | 14.10 | 14.43 | 14.42 |
| 0.75x | 50 | 6.62 | 21.66 | 20.86 | 21.71 |
| 1x | 48 | 3.92 | 29.39 | 37.37 | 33.67 |
| 1x | 49 | 7.17 | 26.73 | 27.49 | 28.55 |
| 1x | 50 | 6.37 | 38.54 | 37.24 | 37.93 |

### Fresh treatment effects

Effects are paired differences from the unchanged 0.5x control over days
(80,110]. Gain is predicted divided by true day-110 burden change: 1 is exact
and 0 is no predicted response.

| Future strength | Acquisition | Production effect field % | NODE filtered effect field % | NODE all20 median effect field % | NODE all20 mean effect field % | Production day-110 gain | NODE filtered day-110 gain |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.25x | 48 | 21.39 | 92.87 | 94.23 | 94.01 | 1.064 | -0.016 |
| 0.25x | 49 | 13.72 | 93.26 | 93.49 | 95.25 | 0.954 | -0.003 |
| 0.25x | 50 | 13.98 | 97.36 | 96.54 | 95.06 | 0.859 | -0.009 |
| 0.75x | 48 | 16.20 | 97.37 | 92.13 | 93.27 | 1.016 | 0.012 |
| 0.75x | 49 | 10.59 | 89.63 | 91.69 | 94.53 | 0.916 | 0.002 |
| 0.75x | 50 | 12.85 | 98.13 | 95.95 | 93.54 | 0.826 | -0.013 |
| 1x | 48 | 14.46 | 97.77 | 91.41 | 92.80 | 1.001 | 0.010 |
| 1x | 49 | 9.71 | 89.03 | 90.87 | 93.75 | 0.904 | 0.020 |
| 1x | 50 | 12.63 | 97.63 | 95.40 | 92.92 | 0.818 | -0.013 |

All nine production endpoint directions were correct. Production gains were
0.818-1.064; the two smallest were acquisition 50 at 0.75x and 1x.

### Production draw reliability

These use the fixed 64 paired operator/IC draws and the primary reference.
Direction counts use the day-110 burden effect. Band columns count queries
whose empirical 5-95% band over complete draws contains truth.

| Future strength | Acquisition | Complete draws | Complete with field <=25% | Correct day-110 effect direction | Burden truth in band | Effect truth in band |
|---|---|---|---|---|---|---|
| 0.5x | 48 | 64/64 | 64/64 | n/a | 152/152 | n/a |
| 0.5x | 49 | 64/64 | 64/64 | n/a | 152/152 | n/a |
| 0.5x | 50 | 64/64 | 64/64 | n/a | 1/152 | n/a |
| 0.25x | 48 | 64/64 | 64/64 | 64/64 | 152/152 | 112/114 |
| 0.25x | 49 | 64/64 | 64/64 | 64/64 | 152/152 | 109/114 |
| 0.25x | 50 | 64/64 | 64/64 | 64/64 | 1/152 | 27/114 |
| 0.75x | 48 | 64/64 | 64/64 | 64/64 | 152/152 | 112/114 |
| 0.75x | 49 | 64/64 | 64/64 | 64/64 | 152/152 | 109/114 |
| 0.75x | 50 | 64/64 | 64/64 | 64/64 | 47/152 | 1/114 |
| 1x | 48 | 64/64 | 64/64 | 64/64 | 151/152 | 113/114 |
| 1x | 49 | 64/64 | 64/64 | 64/64 | 152/152 | 111/114 |
| 1x | 50 | 64/64 | 64/64 | 64/64 | 113/152 | 1/114 |

Every draw completed with field error <=25%, so the frozen reliability guard
passed in all 12 cells, and every draw had the correct effect direction. The
bands are nevertheless not calibrated. A post-hoc descriptive check, not a
screen, found the draw-median burden roughly 6.5-9.3% below truth over the
forecast window on acquisitions 49 and 50. Acquisition 49's wider spread
still covered truth. Acquisition 50's median 5-95% half-width was only about
6-9% of true burden, so its bands missed most burden queries at three
strengths and most effect queries at all three changed strengths. The filtered
NODE member bands also missed often, containing truth at 6-116 of 152
queries. This is stability plus point accuracy, not reliable uncertainty.

### Decoded point-field bounds

Values are minimum and maximum normalized cellularity over all 12 fresh
point forecasts and every forecast query. The final column is the largest
negative-field mass as a percentage of true burden.

| Point policy | Minimum normalized cellularity | Maximum normalized cellularity | Maximum negative mass % of true burden |
|---|---:|---:|---:|
| native | 0.000000 | 0.569712 | 0.0000 |
| filtered_median | -0.000000 | 0.435773 | 0.0000 |
| all20_median | -0.000000 | 0.430950 | 0.0000 |
| all20_mean | -0.000000 | 0.428412 | 0.0000 |

Every point policy decoded all 152 forecast queries in all 12 cells.
Production point fields had no negative voxel values and peaked at 0.570 of
carrying capacity. NODE `-0.000000` entries are round-off negatives of order
1e-18. No policy had materially negative (<-1e-6) or above-capacity values.
These point diagnostics do not certify every draw or untested regimens.

The figure `figures/fresh_burden_and_effects.png` in the artifact namespace
overlays production, filtered-median NODE, and PDE burden trajectories and
incremental effects for the three changed strengths.

## Interpretation

- **Lower-dose transfer and upward extrapolation both passed.** At 0.25x the
  future pulses are weaker than in training, but remain within the scalar
  input range already visited while training pulses decay. The 0.75x and 1x
  pulses have 1.5 and 2 times the training coefficient. Production passed
  point and effect screens at all three. Median gains declined modestly with
  amplitude (0.95, 0.92, 0.90): a mild underestimate of larger responses, not
  a direction error.
- **The earlier nominal-basis limitations did not recur on fresh data, but
  this is not a paired basis ablation.** With the nominal basis on
  acquisitions 45-47, the [earlier dose-switch study](HALF_EXPOSURE_DOSE_SWITCH.md)
  passed 0.75x trajectory accuracy while predicting about half the effect
  (median gain 0.498), and missed the 1x accuracy target (median 23.86%).
  Acquisitions 48-50 were run only with the selected basis. The only paired
  representation comparison is development acquisition 45 (1x: nominal
  19.73%, selected 10.21%), which was used for selection.
- **NODE did not transfer the dose change.** Each NODE ensemble is trained on
  one input history, with no variation in pulse amplitude. Its forecasts
  respond only transiently to the altered pulses and return near the
  unchanged control. Production's `cABN` input and state-input operators can
  carry an amplitude-dependent response. This explanation is interpretive: no
  input-structured or amplitude-varied NODE was evaluated.
- **Fixed-strength forecasting also holds under the shared basis.** The 0.5x
  arm is the unchanged half-strength continuation: production 4.19/6.17/6.95%
  versus NODE filtered 9.53/11.67/8.54% on acquisitions 48-50. It
  supplements, and is not paired with, the single-dose record's nominal-basis
  acquisitions 45-47 (5.39% versus 9.93% median). On the only paired
  acquisition (45), the nominal basis was better for this arm (5.39% versus
  7.35%), so the single-dose recipe keeps the nominal basis.
- **No sign of selection overfitting in this bounded study.** Fresh
  production medians (6.17-6.64%) were comparable to or better than
  development acquisition 45 (6.18-10.21%); only 0.25x was slightly higher
  (6.64% versus 6.18%). Three acquisitions of one anatomy remain limited
  evidence.

## Accounting, audits, and execution

The final inventory has 1,200 primary trajectory slots, all integrated
successfully: 20 development production points and 160 development NODE
member forecasts, then 12 fresh production points, 768 fixed-draw forecasts,
and 240 fresh NODE member forecasts. All 128 strict Radau controls passed
(maximum relative difference 3.41e-9; maximum field-error change 6.2e-7
percentage points).

New training comprised 7 production fits (84,000 SVI updates, all losses
finite) and 4 NODE ensembles (80 members; 480,000 attempted and successful
updates). The nominal production fit and its 20 NODE members were reused.
There were no new full-order solves or algorithm settings.

Independent audits reconstructed the fresh observations and noise, and
rebuilt each production point operator and three fixed operator draws
bitwise. They checked every new NODE member's initial and final checkpoint,
trace, loss, and weight files. They also replayed each completed member's
final update bitwise; those 80 replay updates are discarded hardware checks,
not training. The audits recomputed 952 score rows per fresh acquisition and
56 per shortlisted development candidate. The seal re-derives every screen,
paired effect, and band count from the per-acquisition summaries.

**Execution interruption.** The original attached orchestrator most likely
ended with its CLI session at about 17:47 on 2026-09-23, while NODE member 4
for acquisitions 48 and 49 was between checkpoints. No NODE confirmation
forecast existed yet, and acquisition 50 had not started. The interruption
record preserves the original log's hash. The same frozen stage then resumed
under a new log from the verified step-4,000 checkpoints: model, Adam state,
pre-update state, counters, and loss prefix. Completed fits and evaluations
were reused, not rerun, and no member was restarted or replaced. The
checkpoint preflight had verified nonzero-Adam-state reload with bitwise
next-step replay; an uninterrupted copy of member 4 was not trained.

## Limitations

- The fresh acquisitions share one synthetic anatomy, physics, and set of
  true trajectories; only observation times and noise change. They are not
  new patients or biological realizations, and three is a small number.
- The clean-training POD is privileged simulation information, chosen using
  production development performance. NODE shared that representation but
  was not independently representation-tuned or redesigned for input
  extrapolation.
- Uncentering also changes spectrum-derived priors and NODE conditioning; the
  study does not isolate a causal effect of centering alone.
- The production point is the conditional operator mean at mean
  hyperparameters, not a posterior predictive mean. Draw bands are not
  calibrated uncertainty.
- The references keep the native RK4 dose-onset treatment; the refined
  reference is a source-step sensitivity check, not continuum truth.
- The untreated and fixed-strength cases rely on their own records, including
  their unfavorable results.

## Recommended three-case reporting set

| Case | Configuration to report | Headline evidence | Report alongside it |
|---|---|---|---|
| `untreated-growth` | Faster-spreading example (`k=.05,d=.1`); 40 observations, 1% noise, days 5-60; observed-training mean-centered rank 4; forecast days 60-90. | Median field error 8.14% production versus 14.97% NODE; production wins all three acquisitions. | Nominal good-data growth favors NODE (4.15% versus 6.66%); two-month errors rise to 22.93% versus 31.63%. |
| `single-dose-chemo` | Fixed half-strength regimen; 120 observations, 1% noise, days 5-70; preserved nominal-training mean-centered rank 4; forecast days 70-110. | Median 5.39% production versus 9.93% NODE filtered; production wins all three acquisitions. | The original-exposure control was inaccurate; bands are not calibrated. |
| `multi-dose-chemo` | Same history; future strengths 0.25x-1x; matched-training uncentered rank 4 (`--pod-rank 4 --pod-source matched_training --pod-centering none`). | Fresh medians 6.17-6.64% production versus 8.66-37.24% NODE; effect errors 12.63-13.98% versus 89-98%. | Clean-training basis selected on development data; same anatomy and truths; acquisition-50 bands missed truth. |

## Provenance

The committed recovery point is
`452b3e6727aea312ed68255b9c03f1f3acb7265c`, also retained as
`backup/tumor-three-benchmarks-before-pod-452b3e6`. Production code, historical
defaults, canonical caches, manuscript, and earlier reports are not replaced.

The owned artifact namespace is:

```text
/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics/tumor_benchmark_pod
```

The protocol SHA256 is
`ab55a14c937c4c082d725899946430ce17c741ca75df9a696c8d9932f2b9129d`.
The selection SHA256 is
`1420fe32b1de9385e7abc6d49bd268c55fa889c902bad636a7f007bd0d4e6dac`.
Both are pinned before confirmation data generation. Numerical data and
session-local `tumor_pod_*.py` drivers are not bundled in Git. The sealed
release, `study_release.json` in that namespace, records this record's SHA256,
the four repository-file hashes, the artifact manifest, the protocol and
selection identities, and the accounting above.
