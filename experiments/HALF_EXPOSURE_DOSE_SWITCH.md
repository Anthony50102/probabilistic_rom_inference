# Half-exposure future-dose transfer

## Verdict and reportable scope

The unchanged production algorithm supports useful **local future-dose
transfer**, but this is not a general solution to dose extrapolation.
All fits used the same half-exposure history through day 70. Only the strengths
of the future day-80 and day-100 pulses changed.

- **0.25x future strength:** a convincing lower-dose transfer result. Production
  is accurate, beats both prescribed Neural ODE medians on all three acquisition
  seeds, and captures the direction and much of the dose-change magnitude.
  This is an unseen pulse strength, but not extrapolation beyond the scalar
  input range already visited during decay of the training pulses.
- **0.75x future strength:** useful, limited upward extrapolation. Production
  meets the trajectory-accuracy and NODE-superiority criteria, but predicts only
  about half the lasting treatment effect. It narrowly fails the additional
  prospectively specified endpoint-magnitude screen; do not present it as
  accurate quantitative treatment-response prediction.
- **1x future strength:** retain as a limitation/stress test, not a successful
  accurate-forecast example. Production beats the prescribed NODE medians, but
  its median field error is about 24%, above the frozen 15% accuracy target.
  Treatment-effect magnitude is also substantially underestimated.

Keep known-regimen 0.5x continuation as the main chemo result: its previously
reported median field errors are **5.39% production versus 9.93% NODE**.
Use the complete dose sweep as a secondary result, distinguishing the successful
lower-dose transfer, limited upward extrapolation, and full-dose limitation.
Do not omit the unfavorable dose or imply calibrated treatment-effect uncertainty.

This study establishes performance under the half-exposure setup. It does not
contain a matched full-exposure-training future-switch experiment, so it cannot
quantify how much changing the training exposure itself improved generalization.

## Frozen design

The protocol was persisted before generating new sources or model forecasts.
Recovery commit `c39714379af95cf40ea6989f9f89b125d8a352e4` is retained at
`backup/dose-switch-before-testing-c397143`. No production, default, canonical
cache, prior research record, or manuscript changes were made.

The source is the same TNBC demo anatomy and native TumorTwin reaction-diffusion
model as [the continuation comparison](TUMOR_CHEMO_TASK_COMPARISON.md):
`k=.025`, `d=.05`, carrying capacity 1, base sensitivity `.5`, and drug decay `.7`.
Strengths are relative to the original modeled exposure, not clinical dose advice.

| Pulse days | Training/control strength | Changed future strengths | Effective pulse coefficients |
|---|---|---|---|
| 20, 40, 60 | 0.5x | Unchanged in every arm | 0.25 |
| 80, 100 | 0.5x | 0.25x, 0.75x, 1x | 0.125, 0.375, 0.5 |

Day 70 is the decision/forecast cutoff; the first changed pulse is on day 80.
The residual drug from the three earlier pulses is **not** rescaled. Separate
past/future chemotherapy specifications with unit dose arrays and the stated
sensitivities avoid TumorTwin's dose-array normalization changing the past.

All three existing M acquisitions, seeds 45/46/47, were reused: 120 observations
over days 5-70, 1% masked voxel-level noise, and the same archived clean four-mode
basis and affine decoder. The privileged clean basis remains a limitation.
These are dense synthetic acquisitions on one anatomy, not three patients and
not a sparse/noisy clinical acquisition study.

There were **zero new production, NODE, or POD fits and zero new training
updates**. Production used each original hyperpoint, plus the same 64 operator
draws at `linspace(0,499,64,dtype=int)` with their original paired GP IC draws.
The hyperpoint is not the posterior predictive mean. Each NODE used its original
20 final models and original initializations.

The original NODE policy remains the coordinatewise trajectory median of members
whose last recorded **pre-update** training loss is at most three times the
20-member median. Seed 45 retains 18/20, rejecting members 1 and 16; seeds 46 and
47 retain 20/20. No future-dependent filtering was allowed. Strict all-20 median
and all-20 mean outputs, every rejected member, and every expected slot remain
in the artifacts. Centers are formed in reduced coordinates before decoding.

Learned trajectories start from their original observed day-5 IC, or the paired
native GP IC draw. They are **not** restarted from the true day-70 PDE state.
All 765 learned trajectories' training predictions are bitwise equal to their
archived unchanged-dose counterparts.

## Metrics and prospective criteria

Forecast errors use all 646,812 voxels on the original 400-query day-5-to-110
grid, with equal retained-query weights and the affine decoder shift included.
The main window is `70 < t <= 110` (152 queries):

`field error = 100 * ||decoded_prediction - reference|| / ||reference||`.

Dose effects are the paired changed arm minus unchanged 0.5x control, measured
on `80 < t <= 110` (114 queries). Their full-field error includes the
unrepresented spatial residual; the decoder shift cancels. Point effects are
differences of each policy's centers, not medians of paired member effects.
Integrated normalized cellularity at 1 mm^3 voxel volume is called burden;
it is not a clinical cell count.

The original trajectory screen requires all three points complete and qualified,
median native field error <=15%, every seed <=25%, and, against **both** NODE
medians, median paired native/NODE error ratio <=0.90 with at least two wins.
Missing or zero-error NODE baselines do not count as automatic native wins.

The additional prospective dose-effect screen requires median field-effect error
<=50%, every seed <=75%, correct endpoint burden direction on all three seeds,
median endpoint gain in [0.5,1.5], every gain in [0.25,1.75], and the same
paired superiority rule against both NODE medians. Gain means predicted divided
by true signed burden change. These are pragmatic research screens, not clinical
thresholds. Tiny/undefined effects cannot count as successful responses.
An exactly zero-response predictor has 100% effect error.

### Primary median results

All numbers below are percentages except the final screen. "Full dose screen"
includes trajectory accuracy, dose-effect direction/magnitude, both NODE
comparisons, and preservation of the conclusion under source refinement.

| Future strength | Production field | NODE filtered field | NODE all20 field | Production effect field | NODE filtered effect field | Full dose screen |
|---|---:|---:|---:|---:|---:|---|
| 0.25x | 6.27 | 15.48 | 15.48 | 15.55 | 95.34 | Pass |
| 0.75x | 11.59 | 23.00 | 22.99 | 49.29 | 91.05 | Fail |
| 1x | 23.86 | 37.26 | 37.26 | 59.71 | 85.16 | Fail |

The 0.75x failure is **not** a trajectory failure. Its unrounded median endpoint
gain is `0.4983087762`, just below the frozen 0.5 cutoff; the finer-reference gain
is `0.4960785709`. Its effect-field screen and both NODE comparisons pass.
The exact cutoff is not a scientific cliff: the substantive finding is a
consistent roughly twofold underestimation of the lasting effect, with gains
0.519, 0.498, and 0.389 across the three seeds.

### Every primary point result

| Future strength | Seed | Production field % | NODE filtered field % | NODE all20 field % | Production burden % | NODE filtered burden % |
|---|---:|---:|---:|---:|---:|---:|
| 0.25x | 45 | 6.27 | 16.15 | 16.17 | 6.11 | 19.58 |
| 0.25x | 46 | 6.18 | 12.96 | 12.96 | 4.41 | 15.93 |
| 0.25x | 47 | 7.57 | 15.48 | 15.48 | 8.68 | 18.50 |
| 0.75x | 45 | 8.76 | 23.00 | 22.99 | 5.71 | 18.04 |
| 0.75x | 46 | 11.59 | 24.97 | 24.97 | 7.00 | 20.51 |
| 0.75x | 47 | 11.77 | 22.24 | 22.24 | 8.13 | 18.22 |
| 1x | 45 | 19.73 | 23.99 | 23.82 | 15.04 | 19.40 |
| 1x | 46 | 23.86 | 42.19 | 42.19 | 17.62 | 38.01 |
| 1x | 47 | 24.29 | 37.26 | 37.26 | 19.53 | 33.08 |

Production wins all nine field-error pairs against both prescribed NODE medians.
This does **not** mean every NODE aggregation loses. At 1x, the secondary all-20
mean has median field/effect errors 23.51%/51.16%, versus production
23.86%/59.71%. Neither achieves the absolute trajectory target there. The
predeclared original median policy is retained rather than selecting an
aggregation after seeing the forecasts.

### Treatment-effect direction and magnitude

| Future strength | Seed | Production effect-field % | NODE filtered effect-field % | NODE all20 effect-field % | Production endpoint gain | NODE filtered endpoint gain |
|---|---:|---:|---:|---:|---:|---:|
| 0.25x | 45 | 15.55 | 96.54 | 96.48 | 0.84 | -0.01 |
| 0.25x | 46 | 15.18 | 90.57 | 90.57 | 0.81 | 0.04 |
| 0.25x | 47 | 30.55 | 95.34 | 95.34 | 0.60 | -0.01 |
| 0.75x | 45 | 46.82 | 92.06 | 91.64 | 0.52 | 0.03 |
| 0.75x | 46 | 49.29 | 91.05 | 91.05 | 0.50 | 0.04 |
| 0.75x | 47 | 55.97 | 85.85 | 85.85 | 0.39 | 0.09 |
| 1x | 45 | 57.26 | 53.52 | 53.02 | 0.44 | 0.18 |
| 1x | 46 | 59.71 | 92.59 | 92.59 | 0.42 | 0.03 |
| 1x | 47 | 64.17 | 85.16 | 85.16 | 0.34 | 0.04 |

Every native point has the correct burden-change direction at all 114 effect
queries. The stronger-dose problem is magnitude, not a sign reversal:

- At 0.25x, true endpoint burden rises **38.41%** relative to unchanged treatment;
  median predicted changes are **+31.16% native** and **-0.21% NODE filtered**.
- At 0.75x, true endpoint burden falls **28.30%**; predicted changes are
  **-14.10% native** and **-1.00% NODE filtered**.
- At 1x, true endpoint burden falls **48.90%**; predicted changes are
  **-20.66% native** and **-1.96% NODE filtered**.

These percentages describe an incremental effect relative to the paired
unchanged regimen, not shrinkage relative to initial burden.

The truth-level predictor that simply reuses the unchanged treatment trajectory
has field-forecast errors 17.46%, 18.25%, and 36.38% in the three arms.
Production improves on these no-response references and on reusing its own
unchanged-regimen prediction in every seed. NODE's lasting median response is
usually much weaker, although at 1x/seed45 its full-window effect error is lower
than native's. No universal treatment-metric win is claimed.

## Source and integration qualification

Eight new native PDE solves were performed: two same-regimen restart controls,
then six changed-dose branches (three strengths at steps 0.25 and 0.125 day).
Each starts at the corresponding archived half-exposure day-70 state and saves
through day 120 at 0.5-day intervals. Source branches are shared across the
three acquisitions, not independently regenerated per seed.

TumorTwin's `Solver.solve` normally resets its time origin. The study instead
uses native `torchdiffeq.odeint` with absolute times 70-120, the original model
clock, and the native grid constructor, RHS, boundary/mask callback, and RK4
implementation. Both unchanged-dose restarts reproduce **every saved voxel
bitwise**. All changed arms also match the old source at every saved time before
day 80. No native simulator behavior was changed.

The archived 4001-knot forcing table and linear interpolation were retained,
adding only the change in future pulse contributions. Training inputs are
bitwise unchanged. The sub-knot ramp around day 80 starts at 79.99625 and ends
at 80.0225. The native hard-onset RK4 endpoint-anticipation behavior remains;
it was not silently repaired or confused with model quality.

References reuse the old cubic reference exactly at prediction times before day
80. From day 80 onward they use the original full-grid cubic interpolation of
the joined old-past/new-future source. This explicitly suppresses retrospective
cubic influence from changed future knots without changing any observations,
fit, or unchanged-control score.

| Future strength | Primary/refined forecast-field discrepancy % | Effect-field discrepancy % | Endpoint effect discrepancy % | True endpoint burden change % |
|---|---:|---:|---:|---:|
| 0.25x | 0.9884 | 0.2442 | 0.0658 | 38.41 |
| 0.75x | 1.3145 | 0.3515 | 0.4496 | -28.30 |
| 1x | 1.4119 | 0.4514 | 0.6189 | -48.90 |

All three reference pairs pass the prospectively specified 5% trajectory / 10%
effect / 10% endpoint-effect discrepancy gates. The finer sources are a
sensitivity analysis, not a claim of continuum truth.

Rescoring the **identical learned trajectories** against the finer references:

| Future strength | Production field % | NODE filtered field % | NODE all20 field % | Production effect-field % | NODE filtered effect-field % | Full dose screen |
|---|---:|---:|---:|---:|---:|---|
| 0.25x | 6.82 | 16.21 | 16.21 | 15.63 | 95.34 | Pass |
| 0.75x | 10.67 | 21.94 | 21.94 | 49.41 | 91.06 | Fail |
| 1x | 22.59 | 35.84 | 35.84 | 59.85 | 85.21 | Fail |

All **765/765** primary trajectories complete: nine native hyperpoints,
576 paired native draws, and 180 NODE member trajectories. All **63/63** fixed
Radau controls pass. The controls are the native point, native positions
0/31/63 (original indices 0/245/499), and NODE members 0/9/19 in each cell.
The largest full-trajectory solver discrepancy is `2.51e-9` relative; the
forecast-only maximum is `4.34e-9`. The largest field-error change is
`1.48e-7` percentage points (rounded upward).

The same all-input-knot DOP853 and stricter Radau evaluators, safety threshold,
time/RHS budgets, unclipped predictions, and failure-preserving denominators
as the continuation study were used. No censors, numerical failures, replacements,
or surviving-prefix success scores occurred in this new matrix.

## Reliability and uncertainty are separate

All 576 native draw forecasts complete. Every primary cell passes the existing
reliability guard (>=58/64 complete and >=32/64 with field error <=25%).
This is not evidence of calibrated intervals.

| Future strength | Seed | Complete native draws | Native field error <=25% | Correct endpoint effect direction | True effect in native 5-95% band | True burden in native 5-95% band |
|---|---:|---:|---:|---:|---:|---:|
| 0.25x | 45 | 64/64 | 64/64 | 64/64 | 47/114 | 91/152 |
| 0.25x | 46 | 64/64 | 64/64 | 64/64 | 75/114 | 100/152 |
| 0.25x | 47 | 64/64 | 64/64 | 64/64 | 1/114 | 0/152 |
| 0.75x | 45 | 64/64 | 64/64 | 64/64 | 1/114 | 105/152 |
| 0.75x | 46 | 64/64 | 64/64 | 64/64 | 1/114 | 102/152 |
| 0.75x | 47 | 64/64 | 64/64 | 64/64 | 1/114 | 40/152 |
| 1x | 45 | 64/64 | 57/64 | 64/64 | 1/114 | 56/152 |
| 1x | 46 | 64/64 | 39/64 | 64/64 | 1/114 | 33/152 |
| 1x | 47 | 64/64 | 34/64 | 64/64 | 1/114 | 3/152 |

The dose-effect bands use matched operator/IC draws across treatments, not
independent draws subtracted afterward. Their near-total exclusion of the true
effect in the upward-dose arms is an important failure despite stable solves.
Even the successful lower-dose point experiment has poor interval inclusion on
seed 47. **Do not claim calibrated predictive or treatment-effect uncertainty.**

Actual decoded point fields, rather than signs of POD coordinates, were checked.
Across all nine cells, the worst bounds for each primary point policy are:

| Point policy | Minimum cellularity | Maximum cellularity | Maximum negative mass / true burden % |
|---|---:|---:|---:|
| native_point | -0.000000 | 0.594842 | 0.0000 |
| NODE_filtered_median | -0.000000 | 0.471172 | 0.0000 |
| NODE_all20_median | -0.000000 | 0.471172 | 0.0000 |

Signed zeros above are rounded tiny negatives. Native minimum is approximately
`-5.14e-14`, and filtered NODE minimum `-7.52e-10`, both below the material
negative tolerance of `1e-6`. No material point-field negativity or above-capacity
values occurred. These point checks do not assert bounds for every ensemble draw.

## Audit and artifacts

Independent checks reconstruct all source and effect geometry using a separate
cubic-spline calculation over every voxel/query, rederive the original
pre-update NODE filters, verify the 500-to-64 operator/IC mapping, check all
exported training histories, strict centers, paired effects and bands, and
recompute **11,076** score rows. Forecast and source audits pass. Prior release
artifacts and source identities are rechecked at sealing.

The new local artifact root is:

```text
/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics/half_exposure_dose_switch
```

Important artifacts:

- `protocol.json`, SHA256
  `1d6109f846725db2a5faa331176d1336dbac19d415813621002af91afce70665`.
- `sources/`, `inputs/`, and `geometry/`: all branch sources, restart controls,
  input tables, and primary/refined full-field/effect reference geometry.
- `evaluation/<arm>/<run>/`: every trajectory, fixed Radau control, original
  member identity, point/ensemble score, physical bound, and paired-effect band.
- `analysis/summary.json` and `analysis/all_point_results.csv`: exact unrounded
  results, all point policies, source-sensitive screens, and endpoint effects.
- `figures/paired_burden_and_dose_effects.png`: all three acquisition seeds,
  both prescribed methods, true burden, and incremental treatment effects.
- `audit/`, `final_checks.json`, `study_artifact_manifest.json`, and
  `study_release.json`: independent checks, report-table verification, source
  snapshots, prior-artifact preservation, and final hashes.

The session-local scripts `dose_switch_{common,sources,geometry,evaluate,analysis,
audit,seal}.py` retain the complete study implementation; run artifacts are not
bundled in Git. Execution used the existing `prob_rom` environment with
single-threaded numerical libraries. The repository change is this research
record only, committed locally without a push.
