# Tumor benchmarks on segmented noisy scans

## Status and scope

**Complete, with a mixed outcome.** Every scan is now noisy and every basis
is fitted to noisy scans. Under this design production (Bayesian OpInf) is
more accurate than the NODE in two benchmarks:

- untreated growth: 7.33% against 12.33% median over days 60-90, lower on
  every acquisition;
- multi-dose chemo: 9.63-12.69% against 11.36-26.65% median at the four
  future strengths, lower on 11 of the 12 acquisition-strength pairs (not on
  the unchanged 0.5x arm of acquisition 50: 13.02% against 11.64%). The
  frozen development-selected basis passed every pre-declared confirmation
  check.

Production's median day-110 gains are 0.79-0.80, so it recovers most of the
true treatment effect, while the NODE predicts almost none (gains within
0.03 of zero).

The single-dose continuation is not a production win. Its medians are
18.90% against 11.15%, and the NODE is lower on two of three acquisitions.
The same unchanged-regimen forecast is also the multi-dose 0.5x arm. Over all
six reported chemo acquisitions it gives production 4.10-19.69% (median
12.36%) against NODE 9.75-14.15% (median 12.62%).

**Operator prior.** All values use the nondimensional operator prior and the
round-off GP nugget (`sigma_O=None`, `gp_jitter_rel=None`, the
`WeakFormConfig` defaults), which replaced the hand-set `sigma_O=5.0` and
`gp_jitter_rel=1e-3` here and the corresponding settings in every other
experiment. The replacement was chosen because it has no per-experiment
constant, but only after an ablation that included these acquisitions, so
the choice was not blind to them. The task design and the chemo basis were
fixed under the earlier prior. Values marked "earlier prior" come from it;
the full earlier record is at git tag `pre-universal-prior`.

**Standardized settings.** All production values below are from the rerun
under the settings shared by every experiment (README): unit block weights,
one optimizer schedule and grid, the QR operator solve, and the dof-tempered
dynamics rows (`closure="tempered"`), which have no closure constant. The
chemo fits use the Galerkin structure (`cAN` on the uncentered basis). The
task design, the bases, and the NODE ensembles did not change. These settings
were adopted after the reported acquisitions had been scored twice:

- under the earlier, per-experiment settings (git tag
  `pre-standard-settings`): 7.45% untreated growth, 11.20% single-dose, and
  5.60-7.08% multi-dose, with the growth draw band containing the true
  burden at every forecast time;
- under the standardized settings with a closure variance whose constant,
  `gamma2_nd = 0.1` for the tumor tasks, was chosen on development
  acquisitions 45-47 (git tag `pre-closure-tempering`): 7.32%, 18.99%, and
  8.23-8.84%, with burden bands containing the truth at 7-9% of the growth
  forecast times, 0-21% of the single-dose ones, and 3-100% of the
  multi-dose ones.

The tempering was selected among constant-free rules on development data
only, by criteria fixed before any of them was fitted (README), and then run
once here. Against the closure constant it leaves untreated growth and
single-dose within 0.1 points, raises the multi-dose error, and widens the
chemo bands (below). The `production_config` of each result still records
`gamma2_nd = 0.1`, the configuration default, which only `closure="slack"`
uses. No method or configuration was changed after the rerun was scored.

The three reported tumor benchmarks now observe the simulated tumor through
noisy scans from which the lesion is segmented (`--observation segmented`, the
default of `benchmark_cases.py` and the 04/05/06 runners). This replaces the
earlier oracle-masked design of the
[three-benchmark POD record](TUMOR_BENCHMARK_POD_COMPARISON.md), in which
noise was added only where the true tumor was present, the initial state was
noise-free, and the chemo PODs were fitted to clean simulated snapshots. That
design remains available with `--observation oracle_masked` and reproduces its
sealed results.

Apart from the operator prior, nugget, and standardized settings above, the
production Bayesian algorithm and the Neural ODE (NODE) recipe are those of
the earlier design. Only the data, the POD bases fitted to them, and
the POD ranks changed. All acquisitions share one synthetic anatomy and one
set of true trajectories; only the scan times and noise differ.

## Observation model

For every scan, the first included (both methods start from it):

1. **Measurement.** Independent Gaussian noise is added to every breast-tissue
   voxel (`geometry["breast_mask"]`). Its standard deviation is 1% of the range
   of the noise-free training fields: 0.0095 for untreated growth and 0.0070
   for chemo.
2. **Segmentation from the noisy scan alone.** The scan is smoothed by a 1 mm
   Gaussian and thresholded at z times the smoothed noise level. z = 5.21 is
   the Bonferroni value that holds the chance of any false-positive voxel in a
   scan at 5% over the tissue voxels.
3. **Reported cellularity.** As in TumorTwin, the noisy value is clipped to
   [0, 1] inside the segmented lesion and set to zero elsewhere.

The zero background therefore comes from segmenting each scan, not from
knowing where the true tumor is. Evaluation-only checks over the nine reported
acquisitions (840 scans) are listed below. `data/metadata.json` records them
for every acquisition.

| Check | Untreated growth (42-44) | Chemo (48-53) |
|---|---|---|
| Lesion voxels per scan | 34,165-94,407 | 32,644-57,558 |
| False-positive voxels, all scans | 0 | 3 |
| Field energy missed (faint margin) | 0.285-0.290% | 0.269-0.297% |
| Scan error against the truth | 2.25-2.27% | 2.58-2.68% |

## POD and the noise-rank guard

Each acquisition's POD is fitted to its own segmented training scans (source
`observed_training`). No clean, future, or other-acquisition fields are used.
`benchmark_data.noise_threshold` gives the Gavish-Donoho optimal hard
threshold `lambda*(n/N) sqrt(N) sigma` for the singular values of the scan
matrix, with n the number of scans, N the mean number of segmented voxels per
scan, and sigma the known voxel noise (Gavish and Donoho, IEEE Trans. Inf.
Theory 60(8), 2014). Preparation refuses a declared rank above the number of
singular values that exceed it, because those modes cannot be told apart
from noise.

| Case | Leading singular values (acquisition 42 or 48) | Threshold | Modes admitted | Retained / threshold | First rejected / threshold |
|---|---|---|---|---|---|
| Untreated growth | 176.9, 26.7, 5.10, 2.90, 2.86 | 3.30-3.37 | 3 | 1.40-1.55 | 0.85-0.88 |
| Chemo | 602.7, 56.7, 10.9, 2.99, 1.70 | 2.12-2.14 | 4 | 1.39-1.52 | 0.79-0.80 |

## How the configurations were fixed

**Untreated growth: mean-centered rank 3.** The earlier design's mean-centered
rank 4 was fitted first. On acquisition 42 its fourth singular value lies in
the noise bulk. The GP interpolated that mode, the learned operator acquired a
growing eigenvalue (+0.76 per day), and the forecast diverged
(`threshold_censored`). Refitted under the nondimensional prior (+1.0 per
day) and again under the standardized settings with the closure constant
(+1.1 per day), the point forecast and all 64 draws still diverged. With the
tempered rows it no longer diverges: the largest eigenvalue is +0.005 per
day and the error 7.85% over days 60-90 (6.99% at rank 3), although 83% of
the draws have an eigenvalue with positive real part. On 43 and 44, rank 4
gave 7.35% and 7.44% over days 60-90 (earlier prior; not refitted). This
failure is why the rank guard was added. The guard admits three modes on
every growth acquisition, and all three acquisitions report rank 3.

**Chemo: uncentered rank 4.** This was selected on development acquisition 45
among four candidates by the lowest mean production error over the four
future strengths, then frozen before any reported chemo acquisition was
fitted. The NODE was not consulted. The selection used the earlier prior:

| Candidate on acquisition 45 | 0.25x | 0.5x | 0.75x | 1x | Mean |
|---|---|---|---|---|---|
| **Uncentered rank 4 (selected)** | 6.21 | 7.04 | 8.51 | 9.73 | **7.87** |
| Uncentered rank 3 | 7.72 | 8.83 | 10.27 | 11.39 | 9.55 |
| Mean-centered rank 4 | 14.31 | 29.88 | 34.28 | 19.26 | 24.44 |
| Mean-centered rank 3 | 16.88 | 34.77 | 46.52 | 42.03 | 35.05 |

The shortlist was uncentered ranks 4 and 3. Stage 2 of the pre-declared
rule prefers a shortlisted candidate within 10% of the best production mean
that also clears a NODE margin. It could not change the choice, because only
the selected candidate was within 10% (uncentered rank 3 was 21% worse), so
the rank-3 NODE was never trained. The selected basis's development NODE
comparison finished after the freeze and passes every stage-2 check:

- NODE filtered medians of 21.79 / 9.84 / 15.03 / 30.59% at 0.25x / 0.5x /
  0.75x / 1x (all-member 23.39 / 10.29 / 14.74 / 29.99%);
- mean paired ratios of 0.47 (filtered) and 0.46 (all-member), with 4/4 wins
  against each;
- 19 of 20 members kept.

Repeated under the current prior, the same rule selects the same basis:

| Candidate on acquisition 45, current prior | 0.25x | 0.5x | 0.75x | 1x | Mean |
|---|---|---|---|---|---|
| **Uncentered rank 4 (selected)** | 5.97 | 6.90 | 8.31 | 9.42 | **7.65** |
| Uncentered rank 3 | 7.79 | 9.27 | 10.83 | 11.96 | 9.96 |
| Mean-centered rank 4 | 13.27 | 26.60 | 26.50 | 19.19 | 21.39 |
| Mean-centered rank 3 | 15.74 | 30.95 | 37.30 | 27.05 | 27.76 |

Uncentered rank 3 is 30% worse than the selected basis, and against the same
NODE ensembles the selected basis has mean paired ratios of 0.46 (filtered)
and 0.45 (all-member), with 4/4 wins against each.

Under the standardized settings with the closure constant the rule again selected it:

| Candidate on acquisition 45, standardized settings | 0.25x | 0.5x | 0.75x | 1x | Mean |
|---|---|---|---|---|---|
| **Uncentered rank 4 (selected)** | 4.06 | 3.67 | 3.84 | 4.11 | **3.92** |
| Uncentered rank 3 | 4.83 | 4.22 | 4.08 | 4.08 | 4.30 |
| Mean-centered rank 4 | 9.27 | 20.03 | 101.98 | 345.65 | 119.23 |
| Mean-centered rank 3 | 8.32 | 4.29 | 54.96 | 230.52 | 74.52 |

Uncentered rank 3 is now within 10% of the best (9.8% worse), so stage 2
applies to both shortlisted bases. The selected basis clears the NODE
preference (mean paired ratios 0.24 filtered and 0.23 all-member, 4/4 wins
against each), and remaining ties go to the lower production mean, so the
rank-3 NODE ensemble, which was not trained, could not change the choice. The
mean-centered bases, which keep the `B` block (`cABN`), fail at the 0.75x and
1x doses (55-346%).

With the tempered rows (the current settings) the rule selects uncentered
rank 3 instead:

| Candidate on acquisition 45, tempered rows | 0.25x | 0.5x | 0.75x | 1x | Mean |
|---|---|---|---|---|---|
| Uncentered rank 4 (frozen, reported) | 5.01 | 4.95 | 5.21 | 5.56 | 5.18 |
| **Uncentered rank 3 (rule's choice)** | 5.30 | 4.95 | 4.94 | 5.07 | **5.06** |
| Mean-centered rank 4 | 6.38 | 16.99 | 94.21 | 312.24 | 107.45 |
| Mean-centered rank 3 | 8.83 | 4.14 | 64.41 | 253.64 | 82.76 |

Both uncentered bases are within 10% of the best, so stage 2 needed the
rank-3 NODE ensemble, which was trained for it (19 of 20 members kept;
filtered medians 13.89 / 12.54 / 27.03 / 43.38% at 0.25x / 0.5x / 0.75x /
1x). Both bases pass every stage-2 check (mean paired ratios 0.27 for rank 3
and 0.32 for rank 4 against the filtered NODE, 0.27 and 0.31 against the
all-member one, 4/4 wins each), and the tie goes to the lower production
mean, rank 3, by 2%. The reported results keep the rank-4 basis frozen
before 48-50 were fitted. That decision was made before rank 3 was scored on
the reported acquisitions; scored afterwards as a sensitivity check (point
forecasts only, `--pod-rank 3 --no-draws`), rank 3 gives:

| Rank 3, reported acquisitions | 0.25x | 0.5x | 0.75x | 1x |
|---|---|---|---|---|
| 48 / 49 / 50 (multi-dose) | 5.30 / 14.75 / 15.93 | 4.66 / 13.73 / 14.02 | 4.94 / 12.57 / 12.23 | 5.34 / 11.42 / 10.75 |
| 51 / 52 / 53 (single-dose) | | 18.86 / 21.09 / 10.55 | | |

Its multi-dose medians, 14.75 / 13.73 / 12.23 / 10.75%, are above rank 4's
12.69 / 11.70 / 10.63 / 9.63%, and its single-dose median is 18.86% against
18.90%.

**Single-dose: the same basis on fresh acquisitions 51-53.** The single-dose
task was first declared with the mean-centered rank-4 basis on acquisitions
45-47, where it failed (29.88%, 120.59%, and 33.25%; under the nondimensional
prior 26.60%, 104.22%, and 33.78%; under the standardized settings with the
closure constant 20.03%, 39.98%, and 9.78%). With the tempered rows it gives
16.99%, 4.43%, and 3.46%, but the task had moved before. A single-dose forecast is
the unchanged 0.5x arm of the multi-dose forecast on the same acquisition,
with identical data, basis, settings, and fit. The task therefore adopted the
selected chemo basis. It is reported on fresh acquisitions 51-53, because 45
was the development acquisition and 48-50 are the multi-dose confirmation.

## Reported results

Relative full-field error (%), including the POD residual. The NODE center is
the all-member median for growth and the loss-filtered median for chemo, as
in the earlier studies. The floor is the error of the truth's orthogonal
projection onto the basis.

**Untreated growth** (acquisitions 42 / 43 / 44)

| Window | Production | NODE all-member | NODE filtered | Floor (median) |
|---|---|---|---|---|
| Training, days 5-60 | 0.76 / 0.75 / 0.80 | 0.75 / 0.75 / 0.77 | 0.75 / 0.75 / 0.77 | 0.73 |
| Days 60-90 | 6.99 / 7.33 / 7.75 | 12.33 / 14.82 / 11.72 | 13.92 / 14.82 / 11.65 | 6.53 |
| Days 60-120 | 20.25 / 21.09 / 22.71 | 28.01 / 31.84 / 27.42 | 29.69 / 31.84 / 27.26 | 18.45 |

NODE kept 16, 18, and 17 members after its loss filter.

**Single-dose chemo** (acquisitions 51 / 52 / 53), days 70-110

| Window | Production | NODE filtered | NODE all-member | Floor (median) |
|---|---|---|---|---|
| Training, days 5-70 | 3.41 / 3.95 / 2.19 | 1.31 / 1.34 / 1.87 | 1.31 / 1.34 / 1.90 | 0.51 |
| Days 70-110 | 18.90 / 19.69 / 9.70 | 11.15 / 9.75 / 13.61 | 11.15 / 9.75 / 14.27 | 2.46 |

The NODE kept 20, 20, and 19 members, and production wins only on 53.

**Multi-dose chemo** (acquisitions 48 / 49 / 50), days 70-110

| Future pulses | Production | NODE filtered | NODE all-member | Floor (median) |
|---|---|---|---|---|
| Training, days 5-70 | 4.74 / 2.60 / 2.45 | 1.48 / 1.32 / 1.41 | 1.46 / 1.32 / 1.41 | 0.51 |
| 0.25x | 4.97 / 12.69 / 15.07 | 28.02 / 26.65 / 24.10 | 24.98 / 26.65 / 24.10 | 2.85 |
| 0.5x (unchanged) | 4.10 / 11.70 / 13.02 | 14.15 / 13.87 / 11.64 | 11.76 / 13.87 / 11.64 | 2.54 |
| 0.75x | 4.32 / 10.63 / 11.17 | 10.66 / 11.36 / 13.84 | 12.06 / 11.36 / 13.84 | 2.30 |
| 1x | 4.76 / 9.63 / 9.71 | 23.97 / 23.57 / 27.59 | 26.01 / 23.57 / 27.59 | 2.10 |

Production wins every seed at every strength against both centers except the
unchanged 0.5x arm on acquisition 50. The NODE kept 19, 20, and 20 members.
Every frozen confirmation check passes at every strength:

- production median at most 15% and every seed at most 25% (observed at most
  12.69% and 15.07%);
- median paired ratio at most 0.90 against both NODE centers (observed
  0.35-0.84);
- at least two of three wins (observed 3/3 at the changed strengths and 2/3
  for the unchanged arm).

The treatment effect is the change from the unchanged arm after day 80. The
true day-110 changes are +38.4%, -28.3%, and -48.9% of the control burden.
Gain is the predicted day-110 change divided by the true one.

| Effect versus 0.5x | Production error | Production gain | NODE filtered error | NODE gain |
|---|---|---|---|---|
| 0.25x | 15.51 / 17.16 / 24.91 | 1.10 / 0.80 / 0.71 | 98.71 / 94.75 / 95.98 | -0.03 / 0.01 / -0.02 |
| 0.75x | 10.41 / 17.39 / 23.87 | 1.04 / 0.79 / 0.71 | 100.52 / 95.03 / 94.97 | -0.02 / 0.01 / -0.02 |
| 1x | 8.81 / 17.30 / 23.25 | 1.02 / 0.79 / 0.72 | 100.03 / 94.86 / 94.52 | -0.00 / 0.02 / -0.03 |

## The unchanged regimen on every chemo acquisition

The unchanged half-strength forecast was made on seven acquisitions: 45
(development), 48-50 (the multi-dose 0.5x arm), and 51-53 (single-dose). The
day-110 column is the predicted over the true tumor burden.

| Acquisition | Role | Production | NODE filtered | NODE all-member | Floor | Day-110 burden |
|---|---|---|---|---|---|---|
| 45 | development | 4.95 | 9.84 | 10.29 | 2.65 | 1.01 |
| 48 | multi-dose | 4.10 | 14.15 | 11.76 | 2.55 | 0.96 |
| 49 | multi-dose | 11.70 | 13.87 | 13.87 | 2.54 | 0.81 |
| 50 | multi-dose | 13.02 | 11.64 | 11.64 | 2.53 | 0.79 |
| 51 | single-dose | 18.90 | 11.15 | 11.15 | 2.46 | 0.70 |
| 52 | single-dose | 19.69 | 9.75 | 9.75 | 2.59 | 0.73 |
| 53 | single-dose | 9.70 | 13.61 | 14.27 | 2.46 | 0.84 |

Over the six reported acquisitions (48-53), production's median is 12.36%.
The NODE's is 12.62% filtered and 11.70% all-member. Production is lower on
three of the six, with a median paired ratio of 0.98 against each.

Production's error varies much more between acquisitions than the NODE's.
On every acquisition except 45 (+1%) it under-predicts the day-110 burden,
most on 51 and 52 (by 30% and 27%). Between the day-80 and day-100 pulses,
the rise of the burden from its post-pulse minimum to just before day 100, it
recovers only 32% and 39% of the true regrowth there, against 46-86% on the
other five acquisitions. The training fit does not single them out: 3.41%
and 3.95% on 51 and 52, against 4.74% on 48 (forecast 4.10%) and 2.19% on 53
(forecast 9.70%).

Post hoc diagnostics on 51 and 52:

- Before the standardization (errors 17.11% and 11.20%), starting the
  fitted ROMs from the noise-free reduced initial state changed every
  unchanged-regimen error by at most 0.15 points (51: 17.08%, 52: 11.27%),
  so the noisy first scan was not the cause. This was not repeated.
- Under the hand-set prior, 51 and 52 had the longest second-mode GP
  length-scales (6.2 and 5.9 days, against 4.3-4.7). That no longer held
  under the standardized settings with the closure constant (51: 3.7 days,
  52: 3.0, the others 1.9-3.4) and does not hold with the tempered rows (51:
  2.7 days, 52: 2.0, the others 1.8-2.4). Under both, 51 and 52 have the
  largest fitted first-mode noise (tempered: 0.80 and 0.65, against
  0.03-0.55), but with the tempered rows 50 has a larger error (13.02%,
  noise 0.38) than 49 (11.70%, noise 0.55).

With seven acquisitions this is a post hoc association, not a validated
predictor, and no setting was changed because of it.

## Posterior draws

Production scores 64 posterior draws per arm. These are empirical draw
bands, not calibrated uncertainty.

- **Completion:** every draw completes for every arm and acquisition.
- **Field error:** every growth draw is at most 8.88%. On multi-dose, 51 of
  the 768 draws exceed 25% (5 on 48, 21 on 49, 25 on 50; worst 37.12%). On
  single-dose, 40, 42, and 63 of 64 draws are at most 25% (worst 47.37%).
- **Growth burden coverage:** the 90% band contains the true burden at none
  of the 105 forecast queries on any acquisition (7-9 with the closure
  constant, all 105 under the earlier settings). The band is about 1% of the
  burden wide, while the forecast burden is about 2% low.
- **Single-dose burden coverage:** 151, 115, and 113 of 152 queries (16, 0,
  and 32 with the closure constant).
- **Multi-dose burden coverage:** all 152 queries for every strength and
  acquisition (5-152 with the closure constant).
- **Treatment-effect direction:** on every multi-dose acquisition and changed
  strength, every draw has the correct sign at all 114 meaningful queries
  after day 80.
- **Treatment-effect gains:** 0.47-1.85 per draw.
- **Effect-band coverage** (out of 114): 114 on 48, 83-84 on 49, and 0 on
  50.

## Interpretation and limitations

- The earlier design was idealized, and the realistic one costs production
  mostly on the unchanged regimen. With a clean nominal basis, production's
  single-dose error was 5.31-5.51% on 45-47, against NODE 9.90-10.04%. With
  segmented scans and each acquisition's own basis, the unchanged-regimen
  forecast on 48-53 spans 4.10-19.69% for production and 9.75-14.15% for
  the NODE.
- Production is more accurate on every acquisition for untreated growth and
  for each of the three changed doses. The NODE, trained on one input
  history, carries almost no dose dependence.
- The single-dose shortfall is not a representation limit: the floor is
  about 2.5%, and the basis is the one that succeeds on 48-50.
- All acquisitions share one anatomy and one set of true trajectories. Three
  acquisitions per task support no population-level claim.
- Several choices were made with knowledge of results:
  - the task design (half-strength regimen, growth parameters, scan counts)
    was fixed under the earlier observation model;
  - the chemo basis was chosen by production's accuracy on development
    acquisition 45;
  - the growth rank cap was introduced after rank 4 failed on reported
    acquisition 42, although the cap itself is a noise criterion;
  - the operator prior and GP nugget were replaced after an ablation that
    included the reported acquisitions;
  - the standardized settings were adopted after the reported acquisitions
    had been scored, although they were selected on development
    acquisitions 45-47 only. On the reported chemo acquisitions they are
    less accurate than the earlier settings;
  - the tempered dynamics rows replaced the closure constant after the
    reported acquisitions had been scored with it, although the rule was
    selected on development data only. On the reported multi-dose
    acquisitions it is less accurate than the constant (9.63-12.69% against
    8.23-8.84%).
- The NODE shares the basis but was not tuned for it. Input-structured or
  multi-dose-trained NODE variants were not evaluated.
- Production draw bands are not calibrated uncertainty (see above).

## Reproduction

```bash
cd experiments/tumor
conda run -n prob_rom python 04_unified_benchmark.py      # prepares data, fits, scores 64 draws
conda run -n prob_rom python 05_neural_ode_benchmark.py
conda run -n prob_rom python 06_compare_benchmark.py
```

The paper's three-dimensional multi-dose figure (acquisition 49, day 110) is a NiiVue
rendering of the saved evaluations on a slice of the demonstration patient's T1
post-contrast MRI; no model is refitted:

```bash
conda run -n prob_rom python niivue_benchmark_figure.py export
(cd niivue_viewer && npm ci && npm run build && npm run render-figure)
conda run -n prob_rom python niivue_benchmark_figure.py compose \
  --paper-figure ../../../GP-Bayes-Refactor/manuscript_v2/figures/selected/tumor_multidose_mri.png
```

See `niivue_viewer/README.md` (Publication figure) for the rendering settings.

Development runs use the same runners with explicit overrides. Examples are
`04_unified_benchmark.py multi-dose-chemo --seeds 45 --pod-centering mean --pod-rank 3 --no-draws`
and `single-dose-chemo --seeds 45 46 47 --pod-centering mean`. Rank-4 growth now stops at
preparation, which is intended. The implementation is in
`experiments/tumor/benchmark_data.py`, in the functions `detection_rule`, `noisy_scan`,
`cellularity_map`, `observed_pod`, `noise_threshold`, and `_segmented_acquisition`.
`tests/test_benchmark_cases.py` and `tests/test_benchmark_runners.py` cover the
segmentation, the guard, and the unchanged oracle-masked fingerprints.
