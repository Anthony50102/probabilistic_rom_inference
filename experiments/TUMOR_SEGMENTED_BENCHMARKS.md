# Tumor benchmarks on segmented noisy scans

## Status and scope

**Complete, with a mixed outcome.** Every scan is now noisy and every basis
is fitted to noisy scans. Under this design production (unchanged Bayesian
OpInf) stays more accurate than the NODE on every acquisition in two
benchmarks:

- untreated growth: 7.11% against 12.33% median over days 60-90;
- multi-dose chemo: 6.02-7.04% against 11.36-26.65% at every future strength.
  Here the frozen development-selected basis passed every pre-declared
  confirmation check.

Production's median day-110 gains are 0.91-0.96, so it recovers most of the
true treatment effect, while the NODE predicts almost none (gains within
0.02 of zero).

The single-dose continuation is no longer a production win. Its medians are
11.38% against 11.15%, and the NODE is lower on two of three acquisitions.
The same unchanged-regimen forecast is also the multi-dose 0.5x arm. Over all
six reported chemo acquisitions it gives production 4.78-16.96% (median
7.55%) against NODE 9.75-14.15% (median 12.62%). No method or configuration
was changed after the reported acquisitions were scored.

The three reported tumor benchmarks now observe the simulated tumor through
noisy scans from which the lesion is segmented (`--observation segmented`, the
default of `benchmark_cases.py` and the 04/05/06 runners). This replaces the
earlier oracle-masked design of the
[three-benchmark POD record](TUMOR_BENCHMARK_POD_COMPARISON.md), in which
noise was added only where the true tumor was present, the initial state was
noise-free, and the chemo PODs were fitted to clean simulated snapshots. That
design remains available with `--observation oracle_masked` and reproduces its
sealed results.

The production Bayesian algorithm, its inference settings, and the Neural ODE
(NODE) recipe are unchanged. Only the data, the POD bases fitted to them, and
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
(`threshold_censored`). On 43 and 44, rank 4 gave 7.35% and 7.44% over days
60-90. This failure is why the rank guard was added. The guard admits three
modes on every growth acquisition, and all three acquisitions report rank 3.

**Chemo: uncentered rank 4.** This was selected on development acquisition 45
among four candidates by the lowest mean production error over the four
future strengths, then frozen before any reported chemo acquisition was
fitted. The NODE was not consulted.

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

**Single-dose: the same basis on fresh acquisitions 51-53.** The single-dose
task was first declared with the mean-centered rank-4 basis on acquisitions
45-47, where it failed (29.88%, 120.59%, and 33.25%). A single-dose forecast is
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
| Training, days 5-60 | 0.79 / 0.79 / 0.80 | 0.75 / 0.75 / 0.77 | 0.75 / 0.75 / 0.77 | 0.73 |
| Days 60-90 | 6.96 / 7.11 / 7.27 | 12.33 / 14.82 / 11.72 | 13.92 / 14.82 / 11.65 | 6.53 |
| Days 60-120 | 19.42 / 19.76 / 20.14 | 28.01 / 31.84 / 27.42 | 29.69 / 31.84 / 27.26 | 18.45 |

NODE kept 16, 18, and 17 members after its loss filter.

**Single-dose chemo** (acquisitions 51 / 52 / 53), days 70-110

| Window | Production | NODE filtered | NODE all-member | Floor (median) |
|---|---|---|---|---|
| Training, days 5-70 | 2.83 / 2.50 / 3.42 | 1.31 / 1.34 / 1.87 | 1.31 / 1.34 / 1.90 | 0.51 |
| Days 70-110 | 16.96 / 11.38 / 8.33 | 11.15 / 9.75 / 13.61 | 11.15 / 9.75 / 14.27 | 2.46 |

The NODE kept 20, 20, and 19 members, and production wins only on 53.

**Multi-dose chemo** (acquisitions 48 / 49 / 50), days 70-110

| Future pulses | Production | NODE filtered | NODE all-member | Floor (median) |
|---|---|---|---|---|
| Training, days 5-70 | 2.80 / 1.77 / 1.75 | 1.48 / 1.32 / 1.41 | 1.46 / 1.32 / 1.41 | 0.51 |
| 0.25x | 7.04 / 5.94 / 7.23 | 28.02 / 26.65 / 24.10 | 24.98 / 26.65 / 24.10 | 2.85 |
| 0.5x (unchanged) | 4.78 / 6.02 / 6.76 | 14.15 / 13.87 / 11.64 | 11.76 / 13.87 / 11.64 | 2.54 |
| 0.75x | 4.16 / 6.53 / 6.48 | 10.66 / 11.36 / 13.84 | 12.06 / 11.36 / 13.84 | 2.30 |
| 1x | 4.64 / 7.03 / 6.32 | 23.97 / 23.57 / 27.59 | 26.01 / 23.57 / 27.59 | 2.10 |

Production wins every seed at every strength against both centers. The NODE
kept 19, 20, and 20 members. Every frozen confirmation check passes at every
strength:

- production median at most 15% and every seed at most 25%;
- median paired ratio at most 0.90 against both NODE centers (observed
  0.23-0.47);
- at least two of three wins (observed 3/3).

The treatment effect is the change from the unchanged arm after day 80. The
true day-110 changes are +38.4%, -28.3%, and -48.9% of the control burden.
Gain is the predicted day-110 change divided by the true one.

| Effect versus 0.5x | Production error | Production gain | NODE filtered error | NODE gain |
|---|---|---|---|---|
| 0.25x | 21.72 / 13.90 / 13.90 | 1.07 / 0.96 / 0.87 | 98.71 / 94.75 / 95.98 | -0.03 / 0.01 / -0.02 |
| 0.75x | 16.49 / 10.58 / 12.57 | 1.02 / 0.92 / 0.83 | 100.52 / 95.03 / 94.97 | -0.02 / 0.01 / -0.02 |
| 1x | 14.73 / 9.62 / 12.29 | 1.00 / 0.91 / 0.83 | 100.03 / 94.86 / 94.52 | -0.00 / 0.02 / -0.03 |

## The unchanged regimen on every chemo acquisition

The unchanged half-strength forecast was made on seven acquisitions: 45
(development), 48-50 (the multi-dose 0.5x arm), and 51-53 (single-dose). The
day-110 column is the predicted over the true tumor burden.

| Acquisition | Role | Production | NODE filtered | NODE all-member | Floor | Day-110 burden |
|---|---|---|---|---|---|---|
| 45 | development | 7.04 | 9.84 | 10.29 | 2.65 | 0.87 |
| 48 | multi-dose | 4.78 | 14.15 | 11.76 | 2.55 | 0.97 |
| 49 | multi-dose | 6.02 | 13.87 | 13.87 | 2.54 | 0.90 |
| 50 | multi-dose | 6.76 | 11.64 | 11.64 | 2.53 | 0.88 |
| 51 | single-dose | 16.96 | 11.15 | 11.15 | 2.46 | 0.72 |
| 52 | single-dose | 11.38 | 9.75 | 9.75 | 2.59 | 0.82 |
| 53 | single-dose | 8.33 | 13.61 | 14.27 | 2.46 | 0.89 |

Over the six reported acquisitions (48-53), production's median is 7.55%.
The NODE's is 12.62% filtered and 11.70% all-member. Production is lower on
four of the six, with median paired ratios of 0.60 and 0.58.

Production's error varies much more between acquisitions than the NODE's.
On every acquisition it under-predicts the day-110 burden, most on 51 and 52.
There it recovers only 54% and 63% of the true regrowth between the day-80
and day-100 pulses, against 69-90% on 48, 49, and 53. The training fit does
not single them out: 2.83% and 2.50% on 51 and 52, against 2.80% on 48
(forecast 4.78%) and 3.42% on 53 (forecast 8.33%).

Post hoc diagnostics on 51 and 52:

- Starting the fitted ROMs from the noise-free reduced initial state changes
  every unchanged-regimen error by at most 0.09 points (51: 16.93%, 52:
  11.44%), so the noisy first scan is not the cause.
- On 51 and 52 the GP settles at a longer second-mode length-scale (6.2 and
  5.9 days, against 4.3-4.7 on the other five acquisitions) and the largest
  fitted noise in the first two modes. Their first-mode length-scales
  (2.0-2.1 days) do not stand out: 49 and 53 have 2.0 and 2.3 days, with errors
  of 6.02% and 8.33%.

With seven acquisitions this is a post hoc association, not a validated
predictor, and no setting was changed because of it.

## Posterior draws

Production scores 64 posterior draws per arm. These are empirical draw
bands, not calibrated uncertainty.

- **Completion:** every draw completes for every arm and acquisition.
- **Field error:** every growth draw is at most 9.88% and every multi-dose
  draw at most 24.94%. On single-dose, 59, 63, and 64 of 64 draws are at most
  25% (worst 36.14%).
- **Growth burden coverage:** the 90% band contains the true burden at 11, 12,
  and 13 of 105 forecast queries.
- **Single-dose burden coverage:** 4, 0, and 0 of 152 queries.
- **Multi-dose burden coverage** (0.25x / 0.5x / 0.75x / 1x, out of 152):
  - 48: 152 / 152 / 152 / 152;
  - 49: 152 / 71 / 122 / 151;
  - 50: 0 / 0 / 35 / 79.
- **Treatment-effect direction:** on every multi-dose acquisition and changed
  strength, every draw has the correct sign at all 114 meaningful queries
  after day 80.
- **Treatment-effect gains:** 0.68-1.20 per draw.
- **Effect-band coverage** (out of 114): 113 on 48, 109-111 on 49, and 1-11
  on 50.

## Interpretation and limitations

- The earlier design was idealized, and the realistic one costs production
  mostly on the unchanged regimen. With a clean nominal basis, production's
  single-dose error was 5.31-5.51% on 45-47, against NODE 9.90-10.04%. With
  segmented scans and each acquisition's own basis, the same task spans
  4.78-16.96% for production and 9.75-14.15% for the NODE.
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
    acquisition 42, although the cap itself is a noise criterion.
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
