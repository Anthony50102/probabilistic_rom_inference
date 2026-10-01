# Tumor benchmarks on segmented noisy scans

## Status and scope

**Complete, with a mixed outcome.** Every scan is now noisy and every basis
is fitted to noisy scans. Under this design production (Bayesian OpInf)
stays more accurate than the NODE on every acquisition in two benchmarks:

- untreated growth: 7.45% against 12.33% median over days 60-90;
- multi-dose chemo: 5.60-7.08% against 11.36-26.65% at every future strength.
  Here the frozen development-selected basis passed every pre-declared
  confirmation check.

Production's median day-110 gains are 0.94-1.00, so it recovers most of the
true treatment effect, while the NODE predicts almost none (gains within
0.02 of zero).

The single-dose continuation is not a production win. Its medians are
11.20% against 11.15%, and the NODE is lower on two of three acquisitions.
The same unchanged-regimen forecast is also the multi-dose 0.5x arm. Over all
six reported chemo acquisitions it gives production 4.90-17.11% (median
6.16%) against NODE 9.75-14.15% (median 12.62%).

**Operator prior.** All values use the nondimensional operator prior and the
round-off GP nugget (`sigma_O=None`, `gp_jitter_rel=None`, the
`WeakFormConfig` defaults), which replaced the hand-set `sigma_O=5.0` and
`gp_jitter_rel=1e-3` here and the corresponding settings in every other
experiment. The replacement was chosen because it has no per-experiment
constant, but only after an ablation that included these acquisitions, so
the choice was not blind to them. The task design and the chemo basis were
fixed under the earlier prior. Values marked "earlier prior" come from it;
the full earlier record is at git tag `pre-universal-prior`. No other method
or configuration was changed after the reported acquisitions were scored.

The three reported tumor benchmarks now observe the simulated tumor through
noisy scans from which the lesion is segmented (`--observation segmented`, the
default of `benchmark_cases.py` and the 04/05/06 runners). This replaces the
earlier oracle-masked design of the
[three-benchmark POD record](TUMOR_BENCHMARK_POD_COMPARISON.md), in which
noise was added only where the true tumor was present, the initial state was
noise-free, and the chemo PODs were fitted to clean simulated snapshots. That
design remains available with `--observation oracle_masked` and reproduces its
sealed results.

Apart from the operator prior and nugget above, the production Bayesian
algorithm, its inference settings, and the Neural ODE (NODE) recipe are those
of the earlier design. Only the data, the POD bases fitted to them, and
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
(`threshold_censored`). Refitted under the current prior, the eigenvalue is
+1.0 per day and the point forecast and all 64 draws still diverge. On 43 and
44, rank 4 gave 7.35% and 7.44% over days 60-90 (earlier prior; not refitted). This failure is why the rank guard was added. The guard admits three
modes on every growth acquisition, and all three acquisitions report rank 3.

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

**Single-dose: the same basis on fresh acquisitions 51-53.** The single-dose
task was first declared with the mean-centered rank-4 basis on acquisitions
45-47, where it failed (29.88%, 120.59%, and 33.25%; under the current prior
26.60%, 104.22%, and 33.78%). A single-dose forecast is
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
| Training, days 5-60 | 0.78 / 0.77 / 0.80 | 0.75 / 0.75 / 0.77 | 0.75 / 0.75 / 0.77 | 0.73 |
| Days 60-90 | 7.14 / 7.45 / 7.52 | 12.33 / 14.82 / 11.72 | 13.92 / 14.82 / 11.65 | 6.53 |
| Days 60-120 | 20.59 / 21.55 / 21.75 | 28.01 / 31.84 / 27.42 | 29.69 / 31.84 / 27.26 | 18.45 |

NODE kept 16, 18, and 17 members after its loss filter.

**Single-dose chemo** (acquisitions 51 / 52 / 53), days 70-110

| Window | Production | NODE filtered | NODE all-member | Floor (median) |
|---|---|---|---|---|
| Training, days 5-70 | 2.84 / 2.51 / 4.09 | 1.31 / 1.34 / 1.87 | 1.31 / 1.34 / 1.90 | 0.51 |
| Days 70-110 | 17.11 / 11.20 / 4.90 | 11.15 / 9.75 / 13.61 | 11.15 / 9.75 / 14.27 | 2.46 |

The NODE kept 20, 20, and 19 members, and production wins only on 53.

**Multi-dose chemo** (acquisitions 48 / 49 / 50), days 70-110

| Future pulses | Production | NODE filtered | NODE all-member | Floor (median) |
|---|---|---|---|---|
| Training, days 5-70 | 2.88 / 1.76 / 1.76 | 1.48 / 1.32 / 1.41 | 1.46 / 1.32 / 1.41 | 0.51 |
| 0.25x | 7.41 / 4.99 / 6.66 | 28.02 / 26.65 / 24.10 | 24.98 / 26.65 / 24.10 | 2.85 |
| 0.5x (unchanged) | 4.99 / 5.60 / 6.72 | 14.15 / 13.87 / 11.64 | 11.76 / 13.87 / 11.64 | 2.54 |
| 0.75x | 4.38 / 6.48 / 6.93 | 10.66 / 11.36 / 13.84 | 12.06 / 11.36 / 13.84 | 2.30 |
| 1x | 4.90 / 7.14 / 7.08 | 23.97 / 23.57 / 27.59 | 26.01 / 23.57 / 27.59 | 2.10 |

Production wins every seed at every strength against both centers. The NODE
kept 19, 20, and 20 members. Every frozen confirmation check passes at every
strength:

- production median at most 15% and every seed at most 25%;
- median paired ratio at most 0.90 against both NODE centers (observed
  0.26-0.50);
- at least two of three wins (observed 3/3).

The treatment effect is the change from the unchanged arm after day 80. The
true day-110 changes are +38.4%, -28.3%, and -48.9% of the control burden.
Gain is the predicted day-110 change divided by the true one.

| Effect versus 0.5x | Production error | Production gain | NODE filtered error | NODE gain |
|---|---|---|---|---|
| 0.25x | 23.05 / 13.00 / 12.93 | 1.11 / 1.00 / 0.90 | 98.71 / 94.75 / 95.98 | -0.03 / 0.01 / -0.02 |
| 0.75x | 17.20 / 8.88 / 11.01 | 1.05 / 0.96 / 0.87 | 100.52 / 95.03 / 94.97 | -0.02 / 0.01 / -0.02 |
| 1x | 15.21 / 7.74 / 10.62 | 1.04 / 0.94 / 0.86 | 100.03 / 94.86 / 94.52 | -0.00 / 0.02 / -0.03 |

## The unchanged regimen on every chemo acquisition

The unchanged half-strength forecast was made on seven acquisitions: 45
(development), 48-50 (the multi-dose 0.5x arm), and 51-53 (single-dose). The
day-110 column is the predicted over the true tumor burden.

| Acquisition | Role | Production | NODE filtered | NODE all-member | Floor | Day-110 burden |
|---|---|---|---|---|---|---|
| 45 | development | 6.90 | 9.84 | 10.29 | 2.65 | 0.87 |
| 48 | multi-dose | 4.99 | 14.15 | 11.76 | 2.55 | 1.00 |
| 49 | multi-dose | 5.60 | 13.87 | 13.87 | 2.54 | 0.92 |
| 50 | multi-dose | 6.72 | 11.64 | 11.64 | 2.53 | 0.89 |
| 51 | single-dose | 17.11 | 11.15 | 11.15 | 2.46 | 0.72 |
| 52 | single-dose | 11.20 | 9.75 | 9.75 | 2.59 | 0.83 |
| 53 | single-dose | 4.90 | 13.61 | 14.27 | 2.46 | 0.96 |

Over the six reported acquisitions (48-53), production's median is 6.16%.
The NODE's is 12.62% filtered and 11.70% all-member. Production is lower on
four of the six, with median paired ratios of 0.49 and 0.50.

Production's error varies much more between acquisitions than the NODE's.
On every acquisition it under-predicts the day-110 burden (on 48 by 0.1%),
most on 51 and 52. Between the day-80 and day-100 pulses, the rise of the
burden from its post-pulse minimum to just before day 100, it recovers only
45% and 67% of the true regrowth there, against 76-94% on the other five
acquisitions. The training fit does not single them out: 2.84% and 2.51% on
51 and 52, against 2.88% on 48 (forecast 4.99%) and 4.09% on 53 (forecast
4.90%).

Post hoc diagnostics on 51 and 52:

- Starting the fitted ROMs from the noise-free reduced initial state changes
  every unchanged-regimen error by at most 0.15 points (51: 17.08%, 52:
  11.27%), so the noisy first scan is not the cause.
- Under the earlier prior, 51 and 52 had the longest second-mode GP
  length-scales (6.2 and 5.9 days, against 4.3-4.7). Under the current prior
  that no longer holds (51: 6.5 days, 52: 3.3, the others 2.8-5.0). Both have
  the largest fitted first-mode noise (4.1 and 4.2, against 0.07-3.3), but
  49 is close (3.3) with an error of 5.60%.

With seven acquisitions this is a post hoc association, not a validated
predictor, and no setting was changed because of it.

## Posterior draws

Production scores 64 posterior draws per arm. These are empirical draw
bands, not calibrated uncertainty.

- **Completion:** every draw completes for every arm and acquisition.
- **Field error:** every growth draw is at most 10.34%. Every multi-dose draw
  is at most 25.09%, and one (on 49 at 1x) exceeds 25%. On single-dose, 59,
  62, and 64 of 64 draws are at most 25% (worst 33.76%).
- **Growth burden coverage:** the 90% band contains the true burden at all
  105 forecast queries on every acquisition.
- **Single-dose burden coverage:** 3, 0, and 36 of 152 queries.
- **Multi-dose burden coverage** (0.25x / 0.5x / 0.75x / 1x, out of 152):
  - 48: 152 / 152 / 152 / 152;
  - 49: 152 / 152 / 151 / 152;
  - 50: 0 / 0 / 13 / 46.
- **Treatment-effect direction:** on every multi-dose acquisition and changed
  strength, every draw has the correct sign at all 114 meaningful queries
  after day 80.
- **Treatment-effect gains:** 0.69-1.29 per draw.
- **Effect-band coverage** (out of 114): 98-114 on 48, 112 on 49, and 2-58
  on 50.

## Interpretation and limitations

- The earlier design was idealized, and the realistic one costs production
  mostly on the unchanged regimen. With a clean nominal basis, production's
  single-dose error was 5.31-5.51% on 45-47, against NODE 9.90-10.04%. With
  segmented scans and each acquisition's own basis, the same task spans
  4.90-17.11% for production and 9.75-14.15% for the NODE.
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
    included the reported acquisitions.
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
