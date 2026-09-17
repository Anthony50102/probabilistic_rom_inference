# Tumor applications: multiple treatment histories and sparse growth forecasts

## Main findings

**Treatment-history diversity did not rescue chemotherapy inference at a fixed
observation budget. Untreated growth is a more workable application, but longer
forecasts, a noisy learned spatial basis, and uncertainty remain substantive
challenges.**

The two branches used the unchanged native hierarchical Bayesian method, not
the fixed-hyperparameter shared-latent MAP prototype from the
[previous study](SHARED_LATENT_WEIGHTING_RELIABILITY_STUDY.md).

In the diverse-history chemotherapy branch, the model learned the untreated
trajectory while largely explaining away the treated trajectories as noise.
Its new-treatment forecasts remained grossly inaccurate. This strengthens the
existing [GP-collapse evidence](tumor/INPUT_AWARE_GP_EXPERIMENT.md); it does not
prove that every richer acquisition design would fail.

Untreated growth was reasonably predictable for the first forecast month with
40 low-noise observations: median full-field errors were 6.66% and 8.14% for the
two biologies. The second month was appreciably harder. Sparse/noisy
observations exposed both representation limits and occasional severe
inference failures. Simple linear extrapolation was competitive.

An important acquisition distinction: **20% voxel-level generator noise did
not mean 20% noise in total burden.** Independent spatial noise largely averaged
out; the largest realized training-burden relative RMS noise was 0.3571%.
Consequently, burden forecasting alone is not a sufficiently demanding
demonstration of this method's value.

No production code, defaults, canonical caches, TumorTwin source, or manuscript
were changed. Recovery commit `f93fb04015eb2767c5d9c5d65060490cbc4d3c4e`
was retained as `backup/tumor-applications-before-parallel-f93fb04`.
This is a local research record, with no push or model promotion.

## Frozen design and interpretation

There were **39 native fits**, three acquisition seeds (42, 43, 44), and the
original 12,000 SVI updates per fit. Inference seed 42 was fixed independently
of the acquisition seed. All conditions and seeds are retained; no warm-start
rescue, continuation, alternate optimization, or forecast-selected variant
was introduced.

Both branches retained four POD modes, 200 GP/constraint evaluation points,
combined derivative and integration-by-parts weak constraints, diagonal
covariance blocks, derivative/weak weights 1/8, `gamma2=0.035`,
`mll_weight=0.1`, and the hierarchical zero-mean operator prior with
`sigma_O=5`. The native spectrum-anchored GP/noise priors, jittered
normal-equation backend, `AutoNormal`, `ClippedAdam`, learning rate 0.003,
and default float32 inference were unchanged. ODE solves occurred only in
simulation and post-fit forecasting, never in an inference objective.

The reported native **point** is the conditional operator mean evaluated at
arithmetic means of positive GP/block-scale samples. It is **not a posterior
predictive mean**. All 500 native operator draws were saved per fit; the
predeclared 64 indices `linspace(0,499,64,dtype=int)` were evaluated, using their
paired native initial-state draws. These are selected from all 500, not the
older export's auxiliary 32-draw view.

The ensembles include native variational hyperparameter, conditional operator,
and initial-state uncertainty. They are not a full latent-path posterior.
The observed reduced initial state is used for the point. The original
initial-state nugget (`max(1e-5, signal_variance*1e-4)`) is preserved, distinct
from the configured training-GP relative jitter of `1e-3`.

### Common measurements

Full-field, reduced-coordinate, and burden errors are distinct quantities.
Each percentage is `100*sqrt(sum(error^2)/sum(reference^2))` over the retained
query grid in its stated window, not a time-integral-weighted or endpoint error.
Full-field scoring includes the affine decoder shift and the unrepresented
physical residual, using stable QR geometry across all 646,812 voxels.

**Burden** means `voxel_volume_mm3 * sum(normalized_cellularity)`, not an
absolute clinical cell count. The voxel volume here is 1 mm^3.

Primary integration used DOP853 with `rtol=1e-8`, `atol=1e-10`, and maximum
step equal to the horizon divided by 400. The training endpoint and known
forcing knots were split; chemotherapy also split exact dose days. The fixed
safety rule was `max(abs(q_i)/training_RMS_i)=1e6`, with 200,000 RHS-call and
120-second per-target limits.

**C** denotes a large-state safety censor. Its forecast score is unavailable,
not zero and not a score on the surviving prefix. Numerical completion is
neither accuracy nor physical plausibility; a censor is not proof of
mathematical finite-time blow-up. NaN tails and negative predictions were
retained without clipping.

Pointwise 90% bands are the finite-draw 5th/95th percentiles, with explicit
per-time availability counts. The compound accuracy counts require both
completion and reduced error <=10% or <=25%, always with all 64 draws in the
denominator. Completed-only quantiles are not all-draw quantiles.

## Branch 1: multiple chemotherapy histories

All trajectories share the TNBC demo anatomy, initial MRI state, and biology:
`k=0.025`, `d=0.05`, `theta=1`, base sensitivity 0.5, and decay rate 0.7.
The native model is `cABN`, with operator shape `(4,10)` and column order
`c, A0..A3, B, N0..N3`. Every arm uses the same archived four-mode **clean
nominal training basis**, a privileged preprocessing step inherited from the
existing chemotherapy benchmark.

**Pre-fit amendment 001:** the initial protocol mistakenly copied sensitivity
0.2 from the generator default, whereas the mandatory archived nominal source
and decoder use 0.5. Before any chemotherapy fit, the original protocol was
preserved and an explicit amendment corrected only this parameter to 0.5.
The source NPZ and decoder identities were independently confirmed. Growth
was unaffected; this was not an outcome-driven parameter change.

Training observations cover days 5-70 with 1% native physical generator noise.
Every arm has the **same total budget of 120 observation columns**:

| Arm | Training histories | Observations per history |
| --- | --- | --- |
| `single_nominal` | Dose scale 1.0 | 120 |
| `repeated_nominal` | Three independent observations of dose scale 1.0 | 40 each |
| `diverse_histories` | Dose scales 0.0, 1.0, 1.5 | 40 each |

Thus this tests reallocating a fixed budget across histories, **not adding
more total observations**. Corresponding repeated/diverse histories share
observation grids and Gaussian innovations before their history-specific
masks/scales. Their nominal history at index 1 is exactly paired.

The nominal treatment days are 20, 40, 60, 80, 100. Evaluation-only targets
use dose scales 0.8 and 1.2 on that schedule, plus dose scale 1.0 on days
22, 42, 62, 82, 102. No heldout observations or heldout GP were generated,
and no heldout source contributed to POD. The native heldout IC uncertainty
uses averaged training-GP hyperparameters and the designated nominal
training time grid, not heldout data.

TumorTwin normalizes dose amounts. Its unchanged generator implements dose
scaling through effective sensitivity, `0.5*dose_scale`; the known input
uses the corresponding exposure scaling exactly once. All 4,001 input-table
knots were retained rather than substituting an analytic hard-onset input.

### Forecast results

These point errors pool all three heldout targets by summing their
error/reference energies over **days (70,110]**, 152 query points per target.
Training-target metrics are separate in the artifacts. Every point completed.

<!-- table:chemo-forecast -->
| Arm | Seed | Reduced % | Full-field % | Burden % |
| --- | --- | --- | --- | --- |
| single_nominal | 42 | 1015.85 | 1479.43 | 1232.26 |
| single_nominal | 43 | 412.96 | 601.41 | 583.22 |
| single_nominal | 44 | 217.94 | 317.40 | 236.49 |
| repeated_nominal | 42 | 11684.35 | 17016.43 | 11793.10 |
| repeated_nominal | 43 | 174.33 | 253.90 | 198.64 |
| repeated_nominal | 44 | 189.78 | 276.39 | 177.56 |
| diverse_histories | 42 | 418.09 | 608.89 | 639.24 |
| diverse_histories | 43 | 316.80 | 461.38 | 486.85 |
| diverse_histories | 44 | 416.32 | 606.31 | 640.37 |
<!-- endtable:chemo-forecast -->

Diversity changes the answer, but does not produce consistently better or
accurate treatment generalization. It is not sufficient to highlight its
improvement over the particularly poor single-history seed 42.

The heldout physical projection errors are only **1.99%, 1.33%, and 1.34%**
for dose 0.8, dose 1.2, and the shifted schedule. Representation error cannot
explain the hundreds-to-thousands-percent forecast errors.

In contrast, the diverse arm's untreated target has reduced forecast errors
**5.47%, 7.37%, 6.07%** in seed order. Its untreated clean-reference GP errors
are 0.438-1.124%, whereas treated-history GP errors are **92.351-99.819%**.
Across treated histories, mode-1 inferred noise SDs at the mean variance are
10.517-17.770 reduced units, versus diagnostic generator-projected SDs of
0.006937-0.007056. The diagnostic noise truth did not enter inference.

This is evidence of a selective failure: learning natural growth does not
force the model to honor treatment-driven changes. In the representative
diverse-history plots, treated forecasts largely continue growing despite
the observed treatment-related declines. The inference failure is already
visible on training histories, not just unseen doses.

All nine runs have **0/64** complete-and-reduced-accurate pooled heldout
draws at both thresholds. Every individual heldout target also has 0/64.
Of 3,072 draw-target trajectories, 3,068 complete; all four censors are
`single_nominal__seed44`, `heldout_dose_1p2`, original indices
23, 87, 174, 491, at approximately days 102.03, 103.42, 101.35, 103.31.
The diverse arm's heldout 90% burden bands include truth at **0/1,368**
defined time/target/seed entries, with all 64 draws finite there.

These observations strengthen the hypothesis that the present objective/GP
coupling can discount real treatment dynamics. They do not isolate one
objective term, establish a global optimum, rule out a different acquisition
design, or show that genuinely adding more data cannot help.

## Branch 2: future untreated growth from sparse/noisy snapshots

Two separate no-treatment biologies were fitted, not pooled into a
cross-parameter model: nominal `k=0.025,d=0.05` and faster-spreading
`k=0.05,d=0.1`, both with `theta=1` and the same initial anatomy.
The native model is `cA`, operator shape `(4,5)`.

Training is restricted to days 5-60. The five conditions are 40 observations
at 1% noise, and 16 or 8 observations at 10% or 20% noise. Each acquisition
seed uses nested subsets of a 40-point irregular grid with shared Gaussian
innovations; endpoints 5 and 60 are retained.

Noise SD is the stated fraction of the clean physical observation range,
applied where cellularity exceeds 0.001 of its maximum. The first physical
observation is exact and there is no clipping. A new native POD basis and
shift are learned from **only that run's noisy training observations**.
No clean/future basis rescue is used.

Each 120-day FOM was generated from day 0 at 0.5-day solve/save spacing.
Overlapping source snapshots through day 90 match the earlier canonical
caches bitwise. Prediction uses 402 points, explicitly including days
60 and 90; each separate forecast month has 105 query points.

### Every full-window point result

Entries are **full-field relative error % over days (60,120]**. The very
large finite result and all four censors are intentional parts of the table.

<!-- table:growth-forecast -->
| Biology | Obs / noise | Seed 42 | Seed 43 | Seed 44 |
| --- | --- | --- | --- | --- |
| Nominal | 40 / 1% | 16.40 | 18.29 | 18.87 |
| Nominal | 16 / 10% | 21.94 | C | 21.17 |
| Nominal | 16 / 20% | 36.19 | C | 34.84 |
| Nominal | 8 / 10% | 26.65 | 25.11 | 25.45 |
| Nominal | 8 / 20% | 55.87 | 55.87 | 55.36 |
| Faster | 40 / 1% | 22.18 | 22.93 | 25.00 |
| Faster | 16 / 10% | 38.44 | 420546.75 | 37.65 |
| Faster | 16 / 20% | 46.00 | C | C |
| Faster | 8 / 10% | 40.24 | 39.25 | 39.53 |
| Faster | 8 / 20% | 52.54 | 50.59 | 50.99 |
<!-- endtable:growth-forecast -->

For the 40-observation, 1%-noise reference, median full-field errors in the
two **separate** 30-day windows are:

<!-- table:growth-horizon -->
| Biology | Days (60,90] % | Days (90,120] % | 90% burden-band inclusion, (60,120] |
| --- | --- | --- | --- |
| Nominal | 6.66 | 23.12 | 121/630 |
| Faster | 8.14 | 28.30 | 54/630 |
<!-- endtable:growth-horizon -->

The final column is descriptive time-point inclusion across three seeds,
not a coverage estimate across independent patients or experiments. All
64 draws are finite at these reference forecast entries. The second month
is not a cumulative 60-day metric and should not be compared as one.

### Representation and simple baselines

Two controls were specified before reading fitted outcomes: holding the
last reduced observation fixed, and ordinary unweighted linear regression
against time using all reduced training observations. They reuse exactly
the native noisy-data basis and shift. They have no invented confidence
intervals.

The following entries are medians over **all three seeds**, in full-field
error %, days (60,120]. The projection floor is the best possible physical
representation in the fixed affine basis, not a fitted forecast. It is
evaluated even when the native ODE is censored.

<!-- table:growth-baselines -->
| Biology | Obs / noise | Projection floor % | Persistence % | Linear trend % |
| --- | --- | --- | --- | --- |
| Nominal | 40 / 1% | 7.55 | 26.52 | 14.73 |
| Nominal | 16 / 10% | 20.55 | 32.14 | 21.15 |
| Nominal | 16 / 20% | 29.88 | 44.45 | 34.34 |
| Nominal | 8 / 10% | 24.33 | 32.10 | 25.78 |
| Nominal | 8 / 20% | 35.99 | 45.14 | 45.36 |
| Faster | 40 / 1% | 18.70 | 41.77 | 36.36 |
| Faster | 16 / 10% | 36.26 | 44.75 | 38.76 |
| Faster | 16 / 20% | 42.64 | 53.57 | 45.08 |
| Faster | 8 / 10% | 38.52 | 44.99 | 40.57 |
| Faster | 8 / 20% | 47.26 | 53.47 | 52.20 |
<!-- endtable:growth-baselines -->

For nominal reference growth, native median error is 18.29% versus 14.73%
for linear extrapolation. For faster reference growth, native is better:
22.93% versus 36.36%. Neither method wins universally.

For faster growth with 8 observations and 10% noise, native median error
39.53% is close to the 38.52% representation floor. Much of that error cannot
be removed by changing the operator learner while keeping this basis.
Conversely, the huge finite outlier and censors cannot be explained by a
modest representation floor alone.

Fewer observations do not produce a monotone outcome: the 16-observation
conditions have catastrophic seeds that the 8-observation conditions do
not. Avoid treating numerical stability after discarding data as evidence
of improved learning.

### Burden is a different, often easier observation problem

Despite up to 20% generator noise in voxel fields, the maximum realized
relative RMS noise in observed total burden is **0.357113%**, excluding the
exact first observation. This is spatial averaging of independent noise,
not evidence that heavily corrupted clinical burden measurements were solved.

Over days (60,120], median native/linear burden errors are **14.41%/10.02%**
for nominal reference growth and **13.91%/22.95%** for faster reference
growth. At nominal 8 observations/20% noise, they are **60.65%/11.87%**.
All three seeds complete in these examples. Per-seed burden scores and
availability for every other condition are retained in the readout;
completed-only medians must not conceal missing or extreme results.

The baseline burden is the affine functional of the same reduced forecast,
not a separately optimized clinical burden predictor. It is already
competitive without tailoring the model to this quantity.

### Posterior trajectory reliability

All 1,920 selected growth draws remain in each denominator:

<!-- table:growth-uncertainty -->
| Window | Complete | Complete + reduced <=10% | Complete + reduced <=25% |
| --- | --- | --- | --- |
| forecast_60_90 | 1689/1920 | 264/1920 | 962/1920 |
| forecast_90_120 | 1646/1920 | 46/1920 | 585/1920 |
| forecast_60_120 | 1646/1920 | 38/1920 | 696/1920 |
<!-- endtable:growth-uncertainty -->

These pooled counts describe this deliberately heterogeneous stress panel,
not probabilities for a population. There are 274 draw censors through day
120. An auxiliary worker count of 426/1,920 used <=25% **physical** error,
not the reduced-error criterion above.

Negative forecast burdens and extreme uncertainty tails remain in the
artifacts and figures. In particular, the severe nominal-noise overview
retains its large negative band rather than clipping it to make the
central curve look well behaved. Numerical completion does not establish
physically sensible or calibrated uncertainty.

## What this establishes, and what it does not

The chemotherapy result is not simply a lack of examples of untreated
growth: that component can be learned while treatment responses are
discounted. Fixed-budget history diversity alone did not solve the native
inference problem. A future chemo redesign would need to address fidelity
to the observed treated trajectories, not just add a heldout-dose score.

Untreated growth is a viable source of nontrivial forecasting questions,
especially spatial prediction beyond the observation window. But this
study separates three issues that should not be conflated: the amount of
information left after noisy POD, extrapolation of the reduced dynamics,
and the usefulness of posterior uncertainty. Good total-burden curves
alone do not resolve those issues or outperform simple forecasts.

This is one demo anatomy, two untreated parameter settings, three
acquisition seeds, and a finite synthetic acquisition panel. It is not
clinical validation, calibrated population uncertainty, proof of
identifiability, or evidence about every tumor or chemotherapy task.
The existing algorithm was not retuned to make either branch succeed.

## Evidence, numerical controls, and persistence

The parent independently reconstructed all primary trajectory metrics,
role pooling, finite-draw bands and denominators; reproduced source,
observation/noise, POD/decoder and physical geometry; and checked native
settings, checkpoint losses, hyperparameter exports and IC pairing.
Chemotherapy GP means, derivative means, conditional variances, MLLs,
noise summaries and physical/reduced fidelity were independently checked.

The frozen numerical panel contains 22 chemo controls using independent
manual-polynomial Radau and 18 growth controls using augmented matrix
exponentials. Maximum relative prediction differences were
`7.9957e-12` and `7.3785e-15`, respectively. These are numerical
reproductions, not additional scientific fits.

The original 22 parent chemo controls split every input-table knot but
omitted the redundant non-knot dose-day cuts. They are retained as
auxiliary evidence; a separate unchanged-operator rerun includes the exact
frozen cut set. Primary forecasts were not replaced.

Chemotherapy retains the native two-stage reference interpolation:
source-knot cubic to a 200-point clean reference, then query-grid cubic.
Observations use direct source-knot cubic. An independent direct-source
comparison found only 0.0804-0.1345% full-field forecast-reference
differences on the three heldouts, much smaller than the observed failures.
Growth uses direct source-knot interpolation. Its clean simulator
interpolation is data generation, not an inference input of future states.

Chemo GP diagnostics are post-fit float64 conditionals; growth retains its
native float32 GP diagnostics with separately labeled float64 controls.
Raw conditional variance diagonals are retained; displayed GP SDs use
`sqrt(max(variance,0))`. That arithmetic display convention is distinct
from clipping an ODE forecast. An exact-IC compression roundoff guard in
growth was replaced by an explicitly retained pre-fit tolerance-only
validation; no observation or initial state changed.

The final growth readout includes projection floors for all bases,
including censored native points. The earlier intermediate readout that
omitted those fields is preserved. No forecast score changed.

Accounting is **468,000 SVI updates**, 51 training-history/204 mode-wise
GP contexts, 19,500 saved operator draws, and 2,496 unique selected operator
draws. There are **78 point-target and 4,992 draw-target trajectories**:
74 points and 4,714 draw-targets complete, with 4 and 278 censors,
respectively. Reusing one operator draw across chemo targets does not make
those targets independent. These totals exclude numerical controls.

Numerical artifacts are persistent **local session files**, not checked
into Git or a portable dataset bundled with this document. Their root is:

```text
/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics/tumor_applications/
```

| Relative artifact | Purpose |
| --- | --- |
| `study_release.json`, `study_artifact_manifest.json` | Combined content-addressed release and file inventory |
| `protocol_freeze.json`, `protocol_amendment_001.json` | Original source freeze and pre-fit sensitivity correction |
| `verification_protocol.json`, `growth_baselines_protocol.json` | Predeclared numerical controls and training-only baselines |
| `combined_readout.json` | Verified tables, GP comparisons and total accounting |
| `growth_final_summary.json` | All growth seeds/windows, baselines, burden noise and all-basis floors |
| `multi_history/release.json`, `growth/release.json` | Immutable producer releases, arrays, losses and source snapshots |
| `parent_multi_history/summary.json`, `parent_growth/summary.json` | Independently reconstructed primary scores and availability |
| `parent_multi_history_data_verification.json`, `parent_growth_data_verification.json` | Independent source, acquisition, geometry and native-fit audits |
| `parent_multi_history_numerical_exact_cuts/`, `parent_growth_numerical/` | Fixed independent numerical reproductions |
| `multi_history/figures/preferred_figures.json` | Chemo burden, GP-fidelity and loss figure index |
| `growth_readout/nominal_growth__baseline_comparison.png` | Fixed-seed nominal reference/severe-noise overview |
| `growth_readout/faster_spreading_growth__baseline_comparison.png` | Corresponding faster-growth overview |

The multi-history producer release SHA256 is
`09eb3f008c3d8aa3c7a84f86320665db54b13c54b9f16859dad66be4b818c0c5`;
the growth producer release SHA256 is
`43621d03fe484b6a8976a604e3e0508f9c8a790dcf00effabb89806bc8e6cec1`.
The combined release seals this record, parent scripts/readouts, both
producer manifests and the protocol. The previous weighting/reliability
release (`532d511969e5bbc27992fb7c07071777b4ed7793855ce4658f7738141f13927c`)
and mechanism release
(`227711e5a1f4010cd78d3db338112e74a2d6bf090f9032c67670bf0fcfb82bc9`)
remain unchanged, including their 16,381 and 3,247 manifest entries.
