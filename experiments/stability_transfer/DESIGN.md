# Local-contraction prior on the existing heat and diffusion-reaction benchmarks

## Design recorded before any new fit

The user requested applying the cylinder stability-aware prior to heat and
`burgers_2d`, excluding tumor. The latter directory now implements
2-D diffusion-reaction, not viscous Burgers. No existing paper results,
defaults, or tumor code will be replaced.

The starting commit is `49befeb`, protected by local tag
`pre-stability-transfer-20261112`. This is a cross-system exploratory comparison,
not a claim that the cylinder settings have already been validated universally.

### Cases and comparisons

Use the existing adapters, POD rules, observation counts, noise levels, training
intervals, inputs and forecast intervals:

| System | Rank | Observations | Noise | Training | Evaluation |
|---|---|---|---|---|---|
| Cubic heat | 5 | 20 per training trajectory; 5 trajectories | 1%, 3%, 5% | [0,1] | [0,2], plus separate held-out input/IC case |
| 2-D diffusion-reaction (`burgers_2d`) | 3 | 60 | 3% | [0,1] | [0,3] |

For each of the four cases, run seeds 42 and 43 with five variants:

1. `hierarchical`: original block-hierarchical zero-mean operator prior.
2. `fixed`: unit-multiplier nondimensional diagonal Gaussian, without contraction.
3. `contract1`: local contraction with dimensionless damping 1 and width 3.
4. `contract5`: damping 5, width 3.
5. `contract20`: damping 20, width 3 (the cylinder setting).

This is 40 fits. The fixed control separates changes due to removing the
hierarchy from the additional contraction information. All settings are fixed
before results; report every variant, rather than selecting on forecast truth.
Seeds change acquisition and inference randomness, not the physical systems.
Use float64 throughout this comparison, including controls, 12000 SVI updates,
the existing initialization and learning rate, and 500 operator draws.
Do not treat differences from the older float32 paper runs as prior effects.

### Generalized prior

Obtain states and rates from the training-only data-GP MAP already used by the
tempering rule. At 12 interior anchors per training trajectory, differentiate
the ROM's actual feature assembler with respect to state, holding inputs fixed.
This handles autonomous features and multiple time-varying input channels.
Use the original within-trajectory fluctuation scales D and nondimensional
base Gaussian scales. Favor T D^-1 J D P = -lambda P, where T is the mean
training duration and P removes the normalized D^-1 qdot tangent.
Normalize the residual by width times sqrt(total accepted anchors).

Skip and count anchors whose nondimensional scaled rate norm is at most
sqrt(float64 epsilon); if all are stationary, fail explicitly rather than
inventing a tangent. Reject zero modal fluctuation scales. Geometry is fixed:
its uncertainty is not propagated. This is a soft, local preference, not
global stability, positivity, conservation, or a physical energy certificate.
No held-out trajectory is used to construct the prior.

### Evaluation and safeguards

Save posterior arrays before integration. Evaluate the same 200 evenly spaced
operator draws out of 500 used by the production prediction path, with its
initial-condition uncertainty convention. Also evaluate the Rao-Blackwell
mean operator. Use RK45 with rtol 1e-7 and atol 1e-9 for all variants; an
explicit RHS work budget scales with horizon (20000 calls per training-time
unit). A common amplitude guard is 100 times the largest absolute training
reduced observation. Record amplitude, work-budget and solver failures
separately; computational censoring is not proof of instability.

Report physical-state errors with full-state and fluctuation
normalization, POD floors, reduced-state survivor-median error, survivor-only
90% coverage and width, and draw completion fractions. Heat's training
trajectories and held-out trajectory remain separate. The original paper's
unbudgeted/default-tolerance completion definition is not silently equated
with this guarded evaluation.

For the same first 50 selected draws and mean operator, test boundedness to
10 training durations using the known input function. Do not claim long-time
accuracy where no truth is available. Report all failures and any skipped
geometry anchors. Examine convergence diagnostics before interpreting a change.

No automatic adoption or all-cases pass/fail rule. Improvement means a useful
accuracy/stability/uncertainty tradeoff; a bounded but biased or excessively
narrow prediction is not a success. No NODE, tumor experiments, or paper
rewrites are part of this request.

Heavy preparation, fitting and tests run remotely. Download and verify
artifacts, export plots, then remove only this study's Azure resource group
`rg-thesis-stability-transfer-20261112` and record estimated cost.

### Implementation checks before production fits

The shared helper differentiates each model's feature assembler; independent
Gaussian-algebra checks cover two input channels, time-unit changes, duplicate
anchors, stationary-anchor reporting and invalid inputs. Together with the
existing nonzero-mean/full-covariance tests and the study's integration guards,
12 focused tests passed on the worker. End-to-end plumbing checks used seed 45,
100 updates and three draws for heat at 1% noise and diffusion-reaction, both
with `contract20`; these are not scientific results or hyperparameter selection.

The first worker request in East US failed quota preflight because unrelated
work used eight regional cores. A single Central US worker was created instead;
no unrelated resource was modified. Its auto-shutdown schedule required the
VM's region explicitly rather than the resource group's region. Source staging
initially omitted `core/plotting/figures.py` along with figure directories; the
directory-only exclusion was corrected before the successful checks.

## Completed comparison (2026-11-12)

All 40 declared fits completed using source
`94e24bf2444daa5e9120cd33dd90a211da635549`. All five variants used identical
training data and POD bases within each case/seed, verified by hashes.
The downloaded archive contains 224 files (656,781,398 bytes), each verified
against its worker SHA256 digest. The complete numerical record is in
[`report_results/summary.json`](report_results/summary.json) and
[`report_results/results.csv`](report_results/results.csv); the manifest is
[`report_results/artifact_checksums.json`](report_results/artifact_checksums.json).
Reporting changes after the fitting revision do not change any posterior.

**Decision: keep the original hierarchical prior on these benchmarks.**
The tested fixed-Gaussian contraction priors are not a general upgrade.
Contraction helps several metrics relative to the fixed-Gaussian control,
but does not recover the original accuracy/stability tradeoff.
There was no additional tuning or selection after seeing these results.

### Forecast accuracy and metric definitions

The following are physical full-state relative errors (%) of the
Rao-Blackwell mean-operator trajectory, scored only after training:
`1 < t <= 2` for heat and `1 < t <= 3` for diffusion-reaction.
Values average seeds 42/43; heat training rows also average the five training
targets. The held-out forcing/IC case remains separate.

| Case | Hierarchical | Fixed | Contract 1 | Contract 5 | Contract 20 |
|---|---:|---:|---:|---:|---:|
| Heat 1%, training | 1.646 | 1.736 | 1.721 | 1.713 | 1.702 |
| Heat 1%, held-out | 1.381 | 1.256 | 1.260 | 1.266 | 1.297 |
| Heat 3%, training | 1.931 | 2.279 | 2.249 | 2.270 | 2.357 |
| Heat 3%, held-out | 1.625 | 1.673 | 1.675 | 1.694 | 1.768 |
| Heat 5%, training | 2.307 | 3.232 | 3.167 | 3.194 | 3.295 |
| Heat 5%, held-out | 1.906 | 2.317 | 2.320 | 2.332 | 2.396 |
| Diffusion-reaction 3% | 4.064 | 4.370 | 4.399 | 4.556 | 5.487 |

These are not the percentages from the paper's `aggregate_table.py`: that
table uses the error of the pointwise posterior-predictive median in reduced
coordinates, conditional on surviving draws. Integrating a mean operator
also differs from taking the median of nonlinear trajectories. The report
includes both definitions, with full-state and reduced normalization named
explicitly. On the paper-style reduced median metric, **all three contraction
strengths are worse than the original hierarchical prior in all seven groups,
for each seed separately** (42 matched comparisons), not just after averaging.
The two seeds do not establish population-level statistical significance.

For orientation, original versus contract20 reduced median errors (%) are:
heat training 1/3/5%: 1.905/2.549/3.329 versus 3.172/5.056/7.489;
heat held-out: 2.269/2.746/3.450 versus 2.807/4.672/6.508;
diffusion-reaction: 2.127 versus 4.051. These fresh float64/tighter-integration,
two-seed results should not be silently substituted for the older paper runs.

The best attainable physical-field reconstruction errors in these learned
affine spaces are 0.968/1.366/1.795% for heat training at 1/3/5% noise,
0.819/1.142/1.515% for its held-out case, and 3.431% for diffusion-reaction.
They are computed by least squares on the physical reconstruction basis.
For lifted heat, projecting the joint `(q, q^2)` state need not minimize
physical `q` error; that joint projection error is saved separately and is
not mislabeled as the physical floor.

### Boundedness and uncertainty

Every original hierarchical draw completed the original evaluation horizon.
The long window extends to ten training durations and has no truth-based
accuracy claim. Original versus contract20 completion fractions (%) are:

| Case | Short: original | Short: contract20 | Long: original | Long: contract20 |
|---|---:|---:|---:|---:|
| Heat 1%, training | 100 | 100 | 90.0 | 52.8 |
| Heat 1%, held-out | 100 | 100 | 93.0 | 53.0 |
| Heat 3%, training | 100 | 99.8 | 90.4 | 55.6 |
| Heat 3%, held-out | 100 | 100 | 92.0 | 56.0 |
| Heat 5%, training | 100 | 99.0 | 84.2 | 52.6 |
| Heat 5%, held-out | 100 | 100 | 86.0 | 53.0 |
| Diffusion-reaction 3% | 100 | 100 | 92.0 | 74.0 |

All failed draw-target solves hit the amplitude guard; none exhausted the
RHS budget or failed for another solver reason. These are escapes past
100 times the largest absolute training reduced observation, not a proof
of mathematical finite-time blowup. Each operator is evaluated on multiple
targets for heat, so these counts are not independent posterior draws:

| Variant | Short failures / 7,600 | Long failures / 1,900 | Long mean failures / 38 |
|---|---:|---:|---:|
| Hierarchical | 0 | 214 | 0 |
| Fixed | 75 | 1122 | 9 |
| Contract 1 | 56 | 1106 | 5 |
| Contract 5 | 48 | 1024 | 4 |
| Contract 20 | 24 | 859 | 1 |

All 190 short-window mean-operator solves completed. The 19 failed long-window
mean solves also hit the amplitude guard. No stationary anchors were skipped
in any contraction fit.

Survivor-only 90% bands must be read with their completion fractions and
widths. A real exception to an otherwise unfavorable comparison is low-noise
held-out heat: contract20 improves physical mean error from 1.381% to 1.297%
and reduced-coordinate coverage from 78.25% to 94.50%. However, relative band
width roughly doubles (0.084 to 0.168), reduced median error worsens, and
long completion falls from 93% to 53%. Most other original bands already
overcover rather than undercover; higher coverage alone is not an improvement.

### Interpretation and diagnostics

The fixed control shows that removing adaptive block scales is an important
part of the regression, not just adding the contraction preference. Posterior
median standard-deviation multipliers learned by the hierarchical model are:

| System | Constant | Linear | Quadratic | Input | State-input |
|---|---:|---:|---:|---:|---:|
| Heat, range over six fits | 1.501-1.568 | 0.906-0.976 | 0.049-0.061 | 3.652-3.872 | 0.230-0.256 |
| Diffusion-reaction, two fits | 0.022-0.032 | 0.033-0.054 | 0.648-0.681 | n/a | n/a |

The fixed control instead sets all these multipliers to one. The contraction
variants add correlated, nonzero-mean information to that fixed Gaussian,
not to the original learned hierarchy. The learned scales support the
interpretation that block-specific regularization matters here; they do not
prove that this is the only cause or that an untested combined model would win.
These small ROMs (5 and 3 modes) already work well without the extension.
The strong-damping diffusion-reaction trajectory also visibly biases the
third reduced coordinate. A local contraction preference neither guarantees
global boundedness nor automatically matches the geometry of every system.

Across all 40 fits, the absolute change between the last two 500-update loss
means was 0.022-0.160 times the last-1000-update loss standard deviation.
This is a coarse late-training drift check, not a certificate of variational
convergence or posterior correctness. Geometry uncertainty remains unmodeled,
and two acquisition/inference seeds are a limited robustness check.

The positive cylinder result therefore remains a regime-specific benefit,
not evidence that this prior should replace the original method everywhere.
No default algorithm, original benchmark outputs, tumor code, or paper text
was changed by this study.

### Artifacts, reproduction, and resource retirement

Nine figures plus the numerical report, definitions, and cloud record are
exported to `~/Downloads/stability_transfer_plots/`. The overview shows all
five variants. Trajectory panels show seed 42, the first training target or
the held-out target, and the first three modes, rather than a selected best
fit. Observations are black stars; the held-out panel shows only its initial
observation. Bands are shown for the original prior and contract20 only.

Raw posteriors, per-draw status records, and plotting arrays remain in the
session artifact directory `files/stability-transfer/results/`. They are not
committed. Given that directory, rerender with:

```bash
conda run -n prob_rom python experiments/stability_transfer/report.py \
  --results /path/to/stability-transfer/results \
  --output /path/to/stability-transfer/report
```

The declared runner accepts `--system`, `--schema`, `--seed`, `--variant`,
`--source-commit`, and `--output`; production defaults are 12000 updates,
500 posterior samples and 50 long-window draws. Reproducing the original
fits requires the recorded source revision, not relabeling a newer checkout.

The Central US F16as_v6 worker was verified deallocated at
2026-11-12 18:57:19 UTC; its eight-resource study group was verified deleted
at 19:00:26 UTC. No unrelated resource was modified. The creation-to-deallocation
interval gives an estimated compute upper bound of **$1.45** using the cached
public Linux rate of $1.235/hour. This excludes disk/IP charges and is not
billed spend. Timestamps and calculation inputs are retained in
[`report_results/cloud_run.json`](report_results/cloud_run.json).
