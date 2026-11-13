# Shared Gaussian prior search

## Question and scope (declared before fits)

Can one simple, approximately shared Gaussian prior support the existing Euler,
heat, diffusion-reaction (`burgers_2d`, not viscous Burgers), and high-rank
multi-Re cylinder examples? The successful cylinder local-contraction prior is
an optional physics-informed comparison, not the general-method requirement.
No tumor, NODE, manuscript edits, production-default changes, new basis-selection
rules, or changes to the observations are part of this study.

The protected starting commit is `c3f9d0f`, tagged `pre-general-prior-search`.
Earlier fixed/global shrinkage, warm-start, shifted-feature, modular-GP, and
contraction investigations remain part of the development history. This search
is not independent evidence about the already-inspected Re200 trajectory.

## Candidate families

The complete initial grid is `run.py::CANDIDATES`; every setting is applied to
all four development systems, not chosen separately by system.

- Original dimensional block hierarchy, log-scale widths 3 and 1.
- Divide the original column SDs by the square root of the block's feature count.
- Scale each output by its within-trajectory observed fluctuation SD / T, and
  each feature by its training GP RMS and square root of block feature count.
- Replace that output rate scale by the quadrature sum of observed SD / T and
  training-only GP derivative RMS. Test log-scale widths 0.5, 1 and 3.
- Repeat selected families in coordinates centered at the pooled training mean;
  this is an exact feature-coordinate change, not a new POD basis.
- Modest global SD factors 0.3 and 3; optional linear mean -d/T times the identity
  with d = 0.1, 1 or 3. In centered coordinates this means mean reversion toward
  the training mean. It is not a Jacobian/tangent constraint or a guarantee of
  stable dynamics, and may be inappropriate for Euler.
- A secondary generic feature-support covariance: regularized normalized feature
  Gram matrix, NOT its inverse. Test ridge fractions 0.01 and 0.0001, within
  blocks (retaining learned block scales) or across all features (fixed scales).
- Fixed rate-scaled controls at global SD factors 0.3, 1 and 3, to separate the
  role of scaling from learning the block hierarchy.

For the simplest rate family, coefficient (i,j) in block b has conditional SD
`kappa_b * rate_i / (RMS(feature_j) * sqrt(n_b))`. This equalizes each block's
prior-predictive contribution to a given output rate and is invariant to positive
changes of modal, input, and time units. Centering and correlation are optional
and reported separately. Data-only GP statistics are frozen empirical-prior
inputs: their uncertainty is not propagated, and they use training data twice.
No forecast truth, tangent projectors, fitted stability Jacobians, or Reynolds-
specific knowledge enters these priors.

The shared model retains exact Gaussian operator marginalization. With a supplied
factor R, O_i = M_i + R_i theta_i; the existing block multipliers scale columns
of R through the latent theta prior. For centered blockwise roots, block meaning
is preserved in centered feature coordinates. Full mixed roots use fixed scales
to avoid pretending their latent groups are physical operator blocks.

## Development and inference

Development seed 42, original observations, basis ranks, polynomial libraries,
GP priors and tempering rules:

| System | Development schema | Rank |
|---|---|---:|
| Euler | dense_low_noise, 250 observations, 1% | 6 |
| Heat | sparse_medium_noise, five trajectories, 20 each, 3% | 5 |
| Diffusion-reaction | dense_medium_noise, 60 observations, 3% | 3 |
| Cylinder | Re150/300, 250 observations each, 1%; [16,17] training | 25 |

Every initial candidate receives full joint GP/operator inference on all four
systems: float64, 12,000 SVI updates, the existing 0.003 learning rate, 200
posterior draws and 20 long-horizon draws. All start at the same data-only GP
MAP and unit block multipliers. This differs from the original small-system
`init_to_median` initialization; original-prior refits under this same optimizer
initialization are the matched controls. Previously published cold-start
results are secondary references, not substituted controls.

Conditional/prior-predictive diagnostics can explain behavior but cannot
establish a successful candidate. Checkpoint before evaluation; retain all
negative results and nonfinite/failed fits. A failed optimizer is not a bounded
or zero-error forecast. Record GP rate drift and late loss drift; conspicuously
unfinished optimization needs a disclosed continuation, not a prior verdict.

## Evaluation and promotion

Reuse `stability_transfer.run.evaluate`, extended to honor the cylinder's
established 2x observed-amplitude cap and held-out Re200 score interval. Other
systems keep their existing 100x cap. These percentages are not equivalent
cross-system stability thresholds. RK45 tolerances remain 1e-7 / 1e-9; work
budget is 20,000 RHS calls per training duration (ceiling), including long
checks over ten training durations. Report amplitude and work failures separately.

Primary cylinder accuracy is full-field fluctuation-relative forecast error.
Other systems report full-state AND reduced-coordinate errors so mean fields
cannot hide a bad dynamical fit. Report mean-operator and survivor-median
trajectories, survival, survivor-only 90% coverage and width, and long behavior.
For cylinder, retain the finite-sample mean of the same sampled initial states
for mean-operator evaluation; other systems retain the observed initial state.

Desired development outcome: both cylinder training-Re mean errors <=5%, at least
90% short-draw survival at each, and no substantial loss on the smaller systems
(mean physical AND reduced median errors no more than 25% above their matched
original-prior values, short survival >=95%). These are investigator-defined
search targets, not the user's publication threshold or a license to hide
partial improvements. Re200 is reported, not used to select the initial winner.

Shortlist at most three settings, prioritizing simple uncorrelated zero-mean
families if they meet these targets; otherwise use the accuracy/survival tradeoff
across all four systems and explicitly report that the target was not met.
Freeze the shortlist before confirmation. If the first grid fails, use its
diagnostics to declare a separate scientifically motivated follow-up; do not
silently replace failed candidates or claim conditional success.

## Confirmation, provenance and persistence

Confirm shortlisted settings and the matched original prior with 500 draws:
remaining heat noise levels, remaining Euler sparsity/noise schemas, cylinder
r29 seed42 and r25 seed43, and fresh acquisition/inference seed47 on the four
development systems. Also test cylinder r25 seed47 on the later [20,21] training
window and forecast through 23. Seed47 changes noise/timing and inference, not
the underlying PDE truth; the later periodic window is robustness, not a new
flow regime. If confirmation motivates another change, label it development and
do not continue calling those cases held back.

Use isolated, purpose-tagged Azure CPU workers; keep the Mac light. Record source
commits, requests, data fingerprints, checkpoints, all posteriors, and failures.
Download and checksum results before retiring only this study's resources.
Export readable plots with observations as black stars. Report both successes
and limitations, approximate compute cost, and whether a shared-prior main-paper
story is actually supported.

## Pre-development amendment: the clock in a damping mean

Before any full development fit, add six generic mean controls. The original
29 settings use damping relative to the whole training duration T. For a rapidly
oscillating trajectory this clock can be much slower than the observed dynamics,
making a damping mean negligible compared with the rate-scaled prior variance.
The additional controls use each mode's rate_scale / observed_amplitude as its
inverse clock. Test damping multipliers 0.3/1/3 for centered rate scaling, plus
multiplier 1 with centered amplitude scaling, dimensional scaling, and blockwise
feature-support covariance (ridge 0.01). No tangent or Jacobian is computed.

The initial full grid is therefore **35 settings x four systems = 140 joint
fits**, followed by the same frozen-shortlist confirmation. Four seed45
100-update plumbing smokes were started under the original 29-setting code;
they are not used to select this amendment or as scientific results. The
amendment follows from matching the prior mean's clock to its variance, not from
inspecting any full-fit forecast.

## Execution record (initial grid completed)

All four seed45 plumbing fits completed successfully (100 updates, two draws;
not accuracy evidence). The numerical development source is
`fb670542c10956cd3d6a2cd80b3a2bf92bce1821`. Its last change makes the cylinder's
request metadata explicitly record the initial-state uncertainty already used
by the evaluator, and adds block timings; it does not change the sampled states.

The 140-fit grid started on four isolated Standard_F16as_v6 workers. Observed
1000-update cylinder blocks took approximately 13 minutes, so the same frozen
jobs were redistributed across nine workers (centralus, eastus, eastus2,
westus3, westus2, northcentralus, southcentralus, canadacentral, westus).
No experimental setting or iteration count changed. The 16 started fits stayed
on their original workers and resumed their saved SVI states; unstarted fits
moved to the new workers. Each worker uses four disjoint four-CPU slots.
The original and redistributed job manifests have the same 140 IDs/arguments.

The queue interruption is an operational event, not a new inference run or a
forecast-based selection. An independent two-mode unit test compares an
uninterrupted 200-update fit with a 100+100 continuation, including the complete
loss history, GP/block-scale samples, and Gaussian operator samples (agreement
to 1e-9 relative tolerance). Failed fits and failed forecast integrations remain
separate report entries; missing or escaped forecasts never become zero errors.

## Initial development results (2026-11-13)

**All 140 fits completed, but none met the shared-prior development targets.**
The report contains 210 training/held-out fit-groups, not 210 independent fits.
Every fit retained the declared source, 12,000 updates, 200 posterior draws and
20 long-horizon draws. Comparisons use matching training-data/basis fingerprints.
Successful inference execution does not imply successful forecast integration.
No shortlist was promoted and no confirmation fits were run.

### Cylinder: excellent mean trajectories, unusable sampled forecasts

The following are full-field **fluctuation-relative** errors after the training
interval, reported separately for the two training Reynolds numbers.

| Prior | Re150 mean error | Re300 mean error | Re150 short survival | Re300 short survival |
|---|---:|---:|---:|---:|
| Original | failed | failed | 0/200 | 0/200 |
| `rate-centered-h0.5` | 3.202% | 3.897% | 0/200 | 0/200 |
| `dimensional-clock-d1` | 1.930% | 1.828% | 0/200 | 0/200 |
| `full-support-0.0001` | 46.507% | 77.680% | 191/200 | 142/200 |

The corresponding POD reconstruction floors are 1.895% and 1.817%. The simple
rate-clock damping mean nearly reaches those floors, but **a good mean operator
is not a usable posterior over forecast trajectories**. Full feature-support
shrinkage improves short survival at the cost of severe mean bias; its long
survival is only 1/20 and 0/20. All recorded cylinder short-draw failures in this
grid are amplitude failures, not RHS-budget exhaustion.

Re200 also remains poor: rate-centered and dimensional-clock mean errors are
152.109% and 72.237%, respectively, with zero sampled survival. Full-support
ridge 0.0001 has a failed mean and 38/200 short survivors at Re200. These are
reported development results, not independent held-out confirmation after the
earlier investigations. No survivor-only band is presented as unconditional UQ.

### Smaller systems: retain the hierarchy; centering is not harmless

Matched-original mean full-state errors are 1.230% (Euler), 1.864% (heat), and
4.188% (diffusion-reaction); survivor-median reduced errors are 7.540%, 1.995%,
and 2.373%, respectively. All their training-target short draws survive.

`balanced-h3` essentially preserves these results: mean-error ratios to original
are 1.000, 1.000 and 1.002; reduced-median ratios are 1.000, 1.001 and 1.016.
Changing base block scales alone can be offset by the learned block multipliers,
so this near-equivalence is not evidence that arbitrary prior changes help.
Uncentered `rate-h3` improves diffusion-reaction mean error to 3.894% with 99.5%
short survival, but increases heat's reduced-median error from 1.995% to 3.307%.
Centered rate families often damage diffusion-reaction substantially.
Fixed-scale and full-support controls also lose important small-system
performance. Thus the original/shared hierarchy remains the reasonable baseline;
the cylinder problem has not been solved by replacing it with one new family.

### After-the-fact diagnosis and rejected explanation

For `rate-centered-h0.5`, an additional 20 sampled operators at each training Re
were integrated from the **same fixed observed initial state**. Both cases still
had 0/20 survivors, all amplitude failures. Initial-state uncertainty alone
therefore cannot explain the failure.

At 24 training anchors, using observed mode SD to scale perturbations and GP
rate scales for vector fields, normal-direction derivative SD was 0.07004
versus 0.002979 tangentially: approximately 24 times larger. Conditional-mean
variation was much smaller (0.003849 normal, 0.000387 tangential). This points
to the conditional Gaussian operator uncertainty, particularly off the observed
paths, rather than only GP-hyperparameter variation. It is a local uncertainty
diagnostic, **not** a Floquet analysis or proof of global instability.

The proposed observation-noise/rate-floor explanation was checked and rejected
for this candidate. In none of the 25 modes did observed amplitude/T exceed GP
derivative RMS; the latter was 19--147 times larger. Observed and GP-smoothed
amplitudes were close, and feature RMS values were not near machine zero.
No noise-floor correction is justified by these data.

The largest absolute recorded late-loss change was 0.095 within-block SD; the
largest GP-rate relative shift was 0.276. These diagnostics do not reveal gross
late loss drift, but are not proofs of SVI convergence or posterior calibration.

### Interpretation and next experimental direction

This is a negative result for the **tested grid**, not proof that no simple
general prior can work. It does not support a claim that one new generic prior
now handles all four systems with reliable UQ. The earlier specialized
contraction-prior results remain an optional physics-informed comparison;
they used a different run budget and are not matched controls in this grid.
No production defaults or manuscript claims have been changed.

A separately declared next round could combine full feature-support covariance
with a simple damping mean, or constrain quadratic strength relative to linear
strength while retaining hierarchy. Naively rescaling quadratic columns can be
undone by learned multipliers, so that hypothesis needs a real prior change and
a matched control. **Neither direction has been implemented or fitted.**

### Preservation, compute and external blocker

All 140 metadata sets are local. A streaming SHA256 audit verified 873 files,
representing **97 complete nine-file artifact sets**. The remaining 43 fits
have complete summaries but still require their posterior, prior-root, plotting
and checkpoint files from worker disks. The exact per-job recovery inventory
and hashes of all 700 collected metadata files are retained in the private
operational workspace. Do not delete the disks before recovery.

At 2026-11-13 06:03:06 UTC, Azure's subscription API reported **Disabled** and
all nine workers were deallocated. The reason is not established; credit
exhaustion must not be inferred from this status. No billing restrictions were
changed, no subscription was switched, and no resources were deleted.
Further experiments and the remaining downloads require subscription recovery.

Assuming continuous allocation from each worker's creation until that
confirmed-deallocated snapshot gives 46.397 VM-hours and a conservative
**USD 54.05 compute estimate** at the regional public Linux F16as_v6 rates
retrieved on 2026-11-13. This is not an invoice or an exact shutdown timestamp.
Disk, public-IP, network, tax and account-specific pricing are excluded;
deallocation does not eliminate all ongoing resource charges.

### Report artifacts and reproduction

`initial_results/` contains all-candidate CSV/JSON summaries and the heatmap,
per-Re cylinder metrics, selected mean trajectories, the accuracy/survival
comparison, diagnostic outputs and sanitized provenance. Blank CSV mean errors
are failed forecasts, not zeros. The trajectory figure subtracts an observation
mean only for display and shows the two modes with largest observed temporal
variation. It does not display uncertainty bands when all draws fail.
The metadata-only full-support case appears in the scalar comparison, not as
a fabricated trajectory. Black stars at Re200 are explicitly held-out data.

From the repository root, with the original collected-results directory
available and the `prob_rom` environment active:

```bash
python experiments/general_prior_search/report.py \
  --results /path/to/collected/development --output /path/to/report
python experiments/general_prior_search/plot_initial.py \
  --results /path/to/collected/development --output /path/to/report
```
