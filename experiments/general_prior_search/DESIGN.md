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
