# Shared latent states: constraints, optimization and interpolation

## Main findings

Adding weak form can substantially improve the shared latent interpolation
while maintaining close measurement agreement. It does not yet yield
reliable ODE rollouts. The study separates three contributing issues:

- The correlated formulation has severe optimization/conditioning problems;
  an alternative optimizer materially improves the same objective.
- Distinct near-stationary solutions can have the opposite training-objective
  and rollout-quality ranking. Longer optimization is not a complete remedy.
- The major GP interpolation defect persists after removing measurement noise.
  Its conditional uncertainty can understate shape error, but weak constraints
  can still pay the GP prior cost to produce useful corrections.

These findings support further study of shared latent states, not promotion
of this point-MAP prototype or a claim that weak-form inference fails.

## Scope

This study investigates three separate questions: whether adding weak form
helps the experimental shared-latent model; whether its poor results reflect
unfinished optimization or different local solutions; and whether conditional
GP uncertainty permits dynamics to improve interpolation between measurements.

The recovery point is `2124cd9`, also preserved as
`backup/shared-latent-before-joint-constraints-2124cd9`. This is session-only
experimental inference, not a replacement of production code, inference
defaults, canonical benchmark outputs or neural checkpoints.

The two datasets are the fixed chemotherapy comparison observations
(`matched`, A) and replacement times generated with seed 43 (`resample43`, B).
Both have 80 observations between days 5 and 70, the same four-mode basis,
nominal treatment, and prediction through day 110. The reduced observation
noise residual is held fixed by column index; they are not independent noise
realizations. "Matched" means methods share the comparison data, not that a
prediction matches observations.

The prototype conditions a time-only RBF GP on measurements with fixed
data-only MAP hyperparameters and declared measurement-noise variance.
It adjusts whitened latent states and analytically profiles the conditional
Gaussian operator MAP. It does not marginalize operators or infer a full
latent-state/hyperparameter posterior. Fixed scaled operator priors and
fixed discrepancy SDs remain those of the original 40-state prototype,
not the production model's hierarchical scales.

All operator integrations in these diagnostics are post-fit evaluation.
No ODE integration is used in the strong/weak fitting objectives.
No forecast errors select hyperparameters, initializations or methods.

## Adding weak constraints to the shared trajectory

The original shared-latent prototype used strong constraints only, although
production uses both strong and weak constraints. This experiment explicitly
adds the weak block while retaining the same state variables, GP fits,
operator priors and strong discrepancy SDs.

Both blocks now use the **same smooth conditional GP trajectory**, rather than
the piecewise-linear path used in the preceding weak-only experiment.
There are 20 compact `psi = (1-z^2)^6` test functions with radius
`65 * 20/199` days. The weak target is `-integral(psi' X)` and the design
integrates `[1, X, alpha, alpha X]` against psi. No derivative mean constructs
the weak target; integrating that mean is an independent IBP identity check.

Two versions of the addition are reported:

| Formulation | Covariance treatment |
|---|---|
| `combined_blockdiag` | Full covariance within each block, cross-block covariance omitted |
| `combined_correlated` | GP strong/weak functional cross-covariance and common discrepancy cross-covariance retained |

The weak covariance includes its GP target uncertainty conditional on measured
and latent nodal states, plus `slack_i^2 M M^T`, where M integrates psi against
piecewise-linear nodal hats. Only the discrepancy uses these hats, not the
latent mean trajectory. The correlated cross block additionally contains
`slack_i^2 M^T`. No additional combined-block nugget was added in these runs.
The existing state/strong roundoff nuggets remain.

Weak weight is one, not the production chemo weight of eight. The two versions
are declared comparisons, not candidates selected by forecast. Omitting
correlation is a composite/independence approximation that can count overlapping
information twice. Retaining correlation is still **not an exact full random
GP residual likelihood**: off-node features use conditional means, whereas
their full randomness would introduce operator-dependent covariance. This
plug-in approximation preserves analytic conditional operator profiling.

### Longer runs and different initializations

There are 48 predefined fits: two datasets, 40/80 uniform state grids, three
formulations including strong-only, and four starts. Starts are zero whitened
states, the saved same-grid fit, and two `0.5 * Normal(0,I)` perturbations
(seeds 101 and 202). All starts are paired between formulations. Maximum
iterations increased from 200 to 2,000, with `ftol=1e-13`, `gtol=1e-6`.
Iteration-200 states and objective traces were preserved.

All 48 terminal states were recovered. Forty-four rollouts were scored;
the four A/80/block-diagonal candidates exceeded the unchanged 200,000-RHS
evaluation budget. Three correlated terminal diagnostics raised before
serialization; identical-input replays with cached preparation arrays
recovered exactly matching original objective traces. The initial failures
and one roundoff-different replay remain archived and were not substituted
or used for selection.

For each formulation/grid/dataset, the reported start minimizes its own
training objective. Absolute objectives from different formulations are not
compared. Full-field training / forecast errors:

| Constraints | Nodes | A | B |
|---|---:|---:|---:|
| Strong only | 40 | 9.91% / 26.10% | 9.41% / 27.72% |
| Strong only | 80 | 16.41% / 32.94% | 10.53% / 10.90% |
| Strong + weak, block diagonal | 40 | 8.29% / 32.27% | 4.14% / 33.66% |
| Strong + weak, block diagonal | 80 | Evaluation budget exceeded | 4.53e13% / 4.28e22% |

Longer strong-only optimization does not substantially repair the old result.
Strong40 returns essentially the same solution from all four starts.
Strong80/A decreases its zero-start objective by about 0.491% after iteration
200, without a substantial rollout improvement. Strong80/B has worse
random-start solutions, establishing initialization sensitivity without
demonstrating a better attainable solution than the saved basin.

The B/80/block-diagonal comparison is particularly informative. Zero/saved
starts attain approximately **3.97% training / 6.01% forecast**, but the
random-start catastrophic solution has the **lower** training objective:
**177.380868 versus 177.719446**, a difference of about 0.191%.
Selecting the attractive forecast would violate the stated selection rule.
This is evidence of both distinct solutions and imperfect alignment between
the training criterion and ODE trajectory quality.

The 32 non-correlated runs stopped by function tolerance, not the requested
absolute gradient tolerance; final gradient infinity norms range from about
5.5e-6 to 4.1e-3. Thus "converged" flags are not strict stationarity certificates.
All 16 correlated runs hit 2,000 iterations, with gradients approximately
4.15e2-1.52e5 and worst covariance condition numbers around 3e11.
Their initial poor rollouts cannot establish a failure of a converged
correlated model.

Raw quadrature convergence does not guarantee inverse-covariance stability.
Although the checked raw integral relative changes are at most 6.5e-10,
near-cancelled correlated covariance directions amplify Gauss8/Gauss16
differences: standardized covariance changes reach 0.118-0.178 in spectral
norm. At identical final states, profiled objectives change by up to 0.0704%.
This is a material numerical limitation, not remedied by the optimizer's
success flag or small raw quadrature error.

### Exact fixed-operator conditional-state audit

For fixed O, the state-fitting problem is a strictly convex quadratic.
An augmented least-squares solve therefore finds its global conditional
minimum without ODE integration or nonlinear iterations. We evaluated both
each selected operator and the archived direct-integration reference operator.
The latter is diagnostic only, not a proposed fitting method or a new prior.

Holding the selected non-correlated operators fixed and solving for states
exactly changes objectives by at most approximately **1.5e-8**. This provides
a stronger conditional convergence check than a function-tolerance flag.
In contrast, the initial correlated objectives drop substantially:
A40 739.84 to 537.09; A80 65,262.79 to 44,388.86;
B40 856.36 to 508.51; B80 1,733.21 to 1,164.45.
Those correlated runs are demonstrably unfinished optimization.

Even with globally optimal states for the particular direct-reference
operator, the original strong40 objective is **513.37 versus 36.99** on A
and **281.09 versus 39.22** on B. Thus the existing criterion prefers the
poor-rollout operator to this much more accurate reference, even after
removing conditional-state optimization error from that comparison.
This does not prove a global optimum or rule out other accurate operators.

### Same-objective curvature-aware optimizer

To test conditioning rather than merely repeat L-BFGS longer, six predefined
archived endpoints were continued with trust-region nonlinear least squares.
Joint variables are the same whitened states w and prior-standardized
operators eta. The residual vector is
`[w, eta, whitened_design(w) eta - whitened_target(w)]`.
Half its squared norm is exactly the original joint MAP objective; analytic
Jacobians include the state/operator bilinear terms. There is no new
likelihood, prior, covariance, GP preparation or ODE fit.

Controls comprise the four selected correlated endpoints and the two B80
block-diagonal basins. SciPy `least_squares(method="trf", x_scale="jac")`
used linear loss, `ftol=xtol=1e-12`, `gtol=1e-6` and at most 1,500 function
evaluations. The terminal states were reprofiled with the unchanged operator
solve before scoring.

The following are the initial six optimizer-control endpoints; the final
continuations below supersede the capped A80 and B40 correlated endpoints.

| Case and formulation | Previous objective | Reprofiled objective | Field fit / forecast |
|---|---:|---:|---:|
| A40 correlated | 739.84 | 242.79 | 5.01% / 43.85% |
| A80 correlated | 65,262.79 | 7,002.53 | 1.11e5% / 2.09e8% |
| B40 correlated | 856.36 | 138.34 | 19.73% / 64.06% |
| B80 correlated | 1,733.21 | 673.37 | 8.76e5% / 7.67e8% |
| B80 block diagonal, zero start | 177.719446 | 177.719446 | 3.97% / 6.01% |
| B80 block diagonal, random101 | 177.380868 | 177.380868 | 4.53e13% / 4.28e22% |

Correlated objectives decrease by **61-89%** on exactly the same model.
Optimization conditioning is therefore a major confound in their original
outcomes. A40's training rollout improves from 1,228.7% error to 5.01%.
It would be incorrect to describe the original correlated endpoints as
optimized-model results.

This does not establish a complete optimization rescue. A80 and B40 still
hit the evaluation limit; A40 and B80 stop by function tolerance, not the
requested gradient tolerance. Their profiled gradient infinity norms are
approximately 2.99e-4, 2.08e3, 2.63e2 and 2.75e-2, respectively.
Column-normalized Jacobians still have condition numbers around 1e6-1e7.
Better search behavior does not eliminate covariance cancellation or certify
a global optimum.

The two B80 block-diagonal endpoints each stop after three evaluations,
with changes of only about 1e-9 objective units. The catastrophic basin
retains the lower training objective. This ranking mismatch is not explained
solely by L-BFGS conditioning or unfinished conditional-state optimization.

### Final continuation of the two capped cases

Only A80 and B40 were restarted from their exact saved joint terminal
states, including unprofiled operators, allowing up to 10,000 further
function evaluations with a shared 15-minute wall-clock ceiling.
All other settings remained unchanged. This is an endpoint restart, not a
restoration of SciPy's internal trust-region history.

Both stopped by function-tolerance plateau, not the enlarged evaluation or
time limit. A80 used 2,162 further evaluations, lowering its objective from
7,002.53 to **5,553.28**. Its profiled gradient remains **10.58**, and
post-fit scoring exceeds the unchanged RHS budget. B40 used 112 further
evaluations, lowering 138.336778 to **138.334442**, with profiled gradient
**0.00485** and field fit/forecast **20.25% / 68.30%**.

Neither meets the requested absolute gradient tolerance, so stationarity
and conditioning remain unresolved. The remaining difficulties cannot
simply be attributed to the earlier small evaluation cap, but neither
are these certified MAP optima or proof of a floating-point floor.
The prescribed unchanged stopping criterion is reached without a reliable
rollout rescue. No additional fitting, covariance adjustment or start
selection was performed after these two continuations.

## Does the GP mainly overfit measurement noise?

The exact reference curve was reconstructed by projecting the original
simulation snapshots through the frozen basis and applying the same cubic
interpolant as the frozen experiments. It agrees with all 400 stored truth
values. The simulation knots are spaced 0.5 days apart; values between them
are a reference interpolation, not additional FOM solves.

Two paired data-only GP starts (saved hyperparameters and the original
training anchor) were used for both noisy observations and a diagnostic
replacement by clean reference values at exactly the same observation times.
The known positive likelihood variance, numerical nugget, prior anchors and
bounds remained unchanged. Thus the control removes the particular observed
noise perturbation, not the likelihood variance. Selection uses each
data-only training MAP objective.

| Data-only GP | A: reduced training-truth error | B: reduced training-truth error |
|---|---:|---:|
| Original noisy observations | 24.48% | 17.23% |
| Clean observations, same likelihood variance | 24.46% | 17.09% |

The dominant matched-mode lengthscale changes only from 1.075 to 1.094 days.
Removing the perturbations does not remove the large overshoot following
treatment. Ordinary fitting of measurement noise is therefore not the main
explanation for this defect. The chosen GP family, acquisition geometry and
plug-in hyperparameter treatment are more relevant limitations.

Saved abnormal GP line-search terminations were investigated rather than
ignored. Paired starts reproduce essentially the same noisy objectives and
curves. Some reported terminations and finite-difference gradients remain
limited by numerical precision; this is not a proof of a global optimum.
It does show that the gross interpolation error is not resolved by these
data-only optimization continuations.

## Does the conditional GP admit the correct interpolation?

These are **pointwise conditional latent-function GP bands given measurements
and fixed hyperparameters**, excluding predictive observation noise and
hyperparameter uncertainty. They are not posterior bands from the new
joint physics inference. Coverage is descriptive for these two designs,
not a general calibration assessment.

For A, the first-treatment observation gap is days 20.206-23.290.
In the dominant mode, the median absolute reference error is **5.44 GP SD**;
only **1.75%** of 401 sampled gap times have the reference inside the
pointwise 95% band. The GP is more uncertain there than at measured times,
but not sufficiently uncertain about its incorrect shape.

For B, the corresponding gap is days 19.606-22.140. All four reference
coordinates are inside their pointwise 95% bands at all 401 sampled gap
times. The old strong-only 40-state latent fit improves reduced
training-truth error from **17.23% to 5.42%**. This is direct evidence that
the intended adjustment can work for one observation design.

Pointwise support does not establish joint trajectory support. We also
computed full-covariance nodal Mahalanobis costs, separating modes and
gap-only from whole-window constraints. Some large full-reference costs,
particularly the fourth mode at 80 nodes, are dominated by near-null
RBF covariance directions and depend strongly on the roundoff nugget.
Their enormous exact magnitudes are not numerically robust evidence, nor
do they establish that every physically adequate smooth correction is
excluded. A cubic reference need not itself have the smoothness preferred
by a squared-exponential GP.

Conditioning again on fitted MAP states reduces GP variance but excludes
uncertainty in those estimated states. Such a conditional covariance must
not be presented as a full physics-informed posterior.

### Does adding weak form actually correct the latent states?

Yes, in some cases it does, rather than merely changing the operators.
Using the longer-run, objective-selected block-diagonal fits:

| Dataset and state count | Strong-only latent training-truth error | Strong + weak latent training-truth error |
|---|---:|---:|
| A, 40 | 22.77% | 5.97% |
| A, 80 | 22.88% | 21.73% |
| B, 40 | 5.42% | 4.83% |
| B, 80 | 6.36% | 4.99% |

These are reduced-coordinate errors of the **latent curve**, not the
full-field ODE errors reported earlier. On A40 the observation residual
remains small, changing from 0.0556% to 0.0735%, while the major treatment-gap
error decreases substantially. The change from strong to combined has
dominant-mode gap RMS approximately 5.09 data-only GP SD; latent prior cost
increases from 19.33 to 70.61. This is the intended physics-driven adjustment.

The data-only 95% bands are not hard constraints. A useful smooth correction
can move outside them and pay a prior penalty; the huge cost of imposing
every exact cubic-reference nodal value is not the cost of all useful
corrections. The improvement is not uniform: B40's first-treatment gap
gets worse despite its small improvement over the entire training interval.

Even substantially better interpolation does not ensure a good learned ODE.
A40's ODE forecast worsens despite its improved latent trajectory. In B80,
the objective-selected catastrophic solution and the unselected zero-start
solution have continuous training paths differing by only approximately
**0.951%**, but operators differing by **344.6%** relative to the zero-start
operator norm. The attractive zero-start forecast is not a selected result.
This demonstrates sensitivity in the operator/dynamics fit beyond simply
whether the GP path is close to the reference.

## Frozen-trajectory operator controls

To distinguish interpolation quality from operator estimation, operators
were fitted to three fixed curves: the clean reference, the data-only GP
mean and the saved strong-only 80-state GP path. There is no nonlinear
latent optimization in these controls: each row is a regularized linear
operator solve with an independently checked stationarity condition.

Strong controls use rates at 80 nodes and the saved conditional derivative
covariance plus discrepancy. Weak controls use 20 compact degree-12 bumps
and state-only integration by parts on each continuous curve, with full
overlap covariance `slack_i^2 M M^T`. Integration is partitioned at source
knots, input knots, latent knots and bump boundaries. Gauss12/Gauss16
agreement and integration-by-parts identities are checked.

The weak frozen-curve control deliberately treats its curve as known and
does not add GP weak-target uncertainty. Therefore it is a diagnostic of
fixed-trajectory regression, not an otherwise identical ablation of the
new joint model.

Full-field relative L2 percentages, training / forecast:

| Fixed curve and fitting constraints | A | B |
|---|---:|---:|
| Clean reference, strong80 | 5.65% / 56.99% | 5.68% / 60.35% |
| Clean reference, weak20 | 1.37% / 15.87% | 1.37% / 15.91% |
| Data-only GP, strong80 | 57.81% / 775.54% | 43.11% / 125.53% |
| Data-only GP, weak20 | 9.39% / 53.68% | 1,176.01% / 104,059.01% |
| Saved strong80 GP, strong80 | 16.28% / 33.62% | 10.53% / 10.89% |
| Saved strong80 GP, weak20 | 4,924.11% / 1,750,909.97% | 3.74% / 5.71% |

Clean-reference controls use privileged information unavailable to inference.
They are not candidate benchmark results. Nevertheless, weak operator
regression works much better when supplied the actual training shape.
Conversely, correct interpolation alone does not guarantee the best rollout
under a particular strong residual weighting. Interpolation, residual
weighting and subsequent dynamical amplification are separate issues.

## Artifacts and reproducibility

Artifacts are preserved under session
`270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics/`.
The study uses the `prob_rom` conda environment and one numerical thread.

`joint_physics_uncertainty.py` and `joint_physics_uncertainty/` contain
paired GP fits, conditional covariance arrays, nodal support diagnostics,
the compact exact reference knots, source hashes, and day-18-to-25 plots.
Its `selected_path_extension/` subdirectory compares the new selected
strong/combined trajectories and the explicitly unselected B80 contrasts.
`joint_physics_oracle.py` and `joint_physics_oracle/` contain the fixed-curve
operator controls, quadrature checks, saved operators and scored predictions.
`joint_physics_model.py`, `joint_physics_runs.py` and `joint_physics_study/`
contain all 48 fits, cached preparation arrays, convergence audits, terminal
replay records and the complete `completed_summary.json`.
`joint_physics_fixed_operator.py` and `joint_physics_fixed_operator/` contain
the exact fixed-operator conditional-state solves and their objective terms.
`joint_physics_optimizer_control.py` and `joint_physics_optimizer_control/`
contain the identical-objective trust-region controls, exact-cache loader,
analytic Jacobians, all accepted states and terminal/profiled diagnostics.
`joint_physics_optimizer_continuation.py` and
`joint_physics_optimizer_continuation/` preserve the final two endpoint
continuations and their plateau/gradient/scoring diagnostics. The original
48-fit and six-control artifacts were retained unchanged.
