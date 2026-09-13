# Full-feature covariance correction

This incremental follow-up to
[the four-hypothesis study](SHARED_LATENT_HYPOTHESIS_TESTS.md) investigates
the missing weak-feature uncertainty. The pre-study recovery point is
`6523fd4`, preserved as `backup/shared-latent-before-fullcov-6523fd4`.
All 48 chemotherapy refits, post-fit evaluations, six known-system fits,
and two bounded optimization controls are complete.

The correction reverses the runaway preference in the B80 block-diagonal
case and yields a selected forecast error of 6.217%. Other results are
mixed, and 80-point correlated fits remain underoptimized. There is no
production promotion or added physical restriction.

## Declared comparison

Observations, GP hyperparameters, measurement noise, nodal state coordinates,
mean-function maps, operator priors and discrepancy models remain fixed.
Both strong and weak constraints remain present. The chemotherapy
discrepancy is not replaced by the toy's continuous OU discrepancy.
ODE solves are allowed only for synthetic generation and post-fit evaluation.
Production code, defaults and canonical benchmark results are unchanged.

The frozen-state audit separates two changes: restoring uncertain weak
features, and restoring strong/weak cross covariance. Its block-diagonal
full-feature variant omits strong/weak cross blocks but retains cross-output
weak covariance. A pair of useful/runaway B80 solutions and the selected A40
solution are held completely fixed while their densities are evaluated.

All chemotherapy fits use the frozen nominal-dose, 1%-noise experiment.
A denotes the original matched acquisition and B the replacement observation
schedule. Both contain 80 observations. The 40/80 grid sizes below count
latent-state points, not observations or patients.

The declared refits compare the known scalar system at 40/80/160 nodes,
then the two frozen chemotherapy acquisitions at 40/80 nodes. The chemo
comparison uses target-only block-diagonal, target-only correlated and
full-feature correlated covariance, with the same three archived operator
starts in each comparison: zero, saved and random101. Every result is
retained; selection uses its own training objective, never forecast error.

After the frozen audit and before any chemotherapy refit, an explicit
amendment adds full-feature block-diagonal fits. These twelve additional
controls complete the two-by-two feature-uncertainty/correlation comparison,
bringing the total from 36 to 48. They are an outcome-informed follow-up
to the frozen ranking, not retroactively labeled part of the initial
declaration. The original protocol remains unchanged in the artifact record.

## Normalized objective and conditional-state control

At fixed operators O, the residual mean is `r = b(O) + J(O) w`, where w
contains the standard-normal nodal latent coordinates. Integrating the
uncertain off-node GP functionals gives a covariance R(O), including the
unchanged discrepancy. Unlike the target-only approximation, this covariance
depends on O and can couple different output coordinates.

The joint MAP cost, up to fixed additive constants, is
`0.5||w||^2 + 0.5||O/prior_std||^2 + 0.5 r^T R^-1 r + 0.5 logdet(R)`.
The log determinant must not be discarded: this is a normalized Gaussian
residual likelihood, not just a change of least-squares weights.

This likelihood is not generally Gaussian in O. Consequently the old
conditional Gaussian operator profiler is not valid for the full-feature
variant. However, **the latent-state problem is exactly quadratic at fixed
O**. With `S = R + J J^T`, its global conditional solution is
`w_star = -J^T S^-1 b`, and the profiled joint MAP cost is
`0.5 b^T S^-1 b + 0.5||O/prior_std||^2 + 0.5 logdet(R)`.

This is a change of optimization coordinates, not a new posterior integral.
Replacing `logdet(R)` with `logdet(S)` would integrate states rather than
profile them and would change the statistical objective. Small independent
Gaussian integration and conditional-state identities verify this distinction.
The same state-profiling optimizer is used for both corrected and
target-only controls, rather than attributing an optimizer change to covariance.
The implementation uses Householder QR and orthogonal residual projection
instead of numerically forming the normal equations for this identity.
Independent augmented least-squares and small Gaussian controls agree with it.

## Frozen-state result: the feature correction reverses one bad ranking

Positive bad-minus-good cost means the useful B80 solution is preferred:

| Covariance at the unchanged B80 endpoints | Bad minus good |
|---|---:|
| Target-only, block diagonal | -0.338578 |
| Full-feature, block diagonal | +4.526254 |
| Target-only, correlated | +8.02223e9 |
| Full-feature, correlated | +87,919.96 |

The first two rows isolate the feature-uncertainty correction without adding
strong/weak cross covariance. Thus it really does reverse this particular
ranking under the fully normalized score. It is not achieved by changing
states, priors, observations or operators.

The complete block-diagonal contributions are:

| Endpoint and covariance | State/operator prior cost | Residual quadratic | Half log determinant | Total |
|---|---:|---:|---:|---:|
| Useful, target-only | 66.0494 | 111.6700 | -1104.9579 | -927.2384 |
| Runaway, target-only | 84.6875 | 92.6934 | -1104.9579 | -927.5770 |
| Useful, full-feature | 66.0494 | 103.5562 | -1103.5840 | -933.9784 |
| Runaway, full-feature | 84.6875 | 87.5066 | -1101.6462 | -929.4521 |

Constants shared within a case/grid comparison are omitted. Negative costs
are not negative variances or errors. These are point-density comparisons,
not integrated basin probabilities or forecast improvements.

Both correlated constructions penalize these old block-diagonal endpoints
heavily. Full-feature covariance reduces their costs from approximately
1.17e10/1.97e10 to 352,492/440,411, but that still leaves considerable room
for refitting. Correlation alone already preferred the useful frozen point;
its enormous original penalties are not evidence of a useful fitted model.
The A40 full-feature correlated cost at its unchanged endpoint is 6,199.

The independent calculation starts with conditional Fourier coefficient
square roots and propagates them through the residual maps and discrepancy,
then uses QR factors. It does not form a cancellation-prone GP covariance
subtraction or inflate eigenvalues. Archived block-diagonal factors reproduce
the totals essentially exactly. Archived correlated factors give roughly
4% different, still enormous frozen penalties because of their previously
documented near-null numerical sensitivity. Both versions are retained,
not silently conflated. Direct calculations from the archived full covariance
also reproduce the full-feature block-diagonal ranking.

**Limit:** a better ranking of two unchanged points does not establish
where the new optimization will go. No covariance-induced prediction gain
has been established by this frozen calculation alone.

## Known-system refits: a modest coarse-grid gain, no fine-grid regression

The six declared scalar fits use the original observations, GP parameters,
unit operator priors, continuous OU discrepancy and one common initial
operator per grid. Both variants use the same exact conditional-state
optimizer; neither receives the true operators or clean initial state
during fitting.

| Nodes | Covariance | Known-operator error | True-IC training trajectory error | Observed-IC training trajectory error |
|---|---|---:|---:|---:|
| 40 | Target-only correlated | 3.079% | 1.140% | 1.067% |
| 40 | Full-feature correlated | 2.744% | 1.061% | 1.001% |
| 80 | Target-only correlated | 3.684% | 0.990% | 1.115% |
| 80 | Full-feature correlated | 3.684% | 0.990% | 1.115% |
| 160 | Target-only correlated | 3.591% | 0.994% | 1.115% |
| 160 | Full-feature correlated | 3.591% | 0.994% | 1.115% |

All trajectory scores are within the same day-5-to-70 training interval.
There is no held-out forecast in this toy. The correction has a modest
effect at 40 nodes; at 80/160, the remaining off-node GP uncertainty is
already small relative to the fixed discrepancy, and the correction has
negligible effect. Latent training errors are approximately 2.95% versus
2.94% at 40 nodes and 1.40% on the finer grids.

Three fits meet the requested operator-gradient tolerance of 1e-6.
The other three terminate on function tolerance, with operator-gradient
infinity norms 8.28e-5, 6.19e-6 and 6.19e-6. All six take 17-22 iterations
and have exact conditional-state gradients below 3e-14. No fit encounters
a rejected numerical trial. Target-only operators reproduce the old
joint-MAP endpoints within 1.18e-7 relative operator norm.

Independent dense-covariance Cholesky and augmented least squares reproduce
the saved conditional states within 4e-14 absolute and normalized costs
within 2.3e-13. A separate scalar integrating-factor calculation reproduces
all post-fit trajectories within 2e-7 absolute, without using the evaluation
ODE solver.

The toy target-only fits retain the archived factors for direct historical
reproduction. For chemotherapy, both target-only and full-feature variants
use the same stable primitive roots, avoiding the much larger correlated
roundoff sensitivity documented above. The original loader remains available
for reproduction and is not silently replaced.

**Conclusion:** the correction preserves the known-system behavior and
slightly improves its coarse-grid estimate. This is a useful control, not
evidence that the chemotherapy forecast problem is solved.

## Chemotherapy refit accounting

All 48 declared primary/control fits returned finite optimizer endpoints
without a numerical exception. That is not a convergence claim:
34 stopped by function tolerance, 13 reached 4,000 iterations, and one
stopped with an abnormal line-search termination. None met the requested
absolute operator-gradient tolerance of 1e-6.

The abnormal endpoint is the selected A40 target-only correlated saved start.
It remains explicitly labeled, not discarded or reclassified as a converged
minimum. Selection always takes the lowest retained training-objective
endpoint within a case, grid and covariance variant, regardless of its
forecast or optimizer success flag.

The selected 80-node correlated runs are particularly unfinished:
operator-gradient infinity norms are approximately 11,900-26,800. Their
forecast behavior cannot establish the performance of converged correlated
fits. This assessment does not rely only on gradient size: their final
100 accepted iterations still decrease their own objectives by 0.56-4.32,
and their last individual decreases range 0.00086-0.021. They were not
at the numerical plateaus observed in the A40 controls.

Independent endpoint replay reproduces every saved operator, conditional
state, gradient and objective, and confirms all sixteen selections.
Direct fixed-state likelihood evaluation differs from the stable profiled
cost by at most 1.4e-8 across the 48 endpoints. These numerical agreements
do not turn capped or abnormal optimization into successful convergence.

### Selected forecast results

Full-field relative L2 errors over days after 70 through 110:

| Acquisition / latent points | Target-only block diagonal | Full-feature block diagonal | Target-only correlated | Full-feature correlated |
|---|---:|---:|---:|---:|
| A / 40 | 32.272% | 39.854% | 539.025% | 47.105% |
| A / 80 | Budget limit; alternative 81.127% | Budget limit; alternative 82.509% | 2,403.026% | 17,032.199% |
| B / 40 | 33.662% | 33.723% | 68.336% | 15.735% |
| B / 80 | 4.28294e22% | 6.217% | 5.42834e9% | 2,162.667% |

All sixteen selections use training objective alone. The 80-point correlated
columns are retained for completeness, not treated as converged comparisons.
Of the 48 primary point evaluations, 39 return scores and nine exceed the
unchanged 200,000-RHS budget. All failures and unselected endpoints remain
in the artifacts. Every selected in-window amplification diagnostic completes
its refinement criteria; a finite amplified trajectory is not necessarily
an accurate or stable one.

The clearest effect is B80 with strong/weak cross correlation still omitted:
the target-only likelihood selects random101's runaway operator, while
full-feature uncertainty selects the useful zero-start basin. Its forecast
error is 6.217%. Thus the frozen ranking reversal survives refitting in
this controlled example, without forecast-based choice or changing priors.

It is not a general improvement. A40 block-diagonal forecast error worsens,
B40 block-diagonal error is nearly unchanged, and A80 block-diagonal
forecasts remain poor. Among the 40-point correlated fits, restoring
feature uncertainty helps both acquisitions, but does not guarantee that
adding correlations improves on the simpler block-diagonal result.

### The two missing selected forecasts are inaccurate, not hidden successes

The original DOP853 budget failures are preserved. A separately labeled
post-fit calculation uses the affine 5-by-5 lifted cABN system, with
fourth-order Magnus steps aligned to every input knot and prediction query.
It does not extrapolate a GP, change a fitted operator, or run inside inference.
Controls against successful B80 DOP853 predictions agree within 5.44e-7
scaled error; failed coarse refinements remain recorded.

Both selected A80 block-diagonal trajectories satisfy refinement at sixteen
subdivisions, using 138,756 matrix exponentials each. They have field
training/forecast errors 19.847%/81.127% and 20.107%/82.509%.
An independent implicit Radau calculation with analytic Jacobians and
all input knots as boundaries reproduces these trajectories within
1.2e-10 maximum absolute difference. It uses approximately 84,000 RHS
calls per endpoint.

These are finite but inaccurate forecasts. Large sampled matrix norms
coexist with moderate sampled eigenvalue timescales, so budget exhaustion
alone did not establish either divergence or classical stiffness.
The alternative values supplement, rather than overwrite, the primary
evaluation outcomes.

### Bounded same-objective A40 optimization controls

Two controls start at the selected A40 correlated endpoints, with every
model term unchanged. Each has a 2,000-call budget that includes numerical
derivative and curvature probes. The target-only control uses an exact
variable-projection residual/Jacobian with trust-region least squares;
the full-feature control uses curvature-scaled trust-exact optimization
while retaining the parameter-dependent covariance normalizer.

| A40 control | Objective decrease | Operator-gradient infinity norm before / after | Calls |
|---|---:|---:|---:|
| Target-only correlated | 5.84e-8 | 0.0257 / 0.00303 | 419 |
| Full-feature correlated | 7.44e-8 | 0.0325 / 1.34e-5 | 561 |

Local Hessians are numerically consistent and positive in the inspected
neighborhoods, with raw eigenvalues ranging approximately 8-11 through
4.3e7-9.2e7. The apparently large gradients coexist with very small
remaining predicted objective gains because of this strong anisotropy.

The target-only control stops on function tolerance. The full-feature
control accepts one step and then cannot resolve a further improving step;
that unsuccessful termination is retained. Neither meets the strict
gradient threshold. Estimated remaining local Newton improvements are
7e-12 and 3e-16, below the observed numerical consistency scales.
This supports a local objective plateau for these A40 endpoints, not a
global optimum or a claim that every 80-point fit has converged.

Independent post-fit evaluation confirms that these changes do not
materially repair the forecasts:

| A40 covariance | Original / control field training error | Original / control field forecast error |
|---|---:|---:|
| Target-only correlated | 23.143% / 23.137% | 539.025% / 538.878% |
| Full-feature correlated | 9.677% / 9.676% | 47.105% / 47.104% |

Both control amplification diagnostics also satisfy their refinement
criteria. Original predictions and optimizer endpoints remain unchanged
in the primary record.

## Physical-source audit: the uniform treatment block is known

The actual simulator computes the scalar chemotherapy cell-death rate and
subtracts that rate times the density. The ROM input alpha already includes
drug sensitivity; multiplying by sensitivity again would be incorrect.
The generator's comments using sensitivity and alpha separately should not
be substituted for the actual input implementation.

For the centered orthonormal POD coordinates `q = V^T (u - shift)`, the
uniform treatment contribution therefore projects to
`q'_chemo = -alpha(t) (q + V^T shift)`.
In the fitted `alpha(t) (B + Nq)` convention, that contribution has
`N = -I` and `B = -V^T shift`.

This identity is exact for uniform scalar killing, even when other
full-order state components lie outside the retained POD subspace.
Scalar multiplication commutes with orthogonal projection. It does not
make the autonomous cA dynamics exact or remove nonlinear closure error.

The frozen basis has maximum orthogonality error 2.54e-14. Direct projection
of the full-space treatment term at three archived states and three input
values agrees with the reduced identity within 7.5e-13 absolute. The
resulting B is approximately `[38.1153, -7.06963, -2.27111, 0.0609354]`.

The useful B80 operator is not physically correct merely because its
forecast is useful: its Frobenius distance from -I is approximately 369
times the norm of I, and the runaway operator's is approximately 894
times that norm. The useful N has negative eigenvalues but large nonnormal
couplings; the runaway N also has positive
eigenvalues. A blanket eigenvalue sign rule would not reproduce the
actual known treatment block.

Sources:

- `TumorTwin/tumortwin/models/reaction_diffusion_3d.py:161-180` subtracts the
  treatment rate times the state.
- `TumorTwin/tumortwin/treatments/chemotherapy.py:60-103` includes sensitivity
  in that rate.
- `experiments/tumor/config.py:218-261` wraps the same rate as the ROM input.
- `experiments/tumor/config.py:585-624` defines centered POD compression.
- `experiments/tumor/generate_fom_data_chemo.py:107-140` records the generator's
  treatment scaling and model construction.

**No treatment restriction is fitted in this round.** Fixing these operators
or introducing a restricted treatment family would supply additional
mechanistic information. It is a distinct next experiment, not a disguised
covariance-only correction. The identity is specific to this simulator and
input convention, not a general assertion about clinical treatment response.

## Interpretation

The uncertainty correction changes a consequential ranking even before
optimization, and that benefit survives the B80 block-diagonal refit.
This establishes a real contribution to that failure, not a generic
forecast repair. The other block-diagonal outcomes are worse or essentially
unchanged. Forty-point correlated fits benefit from the correction, while
the 80-point correlated comparisons remain limited by unfinished optimization.

The known-system control remains useful, and the two A40 curvature controls
show that simply requesting more iterations there is unlikely to change the
local objective materially. No global optimum or posterior-calibration claim
follows from these MAP experiments.

The independent simulator audit identifies a concrete subsequent structural
experiment: constrain the treatment response to its known uniform-removal
form while continuing to learn untreated dynamics. That would be an explicit
additional use of mechanistic information. It is deliberately not combined
with this covariance correction in the present study.

## Artifacts

Artifacts are under session
`270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics/`.
`fullcov_protocol.json` records the declared comparison.
`fullcov_rescore.py` and `fullcov_rescore/` contain independent frozen scores,
all contributions, square-root factors and numerical comparisons.
`fullcov_reference_identities.py` contains the small Gaussian controls.
`fullcov_physics_audit.py` and `fullcov_physics_audit/` preserve the numerical
projection checks and archived operator comparisons.
`fullcov_refit_amendment.json` records the twelve additional covariance
factorial controls before chemotherapy fitting.
`fullcov_model.py` and `fullcov_stable_target.py` implement the normalized
likelihood, exact state profile and explicitly separate numerical conventions.
`fullcov_toy.py` and `fullcov_toy/` contain all six fits, traces, numerical
controls and API descriptions. `fullcov_verify_toy.py` preserves the independent
state/cost/trajectory reproduction.
`fullcov_chemo_runs.py` and `fullcov_chemo/` preserve the 48 refits, all
accepted/trial records, sixteen selections, source hashes and exact
termination metadata. `fullcov_verify_refits.py` and
`fullcov_chemo/parent_validation.json` record independent endpoint replay
and selection/convergence checks.
`fullcov_evaluate.py` and `fullcov_chemo_evaluation/` contain all 48 primary
evaluations, sixteen selected detailed diagnostics, and separately labeled
budget-limited alternatives. `fullcov_verify_predictions.py` reproduces
all 39 available field metrics and the independent Radau comparisons.
`fullcov_optimizer_controls.py` and `fullcov_optimizer_controls/` preserve
the two bounded same-objective A40 controls and curvature evidence.
`fullcov_optimizer_control_evaluation/` records their separate post-fit
comparisons without overwriting the primary results.
