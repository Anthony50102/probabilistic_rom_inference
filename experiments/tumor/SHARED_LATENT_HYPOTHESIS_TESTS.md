# Shared-latent hypothesis tests

## Main findings

The strongest positive diagnoses are coupled state/operator sensitivity,
large dynamical amplification, and materially omitted off-node feature
uncertainty. Simple conditional operator ambiguity, a narrow bad operator-MAP
solution, and omitted GP hyperparameter uncertainty do not explain the
dominant failures by themselves.

A known, well-excited scalar system is recovered with approximately 1%
in-window trajectory error and stable grid refinement under a fixed
continuous discrepancy model. This is evidence that the shared-latent
strong/weak machinery can work, not a validated chemotherapy repair.

## Scope and controls

This follows [the combined-constraint study](SHARED_LATENT_CONSTRAINT_STUDY.md).
The pre-study recovery commit is `28983e7`, also preserved as
`backup/shared-latent-before-hypotheses-28983e7`.

Four hypotheses were declared before the new experiments: insufficient operator
identification/amplification, operator profiling versus marginalization,
constraint-covariance fidelity and grid consistency, and underestimated
interpolation uncertainty from fixed GP hyperparameters.

The chemotherapy experiments retain the same two 80-observation datasets,
training days 5-70, prediction through 110, four-mode basis, noise residuals and
known input. A (`matched`) is the fixed comparison acquisition; B (`resample43`)
uses replacement observation times. No forecast-based fit selection is used.
Synthetic generation and post-fit diagnosis can solve ODEs; fitting cannot.
Production code, defaults, canonical benchmarks and prior results are unchanged.

## H1: Conditional identification versus joint sensitivity and amplification

The main comparison uses the archived B80 block-diagonal zero-start
approximately 6%-forecast solution and random101 catastrophic solution.
No new nonlinear fitting or regularization is introduced.

The prior-scaled, noise-whitened operator designs are full rank. Their
smallest likelihood singular values are **15.86** and **7.13**, respectively,
both above one in prior-standardized operator coordinates. Although 98.54%
of the squared operator difference lies in the three relatively weakest
joint directions, the operators are not cheaply interchangeable with X fixed:

| Frozen latent curve | Operator substituted | Increase in training objective |
|---|---|---:|
| Useful solution's curve | Catastrophic operator | 7,041.08 |
| Catastrophic solution's curve | Useful operator | 1,276.27 |

Those costs correspond to conditional posterior Mahalanobis distances of
118.67 and 50.52. Relative spectral weakness is not low absolute cost.
The differences primarily involve B/N; isolated-block costs contain large
cancelling cross terms, so separate raw coefficient norms are misleading.

The aggregate dense path difference is about 0.949%, but the first mode
contains 98.56% of the path energy. Per-mode differences are
0.913%, 2.451%, 2.992% and 1.158%. Time-RMS differences in actual
data-conditioned GP SD units are 0.124, 0.342, 0.554 and 0.428; the third
mode reaches 3.90 SD locally. Observation-time differences remain much
smaller than some between-observation differences.

The whitened-state displacement has norm 8.49, or RMS 0.475 per coordinate
across 320 coordinates. Five predefined points interpolating the whitened
states have profiled objectives approximately
177.72, 189.02, 256.87, 375.20 and 177.38. This establishes sensitivity
of the coupled state/operator fit and a sampled barrier, not a complete
description of global basin geometry or proof of strict local minima.

### Controlled conditional-operator perturbations

The three largest conditional operator-covariance directions were perturbed
by plus/minus one and two SD at each frozen latent state, giving 24
predeclared perturbations. Every fixed-X objective increase matches the
Gaussian quadratic prediction `0.5 * amplitude^2`.

All 12 useful-basin perturbations scored, with forecast errors **6.13-11.72%**,
compared with 6.00% at the original operator. No catastrophic result occurred
in this finite set. Seven bad-basin perturbations scored but remained very
inaccurate; five exceeded the unchanged evaluator budget. All were retained.
These directed probes are not random posterior samples, a posterior success
probability, or a full joint uncertainty assessment.

### The important dynamical distinction

Post-fit error decompositions separate the local derivative defect from its
propagation by the learned linear time-varying matrix `A + alpha(t) N`.

| Own-curve diagnostic | Useful operator | Catastrophic operator |
|---|---:|---:|
| Total reduced derivative-defect RMS | 1.486 | 1.254 |
| Homogeneous transition norm gain, day 5 to day 70 | 4.46 | approximately 1.56e14 |

The bad operator actually has the smaller local defect, but amplifies errors
enormously. The transition matrix is independent of the chosen latent path;
the same gain contrast appears when each operator is evaluated on the other
path. Propagated derivative defects supply most of the reconstructed error,
not the tiny initial mismatch alone. Their norm ratios are not additive
variance fractions.

Independent input-knot fourth-order Magnus refinement confirms the huge gain;
it resolves an initially insufficiently accurate standalone RK45 comparison.
The useful operator also permits positive local growth, so this does not
justify forcing every instantaneous eigenvalue or symmetric-part eigenvalue
to be nonpositive.

**Conclusion:** the simple hypothesis of cheap conditional operator ambiguity
at an otherwise identical fixed trajectory is not supported. Coupled X-O
sensitivity and severe dynamical amplification are supported. Small local
constraint error is not a reliable surrogate for integrated trajectory error.

## H2: Does operator marginalization repair the ranking?

### Exact isolated change

The prototype profiled the conditional Gaussian operator MAP while optimizing
whitened latent states. The new objective instead integrates operators exactly,
retaining the same cached GP means/maps, covariance factors, measurement noise,
operator priors, discrepancy and initial states. No new numerical ridge is added.
Latent states are still optimized at MAP; this does not restore a full posterior
over latent trajectories or hyperparameters.

For each prior-scaled, noise-whitened design D, integration adds
`0.5 * logdet(I + D^T D)` to the profiled objective. The determinant and its
gradient are evaluated by SVD. At any fixed state the conditional Gaussian
operator mean is unchanged. Independent observation-space Gaussian-density
and directional-derivative identities confirm the implementation.

All 48 prior endpoints were rescored. The 32 predefined strong-only and
block-diagonal combined combinations were then refitted: two datasets,
40/80 states, four identical initializations, up to 4,000 L-BFGS iterations,
`ftol=1e-13`, `gtol=1e-6`. Correlated endpoints were rescored only because
their conditioning and stationarity limitations were already established.

### Existing good/bad B80 states

For the block-diagonal combined formulation:

| Archived state | Profiled objective | Added determinant term | Marginalized objective |
|---|---:|---:|---:|
| Zero-start, useful rollout | 177.719446 | 267.169956 | 444.889402 |
| Random101, catastrophic rollout | 177.380868 | 263.716055 | 441.096923 |

The bad state remains preferred, and its advantage increases. In particular,
its conditional Gaussian operator covariance ellipsoid has approximately
**31.6 times the volume** of the good state's, with the same operator prior
scaling. It is not a narrower conditional-operator solution being unfairly
favored only by profiling.

This volume comparison is conditional on the respective latent states. It
does not integrate neighborhoods in latent-state space and is not a posterior
probability or full joint basin-mass ratio.

### Refitted results

Each row selects only by its own marginalized training objective.
Full-field relative L2 training / forecast percentages:

| Dataset | States | Constraints | Profiled baseline | Marginalized refit |
|---|---:|---|---:|---:|
| A | 40 | Strong | 9.91% / 26.10% | 38.89% / 395.52% |
| A | 40 | Strong + weak, block diagonal | 8.29% / 32.27% | 8.69% / 33.53% |
| A | 80 | Strong | 16.41% / 32.94% | 16.41% / 33.11% |
| A | 80 | Strong + weak, block diagonal | Evaluation budget exceeded | Evaluation budget exceeded |
| B | 40 | Strong | 9.41% / 27.72% | 17.24% / 22.79% |
| B | 40 | Strong + weak, block diagonal | 4.14% / 33.66% | 5.31% / 32.07% |
| B | 80 | Strong | 10.53% / 10.90% | 9.73% / 8.13% |
| B | 80 | Strong + weak, block diagonal | Catastrophic growth | Evaluation budget exceeded |

The B80 combined zero-start refit still gives a useful point forecast,
**3.93% training / 6.56% forecast**, but its objective is 444.638585 versus
440.917478 for the selected random101 result, which exceeds the unchanged
post-fit evaluator budget. It is not substituted for the selected result.

All 32 fits reached function-tolerance termination before their iteration cap;
none met the requested absolute gradient tolerance. Final gradient infinity
norms span approximately 1.17e-5 to 2.02e-3. Twenty-six point rollouts were
scored; six explicitly exceeded the unchanged 200,000-RHS budget.

**Conclusion:** Gaussian operator marginalization alone does not repair the
observed ranking failure or consistently improve point rollouts. It improves
some rows, leaves others similar, and worsens others. This rules against
the particular narrow-conditional-operator-MAP explanation, not against
Bayesian inference in general. GP hyperparameter uncertainty, uncertainty in
latent states, off-node feature covariance and physical identifiability are
separate issues. No posterior predictive calibration claim follows from
these conditional-mean-operator point forecasts.

## H3: Constraint-covariance fidelity and a known-system control

### Off-node feature uncertainty is material

At three archived endpoints, operators and nodal latent states were fixed.
For each GP coordinate j, the joint conditional Gaussian functionals are
`Z_j = X'_j(tc)`, `C_j = integral(psi X_j)`,
`U_j = integral(psi alpha X_j)` and `T_j = -integral(psi' X_j)`.
The weak random residual in output i is
`sum_j(A_ij C_j + N_ij U_j) - T_i`.

The current prototype uses conditional means in the C/U feature terms and
propagates only target uncertainty through T. Restoring those random feature
terms changes both within-output covariance and cross-output covariance.
The same nodal discrepancy and original numerical nuggets are retained;
the comparison does not inflate noise or retune parameters.

| Fixed endpoint | Maximum weak variance ratio, full / target-only | Median ratio | Maximum cross-output absolute correlation |
|---|---:|---:|---:|
| B80 useful zero-start solution | 2.34 | 1.00014 | 0.491 |
| B80 catastrophic random101 solution | 6.92 | 1.00092 | 0.830 |
| A40 selected block-diagonal solution | 14.22 | 1.01275 | 0.917 |

The maxima are not uniform variance inflation. Most marginal changes are
much smaller, and some variances decrease because feature-target cross terms
can be negative. The corresponding weak-covariance Frobenius changes are
15.5%, 82.3% and 210.6% of target-only weak covariance. Adding a positive
diagonal slack term would not reproduce the missing covariance structure.

The effect is particularly severe in near-null directions of the correlated
target-only construction: largest eigenvalues of the full covariance whitened
by its implemented correlated factors reach approximately 1.13e8, 2.38e9
and 4.64e9. These values are not physical-error calibration factors; they
quantify disagreement within the assumed conditional GP model. Comparisons
against block-diagonal factors also restore omitted strong/weak correlation,
so they are not purely comparisons of individual marginal variances.

Twenty thousand underlying joint GP-functional draws per endpoint were
propagated through the residual formula and shared discrepancy. They were
not sampled directly from the final residual covariance. Empirical full
covariance errors are 1.59-1.67% in relative Frobenius norm, consistent
with their Gaussian Monte Carlo precision; all marginal variances pass
simultaneous bounds. Smaller cross-covariance blocks have worse relative
Monte Carlo precision, which is reported separately.

A positive-semidefinite Fourier representation and coefficient-space
conditioning avoid cancellation-prone GP Schur-complement subtraction.
Kernel, quadrature and frequency-refinement errors were checked separately.
No material negative eigenvalues or extra covariance noise were repaired
away. With A=N=0 the reduction to target-only covariance is algebraically
exact. Archived floating-point matrices differ by up to 7.34e-10 absolute;
their near-null directions amplify that small historical cancellation error.
It is much smaller than the nonzero-operator feature-uncertainty effects.

**Consequence:** after marginalizing random off-node features, full residual
covariance depends on O and couples outputs. The operator posterior is not
generally Gaussian under that marginal likelihood. H2's Gaussian operator
integration is correct for the fixed-covariance approximation, not an exact
integration of this full-feature model. This does not require ODE solves
inside fitting, but a consistent replacement would need a different
factorization or latent representation. No replacement was fitted here,
and the covariance audit does not establish a ranking or forecast repair.

### Known scalar cABN system with grid-consistent discrepancy

The predeclared synthetic system is
`x' = 0.3 - 0.22 x + u(t) (0.6 - 0.12 x)`, starting from x(5)=-1.
The input is a fixed three-frequency sum, deliberately rich and not a
nonnegative chemotherapy dosing schedule. Thirty-two noisy observations
use fixed seeds, noise SD 0.04, RBF lengthscale 1.4, variance 4 and unit
operator-prior SDs. None was tuned after seeing results.

One continuous Ornstein-Uhlenbeck discrepancy is used at every resolution:
`Cov(epsilon(t), epsilon(s)) = 0.035^2 exp(-abs(t-s)/0.8)`.
Its values and weak integrals share full covariance. This is not white
noise evaluated at individual points or a count of independent nodal errors.
It is an explicitly different toy control, not a silent chemo-model change.

Nine fits compare 40/80/160 nodes with strong-only, weak-only and correlated
combined constraints, using the same mean-feature/target-covariance MAP
approximation and one predefined initialization. All profiled gradients
meet the requested 1e-6 tolerance.

Post-fit relative trajectory errors from the exact reference initial state:

| Nodes | Strong only | Weak only | Combined |
|---|---:|---:|---:|
| 40 | 1.821% | 1.679% | 1.140% |
| 80 | 1.015% | 1.681% | 0.990% |
| 160 | 1.002% | 1.681% | 0.994% |

These errors are on the common **day-5-to-70 training interval**, not held-out
forecasts. The exact initial state is privileged and used only after fitting;
observed-initial-state errors are also reported and range 1.07-1.85%.
No clean initial state, oracle operators or dense truth entered fitting.

Known-operator errors range 3.08-7.85%. The inferred and oracle feature
matrices have full rank four and conditions approximately 6.5-17.8.
Unregularized true-path regression recovers the known operator within
1.92e-8 relative error, providing a separate code/quadrature oracle.
These controls do not select an inferred endpoint.

The data-only GP has 38.53% dense state error. Physics reduces latent
path error to roughly 1.4% on the finer strong/combined grids and 6.33%
for weak-only. From 80 to 160 nodes, strong/combined operators change
approximately 0.15%/0.10%, and weak-only changes negligibly.
Weak OU covariance changes only around 1e-7 relatively across grids,
consistent with one physical process rather than changing noise strength.
Independent integrating-factor quadrature reproduces all nine post-fit
trajectories within 2.4e-7 maximum absolute discrepancy.

**Conclusion:** the uncertainty approximation omits material terms, and the
shared-latent strong/weak method can recover this known well-excited system
with stable refinement. The toy does not isolate which difference from
chemotherapy is decisive, prove universal consistency, or validate a chemo
discrepancy replacement.

## H4: Does integrating GP hyperparameters repair interpolation uncertainty?

All eight two-parameter GP hyperposteriors (four modes, two datasets) were
integrated over their full original MAP bounds, retaining the same zero-mean
RBF family, positive known likelihood variance, roundoff and original
noisy-data empirical hyperprior anchors. The independent unit-Normal priors
in log lengthscale/log variance were normalized over those bounds.

A full-bound refined density search was followed by adaptive Gauss integration.
The procedure captures multiple modes rather than assuming a local Gaussian
posterior. Predictive intervals are actual Gaussian-mixture CDF quantiles,
not a mean plus/minus a multiple of a moment-matched SD. Variance is separated
into average conditional-GP variance and variability of conditional means
across hyperparameters. Predictive observation noise is not added to
latent-function bands.

| Diagnostic | Fixed hyperparameters | Integrated hyperparameters |
|---|---:|---:|
| A: reduced training-truth error | 24.48% | 23.78% |
| A: dominant-mode first-gap pointwise 95% coverage | 1.75% | 3.99% |
| B: reduced training-truth error | 17.23% | 17.32% |
| B: dominant-mode first-gap pointwise 95% coverage | 100% | 100% |

A's dominant-mode median gap SD increases by about 10.9%, but the wrong
overshoot remains. Its fourth mode is genuinely bimodal, with approximately
22%/78% mass in the two lengthscale-marginal basins; both are included.
Those basin weights are not joint-mode probabilities.

Full-bound and local quadrature refinement, tail mass, predictive-quantile
refinement, total-variance identities and direct Cholesky comparisons were
recorded. Independent direct-Cholesky mixture-CDF calculations at three
interior gap times reproduce the reported 95% endpoints within 1.6e-10
in CDF probability for A and 2.7e-11 for B.

**Conclusion:** omitted hyperparameter uncertainty contributes and is not
negligible in every mode, but does not explain or resolve the dominant
interpolation defect in the matched acquisition. These are data-only GP
posteriors with fixed empirical anchors/noise, not a refitted joint
state/operator/GP posterior. The result does not rule out effects of
hyperparameter uncertainty within a future full physics inference.

## Interpretation

There are two distinct supported failure mechanisms: the statistical
approximation drops meaningful uncertainty terms, and the learned dynamics
can amplify small local errors catastrophically. Correcting one is not
proven to correct the other. The operator-integral and GP-hyperposterior
experiments do not supply a standalone repair.

The known-system control gives a constructive counterexample to the claim
that shared-latent derivative/weak inference cannot work. It does not
establish robustness on the chemotherapy acquisition or held-out forecasts.
The evidence motivates a consistent latent-feature/constraint representation
and investigation of physically justified input-feedback behavior, rather
than indiscriminate covariance inflation, longer optimization or a switch
to fitting through ODE integration. Those are future research directions,
not changes implemented or validated by this study.

## Reproducibility

Session artifacts are under
`270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics/`.
`hypothesis_study_protocol.json` records the four declared investigations.
`hypothesis_marginalization.py` and `hypothesis_marginalization/` preserve
the isolated objective, 48 archived scores, 32 fitted states, conditional
operator covariances, optimization traces, point predictions and source hashes.
The `prob_rom` environment and one BLAS/OMP thread were used.

`hypothesis_operator_audit.py` and `hypothesis_operator_audit/` contain the
frozen-design spectra, swaps, all perturbations, standardized comparisons
and four training-time error decompositions. `parent_gain_verification.json`
records the independent propagation refinement.

`hypothesis_gp_uncertainty.py` and `hypothesis_gp_uncertainty/` preserve all
hyperparameter integration weights, global/refined grids, mixture predictions,
variance decompositions, plots and numerical diagnostics.

`hypothesis_covariance_audit.py` and `hypothesis_covariance_audit/` contain
three conditional-feature covariance audits, underlying functional
covariances/draw summaries, numerical references and the nine known-system
fits. Its `toy_control.py` and `test_covariance_hypothesis.py` preserve the
continuous-discrepancy control and six targeted analytic/numerical tests.
