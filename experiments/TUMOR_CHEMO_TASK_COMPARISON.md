# Matched chemotherapy continuation: task construction and Neural ODE comparison

## Main result

**The selected half-exposure task met both requested point-performance goals:
reasonably accurate forecasts and better forecasts than Neural ODE.**

Over the full 40-day forecast, native full-field errors were **5.39%, 5.51% and
5.31%** on the three fresh acquisitions. The original training-loss-filtered
NODE medians gave **9.90%, 10.04% and 9.93%**. Native won every matched pair,
also beat the strict all-20 NODE median, and retained the result against the
finer source reference. No change to the Bayesian inference algorithm was
needed.

This is a useful synthetic continuation task, not a general chemotherapy
solution: it predicts the next cycles of the same known regimen, with a
half-exposure intervention, dense low-noise observations and a privileged
archived clean spatial basis. The original-exposure control is retained.
Native also beat NODE there, but its median error was 19.03% and worst error
33.90%, failing the frozen absolute-accuracy requirement.

The diagnosis is more nuanced than "chemo always breaks the GP." Development
seed 42 exhibited collapse, whereas all three fresh original-exposure GP
reconstructions were faithful. Good GP reconstruction was still insufficient:
original-exposure seed 46 had 2.20% GP training-field error but 33.90% native
ODE forecast error. Halving exposure is therefore not established as a
necessary cure for GP collapse, nor have timing, noise realization and their
data-dependent priors been separately isolated as its cause.

The [matched untreated-growth comparison](TUMOR_GROWTH_NODE_COMPARISON.md)
found one narrower success with good data, but no sparse/noisy condition met
both goals. Among these tested applications, half-exposure chemotherapy
continuation is the stronger point-forecast example. Neither result establishes
calibrated uncertainty, clinical utility, or unseen-dose generalization.

This follows the [multi-history and untreated-growth study](TUMOR_APPLICATION_STUDIES.md)
without changing that study or its conclusions. All cells use one demo anatomy
and three acquisition seeds, not independent patients. The criteria are
exploratory synthetic-benchmark criteria, not population significance.

## Primary 40-day forecast results

Full-field relative RMS percentage is measured over days (70,110], using 152
equally weighted retained queries. The affine decoder shift and unrepresented
residual are included. U would mean unavailable, never zero or a score restricted
to a surviving prefix. All primary trajectories completed in this confirmation.

| Task | Seed | Native | NODE original filtered median | NODE all20 median | NODE kept | Point numerics qualified |
| --- | --- | --- | --- | --- | --- | --- |
| Original exposure | 45 | 11.35 | 48.57 | 49.47 | 19/20 | Yes |
| Original exposure | 46 | 33.90 | 51.28 | 50.58 | 19/20 | Yes |
| Original exposure | 47 | 19.03 | 49.18 | 50.42 | 19/20 | Yes |
| Half exposure | 45 | 5.39 | 9.90 | 9.96 | 18/20 | Yes |
| Half exposure | 46 | 5.51 | 10.04 | 10.04 | 20/20 | Yes |
| Half exposure | 47 | 5.31 | 9.93 | 9.93 | 20/20 | Yes |

The native point is the conditional operator mean at arithmetic means of
positive hyperparameter/block-scale samples, not a posterior predictive mean.
NODE medians are coordinatewise trajectory medians, decoded before computing
physical errors or burden.

## Frozen success criteria

Native median full-field error must be <=15%, every seed <=25%, and all three
points complete. Against both the original training-loss-filtered NODE median
and strict all-20 median, the median paired native/NODE error ratio must be
<=0.90 and native must have lower error on at least 2/3 matched seeds. Missing or
unqualified NODE comparisons are not native wins. Posterior reliability is
separate, not an additional hidden point-performance requirement.

| Task | Native median% | Ratio vs filtered | Ratio vs all20 | Wins vs filtered | Wins vs all20 | Primary criteria met | Refined-reference control met |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original exposure | 19.03 | 0.387 | 0.377 | 3/3 | 3/3 | No | No |
| Half exposure | 5.39 | 0.544 | 0.541 | 3/3 | 3/3 | Yes | Yes |

The refined-reference column rescores identical saved predictions without
refitting. It does not redefine the primary reference or permit task
reselection. Against that reference, half-exposure native errors were 5.93%,
5.46% and 5.59%, versus filtered NODE's 9.85%, 9.64% and 9.83%.

## Simple baselines and training-versus-forecast behavior

These training-only baselines were specified before model forecasts.
Persistence repeats the last observation; the linear trend is unweighted OLS
over all 120 training observations. Neither uses future truth.

| Task | Seed | Native forecast% | NODE filtered forecast% | Persistence% | Linear trend% |
| --- | --- | --- | --- | --- | --- |
| Original exposure | 45 | 11.35 | 48.57 | 36.47 | 134.28 |
| Original exposure | 46 | 33.90 | 51.28 | 36.50 | 138.08 |
| Original exposure | 47 | 19.03 | 49.18 | 36.47 | 124.98 |
| Half exposure | 45 | 5.39 | 9.90 | 12.36 | 21.28 |
| Half exposure | 46 | 5.51 | 10.04 | 12.38 | 23.30 |
| Half exposure | 47 | 5.31 | 9.93 | 12.37 | 21.57 |

The half-exposure task is not won simply by repeating the last scan or extending
a straight line. Its native median error is 5.39%, versus 12.37% for persistence
and 21.57% for the linear trend.

In every confirmation cell, NODE had a lower clean-reference training-trajectory
field error than native, but a higher 40-day forecast error. Better training
reconstruction did not imply better extrapolation. This is a descriptive
comparison, not proof that one particular network or inference mechanism
caused the difference.

## Development selection and fresh training

Source inputs, observation count and inference were frozen before the four
seed-42 development screens. Candidate priority was S, M, A; M alone passed
all state-fidelity gates. No future model forecast or NODE score selected it.

| Cell | Reduced GP% | Full-field GP% | Variation-normalized GP% | Eligibility |
| --- | --- | --- | --- | --- |
| C_original_pulses | 96.15 | 38.09 | 96.20 | ineligible |
| S_distributed_exposure | 88.51 | 33.06 | 90.13 | ineligible |
| M_half_exposure | 6.61 | 2.52 | 13.84 | eligible |
| A_event_acquisition | 94.16 | 37.30 | 94.21 | ineligible |

M retained a 43.03% relative full-field treatment contrast against a matched
untreated training reference. This is not 43% tumor shrinkage or clinical
efficacy. Distributed exposure had a lower peak than M while preserving the
original total exposure, so peak height alone was not isolated as the cause.
Event-centered observations at the same budget did not cure the development
collapse.

| Fresh task | Seed | Training GP field% | Training GP reduced% |
| --- | --- | --- | --- |
| Original exposure | 45 | 2.27 | 5.71 |
| Original exposure | 46 | 2.20 | 5.53 |
| Original exposure | 47 | 1.62 | 4.08 |
| Half exposure | 45 | 1.47 | 2.46 |
| Half exposure | 46 | 1.52 | 2.70 |
| Half exposure | 47 | 1.50 | 2.62 |

Fresh-seed changes alter observation timing, noise realization and consequently
the unchanged data-dependent GP/noise priors. The evidence establishes
acquisition sensitivity, not which coupled factor caused it. The successful
task changes the operating regime; it does not prove a unique mechanistic
explanation or establish robust arbitrary treatment-response identification.

## Numerical source and representation limits

The initial 0.5-day versus 0.25-day source comparison failed the unchanged 1%
aggregate training gate in C, S, M (1.53635%, 1.39280%, 1.01806%). No fits ran
in that failed qualification stage. One explicit pre-fit refinement to 0.125
day qualified 0.25-day primaries, keeping the same 0.5-day save grid. The
0.25-day source is not continuum truth.

Twelve independent scalar controls isolate RK4 endpoint anticipation at hard
dose onsets: the installed 3/8 rule gives onset state `1-h*0.5/8` rather than 1
without endpoint perturbation. This numerical mechanism is not an explanation
of the near-95% reduced-state GP error. No production simulator was changed.

| Task | 0.25/0.125 forecast discrepancy% | Day-110 discrepancy% | Primary four-mode forecast floor% |
| --- | --- | --- | --- |
| Original exposure | 2.5846 | 3.4544 | 1.2670 |
| Half exposure | 1.1727 | 1.4092 | 4.7725 |

Both methods use the same archived clean nominal four-mode decoder. This is
privileged synthetic preprocessing, not a deployment-realistic noisy-basis
estimate. The half-exposure native error is close to, but above, its 4.77%
representation floor. Comparison within each task uses the same reference;
cross-task percentages also reflect the changed reference trajectory and
normalization, not an isolated algorithm change.

## Point-estimator and uncertainty controls

These secondary centers are not replacements selected after seeing errors.
Strict centers would be unavailable wherever any expected contributing member
was unavailable.

| Task | Seed | Native hyperpoint% | Native strict-64 median% | NODE strict-20 mean% |
| --- | --- | --- | --- | --- |
| Original exposure | 45 | 11.35 | 10.21 | 52.43 |
| Original exposure | 46 | 33.90 | 29.33 | 50.17 |
| Original exposure | 47 | 19.03 | 18.78 | 50.31 |
| Half exposure | 45 | 5.39 | 5.46 | 9.53 |
| Half exposure | 46 | 5.51 | 5.47 | 9.96 |
| Half exposure | 47 | 5.31 | 5.42 | 8.95 |

Native uncertainty combines variational GP/block-scale samples, conditional
operators and original IC draws. NODE spread is initialization variability,
not the same probability construction. Bands below are empirical 5th/95th
percentiles with full-denominator availability; interval inclusion is not a
calibration claim.

| Task | Seed | Native complete | Native complete + field <=25% | Native all-64 finite + truth in band | NODE all-20 finite + truth in band | Native reliability guard |
| --- | --- | --- | --- | --- | --- | --- |
| Original exposure | 45 | 64/64 | 29/64 | 152/152 | 43/152 | No |
| Original exposure | 46 | 64/64 | 16/64 | 152/152 | 22/152 | No |
| Original exposure | 47 | 64/64 | 28/64 | 152/152 | 48/152 | No |
| Half exposure | 45 | 64/64 | 64/64 | 133/152 | 111/152 | Yes |
| Half exposure | 46 | 64/64 | 64/64 | 128/152 | 111/152 | Yes |
| Half exposure | 47 | 64/64 | 64/64 | 39/152 | 112/152 | Yes |

The reliability guard is 58/64 complete and 32/64 complete with field error
<=25%. It was not added to the point-performance criteria. All 20 NODE models
remain in reliability accounting, including training-loss-rejected members.

All 192 evaluated half-exposure native draws completed with field error <=25%.
That is encouraging stability, not reliable uncertainty calibration: for
half-exposure seed 47, the nominal 5-95% burden band contained truth on only
39/152 forecast queries. Conversely, original-exposure bands contained truth
throughout while most draws exceeded 25% field error. Stability, point accuracy,
band width and calibration are different properties.

## Decoded point-field diagnostics

Negative reduced coordinates alone are normal for a centered POD basis.
These diagnostics decode actual fields. Material negativity uses cellularity
<-1e-6; values >1+1e-6 are also retained. These are diagnostics, not clipping or
selection gates. Fractions use available voxel-query pairs only; availability
is explicit. Point bounds do not establish positivity of every uncertainty
member or physical validity for untested regimens.

| Task | Seed | Point | Minimum | Maximum | Material-negative voxel-query% | Max negative mass / true burden% | Queries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original exposure | 45 | NODE_all20_median | -0.00000 | 0.28762 | 0.0000 | 0.0000 | 152/152 |
| Original exposure | 45 | NODE_original_policy_median | -0.00000 | 0.28470 | 0.0000 | 0.0000 | 152/152 |
| Original exposure | 45 | native_point | -0.00000 | 0.23460 | 0.0000 | 0.0000 | 152/152 |
| Original exposure | 46 | NODE_all20_median | -0.00000 | 0.28593 | 0.0000 | 0.0000 | 152/152 |
| Original exposure | 46 | NODE_original_policy_median | -0.00000 | 0.28587 | 0.0000 | 0.0000 | 152/152 |
| Original exposure | 46 | native_point | -0.00000 | 0.25389 | 0.0000 | 0.0000 | 152/152 |
| Original exposure | 47 | NODE_all20_median | -0.00000 | 0.27538 | 0.0000 | 0.0000 | 152/152 |
| Original exposure | 47 | NODE_original_policy_median | -0.00000 | 0.27378 | 0.0000 | 0.0000 | 152/152 |
| Original exposure | 47 | native_point | -0.00000 | 0.22727 | 0.0000 | 0.0000 | 152/152 |
| Half exposure | 45 | NODE_all20_median | -0.00000 | 0.45232 | 0.0000 | 0.0000 | 152/152 |
| Half exposure | 45 | NODE_original_policy_median | -0.00000 | 0.45257 | 0.0000 | 0.0000 | 152/152 |
| Half exposure | 45 | native_point | -0.00000 | 0.49985 | 0.0000 | 0.0000 | 152/152 |
| Half exposure | 46 | NODE_all20_median | -0.00000 | 0.46614 | 0.0000 | 0.0000 | 152/152 |
| Half exposure | 46 | NODE_original_policy_median | -0.00000 | 0.46614 | 0.0000 | 0.0000 | 152/152 |
| Half exposure | 46 | native_point | -0.00000 | 0.51544 | 0.0000 | 0.0000 | 152/152 |
| Half exposure | 47 | NODE_all20_median | -0.00000 | 0.45230 | 0.0000 | 0.0000 | 152/152 |
| Half exposure | 47 | NODE_original_policy_median | -0.00000 | 0.45230 | 0.0000 | 0.0000 | 152/152 |
| Half exposure | 47 | native_point | -0.00000 | 0.50180 | 0.0000 | 0.0000 | 152/152 |

Displayed negative zeros are small unrounded values, not evidence of materially
negative cellularity. No positivity clipping was applied.

## Matched recipes and provenance

All six cells used 120 observations with 1% native masked physical Gaussian
noise over days 5-70, exactly paired C/M times and Gaussian innovations per
seed, and the same day-5 observed IC. Noise masks and absolute noise scales
follow each source trajectory. Both methods receive the same qualified
0.25-day source and frozen 4001-knot forcing table within each cell. Training
interpolation uses source knots no later than day 70. The known cyclic regimen
continues at days 80 and 100; this is not unseen-dose or held-out-IC prediction.

The original source has growth k=.025, diffusion d=.05, capacity theta=1,
drug decay .7 and effective sensitivity .5, with dose dates 20,40,60,80,100.
TumorTwin normalizes equal dose arrays, so the half-exposure factor is realized
exactly once through effective sensitivity .25 rather than by relying on a
uniform rescaling of those dose arrays. This is a synthetic effective-exposure
change, not a clinical dosing recommendation.

Native inference retains cABN (40 operator coefficients), four modes, 200
GP/constraint nodes, derivative/IBP weights 1/8, diagonal covariances,
`gamma2=.035`, `mll_weight=.1`, spectrum/cadence priors, zero-mean block
hierarchy, `sigma_O=5`, normal equations, AutoNormal/ClippedAdam, learning rate
.003, seed 42 and 12,000 updates. **There is no ODE in its inference objective.**

NODE retains the original 5-128-128-128-4 tanh network (34,308 parameters), 20
original seed-42 keys, 6,000 Adam updates, learning rate .0005, global clipping
at 1 and unweighted observation rollout MSE, without rescaling or regularization
changes. The model classes are not capacity-matched. Adaptive force-aware
Tsit5 (`rtol=1e-5`, `atol=1e-7`, `dtmax=.25`) is an explicit numerical
adaptation, not unchanged legacy constant-step optimization.

The genuine source NODE policy filters last recorded PRE-UPDATE losses above
three times their all-20 median. That policy was restored before any
confirmation NODE fit or forecast, correcting the earlier proposal's
unfiltered-mean comparison. All six filter masks were frozen before any fitted
confirmation forecast. Scanned optimization was rejected when it differed from
scalar updates; the exact qualified scalar loop was retained.

Independent NumPy float64 DOP853 splits every input knot and dose/window
boundary, with 42 fixed Radau controls. The 1e6 common-RMS, 200,000-RHS-call and
120-second safety policy preserves censored/unavailable tails and negative
predictions. A censor would not prove mathematical finite-time blow-up.
Native-draw numerical/reliability diagnostics remain separate from the point.

Accounting: six native confirmation fits, 72,000 SVI updates and 3,000 saved
operator draws; 120/120 NODE members completed, all 720,000 nominal updates;
510/510 primary trajectories completed and 42/42 Radau controls passed. The
earlier four development fits add 48,000 SVI updates. No new confirmation FOM
or basis fits were added.

## Execution recovery and persistence

Execution was interrupted after all six native fits, one full NODE ensemble,
18 members of the second ensemble and a durable step-1000 checkpoint for its
next member. All completed outputs remained unchanged. The missing first-1000
loss-history prefix was reconstructed with the original deterministic scalar
updates, requiring bitwise identity at every existing checkpoint before
continuing. The extra 1,000 replayed hardware updates are not new scientific
optimization steps. Unknown unlogged work after the last durable checkpoint
cannot be counted precisely. Subsequent recovery checkpoints also retain loss
prefixes. No alternate initialization or favorable checkpoint was selected.

Parent audits verified all six physical acquisitions, paired innovations,
training-only interpolation, input/IC/decoder identities, native operator-draw
exports, and all 120 NODE final update/Adam/filter contracts. A separate final
array audit independently replayed all 510 trajectory scores across the
training and forecast windows, all 42 saved DOP853/Radau comparisons, both NODE
centers, posterior counts, burden-band inclusion, simple baselines and the
primary/refined-reference cohort rules. No fits or forecasts were rerun for
that array audit.

Numerical data are persistent local session artifacts, not a portable dataset
bundled with this record. No production defaults, canonical caches, previous
reports or manuscript were changed; no push or model promotion is implied.
Recovery commit `0f5e4ba0b358469656efd18489e78bd3c0dee4f1` remains available as
`backup/growth-node-before-comparison-0f5e4ba`. The completed growth research
record was committed separately as `7784db13856b290e99ff7d095e2d1c93a4e95d64`.

Artifact root:
`/Users/anthonypoole/.copilot/session-state/270528ea-ceb6-45be-9e3d-35eb315a865f/files/chemo-regression-diagnostics`.

Confirmation outputs are under `chemo_task_confirmation/`. Supplemental
audits, all eight development/fresh-acquisition GP reconstruction plots and
machine-readable report tables are under `chemo_parent_confirmation/`.

| Evidence | SHA256 |
| --- | --- |
| Confirmation authorization | `a69d1213c7a0e369e35bb18b414acecb86e42155a5f4bb5bc9fcceb6fd5bbbe1` |
| Confirmation release | `4b683e339fee72f3d2fe0002f7fcd08a4caf9a49b64e16d03758919ba80edf4c` |
| Confirmation artifact manifest | `37ce13b1a2c73a7a17dfcdff9f89de7d0f3c3b9aa0bc0a6fe6488807ba08c84c` |
| Primary comparison summary | `41ee85a3d3ae89d7e983d299ddba294e5083df0222b859eb67521e3078365d27` |
| Machine-readable report tables | `947752008015e6592841ee35f6642d412112ded1939ddfb8e30de2308100cf98` |
| Independent final array checks | `782b19fb83433eb4bfe04a2619a082b93c3e35b4a50715475fd60f5d38b12138` |

Earlier evidence remains in `chemo_task_design/release.json`,
`chemo_task_refinement/release.json`,
`chemo_parent_verification/development_audit.json` and
`chemo_parent_confirmation/data_checks_v2/summary.json`. The first parent input
audit's float64-versus-native-float32 assumption failure is preserved; only the
audit was corrected.
