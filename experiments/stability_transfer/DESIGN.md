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

Report full-order velocity/state errors with full-state and fluctuation
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
