# Cylinder wake (incompressible Navier–Stokes, Re = 100): design

Written before any ROM fit; the code (`fom_dolfinx.py`, `config.py`, `04_unified.py`) implements exactly this.
Purpose: test the method beyond the paper's small benchmarks. All of those use r ≤ 6 modes, and none has an
advection-dominated flow. NODE is not run here (gated).

## Full-order model
- DFG benchmark 2D-2 (Schäfer & Turek 1996). Channel [0, 2.2] × [0, 0.41], cylinder of radius 0.05 at (0.2, 0.2),
  ν = 1e-3, parabolic inflow with mean 1 (Re = 100), no slip, zero outflow pressure. The flow starts from rest and
  the inflow is ramped over 0.5 time units.
- FEniCSx 0.9: Taylor–Hood P2–P1 elements and incremental pressure correction (Crank–Nicolson diffusion,
  Adams–Bashforth convection).
- Three resolutions:
  - coarse: h 0.008 / 0.02 / 0.04, dt 1.25e-3;
  - medium (production): h 0.004 / 0.0125 / 0.03, dt 6.25e-4;
  - fine: h 0.002 / 0.008 / 0.02, dt 3.125e-4, T = 12.
- Validation: the medium run must fall in the benchmark's reference ranges, or within 2% of the fine run:
  - Cd_max 3.22–3.24;
  - Cl_max 0.99–1.01;
  - St 0.295–0.305.
- Observed state: the velocity (u, v) on a uniform 0.01 grid over the channel, like a PIV measurement. That is
  221 × 42 points, zero inside the cylinder, 18,564 values per snapshot. Snapshots are saved every 0.005 to T = 24,
  and training samples between saved times use a cubic interpolant in time.

## Tasks (absolute FOM time)
- **lc**, periodic shedding: train on [16, 17] (about three periods), forecast to 19 (about six more).
- **tr**, onset of shedding: let A_k be the lift amplitude of cycle k relative to its limit-cycle median
  (t ≥ 16).
  - t_b = the first cycle with A_k ≥ 0.6.
  - t_a = the last cycle before t_b with A_k ≤ 0.1.
  - Train on [t_a, t_b] and forecast to t_a + 3 (t_b − t_a): the ROM must predict the saturation it has not seen.

## Data regimes (the paper's Euler regimes, scaled per time unit)
| regime | samples per time unit | noise |
|---|---|---|
| dense_low_noise | 250 | 1% |
| sparse_low_noise | 55 | 3% |
| dense_high_noise | 250 | 10% |

- Noise is pointwise Gaussian proportional to the state, as in the 2D diffusion–reaction benchmark.
- Sample times are uniform random in the window, with both ends included. The seed is 42 via the pipeline.

## ROM and method
- Uncentred POD of the noisy training snapshots and a cAH ROM. Incompressible Navier–Stokes is quadratic, so cAH
  is the exact Galerkin structure.
- The shared production configuration, identical to Euler's, with no tuning: SVI 12,000 steps, 500 posterior
  draws, 200 integrated, IC uncertainty on, prediction on the FOM's saved grid.
- Runs:
  - headline: r = 9 (mean + four shedding pairs) for both tasks × three regimes;
  - mode sweep: r ∈ {5, 9, 13, 17, 21} for lc and {5, 9, 13} for tr, dense_low_noise.

## Metrics (`summary.json`)
- Paper-style metrics, as in tab:aggregate: train/forecast relative error of the median in reduced coordinates,
  5–95% coverage and width on the forecast window, and the stable %.
- Metrics the mean flow cannot hide. The mean flow dominates the norm, so the paper-style error understates shedding
  errors. These include:
  - errors relative to the fluctuating part of the truth;
  - full-order errors;
  - the POD projection floor;
  - the Strouhal number and amplitude of the leading fluctuating mode: truth vs the median, plus the 5/50/95%
    range over the draws.
- Long-horizon stability: 50 draws integrated to 10× the training window; bounded means |q| ≤ 2 × the largest
  training |q|.
- Reference (not a competitor, no NODE): deterministic least-squares OpInf with finite-difference derivatives of
  the noisy reduced data. Its regulariser comes from 1e-6 to 1e6 and is chosen without the truth: the bounded
  forecast with the smallest error against the noisy training data.

## Amendment after the declared fits started (after the fact; disclosed)
Written on 9 Nov 2026 after 11 of the 12 declared fits had finished. Nothing above is changed. The declared results
are reported as declared; everything here is labelled "after the fact".

1. **Centring was considered and rejected.** The onset fits blew up (0% stable draws). My first guess was that the
   uncentred basis makes the columns 1, q₁, q₁² nearly collinear. A noise-free least-squares check (session script
   `centre_diag.py`) refuted this. On periodic shedding, uncentred least squares is stable with mild
   regularisation, and centring made it worse. On the onset task, both versions blow up. The cause is the basis: the
   POD of the training window cannot represent the saturated wake. Its noise-free projection floor over the forecast
   is 56–71% of the shedding part at r = 5–13. So no ROM on that basis can forecast it. No centred fits were run.
2. **Posterior-mean rescoring.** The declared point forecast is the median of the stable draws. At larger r, few
   draws stay finite (64.5% / 22% / 1% at r = 5 / 9 / 13), so that median rests on a handful of draws. The session
   script `rescore_mean.py` also integrates the posterior-mean operator from the posterior-mean initial state. It
   reads only the saved `04_unified.npz`; there are no refits. It is reported next to the declared numbers, never
   instead of them.
3. **Float64 rerun of r = 21.** The declared r = 21 fit diverged in float32: the SVI loss became NaN after
   3656 s, and the run stopped with NaN operators. Its declared outcome is "failed". It is rerun once with
   `--precision float64`, with everything else identical; the output is `lc_dense_low_noise_r21_f64`. If r = 17 also
   diverges, it gets the same treatment.

## Results (9 Nov 2026; one Azure F16as_v6, each fit pinned to 4 cores)
**FOM.** Medium: Cd_max 3.2167, Cl_max 0.9853, St 0.3019. Fine: 3.2245 / 0.9862 / 0.3019. The two agree within
0.25%, so the medium mesh passes the declared check ("in range or within 2% of fine"). Cd_max and Cl_max sit
0.1–0.5% below the reference ranges on both meshes.

**Full-order forecast error, % of the fluctuating part.** "Declared" is the median of the stable draws.
"PM*" is the posterior mean, scored after the fact (Amendment 2). "Floor" is the POD projection error.
"LS" is least-squares OpInf.

| run | floor | PM* | declared | stable draws | St PM* (truth 0.3019) | LS | fit time |
|---|---|---|---|---|---|---|---|
| lc dense r5 | 13.4 | 13.5 | 14.1 | 64.5% | 0.3019 | 158 | 8 min |
| lc dense r9 | 3.1 | 3.2 | 8.2 | 22% | 0.3019 | 96 | 15 min |
| lc dense r13 | 1.2 | 1.4 | 55.3 | 1% | 0.3018 | 135 | 25 min |
| lc dense r17 | 1.0 | 1.9 | — | 0% | 0.3019 | 149 | 73 min |
| lc dense r21 (float32, declared) | — | — | failed (SVI NaN) | — | — | — | 61 min |
| lc dense r21 float64* | 1.0 | 2.7 | — | 0% | 0.3016 | 151 | 152 min (8 cores) |
| lc sparse r9 | 5.8 | 21.5 | 36.0 | 7.5% | 0.304 | 142 | 46 min |
| lc 10% noise r9 | 7.5 | 18.2 | 19.9 | 13.5% | 0.302 | 171 | 15 min |
| tr dense r5 / r9 / r13 | 71 / 61 / 58 | blows up | — | 0% | — | 125–141 | 13–55 min |
| tr sparse r9 | 61 | blows up | — | 0% | — | 134 | 6 min |
| tr 10% noise r9 | 62 | 158 | 100 | 1.5% | — | 144 | 23 min |

The LS column is the reduced error. Full order = hypot(ratio × reduced, floor), where ratio is 0.99–1.0 for lc
(under 1 point of difference) and 0.78–0.86 for tr (121–136% full order). The declared reduced-coordinate metrics (tab:aggregate style) are in `after_the_fact/results_table.txt`.

**Reading.**
1. **Point forecasts scale.** At r = 5–13 the posterior mean sits within 0.25 points of the projection floor. It
   reproduces the Strouhal number and the amplitude, and it stays bounded at 10× the training window.
   r = 13 is about twice the largest r in the paper. At r = 17 and 21 the mean drifts above the floor
   (1.9 and 2.7 vs 1.0). LS fails everywhere (≥ 96%).
2. **Sampling-based UQ does not scale.** The stable fraction falls from 64.5% to 22%, 1% and 0% as r grows,
   so the declared median degrades and is undefined at r ≥ 17. The posterior is wide in quadratic directions
   that the limit cycle never excites, and draws pick up unstable growth there (`draw_diag.py`). The mean is
   stable. Candidate fix, not tested: a stability-aware operator prior, or rejecting unstable draws inside the
   sampler.
3. **The onset task is infeasible on a training-window basis.** The floor is 56–71% at every r (Amendment 1).
   This limits POD-based ROMs, not the fitting method.
4. **Cost.** Fit time grows roughly like r² to r³ on 4 cores (8 → 73 min from r = 5 to 17). float64 at r = 21 took
   152 min on 8 cores. The sparse r9 fit (46 min vs 15 for dense) was slow for a reason I did not find; it was not
   hung (py-spy showed it inside SVI).

**Disclosures.**
- Posterior-mean scoring, the centring check and the float64 rerun were all added after the fact (Amendment).
- The r = 17 and r = 21 posterior-mean results are in the same category.
- No refit used the truth.
- NODE was not run.

## Stability prior (design written before any fit; Nov 2026)

**Why.** The posterior mean forecasts well, but most posterior draws blow up at r ≥ 9 (Results, reading 2). Goyal,
Pontes Duff & Benner (2023, arXiv:2308.13819) make quadratic OpInf models stable by (i) an energy-preserving (EP)
quadratic term, Σᵢ qᵢ Hᵢ(q⊗q) ≡ 0, plus (ii) a learned trapping region. Incompressible Navier–Stokes with the
Galerkin quadratic term has property (i) exactly in the fluctuation (the convective term conserves kinetic energy).

**What is tested.** Property (i) only, as an exact constraint on the Gaussian operator posterior
(`WeakFormConfig.quad_constraint="energy_preserving"`, `core/weakform_opinf/constraints.py`): C vec(O) = 0 is linear,
so the constrained posterior and its evidence are closed form (Matheron's rule per draw; the evidence gains
log N(0; Cμ, CΣCᵀ) − log N(0; 0, CΣ₀Cᵀ), so the GP hyperparameters are fitted under the constrained model).
Part (ii) is not tested: with Q = I the trapping certificate is never negative even on noise-free EP least-squares
fits (below), and a learned Q needs a nonlinear parameterisation that loses the closed form. The certificate is
reported as a diagnostic only.

**Shifted features** (`feature_shift="training_mean"`). On the limit cycle the first POD coefficient is nearly
constant, so the constant, linear and q₁-quadratic columns are almost collinear. The operator is therefore fitted to
features of z = q − q̄ (q̄ = mean training reduced state) under the same prior, then mapped back to q exactly
(O_q = O_z Gᵀ; H unchanged, so the constraint is the same in both coordinates). The GP, the observations and the
prior scales (computed from q, as before) are unchanged.

**Runs** (all float64, lc task, the declared data regimes and settings otherwise; `04_unified.py` flags):

| variant | flags | runs |
|---|---|---|
| f64 control | `--precision float64` | dense r9, r13 |
| EP | `+ --quad-constraint energy_preserving` | dense r5, r9, r13, r17, r21 |
| shift | `+ --feature-shift training_mean` | dense r9, r13 |
| **EP + shift (primary)** | both | dense r5, r9, r13, r17, r21; sparse r9; 10% noise r9 |

16 fits. The earlier float32 runs (Results) are the free reference at r5 and r17, and the r21 float64 run.
Practicality rule: a fit not finished 8 h after its start is stopped and reported as not run.

**Metrics** (unchanged definitions): stable fraction of the 500 draws; declared median of stable draws (full-order
forecast error); posterior mean PM* (mean of the 500 draws; `after_the_fact/rescore_mean.py`); long-horizon bounded
draws (10× window); band coverage in the forecast window (`ci_cov_ext`); Strouhal. Diagnostics: relative EP violation
‖C vec(PM*)‖/‖C‖‖PM*‖ (≈ 0 by construction for EP variants) and the Q = I trapping certificate of PM*.

**Reading rule.** The stability prior "fixes the sampling UQ" at a given r if EP + shift has ≥ 80% stable draws and
its declared median is within 3 points (full order) of the free PM* at that r. Judged at r9 and r13 (where the free
mean is near the floor); r5, r17, r21 and the sparse/noisy runs are reported, not judged.

**Expectations** (written before any fit):
1. EP + shift raises the stable fraction at every r above the free runs (64.5/22/1/0/0%).
2. EP + shift PM* is within 3 points of the free PM* at r ≤ 13 (the constraint costs a little accuracy).
3. EP without the shift gives a clearly worse PM* than EP + shift (collinearity; see disclosure).
4. Shift alone does not fix stability (stable < 50% at r13).
5. The Q = I trapping certificate is never negative (no certified trapping region).
I am least sure of 1 at r17/r21 and of the reading rule at r13.

**Disclosure: after-the-fact diagnostics that shaped this design** (session scripts, no refits, no repo change).
(a) Projecting the stored r5 posterior draws onto EP (Matheron with per-row sample moments): 94% stable (free 61.4%)
but the mean error rose to 406% (free 13.5%); its operators were 5–11× larger with near-identical training residuals
(0.62 vs 0.39%) → collinearity, which motivated the shift. (b) Noise-free least squares on the truth's training
window: EP costs little (reduced forecast error r5 0.078 vs 0.064, r9 0.102 vs 0.070, r13 0.111 vs 0.071), all
bounded at 10×; the Q = I certificate was ≈ 0 and never negative; a centred POD basis made EP fits bad (30–155%), so
the basis stays uncentred. (c) A crude Bayesian stand-in (Savitzky–Golay derivatives) was inconclusive.

## Centring the existing benchmarks (design written before any fit; 9 Nov 2026)

Question (user): does fitting the operator about the mean training state (`feature_shift=training_mean`), which
helped the cylinder, also improve the paper's existing benchmarks?

Which benchmarks it can touch (checked in code before any fit):
- Euler: the basis subtracts the mean of the training snapshots before POD (`euler/config.py` `fit`), so the
  mean reduced training state is 0 and the shift is the identity -> same fit; not run.
- Heat: also mean-shifted, and its library has inputs (cAHBN), which `feature_shift` does not support; not run.
- Tumour (dose change): uncentred basis but with inputs (cAN); shifting there adds a pure-input term, i.e. a
  different model; not run (needs its own design).
- 2D diffusion-reaction (`burgers_2d`, dense medium noise, cAH, r = 3, plain uncentred POD): the only one it
  applies to. Run: `04_unified.py dense_medium_noise --feature-shift` (output `dense_medium_noise_shift`) and,
  on the same machine, the unchanged `04_unified.py dense_medium_noise` into a copy of the repo (platform check
  against the paper's saved npz).

Scored with `aggregate_table.score` (the paper's table metrics, reduced coordinates): stable %, training and
forecast error of the draw median, 5-95% band coverage and width in the forecast window. No pass/fail; the shift
is adopted for this row only if forecast error is lower AND coverage is not lower. Expectation: little change
(the paper's run is already 100% stable with 2.4% forecast error).
