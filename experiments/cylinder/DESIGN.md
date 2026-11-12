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

### Results (9-10 Nov 2026; 5 Azure F16as_v6, float64; `after_the_fact/score_extra.py` + `report_stability.py`)

13 of the 16 declared fits finished; r5 EP, r5 EP + shift and sparse r9 EP + shift hit the 8 h limit (rc 124)
and are not run. Azure: rg-thesis-stability, VMs 15:42-01:55 UTC (34 VM-hours), deleted (ResourceGroupNotFound);
cost estimate ~$38.

Full-order forecast error, % of the fluctuating part; floor = POD projection. "Stable" = draw stays finite over
the window (the declared definition), so a stable draw can still be wildly wrong. Long = draws bounded at 10x.

| run | floor | PM* | declared median | stable % | long | St (PM*) |
|---|---|---|---|---|---|---|
| r5 free (earlier) | 13.5 | 13.5 | 14.1 | 64.5 | 25/50 | 0.302 |
| r5 EP | 13.5 | not run (8 h limit) | | | | |
| r5 EP + shift | 13.5 | not run (8 h limit) | | | | |
| r9 free (earlier) | 3.1 | 3.2 | 8.2 | 22.0 | 2/50 | 0.302 |
| r9 f64 control | 3.1 | 5.3 | 7.6 | 18.5 | 2/50 | 0.302 |
| r9 shift | 3.1 | 3.1 | 3.1 | 37.5 | 5/50 | 0.302 |
| r9 EP | 3.1 | 260 | 203 | 100 | 49/50 | 0.332 |
| r9 EP + shift | 3.1 | 79.8 | 213 | 99.5 | 4/50 | 0.303 |
| r13 free (earlier) | 1.2 | 1.4 | 55.3 | 1.0 | 0/50 | 0.302 |
| r13 f64 control | 1.2 | 12.1 | 29.6 | 1.5 | 0/50 | 0.303 |
| r13 shift | 1.2 | 1.2 | 1.4 | 20.5 | 0/50 | 0.302 |
| r13 EP | 1.2 | 2.2 | 156 | 100 | 50/50 | 0.302 |
| r13 EP + shift | 1.2 | 1.2 | 15.5 | 100 | 7/50 | 0.302 |
| r17 free (earlier) | 1.0 | 1.9 | - | 0 | 0/50 | 0.302 |
| r17 EP | 1.0 | 3.5 | 250 | 100 | 50/50 | 0.302 |
| r17 EP + shift | 1.0 | 168 | 112 | 100 | 8/50 | 0.506 |
| r21 free (earlier, f64) | 1.0 | 2.7 | - | 0 | 0/50 | 0.302 |
| r21 EP | 1.0 | 1.9 | 344 | 100 | 50/50 | 0.302 |
| r21 EP + shift | 1.0 | 2.0 | 1.0e8 | 100 | 0/50 | 0.302 |
| sparse r9 free (earlier) | 5.8 | 21.5 | 36.0 | 7.5 | 1/50 | 0.304 |
| sparse r9 EP + shift | 5.8 | not run (8 h limit) | | | | |
| 10% noise r9 free (earlier) | 7.5 | 18.2 | 19.9 | 13.5 | 2/50 | 0.302 |
| 10% noise r9 EP + shift | 7.5 | 10.5 | 69.7 | 100 | 13/50 | 0.303 |

**Declared reading.** The stability prior does NOT fix the sampling UQ at r9 (99.5% stable but declared median
213 vs free PM* 3.2) or r13 (100% stable, 15.5 vs 1.4 + 3). Expectations (judged only where the fits ran): 1 MET at r9-r21 (r5 not run), 2 NOT MET
(r9 PM* 79.8), 3 NOT MET (EP alone is better than EP + shift at r17 and ties at r21), 4 MET (shift alone 20.5%
stable at r13), 5 MET (certificate min 3e-4, never negative).

**What the energy-preserving constraint does.** It makes every draw finite (99.5-100%) and keeps the posterior
mean close to the floor at r13-r21 (1.2-3.5%; Strouhal exact). But the draws are not useful forecasts: the median
of the "stable" draws is 15-344% (1e8% at r21 with the shift), because the constraint only removes energy
production by the quadratic term; the linear part can still amplify, and with the shift the operators are huge
(|PM*| 1374-5806), so draws are stiff and grow without leaving the finite range inside the window. At r5 and on the sparse data the
draws were so stiff that the 500-draw forecast did not finish in 8 h.

**Run-to-run spread.** The float64 controls differ from the earlier float32 runs at the same settings (PM* r9
5.3 vs 3.2, r13 12.1 vs 1.4; stable 18.5 vs 22, 1.5 vs 1), so a single fit's PM* moves by several points with
the arithmetic path alone. Comparisons above are within this round where possible (shift vs f64 control).

**After the fact (not declared): the shift alone.** Fitting about the mean training state, with no constraint,
puts the declared median of the stable draws at the floor (r9 3.09 vs 3.07; r13 1.41 vs 1.16) and the posterior
mean at the floor too, with 100% band coverage. It doubles (r9 22 -> 37.5%) or 20x (r13 1 -> 20.5%) the stable
fraction, but most draws still blow up, and few stay bounded at 10x (5/50, 0/50). This finding was not a declared
expectation (expectation 4 only said it would not fix stability), so it needs its own declared run before it is
claimed (proposed: shift alone at r5, r17, r21, sparse and 10% noise).

**Disclosures.**
- An ssh connection problem with stab3 during the round (session record); both stab3 fits ended with rc 0.
- Core layout per fit: 4 cores (r5-r13, sparse, 10%), 8 (r17), 16 (r21, one fit per VM); declared settings otherwise.
- Expectation 3's "clearly worse" was operationalised as "> 1 point worse at every r" before any result was read.
- f64 r13 control was queued behind the sparse run on stab5 and relaunched early on idle cores (same command);
  the queue's lane-1 shell was killed to do so, so the sparse run wrote no .rc (exit status from its .time file).
- Runs that hit the 8 h limit did so while integrating stiff posterior draws with explicit RK45 (py-spy on the
  VM: `generate_rom_predictions` -> `solve_ivp` -> `rk_step`), inside the main forecast (`_predict_targets`), before
  `04_unified.npz` is written. Per the practicality rule they are reported as not run. (During the round I wrote
  `summary_fallback.py` for runs killed later, in the 10x long-horizon check; no run ended up needing it.)
- `timeout` killed `/usr/bin/time` too, so killed runs have an empty `.time` file.
- `report_stability.py` figure 2 gained the shift-alone column after the results were seen.

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

### Result (9 Nov 2026; stab5, 4 cores each, both fits ~88 s, rc 0)

| run | stable % | train | forecast | coverage | band width (forecast) |
|---|---|---|---|---|---|
| paper npz (as published) | 100 | 0.35% | 2.36% | 100% | 0.696 |
| unchanged rerun (VM) | 100 | 0.35% | 2.36% | 100% | 0.693 |
| centred (`--feature-shift`) | 96.5 | 0.38% | 1.99% | 100% | 7.31 |
| Neural ODE (paper) | 100 | 1.95% | 25.1% | 12% | 0.731 |

- Platform check: the rerun matches the paper's metrics to 3 significant figures; individual draws differ (max
  |diff| 1.4 in reduced units) because the VM's float32 SVI path differs from the Mac's.
- The declared rule (lower forecast error, coverage not lower) is met: 2.36 -> 1.99%, 100% -> 100%. But the
  band is 10.5x wider (all three modes, 5.3-8.4 vs 0.61-0.84) and 7 of 200 draws blow up. Band width was left
  out of the rule; that was my mistake, noticed only after the result. A 10x wider band at the same coverage
  is a worse uncertainty estimate, and it would turn the row's band from as sharp as the Neural ODE's (ratio
  1.05) to 10x wider. Reading: centring does not help this benchmark; the paper keeps the uncentred fit.
- Why plausibly: on a 3-mode basis whose mean is far from 0, the zero-mean operator prior is a weaker
  constraint about the mean than about 0, so the posterior spreads more. On the cylinder (r >= 9) the same
  shift helps because the uncentred fit's draws mostly blow up.
- Scores: session `centring-existing/score_centring.py` -> `centring_scores.json` (aggregate_table.score).

## Higher-mode variant: choosing the Reynolds number (written before any run at Re ≠ 100; 10 Nov 2026)
Why: at Re 100 the flow needs only about 11 modes. With 1% noise, the forecast-window POD floor (relative to the
fluctuating part) is 3.1% at r9, 1.2% at r13 and plateaus near 1% (the noise in the basis) from r15. So the r17/r21
fits do not test scalability. Goal: a variant of the same benchmark that needs ≥ 20 modes for good accuracy. Raising
Re adds harmonics and sharper wake structures, so the POD spectrum decays more slowly.

- Only change: viscosity ν = 0.1/Re via `fom_dolfinx.py --nu` (default unchanged; Re 100 data untouched). Same
  geometry, inflow, observed grid, saving (every 0.005), lc task (train [16, 17], forecast to 19), regimes.
- Scout runs (Azure, serial FEniCSx jobs, no ROM fits):
  - medium mesh (h 0.004 / 0.0125 / 0.03, dt 6.25e-4, T 24) at Re 150, 200, 300, 400, 600, 800, 1000;
  - fine mesh (h 0.002 / 0.008 / 0.02, dt 3.125e-4, save_every 16, T 20) at Re 300, 400, 600, 800, 1000.
  - A run that stops with a non-finite drag is rerun once at half the time step (disclosed).
- Scout quantities (`scout_modes.py`; reproduces the stored Re 100 floors: r9 3.07%, r13 1.16%, r17 0.99%):
  - Cd_max, Cl_max, St over t ≥ 16;
  - periodicity over t ≥ 16: range of per-cycle lift amplitudes and periods (relative), share of the lift's
    fluctuating power at the fundamental and its harmonics;
  - POD floor vs r (1-48), basis built exactly as `04_unified.prepare` (250 samples, 1% noise, seed 42), and
    r_2% = the smallest r with floor ≤ 2% (Re 100: 11).
- Selection rule (applied once, to the scout table):
  1. periodic: amplitude range ≤ 2%, period range ≤ 1%, harmonic share ≥ 0.99 (Re 100: 0%, 0%, 1.000), on every mesh
     run at that Re. A quasi-periodic or chaotic wake cannot be forecast from a 3-period window by any model and is
     excluded.
  2. needs ≥ 20 modes: r_2% ≥ 20 on the production mesh.
  3. mesh check, where a fine run exists: medium within 2% of fine on Cd_max, Cl_max and St, and r_2% within ±2.
     Medium is production if it passes; otherwise fine is production (T 20 covers the lc window).
  - Choose the lowest Re meeting 1-3 (fewest modes that still reach 20; cheapest fits).
  - If none qualifies, report the table and do not run fits. The fallback (a periodic Re with the largest r_2%, or a
    different geometry) is then a separate, declared decision.
- The ROM fits (r sweep, regimes, expectations, time limit) are declared in a separate section after the scout,
  before any fit.

### Re scout outcome (medium mesh; fine Re 300/400/600 still running when written)
| Re | Cd_max | Cl_max | St | amp / period range | harmonic share | r_2% | floor r9 / r13 / r17 / r21 |
|---|---|---|---|---|---|---|---|
| 150 | 3.2094 | 1.6583 | 0.3131 | 0.00% / 0.00% | 1.0000 | 11 | 3.66 / 1.47 / 0.96 / 0.89% |
| 200 | 3.2762 | 2.1203 | 0.3206 | 0.00% / 0.00% | 1.0000 | 13 | 3.95 / 1.74 / 1.04 / 0.87% |
| 300 | 3.3784 | 2.6075 | 0.3321 | 0.00% / 0.00% | 1.0000 | 14 | 4.19 / 2.05 / 1.21 / 0.92% |
| 400 | 3.4353 | 2.8227 | 0.3434 | 0.00% / 0.00% | 1.0000 | 14 | 4.37 / 2.14 / 1.29 / 0.94% |
| 600 | 3.6360 | 3.0349 | 0.3527 | 0.00% / 0.00% | 1.0000 | 14 | 4.50 / 2.28 / 1.38 / 1.01% |

- Re 800 and 1000 diverged on both meshes (drag grew to 1e27-1e83 without turning non-finite, so the declared
  automatic rerun never triggered). Medium 800/1000 were killed by hand and rerun once at half dt; both diverged again
  and were killed. Fine 800/1000 were killed by hand. Fine 800 was not rerun at half dt (≈ 20 h; it could only matter
  if nothing ≤ 600 qualified, and the medium half-dt run had already failed). Disclosed deviation.
- The 12 jobs ran at once on 16 cores and were ~3× slower than a lone run (memory bandwidth).
- **No Re qualifies**: every resolvable Re is strictly periodic, but r_2% saturates at 14. A single 2-D cylinder wake
  stays low-dimensional. No ROM fits at higher Re.
- Exploratory checks after the outcome (session scripts, no fits, not part of the rule):
  - the start-up transient (onset to the limit cycle) still has a 9-12% floor at r29;
  - a joint basis of several Re needs 24-28 modes for 2 Re and 33-38 for 3 Re, but a held-out Re is not
    representable (floor plateau 7-13%), so only in-sample multi-Re forecasts would be possible.

## Fallback: the fluidic pinball (written before any pinball run; 11 Nov 2026)
Why: the decision deferred by the rule above. The fluidic pinball (three cylinders on an equilateral triangle; Deng,
Noack, Morzyński & Pastur 2020, JFM 884 A37) is a standard ROM benchmark. Its periodic regimes have gap jets and a
deflected wake, so they may need more modes than one cylinder at the same Re, without leaving the periodic class that
rule 1 requires.

- `fom_dolfinx.py --geometry pinball`: three cylinders of D = 0.05, side 1.5 D, pointing upstream, front centre
  (0.2, 0.2); same channel, inflow, observed grid, saving and lc task. Re = 0.05/ν; forces summed over the cylinders.
  Blockage 0.125/0.41 = 30% (the unconfined bifurcation Re, about 18 / 68 / 104 / 115 for shedding / asymmetric /
  quasi-periodic / chaotic, will shift; hence a scan).
- Scout (Azure, no ROM fits): pinball mesh h 0.002 / 0.00625 / 0.015 (the medium mesh scaled to D = 0.05), dt
  3.125e-4, save_every 16, T 20, at Re 60, 75, 90, 105, 120. Non-finite or divergent runs: rerun once at half dt.
- `scout_modes.py` as for the cylinder (cylinder output unchanged; pinball: lift crossings about the mean lift,
  Strouhal with D = 0.05, floors to r 96).
- Selection: rules 1 and 2 above unchanged (periodic; r_2% ≥ 20). Choose the lowest qualifying Re. Rule 3 (mesh
  check) on that Re only, against a finer pinball mesh (h 0.0013 / 0.0042 / 0.01, dt 2.0833e-4, save_every 24, T 20).
- If none qualifies: report and stop. The ROM fits for a qualifying Re are declared in a separate section, before
  any fit.

### Amendment 1 (11 Nov 2026, 03:05 UTC, before any pinball result; runs at t ≤ 0.5 were stopped)
- The first launch crashed at start-up (a `PointSampler` argument shadowed `dolfinx.geometry`; fixed in 334c850, no
  step taken). The relaunch ran at ~2,600 s per time unit, so T 20 would take ~14.5 h and the fine check > 40 h.
- The lc task is now defined in convective units (D/U), as it should be for a different diameter. The cylinder's
  [16, 17] → 19 is 160 / 10 / 30 D/U, so the pinball's (D = 0.05) is train [8, 8.5], forecast to 9.5, with the same 250
  basis samples (`scout_modes.py`; cylinder output unchanged, checked on the stored Re 100 file). Pinball runs (scout
  and fine check) use T 10 (200 D/U). Everything else is as declared.
- The fine cylinder runs (Re 300/400/600, at t 7-9.5 of 20) were stopped: no Re qualified, so rule 3 does not apply to
  them, and they were slowing the pinball runs. The scout table above is the medium mesh only.

### Pinball scout outcome (11 Nov 2026; T 10, pinball mesh, all runs rc 0, none diverged)

| Re | St (D/U) | amplitude range | period range | harmonic share | floor r9 / 13 / 21 / 29 | best floor (r ≤ 96) | r_2% |
|---:|---:|---:|---:|---:|---|---:|---:|
| 60 | 0.159 | 19.0% | 1.5% | 0.998 | 17.7 / 14.0 / 11.8 / 11.6% | 11.5% | never |
| 75 | 0.170 | 13.0% | 5.8% | 0.986 | 24.9 / 21.6 / 19.9 / 19.2% | 19.1% | never |
| 90 | 0.191 | 55.6% | 7.8% | 0.986 | 34.2 / 32.0 / 30.7 / 30.3% | 30.3% | never |
| 105 | 0.184 | 50.6% | 56.3% | 0.685 | 75.3 / 74.2 / 72.4 / 71.9% | 71.7% | never |
| 120 | 0.237 | 96.5% | 59.2% | 0.640 | 79.9 / 79.3 / 78.3 / 77.9% | 77.7% | never |

**No pinball Re qualifies; per the declaration, report and stop.** Every run fails rule 1. At Re 60-90 the shedding is
still growing at t = 10 (Cd still drifting; mean lift away from zero, i.e. a slow symmetry break), so the forecast
window holds states the training window never visits and the floor plateaus at 12-30% whatever the rank. At Re 105
and 120 the wake is irregular (harmonic share 0.64-0.69). Rule 3 was not run. No ROM was fitted.

Reading: the lc task needs a flow that has settled onto a periodic cycle. With 30% blockage the pinball's shedding
saturates more slowly than 200 D/U at Re ≤ 90, and is already irregular at Re ≥ 105. Longer runs might let Re 60-75
settle, but a settled single-frequency cycle is the case where the single cylinder needed only 11-14 modes, so this
was not pursued.

Disclosures (run events):
- The first stop of the T 20 pinball runs and the fine cylinder runs (amendment 1) missed its PIDs. These jobs ran
  alongside the T 10 relaunch for about 1.5 minutes until they were killed by PID (03:03 UTC). None of them wrote an
  output file. Their logs are kept in the VM's `logs/pinball_T20_stopped/` and `logs/fine_stopped/` (session copy).
- The T 10 logs contain stray null bytes, so the VM's divergence watchdog (a plain `grep`) could not read them. No
  divergence happened: Cd stayed at 6.4-7.4 in every run (checked by hand during the runs), and all runs ended rc 0.
- Azure VM (F16as_v6, eastus) ran 23:19 UTC 10 Nov to 08:30 UTC 11 Nov (~9.2 h, ~$10.5 with disk and IP). The
  resource group was deleted. The FOM files (5 cylinder Re, 5 pinball Re; 2.4 GB) were kept in the session store,
  sha256 checked.

### Where this leaves "20+ modes"
A periodic two-dimensional wake (one cylinder at Re 100-600, or the pinball where it is periodic) needs at most
14 modes for 2%. Flows that need 20+ modes either never settle onto the cycle seen in training (onset transients,
irregular wakes), so no basis from one short window can forecast them, or mix several cycles. The only forecastable
20+-mode setting found is parametric: one basis over 2-5 Reynolds numbers (r_2% 23-28 for 2 Re, 33-38 for 3, 48-55
for 5; exploratory scout, no fits). This needs ν as an input to the model, and `feature_shift` does not yet support
inputs. A held-out Re is not representable on such a basis (floor 7-13%), so only in-sample Re can be forecast.
Not declared; awaiting a decision.

## Parametric multi-Re ROM (design written before any fit; 11 Nov 2026)
Why: the only forecastable 20+-mode setting found above. One POD basis is shared by two Reynolds numbers, and the
viscosity enters as a known input, so a single learned model covers both flows.

- Data: medium-mesh cylinder FOMs at Re 150 and 300 (training) and Re 200 (held out), the same files as the Re scan.
  lc task, training window [16, 17], forecast to 19. 250 samples per Re (random times, ends pinned, drawn in turn from
  the pipeline's seeded RNG), 1% noise. The scout used seed 42 for every Re, so floors here may differ slightly; the
  floors reported are those of the run.
- Basis: uncentred POD of the stacked noisy training samples of both Re (scout r_2%: 24 at Re 150, 25 at Re 300).
- Model: Galerkin Navier–Stokes with ν = ν_ref·u is exactly `cAHBN` with a constant input u = Re_ref / Re (Re_ref
  200; u 4/3 at Re 150, 2/3 at Re 300, 1 at Re 200): dq/dt = c + Aq + H(q⊗q) + u(b + Nq). Same weak-form Bayesian
  method, settings and float64 precision as `04_unified.py`; one GP per trajectory; input-aware `feature_shift` and
  `quad_constraint` (commit 6ac04d6, tests). Runner `05_multi_re.py` (commit 91d8fcf; smoke at r3 on subsampled
  files only, not results).
- Sweep: r ∈ {17, 21, 25, 29} × {free, shift (`--feature-shift training_mean`)} = 8 fits. The energy-preserving
  constraint is not run (its draws were stiff and its forecasts 15-344% in the stability round).
- Scored per Re (`summary.json` `per_re`): training Re on the forecast window (17, 19]; held-out Re on all of
  (16, 19], since the model never saw it. PM* = the mean operator from the mean initial state (as in
  `after_the_fact/rescore_mean.py`): full-order fluctuation error, Strouhal, bounded to 10× the training length.
  Draws: median forecast error, % stable, 5-95% band coverage, 50 long draws bounded. Floor = the true state projected
  on the basis. LS reference = deterministic cAHBN OpInf on both trajectories (λ: smallest training misfit among the
  values bounded at both training Re; no truth used).
- Primary reading (fixed now): the shift arm at r 25 succeeds if PM* is bounded and ≤ 5% full-order fluctuation error
  at BOTH training Re, with the Strouhal number within 2%. Everything else is descriptive.
- Practicality: a fit not finished 14 h after its start is stopped and reported as "not run".
- Azure: 4 × F16as_v6 (one region each), two fits per VM on 8 cores each (pairs r29+r17 and r25+r21, one arm per VM).
  Estimate 8-13 h for r29, ≤ $60. The VM smoke (r5, `--smoke`) is a plumbing check, not a result.

Expectations (written before any fit):
1. Floors at r 25 ≤ 2.5% at both training Re.
2. Primary: shift r 25 PM* ≤ 5% at both training Re, St within 2%.
3. In the shift arm, PM* error falls from r 17 to r 25 at both training Re.
4. Draws: < 50% stable at r ≥ 21 in both arms; the stable-draw median is not near the floor.
5. Held-out Re 200: PM* error ≥ its floor (~8%) and ≤ 20%.
6. LS: unbounded or > 50% at every r.
7. Every fit finishes within 14 h.

### Multi-Re results (12 Nov 2026; 4 Azure F16as_v6, 8 cores per fit, float64; all 8 fits rc 0)
Forecast-window full-order fluctuation error (%), training Re on (17, 19], held-out Re 200 on (16, 19]. "unb" = the
mean operator (PM*) is unbounded on the window. Floors and LS are identical across arms (same basis and data).

| r | POD energy | floor 150 / 300 / 200 | PM* free 150 / 300 / 200 | PM* shift | draws stable | LS 150 / 300 / 200 (λ) | runtime free / shift (h) | peak RSS (GB) |
|---|---|---|---|---|---|---|---|---|
| 17 | 0.99972 | 4.36 / 3.76 / 8.29 | unb / unb / unb | unb everywhere | 0% | 95 / 231 / 174 (3162) | 1.74 / 1.75 | 6.7-6.9 |
| 21 | 0.99983 | 2.45 / 2.65 / 7.89 | 99.4 / unb / unb | unb everywhere | 0% | 167 / 254 / 174 (3162) | 3.14 / 3.34 | 8.1-8.2 |
| 25 | 0.99987 | 1.89 / 1.82 / 7.78 | 18.9 / 49.0 / unb | unb everywhere | 0% | 177 / 212 / 156 (1e4) | 6.55 / 7.94 | 9.3-9.5 |
| 29 | 0.99989 | 1.33 / 1.39 / 7.71 | unb / unb / unb | unb everywhere | 0% | 152 / 381 / 187 (3162) | 12.27 / 12.34 | 10.8-10.9 |

r 25 free: Strouhal 0.3148 vs 0.3131 (Re 150), 0.3385 vs 0.3321 (Re 300). The shedding mode keeps phase and
amplitude; the mean-flow mode q1 drifts (Re 300 strongly after t ≈ 1.3). Long-horizon (10×) PM* and all 50 long
draws are unbounded in every run. Draw medians, bands and coverage are undefined (no stable draws).

Expectations: 1 met (r 25 floors 1.89 / 1.82); 2 (primary) NOT met (shift r 25 PM* unbounded); 3 NOT met (shift PM*
unbounded at every r); 4 met (0% stable at every r; no stable-draw median exists); 5 NOT met (held-out PM* unbounded
at every r); 6 met; 7 met (longest 12.34 h).

After the fact (descriptive; session `diag.py`, run on the VMs from the saved fit npz, r 21 / r 25 / r 29 free):
- The mean operator's derivative residual on projected training truth is 1.1-1.7% with the correct input u and
  1.9-3.1% with inputs swapped. This does not establish identifiability of the viscosity dependence or rule out
  every implementation bug (the original description incorrectly called these GP-smoothed states).
- The posterior does not concentrate: posterior std of an operator entry is 8.5-11× |mean| (median; r 29 11.4×),
  with 275-495 unknowns per row (r 21-29) from two trajectories of one time unit each.
- Draws have unstable Jacobians (max real eigenvalue at the mean state ~13-18 vs 1-2 for the mean operator) and
  diverge at t ≈ 0.27-0.43, before the end of the training window; 0% reach t = 1.
- The mean operator is marginal: from the TRUE initial state it stays bounded over [0, 3] at both training Re for
  r 25, reaches 2.43 (r 21, Re 300), and fails at 2.0 / 1.04 for r 29 (Re 150 / 300); Re 200 fails at 0.32-1.6.

Reading: at 17-29 modes the method fits rates along the observed curves, but with this much data
the posterior is too wide for sampling-based forecasts and the mean operator is not reliably stable; the shift arm,
which helped at single Re, does not help here. This is reported as a scalability limitation, not a result.

Run record and disclosures:
- Smoke fix before the declaration: the layout check called `_assemble_data_matrix` in float32 outside the float64
  context; it now uses `constraints._features` (in 91d8fcf, before any fit).
- FOM files carry no `D` key; `load_fom` defaults D = 0.1 (the cylinder diameter used to make them).
- Launch 16:07 UTC 11 Nov; mre3/mre4 (r 21/25) deleted after their fits finished and results were pulled; the slim
  npz (`O_samples` etc., no true states) was kept for the diagnostic. mre1/mre2 (r 17/29) deleted 04:32 UTC 12 Nov
  after `diag.py` on r 29 free. Resource group `rg-thesis-multire` deleted (confirmed ResourceGroupNotFound).
- Cost ≈ 50 VM-hours × $1.09 ≈ $55-57 (estimate; billing lags).

## Diagnosing and regularising multi-Re instability (12 Nov 2026; before new fits)

The user requested a substantial investigation of priors and algorithm tuning, not another rank sweep.
Safety tag `pre-multire-stabilization` preserves the previous result. Heavy computation runs only on Azure,
in `rg-thesis-rom-stability-20261112`; no NODE experiments.

First, diagnose the saved r25 free/shift fits: compare RK45 tolerances with Radau; assess the feature spectrum,
derivative errors on projected truth, tangent/normal Jacobians, loss progression, and variability of the mean
operator estimated from subsets of its 500 draws. Shrinking saved deviations around their sample mean is a
sensitivity diagnostic ONLY, not a posterior or a proposed UQ procedure. The old diagnostic used projected
truth, not GP-smoothed states as the previous result paragraph incorrectly said. Low derivative residuals do
not establish transverse stability, identifiability of the Re dependence, or absence of all implementation bugs.
The old exports omitted operator-scale hyperparameter draws, so an exact replay of their conditional posterior
is unavailable; new exports must retain those draws and analytic conditional means.

**Stage 1: conditional screening, exploratory, not full Bayesian refits.** Freeze each GP's hyperparameters at
the medians of the saved r25 free run. Reconstruct the derivative/IBP likelihood from the core GP and weak-form
helpers. Recompute its tempering exponent from those fixed hypers (not the original data-only MAP); record it.
Keep the original observations, basis, train/test split, all three Re, and full-order fluctuation metric.
No evaluation truth enters a prior, hyperparameter selection, or derivative training target.

- Coordinates: unshifted or shifted by the pooled training mean.
- Gaussian zero-centred priors: original global scale; output-mode fluctuation-scaled (within-trajectory
  fluctuations); feature-balanced (output fluctuation divided by feature RMS and square root of block width).
  Global SD multiplier: 0.03, 0.1, 0.3, 1, 3, 10, 30. Also likelihood-power multipliers 4 and 16 for the original
  prior at multiplier 1, explicitly testing the effect of tempering, not asserting independence of pseudo-data.
- Additional Gaussian local-Jacobian priors: in fluctuation-scaled coordinates, favour contraction normal to
  the GP trajectory at 12 evenly spaced interior anchors per training Re. Leave its tangent unpenalised.
  Target normal damping 0, 1, 5, 20 per training time; prior SD 0.3, 1, 3, 10 per training time; normalise by the
  square root of the number of anchors so extra anchors do not automatically strengthen the prior.
  This is a new empirical, local stability assumption, NOT a global stability certificate or the EP prior.
- Common random numbers, 32 draws per screen; retain failures, analytic mean, derivative residual, forecast
  accuracy, bounded fraction, and conditional 90% interval coverage. Solver-budget exhaustion is a failure,
  not a silently discarded draw. Stable-only intervals are labelled conditional on survival.

These are development results on an already inspected problem. Advance promising mechanisms, not just the
lowest forecast error: useful accuracy and survival, a clear causal interpretation, and no fabricated posterior
shrinkage. Declare any full joint refits/data extensions and independent confirmation before running them.
Report all screened variants, including negative results. Expectations: stronger likelihood alone will not
identify directions absent from the data; shrinking all coefficients can trade instability for bias;
mode-aware or normal-contraction priors may help, but can suppress real dynamics. A longer periodic record alone
may improve noise averaging without supplying missing off-cycle excitation.

### Stage 1 outcome and joint-refit declaration (before any joint refit)

All 62 conditional screens completed. Ordinary unshifted Gaussian priors were the useful direction:
SD multiplier 0.3 gave PM errors 5.2 / 2.4% (Re 150/300), bounded fractions 31/32 and 30/32, and conditional
median errors 4.8 / 2.3%. Re 200 remained poor: PM 65.7%, 14/32 bounded. Multiplier 0.1 bounded every draw
at both training Re but biased the mean to 7.5 / 4.2%; multiplier 1 gave mean errors 6.1 / 2.9% but only
2/32 and 0/32 bounded draws. Stronger likelihood weights did not fix survival. The mode-aware and local
contraction priors generally traded instability for substantial forecast bias. This is exploratory evidence,
not joint inference and not a new paper claim.

Saved-fit diagnostics: the 377-column noise-free design has 61 directions above 1e-6 of its largest singular
value, only 28 above 1e-3; the noisy design has 274 and 28 respectively. RK45 (two tolerances) and Radau agree
on representative cap-crossing times. The loss plateau precedes the final several thousand updates. Independent
100-draw groups yield substantially different mean-operator forecasts, motivating a Rao-Blackwell estimate:
average the analytic conditional means over hyperparameter draws instead of averaging noisy operator draws.
The IC protocol remains the first noisy reduced observation plus GP-derived uncertainty, NOT a smoothed GP IC.

**Stage 2, ten joint refits:** same shared `build_model`, AutoNormal, learning rate, seeded preparation and
Gaussian operator posterior. Use `06_stabilization.py`, 6000 updates (the old control plateaued by ~4000),
500 operator/hyperparameter draws, checkpoint every 100 updates. Every run starts from the standard initializer,
not from a selected good forecast. Save all losses, hyperparameters including block scales, conditional operator
means, and draws before prediction. All new mean-model metrics use the Rao-Blackwell estimate; also report the
finite-sample operator average to distinguish estimator effects from prior effects.

| name | rank | prior SD multiplier / log-block SD | data seed | data |
|---|---|---|---|---|
| broad-control | 25 | learned, 1 / 3 | 42 | original |
| fixed01 | 25 | fixed 0.1 | 42 | original |
| fixed03 | 25 | fixed 0.3 | 42 | original |
| fixed1 | 25 | fixed 1 | 42 | original |
| narrow03 | 25 | learned, 0.3 / 0.35 | 42 | original |
| rank29 | 29 | fixed 0.3 | 42 | original |
| noise43 | 25 | fixed 0.3 | 43 | fresh noise/times, same two flows |
| noise44 | 25 | fixed 0.3 | 44 | fresh noise/times, same two flows |
| longer-fixed03 | 25 | fixed 0.3 | 42 | [14,17], 450 samples/Re, 600 quadrature points |
| longer-broad | 25 | learned, 1 / 3 | 42 | same longer window |

Original cases forecast (17,19]; the longer cases forecast exactly the same absolute window, with a basis fitted
only to their longer training data and its own projection floor. Re 200 is never used in either fit or basis,
but is an already-inspected development target, not a fresh confirmation problem. Seeds 43/44 assess acquisition
robustness only, not a different flow. Long-horizon check: 50 draws and the mean to 10 training-window lengths,
boundedness only where FOM truth is unavailable.

Keep the original two-times-largest-training-coefficient cap, use RK45 with tighter tolerances, and report
solver-budget exhaustion as a failure. Integrate all 500 new draws (the original shared pipeline integrated
at most 200, although it saved 500). Report 90% interval coverage and width conditional on survival alongside
the survival rate; do not present survivor-only bands as calibrated unconditional UQ. No posterior rejection,
draw shrinkage, trajectory clipping, or future-truth calibration is used.

Expectations: fixed 0.3 improves training-Re survival and accuracy; fixed 0.1 trades accuracy for survival;
the tight hierarchy may move away from the useful shrinkage; the broad control exposes large null-direction
prior variance; longer records may help less than shrinkage; held-out Re 200 may remain poor. These are hypotheses,
not required outcomes. If losses are nonfinite, stop and report failure; per-fit cap 18 h. Any change or extra
experiment after these results must be disclosed.

**Amendment before reading any stage-2 forecast:** add `longer-fixed09`, identical to `longer-fixed03` except
SD multiplier 0.9. The nondimensional prior contains 1/T: merely extending the window from 1 to 3 also narrows
its dimensional width threefold. This extra control compensates for that change (apart from the basis/state
scale changing with the data), partly separating more observations from automatic extra shrinkage. The longer
fits also integrate from t=14 rather than t=16, so they accumulate more rollout error; this is not an isolated
"more data only" comparison. It runs after the
two short-window fits on fix1; all other rules stay unchanged. A screen-only status-label correction now names
an original-state-cap failure explicitly rather than leaving its successful numerical-solver message; bounded
flags and every reported score are unchanged.

**Feedback and Jacobian ablations, declared before any stage-2 forecast is read.** Two further r25 runs,
`modular42` and `modular43`, use fixed SD multiplier 0.3 and the original data windows/seeds 42/43. Fit the
GP hyperparameter posterior to observations only (same priors, AutoNormal, 6000 updates), then sample the
unchanged closed-form conditional operator posterior given each GP draw. This removes feedback from the
operator evidence into the smoother while retaining GP uncertainty. It is modular Bayesian inference, not
the paper's joint posterior, and will be labelled separately even if it helps.

The first local-Jacobian screens used the row-scaled shifted prior, whose own mean fit was poor. To avoid
attributing that baseline's bias to the stability idea, repeat the same 16 damping/SD combinations on the
unshifted global prior (multiplier 1), still with fixed GP hypers and 32 draws. No new parameter grid is
selected from forecast truth. All 78 conditional screens and 13 full/modular fits will be reported.

**Initialization ablation (before reading a joint forecast).** Add `warm42` / `warm43`, otherwise identical to
fixed03 / noise43, initializing GP variational locations at the training-only GP MAP already computed for
tempering. No parameters or truths from a good forecast are supplied. This changes optimization only, not the
joint posterior or priors, and tests whether tight priors and prior-median initialization lead to a bad local
solution. Same 6000 updates and 500 draws. The total is now 15 full/modular fits; no conditional success will be
presented as success of the original joint method without this distinction.

### Full local-contraction prior (declared after conditional screens, before these refits)

The matched-baseline screen materially changes the assessment of this prior: damping 20, SD 3 yielded
PM errors 2.02 / 1.84 / 19.35% at Re 150 / 300 / 200, with 32/32 bounded draws at all three Re.
Damping 5, SD 1 yielded 1.95 / 2.49 / 84.59%, with 32/32, 32/32, 31/32 bounded. The earlier conclusion that
the tested contraction construction was mostly biased was confounded by its row-scaled shifted base prior.
These remain conditional, already-inspected development results.

Four full joint fits: `jac5` (damping 5, SD 1, r25, seed42), `jac20` (20, 3, r25, seed42),
`jac20-noise43` (20, 3, r25, seed43), and `jac20-r29` (20, 3, r29, seed42).
Original windows, global unshifted base prior of multiplier 1, 6000 updates, 500 draws, and the same forecast
scoring. Initialize at the data-only GP MAP, as in the declared initialization ablation.

Construct the Gaussian prior from the training-only data-GP MAP (not from the old joint GP fit used in the
conditional screen). At 12 interior anchors per Re, let D be the diagonal within-trajectory fluctuation scales,
v the normalized tangent D^-1 qdot, and P=I-vv^T. Favour D^-1 J(q,u) D P = -damping P, dividing each residual
by SD times the square root of the anchor count so adding anchors does not automatically strengthen the prior.
The residual is linear in each operator row.
Multiplying this factor by the original Gaussian prior gives a proper full-covariance Gaussian per row;
retain its nonzero mean and covariance, including its normalization, in both the marginal evidence and
conditional posterior. `build_model(prior_root=..., prior_mean=...)` implements this by exact whitening,
not by projecting posterior draws after fitting. Tests compare its evidence and moments to Gaussian algebra.

This is an empirical, training-geometry-informed local contraction prior. It is not a global trapping
certificate, does not impose negative eigenvalues on the tangent dynamics, and does not guarantee boundedness
away from the training curves. The held-out Re has already influenced development, so fresh acquisition seeds
are only a robustness check. The runner's seed also changes inference and predictive Monte Carlo randomness,
not only observation noise/timing; no new physical flow is generated. Total declared: 78 conditional screens
and 19 joint/modular fits.

Run-record correction: the shell launch wrapper was initially deployed in place while the four original
fix2/fix3 jobs were running. This prevented completion markers from being written after their Python commands,
delaying the dependent queues. Recover markers from GNU time's recorded command exit status; for successful
commands require the final three-target summary and CRC-valid posterior/plot archives. Keep recovery provenance
beside each job's logs, using the final timing file's modification time for the reconstructed finish timestamp.
Future wrapper deployments use a new file followed by atomic rename. No fit is rerun or resampled by this repair.
