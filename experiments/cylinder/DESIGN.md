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
