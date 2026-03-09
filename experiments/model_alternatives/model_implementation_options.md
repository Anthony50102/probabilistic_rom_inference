# Model Implementation Options: Joint Bayesian OpInf

## The Problem

We need to jointly infer GP states **X** and operator **O** such that:
- GPs fit the observed POD coefficient trajectories: `y ~ GP(0, K(ℓ,σ²) + νI)`
- The operator satisfies the ODE constraint: `dX/dt ≈ f(X) @ O^T`

The fundamental challenge is the **null basin**: the observation likelihood depends only on X (not O), so the model can satisfy everything with `X ≈ const, O = 0` — the GP absorbs all signal as noise, derivatives vanish, and the ODE constraint is trivially satisfied.

---

## Option 1: Two-Stage (MLE GP → Least-Squares O)

**How it works:**
1. Fit each GP independently via MLE (maximize marginal likelihood)
2. Extract posterior mean states X and derivatives dX/dt
3. Solve `O = argmin ‖dX/dt - f(X) @ O^T‖²` (linear least-squares)
4. Optionally: Bayesian linear regression on O for uncertainty

**Pros:**
- Rock solid — GP fitting is a solved problem (no null basin, convex marginal likelihood)
- Fast: GP MLE ~seconds, least-squares ~milliseconds
- Easy to debug: each stage is independently verifiable
- Already implemented and working (`02_full_bayesian_euler.ipynb`)

**Cons:**
- No feedback loop: O cannot inform the GP fit (e.g., physics-aware smoothing)
- Uncertainty doesn't propagate: GP uncertainty frozen before O inference
- Not consistent with the joint posterior `p(X, O | y)` — it's `p(O | X_MAP) × δ(X - X_MAP)`
- Less theoretically elegant for a thesis claiming "joint Bayesian inference"

**Implementation complexity:** Low — already done.

---

## Option 2: Joint SVI with KL Annealing (Current Approach)

**How it works:**
- Single model with GP marginal likelihood + β-scaled ODE constraint
- β ramps from 0→1: GP converges first, then ODE constraint activates
- AutoNormal mean-field guide optimized via SVI (Trace_ELBO + ClippedAdam)

**Pros:**
- Truly joint optimization — GP and O inform each other through shared ELBO
- Fast compared to MCMC (compiled JAX scan, ~minutes)
- Elegant: single model, single objective, principled β-annealing
- Marginal likelihood approach eliminates X_raw funnel (18 hyperparams vs 1500)

**Cons:**
- Null basin still structurally exists — β-annealing suppresses but doesn't eliminate it
- ODE constraint gradients corrupt GP fit when β activates (the current pain point)
- Sensitive to schedule tuning (delay, ramp shape, ramp speed)
- Mean-field guide can't capture GP-operator correlations
- Derivative covariance Schur complement can go non-PD → NaN

**Implementation complexity:** High — we've been debugging this for a while.

---

## Option 3: Warm-Started Joint SVI

**How it works:**
1. Fit GPs via MLE (same as two-stage step 1)
2. Compute least-squares O from GP derivatives
3. Initialize the AutoNormal guide means from these MLE/LS values
4. Run joint SVI from this warm start (with light or no annealing)

**Pros:**
- Breaks the null basin symmetry: optimizer starts in the correct basin
- Preserves joint optimization — both GP and O refine together from a good starting point
- ODE constraint gradients are constructive (not destructive) because O already approximately matches dX/dt
- Much less sensitive to β schedule — the "spike" disappears because O isn't random when β activates
- Relatively simple to implement given existing MLE GP infrastructure

**Cons:**
- Still relies on MLE GP being a good initialization (it is, for this problem)
- Some might argue this "biases" the posterior toward the MLE solution (counterpoint: the posterior *should* be near MLE for well-identified models)
- Need to implement guide parameter injection (init_to_value in NumPyro)
- If the MLE solution is in a local optimum, joint SVI may not escape it

**Implementation complexity:** Low-moderate — MLE GP already exists, need to wire up init_to_value.

---

## Option 4: EM-Style Alternating Optimization

**How it works:**
- **E-step:** Fix O, optimize GP hyperparameters via marginal likelihood (with ODE constraint as additional likelihood term conditioned on fixed O)
- **M-step:** Fix GP posterior (X, dX/dt), optimize O via penalized least-squares or SVI
- Iterate until convergence

**Pros:**
- Each sub-problem is well-conditioned: GPs fit data when O is fixed, O fits derivatives when GPs are fixed
- Natural feedback loop: O influences GP fit through ODE likelihood, GP derivatives inform O
- No null basin per step — with O fixed, GP must still fit data; with GP fixed, O has a clear regression target
- Each step is fast and numerically stable

**Cons:**
- Not fully Bayesian — gives point estimates unless you do full posterior sampling in each step
- Can get stuck in local optima (though less likely than joint SVI)
- Convergence monitoring is ad-hoc (no single loss to track)
- More engineering: two separate optimization loops, state passing between them
- Uncertainty quantification requires additional work (e.g., Laplace at convergence)

**Implementation complexity:** Moderate — two optimization loops, convergence criteria.

---

## Option 5: Sequential Monte Carlo (SMC)

**How it works:**
- Tempered likelihood: `p_t(θ) ∝ prior(θ) × likelihood(θ)^t` for t ∈ [0, 1]
- Particles evolve from prior → posterior through temperature schedule
- At each temperature: resample (prune bad particles) + MCMC moves (diversify)
- NumPyro has `SMCMC` built-in

**Pros:**
- Principled tempering that naturally handles the null basin — particles in the null mode get pruned as likelihood increases
- Truly joint — all parameters evolve together
- Sample-based posterior (not Gaussian approximation)
- No guide to design — works with the model directly
- Temperature schedule is adaptive (effective sample size criterion)

**Cons:**
- Slower than SVI: ~5-10× wall-clock time (each temperature step requires MCMC moves)
- Particle count vs dimensionality: with 18 dimensions (marginal likelihood), ~100-500 particles should suffice; with 1500 (X_raw), impractical
- Less mature in NumPyro than SVI — fewer examples, harder to debug
- Memory: storing particle populations

**Implementation complexity:** Moderate — NumPyro has SMC support, but tuning MCMC kernel + resampling is non-trivial.

---

## Option 6: Integral Form ODE Constraint (Model E)

**How it works:**
- Replace derivative constraint `dX/dt ≈ f(X) @ O^T` with integral form:
  `X(t_b) - X(t_a) ≈ ∫_{t_a}^{t_b} f(X(s)) @ O^T ds`
- State differences `ΔX = X(t_b) - X(t_a)` are anchored to observed data
- If O = 0, predicted integral = 0, but observed ΔX ≠ 0 → null basin eliminated structurally

**Pros:**
- Structurally eliminates the null basin (not just suppresses it)
- Integration averages noise (like weak SINDy) — more robust than pointwise derivative matching
- No derivative covariance matrix needed → no Schur complement PD issues
- Joint optimization works because the constraint is data-anchored

**Cons:**
- Quadrature approximation introduces discretization error
- No direct derivative uncertainty quantification
- More complex model assembly (sliding windows, quadrature weights)
- Less standard in the literature — needs careful mathematical justification for thesis
- May require finer quadrature for stiff systems

**Implementation complexity:** Moderate-high — needs quadrature implementation, window selection.

---

## Comparison Matrix

| | Null Basin | Joint Opt | Speed | UQ Quality | Complexity | Debuggability |
|---|---|---|---|---|---|---|
| **1. Two-Stage** | ✅ Eliminated | ❌ None | ⚡ Fast | 🟡 Partial | ✅ Low | ✅ Easy |
| **2. Joint SVI (current)** | ❌ Persists | ✅ Full | ⚡ Fast | ✅ Full | ❌ High | ❌ Hard |
| **3. Warm-Start SVI** | ✅ Broken | ✅ Full | ⚡ Fast | ✅ Full | 🟡 Low-Med | 🟡 Medium |
| **4. EM Alternating** | ✅ Per-step | 🟡 Iterative | 🟡 Medium | 🟡 Partial | 🟡 Medium | 🟡 Medium |
| **5. SMC** | ✅ Pruned | ✅ Full | 🔴 Slow | ✅ Full | 🟡 Medium | 🔴 Hard |
| **6. Integral Form** | ✅ Structural | ✅ Full | ⚡ Fast | 🟡 No deriv | 🟡 Med-High | 🟡 Medium |

---

## Recommendations

### For the thesis (balancing rigor + pragmatism):

**Primary recommendation: Option 3 (Warm-Start SVI)**
- Gets you joint optimization (theoretically satisfying) without the null basin pain
- Can present as: "We initialize the variational parameters from MLE/LS estimates, then refine jointly" — this is standard practice in VI literature
- If it works, you get the best of both worlds: joint posterior + stable convergence

**Fallback: Option 1 (Two-Stage)**
- If warm-start SVI still shows instability, the two-stage approach is scientifically defensible
- Frame as: "We decompose the posterior via conditional independence: p(O|X)p(X|y)"
- Already working — guaranteed results for thesis deadline

**Ambitious option: Option 6 (Integral Form)**
- Structurally the most elegant solution to the null basin
- Worth exploring if time permits — could be a genuine contribution

**For future work / paper: Option 5 (SMC)**
- Most principled but slowest — good "future directions" material
