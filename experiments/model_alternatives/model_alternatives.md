# Probabilistic Model Structures for Bayesian Operator Inference

## The Problem

We observe noisy snapshots $y(t_k) \in \mathbb{R}^r$ of a PDE solution projected onto $r$ POD modes. The goal is to infer a ROM operator $O$ such that the latent dynamics satisfy

$$\frac{dX}{dt} = f(X) \cdot O^\top$$

in a Bayesian framework, producing posterior uncertainty over $O$.

### The Null Basin Problem

In a joint model where $X \sim \mathcal{GP}$ and $O \sim \mathcal{N}(0, \gamma I)$, the observation likelihood $y \mid X \sim \mathcal{N}(X, \nu I)$ does not involve $O$. The GP can fit the data independently: it over-smooths to produce nearly flat states $X \approx \text{const}$, yielding $dX/dt \approx 0$, which is perfectly consistent with $O = 0$ under the ODE constraint. The prior $O \sim \mathcal{N}(0, \gamma I)$ actively rewards this collapse.

This is **structural**, not a tuning problem. No choice of prior variance $\gamma$, observation noise $\nu$, or ODE constraint variance $\Sigma$ eliminates the basin — it exists because $O$ is not part of the generative path from latent states to observations.

---

## Model A — GP + ODE Constraint (Current Joint Model)

$$
X \sim \mathcal{GP}(0, K(\theta))
$$
$$
y \mid X \sim \mathcal{N}(X, \nu I)
$$
$$
\frac{dX}{dt} \bigg| X, O \sim \mathcal{N}(f(X) O^\top, \Sigma)
$$
$$
O \sim \mathcal{N}(0, \gamma I)
$$

The observation path $y \leftarrow X$ bypasses $O$ entirely. The GP generates $X$, and $X$ generates $y$ — the operator $O$ only appears in a soft constraint on the derivatives. This means:

- $X$ can absorb the entire data fit without $O$ contributing.
- The configuration $X \approx \bar{y}$ (flat), $O = 0$ is a valid local minimum of the ELBO.
- The null basin is **structural**: it exists for any hyperparameter setting.
- This is analogous to **posterior collapse** in VAEs, where a powerful decoder (here, the GP) renders the latent code (here, $O$) unnecessary.

---

## Model B — State-Space Model (Dynamics Generate States)

$$
X_0 \sim \mathcal{N}(y_0, \sigma_0^2 I)
$$
$$
X_{k+1} \mid X_k, O \sim \mathcal{N}(X_k + \Delta t \cdot f(X_k) \cdot O^\top, Q)
$$
$$
y_k \mid X_k \sim \mathcal{N}(X_k, R)
$$
$$
O \sim \mathcal{N}(0, \gamma I)
$$

Here $O$ is part of the **generative process** — the dynamics produce the states, not an independent GP. Setting $O = 0$ forces $X_{k+1} = X_k + \text{noise}$, i.e., a random walk. For oscillatory or transient PDE data, this contradicts the observations, so the null basin is eliminated.

**Advantages:**
- $O = 0$ is no longer a local minimum — dynamics must match data.
- Implementation is compact (~15 lines of model code), no GP kernel machinery.
- ODE integration provides implicit smoothness (solutions of ODEs are smooth).
- Well-studied framework: state-space models, Kalman filtering, particle methods.

**Tradeoffs:**
- Loses GP kernel-based uncertainty quantification on states and derivatives.
- Forward Euler discretization introduces truncation error (higher-order integrators add complexity).
- Process noise $Q$ must be tuned to balance model trust vs. observation fit.

---

## Model C — GP with Dynamics-Informed Mean

$$
X \sim \mathcal{GP}(m_O(t), K(\theta))
$$
$$
m_O(t) = X_0 + \int_0^t f(m_O(s)) \cdot O^\top \, ds
$$
$$
y \mid X \sim \mathcal{N}(X, \nu I)
$$
$$
O \sim \mathcal{N}(0, \gamma I)
$$

The GP prior **depends on $O$** through its mean function $m_O(t)$, which is the solution of the ODE $\dot{m} = f(m) \cdot O^\top$. The GP captures uncertainty around the deterministic dynamics trajectory.

**Advantages:**
- Most theoretically elegant: the GP quantifies deviation from the ODE solution.
- No null basin — setting $O = 0$ collapses the mean to $m_O(t) = X_0$, forcing the GP kernel alone to explain all temporal variation. This is penalised by the marginal likelihood.
- Retains GP-based UQ on states and derivatives.

**Tradeoffs:**
- Requires an ODE solve inside the model at every SVI step.
- The ODE solve must be differentiable (adjoint methods or forward-mode AD).
- Computational cost is substantially higher than Models A, B, or D.
- Implementation complexity is significant.

---

## Model D — Staged Posterior Passing (Paper's Algorithm 1)

$$
\text{Stage 1:} \quad \theta_{\text{GP}} \mid y \;\longrightarrow\; q_1(\theta_{\text{GP}})
$$
$$
\text{Stage 2:} \quad X \mid y, \, q_1(\theta_{\text{GP}}) \;\longrightarrow\; q_2(X)
$$
$$
\text{Stage 3:} \quad O \mid q_2(X) \;\longrightarrow\; q_3(O)
$$

Each stage's posterior becomes the next stage's prior or fixed input. The joint inference problem — which is structurally non-identifiable (Model A) — is decomposed into a sequence of **individually well-identified** subproblems.

**Why each stage is identified:**
- **Stage 1:** Standard GP marginal likelihood optimisation — well-posed, convex in many cases.
- **Stage 2:** GP posterior conditioned on data — unique given $\theta_{\text{GP}}$.
- **Stage 3:** Linear-Gaussian regression of $O$ given fixed $X$ — closed-form or straightforward VI.

**Advantages:**
- No null basin at any stage — $O$ is inferred from a fixed, data-informed $X$.
- Staging is the **theoretical contribution**: it converts a non-identifiable joint problem into identifiable subproblems.
- Uncertainty propagates forward via variational message passing.
- Each stage can use its own optimiser, convergence criteria, and diagnostics.

**Tradeoffs:**
- Information does not flow backward: Stage 3 cannot improve the GP fit from Stage 1.
- If Stage 2 produces a poor $q_2(X)$ (e.g., oversmoothed derivatives), Stage 3 inherits the error.
- The factorisation $q(X, O, \theta) = q_1(\theta) \, q_2(X) \, q_3(O)$ is an approximation to the true joint posterior.

---

## Model E — GP + Integral Form Constraint (Weak Form)

Instead of computing derivatives (a noise-amplifying operation), use the integral form of the ODE. Integrating $\dot{X} = f(X) \cdot O^\top$ over an interval $[t_a, t_b]$ gives:

$$X(t_b) - X(t_a) = \int_{t_a}^{t_b} f(X(s)) \cdot O^\top \, ds$$

The left side is a state difference (two noisy point evaluations, noise $\sim \sqrt{2}\sigma$). The right side is a quadrature sum — integration *averages out* noise rather than amplifying it. No differentiation is performed anywhere.

The probabilistic model:

$$X \sim \mathcal{GP}(0, K(\theta))$$
$$y \mid X \sim \mathcal{N}(X, \nu I)$$
$$O \sim \mathcal{N}(0, \gamma I)$$
$$\Delta X_{ab} \mid X, O \sim \mathcal{N}\!\left(\sum_k \Delta t_k \, f(X(t_k)) \cdot O^\top,\; \sigma_{\text{int}}^2 I\right)$$

where $\Delta X_{ab} = X(t_b) - X(t_a)$ are precomputed state differences over chosen intervals.

**Why this avoids the null basin:** The state differences $\Delta X_{ab}$ are fixed from the data and non-zero. The integral constraint forces $\sum \Delta t_k f(X(t_k)) O^\top$ to match these non-zero differences. If $O = 0$, the predicted integral is zero, which contradicts the observed $\Delta X \neq 0$. Unlike derivative-based constraints, the GP cannot conspire with $O$ to make this constraint vacuous — the anchor is data-derived and immutable.

**Advantages:**
- Noise-robust: integration smooths noise; differentiation amplifies it (this is the core insight from weak SINDy, Messenger & Bortz 2021).
- No derivative estimation machinery (no kernel conditioning, no $K'$, $K''$ matrices).
- The integral anchor prevents the null basin without staging.
- Fully joint, single-pass model.
- Simple implementation: just state differences and quadrature sums.

**Tradeoffs:**
- Quadrature approximation introduces integration error (mitigated by dense time sampling).
- Choice of integration intervals affects conditioning — many short intervals vs. fewer long ones.
- Does not directly provide derivative uncertainty (though this can be recovered from the GP posterior post-hoc).

---

## Model F — GP + ODE Constraint with KL Annealing (Model A + β schedule)

Same generative structure as Model A, but with a **KL annealing schedule** that ramps the prior penalty weight β from 0 → 1 over the first ~30–40% of SVI iterations.

$$\mathcal{L}_\beta = \mathbb{E}_q[\log p(y \mid X)] + \mathbb{E}_q[\log p(\dot{X} \mid X, O)] - \beta \, \text{KL}(q \| p)$$

**Why a single β schedule is enough:**

Even with β = 0, both the GP and the operator receive learning signal. The GP is pushed to fit the data by the observation likelihood. The operator is pushed to match the GP's derivatives by the ODE constraint. They are coupled through the physics constraint — no separate annealing phase is needed for each.

A natural ordering emerges automatically: early in optimisation, the GP is uncertain — Σ_z (the derivative covariance) is large. The operator sees a very wide Gaussian for the physics constraint, giving it weak, noisy gradients. Meanwhile the GP gets strong, direct gradients from the observation likelihood. So the GP moves fast and the operator moves slow, without engineered scheduling. As the GP tightens and Σ_z shrinks, the operator starts getting sharper signal and begins converging.

**The single-shot recipe:**
- **β ≈ 0 phase**: GP chases the data hard (avoids the flat mode); operator drifts gently toward matching whatever derivatives the GP produces.
- **β ramp-up**: Priors kick in, regularising both GP hyperparameters and operator.
- **β = 1 phase**: True ELBO; both components refine together.

**Implementation:** Each prior sample site uses `dist.mask(False)` (so NumPyro doesn't auto-score it), then a `numpyro.factor` adds `β · log p(x)` manually. Observation and ODE constraint likelihoods remain at full strength throughout. A custom training loop calls `svi.update()` per step with the current β value.

**Advantages:**
- Addresses the null basin without changing the generative model structure.
- Single optimiser, single ELBO, single β schedule.
- The model's own uncertainty structure creates a built-in curriculum.
- Retains all GP-based UQ machinery.

**Tradeoffs:**
- Requires tuning the anneal fraction and schedule shape.
- During the β < 1 phase, we are not optimising the true ELBO — the final β = 1 phase must be long enough for convergence.
- The null basin is suppressed but not structurally eliminated — poor schedule choices could still lead to collapse.

---

## Comparison

| | Null Basin? | Joint Optimisation? | GP-based UQ? | Implementation Complexity | Theoretical Elegance |
|---|---|---|---|---|---|
| **Model A** — GP + ODE Constraint | Yes | Yes | Yes | Moderate | Low — structurally flawed |
| **Model B** — State-Space | No | Yes | No | Low | Moderate |
| **Model C** — GP + Dynamics Mean | No | Yes | Yes | High | High |
| **Model D** — Staged | No | No (sequential) | Yes | Moderate | High — the paper's contribution |
| **Model E** — GP + Integral Form | No | Yes | Yes | Low | High — noise-robust, no staging |
| **Model F** — GP + KL Annealing | Suppressed | Yes | Yes | Moderate | Moderate — pragmatic fix |

---

## Key Insight

In any model where $X$ has an **independent generative mechanism** (the GP with a zero or fixed mean) that can produce flat states, the null basin exists. The operator $O$ becomes structurally unnecessary because the observation likelihood $y \mid X$ does not depend on it.

There are exactly three ways to address this:

1. **Make $X$'s generative process depend on $O$** — so that $O = 0$ produces states that contradict the data (Models B and C).
2. **Fix $X$ before inferring $O$** — so that $O$ is conditioned on a data-informed $X$ rather than jointly optimised with it (Model D).
3. **Anchor constraints to immutable data-derived quantities** — so that the GP cannot conspire with $O$ to satisfy the constraint vacuously (Model E).
4. **Suppress the basin via optimisation dynamics** — KL annealing (Model F) lets the GP establish structure before the prior can pull it flat, exploiting the natural curriculum created by the model's own uncertainty structure.

Model D (staging) is the approach taken in the paper. It sacrifices joint optimality for identifiability — a principled tradeoff when the joint problem is structurally degenerate. Model E offers an alternative that preserves joint optimisation by replacing derivative constraints with integral constraints anchored to observed state differences. Model F takes a pragmatic approach: the null basin still exists structurally, but the annealing schedule prevents the optimiser from finding it.
