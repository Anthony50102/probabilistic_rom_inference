"""Typed configuration for the marginalised-O × weak-form Bayesian OpInf method.

All per-experiment behaviour is expressed as fields on :class:`WeakFormConfig`.
There are deliberately **no environment-variable toggles and no MLE anywhere**:
GP-hyperparameter priors are spectrum-anchored (derived from the observation
window and the POD singular-value spectrum), the operator prior is
nondimensionalised by the training window and the state/input scales, the
pseudo-likelihood of the physics rows is tempered by the GP's effective degrees
of freedom, the GP nugget sits at floating-point round-off, and SVI/NUTS
explores the hyperparameters. This makes every case study run the identical
algorithm, with no per-experiment constant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeakFormConfig:
    """Configuration for :func:`core.weakform_opinf.model.build_model`.

    The defaults are the production algorithm shared by every reported
    experiment; experiment scripts set only the model structure (operators,
    num_modes) and their data-specific options.
    """

    # ── Reduced-model structure ──────────────────────────────────────────
    operators: str = "cAH"
    """opinf operator string, e.g. 'cA', 'cAH', 'cABN', 'cAHBN'."""
    num_modes: int = 6
    """Number of POD modes (fixed; no MLE/SNR selection)."""
    num_eval_points: int = 200
    """GP densification / weak-form quadrature grid size."""

    # ── GP-hyperparameter priors (spectrum-anchored, never MLE) ──────────
    ell_prior_mode: str = "principled"
    """'principled' → LogNormal(log Δt, log(T/Δt)/z_0.99); 'legacy' → T/20."""
    sig2_prior_scale: float = 1.0
    """LogNormal scale for the per-mode variance prior."""
    nu_prior_scale: float = 1.0
    """LogNormal scale for the per-mode noise prior."""
    gp_jitter_rel: float | None = None
    """GP training-kernel nugget. None (default, every reported experiment):
    the round-off level n·ε·max diag K of the n×n training kernel (ε = machine
    epsilon of the working dtype); scale invariant, no floor, no smoothing.
    A float selects the historical relative nugget max(1e-5, σ²·gp_jitter_rel)."""
    gp_input_trend: bool = False
    """Marginalize a Gaussian trend in time and integrated scalar input."""
    gp_noise_prior: str = "spectrum"
    """'measurement' uses supplied projected measurement variances for noise priors."""

    # ── Weak-form test functions ─────────────────────────────────────────
    window_size: int = 20
    bump_p: int = 6
    num_test_funcs: int | None = None
    bump_radius_frac: float | None = None
    weakform_mode: str = "ibp"
    """'ibp' (WSINDy integration-by-parts, state-based) or 'deriv'."""

    # ── Constraint covariance models ─────────────────────────────────────
    deriv_cov: str = "diag"
    """'diag' (marginal derivative variance) or 'full' (dense Σ_z)."""
    weakform_cov: str = "diag"
    """'diag' or 'full' (dense K×K weak-form covariance)."""
    deriv_weight: float = 1.0
    weakform_weight: float = 1.0

    # ── Operator prior ───────────────────────────────────────────────────
    op_prior_mode: str = "block_hier"
    """'block_hier' (per-block ARD scales, learned) or 'fixed' (scales held at
    their prior centre)."""
    sigma_O: float | None = None
    """None (default, every reported experiment): nondimensional operator prior
    O_ij ~ N(0, (κ_b s_j)²), s_j = S^(1-p_j) U^(-q_j) / T, with T the training
    window, S the RMS training POD coefficient, U the RMS input and p_j, q_j the
    state/input degree of column j; log κ_b ~ N(0, hier_tau_scale²). A float
    selects the historical dimensional prior log τ_b ~ N(log σ_O, hier_tau_scale²)."""
    hier_tau_scale: float = 3.0
    operator_solver: str = "qr"
    """'qr' (default) factors the whitened likelihood and the declared Gaussian
    prior as one augmented least-squares system: exact and unit invariant.
    'normal' retains the historical normal equations with the trace-scaled
    ridge 1e-6 max(tr(M)/m, 1) I, which is not unit invariant and can move
    weakly identified coefficients substantially."""

    # ── Pseudo-likelihood calibration + GP marginal-likelihood weight ────
    closure: str = "tempered"
    """How the derivative and weak-form rows, which are correlated functionals
    of one GP fit, are weighted. 'tempered' (default, every reported experiment):
    each mode's rows enter as a power likelihood with exponent
    α_i = min(1, df_i / (n_e + K)), df_i = tr K(K + νI)⁻¹ the effective degrees
    of freedom of the mode's GP smoother at a data-only GP fit (n_e derivative
    rows, K weak rows), and no closure variance (γ² = 0); no constant to set.
    Needs the diagonal derivative and weak blocks of the IBP weak form.
    'slack': untempered rows with the closure-error variance γ² below (the
    earlier rule; its constant c_γ had to be selected per experiment)."""
    gamma2: float | None = None
    """closure='slack' only. Closure-error (derivative-slack) variance γ². None:
    γ² = gamma2_nd (S/T)², with S/T the reference rate of the nondimensional
    operator prior (S RMS training POD coefficient, T training window), so the
    slack is unit invariant. A float fixes γ² in the data's units (historical;
    keeps its 1e-4 absolute floor on the derivative variance)."""
    gamma2_nd: float = 0.1
    """closure='slack' only. Closure constant c_γ: closure-error variance in
    units of (S/T)², used when gamma2 is None. No single value suited every
    benchmark, which is why the tempered rule replaced it."""
    weak_slack: str = "grid"
    """closure='slack' only. Weak-form closure variance. 'grid' (default): the derivative block's
    closure error (independent, variance γ², at every grid point) carried through
    the weak-form quadrature, γ² Σ_j (w_j ψ_k(t_j))²; 'support': one closure error
    shared over each test-function support, γ² (∫ψ_k)²; 'legacy': γ² ∫ψ_k², an
    implicit correlation time of one time unit (historical)."""
    mll_weight: float = 1.0

    # ── Inference ────────────────────────────────────────────────────────
    infer: str = "svi"
    """'svi' (AutoNormal) or 'nuts'."""
    num_steps: int = 12000
    learning_rate: float = 3e-3
    num_posterior_samples: int = 500
    nuts_warmup: int = 500
    nuts_samples: int = 500
    precision: str = "default"
    """'default' preserves the caller's JAX setting; or select float32/float64."""

    # ── Least-squares ROM prior fit (structure only; values marginalised) ─
    regularizer: float = 1.0

    # ── Prediction ───────────────────────────────────────────────────────
    num_pred_points: int = 400
    ic_uncertainty: bool = False
    ic_scale: float = 1.0

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42

    def __post_init__(self):
        _one_of("ell_prior_mode", self.ell_prior_mode, {"principled", "legacy"})
        _one_of("weakform_mode", self.weakform_mode, {"ibp", "deriv"})
        _one_of("deriv_cov", self.deriv_cov, {"diag", "full"})
        _one_of("weakform_cov", self.weakform_cov, {"diag", "full"})
        _one_of("op_prior_mode", self.op_prior_mode, {"block_hier", "fixed"})
        _one_of("operator_solver", self.operator_solver, {"normal", "qr"})
        _one_of("gp_noise_prior", self.gp_noise_prior, {"spectrum", "measurement"})
        _one_of("precision", self.precision, {"default", "float32", "float64"})
        _one_of("infer", self.infer, {"svi", "nuts"})
        _one_of("weak_slack", self.weak_slack, {"grid", "support", "legacy"})
        _one_of("closure", self.closure, {"tempered", "slack"})
        if self.closure == "tempered":
            if self.gamma2 is not None:
                raise ValueError("WeakFormConfig.gamma2 fixes a closure slack: set closure='slack'")
            if (self.deriv_cov, self.weakform_cov, self.weakform_mode) != ("diag", "diag", "ibp"):
                raise ValueError("closure='tempered' needs deriv_cov='diag', weakform_cov='diag' and "
                                 "weakform_mode='ibp'; use closure='slack' otherwise")
        for name in ("sigma_O", "gp_jitter_rel", "gamma2"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"WeakFormConfig.{name}={value!r} must be None or positive")
        if not self.gamma2_nd > 0:
            raise ValueError(f"WeakFormConfig.gamma2_nd={self.gamma2_nd!r} must be positive")


def _one_of(name, value, allowed):
    if value not in allowed:
        raise ValueError(
            f"WeakFormConfig.{name}={value!r} must be one of {sorted(allowed)}")
