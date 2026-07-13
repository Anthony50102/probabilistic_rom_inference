"""Typed configuration for the marginalised-O × weak-form Bayesian OpInf method.

All per-experiment behaviour is expressed as fields on :class:`WeakFormConfig`.
There are deliberately **no environment-variable toggles and no MLE anywhere**:
GP-hyperparameter priors are spectrum-anchored (derived from the observation
window and the POD singular-value spectrum), and SVI/NUTS explores the
hyperparameters. This mirrors the euler/burgers experiments and makes every
case study provably run the identical algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WeakFormConfig:
    """Configuration for :func:`core.weakform_opinf.model.build_model`.

    The defaults reproduce the single-trajectory autonomous behaviour
    (euler-style). Per-experiment scripts override the relevant fields.
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
    gp_jitter_rel: float = 1e-4
    """Relative GP kernel nugget: diag jitter = max(1e-5, σ²·gp_jitter_rel).
    A numerical stabiliser; larger values add mild extra smoothing. Euler/
    burgers use 1e-4; the tumor cases use 1e-3 (their established value)."""

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
    """'block_hier' (per-block ARD scales, learned) or 'fixed' (uniform σ_O)."""
    sigma_O: float = 10.0
    hier_tau_scale: float = 3.0

    # ── Constraint slack + GP marginal-likelihood weight ─────────────────
    gamma2: float = 10.0
    mll_weight: float = 1.0

    # ── Inference ────────────────────────────────────────────────────────
    infer: str = "svi"
    """'svi' (AutoNormal) or 'nuts'."""
    num_steps: int = 8000
    learning_rate: float = 3e-3
    num_posterior_samples: int = 500
    nuts_warmup: int = 500
    nuts_samples: int = 500

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
        _one_of("infer", self.infer, {"svi", "nuts"})


def _one_of(name, value, allowed):
    if value not in allowed:
        raise ValueError(
            f"WeakFormConfig.{name}={value!r} must be one of {sorted(allowed)}")
