"""Weak-form (bump) test-function construction for the weak-form constraint.

Builds compactly-supported bump test functions ψ_k(t) = (1 - τ²)^p on windows
tiling the evaluation grid, their analytic derivatives ψ'_k, and the trapezoid
quadrature weights used to turn the continuous integrals into fast matvecs.

Extracted verbatim from the per-experiment ``build_model`` implementations so
every experiment shares one construction.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def build_test_functions(time_eval, cfg):
    """Construct quadrature-weighted bump test functions on ``time_eval``.

    Parameters
    ----------
    time_eval : np.ndarray
        Dense evaluation grid (uniform).
    cfg : WeakFormConfig
        Uses ``window_size``, ``bump_p``, ``num_test_funcs``, ``bump_radius_frac``.

    Returns
    -------
    dict with jnp arrays:
        wpsi        (K, T_eval)  trapezoid-weighted ψ_k(t)
        wpsi_dot    (K, T_eval)  trapezoid-weighted ψ'_k(t)
        int_psi_sq  (K,)         ∫ ψ_k(t)² dt   (trapezoid)
        trap_w      (T_eval,)    trapezoid weights
        n_test      int          number of test functions K
    """
    time_eval = np.asarray(time_eval)
    num_eval_points = len(time_eval)
    dt_eval = float(time_eval[1] - time_eval[0])
    T_total = float(time_eval[-1] - time_eval[0])

    num_test_funcs = cfg.num_test_funcs
    if num_test_funcs is None:
        num_test_funcs = max(1, num_eval_points // cfg.window_size)

    if cfg.bump_radius_frac is None:
        radius = cfg.window_size * dt_eval
    else:
        radius = cfg.bump_radius_frac * T_total

    centres = np.linspace(time_eval[0] + radius, time_eval[-1] - radius,
                          num_test_funcs)
    bump_p = cfg.bump_p

    psi_list, psi_dot_list, int_psi_sq_list = [], [], []
    for tc in centres:
        tau = (time_eval - tc) / radius
        in_supp = np.abs(tau) < 1.0
        psi_vals = np.where(in_supp, (1.0 - tau ** 2) ** bump_p, 0.0)
        # dψ/dt = (dψ/dτ)(dτ/dt) = -2 p τ (1-τ²)^(p-1) / r
        psi_d_vals = np.where(in_supp,
            -2.0 * bump_p * tau * (1.0 - tau ** 2) ** (bump_p - 1) / radius,
            0.0)
        psi_vals[~in_supp] = 0.0
        psi_d_vals[~in_supp] = 0.0
        psi_list.append(psi_vals.astype(np.float32))
        psi_dot_list.append(psi_d_vals.astype(np.float32))
        w = np.ones_like(time_eval) * dt_eval
        w[0] *= 0.5
        w[-1] *= 0.5
        int_psi_sq_list.append(float(np.sum(w * psi_vals ** 2)))

    psi_arr = jnp.asarray(np.stack(psi_list))
    psi_dot_arr = jnp.asarray(np.stack(psi_dot_list))
    int_psi_sq_arr = jnp.asarray(np.array(int_psi_sq_list, dtype=np.float32))

    trap_w = np.ones_like(time_eval) * dt_eval
    trap_w[0] *= 0.5
    trap_w[-1] *= 0.5
    trap_w_jnp = jnp.asarray(trap_w.astype(np.float32))

    wpsi = trap_w_jnp[None, :] * psi_arr
    wpsi_dot = trap_w_jnp[None, :] * psi_dot_arr

    return dict(
        wpsi=wpsi,
        wpsi_dot=wpsi_dot,
        int_psi_sq=int_psi_sq_arr,
        trap_w=trap_w_jnp,
        n_test=int(wpsi.shape[0]),
    )
