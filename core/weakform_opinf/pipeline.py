"""Shared inference → operator sampling → prediction → metrics orchestration.

This is the one ``run_experiment`` body for every case study. Given a
:class:`PreparedRun` (from an experiment adapter) and a :class:`WeakFormConfig`,
it builds the model, runs SVI or NUTS over the GP hyperparameters, draws
operators from their closed-form conditional posterior, integrates the ROM for
each evaluation target, and returns/saves metrics.
"""

from __future__ import annotations

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from numpyro.optim import ClippedAdam
from scipy.interpolate import interp1d

from .model import build_model
from core.bayesian_opinf import generate_rom_predictions


def _stack_theta(post, num_traj, num_modes):
    """Stack posterior hyper-samples into (S, num_traj, num_modes) arrays."""
    def g(kind):
        cols = []
        for ic in range(num_traj):
            cols.append(jnp.stack(
                [post[f"{kind}_{ic}_{i}"] for i in range(num_modes)], axis=-1))
        return jnp.stack(cols, axis=1)   # (S, num_traj, num_modes)
    return g("lengthscale"), g("variance"), g("noise")


def _ic_sigma(t_sampled, snapshots_comp, ells_m, sig2s_m, nus_m, num_modes):
    """Posterior GP std of the state at t0 per mode (for IC uncertainty)."""
    t_tr = np.asarray(t_sampled)
    n_tr = len(t_tr)
    sq_tt = (t_tr[:, None] - t_tr[None, :]) ** 2
    sq_0t = (t_tr[0] - t_tr) ** 2
    sig_ic = np.zeros(num_modes)
    for i in range(num_modes):
        ell2 = ells_m[i] ** 2
        K = (sig2s_m[i] * np.exp(-sq_tt / (2 * ell2))
             + (nus_m[i] + max(1e-5, sig2s_m[i] * 1e-4)) * np.eye(n_tr))
        k0 = sig2s_m[i] * np.exp(-sq_0t / (2 * ell2))
        var = sig2s_m[i] - k0 @ np.linalg.solve(K, k0)
        sig_ic[i] = np.sqrt(max(float(var), 0.0))
    return sig_ic


def run_experiment(spec, cfg, schema, script_dir, save=True, verbose=True):
    """Run one data regime end-to-end. Returns a result dict."""
    numpyro.set_platform("cpu")
    rng_key = random.PRNGKey(cfg.seed)
    np.random.seed(cfg.seed)

    prepared = spec.prepare(cfg, schema)
    rom = prepared.rom
    num_modes = cfg.num_modes
    num_traj = len(prepared.trajectories)

    if verbose:
        print(f"\n{'=' * 78}\n  {schema.get('label', schema['name'])}"
              f"  ({num_traj} traj, {num_modes} modes, {cfg.operators})\n{'=' * 78}")

    model, posterior_O_fn, time_evals, prior_info = build_model(
        rom, prepared.trajectories, cfg)

    # ── Inference over GP hyperparameters ────────────────────────────────
    model_kwargs = dict(gamma2=cfg.gamma2)
    rng_key, ik = random.split(rng_key)
    t0 = time.time()
    if cfg.infer == "nuts":
        kernel = NUTS(model, init_strategy=init_to_median, target_accept_prob=0.9)
        mcmc = MCMC(kernel, num_warmup=cfg.nuts_warmup,
                    num_samples=cfg.nuts_samples, num_chains=1,
                    progress_bar=False)
        mcmc.run(ik, **model_kwargs)
        post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()
                if not k.startswith("X_")}
        npost = cfg.nuts_samples
        losses = np.array([0.0])
    else:
        guide = autoguide.AutoNormal(model, init_loc_fn=init_to_median)
        svi = SVI(model, guide, ClippedAdam(step_size=cfg.learning_rate),
                  loss=Trace_ELBO())
        state = svi.init(ik, **model_kwargs)

        @jax.jit
        def _step(s, _):
            return svi.update(s, **model_kwargs)

        state, losses = jax.lax.scan(_step, state, jnp.arange(cfg.num_steps))
        losses = np.array(losses)
        params = svi.get_params(state)
        rng_key, sk = random.split(rng_key)
        npost = cfg.num_posterior_samples
        post = guide.sample_posterior(sk, params, sample_shape=(npost,),
                                      **model_kwargs)
    if verbose:
        print(f"  {cfg.infer.upper()}: loss {float(losses[0]):.2f} → "
              f"{float(losses[-1]):.2f}  ({time.time() - t0:.1f}s)")

    # ── Sample O from its closed-form conditional per θ-sample ───────────
    ells_s, sig2s_s, nus_s = _stack_theta(post, num_traj, num_modes)
    tau_block_s = (jnp.exp(jnp.asarray(post["log_tau_block"]))
                   if "log_tau_block" in post else None)
    sigma_O_j = jnp.asarray(cfg.sigma_O)

    rng_key, ok = random.split(rng_key)
    keys = jax.random.split(ok, npost)

    @jax.jit
    def _draw_O(theta_s, key, tau_s):
        mu_O, C_O = posterior_O_fn(theta_s, cfg.gamma2, sigma_O_j, tau_s)
        eps = jax.random.normal(key, shape=mu_O.shape)
        return mu_O + jnp.einsum("ijk,ik->ij", C_O, eps)

    O_samples = []
    for s in range(npost):
        theta_s = (ells_s[s], sig2s_s[s], nus_s[s])
        tau_s = None if tau_block_s is None else tau_block_s[s]
        O_samples.append(np.asarray(_draw_O(theta_s, keys[s], tau_s)))
    O_samples = np.stack(O_samples)
    op_norms = np.linalg.norm(O_samples.reshape(npost, -1), axis=1)
    if verbose:
        print(f"  ‖O‖ median={np.median(op_norms):.1f}")

    # Mean hypers (first trajectory) for IC uncertainty.
    ell_m = np.asarray(ells_s[:, 0]).mean(0)
    sig2_m = np.asarray(sig2s_s[:, 0]).mean(0)
    nu_m = np.asarray(nus_s[:, 0]).mean(0)

    # ── Predict + score each evaluation target ───────────────────────────
    per_target = []
    for tgt in prepared.eval_targets:
        state0_samples = None
        if cfg.ic_uncertainty:
            sig_ic = _ic_sigma(prepared.t_sampled, prepared.snapshots_comp,
                               ell_m, sig2_m, nu_m, num_modes)
            rng_ic = np.random.default_rng(cfg.seed)
            eps_ic = rng_ic.standard_normal((npost, num_modes))
            state0_samples = (tgt.state0_comp[None, :]
                              + cfg.ic_scale * sig_ic[None, :] * eps_ic)

        samples_for_rom = {"O": jnp.array(O_samples)}
        Os, _, rom_solves = generate_rom_predictions(
            samples=samples_for_rom, rom=rom,
            snapshots_compressed=prepared.snapshots_comp,
            time_eval=tgt.t_pred, num_modes=num_modes,
            num_pulls=min(200, npost), input_func=tgt.input_func,
            state0_samples=state0_samples)
        per_target.append(_score(tgt, rom_solves, Os, prepared.training_span))

    # Aggregate (single-target experiments: just target 0).
    agg = _aggregate(per_target)
    runtime = time.time() - t0
    if verbose:
        _print_results(agg, runtime)

    result = dict(
        schema=schema, cfg=cfg, losses=losses, O_samples=O_samples,
        op_norm_median=float(np.median(op_norms)),
        runtime=runtime, num_modes=num_modes,
        basis=prepared.basis, training_span=prepared.training_span,
        eval_targets=prepared.eval_targets, per_target=per_target,
        extra=prepared.extra, **agg,
    )

    if save:
        _save_npz(result, spec, schema, script_dir)
    return result


def _score(tgt, rom_solves, Os, span):
    n_total = len(Os)
    n_stable = len(rom_solves)
    stability_pct = n_stable / max(n_total, 1) * 100
    out = dict(rom_solves=rom_solves, n_stable=n_stable, n_total=n_total,
               stability_pct=stability_pct, train_error=float("inf"),
               pred_error=float("inf"), ci_coverage=float("nan"),
               ci_width=float("nan"), t_pred=tgt.t_pred)
    if n_stable > 0:
        rom_arr = np.array(rom_solves)
        rom_med = np.median(rom_arr, axis=0)
        ti = interp1d(tgt.t_full, tgt.true_comp, kind="cubic",
                      fill_value="extrapolate")
        ta = ti(tgt.t_pred)
        tm = tgt.t_pred <= span[1]
        pm = tgt.t_pred > span[1]
        out["train_error"] = float(np.linalg.norm(rom_med[:, tm] - ta[:, tm])
                                   / np.linalg.norm(ta[:, tm]))
        out["pred_error"] = float(np.linalg.norm(rom_med[:, pm] - ta[:, pm])
                                  / np.linalg.norm(ta[:, pm]))
        q05 = np.percentile(rom_arr, 5, axis=0)
        q95 = np.percentile(rom_arr, 95, axis=0)
        out["ci_width"] = float(np.mean(q95 - q05))
        out["ci_coverage"] = float(np.mean((ta >= q05) & (ta <= q95)))
    return out


def _aggregate(per_target):
    finite = [t for t in per_target if np.isfinite(t["train_error"])]
    def _mean(key):
        vals = [t[key] for t in finite]
        return float(np.mean(vals)) if vals else float("nan")
    return dict(
        stability_pct=float(np.mean([t["stability_pct"] for t in per_target])),
        train_error=_mean("train_error"),
        pred_error=_mean("pred_error"),
        ci_coverage=_mean("ci_coverage"),
        ci_width=_mean("ci_width"),
    )


def _print_results(agg, runtime):
    print(f"\n  RESULTS — Marginalised-O + Weak-Form")
    print(f"    Stability:   {agg['stability_pct']:.1f}%")
    print(f"    Train error: {agg['train_error']:.2%}")
    print(f"    Pred error:  {agg['pred_error']:.2%}")
    print(f"    CI coverage: {agg['ci_coverage']:.1%} (target 90%)")
    print(f"    CI width:    {agg['ci_width']:.4f}")
    print(f"    Runtime:     {runtime:.0f}s")


def _save_npz(result, spec, schema, script_dir):
    out_dir = os.path.join(script_dir, "results", "comparison", schema["name"])
    os.makedirs(out_dir, exist_ok=True)
    per_target = result["per_target"]
    tgt0 = per_target[0]
    rom_arr = (np.array(tgt0["rom_solves"]) if tgt0["n_stable"] > 0
               else np.empty((0, result["num_modes"], len(tgt0["t_pred"]))))
    suffix = os.environ.get("OUTPUT_SUFFIX", "")
    fname = f"{spec.name}{suffix}.npz"
    # Multi-IC keys (n_ics + rom_solves_i) so multi-trajectory comparison
    # loaders (heat 06) can read each evaluated IC; flat rom_solves (primary)
    # is kept for single-IC loaders.
    multi = {"n_ics": len(per_target)}
    for i, pt in enumerate(per_target):
        multi[f"rom_solves_{i}"] = (np.array(pt["rom_solves"]) if pt["n_stable"] > 0
                                    else np.empty((0, result["num_modes"],
                                                   len(pt["t_pred"]))))
    np.savez(
        os.path.join(out_dir, fname),
        rom_solves=rom_arr, t_pred=tgt0["t_pred"],
        train_error=result["train_error"], pred_error=result["pred_error"],
        stability_pct=result["stability_pct"], ci_coverage=result["ci_coverage"],
        ci_width=result["ci_width"], runtime=result["runtime"],
        op_norm_median=result["op_norm_median"], losses=result["losses"],
        num_modes=result["num_modes"],
        training_span=np.array(result["training_span"]),
        O_samples=result["O_samples"],
        basis_entries=np.asarray(result["basis"].entries),
        **multi,
    )
