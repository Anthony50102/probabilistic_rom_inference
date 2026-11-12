"""Joint Bayesian confirmation of the declared stabilization screen.

Uses the shared model/guide, checkpoints SVI before scoring, exports conditional
means and hyperparameters, and counts capped or budget-limited integrations as
failures. See DESIGN.md for the fixed experimental choices.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_median, init_to_value
from numpyro.optim import ClippedAdam

from core.weakform_opinf import WeakFormConfig
from core.weakform_opinf import gp
from core.weakform_opinf.model import _prior_locs, _sample_gp_hypers, build_model
from core.weakform_opinf.pipeline import _ic_sigma, _stack_theta
from after_the_fact.stabilization import integrate

module_spec = importlib.util.spec_from_file_location("multi_re", HERE / "05_multi_re.py")
multi_re = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(multi_re)


def atomic_pickle(value, path):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(jax.device_get(value), stream)
    temporary.replace(path)


def inference(prepared, cfg, out, gp_only=False, init_data_gp=False):
    model, posterior, grids, priors = build_model(prepared.rom, prepared.trajectories, cfg)
    if gp_only:
        locs = [_prior_locs(tr, cfg) for tr in prepared.trajectories]
        batches = [gp.trajectory_gp_conditional(tr, cfg)(grid)[1]
                   for tr, grid in zip(prepared.trajectories, grids)]

        def model(gamma2=None):
            total = 0.
            for th, batch, tr in zip(_sample_gp_hypers(locs, cfg), batches, prepared.trajectories):
                total += jnp.sum(batch(*th, jnp.asarray(tr["snapshots_comp"]))[4])
            numpyro.factor("gp_mll", total)

        priors["inference"] = "modular: GP posterior uses observations only; O conditional remains Gaussian"
    (out / "priors.json").write_text(json.dumps(priors, indent=2))
    gamma = priors["gamma2"]
    init = init_to_median
    if init_data_gp:
        fitted = priors["closure"]["data_only_gp"]
        values = {f"{site}_{k}_{i}": jnp.asarray(fitted[key][k][i])
                  for site, key in (("lengthscale", "ell"), ("variance", "sig2"), ("noise", "nu"))
                  for k in range(len(prepared.trajectories)) for i in range(cfg.num_modes)}
        init = init_to_value(values=values)
    guide = autoguide.AutoNormal(model, init_loc_fn=init)
    svi = SVI(model, guide, ClippedAdam(step_size=cfg.learning_rate), Trace_ELBO())
    key, ik = jax.random.split(jax.random.PRNGKey(cfg.seed))
    state = svi.init(ik, gamma2=gamma)
    checkpoint = out / "checkpoint.pkl"
    losses, completed = [], 0
    if checkpoint.exists():
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        state, completed, losses = saved["state"], saved["completed"], saved["losses"]
        print(f"resuming {completed} steps", flush=True)

    @jax.jit
    def block(state):
        return jax.lax.scan(lambda s, _: svi.update(s, gamma2=gamma), state, jnp.arange(100))

    while completed < cfg.num_steps:
        started = time.time()
        state, values = block(state)
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite SVI loss after step {completed}")
        losses.extend(values.tolist())
        completed += len(values)
        atomic_pickle(dict(state=state, completed=completed, losses=losses), checkpoint)
        print(f"step {completed} mean_loss {values.mean():.6g} seconds {time.time() - started:.2f}", flush=True)
    key, sk = jax.random.split(key)
    post = guide.sample_posterior(sk, svi.get_params(state), sample_shape=(cfg.num_posterior_samples,), gamma2=gamma)
    hypers = _stack_theta(post, len(prepared.trajectories), cfg.num_modes)
    taus = jnp.exp(post["log_tau_block"]) if "log_tau_block" in post else None
    key, ok = jax.random.split(key)
    keys = jax.random.split(ok, cfg.num_posterior_samples)

    @jax.jit
    def draw(theta, key, tau):
        mean, root = posterior(theta, gamma, None, tau)
        return mean + jnp.einsum("ijk,ik->ij", root, jax.random.normal(key, mean.shape)), mean

    samples, means = [], []
    for j in range(cfg.num_posterior_samples):
        sample, mean = draw(tuple(h[j] for h in hypers), keys[j], None if taus is None else taus[j])
        samples.append(np.asarray(sample))
        means.append(np.asarray(mean))
    arrays = dict(O_samples=np.array(samples), O_conditional_means=np.array(means), losses=np.array(losses),
                  **{name: np.asarray(value) for name, value in zip(
                      ("gp_lengthscale_samples", "gp_variance_samples", "gp_noise_samples"), hypers)})
    if taus is not None:
        arrays["op_log_tau_samples"] = np.asarray(post["log_tau_block"])
    arrays["O_mean_rao_blackwell"] = arrays["O_conditional_means"].mean(0)
    arrays["basis_entries"] = np.asarray(prepared.basis.entries)
    for k, tgt in enumerate(prepared.eval_targets):
        arrays.update({f"{name}_{k}": value for name, value in
                       (("t", tgt.t_pred), ("true_comp", tgt.true_comp),
                        ("t_samp", tgt.t_sampled), ("snaps_comp", tgt.snapshots_comp))})
    np.savez_compressed(out / "posterior.npz", **arrays)
    return arrays


def score(prepared, cfg, arrays, out):
    V = np.asarray(prepared.basis.entries)
    samples = arrays["O_samples"]
    means = dict(rao_blackwell=arrays["O_mean_rao_blackwell"], sample_mean=samples.mean(0))
    hypers = tuple(arrays[key].mean(0) for key in
                   ("gp_lengthscale_samples", "gp_variance_samples", "gp_noise_samples"))
    rng = np.random.default_rng(cfg.seed)
    result = dict(config=asdict(cfg), targets=[])
    plots = {}
    for k, tgt in enumerate(prepared.eval_targets):
        u = multi_re.RE_REF / prepared.extra["runs"][k]["re"]
        t, truth, X = tgt.t_pred, tgt.true_comp, tgt.true_states
        train = prepared.training_span[1]
        fc = t > (train if tgt.training_index is not None else 0)
        denom = np.sum((X[:, fc] - X[:, fc].mean(1, keepdims=True)) ** 2)
        residual = np.sum((V @ truth[:, fc] - X[:, fc]) ** 2)
        cap = 2 * np.max(np.abs(tgt.snapshots_comp))

        def error(sol):
            return float(np.sqrt((np.sum((sol[:, fc] - truth[:, fc]) ** 2) + residual) / denom))

        h = tuple(z.mean(0) if tgt.training_index is None else z[tgt.training_index] for z in hypers)
        sig = _ic_sigma(tgt.t_sampled, *h, cfg.num_modes, roundoff_nugget=True)
        q0s = tgt.state0_comp + sig[None] * rng.standard_normal((len(samples), cfg.num_modes))
        q0 = q0s.mean(0)
        row = dict(re=prepared.extra["runs"][k]["re"], held_out=tgt.training_index is None,
                   floor=float(np.sqrt(residual / denom)), means={})
        for label, mean in means.items():
            stat, pred = integrate(mean, q0, u, t, cap)
            if stat["bounded"]:
                stat["error"] = error(pred)
                plots[f"{label}_{k}"] = pred
            row["means"][label] = stat
        sols, failures = [], {}
        for j, (O, initial) in enumerate(zip(samples, q0s)):
            stat, pred = integrate(O, initial, u, t, cap)
            if stat["bounded"]:
                sols.append(pred)
            else:
                reason = stat["reason"]
                failures[reason] = failures.get(reason, 0) + 1
            if (j + 1) % 100 == 0:
                print(f"Re {row['re']} draws {j+1}/{len(samples)} bounded {len(sols)}", flush=True)
        row.update(n_draws=len(samples), n_bounded=len(sols), survival=len(sols) / len(samples), failures=failures)
        if sols:
            sols = np.array(sols)
            lo, median, hi = np.percentile(sols, (5, 50, 95), axis=0)
            row["median_error"] = error(median)
            row["coverage_survivors"] = float(np.mean((truth[:, fc] >= lo[:, fc]) & (truth[:, fc] <= hi[:, fc])))
            row["width_over_fluct"] = float(np.linalg.norm((hi - lo)[:, fc]) /
                                             np.linalg.norm(truth[:, fc] - truth[:, fc].mean(1, keepdims=True)))
            plots.update({f"{name}_{k}": val for name, val in (("low", lo), ("median", median), ("high", hi))})
        long_t = np.linspace(0, 10 * train, 1001)
        stat, _ = integrate(means["rao_blackwell"], q0, u, long_t, cap)
        row["mean_long"] = stat
        long_status = [integrate(O, initial, u, long_t, cap)[0] for O, initial in zip(samples[:50], q0s[:50])]
        row["long_survival"] = float(np.mean([s["bounded"] for s in long_status]))
        row["long_n_draws"] = len(long_status)
        plots.update({f"{name}_{k}": val for name, val in
                      (("t", t), ("truth", truth), ("t_samp", tgt.t_sampled), ("obs", tgt.snapshots_comp))})
        result["targets"].append(row)
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)
    np.savez_compressed(out / "plotdata.npz", **plots)
    return result


def main(args):
    jax.config.update("jax_enable_x64", True)
    cfg = WeakFormConfig(operators="cAHBN", num_modes=args.modes, op_prior_mode=args.prior,
                         op_prior_scale=args.scale, hier_tau_scale=args.log_sd, seed=args.seed,
                         num_steps=args.steps, num_posterior_samples=args.draws, precision="float64",
                         num_eval_points=args.eval_points, ic_uncertainty=True)
    if cfg.num_steps < 100 or cfg.num_steps % 100:
        raise ValueError("Steps must be a positive multiple of 100")
    if args.gp_only and cfg.op_prior_mode != "fixed":
        raise ValueError("GP-only inference requires fixed operator-prior scales")
    window = (14., 3., 5.) if args.longer else (16., 1., 3.)
    spec = multi_re.MultiReSpec([150., 300.], [200.], window=window, samples=450 if args.longer else 250)
    request = dict(config=asdict(cfg), window=window, samples=spec.samples, gp_only=args.gp_only,
                   init_data_gp=args.init_data_gp)
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / "request.json"
    if request_path.exists():
        previous = json.loads(request_path.read_text())
        previous.setdefault("gp_only", False)
        previous.setdefault("init_data_gp", False)
        if previous != json.loads(json.dumps(request)):
            raise ValueError("Existing experiment has a different request")
    request_path.write_text(json.dumps(request, indent=2))
    np.random.seed(cfg.seed)
    prepared = spec.prepare(cfg, {})
    started = time.time()
    if (args.output / "posterior.npz").exists():
        with np.load(args.output / "posterior.npz") as data:
            arrays = {key: data[key] for key in data.files}
    else:
        arrays = inference(prepared, cfg, args.output, args.gp_only, args.init_data_gp)
    result = score(prepared, cfg, arrays, args.output)
    result["gp_only"] = args.gp_only
    result["init_data_gp"] = args.init_data_gp
    result["seconds_this_invocation"] = time.time() - started
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=25)
    parser.add_argument("--prior", choices=("fixed", "block_hier"), default="fixed")
    parser.add_argument("--scale", type=float, default=.3)
    parser.add_argument("--log-sd", type=float, default=3.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--eval-points", type=int, default=200)
    parser.add_argument("--longer", action="store_true")
    parser.add_argument("--gp-only", action="store_true", help="modular GP posterior without operator-evidence feedback")
    parser.add_argument("--init-data-gp", action="store_true", help="initialize joint SVI at the training-only GP fit")
    main(parser.parse_args())
