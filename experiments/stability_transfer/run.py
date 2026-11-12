"""Matched cross-system prior study; original experiment outputs remain untouched."""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_median
from numpyro.optim import ClippedAdam

from core.weakform_opinf import gp
from core.weakform_opinf.model import build_model, nondimensional_column_scale
from core.weakform_opinf.pipeline import _ic_sigma, _stack_theta
from core.weakform_opinf.stability import local_contraction_prior

VARIANTS = {"hierarchical": None, "fixed": None, "contract1": 1., "contract5": 5., "contract20": 20.}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_adapter(system):
    folder = ROOT / "experiments" / system
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("study_adapter", folder / "04_unified.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.HeatSpec() if system == "heat" else module.BurgersSpec()


def fingerprint(prepared):
    digest = hashlib.sha256(np.asarray(prepared.basis.entries).tobytes())
    for tr in prepared.trajectories:
        for key in ("t_sampled", "snapshots_comp", "inputs_eval"):
            if tr.get(key) is not None:
                arr = np.asarray(tr[key])
                digest.update(str((key, arr.shape, arr.dtype)).encode())
                digest.update(arr.tobytes())
    return digest.hexdigest()


def fit(prepared, cfg, variant, output):
    started = time.monotonic()
    model, posterior, grids, priors = build_model(prepared.rom, prepared.trajectories, cfg)
    damping = VARIANTS[variant]
    if damping is not None:
        fitted = priors["closure"]["data_only_gp"]
        states, rates = [], []
        for k, (tr, grid) in enumerate(zip(prepared.trajectories, grids)):
            th = tuple(jnp.asarray(fitted[key][k]) for key in ("ell", "sig2", "nu"))
            batch = jax.jit(gp.trajectory_gp_conditional(tr, cfg)(grid)[1])
            vals = batch(*th, jnp.asarray(tr["snapshots_comp"]))
            states.append(np.asarray(vals[0]))
            rates.append(np.asarray(vals[1]))
        amp = np.sqrt(np.mean([np.var(tr["snapshots_comp"], axis=1)
                              for tr in prepared.trajectories], axis=0))
        scale, base = nondimensional_column_scale(prepared.rom, prepared.trajectories, cfg.num_modes)
        mean, root, info = local_contraction_prior(
            prepared.rom, states, rates, [tr.get("inputs_eval") for tr in prepared.trajectories],
            amp, scale, base["T"], damping=damping, width=3.)
        np.savez_compressed(output / "operator_prior.npz", mean=mean, root=root,
                            states=np.stack(states), rates=np.stack(rates))
        model, posterior, grids, priors = build_model(
            prepared.rom, prepared.trajectories, cfg, prior_mean=mean, prior_root=root)
        priors["local_contraction"] = info
    write_json(output / "priors.json", priors)
    gamma = priors["gamma2"]
    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_median)
    svi = SVI(model, guide, ClippedAdam(step_size=cfg.learning_rate), Trace_ELBO())
    key, ik = jax.random.split(jax.random.PRNGKey(cfg.seed))
    state = svi.init(ik, gamma2=gamma)

    @jax.jit
    def block(state):
        return jax.lax.scan(lambda s, _: svi.update(s, gamma2=gamma), state, jnp.arange(100))

    losses = []
    for completed in range(0, cfg.num_steps, 100):
        state, values = block(state)
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite loss after step {completed}")
        losses.extend(values.tolist())
        if (completed + 100) % 1000 == 0:
            print(f"step={completed + 100} loss={values.mean():.6g}", flush=True)
    key, sk = jax.random.split(key)
    post = guide.sample_posterior(sk, svi.get_params(state),
                                  sample_shape=(cfg.num_posterior_samples,), gamma2=gamma)
    hypers = _stack_theta(post, len(prepared.trajectories), cfg.num_modes)
    taus = jnp.exp(post["log_tau_block"]) if "log_tau_block" in post else None
    key, ok = jax.random.split(key)
    keys = jax.random.split(ok, cfg.num_posterior_samples)

    @jax.jit
    def draw(theta, key, tau):
        mu, root = posterior(theta, gamma, None, tau)
        return mu + jnp.einsum("ijk,ik->ij", root, jax.random.normal(key, mu.shape)), mu

    draws, means = [], []
    for s in range(cfg.num_posterior_samples):
        O, mu = draw(tuple(h[s] for h in hypers), keys[s], None if taus is None else taus[s])
        draws.append(np.asarray(O))
        means.append(np.asarray(mu))
    arrays = dict(O_samples=np.array(draws), O_conditional_means=np.array(means),
                  O_mean_rao_blackwell=np.mean(means, axis=0), losses=np.array(losses),
                  fit_seconds=np.array(time.monotonic() - started))
    arrays.update({name: np.asarray(value) for name, value in zip(
        ("gp_lengthscale_samples", "gp_variance_samples", "gp_noise_samples"), hypers)})
    if taus is not None:
        arrays["op_log_tau_samples"] = np.asarray(post["log_tau_block"])
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Nonfinite posterior export")
    np.savez_compressed(output / "posterior.npz", **arrays)
    return arrays


class WorkBudgetExceeded(RuntimeError):
    pass


def integrate(model, operator, q0, inputs, times, cap, duration_scale):
    model._extract_operators(operator)
    calls = 0
    budget = 20000 * max(1, int(np.ceil((times[-1] - times[0]) / duration_scale)))

    def rhs(t, q):
        nonlocal calls
        calls += 1
        if calls > budget:
            raise WorkBudgetExceeded
        return model.rhs(t, q, inputs)

    def event(t, q):
        return cap - np.max(np.abs(q))

    event.terminal, event.direction = True, -1
    if np.max(np.abs(q0)) >= cap:
        return dict(complete=False, reason="initial amplitude", calls=0), None
    try:
        result = solve_ivp(rhs, (times[0], times[-1]), q0, t_eval=times,
                           method="RK45", rtol=1e-7, atol=1e-9, events=event)
    except WorkBudgetExceeded:
        return dict(complete=False, reason="work budget", calls=calls), None
    complete = bool(result.success and result.y.shape[1] == len(times) and np.isfinite(result.y).all())
    reason = "complete" if complete else ("amplitude" if len(result.t_events[0]) else result.message)
    return dict(complete=complete, reason=reason, calls=calls), result.y if complete else None


def evaluate(prepared, cfg, arrays, module, system, output, long_draws):
    operators = arrays["O_samples"]
    indices = np.linspace(0, len(operators) - 1, min(200, len(operators)), dtype=int)
    rng = np.random.default_rng(cfg.seed)
    hypers = [arrays[name].mean(0) for name in
              ("gp_lengthscale_samples", "gp_variance_samples", "gp_noise_samples")]
    duration = float(np.mean([tr["t_sampled"][-1] - tr["t_sampled"][0] for tr in prepared.trajectories]))
    cap = 100 * max(float(np.max(np.abs(tr["snapshots_comp"]))) for tr in prepared.trajectories)
    long_t = np.linspace(prepared.training_span[0], prepared.training_span[0] + 10*duration, 401)
    offset = prepared.basis.decompress(np.zeros((cfg.num_modes, 1)))
    physical_basis = prepared.basis.decompress(np.eye(cfg.num_modes)) - offset
    plots = dict(basis_entries=prepared.basis.entries, physical_basis=physical_basis, offset=offset)
    reports = []
    for k, target in enumerate(prepared.eval_targets):
        t = target.t_pred
        idx = target.training_index
        th = tuple(h.mean(0) if idx is None else h[idx] for h in hypers)
        sigma = _ic_sigma(target.t_sampled, *th, cfg.num_modes,
                          roundoff_nugget=cfg.gp_jitter_rel is None)
        q0s = target.state0_comp + cfg.ic_scale * sigma * rng.standard_normal((len(operators), cfg.num_modes))
        inputs = target.input_func
        if system == "heat":
            params = module.input_parameters[k] if idx is not None else module.test_parameters
            a, b = params
            inputs = lambda tt, a=a, b=b: np.array([a*np.sin(2*np.pi*tt), b*np.sin(4*np.pi*tt)])
            for tt in t[::25]:
                np.testing.assert_allclose(inputs(tt), target.input_func(tt), rtol=1e-12, atol=1e-12)
        truth = interp1d(target.t_full, target.true_comp, kind="cubic")(t)
        full = interp1d(target.t_full, target.true_states, kind="cubic")(t)
        score = t > prepared.training_span[1]
        norm_full = np.linalg.norm(full[:, score])
        norm_fluct = np.linalg.norm(full[:, score] - full[:, score].mean(axis=1, keepdims=True))
        norm_reduced = np.linalg.norm(truth[:, score])
        if min(norm_full, norm_fluct, norm_reduced) <= 0:
            raise ValueError("Nonpositive evaluation normalization")

        def errors(pred):
            norm_error = np.linalg.norm((physical_basis @ pred + offset - full)[:, score])
            return dict(full_relative=float(norm_error / norm_full),
                        fluct_relative=float(norm_error / norm_fluct),
                        reduced_relative=float(np.linalg.norm((pred-truth)[:, score]) / norm_reduced))

        best_physical = np.linalg.lstsq(physical_basis, full - offset, rcond=None)[0]
        row = dict(index=k, label=target.label, held_out=idx is None, cap=cap,
                   score_interval=[float(prepared.training_span[1]), float(t[-1])],
                   projected_reconstruction=errors(truth), physical_basis_floor=errors(best_physical),
                   short=[], long=[], draws=len(indices))
        mean_status, mean = integrate(prepared.rom.model, arrays["O_mean_rao_blackwell"],
                                      target.state0_comp, inputs, t, cap, duration)
        row["mean"] = dict(mean_status, errors=None if mean is None else errors(mean))
        solutions, survived = [], []
        for s in indices:
            status, pred = integrate(prepared.rom.model, operators[s], q0s[s], inputs, t, cap, duration)
            row["short"].append(dict(status, draw_index=int(s)))
            if pred is not None:
                solutions.append(pred)
                survived.append(int(s))
        row["survival"] = len(solutions) / len(indices)
        row["median_errors"] = None
        if solutions:
            solutions = np.asarray(solutions)
            median = np.median(solutions, axis=0)
            lo, hi = np.percentile(solutions, (5, 95), axis=0)
            row["median_errors"] = errors(median)
            row["coverage"] = float(np.mean((truth[:, score] >= lo[:, score]) & (truth[:, score] <= hi[:, score])))
            row["relative_band_width"] = float(np.linalg.norm((hi-lo)[:, score]) / norm_reduced)
            plots.update({f"{name}_{k}": value for name, value in
                          (("median", median), ("lo", lo), ("hi", hi), ("rom_solves", solutions))})
        else:
            row["coverage"], row["relative_band_width"] = None, None
        if long_draws:
            status, pred = integrate(prepared.rom.model, arrays["O_mean_rao_blackwell"],
                                     target.state0_comp, inputs, long_t, cap, duration)
            row["long_mean"] = status
            if pred is not None:
                plots[f"long_mean_{k}"] = pred
            for s in indices[:long_draws]:
                status, _ = integrate(prepared.rom.model, operators[s], q0s[s], inputs, long_t, cap, duration)
                row["long"].append(dict(status, draw_index=int(s)))
            row["long_survival"] = sum(s["complete"] for s in row["long"]) / len(row["long"])
        if mean is not None:
            plots[f"mean_{k}"] = mean
        plots.update({f"{name}_{k}": value for name, value in (
            ("truth", truth), ("t", t), ("observations", target.snapshots_comp),
            ("t_sampled", target.t_sampled), ("q0_samples", q0s),
            ("survived_indices", np.asarray(survived, dtype=int)))})
        reports.append(row)
        print(json.dumps({key: value for key, value in row.items() if key not in ("short", "long")}), flush=True)
    plots["long_t"] = long_t
    np.savez_compressed(output / "plotdata.npz", **plots)
    losses = arrays["losses"]
    report = dict(targets=reports, fit_seconds=float(arrays["fit_seconds"]),
                  final_loss_mean=float(losses[-500:].mean()),
                  final_loss_sd=float(losses[-1000:].std()),
                  loss_block_change=float(losses[-500:].mean() - losses[-1000:-500].mean())
                  if len(losses) >= 1000 else None)
    write_json(output / "summary.json", report)


def main(args):
    if args.steps < 100 or args.steps % 100 or args.draws < 1 or not 0 <= args.long_draws <= min(200, args.draws):
        raise ValueError("Steps must be a positive multiple of 100; draw counts must be valid")
    jax.config.update("jax_enable_x64", True)
    module, spec = load_adapter(args.system)
    schema = next(s for s in module.SCHEMAS if s["name"] == args.schema)
    cfg = replace(module.make_config(schema), seed=args.seed, precision="float64",
                  num_steps=args.steps, num_posterior_samples=args.draws,
                  op_prior_mode="block_hier" if args.variant == "hierarchical" else "fixed")
    np.random.seed(cfg.seed)
    prepared = spec.prepare(cfg, schema)
    request = dict(system=args.system, schema=schema, config=asdict(cfg), variant=args.variant,
                   training_hash=fingerprint(prepared), source_commit=args.source_commit,
                   long_draws=args.long_draws, initialization="init_to_median")
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Existing output belongs to a different request")
    write_json(request_path, request)
    posterior_path = args.output / "posterior.npz"
    if posterior_path.exists():
        with np.load(posterior_path) as saved:
            arrays = {key: saved[key] for key in saved.files}
    else:
        arrays = fit(prepared, cfg, args.variant, args.output)
    evaluate(prepared, cfg, arrays, module, args.system, args.output, args.long_draws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("heat", "burgers_2d"), required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--long-draws", type=int, default=50)
    main(parser.parse_args())
