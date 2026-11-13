"""Matched full-joint search for one simple Gaussian prior across four systems."""

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
from numpyro.infer import SVI, Trace_ELBO, autoguide, init_to_value
from numpyro.optim import ClippedAdam

from core.weakform_opinf import WeakFormConfig, gp
from core.weakform_opinf.gaussian_priors import training_scaled_prior
from core.weakform_opinf.model import _block_id_from_rom, build_model
from core.weakform_opinf.pipeline import _stack_theta
from experiments.stability_transfer.run import evaluate, fingerprint, write_json


def candidates():
    cases = {}

    def add(name, normalization="rate", center=False, width=1., fixed=False, **kwargs):
        cases[name] = dict(prior=dict(normalization=normalization, center=center, **kwargs),
                           width=width, fixed=fixed)

    cases["original"] = dict(prior=None, width=3., fixed=False)
    cases["original-h1"] = dict(prior=None, width=1., fixed=False)
    for width in (1., 3.):
        add(f"balanced-h{width:g}", "balanced", width=width)
    add("balanced-centered", "balanced", center=True)
    for center in (False, True):
        suffix = "-centered" if center else ""
        add(f"amplitude{suffix}", "amplitude", center=center)
        for width in (.5, 1., 3.):
            add(f"rate{suffix}-h{width:g}", center=center, width=width)
    for scale in (.3, 3.):
        add(f"rate-centered-scale{scale:g}", center=True, scale=scale)
    add("dimensional-centered-d1", "dimensional", center=True, width=3., damping=1.)
    for damping in (.1, 1., 3.):
        add(f"rate-centered-d{damping:g}", center=True, damping=damping)
    add("rate-d1", damping=1.)
    for center in (False, True):
        for ridge in (.01, .0001):
            name = f"support-{'centered-' if center else ''}{ridge:g}"
            add(name, center=center, correlation="block", ridge=ridge)
    for ridge in (.01, .0001):
        add(f"full-support-{ridge:g}", center=True, correlation="full", ridge=ridge, fixed=True)
    for scale in (.3, 1., 3.):
        add(f"fixed-rate-centered-{scale:g}", center=True, scale=scale, fixed=True)
    for damping in (.3, 1., 3.):
        add(f"rate-clock-d{damping:g}", center=True, damping=damping, mean_clock="rate")
    add("amplitude-clock-d1", "amplitude", center=True, damping=1., mean_clock="rate")
    add("dimensional-clock-d1", "dimensional", center=True, damping=1., mean_clock="rate", width=3.)
    add("support-clock-d1", center=True, damping=1., mean_clock="rate", correlation="block", ridge=.01)
    return cases


CANDIDATES = candidates()


def load_adapter(system):
    folder = ROOT / "experiments" / system
    sys.path.insert(0, str(folder))
    path = folder / ("05_multi_re.py" if system == "cylinder" else "04_unified.py")
    spec = importlib.util.spec_from_file_location("general_prior_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if system == "cylinder":
        return module, None
    cls = {"euler": "EulerSpec", "heat": "HeatSpec", "burgers_2d": "BurgersSpec"}[system]
    return module, getattr(module, cls)()


def gp_arrays(prepared, cfg, grids, hypers):
    states, rates = [], []
    for k, (tr, grid) in enumerate(zip(prepared.trajectories, grids)):
        batch = jax.jit(gp.trajectory_gp_conditional(tr, cfg)(grid)[1])
        values = batch(*(jnp.asarray(h[k]) for h in hypers), jnp.asarray(tr["snapshots_comp"]))
        states.append(np.asarray(values[0]))
        rates.append(np.asarray(values[1]))
    return np.array(states), np.array(rates)


def atomic_arrays(path, arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def fit(prepared, cfg, candidate, output):
    started = time.monotonic()
    model, posterior, grids, priors = build_model(prepared.rom, prepared.trajectories, cfg)
    fitted = priors["closure"]["data_only_gp"]
    map_hypers = tuple(np.asarray(fitted[key]) for key in ("ell", "sig2", "nu"))
    states, rates = gp_arrays(prepared, cfg, grids, map_hypers)
    prior_arrays = dict(gp_map_states=states, gp_map_rates=rates)
    if candidate["prior"] is not None:
        mean, root, info = training_scaled_prior(
            prepared.rom, prepared.trajectories, states, rates, **candidate["prior"])
        model, posterior, grids, priors = build_model(
            prepared.rom, prepared.trajectories, cfg, prior_mean=mean, prior_root=root)
        priors["training_scaled_prior"] = info
        prior_arrays.update(mean=mean, root=root)
    atomic_arrays(output / "operator_prior.npz", prior_arrays)
    write_json(output / "priors.json", priors)
    values = {f"{site}_{k}_{i}": jnp.asarray(fitted[key][k][i])
              for site, key in (("lengthscale", "ell"), ("variance", "sig2"), ("noise", "nu"))
              for k in range(len(prepared.trajectories)) for i in range(cfg.num_modes)}
    if cfg.op_prior_mode == "block_hier":
        values["log_tau_block"] = jnp.zeros(_block_id_from_rom(prepared.rom)[2])
    gamma = priors["gamma2"]
    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_value(values=values))
    svi = SVI(model, guide, ClippedAdam(step_size=cfg.learning_rate), Trace_ELBO())
    key, ik = jax.random.split(jax.random.PRNGKey(cfg.seed))
    state = svi.init(ik, gamma2=gamma)
    completed, losses, previous_seconds = 0, [], 0.
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        state, completed, losses = saved["state"], saved["completed"], saved["losses"]
        previous_seconds = saved["fit_seconds"]
        print(f"resuming step {completed}", flush=True)

    @jax.jit
    def block(state):
        return jax.lax.scan(lambda s, _: svi.update(s, gamma2=gamma), state, jnp.arange(100))

    while completed < cfg.num_steps:
        state, batch_losses = block(state)
        batch_losses = np.asarray(batch_losses)
        if not np.isfinite(batch_losses).all():
            raise FloatingPointError(f"Nonfinite loss after step {completed}")
        losses.extend(batch_losses.tolist())
        completed += len(batch_losses)
        if completed % 1000 == 0 or completed == cfg.num_steps:
            temporary = checkpoint.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(dict(state=jax.device_get(state), completed=completed, losses=losses,
                                 fit_seconds=previous_seconds + time.monotonic() - started), stream)
            temporary.replace(checkpoint)
            print(f"step={completed} loss={batch_losses.mean():.7g}", flush=True)
    key, sk = jax.random.split(key)
    post = guide.sample_posterior(sk, svi.get_params(state),
                                  sample_shape=(cfg.num_posterior_samples,), gamma2=gamma)
    hypers = _stack_theta(post, len(prepared.trajectories), cfg.num_modes)
    taus = jnp.exp(post["log_tau_block"]) if "log_tau_block" in post else None
    key, ok = jax.random.split(key)
    keys = jax.random.split(ok, cfg.num_posterior_samples)

    @jax.jit
    def draw(theta, key, tau):
        mean, root = posterior(theta, gamma, None, tau)
        return mean + jnp.einsum("ijk,ik->ij", root, jax.random.normal(key, mean.shape)), mean

    samples, means = [], []
    for s in range(cfg.num_posterior_samples):
        sample, mean = draw(tuple(h[s] for h in hypers), keys[s], None if taus is None else taus[s])
        samples.append(np.asarray(sample))
        means.append(np.asarray(mean))
    joint_states, joint_rates = gp_arrays(prepared, cfg, grids, [np.asarray(h).mean(0) for h in hypers])
    arrays = dict(O_samples=np.array(samples), O_conditional_means=np.array(means),
                  O_mean_rao_blackwell=np.mean(means, axis=0), losses=np.array(losses),
                  fit_seconds=np.array(previous_seconds + time.monotonic() - started),
                  gp_map_states=states, gp_map_rates=rates,
                  gp_joint_states=joint_states, gp_joint_rates=joint_rates)
    arrays.update({name: np.asarray(value) for name, value in zip(
        ("gp_lengthscale_samples", "gp_variance_samples", "gp_noise_samples"), hypers)})
    if taus is not None:
        arrays["op_log_tau_samples"] = np.asarray(post["log_tau_block"])
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Nonfinite posterior export")
    atomic_arrays(output / "posterior.npz", arrays)
    return arrays


def main(args):
    if (args.steps < 100 or args.steps % 100 or args.draws < 1
            or not 0 <= args.long_draws <= args.draws):
        raise ValueError("Steps must be a positive multiple of 100; draw counts must be valid")
    jax.config.update("jax_enable_x64", True)
    candidate = CANDIDATES[args.candidate]
    module, spec = load_adapter(args.system)
    if args.system == "cylinder":
        schema = dict(name="dense_low_noise", window=[args.window_start, 1., 3.])
        cfg = WeakFormConfig(operators="cAHBN", num_modes=args.modes)
        spec = module.MultiReSpec([150., 300.], [200.], window=tuple(schema["window"]), samples=250)
    else:
        schema = next(s for s in module.SCHEMAS if s["name"] == args.schema)
        cfg = module.make_config(schema)
    cfg = replace(cfg, seed=args.seed, precision="float64", num_steps=args.steps,
                  num_posterior_samples=args.draws, hier_tau_scale=candidate["width"],
                  op_prior_mode="fixed" if candidate["fixed"] else "block_hier")
    np.random.seed(cfg.seed)
    prepared = spec.prepare(cfg, schema)
    request = dict(system=args.system, schema=schema, config=asdict(cfg), candidate=args.candidate,
                   specification=candidate, training_hash=fingerprint(prepared), source_commit=args.source_commit,
                   long_draws=args.long_draws, initialization="data-only GP MAP; block multiplier centers")
    args.output.mkdir(parents=True, exist_ok=True)
    request_path = args.output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != json.loads(json.dumps(request)):
        raise ValueError("Existing output belongs to a different request")
    write_json(request_path, request)
    posterior_path = args.output / "posterior.npz"
    if posterior_path.exists():
        with np.load(posterior_path) as saved:
            arrays = {key: saved[key] for key in saved.files}
    else:
        arrays = fit(prepared, cfg, candidate, args.output)
    evaluate(prepared, cfg, arrays, module, args.system, args.output, args.long_draws, short_draws=args.draws)
    report_path = args.output / "summary.json"
    report = json.loads(report_path.read_text())
    report["gp_rate_relative_shift"] = (
        np.linalg.norm(arrays["gp_joint_rates"]-arrays["gp_map_rates"], axis=2)
        / np.maximum(np.linalg.norm(arrays["gp_map_rates"], axis=2), np.finfo(float).tiny)).tolist()
    if "op_log_tau_samples" in arrays:
        report["block_multiplier_median"] = np.exp(np.median(arrays["op_log_tau_samples"], axis=0)).tolist()
    write_json(report_path, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("euler", "heat", "burgers_2d", "cylinder"), required=True)
    parser.add_argument("--schema", default="dense_low_noise")
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--long-draws", type=int, default=20)
    parser.add_argument("--modes", type=int, default=25)
    parser.add_argument("--window-start", type=float, default=16.)
    main(parser.parse_args())
