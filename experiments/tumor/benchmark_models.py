"""Model fits for the tumor benchmarks: production Bayesian OpInf and the Neural-ODE baselines.

Both methods receive exactly the arrays in ``data/observations.npz`` (sampled
times, reduced observations, the observed initial state and, for chemo, the
training-regimen exposure table). Neither reads a forecast reference.

Every fit is resumable: the production SVI state is checkpointed every 1000
updates, chemo Neural-ODE members every 1000 updates (plus the final
pre-update state), and the batched untreated-growth ensemble every 1000
updates. Checkpoints are plain ``.npz``/Equinox leaf files (no pickles).
"""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import os
import tempfile
import time
from typing import Any, NamedTuple

import numpy as np

import benchmark_data as bd
import benchmark_environment
from benchmark_cases import production_config

PRODUCTION_DRAWS = 500
PRODUCTION_CHUNK = 1000
NODE_MEMBERS = 20
NODE_HIDDEN, NODE_LAYERS = 128, 3
LOSS_OUTLIER_FACTOR = 3.
CHEMO_NODE_STEPS = 6000
GROWTH_NODE_STEPS = 3000
GROWTH_NODE_CHUNK = 100
NODE_CHECKPOINT_EVERY = 1000


def _require_float32():
    import jax
    if jax.config.jax_enable_x64:
        raise RuntimeError("The benchmark fits use JAX's default float32; disable jax_enable_x64.")


def _checked_steps(steps, default):
    total = default if steps is None else int(steps)
    if total < 1 or total > default:
        raise ValueError(f"Training steps must be between 1 and the recipe's {default}.")
    return total


def _same_request(path, requested):
    """Refuse to reuse or resume a fit that was started with different settings."""
    if not path.exists():
        return
    recorded = bd.read_json(path)
    changed = {key: (recorded.get(key), value) for key, value in requested.items() if recorded.get(key) != value}
    if changed:
        raise ValueError(f"{path.parent} was started with different settings {changed} (recorded, requested); "
                         "use another --output-root or remove that directory.")


# =============================================================================
# Production: weak-form Bayesian OpInf (unchanged production configuration)
# =============================================================================
def production_directory(case, seed, pod=None, root=None):
    return bd.seed_directory(case, seed, pod, root) / "production"


def _production_structure(case, cfg, observations):
    """Operator layout for build_model (the least-squares values are not prior centers)."""
    import opinf
    import config as tumor_config
    from core import JaxCompatibleModel
    times = observations["t_sampled"]
    if case.dose_days:
        columns = {"c": 1, "A": cfg.num_modes, "B": 1, "N": cfg.num_modes}
        ridge = np.concatenate([np.full(columns[key], cfg.regularizer) for key in cfg.operators])
        model = tumor_config.ChemoReducedOrderModel(
            operator_string=cfg.operators, solver=opinf.lstsq.TikhonovSolver(regularizer=np.diag(ridge)))
        inputs = np.asarray([np.interp(times, observations["input_times"], observations["input_values"])])
    else:
        model = JaxCompatibleModel(operators=cfg.operators,
                                   solver=opinf.lstsq.L2Solver(regularizer=cfg.regularizer))
        inputs = None
    rom = opinf.ROM(model=model, ddt_estimator=opinf.ddt.NonuniformFiniteDifferencer(times))
    rom.fit(states=observations["y"], inputs=inputs)
    return rom


def _save_svi(path, state, losses, steps):
    import jax
    leaves = jax.tree_util.tree_leaves(jax.device_get(state))
    bd.write_npz(path, steps=np.asarray(steps), losses=np.asarray(losses, dtype=np.float32),
                 **{f"leaf{i:04d}": np.asarray(leaf) for i, leaf in enumerate(leaves)})


def _load_svi(path, template):
    import jax
    import jax.numpy as jnp
    arrays = bd.read_npz(path)
    leaves, tree = jax.tree_util.tree_flatten(template)
    restored = [jnp.asarray(arrays[f"leaf{i:04d}"]) for i in range(len(leaves))]
    for new, old in zip(restored, leaves):
        if new.shape != np.shape(old) or new.dtype != np.asarray(old).dtype:
            raise ValueError(f"SVI checkpoint {path} does not match this model.")
    return jax.tree_util.tree_unflatten(tree, restored), int(arrays["steps"]), arrays["losses"]


def fit_production(case, seed, pod=None, root=None, *, steps=None, log=print):
    """Run the production SVI and export 500 operator/initial-state posterior draws."""
    pod = case.pod if pod is None else pod
    folder = production_directory(case, seed, pod, root)
    cfg = production_config(case, pod)
    total = _checked_steps(steps, cfg.num_steps)
    _same_request(folder / "fit_start.json", {"planned_updates": total, "production_config": asdict(cfg)})
    if (folder / "result.json").exists():
        return bd.read_json(folder / "result.json")
    data = bd.load(case, seed, pod, root, references=False)
    observations = data["observations"]

    import jax
    import jax.numpy as jnp
    from jax import random
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO, autoguide
    from numpyro.infer.initialization import init_to_median
    from numpyro.optim import ClippedAdam
    from core.weakform_opinf import build_model
    _require_float32()
    numpyro.set_platform("cpu")
    np.random.seed(cfg.seed)
    started = time.monotonic()
    rom = _production_structure(case, cfg, observations)
    trajectories = [{"t_sampled": observations["t_sampled"], "snapshots_comp": observations["y"],
                     "inputs_eval": observations["native_inputs_eval"] if case.dose_days else None}]
    model, conditional, _, priors = build_model(rom, trajectories, cfg)
    guide = autoguide.AutoNormal(model, init_loc_fn=init_to_median)
    svi = SVI(model, guide, ClippedAdam(step_size=cfg.learning_rate), loss=Trace_ELBO())
    remaining, init_key = random.split(random.PRNGKey(cfg.seed))
    remaining, posterior_key = random.split(remaining)
    _, operator_key = random.split(remaining)
    state = svi.init(init_key, gamma2=cfg.gamma2)

    checkpoints = folder / "checkpoints"
    done, losses = 0, np.empty(0, dtype=np.float32)
    existing = sorted(checkpoints.glob("step*.npz"))
    if existing:
        state, done, losses = _load_svi(existing[-1], state)
        if len(losses) != done or done > total:
            raise ValueError(f"Inconsistent production checkpoint {existing[-1]}.")
        log(f"  resuming production SVI at update {done}")
    else:
        bd.write_json(folder / "fit_start.json", {
            "case": case.name, "seed": seed, "pod_tag": bd.pod_tag(pod), "observation": case.observation,
            "production_config": asdict(cfg),
            "planned_updates": total, "observations_sha256": data["metadata"]["observations_sha256"],
            "inputs": "Reduced training observations and (chemo) the training exposure table only.",
            "ODE_calls_in_objective": 0, "thread_environment": benchmark_environment.current()})

    @jax.jit
    def step(current, unused):
        return svi.update(current, gamma2=cfg.gamma2)

    for left in range(done, total, PRODUCTION_CHUNK):
        length = min(PRODUCTION_CHUNK, total - left)
        state, chunk = jax.lax.scan(step, state, jnp.arange(length))
        losses = np.r_[losses, np.asarray(chunk)]
        _save_svi(checkpoints / f"step{left + length:05d}.npz", state, losses, left + length)
        log(f"  production SVI {left + length}/{total}: loss {float(losses[-1]):.6g}")
    bd.write_npz(folder / "losses.npz", losses=losses)
    exported = _export_production(folder, cfg, observations, guide, svi, state, conditional,
                                  posterior_key, operator_key)
    complete = bool(np.isfinite(losses).all() and np.isfinite(exported["O_point"]).all())
    result = {
        "status": "complete" if complete else "nonfinite_production_fit",
        "case": case.name, "seed": seed, "pod_tag": bd.pod_tag(pod), "observation": case.observation,
        "steps": total,
        "recipe_steps": cfg.num_steps, "reduced_steps": total != cfg.num_steps,
        "finite_losses": int(np.isfinite(losses).sum()),
        "finite_operator_draws": int(np.isfinite(exported["O_samples"]).all(axis=(1, 2)).sum()),
        "operators_sha256": bd.digest(folder / "operators.npz"),
        "initial_conditions_sha256": bd.digest(folder / "initial_conditions.npz"),
        "production_config": asdict(cfg), "prior_info": priors,
        "point_definition": "Conditional operator mean at the mean of 500 GP/tau draws; observed initial state.",
        "runtime_seconds": time.monotonic() - started, "ODE_calls_in_objective": 0,
        "thread_environment": benchmark_environment.current(),
        "reported_thread_environment": benchmark_environment.matches_reported(),
    }
    bd.write_json(folder / "result.json", result)
    return result


def _export_production(folder, cfg, observations, guide, svi, state, conditional, posterior_key, operator_key):
    import jax
    import jax.numpy as jnp
    from jax import random
    from core.weakform_opinf.pipeline import _ic_sigma, _stack_theta
    r = cfg.num_modes
    params = svi.get_params(state)
    posterior = guide.sample_posterior(posterior_key, params, sample_shape=(PRODUCTION_DRAWS,), gamma2=cfg.gamma2)
    theta = _stack_theta(posterior, 1, r)
    tau = jnp.exp(jnp.asarray(posterior["log_tau_block"]))
    sigma_O = None if cfg.sigma_O is None else jnp.asarray(cfg.sigma_O)
    keys = random.split(operator_key, PRODUCTION_DRAWS)

    @jax.jit
    def draw_operator(theta_s, key, tau_s):
        mean, root = conditional(theta_s, cfg.gamma2, sigma_O, tau_s)
        return mean + jnp.einsum("ijk,ik->ij", root, random.normal(key, shape=mean.shape))

    samples = np.stack([np.asarray(draw_operator(tuple(value[i] for value in theta), keys[i], tau[i]))
                        for i in range(PRODUCTION_DRAWS)])
    mean_hypers = tuple(np.asarray(value).mean(0) for value in theta)
    mean_tau = np.asarray(tau).mean(0)
    point, root = conditional(tuple(jnp.asarray(value) for value in mean_hypers), cfg.gamma2, sigma_O,
                              jnp.asarray(mean_tau))
    arrays = {
        "O_point": np.asarray(point), "O_samples": samples, "conditional_root": np.asarray(root),
        "gp_lengthscale_samples": np.asarray(theta[0]), "gp_variance_samples": np.asarray(theta[1]),
        "gp_noise_samples": np.asarray(theta[2]), "tau_block_samples": np.asarray(tau),
        "mean_lengthscale": mean_hypers[0], "mean_variance": mean_hypers[1], "mean_noise": mean_hypers[2],
        "mean_tau_block": mean_tau, "posterior_key": np.asarray(posterior_key),
        "operator_root_key": np.asarray(operator_key),
    }
    bd.write_npz(folder / "operators.npz", **arrays)
    # Native initial-state rule: GP posterior sd at t0 from the mean hyperparameters, 500 draws.
    sigma = _ic_sigma(observations["t_sampled"], *(value[0] for value in mean_hypers), r,
                      roundoff_nugget=cfg.gp_jitter_rel is None)
    epsilon = np.random.default_rng(cfg.seed).standard_normal((PRODUCTION_DRAWS, r))
    draws = np.broadcast_to(observations["q0"], (PRODUCTION_DRAWS, r))
    if cfg.ic_uncertainty:
        draws = draws + cfg.ic_scale * sigma[None] * epsilon
    bd.write_npz(folder / "initial_conditions.npz", point=observations["q0"], all_draws=np.asarray(draws),
                 sigma=sigma)
    return arrays


def load_production(case, seed, pod=None, root=None):
    folder = production_directory(case, seed, pod, root)
    result = bd.read_json(folder / "result.json")
    for name in ("operators", "initial_conditions"):
        if bd.digest(folder / f"{name}.npz") != result[f"{name}_sha256"]:
            raise RuntimeError(f"{folder / name}.npz changed after the fit.")
    return {"result": result, "operators": bd.read_npz(folder / "operators.npz"),
            "initial": bd.read_npz(folder / "initial_conditions.npz")}


# =============================================================================
# Neural-ODE baselines (the repository's 05_neural_ode*.py architectures)
# =============================================================================
def node_directory(case, seed, pod=None, root=None):
    return bd.seed_directory(case, seed, pod, root) / "neural_ode"


def _baseline(filename):
    """Import a historical baseline module for its model class only (its main() is not run)."""
    import sys
    name = "_tumor_benchmark_" + filename[:-3]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, bd.SCRIPT_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _weights(model):
    values = {}
    for i, layer in enumerate(model.layers):
        values[f"weight_{i}"] = np.asarray(layer.weight)
        values[f"bias_{i}"] = np.asarray(layer.bias)
    if any(value.dtype != np.float32 for value in values.values()):
        raise ValueError("Neural-ODE weights must stay float32.")
    return values


def _filter(last_losses):
    """Original chemo baseline rule: drop members whose final loss exceeds 3x the ensemble median."""
    last = np.asarray(last_losses, dtype=np.float64)
    if not np.isfinite(last).all():
        return {"policy_defined": False, "kept_indices": None, "median": None, "cutoff": None,
                "final_recorded_preupdate_losses": last}
    median = float(np.median(last))
    return {"policy_defined": True, "kept_indices": np.flatnonzero(last <= LOSS_OUTLIER_FACTOR * median).tolist(),
            "median": median, "cutoff": LOSS_OUTLIER_FACTOR * median, "factor": LOSS_OUTLIER_FACTOR,
            "loss_kind": "last recorded pre-update training loss", "final_recorded_preupdate_losses": last}


def _chemo_node():
    """Chemo kernels, identical to the reported study's scalar reference step."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import diffrax
    import optax

    baseline = _baseline("05_neural_ode_chemo.py")
    params = baseline.MODEL_PARAMS
    if (params["HIDDEN_DIM"], params["NUM_LAYERS"], params["ENSEMBLE_SIZE"], params["NUM_TRAIN_STEPS"],
            params["LEARNING_RATE"], params["GRAD_CLIP"], params["SEED"], params["LOSS_OUTLIER_FACTOR"]) != (
            NODE_HIDDEN, NODE_LAYERS, NODE_MEMBERS, CHEMO_NODE_STEPS, 5e-4, 1., 42, LOSS_OUTLIER_FACTOR):
        raise ValueError("05_neural_ode_chemo.py changed its model/optimizer recipe.")
    optimizer = optax.chain(optax.clip_by_global_norm(params["GRAD_CLIP"]), optax.adam(params["LEARNING_RATE"]))

    class InputTable(eqx.Module):
        times: jax.Array
        values: jax.Array

        def __call__(self, t):
            return jnp.atleast_1d(jnp.interp(t, self.times, self.values))

    class TrainingInput(NamedTuple):
        times: Any
        observations: Any
        q0: Any
        forcing: Any
        step_cuts: Any
        dt0: Any

    class MemberState(NamedTuple):
        model: Any
        optimizer: Any
        last_pre_model: Any
        last_pre_optimizer: Any
        active: Any
        steps_completed: Any
        attempts: Any
        failure_step: Any

    def training_input(observations):
        times = jnp.asarray(observations["t_sampled"], dtype=jnp.float32)
        span = float(observations["t_sampled"][-1] - observations["t_sampled"][0])
        return TrainingInput(
            times, jnp.asarray(observations["y"], dtype=jnp.float32),
            jnp.asarray(observations["q0"], dtype=jnp.float32),
            InputTable(jnp.asarray(observations["input_times"], dtype=jnp.float32),
                       jnp.asarray(observations["input_values"], dtype=jnp.float32)),
            jnp.asarray(observations["node_step_cuts"], dtype=jnp.float32),
            jnp.minimum(jnp.minimum(times[1] - times[0], jnp.asarray(span / 200., dtype=jnp.float32)),
                        jnp.asarray(.25, dtype=jnp.float32)))

    def solve(model, data):
        return diffrax.diffeqsolve(
            diffrax.ODETerm(model), diffrax.Tsit5(), t0=data.times[0], t1=data.times[-1], dt0=data.dt0,
            y0=data.q0, args=data.forcing, saveat=diffrax.SaveAt(ts=data.times),
            stepsize_controller=diffrax.ClipStepSizeController(
                diffrax.PIDController(rtol=1e-5, atol=1e-7, dtmax=.25), step_ts=data.step_cuts),
            adjoint=diffrax.RecursiveCheckpointAdjoint(), max_steps=16384, throw=False)

    @eqx.filter_jit
    def predict(model, data):
        result = solve(model, data)
        return result.ys.T, result.result._value

    @eqx.filter_jit
    def scalar_step(model, optimizer_state, data):
        def loss_fn(candidate):
            return jnp.mean((solve(candidate, data).ys - data.observations.T) ** 2)

        loss, gradients = eqx.filter_value_and_grad(loss_fn)(model)
        updates, next_optimizer = optimizer.update(gradients, optimizer_state, model)
        return eqx.apply_updates(model, updates), next_optimizer, loss, gradients

    def checked_step(state, data):
        model, opt_state, loss, gradients = scalar_step(state.model, state.optimizer, data)
        host_loss = np.asarray(loss)
        good = bool(np.isfinite(host_loss)) and all(
            np.isfinite(np.asarray(leaf)).all()
            for tree in (gradients, model, opt_state) for leaf in jax.tree_util.tree_leaves(tree))
        attempts, completed = int(np.asarray(state.attempts)), int(np.asarray(state.steps_completed))
        following = MemberState(
            model, opt_state, state.model, state.optimizer, jnp.asarray(good),
            jnp.asarray(completed + int(good), dtype=jnp.int32), jnp.asarray(attempts + 1, dtype=jnp.int32),
            state.failure_step if good else jnp.asarray(attempts, dtype=jnp.int32))
        return following, np.float32(host_loss), good

    def initial_state(rank, key):
        model = baseline.ChemoNeuralODE(rank, NODE_HIDDEN, NODE_LAYERS, key=key)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        return MemberState(model, opt_state, model, opt_state, jnp.asarray(True), jnp.asarray(0, dtype=jnp.int32),
                           jnp.asarray(0, dtype=jnp.int32), jnp.asarray(-1, dtype=jnp.int32))

    success = int(np.asarray(diffrax.RESULTS.successful._value))
    return dict(training_input=training_input, predict=predict, checked_step=checked_step,
                initial_state=initial_state, success=success)


def _train_chemo_member(kernels, data, folder, index, key, rank, total, log):
    import equinox as eqx
    directory = folder / f"member{index:02d}"
    if (directory / "result.json").exists():
        return bd.read_json(directory / "result.json")
    state = kernels["initial_state"](rank, key)
    losses = np.full(total, np.nan, dtype=np.float32)
    attempted, successful = np.zeros(total, dtype=bool), np.zeros(total, dtype=bool)
    start = 0
    checkpoints = sorted((directory / "checkpoints").glob("step*.eqx"))
    if checkpoints:
        latest = checkpoints[-1]
        state = eqx.tree_deserialise_leaves(latest, state)
        trace = bd.read_npz(latest.with_suffix(".npz"))
        losses, attempted, successful = trace["losses"], trace["attempted"], trace["successful"]
        start = int(np.asarray(state.attempts))
        if len(losses) != total or int(attempted.sum()) != start:
            raise ValueError(f"Neural-ODE checkpoint {latest} does not belong to a {total}-update fit.")
        log(f"  member {index:02d}: resuming at update {start}")
    started = time.monotonic()

    def checkpoint(step):
        directory.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
        path = directory / "checkpoints" / f"step{step:04d}.eqx"
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            eqx.tree_serialise_leaves(stream, state)
        os.replace(stream.name, path)
        bd.write_npz(path.with_suffix(".npz"), losses=losses, attempted=attempted, successful=successful)

    for step in range(start, total):
        if not bool(np.asarray(state.active)):
            break
        state, loss, good = kernels["checked_step"](state, data)
        losses[step], attempted[step], successful[step] = loss, True, good
        if (step + 1) % NODE_CHECKPOINT_EVERY == 0 or step + 1 == total - 1 or not good:
            checkpoint(step + 1)
        if not good:
            break
    weights = _weights(state.model)
    finite = all(np.isfinite(value).all() for value in weights.values())
    prediction, code = np.full(tuple(data.observations.shape), np.nan, dtype=np.float32), -1000
    if finite:
        value, solver_code = kernels["predict"](state.model, data)
        prediction, code = np.asarray(value), int(np.asarray(solver_code))
    bd.write_npz(directory / "weights.npz", **weights)
    bd.write_npz(directory / "losses.npz", losses=losses, optimizer_update_attempted=attempted,
                 successful_update=successful)
    bd.write_npz(directory / "training_prediction.npz", times=np.asarray(data.times), prediction=prediction)
    completed = int(np.asarray(state.steps_completed))
    result = {
        "status": "complete" if completed == total and finite else "training_failed",
        "member_index": index, "steps_completed": completed,
        "optimizer_update_attempts": int(np.asarray(state.attempts)),
        "first_failure_step": int(np.asarray(state.failure_step)),
        "final_recorded_preupdate_loss": float(losses[-1]) if np.isfinite(losses[-1]) else None,
        "final_training_solver_code": code,
        "final_training_solver_success": bool(code == kernels["success"] and np.isfinite(prediction).all()),
        "weights_sha256": bd.digest(directory / "weights.npz"),
        "runtime_seconds": time.monotonic() - started,
    }
    bd.write_json(directory / "result.json", result)
    log(f"  member {index:02d}: {result['status']}, last loss {result['final_recorded_preupdate_loss']}, "
        f"{result['runtime_seconds']:.0f} s")
    return result


def _growth_node():
    """Batched untreated-growth kernels, identical to the reported study's advance()."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import diffrax
    import optax

    baseline = _baseline("05_neural_ode.py")
    params = baseline.MODEL_PARAMS
    if (params["HIDDEN_DIM"], params["NUM_LAYERS"], params["ENSEMBLE_SIZE"], params["NUM_TRAIN_STEPS"],
            params["LEARNING_RATE"], params["SEED"]) != (NODE_HIDDEN, NODE_LAYERS, NODE_MEMBERS,
                                                          GROWTH_NODE_STEPS, 1e-3, 42):
        raise ValueError("05_neural_ode.py changed its model/optimizer recipe.")
    optimizer = optax.adam(params["LEARNING_RATE"])
    success = int(np.asarray(diffrax.RESULTS.successful._value))

    def finite_tree(tree):
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)
                                  if eqx.is_array(x)]))

    def choose(mask, new, old):
        return jax.tree_util.tree_map(
            lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b), new, old)

    def solve(model, q0, times, dt_cap):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(model), diffrax.Tsit5(), t0=times[0], t1=times[-1],
            dt0=jnp.minimum(times[1] - times[0], jnp.float32(dt_cap)), y0=q0, saveat=diffrax.SaveAt(ts=times),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            adjoint=diffrax.RecursiveCheckpointAdjoint(), max_steps=16384, throw=False)
        return solution.ys.T, (solution.result._value, solution.stats["num_steps"],
                               solution.stats["num_accepted_steps"], solution.stats["num_rejected_steps"])

    def objective(model, q0, times, observations, dt_cap):
        prediction, info = solve(model, q0, times, dt_cap)
        return jnp.mean((prediction.T - observations.T) ** 2), (*info, jnp.all(jnp.isfinite(prediction)))

    value_grad = eqx.filter_value_and_grad(objective, has_aux=True)

    def raw_step(model, opt_state, q0, times, observations, dt_cap):
        (loss, aux), gradient = value_grad(model, q0, times, observations, dt_cap)
        updates, opt_state = optimizer.update(gradient, opt_state, model)
        model = eqx.apply_updates(model, updates)
        finite_gradient, finite_state = finite_tree(gradient), finite_tree((model, opt_state))
        valid = (aux[0] == success) & aux[4] & jnp.isfinite(loss) & finite_gradient & finite_state
        return model, opt_state, gradient, (loss, *aux, finite_gradient, finite_state, valid)

    @eqx.filter_jit
    def advance(carry, initial_models, initial_states, q0, times, observations, dt_cap, length):
        def body(carry, unused):
            models, states, active, completed, attempted = carry
            new_models, new_states, _, info = jax.lax.map(
                lambda pair: raw_step(pair[0], pair[1], q0, times, observations, dt_cap),
                (choose(active, models, initial_models), choose(active, states, initial_states)))
            valid = info[-1] & active
            logs = (jnp.where(active, info[0], jnp.nan), jnp.where(active, info[1], -1),
                    jnp.where(active, info[2], 0), jnp.where(active, info[3], 0), jnp.where(active, info[4], 0),
                    active & info[5], active & info[6], active & info[7], valid, active)
            return (choose(active, new_models, models), choose(active, new_states, states), valid,
                    completed + valid.astype(jnp.int32), attempted + active.astype(jnp.int32)), logs

        return jax.lax.scan(body, carry, xs=None, length=length)

    @eqx.filter_jit
    def batch_solve(models, q0, times, dt_cap):
        return jax.lax.map(lambda model: solve(model, q0, times, dt_cap), models)

    def initial(keys, rank):
        models = eqx.filter_vmap(lambda key: baseline.NeuralODE(rank, NODE_HIDDEN, NODE_LAYERS, key=key))(keys)
        states = eqx.filter_vmap(lambda model: optimizer.init(eqx.filter(model, eqx.is_array)))(models)
        return models, states

    return dict(advance=advance, batch_solve=batch_solve, initial=initial, success=success)


GROWTH_LOG_NAMES = ("losses", "solver_codes", "solver_steps", "accepted_steps", "rejected_steps",
                    "finite_prediction", "finite_gradient", "finite_post_update_state", "successful_update",
                    "update_attempted")


def _train_growth(folder, observations, members, total, log):
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    kernels = _growth_node()
    rank = observations["y"].shape[0]
    keys = jax.random.split(jax.random.PRNGKey(42), NODE_MEMBERS)[:members]
    models, states = kernels["initial"](keys, rank)
    q0 = jnp.asarray(observations["q0"], dtype=jnp.float32)
    times = jnp.asarray(observations["t_sampled"], dtype=jnp.float32)
    y = jnp.asarray(observations["y"], dtype=jnp.float32)
    # Static Python float, so the step cap is a trace-time float32 constant as in the reported study.
    dt_cap = float(observations["t_sampled"][-1] - observations["t_sampled"][0]) / 200.
    carry = (models, states, jnp.ones(members, dtype=bool), jnp.zeros(members, dtype=jnp.int32),
             jnp.zeros(members, dtype=jnp.int32))
    logs, done = [], 0
    existing = sorted((folder / "checkpoints").glob("step*.eqx"))
    if existing:
        carry = eqx.tree_deserialise_leaves(existing[-1], carry)
        trace = bd.read_npz(existing[-1].with_suffix(".npz"))
        done = int(trace["steps"])
        if done > total or trace["losses"].shape != (members, done):
            raise ValueError(f"Growth checkpoint {existing[-1]} does not belong to this {members}-member fit.")
        logs = [tuple(trace[name].T for name in GROWTH_LOG_NAMES)]
        log(f"  resuming the growth ensemble at update {done}")
    started = time.monotonic()
    for left in range(done, total, GROWTH_NODE_CHUNK):
        length = min(GROWTH_NODE_CHUNK, total - left)
        carry, chunk = kernels["advance"](carry, models, states, q0, times, y, dt_cap, length)
        logs.append(tuple(np.asarray(x) for x in chunk))
        if (left + length) % NODE_CHECKPOINT_EVERY == 0 or left + length == total:
            path = folder / "checkpoints" / f"step{left + length:04d}.eqx"
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                eqx.tree_serialise_leaves(stream, carry)
            os.replace(stream.name, path)
            bd.write_npz(path.with_suffix(".npz"), steps=np.asarray(left + length), **{
                name: np.concatenate([entry[k] for entry in logs], axis=0).T
                for k, name in enumerate(GROWTH_LOG_NAMES)})
            log(f"  growth ensemble {left + length}/{total}: median loss "
                f"{float(np.nanmedian(logs[-1][0][-1])):.6g}, active {int(np.asarray(carry[2]).sum())}/{members}")
    history = {name: np.concatenate([entry[k] for entry in logs], axis=0).T for k, name in enumerate(GROWTH_LOG_NAMES)}
    trained, _, active, completed, attempted = carry
    predictions, info = kernels["batch_solve"](trained, q0, times, dt_cap)
    predictions = np.asarray(predictions)
    elapsed = time.monotonic() - started
    results = []
    for i in range(members):
        directory = folder / f"member{i:02d}"
        model = jax.tree_util.tree_map(lambda x, index=i: x[index], trained)
        bd.write_npz(directory / "weights.npz", **_weights(model))
        bd.write_npz(directory / "losses.npz", **{name: value[i] for name, value in history.items()})
        bd.write_npz(directory / "training_prediction.npz", times=np.asarray(times), prediction=predictions[i])
        code = int(np.asarray(info[0][i]))
        solved = bool(code == kernels["success"] and np.isfinite(predictions[i]).all())
        complete = int(completed[i]) == total and bool(active[i]) and solved
        last = float(history["losses"][i, -1])
        result = {
            "status": "complete" if complete else "training_failed", "member_index": i,
            "steps_completed": int(completed[i]), "optimizer_update_attempts": int(attempted[i]),
            "final_recorded_preupdate_loss": last if np.isfinite(last) else None,
            "final_training_solver_code": code, "final_training_solver_success": solved,
            "weights_sha256": bd.digest(directory / "weights.npz"),
            "runtime_seconds_shared_batch": elapsed,
        }
        bd.write_json(directory / "result.json", result)
        results.append(result)
    return results


def train_node(case, seed, pod=None, root=None, *, members=None, steps=None, only_members=None, log=print):
    """Train (or resume) the 20-member Neural-ODE ensemble on the shared reduced observations."""
    import jax
    pod = case.pod if pod is None else pod
    folder = node_directory(case, seed, pod, root)
    count = NODE_MEMBERS if members is None else int(members)
    if not 1 <= count <= NODE_MEMBERS:
        raise ValueError(f"Choose between 1 and {NODE_MEMBERS} ensemble members.")
    total = _checked_steps(steps, CHEMO_NODE_STEPS if case.dose_days else GROWTH_NODE_STEPS)
    _same_request(folder / "fit_start.json", {"members": count, "updates": total})
    if (folder / "summary.json").exists():
        return bd.read_json(folder / "summary.json")
    _require_float32()
    data = bd.load(case, seed, pod, root, references=False)
    observations = data["observations"]
    if not (folder / "fit_start.json").exists():
        bd.write_json(folder / "fit_start.json", {
            "case": case.name, "seed": seed, "pod_tag": bd.pod_tag(pod), "observation": case.observation,
            "members": count, "updates": total,
            "baseline": "05_neural_ode_chemo.py" if case.dose_days else "05_neural_ode.py",
            "observations_sha256": data["metadata"]["observations_sha256"],
            "architecture": f"tanh MLP [{'q, alpha(t)' if case.dose_days else 'q'}] -> 128 -> 128 -> 128 -> r",
            "optimizer": "clip(1.0)+Adam(5e-4), 6000 updates" if case.dose_days else "Adam(1e-3), 3000 updates",
            "training_solver": "diffrax Tsit5, PID rtol 1e-5 atol 1e-7, float32",
            "thread_environment": benchmark_environment.current()})
    started = time.monotonic()
    if case.dose_days:
        kernels = _chemo_node()
        training = kernels["training_input"](observations)
        keys = jax.random.split(jax.random.PRNGKey(42), NODE_MEMBERS)
        selected = range(count) if only_members is None else sorted(set(only_members))
        for index in selected:
            if index >= count:
                raise ValueError(f"Member {index} is outside the {count}-member ensemble.")
            log(f"  training chemo Neural-ODE member {index:02d}/{count}")
            _train_chemo_member(kernels, training, folder, index, keys[index], observations["y"].shape[0],
                                total, log)
        missing = [i for i in range(count) if not (folder / f"member{i:02d}" / "result.json").exists()]
        if missing:
            log(f"  members still to train: {missing}")
            return None
        results = [bd.read_json(folder / f"member{i:02d}" / "result.json") for i in range(count)]
    else:
        if only_members is not None:
            raise ValueError("The untreated-growth ensemble trains all members together.")
        results = _train_growth(folder, observations, count, total, log)
    last = [row["final_recorded_preupdate_loss"] if row["final_recorded_preupdate_loss"] is not None else np.nan
            for row in results]
    selection = _filter(last)
    bd.write_json(folder / "filter.json", selection)
    summary = {
        "case": case.name, "seed": seed, "pod_tag": bd.pod_tag(pod), "observation": case.observation,
        "members": count, "steps": total,
        "reduced_members_or_steps": count != NODE_MEMBERS or total != (CHEMO_NODE_STEPS if case.dose_days
                                                                        else GROWTH_NODE_STEPS),
        "completed_members": sum(row["status"] == "complete" for row in results),
        "kept_indices": selection["kept_indices"], "filter_sha256": bd.digest(folder / "filter.json"),
        "runtime_seconds_this_execution": time.monotonic() - started, "source": "trained by this runner",
        "thread_environment": benchmark_environment.current(),
        "reported_thread_environment": benchmark_environment.matches_reported(),
    }
    bd.write_json(folder / "summary.json", summary)
    return summary


def load_node(case, seed, pod=None, root=None):
    folder = node_directory(case, seed, pod, root)
    summary = bd.read_json(folder / "summary.json")
    if bd.digest(folder / "filter.json") != summary["filter_sha256"]:
        raise RuntimeError(f"{folder / 'filter.json'} changed after training.")
    members = []
    for index in range(summary["members"]):
        directory = folder / f"member{index:02d}"
        result = bd.read_json(directory / "result.json")
        if bd.digest(directory / "weights.npz") != result["weights_sha256"]:
            raise RuntimeError(f"{directory / 'weights.npz'} changed after training.")
        members.append({"result": result, "weights": bd.read_npz(directory / "weights.npz"),
                        "training_prediction": bd.read_npz(directory / "training_prediction.npz")["prediction"]})
    return {"summary": summary, "filter": bd.read_json(folder / "filter.json"), "members": members}
