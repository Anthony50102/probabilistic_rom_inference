"""Virtual cohort of past patients behind the population-prior cases (``COHORT_DESIGN``, benchmark_cases.py).

Eight virtual patients share the target's anatomy and baseline tumour (as TumorTwin's own TNBC demo patients do)
and completed the planned course: twelve weekly doses from day 35, three of them with a neuropathy dose
modification. Their growth rate, diffusion and drug sensitivity vary around the tutorial twin. Each is solved with
the target's TumorTwin solver and scanned from day 5 to 119 on one imaging protocol (a scan every 7 or every 8 days,
starting member mod stride days after day 5) by the target's scanner and segmentation. The noisy scans of all
members, pooled, give the cohort's POD basis; the target's own scans are only projected onto it.

``build`` simulates, scans and pools the cohort once (both protocols; cached under data/benchmarks), after checking
that the member solver reproduces the target's cached history and that the member inputs reproduce the target's
exposure table. ``fit CASE`` fits the population operator for CASE's protocol and rank: the production weak-form
fit (the target case's own configuration) to all members' reduced scans at once, one trajectory per member. Its
point operator is the centre of the ``-cohort-prior`` cases' operator prior (benchmark_models.fit_production).

    python benchmark_cohort.py build
    python benchmark_cohort.py fit CASE [--steps N] [--output-root PATH]
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import benchmark_data as bd
import benchmark_environment
from benchmark_cases import (
    CASES, COHORT_CASES, COHORT_DESIGN, COHORT_SOURCES, PATIENT_DOSE_DAYS, TUTORIAL_TWIN, PODSettings, get_case,
    production_config)

PARAMETERS = ("growth_rate", "diffusion", "sensitivity")
OPERATOR_DRAWS = 500


def _common(field):
    values = {getattr(CASES[name], field) for name in COHORT_SOURCES}
    if len(values) != 1:
        raise ValueError(f"The cohort's target cases disagree on {field}: {values}")
    return values.pop()


# The cohort's scanner is the targets' (noise level x each member's own field range).
NOISE_LEVEL = _common("noise_level")


def design():
    """Everything the cohort's simulations, scans and bases depend on."""
    return {"cohort_design": COHORT_DESIGN, "centre": TUTORIAL_TWIN, "parameters": PARAMETERS,
            "dose_days": PATIENT_DOSE_DAYS, "decay_rate": bd.DECAY_RATE, "theta": bd.THETA,
            "solver_dt": bd.CHEMO_SOLVER_DT, "patient": bd.PATIENT, "noise_level": NOISE_LEVEL,
            "detection_alpha": bd.DETECTION_ALPHA, "segmentation_smoothing_mm": bd.SEGMENTATION_SMOOTHING_MM,
            "input_points": bd.INPUT_POINTS}


def design_fingerprint():
    return hashlib.sha256(json.dumps(bd.json_value(design()), sort_keys=True).encode()).hexdigest()


def _stem():
    return bd.CACHE / f"cohort_{design_fingerprint()[:12]}"


def record_path():
    return _stem().with_suffix(".json")


def basis_path(protocol):
    return Path(f"{_stem()}_{protocol}_basis.npz")


def members_path():
    return Path(f"{_stem()}_members.npz")


def check_target(case):
    """The cohort was built for the tutorial twin on the patient's schedule; refuse any other target."""
    if not case.cohort:
        raise ValueError(f"{case.name} has no cohort.")
    twin = {name: getattr(case, name) for name in PARAMETERS}
    if (twin != TUTORIAL_TWIN or tuple(case.dose_days) != PATIENT_DOSE_DAYS or case.training_strength != 1.
            or case.noise_level != NOISE_LEVEL or case.sensitivity != COHORT_DESIGN["input_sensitivity"]):
        raise ValueError(f"{case.name} is not the target the cohort was built for.")


# -----------------------------------------------------------------------------
# Members
# -----------------------------------------------------------------------------
def member_parameters():
    """log k, log D, log sensitivity = twin + log_sd (z_centre + z_member); z_centre drawn first, once."""
    rng = np.random.default_rng(COHORT_DESIGN["parameter_seed"])
    centre_z = rng.standard_normal(len(PARAMETERS))
    member_z = rng.standard_normal((COHORT_DESIGN["members"], len(PARAMETERS)))
    twin = np.log([TUTORIAL_TWIN[name] for name in PARAMETERS])
    values = np.exp(twin + COHORT_DESIGN["log_sd"] * (centre_z + member_z))
    return {"order": PARAMETERS, "centre_z": centre_z, "member_z": member_z, "values": values,
            "centre": np.exp(twin + COHORT_DESIGN["log_sd"] * centre_z),
            # The target relative to the population, in population standard deviations.
            "target_z": -centre_z}


def dose_factors(member):
    """Fraction of the planned dose given at each weekly dose: one held (0), then the reduced fraction."""
    factors = np.ones(len(PATIENT_DOSE_DAYS))
    for who, held, fraction in COHORT_DESIGN["dose_modifications"]:
        if who == member:
            factors[held] = 0.
            factors[held + 1:] = fraction
    return factors


def scan_days(member, protocol):
    """A scan every stride days, from member mod stride days after the first scan day to the last."""
    stride = COHORT_DESIGN["strides"][protocol]
    start, end = COHORT_DESIGN["span"]
    first = start + member % int(stride)
    return first + stride * np.arange(int((end - first) // stride) + 1)


def input_table(factors):
    """alpha(t) = input_sensitivity x sum of the given dose fractions x exp(-decay (t - dose day)): the target's
    exposure units, tabulated on the target's input grid (training start to forecast end)."""
    times = np.linspace(*COHORT_DESIGN["span"], bd.INPUT_POINTS)
    values = sum(factor * np.where(times >= day, np.exp(-bd.DECAY_RATE * (times - day)), 0.)
                 for day, factor in zip(PATIENT_DOSE_DAYS, factors))
    return times, COHORT_DESIGN["input_sensitivity"] * values


def solve_member(parameters, factors, initial, days):
    """TumorTwin fields (float32, len(days) x voxels) of one member at whole days, as the target's history and
    branches are solved: RK4 on the solver's 0.25-day grid merged with the dose times, from the baseline at day 0."""
    torch = bd._single_thread_torch()
    from torchdiffeq import odeint
    import generate_fom_data_chemo  # noqa: F401  (puts TumorTwin on sys.path)
    from tumortwin.models import ReactionDiffusion3D
    from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
    from tumortwin.types import ChemotherapySpecification
    patient = bd._patient()
    origin = patient.visits[0].time
    growth, diffusion, sensitivity = (float(value) for value in parameters)
    specs = []
    for fraction in sorted({float(f) for f in factors if f > 0}, reverse=True):
        given = [day for day, f in zip(PATIENT_DOSE_DAYS, factors) if f == fraction]
        specs.append(ChemotherapySpecification(
            sensitivity=sensitivity * fraction, decay_rate=bd.DECAY_RATE,
            times=[origin + timedelta(days=day) for day in given], doses=[1.] * len(given)))
    model = ReactionDiffusion3D(
        k=torch.tensor(growth), d=torch.tensor(diffusion), theta=torch.tensor(bd.THETA), patient_data=patient,
        initial_time=origin, chemotherapy_specifications=specs, radiotherapy_specification=None, require_grad=False)
    solver = TorchDiffEqSolver(model, TorchDiffEqSolverOptions(
        step_size=timedelta(days=bd.CHEMO_SOLVER_DT), method="rk4", device=torch.device("cpu"), use_adjoint=False))
    query = torch.tensor(np.r_[0., days], dtype=torch.float32)
    solutions = odeint(model, torch.from_numpy(np.array(initial, dtype=np.float32, copy=True)), query, method="rk4",
                       options={"grid_constructor": solver.grid_constructor})
    return solutions[1:].detach().cpu().numpy().reshape(len(days), -1)


def _scan(clean, tissue, shape, spacing, rng):
    """The target's acquisition (benchmark_data._segmented_acquisition) applied to one member's clean fields."""
    noise_sd = NOISE_LEVEL * float(clean.max() - clean.min())
    rule = bd.detection_rule(tissue, spacing, noise_sd)
    scans, lesion_voxels, false_positives = np.empty_like(clean), [], []
    for j in range(len(clean)):
        noise = rng.standard_normal(int(tissue.sum()))
        scans[j], lesion = bd.cellularity_map(bd.noisy_scan(clean[j], noise, tissue, noise_sd), tissue, shape, rule)
        lesion_voxels.append(int(lesion.sum()))
        false_positives.append(int(np.sum(lesion & (clean[j] <= 0.))))
    return scans, {"noise_sd": noise_sd, "detection_threshold": rule["threshold"], "lesion_voxels": lesion_voxels,
                   "false_positive_voxels": false_positives,
                   "scan_relative_error_percent": 100 * float(np.linalg.norm(scans - clean) / np.linalg.norm(clean))}


def build(log=print):
    """Simulate, scan and pool the cohort (both protocols), with the clone and input checks; cached."""
    if record_path().exists():
        return bd.read_json(record_path())
    started = time.monotonic()
    target = CASES[next(iter(COHORT_SOURCES))]
    for name in COHORT_CASES:
        check_target(CASES[name])
    history, geometry = bd.chemo_history(target, log)
    tissue = np.asarray(geometry["breast_mask"], dtype=bool).reshape(-1)
    shape = tuple(int(n) for n in geometry["grid_shape"])
    volume = float(np.prod(geometry["spacing"]))
    initial = np.asarray(history[:, 0]).reshape(shape)
    days = np.arange(1., COHORT_DESIGN["span"][1] + 1.)
    if not np.array_equal(bd.SOURCE_KNOTS[2 * days.astype(int)], days):
        raise RuntimeError("The cached history's knots are not on half days.")

    log("Clone check: the member solver with the twin's parameters and the planned course ...")
    clone = solve_member([TUTORIAL_TWIN[name] for name in PARAMETERS], np.ones(len(PATIENT_DOSE_DAYS)), initial, days)
    cached = np.asarray(history[:, 2 * days.astype(int)]).T
    clone_check = {"days": [float(days[0]), float(days[-1])], "max_abs_difference": float(np.abs(clone - cached).max()),
                   "bitwise_equal": bool(np.array_equal(clone, cached))}
    del clone, cached
    if clone_check["max_abs_difference"] > 1e-6:
        raise RuntimeError(f"The member solver does not reproduce the target's history: {clone_check}")
    log(f"  {clone_check}")

    times, values = input_table(np.ones(len(PATIENT_DOSE_DAYS)))
    reference = bd._training_inputs(target)
    if not np.allclose(times, reference["input_times"], rtol=0., atol=1e-12):
        raise RuntimeError("The member input grid is not the target's.")
    input_check = {"max_abs_difference": float(np.abs(values - reference["input_values"]).max()),
                   "max_value": float(np.abs(reference["input_values"]).max())}
    if input_check["max_abs_difference"] > 1e-9 * input_check["max_value"]:
        raise RuntimeError(f"The member inputs do not reproduce the target's exposure table: {input_check}")

    parameters = member_parameters()
    protocols = COHORT_DESIGN["strides"]
    scans = {protocol: [] for protocol in protocols}
    cleans = {protocol: [] for protocol in protocols}
    members = []
    for member in range(COHORT_DESIGN["members"]):
        log(f"Member {member}: k, D, sensitivity = {np.round(parameters['values'][member], 4).tolist()}, "
            f"doses {dose_factors(member).tolist()}")
        tick = time.monotonic()
        fields = solve_member(parameters["values"][member], dose_factors(member), initial, days)
        record = {"member": member, "parameters": dict(zip(PARAMETERS, parameters["values"][member])),
                  "z": parameters["member_z"][member] + parameters["centre_z"],
                  "dose_factors": dose_factors(member), "solve_seconds": time.monotonic() - tick,
                  "burden_day": {f"{day:g}": volume * float(fields[int(day) - 1].sum()) for day in (5, 35, 57, 84, 119)}}
        for protocol, stride in protocols.items():
            taken = scan_days(member, protocol)
            clean = fields[taken.astype(int) - 1].astype(float)
            rng = np.random.default_rng([COHORT_DESIGN["parameter_seed"], member, int(stride)])
            noisy, acquisition = _scan(clean, tissue, shape, geometry["spacing"], rng)
            scans[protocol].append(noisy)
            cleans[protocol].append(clean.astype(np.float32))
            record[protocol] = {"scan_days": taken, **acquisition}
        members.append(record)
        del fields

    rank = COHORT_DESIGN["rank"]
    arrays, protocol_records = {}, {}
    for protocol in protocols:
        pooled = np.concatenate(scans[protocol])
        D, shift, singular_values = bd.observed_pod(pooled, PODSettings(rank, "cohort_training", "none"))
        noise_sd = max(member[protocol]["noise_sd"] for member in members)
        lesion_voxels = [n for member in members for n in member[protocol]["lesion_voxels"]]
        threshold = bd.noise_threshold(noise_sd, lesion_voxels)
        resolved = int(np.sum(singular_values > threshold))
        if rank > resolved:
            raise ValueError(f"The cohort's {protocol} scans resolve only {resolved} modes above the noise "
                             f"threshold {threshold:.3g}; the declared rank is {rank}.")
        smaller, _, _ = bd.observed_pod(pooled, PODSettings(rank - 1, "cohort_training", "none"))
        nested = float(np.abs(smaller - D[:, :rank - 1]).max())
        if nested > 1e-8:
            raise RuntimeError(f"The rank-{rank - 1} cohort basis is not the first columns of rank {rank}: {nested}")
        bd.write_npz(basis_path(protocol), physical_matrix=D, physical_shift=shift, singular_values=singular_values)
        energy = singular_values**2
        protocol_records[protocol] = {
            "scans": int(len(pooled)), "singular_values": singular_values[:12],
            "singular_value_threshold": threshold, "modes_above_threshold": resolved,
            "threshold_rule": "Gavish-Donoho, the largest member noise level, n = mean lesion voxels of all scans",
            "cumulative_energy": (np.cumsum(energy) / energy.sum())[:6],
            "nested_rank_max_abs_difference": nested}
        for member, (noisy, clean) in enumerate(zip(scans[protocol], cleans[protocol])):
            y, clean_q = D.T @ noisy.T, D.T @ clean.astype(float).T
            arrays[f"{protocol}_m{member}_t"] = scan_days(member, protocol)
            arrays[f"{protocol}_m{member}_y"] = y
            arrays[f"{protocol}_m{member}_clean_q"] = clean_q
            members[member][protocol]["captured_scan_energy"] = float(np.sum(y**2) / np.sum(noisy**2))
            members[member][protocol]["captured_clean_energy"] = float(np.sum(clean_q**2)
                                                                        / np.sum(clean.astype(float)**2))
        del pooled
        scans[protocol] = cleans[protocol] = None
    for member in range(COHORT_DESIGN["members"]):
        arrays[f"m{member}_input_values"] = input_table(dose_factors(member))[1]
    arrays["input_times"] = times
    bd.write_npz(members_path(), **arrays)
    record = {
        "design": design(), "design_fingerprint": design_fingerprint(),
        "population": {key: value for key, value in parameters.items() if key != "values"},
        "members": members, "protocols": protocol_records, "clone_check": clone_check, "input_check": input_check,
        "files": {**{f"{protocol}_basis": bd.file_record(basis_path(protocol)) for protocol in protocols},
                  "members": bd.file_record(members_path())},
        "runtime_seconds": time.monotonic() - started}
    bd.write_json(record_path(), record)
    log(f"Cohort built in {record['runtime_seconds']:.0f} s: {record_path()}")
    return record


def _built():
    if not record_path().exists():
        raise FileNotFoundError(f"The virtual cohort is not built ({record_path()}); run "
                                "`python benchmark_cohort.py build` first.")
    record = bd.read_json(record_path())
    if record["design_fingerprint"] != design_fingerprint():
        raise RuntimeError("The cached cohort was built from a different design.")
    for name, expected in record["files"].items():
        if bd.digest(expected["path"]) != expected["sha256"]:
            raise RuntimeError(f"The cohort file {expected['path']} changed after it was built.")
    return record


def cohort_basis(case, pod):
    """The cohort basis for a target case: decoder (first pod.rank modes), zero shift and its record."""
    check_target(case)
    record = _built()
    stored = record["protocols"][case.cohort]
    if pod.rank > COHORT_DESIGN["rank"] or pod.centering != "none":
        raise ValueError("The cohort basis is uncentred, at most rank "
                         f"{COHORT_DESIGN['rank']}.")
    basis = bd.read_npz(basis_path(case.cohort))
    D = basis["physical_matrix"][:, :pod.rank]
    return D, np.zeros(len(D)), {
        **record["files"][f"{case.cohort}_basis"], "protocol": case.cohort, "rank_used": pod.rank,
        "stored_rank": COHORT_DESIGN["rank"], "design_fingerprint": record["design_fingerprint"],
        "pooled_scans": stored["scans"], "modes_above_threshold": stored["modes_above_threshold"],
        "singular_value_threshold": stored["singular_value_threshold"]}


def load_members(protocol, rank):
    """Each member's scan days, reduced noisy scans and clean coefficients (first rank modes) and input table."""
    record = _built()
    arrays = bd.read_npz(members_path())
    return [{"t_sampled": arrays[f"{protocol}_m{member}_t"], "y": arrays[f"{protocol}_m{member}_y"][:rank],
             "clean_q": arrays[f"{protocol}_m{member}_clean_q"][:rank], "input_times": arrays["input_times"],
             "input_values": arrays[f"m{member}_input_values"], "dose_factors": dose_factors(member)}
            for member in range(len(record["members"]))]


# -----------------------------------------------------------------------------
# Population operator
# -----------------------------------------------------------------------------
def fit_directory(case, pod=None, root=None):
    pod = case.pod if pod is None else pod
    cfg = asdict(production_config(case, pod))
    config_hash = hashlib.sha256(json.dumps(bd.json_value(cfg), sort_keys=True).encode()).hexdigest()[:12]
    return (Path(root or bd.RESULTS) / "_cohorts" / design_fingerprint()[:12]
            / f"{case.cohort}_r{pod.rank}_{config_hash}")


def _trajectory(member, cfg):
    import jax.numpy as jnp
    t = np.asarray(member["t_sampled"], dtype=float)
    query = np.linspace(t[0], t[-1], cfg.num_eval_points)
    # As the target's native inputs: the float32 JAX interpolant of the tabulated exposure on the fit grid.
    inputs = np.asarray(jnp.interp(jnp.asarray(query), jnp.asarray(member["input_times"]),
                                   jnp.asarray(member["input_values"])), dtype=float)[None, :]
    trajectory = {"t_sampled": t, "snapshots_comp": np.asarray(member["y"], dtype=float), "inputs_eval": inputs}
    if cfg.gp_input_trend:
        trajectory["input_table"] = dict(times=np.asarray(member["input_times"], dtype=float),
                                         values=np.asarray(member["input_values"], dtype=float))
    return trajectory


def _in_sample(O_canonical, members, burden_weights):
    """Each member's scans replayed by the population operator from its first scan: relative error (%) of the
    reduced coefficients against the member's clean ones, and the last scan's projected burden ratio."""
    import benchmark_evaluation as be
    out = []
    for member in members:
        forcing = be.InputTable(member["input_times"], member["input_values"])
        field = be.ProductionField(O_canonical, forcing)
        t = np.asarray(member["t_sampled"], dtype=float)
        cuts = np.unique(np.r_[forcing.knots(t[0], t[-1]), PATIENT_DOSE_DAYS])
        status, values = be.integrate_rhs(field.rhs, member["y"][:, 0], t, cuts[(cuts > t[0]) & (cuts < t[-1])])
        clean = np.asarray(member["clean_q"], dtype=float)
        error = (100 * float(np.linalg.norm(values - clean) / np.linalg.norm(clean))
                 if np.isfinite(values).all() else None)
        out.append({"status": status["status"], "relative_error_percent": error,
                    "last_scan_burden_ratio": (float(burden_weights @ values[:, -1] / (burden_weights @ clean[:, -1]))
                                               if error is not None else None)})
    return out


def fit_cohort(case, pod=None, root=None, *, steps=None, log=print):
    """The production fit, the target case's configuration, to all members' reduced scans (one trajectory each)."""
    import benchmark_models as bm
    pod = case.pod if pod is None else pod
    check_target(case)
    if case.structure != "free":
        raise ValueError("The population operator is a free ROM.")
    folder = fit_directory(case, pod, root)
    cfg = production_config(case, pod)
    total = bm._checked_steps(steps, cfg.num_steps)
    decoder, _, basis = cohort_basis(case, pod)
    burden_weights = decoder.sum(axis=0)
    members_sha = _built()["files"]["members"]["sha256"]
    requested = {"planned_updates": total, "production_config": asdict(cfg), "cohort_basis_sha256": basis["sha256"],
                 "members_sha256": members_sha, "rank": pod.rank}
    bm._same_request(folder / "fit_start.json", requested)
    if (folder / "result.json").exists():
        return bd.read_json(folder / "result.json")
    members = load_members(case.cohort, pod.rank)

    import jax
    import jax.numpy as jnp
    from jax import random
    import numpyro
    from core.weakform_opinf import build_model
    from core.weakform_opinf.pipeline import _stack_theta
    bm._require_float32()
    numpyro.set_platform("cpu")
    np.random.seed(cfg.seed)
    started = time.monotonic()
    log(f"Population operator: {case.cohort} protocol, rank {pod.rank}, {len(members)} members -> {folder}")
    trajectories = [_trajectory(member, cfg) for member in members]
    rom = bm._production_structure(case, cfg, members[0])
    model, conditional, _, priors = build_model(rom, trajectories, cfg)
    gamma2 = priors["gamma2"]
    guide, svi, state, losses, posterior_key, operator_key = bm._run_svi(model, cfg, total, folder, gamma2, {
        "cohort": case.cohort, "rank": pod.rank, "target_case": case.name, "members": len(members),
        "production_config": asdict(cfg), "planned_updates": total, "cohort_basis_sha256": basis["sha256"],
        "members_sha256": members_sha, "inputs": "The cohort members' reduced noisy scans and exposure tables only.",
        "thread_environment": benchmark_environment.current()}, log)

    r = cfg.num_modes
    params = svi.get_params(state)
    posterior = guide.sample_posterior(posterior_key, params, sample_shape=(OPERATOR_DRAWS,), gamma2=gamma2)
    theta = _stack_theta(posterior, len(members), r)
    tau = jnp.exp(jnp.asarray(posterior["log_tau_block"]))
    keys = random.split(operator_key, OPERATOR_DRAWS)

    @jax.jit
    def draw_operator(theta_s, key, tau_s):
        mean, root_s = conditional(theta_s, gamma2, None, tau_s)
        return mean + jnp.einsum("ijk,ik->ij", root_s, random.normal(key, shape=mean.shape))

    samples = np.stack([np.asarray(draw_operator(tuple(value[i] for value in theta), keys[i], tau[i]))
                        for i in range(OPERATOR_DRAWS)])
    mean_hypers = tuple(np.asarray(value).mean(0) for value in theta)
    mean_tau = np.asarray(tau).mean(0)
    point, root_point = conditional(tuple(jnp.asarray(value) for value in mean_hypers), gamma2, None,
                                    jnp.asarray(mean_tau))
    point = np.asarray(point)
    canonical = bm.canonical_layout({"O_point": point, "O_samples": samples, "conditional_root": np.asarray(root_point)},
                                    cfg.operators, r)
    arrays = {"O_point_fit": point, "O_samples_fit": samples, "O_point": canonical["O_point"],
              "O_samples": canonical["O_samples"], "mean_tau_block": mean_tau,
              "mean_lengthscale": mean_hypers[0], "mean_variance": mean_hypers[1], "mean_noise": mean_hypers[2]}
    if len(theta) == 4:
        arrays["mean_trend_scale"] = mean_hypers[3]
    bd.write_npz(folder / "operators.npz", **arrays)
    in_sample = _in_sample(canonical["O_point"], members, burden_weights)
    complete = bool(np.isfinite(losses).all() and np.isfinite(point).all())
    result = {
        "status": "complete" if complete else "nonfinite_population_fit", "cohort": case.cohort, "rank": pod.rank,
        "target_case": case.name, "members": len(members), "steps": total, "recipe_steps": cfg.num_steps,
        "reduced_steps": total != cfg.num_steps, "operators": cfg.operators,
        "operators_sha256": bd.digest(folder / "operators.npz"), "cohort_basis_sha256": basis["sha256"],
        "members_sha256": members_sha, "production_config": asdict(cfg), "prior_info": priors,
        "point_definition": "Conditional operator mean at the mean of 500 GP/tau draws (fit layout: O_point_fit).",
        "O_point_fit": point, "mean_tau_block": mean_tau, "in_sample_members": in_sample,
        "final_loss": float(losses[-1]), "runtime_seconds": time.monotonic() - started,
        "thread_environment": benchmark_environment.current()}
    bd.write_json(folder / "result.json", result)
    log(f"  done in {result['runtime_seconds']:.0f} s; in-sample member errors "
        f"{[m['relative_error_percent'] for m in in_sample]}")
    return result


def prior_record(case, pod=None, root=None):
    """The population operator a -cohort-prior fit is centred on (file, digest, steps); it must exist."""
    folder = fit_directory(case, pod, root)
    if not (folder / "result.json").exists():
        raise FileNotFoundError(f"No population operator at {folder}; run "
                                f"`python benchmark_cohort.py fit {case.name}` first (same --output-root).")
    result = bd.read_json(folder / "result.json")
    if result["status"] != "complete":
        raise RuntimeError(f"The population fit at {folder} did not complete ({result['status']}).")
    path = folder / "operators.npz"
    if bd.digest(path) != result["operators_sha256"]:
        raise RuntimeError(f"{path} changed after the population fit.")
    return {"path": str(path), "sha256": result["operators_sha256"], "steps": result["steps"],
            "reduced_steps": result["reduced_steps"], "cohort_basis_sha256": result["cohort_basis_sha256"],
            "members_sha256": result["members_sha256"]}


def prior_mean(case, pod, root, metadata, record, *, full_run):
    """The prior centre M (fit layout) for a target acquisition: the population operator on the same basis."""
    if record != prior_record(case, pod, root):
        raise RuntimeError("The population operator changed since this fit was requested.")
    if metadata["sources"]["basis"]["sha256"] != record["cohort_basis_sha256"]:
        raise ValueError("The population operator was fitted on a different basis than this acquisition's.")
    if full_run and record["reduced_steps"]:
        raise ValueError("A full-length fit needs the full-length population operator.")
    return np.asarray(bd.read_npz(record["path"])["O_point_fit"], dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="Simulate, scan and pool the virtual cohort (cached).")
    fit = commands.add_parser("fit", help="Fit the population operator for a cohort case's protocol and rank.")
    fit.add_argument("case", choices=COHORT_CASES)
    fit.add_argument("--steps", type=int, help="SVI updates (default: the production recipe's).")
    fit.add_argument("--output-root", type=Path, help=f"Results root (default: {bd.RESULTS}).")
    args = parser.parse_args(argv)
    if args.command == "build":
        build()
    else:
        fit_cohort(get_case(args.case), root=args.output_root, steps=args.steps)


if __name__ == "__main__":
    main()
