"""Failure-preserving forecasts and full-field metrics for the tumor benchmarks.

Both methods are integrated by the same float64 DOP853 policy on the same
query grid, and scored against the same evaluation-only references. Nonfinite
or censored trajectories stay in the denominators: an incomplete window is
reported as incomplete, never scored on a surviving prefix.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp

import benchmark_data as bd

DRAW_INDICES = np.linspace(0, 499, 64, dtype=int)
FIELD_TOLERANCE = 25.
TRAINING_REPLAY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Policy:
    rtol: float = 1e-8
    atol: float = 1e-10
    maximum_rhs_calls: int = 200000
    maximum_seconds: float = 120.
    steps_per_horizon: int = 400
    scaled_state_threshold: float = 1e6


POLICY = Policy()


class NumericalStop(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def integrate_rhs(rhs, initial, query, boundaries, policy=POLICY, *, event=None):
    """DOP853 on each interval between forced boundaries, keeping only accepted solution values."""
    query = np.asarray(query, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if query.ndim != 1 or len(query) < 2 or np.any(np.diff(query) <= 0):
        raise ValueError("Integration query must be strictly increasing")
    output = np.full((len(initial), len(query)), np.nan)
    output[:, 0] = initial
    if not np.isfinite(initial).all():
        return dict(status="nonfinite_initial_state", rhs_calls=0, completed_through=None), output
    start, end = query[0], query[-1]
    cuts = np.unique(np.r_[start, end, boundaries])
    if cuts[0] != start or cuts[-1] != end:
        raise ValueError("Integration boundaries outside query interval")
    started = time.monotonic()
    deadline = started + policy.maximum_seconds
    calls, completed, state = 0, start, initial.copy()
    status, message, event_time, warning_count = "ok", None, None, 0
    if event is not None and event(start, state) <= 0:
        return dict(status="threshold_censored", rhs_calls=0, completed_through=float(start),
                    event_time=float(start)), output

    def checked_rhs(t, y):
        nonlocal calls
        calls += 1
        if calls > policy.maximum_rhs_calls:
            raise NumericalStop("rhs_budget_exceeded", "Declared per-trajectory RHS budget exhausted")
        if time.monotonic() >= deadline:
            raise NumericalStop("wall_time_budget_exceeded", "Declared per-trajectory time budget exhausted")
        if not np.isfinite(y).all():
            raise NumericalStop("nonfinite_rhs_state", "Nonfinite internal RHS state")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            value = np.asarray(rhs(t, y))
        if value.shape != initial.shape or not np.isfinite(value).all():
            raise NumericalStop("nonfinite_or_invalid_rhs", "Invalid RHS returned")
        return value

    for left, right in zip(cuts[:-1], cuts[1:]):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                solution = solve_ivp(checked_rhs, (left, right), state, method="DOP853", rtol=policy.rtol,
                                     atol=policy.atol, max_step=(end - start) / policy.steps_per_horizon,
                                     dense_output=True, events=event)
            except NumericalStop as exc:
                status, message, solution = exc.status, str(exc), None
            except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError) as exc:
                status, message, solution = "numerical_integration_failure", f"{type(exc).__name__}: {exc}", None
        warning_count += len(captured)
        if solution is None:
            break
        completed = float(solution.t[-1])
        mask = (query >= left) & (query <= completed)
        if mask.any():
            output[:, mask] = solution.sol(query[mask])
            if not np.isfinite(output[:, mask]).all():
                status, message = "nonfinite_prediction", "Nonfinite dense-output prediction"
                break
        state = solution.y[:, -1]
        if solution.status == 1:
            status, event_time = "threshold_censored", float(solution.t_events[0][0])
            message = "Declared safety threshold reached; not a finite-time blow-up claim"
            break
        if not solution.success:
            status, message = "integration_failed", solution.message
            break
    return dict(status=status, message=message, rhs_calls=calls, completed_through=float(completed),
                event_time=event_time, warnings=warning_count, solver="DOP853",
                wall_seconds=time.monotonic() - started), output


# -----------------------------------------------------------------------------
# Vector fields
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class InputTable:
    """Piecewise-linear exposure alpha(t); extrapolation is an error."""
    times: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        times, values = np.asarray(self.times, dtype=float), np.asarray(self.values, dtype=float)
        if (times.ndim != 1 or values.shape != times.shape or np.any(np.diff(times) <= 0)
                or not np.isfinite(values).all()):
            raise ValueError("Invalid input table")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    def __call__(self, t):
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError("Input-table extrapolation is forbidden")
        return np.interp(t, self.times, self.values)

    def knots(self, start, end):
        return self.times[(self.times > start) & (self.times < end)]


class ProductionField:
    """dq/dt = c + A q (untreated) or c + A q + alpha(t) (B + N q) (chemo)."""

    def __init__(self, operator, forcing=None):
        self.operator, self.forcing = np.asarray(operator, dtype=float), forcing
        r = len(self.operator)
        expected = (r, r + 1) if forcing is None else (r, 2 * r + 2)
        if self.operator.shape != expected or not np.isfinite(self.operator).all():
            raise ValueError("Invalid canonical production operator")

    def rhs(self, t, q):
        O = self.operator
        if self.forcing is None:
            return O @ np.r_[1., q]
        r, u = len(q), self.forcing(t)
        return O[:, 0] + O[:, 1:1 + r] @ q + u * (O[:, 1 + r] + O[:, 2 + r:] @ q)


class NeuralField:
    """The trained tanh MLP on [q] (untreated) or [q, alpha(t)] (chemo), in float64."""

    def __init__(self, arrays, forcing=None):
        self.forcing = forcing
        self.layers = [(np.asarray(arrays[f"weight_{i}"], dtype=float), np.asarray(arrays[f"bias_{i}"], dtype=float))
                       for i in range(4)]
        r = len(self.layers[-1][1])
        sizes = (r + (forcing is not None), 128, 128, 128, r)
        for i, (weight, bias) in enumerate(self.layers):
            if weight.shape != (sizes[i + 1], sizes[i]) or bias.shape != (sizes[i + 1],):
                raise ValueError("Neural ODE architecture differs from the benchmark baseline")
            if not (np.isfinite(weight).all() and np.isfinite(bias).all()):
                raise ValueError("Nonfinite Neural ODE weights")

    def rhs(self, t, q):
        value = q if self.forcing is None else np.r_[q, self.forcing(t)]
        for weight, bias in self.layers[:-1]:
            value = np.tanh(weight @ value + bias)
        weight, bias = self.layers[-1]
        return weight @ value + bias


def forcing_for(case, reference):
    return InputTable(reference["input_times"], reference["input_values"]) if case.dose_days else None


def forecast(field, initial, observations, case, grid, *, policy=POLICY):
    """Predictions on the evaluation grid and at the float32-rounded training times."""
    times, sampled = grid.times, np.asarray(observations["t_sampled"])
    rounded = sampled.astype(np.float32).astype(float)
    if field is None:
        blank = np.full((len(initial), len(times)), np.nan)
        return {"status": "unavailable_fit", "rhs_calls": 0}, {
            "prediction": blank, "training_prediction": np.full((len(initial), len(sampled)), np.nan)}
    query = np.unique(np.r_[times, sampled, rounded, grid.boundaries])
    start, end = times[0], times[-1]
    cuts = np.asarray(grid.boundaries, dtype=float)
    if field.forcing is not None:
        cuts = np.unique(np.r_[field.forcing.knots(start, end), observations["dose_days"], cuts])
        cuts = cuts[(cuts > start) & (cuts < end)]
    scale = np.asarray(observations["state_safety_scale"], dtype=float)

    def event(t, q):
        return policy.scaled_state_threshold - np.max(np.abs(q / scale))

    event.terminal, event.direction = True, -1
    status, values = integrate_rhs(field.rhs, initial, query, cuts, policy, event=event)
    return status, {"prediction": values[:, np.searchsorted(query, times)],
                    "training_prediction": values[:, np.searchsorted(query, rounded)]}


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    times: np.ndarray
    reduced_truth: np.ndarray
    physical_R: np.ndarray
    physical_projection: np.ndarray
    physical_residual2: np.ndarray
    physical_norm2: np.ndarray
    burden_weights: np.ndarray
    burden_offset: float
    burden_truth: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name != "burden_offset":
                object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        r, n = self.reduced_truth.shape
        shapes = {"times": (n,), "physical_R": (r, r), "physical_projection": (r, n),
                  "physical_residual2": (n,), "physical_norm2": (n,), "burden_weights": (r,), "burden_truth": (n,)}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid target geometry: {name}")
        if np.any(self.physical_residual2 < 0) or np.any(self.physical_norm2 < 0):
            raise ValueError("Physical reference energies must be nonnegative")


def field_target(reference, observations):
    return Target(reference["t_pred"], reference["true_comp"], reference["physical_R"],
                  reference["physical_projection"], reference["physical_residual2"], reference["physical_norm2"],
                  observations["burden_weights"], float(observations["burden_offset"]), reference["truth_burden"])


def effect_target(reference, observations):
    """Change relative to the unchanged-regimen arm (no affine offset)."""
    return Target(reference["t_pred"], reference["effect_true_comp"], reference["physical_R"],
                  reference["effect_projection"], reference["effect_residual2"], reference["effect_norm2"],
                  observations["burden_weights"], 0., reference["effect_burden"])


def relative_error(numerator, denominator):
    if not np.isfinite([numerator, denominator]).all() or numerator < 0 or denominator <= 0:
        raise ValueError("Relative error requires finite nonnegative error and positive reference energy")
    return float(100 * np.sqrt(numerator / denominator))


def point_window(prediction, target, mask):
    """Reduced, exact full-field and burden relative errors (percent) on one window."""
    q = np.asarray(prediction)
    if q.shape != target.reduced_truth.shape:
        raise ValueError("Prediction shape does not match the target")
    finite = np.isfinite(q[:, mask]).all(axis=0)
    report = {"query_points": int(mask.sum()), "available_points": int(finite.sum()),
              "projection_floor_percent": relative_error(np.sum(target.physical_residual2[mask]),
                                                         np.sum(target.physical_norm2[mask]))}
    if not finite.all():
        return {**report, "status": "incomplete", "reduced_percent": None,
                "physical_percent": None, "burden_percent": None}
    error = q[:, mask] - target.reduced_truth[:, mask]
    physical = target.physical_R @ q[:, mask] - target.physical_projection[:, mask]
    burden = target.burden_weights @ q[:, mask] + target.burden_offset
    truth = target.burden_truth[mask]
    return {
        **report, "status": "complete",
        "reduced_percent": relative_error(np.sum(error**2), np.sum(target.reduced_truth[:, mask]**2)),
        "physical_percent": relative_error(np.sum(physical**2) + np.sum(target.physical_residual2[mask]),
                                           np.sum(target.physical_norm2[mask])),
        "burden_percent": relative_error(np.sum((burden - truth)**2), np.sum(truth**2)),
        "negative_burden_points": int(np.sum(burden < 0)),
    }


def window_scores(prediction, target, grid):
    return {name: point_window(prediction, target, grid.mask(name)) for name in grid.windows}


def strict_median(values):
    """Coordinate-wise median over members, defined only where every member is finite."""
    values = np.asarray(values, dtype=float)
    center = np.full(values.shape[1:], np.nan)
    if len(values):
        valid = np.isfinite(values).all(axis=(0, 1))
        center[:, valid] = np.median(values[:, :, valid], axis=0)
    return center


def burden_band(values, target):
    values = np.asarray(values)
    burden = np.einsum("i,dit->dt", target.burden_weights, values) + target.burden_offset
    valid = np.isfinite(values).all(axis=1) & np.isfinite(burden)
    counts = valid.sum(axis=0)
    quantiles = np.full((3, len(target.times)), np.nan)
    for index in np.flatnonzero(counts):
        quantiles[:, index] = np.quantile(burden[valid[:, index], index], [.05, .5, .95])
    inside = (target.burden_truth >= quantiles[0]) & (target.burden_truth <= quantiles[2]) & (counts == len(values))
    return {"burden": burden, "finite_counts": counts, "quantiles_05_50_95": quantiles,
            "all_finite_and_truth_in_band": inside}


def ensemble_summary(values, target, mask):
    """Every member stays in the denominator; bands are empirical, not calibrated."""
    rows = [point_window(value, target, mask) for value in values]
    complete = [row for row in rows if row["status"] == "complete"]
    band = burden_band(values, target)
    field = [row["physical_percent"] for row in complete]
    return {
        "members": len(rows), "complete": len(complete),
        f"complete_field_le{FIELD_TOLERANCE:g}": sum(value <= FIELD_TOLERANCE for value in field),
        "field_percent_quantiles_05_50_95": np.quantile(field, [.05, .5, .95]).tolist() if field else None,
        "burden_truth_in_complete_band": int(band["all_finite_and_truth_in_band"][mask].sum()),
        "queries": int(mask.sum()),
        "member_field_percent": [row["physical_percent"] for row in rows],
        "member_burden_percent": [row["burden_percent"] for row in rows],
    }, band


def effect_score(changed, control, reference, control_reference, observations, switch):
    """Score the predicted change from the unchanged regimen after the first changed pulse."""
    delta = np.asarray(changed) - np.asarray(control)
    target = effect_target(reference, observations)
    mask = target.times > switch
    score = point_window(delta, target, mask)
    burden = target.burden_weights @ delta
    truth, baseline = reference["effect_burden"], control_reference["truth_burden"]
    meaningful = mask & (abs(truth) > 1e-6 * baseline)
    valid = np.isfinite(delta).all(axis=0) & np.isfinite(burden)
    endpoint = bool(valid[-1] and abs(truth[-1]) > 1e-6 * baseline[-1])
    return {
        **score,
        "meaningful_direction_queries": int(meaningful.sum()),
        "correct_direction_queries": int(np.sum(meaningful & valid & (np.sign(burden) == np.sign(truth))
                                                & (burden != 0.))),
        "true_final_burden_change": float(truth[-1]),
        "predicted_final_burden_change": float(burden[-1]) if valid[-1] else None,
        "true_final_change_percent_of_control": float(100. * truth[-1] / baseline[-1]),
        "final_burden_gain_predicted_over_true": float(burden[-1] / truth[-1]) if endpoint else None,
    }


def effect_ensemble(changed, control, reference, control_reference, observations, switch):
    """Paired draw-by-draw treatment effects (same operator and initial state in both arms)."""
    rows = [effect_score(a, b, reference, control_reference, observations, switch)
            for a, b in zip(changed, control)]
    target = effect_target(reference, observations)
    band = burden_band(np.asarray(changed) - np.asarray(control), target)
    mask = target.times > switch
    complete = [row for row in rows if row["status"] == "complete"]
    gains = [row["final_burden_gain_predicted_over_true"] for row in complete
             if row["final_burden_gain_predicted_over_true"] is not None]
    return {
        "members": len(rows), "complete": len(complete),
        "effect_field_percent_quantiles_05_50_95": (
            np.quantile([row["physical_percent"] for row in complete], [.05, .5, .95]).tolist()
            if complete else None),
        "final_gain_quantiles_05_50_95": np.quantile(gains, [.05, .5, .95]).tolist() if gains else None,
        "truth_in_complete_band_queries": int(band["all_finite_and_truth_in_band"][mask].sum()),
        "effect_queries": int(mask.sum()),
        "not_calibrated_uncertainty": True,
    }, band


def training_replay(forecast_prediction, native_prediction):
    """Relative difference between this float64 replay and the float32 training solve."""
    native = np.asarray(native_prediction, dtype=float)
    ours = np.asarray(forecast_prediction, dtype=float)
    if not (np.isfinite(native).all() and np.isfinite(ours).all()):
        return None
    norm = np.linalg.norm(native)
    return float(np.linalg.norm(ours - native) / norm) if norm > 0 else None


def compact(status):
    keys = ("status", "message", "rhs_calls", "completed_through", "event_time", "wall_seconds")
    return {key: status.get(key) for key in keys if key in status}


def _check_shared_history(predictions, control, times, end):
    """The arms share inputs before the first changed pulse, so their training forecasts must agree."""
    before = times <= end
    for arm, value in predictions.items():
        if not np.array_equal(value[..., before], predictions[control][..., before], equal_nan=True):
            raise AssertionError(f"Arm {arm} diverged from the unchanged regimen during training.")


# -----------------------------------------------------------------------------
# Per-seed evaluations
# -----------------------------------------------------------------------------
def evaluate_production(case, seed, pod=None, root=None, *, draws=True, log=print):
    import benchmark_models as bm
    pod = case.pod if pod is None else pod
    folder = bd.seed_directory(case, seed, pod, root)
    path = folder / "evaluation" / "production.json"
    fitted = bd.read_json(bm.production_directory(case, seed, pod, root) / "result.json")["operators_sha256"]
    if path.exists():
        cached = bd.read_json(path)
        if cached.get("operators_sha256") == fitted and (cached["draws_evaluated"] or not draws):
            return cached
    started = time.monotonic()
    data = bd.load(case, seed, pod, root)
    observations, references = data["observations"], data["references"]
    fit = bm.load_production(case, seed, pod, root)
    grid, arms = bd.grid(case), bd.arms(case)
    control = next(iter(arms))
    usable = fit["result"]["status"] == "complete"
    records, arrays, points, samples = {}, {"times": grid.times}, {}, {}
    for arm in arms:
        forcing = forcing_for(case, references[arm])
        target = field_target(references[arm], observations)
        field = ProductionField(fit["operators"]["O_point"], forcing) if usable else None
        status, point = forecast(field, observations["q0"], observations, case, grid)
        points[arm] = arrays[f"{arm}_point"] = point["prediction"]
        records[arm] = {"point": {"integration": compact(status),
                                  "scores": window_scores(point["prediction"], target, grid)}}
        if draws:
            values, statuses = [], Counter()
            for index in DRAW_INDICES:
                operator, initial = fit["operators"]["O_samples"][index], fit["initial"]["all_draws"][index]
                valid = usable and np.isfinite(operator).all() and np.isfinite(initial).all()
                status, draw = forecast(ProductionField(operator, forcing) if valid else None,
                                        initial if valid else np.full(len(initial), np.nan), observations, case, grid)
                statuses[status["status"]] += 1
                values.append(draw["prediction"])
            samples[arm] = arrays[f"{arm}_draws"] = np.asarray(values)
            summary, band = ensemble_summary(samples[arm], target, grid.mask(grid.headline))
            records[arm]["draws"] = {**summary, "integration_status_counts": dict(statuses)}
            arrays[f"{arm}_burden_band"] = band["quantiles_05_50_95"]
        log(f"  production {arm}: {grid.headline} field error "
            f"{records[arm]['point']['scores'][grid.headline]['physical_percent']}")
    if case.dose_days:
        _check_shared_history(points, control, grid.times, case.training_span[1])
        switch = bd.first_future_dose(case)
        for arm in arms:
            if arm == control:
                continue
            records[arm]["effect"] = {"point": effect_score(
                points[arm], points[control], references[arm], references[control], observations, switch)}
            if draws:
                records[arm]["effect"]["draws"], band = effect_ensemble(
                    samples[arm], samples[control], references[arm], references[control], observations, switch)
                arrays[f"{arm}_effect_band"] = band["quantiles_05_50_95"]
    arrays["draw_indices"] = DRAW_INDICES
    bd.write_npz(folder / "evaluation" / "production.npz", **arrays)
    result = {
        "method": "production", "case": case.name, "seed": seed, "pod_tag": bd.pod_tag(pod),
        "observation": case.observation,
        "fit_status": fit["result"]["status"], "fit_steps": fit["result"]["steps"],
        "operators_sha256": fit["result"]["operators_sha256"],
        "initial_conditions_sha256": fit["result"]["initial_conditions_sha256"],
        "headline_window": grid.headline, "arms": records, "draws_evaluated": len(DRAW_INDICES) if draws else 0,
        "point_definition": "Conditional operator mean at the mean GP/tau hyperparameters, observed initial state.",
        "draw_definition": "Posterior operator draws paired with native initial-state draws; not calibrated UQ.",
        "policy": asdict(POLICY), "runtime_seconds": time.monotonic() - started,
    }
    bd.write_json(path, result)
    return result


def evaluate_node(case, seed, pod=None, root=None, *, log=print):
    import benchmark_models as bm
    pod = case.pod if pod is None else pod
    folder = bd.seed_directory(case, seed, pod, root)
    path = folder / "evaluation" / "neural_ode.json"
    trained = bd.digest(bm.node_directory(case, seed, pod, root) / "summary.json")
    if path.exists():
        cached = bd.read_json(path)
        if cached.get("node_summary_sha256") == trained:
            return cached
    started = time.monotonic()
    data = bd.load(case, seed, pod, root)
    observations, references = data["observations"], data["references"]
    fit = bm.load_node(case, seed, pod, root)
    kept = np.asarray(fit["filter"]["kept_indices"] or [], dtype=int)
    grid, arms = bd.grid(case), bd.arms(case)
    control = next(iter(arms))
    records, arrays, centers = {}, {"times": grid.times, "kept_indices": kept}, {}
    replay = []
    for arm in arms:
        forcing = forcing_for(case, references[arm])
        target = field_target(references[arm], observations)
        values, statuses = [], Counter()
        for index, member in enumerate(fit["members"]):
            usable = member["result"]["status"] == "complete"
            status, member_forecast = forecast(NeuralField(member["weights"], forcing) if usable else None,
                                               observations["q0"], observations, case, grid)
            statuses[status["status"]] += 1
            values.append(member_forecast["prediction"])
            if arm == control:
                difference = training_replay(member_forecast["training_prediction"], member["training_prediction"])
                replay.append({"member": index, "relative_difference": difference,
                               "qualified": bool(usable and member["result"]["final_training_solver_success"]
                                                 and difference is not None
                                                 and difference <= TRAINING_REPLAY_TOLERANCE)})
        values = np.asarray(values)
        arrays[f"{arm}_members"] = values
        centers[arm] = {"all20_median": strict_median(values),
                        "filtered_median": strict_median(values[kept]) if len(kept)
                        else np.full(values.shape[1:], np.nan)}
        for name, value in centers[arm].items():
            arrays[f"{arm}_{name}"] = value
        summary, band = ensemble_summary(values, target, grid.mask(grid.headline))
        arrays[f"{arm}_burden_band"] = band["quantiles_05_50_95"]
        records[arm] = {"centers": {name: window_scores(value, target, grid) for name, value in centers[arm].items()},
                        "members": {**summary, "integration_status_counts": dict(statuses)}}
        log(f"  neural ODE {arm}: {grid.headline} field error "
            + ", ".join(f"{name} {records[arm]['centers'][name][grid.headline]['physical_percent']}"
                        for name in centers[arm]))
    if case.dose_days:
        _check_shared_history({arm: arrays[f"{arm}_members"] for arm in arms}, control, grid.times,
                              case.training_span[1])
        switch = bd.first_future_dose(case)
        for arm in arms:
            if arm != control:
                records[arm]["effect"] = {name: effect_score(
                    centers[arm][name], centers[control][name], references[arm], references[control],
                    observations, switch) for name in centers[arm]}
    bd.write_npz(folder / "evaluation" / "neural_ode.npz", **arrays)
    result = {
        "method": "neural_ode", "case": case.name, "seed": seed, "pod_tag": bd.pod_tag(pod),
        "observation": case.observation,
        "headline_window": grid.headline, "headline_center": headline_node_center(case),
        "filter": fit["filter"], "members": len(fit["members"]), "node_summary_sha256": trained,
        "node_source": fit["summary"].get("source"),
        "completed_members": sum(m["result"]["status"] == "complete" for m in fit["members"]),
        "training_replay": replay,
        "training_replay_qualified": sum(row["qualified"] for row in replay),
        "arms": records, "policy": asdict(POLICY), "runtime_seconds": time.monotonic() - started,
    }
    bd.write_json(path, result)
    return result


def headline_node_center(case):
    """Each baseline keeps its original ensemble recipe: the chemo script filters loss outliers."""
    return "filtered_median" if case.dose_days else "all20_median"
