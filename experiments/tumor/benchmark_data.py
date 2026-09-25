"""Shared data for the three tumor benchmarks.

TumorTwin sources are cached under ``data/benchmarks`` and reused. Each
acquisition stores the observations handed to *both* methods, plus
evaluation-only reference geometry for every dose arm, under
``results/benchmarks/<case>/<tag>/seed<N>/data``. The tag is
``segmented_<pod>`` for the reported segmented scans (see benchmark_cases.py)
and ``<pod>`` for the earlier oracle-masked design. POD fitting never uses a
field after the end of training, and model fitting never reads a forecast
reference. Segmented acquisitions refuse a POD rank that would include modes
below the scan-noise threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
for _path in (str(ROOT), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmark_cases import (  # noqa: E402
    CASES, LEGACY_OBSERVATION, OBSERVATION_MODELS, BenchmarkCase, PODSettings, describe, get_case)

TUMORTWIN = ROOT.parent / "TumorTwin"
PATIENT = "TNBC_demo_001"
CACHE = SCRIPT_DIR / "data" / "benchmarks"
RESULTS = SCRIPT_DIR / "results" / "benchmarks"
FIGURES = SCRIPT_DIR / "figures" / "benchmarks"
NOMINAL_SOURCE = SCRIPT_DIR / "data" / "TNBC_demo_001_fom_chemo_sparse5_sens0p5.npz"

# Reported acquisitions. Chemo seed 45 was the development acquisition used to
# choose the chemotherapy POD, so the multi-dose confirmation uses 48-50 and the
# single-dose task uses 51-53 (on 48-50 it would repeat the multi-dose 0.5x arm).
# The earlier oracle-masked design keeps its original seeds.
DEFAULT_SEEDS = {
    "untreated-growth": (42, 43, 44),
    "single-dose-chemo": (51, 52, 53),
    "multi-dose-chemo": (48, 49, 50),
}
LEGACY_SEEDS = {**DEFAULT_SEEDS, "single-dose-chemo": (45, 46, 47)}
BLOCK = 2048
GROWTH_BLOCK = 4096
SOURCE_KNOTS = np.arange(0., 120.5, .5)
THETA = 1.
DECAY_RATE = .7
BASE_SENSITIVITY = .5
GROWTH_SOLVER_DT = .5
CHEMO_SOLVER_DT = .25
NOMINAL_SOLVER_DT = .5
INPUT_POINTS = 4001
# Matched chemo PODs use the clean training history at the development
# acquisition's times; the nominal control keeps its earlier 40-time design.
MATCHED_BASIS_SEED = 45
NOMINAL_BASIS_SEED = 42
NOMINAL_BASIS_OBSERVATIONS = 40
# Segmented scans: the lesion is detected on the scan smoothed by a Gaussian of
# this standard deviation, holding the family-wise false-positive rate per scan
# at DETECTION_ALPHA; scan noise is drawn from default_rng([seed, NOISE_STREAM]).
DETECTION_ALPHA = .05
SEGMENTATION_SMOOTHING_MM = 1.
NOISE_STREAM = 1


@dataclass(frozen=True)
class Grid:
    """Evaluation times, scoring windows and forced integration boundaries."""
    times: np.ndarray
    windows: dict
    headline: str
    boundaries: tuple

    def mask(self, name):
        lower, upper, include_left = self.windows[name]
        return ((self.times >= lower) if include_left else (self.times > lower)) & (self.times <= upper)


def grid(case: BenchmarkCase) -> Grid:
    start, end = case.training_span
    if case.dose_days:
        return Grid(
            np.linspace(start, case.prediction_end, 400),
            {"training_5_70": (start, end, True), "forecast_70_90": (end, 90., False),
             "forecast_90_110": (90., case.prediction_end, False),
             "forecast_70_110": (end, case.prediction_end, False)},
            "forecast_70_110", (end, 90.))
    return Grid(
        np.unique(np.r_[np.linspace(start, 120., 400), end, case.prediction_end]),
        {"training_5_60": (start, end, True), "forecast_60_90": (end, case.prediction_end, False),
         "forecast_90_120": (case.prediction_end, 120., False), "forecast_60_120": (end, 120., False)},
        "forecast_60_90", (end, case.prediction_end))


def arms(case: BenchmarkCase) -> dict:
    """Dose arms, with the training-strength control first."""
    if not case.dose_days:
        return {"untreated": None}
    strengths = [case.training_strength] + [s for s in case.future_strengths if s != case.training_strength]
    return {arm_name(s): s for s in strengths}


def arm_name(strength: float) -> str:
    return "strength" + f"{strength:.2f}".replace(".", "p")


def arm_label(strength) -> str:
    return "untreated" if strength is None else f"{strength:g}x"


def first_future_dose(case):
    return min(day for day in case.dose_days if day > case.training_span[1])


def pod_tag(pod: PODSettings) -> str:
    return f"{pod.source.replace('_training', '')}_{pod.centering}_r{pod.rank}"


def acquisition_tag(case, pod=None) -> str:
    """Output tag: the POD recipe, prefixed by the observation model unless it is the earlier design."""
    tag = pod_tag(case.pod if pod is None else pod)
    return tag if case.observation == LEGACY_OBSERVATION else f"{case.observation}_{tag}"


def seed_directory(case, seed, pod=None, root=None) -> Path:
    return Path(root or RESULTS) / case.name / acquisition_tag(case, pod) / f"seed{seed}"


# -----------------------------------------------------------------------------
# Small I/O helpers (atomic writes; no pickles)
# -----------------------------------------------------------------------------
def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _publish(path, writer, suffix):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def write_json(path, value):
    content = (json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return _publish(path, lambda stream: stream.write(content), ".json")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_npz(path, **arrays):
    return _publish(path, lambda stream: np.savez_compressed(stream, **arrays), ".npz")


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_npy(path, array):
    return _publish(path, lambda stream: np.save(stream, array, allow_pickle=False), ".npy")


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            sha.update(block)
    return sha.hexdigest()


def file_record(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def sample_times(span, count, rng):
    times = np.sort(rng.uniform(*span, size=count))
    times[0], times[-1] = span
    return times


def temporal_weights(knots, query):
    """Cubic-spline weights W such that field(query) = field(knots) @ W."""
    return interp1d(knots, np.eye(len(knots)), axis=0, kind="cubic", fill_value="extrapolate")(query).T


def _single_thread_torch():
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch


# -----------------------------------------------------------------------------
# TumorTwin sources (generated once, then memory-mapped)
# -----------------------------------------------------------------------------
def _source(path, generate, log):
    path = Path(path)
    geometry = path.with_name(path.stem + "_geometry.npz")
    if not path.exists() or not geometry.exists():
        log(f"Generating TumorTwin source {path.name} (cached for later runs)...")
        started = time.monotonic()
        snapshots, metadata = generate()
        if not np.isfinite(snapshots).all():
            raise FloatingPointError(f"Nonfinite TumorTwin source: {path.name}")
        write_npz(geometry, **metadata)
        write_npy(path, np.ascontiguousarray(snapshots, dtype=np.float32))
        log(f"  generated in {time.monotonic() - started:.0f} s")
    return np.load(path, mmap_mode="r"), read_npz(geometry)


def _geometry(data):
    return {key: np.asarray(data[key]) for key in ("times_days", "grid_shape", "spacing", "breast_mask")}


def growth_source_path(case):
    return CACHE / f"growth_k{case.growth_rate:.3f}_d{case.diffusion:.3f}_T120_dt{GROWTH_SOLVER_DT:g}.npy"


def growth_source(case, log=print):
    """Untreated 120-day source from day 0 (original generator and solver step)."""
    def generate():
        _single_thread_torch()
        from generate_fom_data import generate_fom_snapshots
        data = generate_fom_snapshots(patient_id=PATIENT, total_days=120., dt_save=.5,
                                      dt_solve=GROWTH_SOLVER_DT, k=case.growth_rate,
                                      d=case.diffusion, theta=THETA)
        np.testing.assert_array_equal(data["times_days"], SOURCE_KNOTS)
        return data["snapshots"], _geometry(data)

    return _source(growth_source_path(case), generate, log)


def _history_stem(case):
    return (f"chemo_k{case.growth_rate:.3f}_d{case.diffusion:.3f}_T120_"
            f"strength{case.training_strength:.2f}_dt{CHEMO_SOLVER_DT:g}")


def chemo_history_path(case):
    return CACHE / (_history_stem(case) + ".npy")


def chemo_history(case, log=print):
    """Chemo source with the training strength at every pulse (days 0-120)."""
    def generate():
        _single_thread_torch()
        from generate_fom_data_chemo import generate_fom_chemo
        data = generate_fom_chemo(
            patient_id=PATIENT, total_days=120., dt_save=.5, dt_solve=CHEMO_SOLVER_DT,
            k=case.growth_rate, d=case.diffusion, theta=THETA, sensitivity=BASE_SENSITIVITY,
            decay_rate=DECAY_RATE, dose_scale=case.training_strength,
            dose_days_override=list(case.dose_days))
        np.testing.assert_array_equal(data["times_days"], SOURCE_KNOTS)
        return data["snapshots"], _geometry(data)

    return _source(chemo_history_path(case), generate, log)


def chemo_branch_path(case, strength):
    return CACHE / f"{_history_stem(case)}_future{strength:.2f}_from{case.training_span[1]:g}.npy"


def chemo_branch(case, strength, log=print):
    """Continue the training-strength PDE from the end of training with new future pulses."""
    history, geometry = chemo_history(case, log)
    start = int(round(case.training_span[1] / .5))

    def generate():
        torch = _single_thread_torch()
        from torchdiffeq import odeint
        import generate_fom_data_chemo  # noqa: F401  (puts TumorTwin on sys.path)
        from tumortwin.models import ReactionDiffusion3D
        from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
        from tumortwin.types import ChemotherapySpecification
        patient = _patient()
        origin = patient.visits[0].time
        end = case.training_span[1]
        past = [d for d in case.dose_days if d <= end]
        future = [d for d in case.dose_days if d > end]
        specs = [ChemotherapySpecification(
            sensitivity=coefficient, decay_rate=DECAY_RATE,
            times=[origin + timedelta(days=day) for day in days], doses=[1.] * len(days))
            for coefficient, days in ((.5 * case.training_strength, past), (.5 * strength, future))]
        model = ReactionDiffusion3D(
            k=torch.tensor(case.growth_rate), d=torch.tensor(case.diffusion), theta=torch.tensor(THETA),
            patient_data=patient, initial_time=origin, chemotherapy_specifications=specs,
            radiotherapy_specification=None, require_grad=False)
        solver = TorchDiffEqSolver(model, TorchDiffEqSolverOptions(
            step_size=timedelta(days=CHEMO_SOLVER_DT), method="rk4",
            device=torch.device("cpu"), use_adjoint=False))
        initial = np.array(history[:, start], copy=True).reshape(tuple(geometry["grid_shape"]))
        # Solver.solve re-zeros the clock; odeint on the original grid keeps absolute days.
        query = torch.tensor(SOURCE_KNOTS[start:], dtype=torch.float32)
        solutions = odeint(model, torch.from_numpy(initial), query, method="rk4",
                           options={"grid_constructor": solver.grid_constructor})
        values = solutions.detach().cpu().numpy().reshape(len(query), -1).T.copy()
        unchanged = SOURCE_KNOTS[start:] < min(future)
        np.testing.assert_array_equal(values[:, unchanged], history[:, start:][:, unchanged])
        return values, {"times_days": SOURCE_KNOTS[start:], "grid_shape": geometry["grid_shape"],
                        "spacing": geometry["spacing"],
                        "effective_pulse_coefficients": np.asarray(case.pulse_coefficients(strength))}

    return _source(chemo_branch_path(case, strength), generate, log)[0]


def _patient():
    import generate_fom_data_chemo  # noqa: F401  (puts TumorTwin on sys.path)
    from tumortwin.types import CropSettings, CropTarget, TNBCPatientData
    folder = TUMORTWIN / "input_files" / PATIENT
    return TNBCPatientData.from_file(folder / f"{PATIENT}.json", image_dir=folder,
                                     crop_settings=CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10))


def _training_inputs(case):
    """Tabulated training-regimen exposure on [5, 110], plus the production fit inputs."""
    path = CACHE / f"inputs_{_history_stem(case)}.npz"
    if not path.exists():
        import config
        patient = _patient()
        amounts = np.asarray([float(item.dose) for item in patient.chemotherapy])
        if not len(amounts) or np.any(amounts <= 0):
            raise ValueError("The patient record has no positive chemotherapy dose amounts.")
        days = np.asarray(case.dose_days)
        cycled = np.asarray([amounts[i % len(amounts)] for i in range(len(days))])
        spec, t0 = config._build_chemo_spec_from_schedule(PATIENT, BASE_SENSITIVITY, DECAY_RATE, days, cycled)
        if not np.array_equal(spec.doses, np.ones(len(days))):
            raise ValueError("TumorTwin no longer normalizes dose amounts; the input table would change.")
        table = config.make_jax_input_func(
            config.chemo_input_func_factory(spec, t0, dose_scale=case.training_strength),
            case.training_span[0], case.prediction_end, n_points=INPUT_POINTS)
        native_times = np.linspace(case.training_span[0], case.training_span[1], 200)
        # The production likelihood consumes the JAX interpolant exactly as the historical adapter did.
        native = np.asarray([float(table(t)[0]) for t in native_times])[None, :]
        write_npz(path, input_times=np.asarray(table.t_grid), input_values=np.asarray(table.alpha_grid),
                  dose_days=days, native_time_eval=native_times, native_inputs_eval=native)
    return read_npz(path)


def chemo_inputs(case, strength):
    """Exposure alpha(t) for one future strength (identical before the first future pulse)."""
    base = _training_inputs(case)
    times = base["input_times"]
    future = sum(np.where(times >= day, np.exp(-DECAY_RATE * (times - day)), 0.)
                 for day in case.dose_days if day > case.training_span[1])
    values = base["input_values"] + .5 * (strength - case.training_strength) * future
    return {"input_times": times, "input_values": values, "dose_days": base["dose_days"]}


# -----------------------------------------------------------------------------
# POD representations shared by both methods
# -----------------------------------------------------------------------------
def chemo_basis(case, pod, log=print):
    """Clean-training chemo POD (the nominal control, or matched to the training regimen)."""
    if pod.source == "nominal_training":
        if pod.centering != "mean":
            raise NotImplementedError("The nominal chemo control is the mean-centered production basis.")
        path = CACHE / f"basis_nominal_mean_r{pod.rank}.npz"
        if not path.exists():
            _write_nominal_basis(path, pod.rank, log)
    elif pod.source == "matched_training":
        path = CACHE / f"basis_{_history_stem(case)}_n{case.observations}_{pod_tag(pod)}.npz"
        if not path.exists():
            _write_matched_basis(path, case, pod, log)
    else:
        raise NotImplementedError(
            "Chemotherapy benchmarks support nominal_training or matched_training POD sources.")
    basis = read_npz(path)
    return basis["physical_matrix"], basis["physical_shift"], path


def _write_nominal_basis(path, rank, log):
    import config
    if NOMINAL_SOURCE.exists():
        with np.load(NOMINAL_SOURCE, allow_pickle=False) as source:
            knots, raw = source["times_days"], source["snapshots"]
    else:
        log("The nominal chemo source is missing; regenerating it with its original generator settings...")
        _single_thread_torch()
        from generate_fom_data_chemo import generate_fom_chemo
        data = generate_fom_chemo(patient_id=PATIENT, total_days=120., dt_save=.5, dt_solve=NOMINAL_SOLVER_DT,
                                  sensitivity=BASE_SENSITIVITY, decay_rate=DECAY_RATE, dose_scale=1.,
                                  dose_days_override=[20., 40., 60., 80., 100.])
        knots, raw = data["times_days"], data["snapshots"]
    np.random.seed(42)
    times = sample_times((5., 70.), NOMINAL_BASIS_OBSERVATIONS, np.random.default_rng(NOMINAL_BASIS_SEED))
    weights = temporal_weights(knots, times)
    clean = np.empty((len(raw), len(times)))
    for left in range(0, len(raw), GROWTH_BLOCK):
        clean[left:left + GROWTH_BLOCK] = np.asarray(raw[left:left + GROWTH_BLOCK], dtype=float) @ weights
    basis = config.Basis(num_vectors=rank)
    basis.fit(clean)
    write_npz(path, physical_matrix=np.asarray(basis.entries), physical_shift=np.asarray(basis.shift_),
              source_training_times=times)


def _write_matched_basis(path, case, pod, log):
    history, _ = chemo_history(case, log)
    start, end = case.training_span
    knots = SOURCE_KNOTS[SOURCE_KNOTS <= end]
    times = sample_times((start, end), case.observations, np.random.default_rng(MATCHED_BASIS_SEED))
    weights = temporal_weights(knots, times)
    count, columns = len(history), len(knots)
    gram = np.zeros((len(times), len(times)))
    for left in range(0, count, BLOCK):
        clean = history[left:left + BLOCK, :columns].astype(float) @ weights
        if pod.centering == "mean":
            clean = clean - clean.mean(axis=1, keepdims=True)
        gram += clean.T @ clean
    # Same eigenproblem truncation as the POD selection study (then keep `rank` columns).
    maximum = max(pod.rank, 6 if pod.centering == "mean" else 4)
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:maximum]
    positive = eigenvalues[order]
    if not np.isfinite(positive).all() or np.any(positive <= 1e-12 * eigenvalues.max()):
        raise ValueError("Declared POD rank is numerically unresolved; refusing to reduce it silently.")
    temporal = vectors[:, order] / np.sqrt(positive)[None, :]
    D, shift = np.empty((count, maximum)), np.zeros(count)
    for left in range(0, count, BLOCK):
        clean = history[left:left + BLOCK, :columns].astype(float) @ weights
        if pod.centering == "mean":
            shift[left:left + BLOCK] = clean.mean(axis=1)
        D[left:left + BLOCK] = (clean - shift[left:left + BLOCK, None]) @ temporal
    D, _ = np.linalg.qr(D, mode="reduced")
    D *= np.sign(D[np.argmax(abs(D), axis=0), np.arange(maximum)])[None]
    np.testing.assert_allclose(D.T @ D, np.eye(maximum), rtol=2e-12, atol=2e-12)
    write_npz(path, physical_matrix=D[:, :pod.rank], physical_shift=shift, source_training_times=times,
              singular_values=np.sqrt(np.maximum(eigenvalues[::-1], 0.)))


# -----------------------------------------------------------------------------
# Acquisitions
# -----------------------------------------------------------------------------
def _chemo_observations(case, seed, history, D, shift):
    rank, count = D.shape[1], case.observations
    start, end = case.training_span
    knots = SOURCE_KNOTS[SOURCE_KNOTS <= end]
    times = sample_times((start, end), count, np.random.default_rng(seed))
    weights = temporal_weights(knots, times)
    columns = len(knots)
    low, high = np.inf, -np.inf
    for left in range(0, len(D), BLOCK):
        clean = history[left:left + BLOCK, :columns].astype(float) @ weights
        low, high = min(low, float(clean.min())), max(high, float(clean.max()))
    scale, threshold = case.noise_level * (high - low), .001 * high
    y, clean_q = np.zeros((rank, count)), np.zeros((rank, count))
    rng, innovations_sha = np.random.RandomState(seed), hashlib.sha256()
    for left in range(0, len(D), BLOCK):
        selected = slice(left, left + BLOCK)
        clean = history[selected, :columns].astype(float) @ weights
        innovations = rng.normal(size=(len(clean), count - 1))
        innovations_sha.update(innovations.tobytes())
        noisy = clean.copy()
        noisy[:, 1:] += scale * innovations * (clean[:, 1:] > threshold)
        y += D[selected].T @ (noisy - shift[selected, None])
        clean_q += D[selected].T @ (clean - shift[selected, None])
    return times, y, clean_q, {"noise_sd": scale, "active_threshold": threshold,
                               "innovations_sha256": innovations_sha.hexdigest(),
                               "latest_source_day_used": float(knots[-1])}


def _interpolate(raw, knots, query):
    out = np.empty((len(raw), len(query)))
    for left in range(0, len(raw), GROWTH_BLOCK):
        out[left:left + GROWTH_BLOCK] = interp1d(
            knots, raw[left:left + GROWTH_BLOCK], axis=1, kind="cubic", fill_value="extrapolate")(query)
    return out


def _growth_observations(case, seed, raw, knots, pod):
    import config
    if pod.source != "observed_training" or pod.centering != "mean":
        raise NotImplementedError("Untreated growth uses the production POD of its noisy training observations.")
    start, end = case.training_span
    rng = np.random.RandomState(seed)
    times = np.sort(rng.uniform(start, end, size=case.observations))
    times[0], times[-1] = start, end
    innovations = rng.standard_normal((len(raw), case.observations - 1))
    # As in the reported study, the PDE state at each observation time is the stored
    # source's all-knot cubic interpolant (the chemo cases stop their spline at day 70).
    clean = _interpolate(raw, knots, times)
    scale, threshold = case.noise_level * (clean.max() - clean.min()), .001 * clean.max()
    noisy = clean.copy()
    noisy[:, 1:] += scale * innovations * (clean[:, 1:] > threshold)
    np.random.seed(42)
    basis = config.Basis(num_vectors=pod.rank)
    basis.fit(noisy)
    y, clean_q = basis.compress(noisy), basis.compress(clean)
    D, shift = np.asarray(basis.entries), np.asarray(basis.shift_)
    return times, y, clean_q, D, shift, {
        "noise_sd": scale, "active_threshold": threshold,
        "innovations_sha256": hashlib.sha256(np.ascontiguousarray(innovations).tobytes()).hexdigest(),
        "basis_cumulative_energy": float(basis.cumulative_energy)}


# -----------------------------------------------------------------------------
# Segmented acquisitions (reported design)
# -----------------------------------------------------------------------------
def detection_rule(tissue, spacing, noise_sd, alpha=DETECTION_ALPHA, smoothing_mm=SEGMENTATION_SMOOTHING_MM):
    """Lesion threshold on the smoothed scan, Bonferroni-controlling false-positive voxels per scan."""
    width = float(smoothing_mm) / np.asarray(spacing, dtype=float)
    radius = [int(4. * w + .5) for w in width]  # gaussian_filter's default truncation
    impulse = np.zeros([2 * r + 1 for r in radius])
    impulse[tuple(radius)] = 1.
    gain = float(np.sqrt(np.sum(gaussian_filter(impulse, width, mode="constant")**2)))
    voxels = int(np.sum(tissue))
    z = float(norm.isf(alpha / voxels))
    return {"smoothing_mm": float(smoothing_mm), "smoothing_voxels": width.tolist(), "family_wise_alpha": alpha,
            "tissue_voxels": voxels, "z": z, "smoothed_noise_gain": gain, "threshold": z * noise_sd * gain}


def noisy_scan(clean, noise, tissue, noise_sd):
    """Measurement: independent Gaussian noise in every tissue voxel, nothing outside the tissue."""
    raw = np.zeros(tissue.size)
    raw[tissue] = clean[tissue] + noise_sd * noise
    return raw


def cellularity_map(raw, tissue, shape, rule):
    """Segment the lesion from the noisy scan alone; clipped cellularity inside it, zero elsewhere."""
    smoothed = gaussian_filter(raw.reshape(shape), rule["smoothing_voxels"], mode="constant").reshape(-1)
    lesion = tissue & (smoothed > rule["threshold"])
    return np.where(lesion, np.clip(raw, 0., THETA), 0.), lesion


def observed_pod(scans, pod):
    """POD of noisy training scans (one row per scan): orthonormal decoder with a fixed column sign."""
    shift = scans.mean(axis=0) if pod.centering == "mean" else np.zeros(scans.shape[1])
    centered = scans - shift
    eigenvalues, vectors = np.linalg.eigh(centered @ centered.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    kept = eigenvalues[:pod.rank]
    if len(kept) < pod.rank or not np.isfinite(kept).all() or np.any(kept <= 1e-12 * eigenvalues[0]):
        raise ValueError("Declared POD rank is numerically unresolved; refusing to reduce it silently.")
    D, _ = np.linalg.qr(centered.T @ (vectors[:, :pod.rank] / np.sqrt(kept)), mode="reduced")
    D *= np.sign(D[np.argmax(abs(D), axis=0), np.arange(pod.rank)])[None]
    return D, shift, np.sqrt(np.maximum(eigenvalues, 0.))


def noise_threshold(noise_sd, lesion_voxels):
    """Gavish-Donoho optimal hard threshold for singular values of the scan matrix, known noise level.

    The noise matrix is treated as n x m with n the mean number of segmented (noisy) voxels per scan
    and m the number of scans; modes below the threshold are indistinguishable from the noise bulk.
    """
    m, n = len(lesion_voxels), float(np.mean(lesion_voxels))
    beta = min(m, n) / max(m, n)
    factor = np.sqrt(2. * (beta + 1.) + 8. * beta / (beta + 1. + np.sqrt(beta**2 + 14. * beta + 1.)))
    return float(factor * np.sqrt(max(m, n)) * noise_sd)


def _training_fields(case, seed, source):
    """Noise-free training fields (scans x voxels) at this acquisition's times, from knots <= end."""
    start, end = case.training_span
    if case.dose_days:
        times = sample_times((start, end), case.observations, np.random.default_rng(seed))
    else:
        # Same scan times as the earlier untreated-growth design for this seed.
        times = np.sort(np.random.RandomState(seed).uniform(start, end, size=case.observations))
        times[0], times[-1] = start, end
    knots = SOURCE_KNOTS[SOURCE_KNOTS <= end]
    weights = temporal_weights(knots, times)
    clean = np.empty((len(times), len(source)))
    for left in range(0, len(source), BLOCK):
        clean[:, left:left + BLOCK] = (np.asarray(source[left:left + BLOCK, :len(knots)], dtype=float) @ weights).T
    return times, clean, float(knots[-1])


def _segmented_acquisition(case, seed, pod, source, geometry, folder):
    """Segmented scans, their POD (saved to basis.npz) and the reduced observations."""
    times, clean, last_knot = _training_fields(case, seed, source)
    tissue = np.asarray(geometry["breast_mask"], dtype=bool).reshape(-1)
    shape = tuple(int(n) for n in geometry["grid_shape"])
    noise_sd = case.noise_level * float(clean.max() - clean.min())
    rule = detection_rule(tissue, geometry["spacing"], noise_sd)
    rng, noise_sha = np.random.default_rng([seed, NOISE_STREAM]), hashlib.sha256()
    scans = np.empty_like(clean)
    lesion_voxels, false_positives, missed2 = [], [], 0.
    for j in range(len(times)):
        noise = rng.standard_normal(int(tissue.sum()))
        noise_sha.update(noise.tobytes())
        scans[j], lesion = cellularity_map(noisy_scan(clean[j], noise, tissue, noise_sd), tissue, shape, rule)
        lesion_voxels.append(int(lesion.sum()))
        false_positives.append(int(np.sum(lesion & (clean[j] <= 0.))))
        missed2 += float(np.sum(clean[j][~lesion]**2))
    D, shift, singular_values = observed_pod(scans, pod)
    threshold = noise_threshold(noise_sd, lesion_voxels)
    resolved = int(np.sum(singular_values > threshold))
    if pod.rank > resolved:
        raise ValueError(f"Declared POD rank {pod.rank} exceeds the {resolved} modes above the scan-noise threshold "
                         f"{threshold:.3g}; the extra modes would be pure noise.")
    y, clean_q = D.T @ (scans - shift).T, D.T @ (clean - shift).T
    write_npz(folder / "basis.npz", physical_matrix=D, physical_shift=shift, singular_values=singular_values,
              source_training_times=times)
    energy = singular_values**2
    return times, y, clean_q, D, shift, {
        "observation_model": "segmented", "noise_sd": noise_sd,
        "noise_sd_rule": "noise_level x range of the noise-free training fields",
        "noisy_voxels": "every breast-tissue voxel of every scan, including the first", "detection": rule,
        "reported_cellularity": f"clipped to [0, {THETA:g}] inside the segmented lesion, zero elsewhere",
        "noise_sha256": noise_sha.hexdigest(), "latest_source_day_used": last_knot,
        "lesion_voxels_per_scan": lesion_voxels,
        "singular_value_threshold": threshold, "modes_above_threshold": resolved,
        "threshold_rule": "Gavish-Donoho optimal hard threshold for the known noise level, n = mean lesion voxels",
        "basis_cumulative_energy": float(energy[:pod.rank].sum() / energy.sum()),
        "evaluation_only": {
            "false_positive_voxels_per_scan": false_positives,
            "missed_field_energy_percent": 100 * float(np.sqrt(missed2 / np.sum(clean**2))),
            "scan_relative_error_percent": 100 * float(np.linalg.norm(scans - clean) / np.linalg.norm(clean))}}


def _reference(field_blocks, D, shift, times, effects):
    """Reduced truth plus exact affine-decoder geometry for the full-field metric."""
    rank, n = D.shape[1], len(times)
    Q, R = np.linalg.qr(D, mode="reduced")
    names = ["true_comp", "physical_projection"] + (["effect_true_comp", "effect_projection"] if effects else [])
    arrays = {name: np.zeros((rank, n)) for name in names}
    scalars = ["physical_norm2", "physical_residual2", "truth_burden"]
    scalars += ["effect_norm2", "effect_residual2", "effect_burden"] if effects else []
    arrays.update({name: np.zeros(n) for name in scalars})
    for selected, field, delta in field_blocks():
        centered = field - shift[selected, None]
        arrays["true_comp"] += D[selected].T @ centered
        arrays["physical_projection"] += Q[selected].T @ centered
        arrays["physical_norm2"] += np.sum(field**2, axis=0)
        arrays["truth_burden"] += field.sum(axis=0)
        if effects:
            arrays["effect_true_comp"] += D[selected].T @ delta
            arrays["effect_projection"] += Q[selected].T @ delta
            arrays["effect_norm2"] += np.sum(delta**2, axis=0)
            arrays["effect_burden"] += delta.sum(axis=0)
    for selected, field, delta in field_blocks():
        arrays["physical_residual2"] += np.sum(
            (field - shift[selected, None] - Q[selected] @ arrays["physical_projection"])**2, axis=0)
        if effects:
            arrays["effect_residual2"] += np.sum((delta - Q[selected] @ arrays["effect_projection"])**2, axis=0)
    return {"t_pred": times, "physical_R": R, **arrays}


def _chemo_reference(case, strength, D, shift, basis_path, log):
    """Evaluation geometry for one arm; cached only for shared (not per-acquisition) bases."""
    path = None if basis_path is None else CACHE / f"reference_{Path(basis_path).stem}_{arm_name(strength)}.npz"
    if path is not None and path.exists():
        return read_npz(path)
    history, _ = chemo_history(case, log)
    branch = None if strength == case.training_strength else chemo_branch(case, strength, log)
    times = grid(case).times
    weights = temporal_weights(SOURCE_KNOTS, times)
    start = int(round(case.training_span[1] / .5))
    switch = first_future_dose(case)

    def blocks():
        for left in range(0, len(history), BLOCK):
            selected = slice(left, min(left + BLOCK, len(history)))
            old = history[selected].astype(float)
            baseline = old @ weights
            delta = (np.zeros_like(baseline) if branch is None else
                     (branch[selected].astype(float) - old[:, start:]) @ weights[start:])
            delta[:, times < switch] = 0.
            yield selected, baseline + delta, delta

    log(f"  reference geometry for {arm_label(strength)} ...")
    reference = _reference(blocks, D, shift, times, effects=True)
    reference.update(chemo_inputs(case, strength))
    if path is not None:
        write_npz(path, **reference)
    return reference


def _growth_reference(case, raw, knots, D, shift, volume):
    times = grid(case).times

    def blocks():
        for left in range(0, len(raw), GROWTH_BLOCK):
            selected = slice(left, min(left + GROWTH_BLOCK, len(raw)))
            yield selected, interp1d(knots, raw[selected], axis=1, kind="cubic")(times), None

    reference = _reference(blocks, D, shift, times, effects=False)
    reference["truth_burden"] = volume * reference["truth_burden"]
    return reference


def node_step_cuts(input_times, dose_days, span):
    """Training-window pulse onsets and their neighbouring input knots, for the NODE solver."""
    cuts = []
    for dose in dose_days:
        if span[0] < dose < span[1]:
            index = int(np.searchsorted(input_times, dose))
            cuts.extend([input_times[index - 1], dose, input_times[index]])
    return np.unique(cuts)


def prepare(case, seed, pod=None, root=None, log=print) -> Path:
    """Create (or reuse) one acquisition and its evaluation-only references."""
    pod = case.pod if pod is None else pod
    replace(case, pod=pod)  # rejects POD sources the observation model does not allow
    segmented = case.observation != LEGACY_OBSERVATION
    folder = seed_directory(case, seed, pod, root) / "data"
    if (folder / "metadata.json").exists():
        return folder
    started = time.monotonic()
    log(f"Preparing {case.name} seed {seed} ({acquisition_tag(case, pod)})")
    recipe = describe(case, pod)
    references = {}
    if case.dose_days:
        history, geometry = chemo_history(case, log)
        if segmented:
            times, y, clean_q, D, shift, acquisition = _segmented_acquisition(
                case, seed, pod, history, geometry, folder)
            basis_path = None
        else:
            D, shift, basis_path = chemo_basis(case, pod, log)
            times, y, clean_q, acquisition = _chemo_observations(case, seed, history, D, shift)
        volume = float(np.prod(geometry["spacing"]))
        inputs = _training_inputs(case)
        extra = {
            "input_times": inputs["input_times"], "input_values": inputs["input_values"],
            "dose_days": inputs["dose_days"], "native_inputs_eval": inputs["native_inputs_eval"],
            "node_step_cuts": node_step_cuts(inputs["input_times"], inputs["dose_days"], case.training_span),
        }
        for arm, strength in arms(case).items():
            references[arm] = _chemo_reference(case, strength, D, shift, basis_path, log)
        sources = {"training_history": file_record(chemo_history_path(case)),
                   "basis": file_record(basis_path or folder / "basis.npz")}
        sources.update({f"future_branch_{arm}": file_record(chemo_branch_path(case, strength))
                        for arm, strength in arms(case).items() if strength != case.training_strength})
    else:
        raw, geometry = growth_source(case, log)
        volume = float(np.prod(geometry["spacing"]))
        if segmented:
            times, y, clean_q, D, shift, acquisition = _segmented_acquisition(case, seed, pod, raw, geometry, folder)
        else:
            times, y, clean_q, D, shift, acquisition = _growth_observations(
                case, seed, raw, geometry["times_days"], pod)
            write_npz(folder / "basis.npz", physical_matrix=D, physical_shift=shift)
        log("  reference geometry ...")
        references["untreated"] = _growth_reference(case, raw, geometry["times_days"], D, shift, volume)
        extra = {}
        sources = {"truth": file_record(growth_source_path(case)), "basis": file_record(folder / "basis.npz")}
    _, R = np.linalg.qr(D, mode="reduced")
    safety = np.sqrt(np.mean(y**2, axis=1))
    if not np.isfinite(y).all() or np.any(safety <= 0):
        raise ValueError("Invalid acquired reduced states.")
    write_npz(folder / "observations.npz", t_sampled=times, y=y, q0=y[:, 0], clean_sampled=clean_q,
              state_safety_scale=safety, physical_R=R, burden_weights=volume * D.sum(axis=0),
              burden_offset=np.asarray(volume * shift.sum()), **extra)
    for arm, reference in references.items():
        write_npz(folder / f"reference_{arm}.npz", **reference)
    write_json(folder / "metadata.json", {
        "case": case.name, "seed": seed, "pod": asdict(pod), "pod_tag": pod_tag(pod),
        "recipe_fingerprint": recipe["recipe_fingerprint"], "recipe": recipe,
        "arms": arms(case), "acquisition": acquisition, "voxel_volume": volume, "sources": sources,
        "observations_sha256": digest(folder / "observations.npz"),
        "observation_model": case.observation,
        "POD_fitted_on": ("noisy segmented training scans of this acquisition" if segmented else
                          "noisy training observations" if not case.dose_days else
                          f"clean training-regimen history on days {case.training_span[0]:g}-"
                          f"{case.training_span[1]:g} ({pod.source})"),
        "observation_interpolation_knots_days": [0., float(case.training_span[1] if segmented or case.dose_days
                                                            else SOURCE_KNOTS[-1])],
        "runtime_seconds": time.monotonic() - started,
    })
    log(f"  prepared in {time.monotonic() - started:.0f} s")
    return folder


def load(case, seed, pod=None, root=None, *, references=True):
    folder = seed_directory(case, seed, pod, root) / "data"
    metadata = read_json(folder / "metadata.json")
    if digest(folder / "observations.npz") != metadata["observations_sha256"]:
        raise RuntimeError(f"Observations changed after preparation: {folder}")
    result = {"metadata": metadata, "observations": read_npz(folder / "observations.npz"), "folder": folder}
    if references:
        result["references"] = {arm: read_npz(folder / f"reference_{arm}.npz") for arm in metadata["arms"]}
    return result


# -----------------------------------------------------------------------------
# Command-line helpers shared by 04/05/06
# -----------------------------------------------------------------------------
def add_common_arguments(parser, *, case_optional=False):
    if case_optional:
        parser.add_argument("case", nargs="?", choices=tuple(CASES),
                            help="Benchmark to process (default: all three).")
    else:
        parser.add_argument("case", choices=tuple(CASES))
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="Acquisition seeds (default: the reported seeds for the case).")
    parser.add_argument("--observation", choices=OBSERVATION_MODELS,
                        help="Observation model (default: segmented, the reported design; oracle_masked "
                             "reproduces the earlier design and its POD defaults).")
    parser.add_argument("--pod-rank", type=int)
    parser.add_argument("--pod-source", choices=("observed_training", "nominal_training", "matched_training"))
    parser.add_argument("--pod-centering", choices=("mean", "none"))
    parser.add_argument("--output-root", type=Path, help=f"Results root (default: {RESULTS}).")
    return parser


def resolve(args, name=None):
    case = get_case(name or args.case, getattr(args, "observation", None))
    changes = {key: value for key, value in (
        ("rank", args.pod_rank), ("source", args.pod_source), ("centering", args.pod_centering))
        if value is not None}
    pod = replace(case.pod, **changes)
    replace(case, pod=pod)
    seeds = tuple(args.seeds) if args.seeds else (
        LEGACY_SEEDS if case.observation == LEGACY_OBSERVATION else DEFAULT_SEEDS)[case.name]
    return case, pod, seeds
