"""Shared nominal-dose data and protocol for the matched chemo experiments."""

from pathlib import Path
import json

import numpy as np

from chemo_artifacts import array_fingerprint
from config import Basis, TumorTwinFOM, load_chemo_fom_data, make_jax_input_func


TRAINING_SPAN = (5.0, 70.0)
PREDICTION_DAYS = 110.0
NUM_PRED_POINTS = 400
NUM_MODES = 4
SEED = 42
DOSE_SCALES = (0.8, 1.0, 1.2)
PROTOCOL_ID = "chemo_matched_80_5_70_110_v1"
SCRIPT_DIR = Path(__file__).resolve().parent
FOM_DATA_PATH = str(SCRIPT_DIR / "data/TNBC_demo_001_fom_chemo_sparse5_sens0p5.npz")
OUTPUT_ROOT = str(SCRIPT_DIR / "results" / PROTOCOL_ID)
INPUT_AWARE_OUTPUT_ROOT = str(SCRIPT_DIR / "results" / f"{PROTOCOL_ID}_input_aware_v1")
SCHEMAS = [
    dict(name=f"dense_{level}_noise", label=f"Dense data, {level} noise",
         NUM_SAMPLES=80, NOISE_LEVEL=noise, NUM_EVAL_POINTS=200)
    for level, noise in (("low", 0.01), ("medium", 0.03), ("high", 0.05))
]


def dose_path(scale):
    if scale == 1.0:
        return Path(FOM_DATA_PATH)
    tag = f"{scale:g}".replace(".", "p")
    return Path(FOM_DATA_PATH).with_name(Path(FOM_DATA_PATH).stem + f"_dose{tag}.npz")


def validate_dose_cache(scale):
    path = dose_path(scale)
    with np.load(FOM_DATA_PATH, allow_pickle=False) as nominal, \
            np.load(path, allow_pickle=False) as candidate:
        if not np.isclose(float(candidate["dose_scale"]), scale):
            raise ValueError(f"Dose scale metadata does not match {scale}: {path}")
        for key in ("k", "d", "theta", "sensitivity", "decay_rate",
                    "chemo_dose_days", "chemo_doses", "spacing", "grid_shape"):
            if not np.array_equal(nominal[key], candidate[key]):
                raise ValueError(f"Dose cache differs in {key}: {path}")
        times = candidate["times_days"]
        if times[0] > TRAINING_SPAN[0] or times[-1] < PREDICTION_DAYS:
            raise ValueError(f"FOM cache does not cover the experiment: {path}")
        if TRAINING_SPAN[0] >= float(candidate["chemo_dose_days"][0]):
            raise ValueError("Shared initial state must precede the first dose")
    return path


def projected_noise_variances(entries, clean_snapshots, noise_level):
    """Project the declared voxel-noise covariance, excluding the exact IC."""
    if (clean_snapshots.ndim != 2 or clean_snapshots.shape[1] < 2
            or entries.ndim != 2 or len(entries) != len(clean_snapshots)):
        raise ValueError("Noise projection requires matching basis and training snapshots")
    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError("Noise level must be finite and nonnegative")
    scale = noise_level * (clean_snapshots.max() - clean_snapshots.min())
    active_fraction = np.mean(
        clean_snapshots[:, 1:] > .001 * clean_snapshots.max(), axis=1)
    return scale ** 2 * np.einsum("vi,v,vi->i", entries, active_fraction, entries)


def prepare_data(schema, seed=SEED, num_modes=NUM_MODES):
    """Both methods receive exactly the same observations, POD basis and input."""
    if schema["NUM_SAMPLES"] != 80 or num_modes != NUM_MODES:
        raise ValueError("The matched protocol requires 80 observations and four modes")
    for scale in DOSE_SCALES:
        validate_dose_cache(scale)
    np.random.seed(seed)
    t_pred = np.linspace(TRAINING_SPAN[0], PREDICTION_DAYS, NUM_PRED_POINTS)
    t_truth = np.linspace(TRAINING_SPAN[0], PREDICTION_DAYS, schema["NUM_EVAL_POINTS"])
    fom, t_full, true_states, t_samp, snaps_noisy, input_raw, chemo_meta = \
        load_chemo_fom_data(FOM_DATA_PATH, t_truth, TRAINING_SPAN,
                           schema["NUM_SAMPLES"], schema["NOISE_LEVEL"], seed=seed)
    basis = Basis(num_vectors=num_modes)
    clean_training = fom.get_states(t_samp)
    basis.fit(clean_training)
    noise_variances = projected_noise_variances(
        basis.entries, clean_training, schema["NOISE_LEVEL"])
    snaps_comp = basis.compress(snaps_noisy)
    true_comp = basis.compress(true_states)
    input_func = make_jax_input_func(input_raw, t_pred[0], t_pred[-1], n_points=4001)
    alpha_pred = np.interp(t_pred, input_func.t_grid, input_func.alpha_grid)
    fingerprint = array_fingerprint(
        t_samp, snaps_comp, basis.entries, basis.shift_,
        t_pred, input_func.t_grid, input_func.alpha_grid)
    return dict(
        schema=dict(schema), fom=fom, basis=basis, t_full=t_full, t_pred=t_pred,
        true_states=true_states, true_comp=true_comp, t_samp=t_samp,
        snaps_noisy=snaps_noisy, snaps_comp=snaps_comp, q0=snaps_comp[:, 0],
        noise_variances_comp=noise_variances,
        input_func=input_func, chemo_meta=chemo_meta, fingerprint=fingerprint,
        alpha_pred=alpha_pred)


def save_protocol(data, out_dir):
    path = Path(out_dir) / "protocol.json"
    record = dict(
        protocol_id=PROTOCOL_ID, schema=data["schema"], seed=SEED,
        training_span=list(TRAINING_SPAN), prediction_end=PREDICTION_DAYS,
        prediction_points=NUM_PRED_POINTS, num_modes=NUM_MODES,
        dose_scales=list(DOSE_SCALES), data_fingerprint=data["fingerprint"],
        nominal_fom=FOM_DATA_PATH, basis_source="clean nominal training snapshots",
        initial_state="noise-free observation at day 5; same nominal state for all doses")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open() as stream:
            if json.load(stream) != record:
                raise ValueError(f"Refusing to mix protocols in {path.parent}")
    else:
        with path.open("w") as stream:
            json.dump(record, stream, indent=2)


def load_dose_fom(scale):
    return TumorTwinFOM(str(validate_dose_cache(scale)))
