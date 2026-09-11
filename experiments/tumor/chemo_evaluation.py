"""Fixed-model dose evaluation with identical metrics for both chemo methods."""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from chemo_protocol import (
    DOSE_SCALES, OUTPUT_ROOT, PROTOCOL_ID, TRAINING_SPAN,
    array_fingerprint, dose_path, load_dose_fom, save_protocol,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, record):
    with Path(path).open("w") as stream:
        json.dump(_json_safe(record), stream, indent=2, allow_nan=False)


def _truth_summary(data, scale):
    """Cache projected truth and exact orthogonal residuals, without huge ensembles."""
    basis, t = data["basis"], data["t_pred"]
    if not np.allclose(basis.entries.T @ basis.entries,
                       np.eye(basis.entries.shape[1]), atol=1e-8):
        raise ValueError("Field-error decomposition requires an orthonormal POD basis")
    source = dose_path(scale)
    stat = source.stat()
    key = array_fingerprint(
        basis.entries, basis.shift_, t, np.array([scale, stat.st_size, stat.st_mtime_ns]))
    directory = Path(OUTPUT_ROOT) / "truth_cache"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            return {k: cached[k] for k in cached.files}
    fom = load_dose_fom(scale)
    # Build the cubic interpolator once, then evaluate small time chunks.
    interpolate = interp1d(fom._times, fom._snapshots, axis=1, kind="cubic",
                          bounds_error=True, copy=False)
    projected, residual2, norm2, volume = [], [], [], []
    voxel_volume = float(np.prod(fom.spacing))
    for start in range(0, len(t), 16):
        truth = interpolate(t[start:start + 16])
        norm2.append(np.sum(truth * truth, axis=0))
        volume.append(truth.sum(axis=0) * voxel_volume)
        truth -= basis.shift_[:, None]
        comp = basis.entries.T @ truth
        projected.append(comp)
        residual2.append(np.maximum(
            np.sum(truth * truth, axis=0) - np.sum(comp * comp, axis=0), 0.0))
    result = dict(
        true_comp=np.concatenate(projected, axis=1),
        residual2=np.concatenate(residual2), norm2=np.concatenate(norm2),
        true_volume=np.concatenate(volume), voxel_volume=np.array(voxel_volume))
    np.savez_compressed(path, **result)
    return result


def _relative(error2, truth2, mask):
    denominator = float(np.sum(truth2[mask]))
    if denominator <= 0:
        raise ValueError("Relative error is undefined for zero truth norm")
    return float(np.sqrt(np.sum(error2[mask]) / denominator))


def score_predictions(data, truth, solves):
    t, basis = data["t_pred"], data["basis"]
    arrays = dict(t_pred=t, true_comp=truth["true_comp"],
                  true_volume=truth["true_volume"])
    metrics = {}
    if len(solves) == 0:
        return metrics, arrays
    median = np.median(solves, axis=0)
    lower, upper = np.percentile(solves, [5, 95], axis=0)
    reduced_error2 = np.sum((median - truth["true_comp"]) ** 2, axis=0)
    field_error2 = reduced_error2 + truth["residual2"]
    vol_weights = basis.entries.sum(axis=0)
    volumes = ((vol_weights @ solves) + basis.shift_.sum()) * truth["voxel_volume"]
    volume_median = np.median(volumes, axis=0)
    volume_lower, volume_upper = np.percentile(volumes, [5, 95], axis=0)
    reduced_inside = (truth["true_comp"] >= lower) & (truth["true_comp"] <= upper)
    volume_inside = ((truth["true_volume"] >= volume_lower)
                     & (truth["true_volume"] <= volume_upper))
    for region, mask in (
            ("fit", t <= TRAINING_SPAN[1]), ("forecast", t > TRAINING_SPAN[1])):
        metrics.update({
            f"field_error_{region}": _relative(field_error2, truth["norm2"], mask),
            f"projection_error_{region}": _relative(truth["residual2"], truth["norm2"], mask),
            f"reduced_error_{region}": _relative(
                reduced_error2, np.sum(truth["true_comp"] ** 2, axis=0), mask),
            f"reduced_coverage_{region}": float(np.mean(reduced_inside[:, mask])),
            f"reduced_interval_width_{region}": float(np.mean(
                (upper - lower)[:, mask])),
            f"volume_error_{region}": _relative(
                (volume_median - truth["true_volume"]) ** 2,
                truth["true_volume"] ** 2, mask),
            f"volume_coverage_{region}": float(np.mean(volume_inside[mask])),
            f"volume_relative_interval_width_{region}": float(
                np.mean((volume_upper - volume_lower)[mask])
                / np.mean(np.abs(truth["true_volume"][mask]))),
            f"negative_volume_fraction_{region}": float(np.mean(volumes[:, mask] < 0)),
        })
    arrays.update(
        median_comp=median, lower_comp=lower, upper_comp=upper,
        median_volume=volume_median, lower_volume=volume_lower, upper_volume=volume_upper,
        field_error=np.sqrt(field_error2 / np.maximum(truth["norm2"], 1e-30)))
    return metrics, arrays


def evaluate_doses(data, predict, method_name, out_dir, *, nominal_solves=None,
                   n_total, n_trained=None, model_id, dose_scales=DOSE_SCALES):
    """Change only the dose input; never refit the model or the nominal POD basis."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_protocol(data, out_dir)
    summaries = []
    for scale in dose_scales:
        tag = f"{scale:g}".replace(".", "p")
        stem = out_dir / f"{method_name}_dose{tag}"
        metadata = dict(
            protocol_id=PROTOCOL_ID, schema=data["schema"]["name"],
            noise=data["schema"]["NOISE_LEVEL"], method=method_name,
            dose_scale=float(scale), model_id=model_id,
            data_fingerprint=data["fingerprint"], n_total=int(n_total),
            n_trained=int(n_total if n_trained is None else n_trained))
        source = dose_path(scale)
        stat = source.stat()
        metadata["truth_source"] = str(source)
        metadata["truth_source_id"] = array_fingerprint(
            np.array([stat.st_size, stat.st_mtime_ns], dtype=np.int64))
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        if json_path.exists() and npz_path.exists():
            with json_path.open() as stream:
                row = json.load(stream)
            if any(row.get(k) != v for k, v in metadata.items()):
                raise ValueError(f"Cached dose predictions have different provenance: {stem}")
            summaries.append(row)
            print(f"  Reusing {method_name} dose {scale:g}", flush=True)
            continue
        truth = _truth_summary(data, scale)
        input_func = lambda t, scale=scale: scale * data["input_func"](t)
        solves = (nominal_solves if scale == 1.0 and nominal_solves is not None
                  else predict(scale, input_func))
        solves = np.asarray(solves)
        expected = (data["snaps_comp"].shape[0], len(data["t_pred"]))
        if solves.size == 0:
            solves = np.empty((0, *expected))
        if solves.ndim != 3 or solves.shape[1:] != expected:
            raise ValueError(f"Invalid prediction shape {solves.shape}; expected (S, {expected})")
        if not np.isfinite(solves).all() or len(solves) > n_total or n_total <= 0:
            raise ValueError("Invalid stable-solve ensemble or denominator")
        metrics, arrays = score_predictions(data, truth, solves)
        row = dict(metadata, n_stable=len(solves),
                   stability_pct=100.0 * len(solves) / n_total,
                   numerical_success_pct=100.0 * len(solves) / n_total,
                   accepted_pct=100.0 * len(solves) / metadata["n_trained"],
                   **metrics)
        np.savez_compressed(npz_path, rom_solves=solves, **arrays)
        write_json(json_path, row)
        summaries.append(row)
        print(f"  {method_name} dose {scale:g}: stable {len(solves)}/{n_total}; "
              f"field forecast error {metrics.get('field_error_forecast', np.nan):.2%}; "
              f"reduced forecast coverage {metrics.get('reduced_coverage_forecast', np.nan):.1%}",
              flush=True)
    return summaries


def plot_comparison(out_dir):
    import matplotlib.pyplot as plt
    from core.plotting.style import save_figure

    out_dir = Path(out_dir)
    with (out_dir / "protocol.json").open() as stream:
        protocol = json.load(stream)
    validated = {}
    for method, fit_name in (("04_unified_chemo", "bayesian_fit"),
                              ("05_neural_ode_chemo", "neural_fit")):
        with (out_dir / f"{fit_name}.json").open() as stream:
            fit = json.load(stream)
        for scale in DOSE_SCALES:
            tag = f"{scale:g}".replace(".", "p")
            path = out_dir / f"{method}_dose{tag}.json"
            with path.open() as stream:
                row = json.load(stream)
            if (not row.get("model_id") or row["model_id"] != fit.get("model_id")
                    or row.get("data_fingerprint") != fit.get("data_fingerprint")
                    or row.get("data_fingerprint") != protocol["data_fingerprint"]
                    or row.get("method") != method or row.get("dose_scale") != scale
                    or row.get("protocol_id") != PROTOCOL_ID
                    or protocol.get("protocol_id") != PROTOCOL_ID
                    or row.get("schema") != protocol["schema"]["name"]
                    or row.get("noise") != protocol["schema"]["NOISE_LEVEL"]):
                raise ValueError(f"Prediction, fit, or protocol provenance mismatch: {path}")
            row["inference_profile"] = (
                fit.get("profile", "historical") if method == "04_unified_chemo"
                else "neural-ensemble")
            validated[method, scale] = row
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)
    for col, scale in enumerate(DOSE_SCALES):
        tag = f"{scale:g}".replace(".", "p")
        for method, label, color in (
                ("04_unified_chemo", "Bayesian OpInf", "tab:purple"),
                ("05_neural_ode_chemo", "Neural ODE", "tab:orange")):
            stem = out_dir / f"{method}_dose{tag}"
            row = validated[method, scale]
            if method == "04_unified_chemo" and row["inference_profile"] == "input-aware":
                label = "Bayesian OpInf (input-aware, experimental)"
            rows.append(row)
            with np.load(stem.with_suffix(".npz")) as d:
                t = d["t_pred"]
                if method == "04_unified_chemo":
                    axes[0, col].plot(t, d["true_volume"], "k-", label="FOM truth")
                if row["n_stable"] > 0:
                    axes[0, col].plot(t, d["median_volume"], color=color, label=label)
                    axes[0, col].fill_between(
                        t, d["lower_volume"], d["upper_volume"], color=color, alpha=.15)
                    axes[1, col].plot(t, d["field_error"], color=color, label=label)
        axes[0, col].set_title(f"Dose {scale:g}x (trained at 1x)")
        for ax in axes[:, col]:
            ax.axvspan(*TRAINING_SPAN, color="gray", alpha=.08)
            ax.axvline(TRAINING_SPAN[1], color="gray", ls=":")
            ax.legend(fontsize=8)
        axes[1, col].set(xlabel="Time (days)", yscale="log")
    axes[0, 0].set_ylabel("Tumor burden (mm3), 90% intervals")
    axes[1, 0].set_ylabel("Relative full-field error")
    fig.tight_layout()
    save_figure(fig, str(out_dir / "dose_comparison.png"))
    write_json(out_dir / "comparison.json", rows)
    return rows
