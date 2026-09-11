"""Export saved matched chemo predictions to local simulation-grid NIfTI volumes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

import nibabel as nib
import numpy as np
from scipy.interpolate import interp1d

from chemo_artifacts import array_fingerprint, PROTOCOL_ID as PROTOCOL, INPUT_AWARE_RESULTS_SUBDIR


HERE = Path(__file__).resolve().parent
SCHEMAS = ("dense_low_noise", "dense_medium_noise", "dense_high_noise")
METHODS = {"04_unified_chemo": "bayesian_fit", "05_neural_ode_chemo": "neural_fit"}
DOSES = (0.8, 1.0, 1.2)
COORDINATES = "simulation grid; not registered patient coordinates"
OWNER = "local-chemo-niivue-export-v1"


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_frames(times, requested):
    times = np.asarray(times)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("Prediction times must be finite and strictly increasing")
    if not np.isfinite(requested).all() or min(requested) < times[0] or max(requested) > times[-1]:
        raise ValueError("Requested times must lie within the saved prediction interval")
    return np.unique([int(np.argmin(np.abs(times - day))) for day in requested])


def reconstruct(entries, shift, median, solves, indices, chunk_voxels):
    """Quantiles are taken AFTER reconstruction, independently at every voxel."""
    nvox = len(shift)
    prediction = np.empty((nvox, len(indices)), dtype=np.float32)
    width = np.empty_like(prediction)
    for start in range(0, nvox, chunk_voxels):
        stop = min(start + chunk_voxels, nvox)
        basis = entries[start:stop]
        offset = shift[start:stop, None]
        prediction[start:stop] = basis @ median[:, indices] + offset
        for frame, index in enumerate(indices):
            fields = np.asarray(basis @ solves[:, :, index].T + offset, dtype=np.float32)
            lower, upper = np.percentile(fields, [5, 95], axis=1)
            width[start:stop, frame] = upper - lower
    if not np.isfinite(prediction).all() or not np.isfinite(width).all():
        raise ValueError("Reconstructed fields exceed finite float32 display range")
    return prediction, width


def interpolate_truth(snapshots, times, selected, chunk_voxels):
    """Use the evaluator's cubic interpolation, bounded in space instead of time."""
    truth = np.empty((len(snapshots), len(selected)), dtype=np.float32)
    for start in range(0, len(snapshots), chunk_voxels):
        stop = min(start + chunk_voxels, len(snapshots))
        interpolate = interp1d(times, snapshots[start:stop], axis=1, kind="cubic",
                              bounds_error=True, copy=False)
        truth[start:stop] = interpolate(selected)
    if not np.isfinite(truth).all():
        raise ValueError("Nonfinite interpolated truth")
    return truth


def save_volume(output, field, shape, spacing, kind):
    data = np.asarray(field, dtype=np.float32).reshape((*shape, field.shape[1]))
    affine = np.diag([*spacing, 1.0])
    fingerprint = array_fingerprint(data, affine)
    name = f"chemo-niivue-{kind}-{fingerprint}.nii.gz"
    path = output / name
    if path.is_symlink():
        raise ValueError(f"Refusing symlink output: {path}")
    if not path.exists():
        image = nib.Nifti1Image(data, affine)
        # Code 0 explicitly avoids claiming scanner/anatomical registration.
        image.set_qform(affine, code=0)
        image.set_sform(affine, code=2)
        image.header.set_xyzt_units("mm", "unknown")
        image.header["descrip"] = COORDINATES.encode("ascii")
        image.header["pixdim"][4] = 1  # Nonuniform actual days live in manifest.
        partial = output / f"{name[:-7]}.partial.nii.gz"
        if partial.exists() or partial.is_symlink():
            raise ValueError(f"Remove stale exporter partial file explicitly: {partial}")
        nib.save(image, partial)
        partial.replace(path)
    else:
        image = nib.load(path)
        if (image.shape != data.shape or not np.allclose(image.affine, affine)
                or not np.array_equal(np.asanyarray(image.dataobj), data)):
            raise ValueError(f"Existing content-addressed volume differs: {path}")
    return {"file": name, "data_fingerprint": fingerprint,
            "min": float(data.min()), "max": float(data.max())}


def prepare_output(output):
    output.mkdir(parents=True, exist_ok=True)
    marker = output / ".chemo-niivue-export.json"
    if marker.is_symlink():
        raise ValueError("Output marker cannot be a symlink")
    if marker.exists():
        if read_json(marker) != {"owner": OWNER}:
            raise ValueError("Output directory belongs to another exporter")
    elif any(output.iterdir()):
        raise ValueError("Choose an empty output directory (unrelated files will not be overwritten)")
    else:
        marker.write_text(json.dumps({"owner": OWNER}) + "\n")
    if (output / "manifest.json").is_symlink():
        raise ValueError("Manifest cannot be a symlink")


def load_case(root, schema, method, dose):
    stem = root / schema / f"{method}_dose{dose:g}".replace(".", "p")
    fit = root / schema / METHODS[method]
    paths = {"evaluation": stem.with_suffix(".npz"), "metadata": stem.with_suffix(".json"),
             "basis": fit.with_suffix(".npz"), "protocol": root / schema / "protocol.json"}
    if fit.with_suffix(".json").is_file():
        paths["basis_metadata"] = fit.with_suffix(".json")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete result: " + ", ".join(missing))
    metadata, protocol = (read_json(paths[k]) for k in ("metadata", "protocol"))
    with np.load(paths["basis"], allow_pickle=False) as basis:
        entries, shift, basis_times = (basis[k] for k in
                                       ("basis_entries", "basis_shift", "t_pred"))
        embedded = {key: basis[key].item() for key in ("model_id", "data_fingerprint")
                    if key in basis}
    fit_meta = read_json(paths["basis_metadata"]) if "basis_metadata" in paths else embedded
    metadata["inference_profile"] = (
        fit_meta.get("profile", "historical") if method == "04_unified_chemo"
        else "neural-ensemble")
    for key, expected in (("protocol_id", PROTOCOL), ("schema", schema), ("method", method),
                          ("dose_scale", dose)):
        if metadata.get(key) != expected:
            raise ValueError(f"Evaluation {key} does not match selection")
    for key in ("model_id", "data_fingerprint"):
        if not metadata.get(key) or fit_meta.get(key) != metadata[key]:
            raise ValueError(f"Basis/evaluation {key} mismatch")
        if key in embedded and embedded[key] != metadata[key]:
            raise ValueError(f"Embedded basis/evaluation {key} mismatch")
    if (protocol.get("protocol_id") != PROTOCOL
            or protocol.get("data_fingerprint") != metadata["data_fingerprint"]
            or protocol.get("training_span") != [5.0, 70.0]
            or protocol.get("schema", {}).get("name") != schema
            or protocol.get("schema", {}).get("NOISE_LEVEL") != metadata.get("noise")):
        raise ValueError("Protocol provenance mismatch")
    with np.load(paths["evaluation"], allow_pickle=False) as evaluation:
        times, solves = evaluation["t_pred"], evaluation["rom_solves"]
        median = evaluation["median_comp"] if "median_comp" in evaluation else None
    if (entries.ndim != 2 or entries.shape[1] != 4 or shift.shape != (len(entries),)
            or not np.array_equal(times, basis_times)
            or solves.shape != (metadata["n_stable"], entries.shape[1], len(times))
            or not all(np.isfinite(a).all() for a in (entries, shift, solves))):
        raise ValueError("Invalid basis or stable prediction ensemble")
    if len(solves) and (median is None or median.shape != solves.shape[1:]
                        or not np.allclose(median, np.median(solves, axis=0))):
        raise ValueError("Saved reduced median is absent or inconsistent with stable solves")
    return metadata, paths, entries, shift, times, solves, median


def export(args):
    root, output = args.results_root.resolve(), args.output_dir.resolve()
    prepare_output(output)
    cases, warnings, truth_cache = [], [], {}
    file_hashes = {}

    def file_record(path):
        if path not in file_hashes:
            file_hashes[path] = digest_file(path)
        return {"file": str(path), "sha256": file_hashes[path]}

    for dose in args.dose:
        truth_cache.clear()
        fom_path = HERE / "data" / ("TNBC_demo_001_fom_chemo_sparse5_sens0p5"
                                   + ("" if dose == 1 else f"_dose{dose:g}".replace(".", "p"))
                                   + ".npz")
        snapshots = None
        for schema in args.schema:
            for method in args.method:
                label = f"{schema}/{method}/dose{dose:g}"
                try:
                    metadata, paths, entries, shift, times, solves, median = load_case(
                        root, schema, method, dose)
                    indices = selected_frames(times, args.times)
                    selected = times[indices]
                    if snapshots is None:
                        with np.load(fom_path, allow_pickle=False) as fom:
                            shape = tuple(int(n) for n in fom["grid_shape"])
                            spacing = np.asarray(fom["spacing"], dtype=float)
                            loaded_snapshots, fom_times = fom["snapshots"], fom["times_days"]
                            source_dose = float(fom["dose_scale"])
                        if (len(shape) != 3 or spacing.shape != (3,) or np.any(spacing <= 0)
                                or not np.isfinite(spacing).all()
                                or any(n <= 0 for n in shape)
                                or fom_times.ndim != 1 or len(fom_times) < 4
                                or not np.isfinite(fom_times).all() or not np.all(np.diff(fom_times) > 0)
                                or loaded_snapshots.shape != (int(np.prod(shape)), len(fom_times))
                                or not np.isclose(source_dose, dose)):
                            raise ValueError("Invalid FOM grid, spacing or dose")
                        snapshots = loaded_snapshots
                        del loaded_snapshots
                    if entries.shape[0] != len(snapshots):
                        raise ValueError("Basis does not match the FOM grid")
                    if metadata.get("truth_source_id"):
                        stat = fom_path.stat()
                        source_id = array_fingerprint(
                            np.array([stat.st_size, stat.st_mtime_ns], dtype=np.int64))
                        if source_id != metadata["truth_source_id"]:
                            raise ValueError("FOM changed since evaluation (truth_source_id mismatch)")
                    key = (dose, tuple(selected))
                    if key not in truth_cache:
                        truth = interpolate_truth(snapshots, fom_times, selected, args.chunk_voxels)
                        truth_volume = save_volume(output, truth, shape, spacing, "truth")
                        truth_cache[key] = truth, truth_volume
                    truth, truth_volume = truth_cache[key]
                    volumes = {"truth": truth_volume}
                    if len(solves):
                        prediction, width = reconstruct(
                            entries, shift, median, solves, indices, args.chunk_voxels)
                        volumes.update({
                            "prediction": save_volume(output, prediction, shape, spacing, "prediction"),
                            "error": save_volume(output, np.abs(truth - prediction), shape, spacing, "error"),
                            "uncertainty": save_volume(output, width, shape, spacing, "uncertainty"),
                        })
                    cases.append({
                        "id": label, "schema": schema, "method": method, "dose_scale": dose,
                        "noise": metadata["noise"], "times_days": selected.tolist(),
                        "prediction_indices": indices.tolist(), "training_end": 70.0,
                        "status": "ready" if len(solves) else "no_stable_solves",
                        "grid_shape": list(shape), "spacing_mm": spacing.tolist(),
                        "affine": np.diag([*spacing, 1.0]).tolist(), "coordinates": COORDINATES,
                        "prediction_definition": "POD reconstruction of componentwise reduced median; not voxelwise median",
                        "uncertainty_definition": "Voxelwise 95th minus 5th percentile of reconstructed stable samples (90% width)",
                        "error_definition": "Absolute truth minus displayed prediction",
                        "basis_fingerprint": array_fingerprint(entries, shift),
                        "metadata": metadata, "sources": {k: file_record(p) for k, p in
                                                         dict(paths, fom=fom_path).items()},
                        "volumes": volumes,
                    })
                    print(f"Exported {label}: {len(indices)} frames, {len(solves)} stable samples", flush=True)
                except (ValueError, KeyError, OSError, EOFError, BadZipFile) as exc:
                    warning = f"{label}: {exc}"
                    if args.strict:
                        raise ValueError(warning) from exc
                    warnings.append(warning)
                    print(f"WARNING: {warning}", file=sys.stderr, flush=True)
        del snapshots
    manifest = {"format_version": 1, "generator": OWNER, "protocol_id": PROTOCOL,
                "created_at": datetime.now(timezone.utc).isoformat(), "coordinates": COORDINATES,
                "requested_times_days": args.times, "cases": cases, "warnings": warnings}
    partial = output / "manifest.json.partial"
    if partial.exists() or partial.is_symlink():
        raise ValueError(f"Remove stale exporter partial file explicitly: {partial}")
    partial.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    partial.replace(output / "manifest.json")
    print(f"{len(cases)} cases written to {output / 'manifest.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=HERE / "results" / INPUT_AWARE_RESULTS_SUBDIR)
    parser.add_argument("--output-dir", type=Path, default=HERE / "niivue_viewer/public/data")
    parser.add_argument("--schema", nargs="+", choices=SCHEMAS, default=list(SCHEMAS))
    parser.add_argument("--method", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--dose", nargs="+", type=float, choices=DOSES, default=list(DOSES))
    parser.add_argument("--times", nargs="+", type=float, default=[5, 20, 40, 60, 70, 80, 90, 100, 110])
    parser.add_argument("--chunk-voxels", type=int, default=8192)
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping incomplete/invalid cases")
    args = parser.parse_args()
    if args.chunk_voxels <= 0:
        parser.error("--chunk-voxels must be positive")
    export(args)


if __name__ == "__main__":
    main()
