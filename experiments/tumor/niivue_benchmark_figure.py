"""
niivue_benchmark_figure.py — NiiVue figure of a chemotherapy benchmark forecast.

No fitting and no FOM simulation: reads the cached full-order fields and the evaluations written by
04_unified_benchmark.py and 05_neural_ode_benchmark.py. Three steps, from experiments/tumor:

    python niivue_benchmark_figure.py export                     # NIfTI volumes + manifest
    (cd niivue_viewer && npm run build && npm run render-figure)  # NiiVue in headless Chrome -> panels, figure
    python niivue_benchmark_figure.py compose \\
        --paper-figure ../../../GP-Bayes-Refactor/manuscript_v2/figures/selected/tumor_multidose_mri.png

Defaults: multi-dose chemotherapy on segmented scans, acquisition 49 (the acquisition of the paper's
dose-transfer figure), day 110, every future pulse strength. Rows are the full-order truth, the production
posterior point forecast and the Neural-ODE headline center (loss-filtered median); both forecasts are
decoded with the acquisition's POD basis (D q + shift, unclipped). Truth uses the evaluator's cubic time
interpolation of the cached fields.

export --style mri (default) overlays every field on the TumorTwin demonstration patient's T1 post-contrast
MRI, on the in-plane slice with the largest summed true burden. The simulation grid is TumorTwin's crop of
that image grid (checked voxel for voxel against the breast mask); the source NIfTI stores no orientation
(qform and sform codes 0), so the volumes use image-grid axes and the figure has no anatomical labels. The
browser lays out the whole figure; compose copies it and writes the provenance sidecar.
export --style volume writes the earlier lesion-cropped volume renderings on the simulation grid (a
spacing-scaled affine, not a registration to patient anatomy), which compose assembles with matplotlib.
"""
import argparse
from pathlib import Path

import benchmark_environment

benchmark_environment.apply()

import numpy as np  # noqa: E402

import benchmark_data as bd  # noqa: E402
import benchmark_evaluation as be  # noqa: E402
from benchmark_cases import CASES, OBSERVATION_MODELS, get_case  # noqa: E402

EXPORT = bd.SCRIPT_DIR / "niivue_viewer" / "public" / "data" / "figure"
COORDINATES = "simulation grid; not registered patient coordinates"
IMAGE_COORDINATES = "TumorTwin image grid (voxel index x spacing); orientation not in source"
ANATOMY = "T1 post-contrast MRI, TumorTwin TNBC demonstration patient"
# Figure rows, top to bottom (a list: the manifest is written with sorted keys).
ROWS = [["truth", "Full-order\ntruth"], ["production", "Ours"], ["neural_ode", "Neural ODE"]]
RENDER = {"width": 720, "height": 560, "azimuth": 110., "elevation": 20., "zoom": 1.25, "illumination": .6,
          "background": [.035, .045, .06, 1.], "colormap": "inferno", "clip": [0., 290., -20.], "margin_voxels": 3}
# MRI style: in-plane field of view centred on the lesion; overlay colours from 0, per-voxel alpha
# (u / threshold)^2 capped at 1, the whole overlay drawn at opacity; the browser layout (CSS px at 100 per inch,
# captured at figure_scale x).
MRI = {"fov_mm": [176., 132.], "px_per_mm": 4., "background": [0., 0., 0., 1.], "colormap": "redyell",
       "threshold": .05, "opacity": .7, "anatomy_percentile": 99.5, "scale_bar_mm": 20.,
       "figure_width_px": 660, "figure_scale": 4}


def truth_fields(case, strength, days):
    """Full-order field (voxels x days) of one arm, exactly as the evaluator's reference builds it."""
    history, geometry = bd.chemo_history(case)
    days = np.asarray(days, dtype=float)
    weights = bd.temporal_weights(bd.SOURCE_KNOTS, days)
    start = int(round(case.training_span[1] / .5))
    branch = None if strength == case.training_strength else bd.chemo_branch(case, strength)
    fields = np.empty((len(history), len(days)))
    for left in range(0, len(history), bd.BLOCK):
        selected = slice(left, min(left + bd.BLOCK, len(history)))
        old = history[selected].astype(float)
        fields[selected] = old @ weights
        if branch is not None:
            delta = (branch[selected].astype(float) - old[:, start:]) @ weights[start:]
            delta[:, days < bd.first_future_dose(case)] = 0.
            fields[selected] += delta
    return fields, geometry


def time_index(times, day):
    hits = np.flatnonzero(np.isclose(times, day, rtol=0., atol=1e-9))
    if len(hits) != 1:
        nearest = np.sort(times[np.argsort(np.abs(times - day))[:2]])
        raise ValueError(f"Day {day:g} is not a saved evaluation time; the nearest are "
                         f"{nearest[0]:.6g} and {nearest[1]:.6g}.")
    return int(hits[0])


def lesion_crop(volumes, shape, margin):
    """Bounding box (with margin) of every voxel above 1% of the largest displayed value."""
    stack = np.stack([np.abs(v).reshape(shape) for v in volumes])
    occupied = np.argwhere((stack > 1e-2 * stack.max()).any(axis=0))
    low = np.maximum(occupied.min(axis=0) - margin, 0)
    high = np.minimum(occupied.max(axis=0) + margin + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def save_display_volume(path, data, spacing, origin, description):
    """NIfTI with spacing-scaled axes whose first voxel sits at origin (voxel indices of the parent grid)."""
    import nibabel as nib
    affine = np.diag([*spacing, 1.])
    affine[:3, 3] = [index * step for index, step in zip(origin, spacing)]
    image = nib.Nifti1Image(data, affine)
    # Code 0 avoids claiming scanner/anatomical registration; code 2 is the aligned display transform.
    image.set_qform(affine, code=0)
    image.set_sform(affine, code=2)
    image.header.set_xyzt_units("mm", "unknown")
    image.header["descrip"] = description.encode("ascii")
    nib.save(image, path)
    return {"file": path.name, "min": float(data.min()), "max": float(data.max())}


def save_volume(path, field, shape, spacing, crop):
    data = np.asarray(field, dtype=np.float32).reshape(shape)[crop]
    return save_display_volume(path, data, spacing, [part.start for part in crop], COORDINATES)


def simulation_box(geometry):
    """Slices of the simulation grid inside the demonstration patient's images, and the T1 post-contrast image.

    TumorTwin crops every image of the patient to one box (the enhancing region of all visits plus padding)
    before simulating, so the simulation's breast mask must reappear exactly inside the uncropped mask."""
    import nibabel as nib
    patient = bd._patient()
    box = tuple(slice(int(part.start), int(part.stop)) for part in patient.crop_bounding_box)
    tissue = np.asanyarray(nib.load(patient.breastmask).dataobj) > 0
    anatomy = nib.load(patient.T1_post)
    grid = np.asarray(geometry["breast_mask"], dtype=bool).reshape(tuple(int(n) for n in geometry["grid_shape"]))
    if anatomy.shape != tissue.shape or not np.array_equal(tissue[box], grid):
        raise ValueError("The simulation grid is not TumorTwin's crop of the patient images (breast masks differ).")
    if not np.allclose(anatomy.header.get_zooms()[:3], geometry["spacing"]):
        raise ValueError("The MRI and the simulation grid have different voxel sizes.")
    return box, anatomy, tissue, Path(patient.T1_post)


def display_window(burden, box, image_shape, fov_voxels):
    """In-plane slice with the largest burden, and an in-plane field of view centred on its centroid there.

    burden is non-negative on the simulation grid, which occupies box of the image grid. Returns the image voxel
    (centroid rounded, slice) and the field of view (fov_voxels in-plane, clipped inside the image; every slice)."""
    k = int(np.argmax(burden.sum(axis=(0, 1))))
    plane = burden[:, :, k]
    centre = [float((plane * index).sum() / plane.sum()) + part.start
              for index, part in zip(np.indices(plane.shape), box)]
    fov = []
    for middle, size, extent in zip(centre, fov_voxels, image_shape):
        if size > extent:
            raise ValueError(f"A {size}-voxel field of view does not fit in an image of {extent} voxels.")
        start = min(max(int(round(middle - size / 2)), 0), extent - size)
        fov.append(slice(start, start + size))
    fov.append(slice(0, image_shape[2]))
    return [int(round(centre[0])), int(round(centre[1])), k + box[2].start], tuple(fov)


def embed(field, shape, box, fov):
    """A simulation-grid field on the field-of-view grid; zero where the field of view leaves the simulation box."""
    grid = np.asarray(field, dtype=np.float32).reshape(shape)
    out = np.zeros(tuple(part.stop - part.start for part in fov), dtype=np.float32)
    source, target = [], []
    for inner, outer in zip(box, fov):
        low, high = max(inner.start, outer.start), min(inner.stop, outer.stop)
        if high <= low:
            return out
        source.append(slice(low - inner.start, high - inner.start))
        target.append(slice(low - outer.start, high - outer.start))
    out[tuple(target)] = grid[tuple(source)]
    return out


def columns(manifest):
    """Figure title and one (arm, day, label) per column, from the manifest panels."""
    shown = [(item["arm"], item["day"], item["strength"]) for item in manifest["panels"] if item["row"] == "truth"]
    future = ", ".join(f"{day:g}" for day in manifest["dose_days"] if day >= manifest["training_span"][1])
    title = f"Strength of the future pulses (days {future}); forecast on day {shown[0][1]:g}"
    return title, [(arm, day, f"{strength:g}×" + (" (as trained)" if strength == manifest["training_strength"]
                                                   else "")) for arm, day, strength in shown]


def forecast_fields(args):
    case = get_case(args.case, args.observation)
    if not case.dose_days:
        raise ValueError("The NiiVue benchmark figure covers the chemotherapy benchmarks.")
    folder = bd.seed_directory(case, args.seed, root=args.output_root)
    data = bd.load(case, args.seed, root=args.output_root, references=False)
    basis = bd.read_npz(folder / "data" / "basis.npz")
    D, shift = basis["physical_matrix"], basis["physical_shift"]
    production = bd.read_npz(folder / "evaluation" / "production.npz")
    node = bd.read_npz(folder / "evaluation" / "neural_ode.npz")
    center = be.headline_node_center(case)
    if not np.array_equal(production["times"], node["times"]):
        raise ValueError("Production and Neural-ODE evaluations use different time grids.")
    indices = [time_index(production["times"], day) for day in args.days]
    arms = sorted(bd.arms(case).items(), key=lambda item: item[1])
    fields, geometry = {}, None
    for arm, strength in arms:
        print(f"  truth {bd.arm_label(strength)} ...")
        truth, geometry = truth_fields(case, strength, args.days)
        for column, (day, index) in enumerate(zip(args.days, indices)):
            fields[arm, day] = {"truth": truth[:, column],
                                "production": D @ production[f"{arm}_point"][:, index] + shift,
                                "neural_ode": D @ node[f"{arm}_{center}"][:, index] + shift}
    return case, folder, data, center, fields, geometry


def write_panels(case, fields, cal_min, cal_max, colormap, save):
    panels = []
    for (arm, day), panel in fields.items():
        truth = panel["truth"]
        for row, field in panel.items():
            key = f"{arm}_day{day:g}_{row}"
            record = save(EXPORT / f"{key}.nii.gz", field)
            error = None if row == "truth" else float(100 * np.linalg.norm(field - truth) / np.linalg.norm(truth))
            panels.append({"id": key, "row": row, "arm": arm, "day": day, "strength": bd.arms(case)[arm],
                           "cal_min": cal_min, "cal_max": cal_max, "colormap": colormap,
                           "relative_field_error_percent": error, **record})
    return panels


def mri_export(case, fields, geometry, cal_max, manifest):
    """Anatomy and overlays on one in-plane field of view of the patient's image grid, plus the figure text."""
    shape = tuple(int(n) for n in geometry["grid_shape"])
    spacing = [float(h) for h in geometry["spacing"]]
    box, anatomy, tissue, source = simulation_box(geometry)
    fov_voxels = [int(round(mm / step)) for mm, step in zip(MRI["fov_mm"], spacing)]
    burden = sum(np.clip(panel["truth"], 0., None) for panel in fields.values()).reshape(shape)
    voxel, fov = display_window(burden, box, anatomy.shape, fov_voxels)
    origin = [part.start for part in fov]
    image = np.asanyarray(anatomy.dataobj)
    window = [0., float(np.percentile(image[tissue], MRI["anatomy_percentile"]))]
    record = save_display_volume(EXPORT / "anatomy_T1_post.nii.gz", image[fov], spacing, origin, ANATOMY)
    panels = write_panels(case, fields, MRI["threshold"], cal_max, MRI["colormap"], lambda path, field: (
        save_display_volume(path, embed(field, shape, box, fov), spacing, origin, IMAGE_COORDINATES)))
    width_mm, height_mm = (n * step for n, step in zip(fov_voxels, spacing))
    render = {**MRI, "width": round(width_mm * MRI["px_per_mm"]), "height": round(height_mm * MRI["px_per_mm"]),
              "fov_width_mm": width_mm, "slice_voxel": [v - o for v, o in zip(voxel, origin)]}
    title, shown = columns({**manifest, "panels": panels})
    ids = {(item["arm"], item["day"], item["row"]): item["id"] for item in panels}
    ticks = np.round(np.arange(0., cal_max + 1e-9, .1), 10)
    day = shown[0][1]
    return {
        "coordinates": IMAGE_COORDINATES,
        "anatomy": {**record, "cal_min": window[0], "cal_max": window[1], "description": ANATOMY,
                    "window": f"0 to the {MRI['anatomy_percentile']:g}th percentile inside the breast mask",
                    "source": bd.file_record(source)},
        "image_grid": {"shape": list(anatomy.shape), "simulation_box": [[p.start, p.stop] for p in box],
                       "field_of_view": [[p.start, p.stop] for p in fov], "slice_voxel": voxel,
                       "slice": "largest summed true burden over the shown panels; in-plane centroid on it",
                       "check": "the simulation grid's breast mask equals the patient's mask inside the box"},
        "figure": {"title": title, "width_px": MRI["figure_width_px"], "scale": MRI["figure_scale"],
                   "columns": [{"label": label, "panels": {row: ids[arm, day, row] for row, _ in ROWS}}
                               for arm, day, label in shown],
                   "corner_label": "T1 post-contrast MRI", "scale_bar_mm": MRI["scale_bar_mm"],
                   "colorbar": {"max": cal_max, "ticks": [float(t) for t in ticks],
                                "label": ["Tumour volume fraction ", "u", f" on day {day:g}"]}},
        "render": render, "panels": panels,
    }


def export(args):
    case, folder, data, center, fields, geometry = forecast_fields(args)
    shape = tuple(int(n) for n in geometry["grid_shape"])
    spacing = [float(h) for h in geometry["spacing"]]
    cal_max = max(float(panel["truth"].max()) for panel in fields.values())
    EXPORT.mkdir(parents=True, exist_ok=True)
    for stale in [*EXPORT.glob("*.nii.gz"), *EXPORT.glob("panels/*.png"), *EXPORT.glob("panels/*.json"),
                  EXPORT / "manifest.json"]:
        stale.unlink(missing_ok=True)
    manifest = {
        "style": args.style, "case": case.name, "seed": args.seed, "acquisition": bd.acquisition_tag(case),
        "observation_model": case.observation, "training_span": list(case.training_span),
        "training_strength": case.training_strength, "dose_days": list(case.dose_days),
        "days": list(args.days), "neural_ode_center": center, "rows": ROWS,
        "grid_shape": list(shape), "spacing": spacing,
        "decoder": "D q + shift with the acquisition's POD basis; unclipped (values below cal_min are transparent)",
        "relative_field_error": "100 ||decoded - truth|| / ||truth|| over every voxel at that day",
        "sources": {name: bd.file_record(folder / "evaluation" / f"{name}.npz")
                    for name in ("production", "neural_ode")} | {"observations": data["metadata"]["sources"]},
    }
    if args.style == "mri":
        manifest |= mri_export(case, fields, geometry, cal_max, manifest)
        grid = manifest["image_grid"]
        where = f"slice {grid['slice_voxel'][2]}, field of view {grid['field_of_view']}"
    else:
        crop = lesion_crop([v for panel in fields.values() for v in panel.values()], shape, RENDER["margin_voxels"])
        panels = write_panels(case, fields, 0., cal_max, RENDER["colormap"],
                              lambda path, field: save_volume(path, field, shape, spacing, crop))
        manifest |= {"coordinates": COORDINATES, "crop": [[p.start, p.stop] for p in crop], "render": RENDER,
                     "panels": panels}
        where = f"crop {manifest['crop']}"
    bd.write_json(EXPORT / "manifest.json", manifest)
    print(f"  {len(manifest['panels'])} volumes ({args.style}), {where}, colour range 0-{cal_max:.3f}: {EXPORT}")
    for item in manifest["panels"]:
        if item["relative_field_error_percent"] is not None:
            print(f"    {item['id']:<36s} {item['relative_field_error_percent']:6.2f}%")


def _colormap(table):
    from matplotlib.colors import LinearSegmentedColormap
    nodes = np.asarray(table["I"], dtype=float) / 255.
    rgb = np.stack([table[c] for c in "RGB"], axis=1) / 255.
    return LinearSegmentedColormap.from_list(table.get("name", "niivue"), list(zip(nodes, rgb)))


def _content_box(images, background, pad=6):
    """Shared pixel box of everything that differs from the render background."""
    mask = np.zeros(images[0].shape[:2], dtype=bool)
    for image in images:
        mask |= np.abs(image[..., :3] - background[:3]).max(axis=2) > 2 / 255
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return (slice(max(rows[0] - pad, 0), rows[-1] + pad + 1), slice(max(cols[0] - pad, 0), cols[-1] + pad + 1))


def provenance(manifest, render):
    fields = ("id", "relative_field_error_percent", "min", "max")
    record = {key: manifest[key] for key in manifest if key not in ("panels", "render")}
    return record | {"render": render, "panels": [{key: item[key] for key in fields} for item in manifest["panels"]]}


def place_figure(args, manifest):
    """MRI style: the browser has laid out the figure; copy it with its resolution and write the sidecar."""
    from PIL import Image
    rendered, settings = EXPORT / "panels" / "figure.png", EXPORT / "panels" / "render.json"
    if not rendered.exists() or rendered.stat().st_mtime < (EXPORT / "manifest.json").stat().st_mtime:
        raise FileNotFoundError(f"{rendered} is missing or older than the manifest; run npm run render-figure.")
    dpi = 100 * manifest["figure"]["scale"]
    first = manifest["panels"][0]
    outputs = [bd.FIGURES / "segmented" / manifest["case"] /
               f"niivue_mri_day{first['day']:g}_seed{manifest['seed']}.png"]
    outputs += [args.paper_figure] if args.paper_figure else []
    with Image.open(rendered) as image:
        for path in outputs:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            image.save(path, dpi=(dpi, dpi))
            print(f"  figure: {path} ({image.width}x{image.height} px, {dpi} dpi)")
    bd.write_json(outputs[0].with_suffix(".json"), provenance(manifest, bd.read_json(settings)))


def compose(args):
    manifest = bd.read_json(EXPORT / "manifest.json")
    if manifest.get("style") == "mri":
        return place_figure(args, manifest)
    from core.plotting.style import apply_style, save_figure
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    apply_style()
    tables = bd.read_json(EXPORT / "panels" / "colormaps.json")
    rendered = EXPORT / "panels" / "render.json"
    render = bd.read_json(rendered) if rendered.exists() else manifest["render"]
    panels = {item["id"]: item for item in manifest["panels"]}
    images = {key: plt.imread(EXPORT / "panels" / f"{key}.png") for key in panels}
    box = _content_box(list(images.values()), np.asarray(render["background"]))
    labels = dict(manifest["rows"])
    rows = list(labels)
    title, shown = columns(manifest)
    height, width = images[next(iter(images))][box].shape[:2]
    fig_width = 6.6
    fig_height = fig_width * .9 * len(rows) / len(shown) * height / width + 1.
    top = 1 - .45 / fig_height
    fig, axes = plt.subplots(len(rows), len(shown), figsize=(fig_width, fig_height), squeeze=False,
                             gridspec_kw={"wspace": .03, "hspace": .05, "top": top})
    fig.suptitle(title, fontsize=9, y=top + .24 / fig_height, va="bottom")
    for c, (arm, day, label) in enumerate(shown):
        axes[0, c].set_title(label, fontsize=8.5, pad=3)
        for r, row in enumerate(rows):
            key = f"{arm}_day{day:g}_{row}"
            ax = axes[r, c]
            ax.imshow(images[key][box], interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            error = panels[key]["relative_field_error_percent"]
            if error is not None:
                ax.text(.97, .05, f"{error:.1f}%", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                        color="0.92")
            if c == 0:
                ax.set_ylabel(labels[row], fontsize=8.5, labelpad=3)
    first = manifest["panels"][0]
    colormap = render.get("colormap", first["colormap"])
    mappable = ScalarMappable(Normalize(render.get("cal_min", first["cal_min"]), first["cal_max"]),
                              _colormap(tables[colormap]))
    bar = fig.colorbar(mappable, ax=axes, orientation="horizontal", fraction=.035, pad=.02, aspect=45)
    bar.set_label(f"Tumour volume fraction $u$ on day {first['day']:g}", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    outputs = [bd.FIGURES / "segmented" / manifest["case"] / f"niivue_day{first['day']:g}_seed{manifest['seed']}.png"]
    outputs += [args.paper_figure] if args.paper_figure else []
    for index, path in enumerate(outputs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, str(path), dpi=300, close=index == len(outputs) - 1)
        print(f"  figure: {path}")
    bd.write_json(outputs[0].with_suffix(".json"), provenance(manifest, render))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export", help="Write NIfTI panels and the render manifest.")
    exporter.add_argument("case", nargs="?", default="multi-dose-chemo",
                          choices=[name for name, case in CASES.items() if case.dose_days])
    exporter.add_argument("--seed", type=int, default=49, help="Acquisition (default: 49, as in the paper).")
    exporter.add_argument("--days", type=float, nargs="+", default=[110.],
                          help="Saved evaluation days to show (default: 110, the end of the forecast).")
    exporter.add_argument("--observation", choices=OBSERVATION_MODELS)
    exporter.add_argument("--output-root", type=Path, help=f"Results root (default: {bd.RESULTS}).")
    exporter.add_argument("--style", choices=("mri", "volume"), default="mri",
                          help="mri: slices over the patient's T1 MRI (default); volume: lesion volume renderings.")
    composer = commands.add_parser("compose", help="Write the figure (MRI style: laid out by render-figure).")
    composer.add_argument("--paper-figure", type=Path, help="Also write the figure to this path.")
    args = parser.parse_args()
    export(args) if args.command == "export" else compose(args)


if __name__ == "__main__":
    main()
