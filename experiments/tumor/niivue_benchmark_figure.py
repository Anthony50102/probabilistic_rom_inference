"""
niivue_benchmark_figure.py — NiiVue volume renderings of a chemotherapy benchmark forecast.

No fitting and no FOM simulation: reads the cached full-order fields and the evaluations written by
04_unified_benchmark.py and 05_neural_ode_benchmark.py. Three steps, from experiments/tumor:

    python niivue_benchmark_figure.py export                     # NIfTI panels + manifest
    (cd niivue_viewer && npm run build && npm run render-figure)  # NiiVue in headless Chrome -> PNG panels
    python niivue_benchmark_figure.py compose \\
        --paper-figure ../../../GP-Bayes-Refactor/manuscript_v2/figures/selected/tumor_multidose_niivue.png

Defaults: multi-dose chemotherapy on segmented scans, acquisition 49 (the acquisition of the paper's
dose-transfer figure), day 110, every future pulse strength. Rows are the full-order truth, the production
posterior point forecast and the Neural-ODE headline center (loss-filtered median); both forecasts are
decoded with the acquisition's POD basis (D q + shift, unclipped). Truth uses the evaluator's cubic time
interpolation of the cached fields. Volumes stay on the simulation grid, cropped to the lesion; the affine
is spacing-scaled and is not a registration to patient anatomy.
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
# Figure rows, top to bottom (a list: the manifest is written with sorted keys).
ROWS = [["truth", "Full-order\ntruth"], ["production", "Ours"], ["neural_ode", "Neural ODE"]]
RENDER = {"width": 720, "height": 560, "azimuth": 110., "elevation": 20., "zoom": 1.25, "illumination": .6,
          "background": [.035, .045, .06, 1.], "colormap": "inferno", "clip": [0., 290., -20.], "margin_voxels": 3}


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


def save_volume(path, field, shape, spacing, crop):
    import nibabel as nib
    data = np.asarray(field, dtype=np.float32).reshape(shape)[crop]
    affine = np.diag([*spacing, 1.])
    affine[:3, 3] = [part.start * step for part, step in zip(crop, spacing)]
    image = nib.Nifti1Image(data, affine)
    # Code 0 avoids claiming scanner/anatomical registration; code 2 is the aligned display transform.
    image.set_qform(affine, code=0)
    image.set_sform(affine, code=2)
    image.header.set_xyzt_units("mm", "unknown")
    image.header["descrip"] = COORDINATES.encode("ascii")
    nib.save(image, path)
    return {"file": path.name, "min": float(data.min()), "max": float(data.max())}


def export(args):
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
    shape = tuple(int(n) for n in geometry["grid_shape"])
    spacing = [float(h) for h in geometry["spacing"]]
    crop = lesion_crop([v for panel in fields.values() for v in panel.values()], shape, RENDER["margin_voxels"])
    cal_max = max(float(panel["truth"].max()) for panel in fields.values())
    EXPORT.mkdir(parents=True, exist_ok=True)
    for stale in [*EXPORT.glob("*.nii.gz"), *EXPORT.glob("panels/*.png"), EXPORT / "manifest.json"]:
        stale.unlink(missing_ok=True)
    panels = []
    for (arm, day), panel in fields.items():
        truth = panel["truth"]
        for row, field in panel.items():
            record = save_volume(EXPORT / f"{arm}_day{day:g}_{row}.nii.gz", field, shape, spacing, crop)
            error = None if row == "truth" else float(100 * np.linalg.norm(field - truth) / np.linalg.norm(truth))
            panels.append({"id": f"{arm}_day{day:g}_{row}", "row": row, "arm": arm, "day": day,
                           "strength": bd.arms(case)[arm], "cal_min": 0., "cal_max": cal_max,
                           "colormap": RENDER["colormap"], "relative_field_error_percent": error, **record})
    manifest = {
        "case": case.name, "seed": args.seed, "acquisition": bd.acquisition_tag(case),
        "observation_model": case.observation, "training_span": list(case.training_span),
        "training_strength": case.training_strength, "dose_days": list(case.dose_days),
        "days": list(args.days), "neural_ode_center": center, "rows": ROWS, "coordinates": COORDINATES,
        "grid_shape": list(shape), "spacing": spacing, "crop": [[p.start, p.stop] for p in crop],
        "decoder": "D q + shift with the acquisition's POD basis; unclipped (values below cal_min are transparent)",
        "relative_field_error": "100 ||decoded - truth|| / ||truth|| over every voxel at that day",
        "sources": {name: bd.file_record(folder / "evaluation" / f"{name}.npz")
                    for name in ("production", "neural_ode")} | {"observations": data["metadata"]["sources"]},
        "render": RENDER, "panels": panels,
    }
    bd.write_json(EXPORT / "manifest.json", manifest)
    print(f"  {len(panels)} volumes, crop {manifest['crop']}, colour range 0-{cal_max:.3f}: {EXPORT}")
    for item in panels:
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


def compose(args):
    from core.plotting.style import apply_style, save_figure
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    apply_style()
    manifest = bd.read_json(EXPORT / "manifest.json")
    tables = bd.read_json(EXPORT / "panels" / "colormaps.json")
    rendered = EXPORT / "panels" / "render.json"
    render = bd.read_json(rendered) if rendered.exists() else manifest["render"]
    panels = {item["id"]: item for item in manifest["panels"]}
    images = {key: plt.imread(EXPORT / "panels" / f"{key}.png") for key in panels}
    box = _content_box(list(images.values()), np.asarray(render["background"]))
    labels = dict(manifest["rows"])
    rows = list(labels)
    columns = [(item["arm"], item["day"], item["strength"]) for item in manifest["panels"] if item["row"] == "truth"]
    height, width = images[next(iter(images))][box].shape[:2]
    fig_width = 6.6
    fig_height = fig_width * .9 * len(rows) / len(columns) * height / width + 1.
    top = 1 - .45 / fig_height
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(fig_width, fig_height), squeeze=False,
                             gridspec_kw={"wspace": .03, "hspace": .05, "top": top})
    future = ", ".join(f"{day:g}" for day in manifest["dose_days"] if day >= manifest["training_span"][1])
    fig.suptitle(f"Strength of the future pulses (days {future}); forecast on day {columns[0][1]:g}",
                 fontsize=9, y=top + .24 / fig_height, va="bottom")
    for c, (arm, day, strength) in enumerate(columns):
        control = " (as trained)" if strength == manifest["training_strength"] else ""
        axes[0, c].set_title(f"{strength:g}×{control}", fontsize=8.5, pad=3)
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
    fields = ("id", "relative_field_error_percent", "min", "max")
    provenance = {key: manifest[key] for key in manifest if key not in ("panels", "render")}
    provenance |= {"render": render, "panels": [{key: item[key] for key in fields} for item in manifest["panels"]]}
    bd.write_json(outputs[0].with_suffix(".json"), provenance)


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
    composer = commands.add_parser("compose", help="Assemble the rendered panels into the figure.")
    composer.add_argument("--paper-figure", type=Path, help="Also write the figure to this path.")
    args = parser.parse_args()
    export(args) if args.command == "export" else compose(args)


if __name__ == "__main__":
    main()
