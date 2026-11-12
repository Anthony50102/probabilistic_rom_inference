"""Lift saved mean-operator trajectories onto the FOM grid; stream movies without refitting."""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation, colors, pyplot as plt
from matplotlib.patches import Circle
import numpy as np


def snapshot_window(path, start, stop):
    """Read only a contiguous time window from a compressed, row-major snapshot array."""
    with zipfile.ZipFile(path) as archive, archive.open("snapshots.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported snapshot NPY version: {version}")
        if len(shape) != 2 or fortran or dtype.hasobject:
            raise ValueError("Snapshots must be a numeric, row-major time-by-state array")
        if not 0 <= start < stop <= shape[0]:
            raise ValueError(f"Invalid snapshot window [{start}, {stop}) for {shape}")
        row_bytes = shape[1] * dtype.itemsize
        stream.seek(stream.tell() + start * row_bytes)
        payload = stream.read((stop - start) * row_bytes)
    if len(payload) != (stop - start) * row_bytes:
        raise ValueError("Truncated snapshot window")
    return np.frombuffer(payload, dtype=dtype).reshape(stop - start, shape[1])


def vorticity(velocity, xs, ys, inside):
    omega = (np.gradient(velocity[:, 1], xs, axis=2, edge_order=2)
             - np.gradient(velocity[:, 0], ys, axis=1, edge_order=2))
    mask = inside.copy()
    mask[1:] |= inside[:-1]
    mask[:-1] |= inside[1:]
    mask[:, 1:] |= inside[:, :-1]
    mask[:, :-1] |= inside[:, 1:]
    omega[:, mask] = np.nan
    return omega.astype(np.float32)


def load_case(fit, fom_dir, basis, plots, row, k, start, train):
    re = int(row["re"])
    path = fom_dir / f"medium_re{re}.npz"
    with np.load(path) as data:
        absolute_t = data["t"]
        xs, ys, inside = data["xs"], data["ys"], data["inside"]
        if "geometry" in data and str(data["geometry"]) != "cylinder":
            raise ValueError("This renderer is for the single-cylinder geometry")
        cells = int(data["cells"])
    t = plots[f"t_{k}"]
    indices = np.searchsorted(absolute_t, start + t - 1e-8)
    np.testing.assert_allclose(absolute_t[indices], start + t, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(np.diff(indices), 1)
    truth = snapshot_window(path, int(indices[0]), int(indices[-1]) + 1).astype(float).T
    if truth.shape != (2 * len(xs) * len(ys), len(t)):
        raise ValueError("FOM grid and velocity snapshot dimensions disagree")
    projected = basis.T @ truth
    np.testing.assert_allclose(projected, plots[f"truth_{k}"], rtol=1e-8, atol=1e-8)
    mean = basis @ plots[f"rao_blackwell_{k}"]
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"Nonfinite reconstructed mean for Re {re}")
    difference = mean - truth
    score = t > (0 if row["held_out"] else train)
    scored_truth = truth[:, score]
    error_squared = np.sum(difference[:, score] ** 2)
    fluct_squared = np.sum((scored_truth - scored_truth.mean(axis=1, keepdims=True)) ** 2)
    relative_error = float(np.sqrt(error_squared / fluct_squared))
    np.testing.assert_allclose(relative_error, row["means"]["rao_blackwell"]["error"],
                               rtol=1e-7, atol=1e-10)
    shape = (len(t), 2, len(ys), len(xs))
    omega_truth = vorticity(truth.T.reshape(shape), xs, ys, inside)
    omega_rom = vorticity(mean.T.reshape(shape), xs, ys, inside)
    velocity_error = np.sqrt(np.sum(difference.T.reshape(shape) ** 2, axis=1)).astype(np.float32)
    velocity_error[:, inside] = np.nan
    metrics = dict(re=re, held_out=bool(row["held_out"]), full_state_values=truth.shape[0],
                   mesh_cells=cells, grid_shape=list(inside.shape),
                   fluctuation_error_pct=100 * relative_error,
                   absolute_state_error_pct=100 * float(np.sqrt(error_squared / np.sum(scored_truth ** 2))),
                   basis_floor_pct=100 * row["floor"], source_fit=str(fit),
                   source_fom=str(path), mean="Rao-Blackwell mean-operator trajectory, not mean trajectory",
                   score_time_interval=[start + (0 if row["held_out"] else train), start + float(t[-1])],
                   score_left_endpoint_excluded=True)
    return dict(re=re, t=start + t, train_end=start + train, xs=xs, ys=ys, metrics=metrics,
                fields=(omega_truth, omega_rom, velocity_error),
                error_max=float(np.nanmax(velocity_error)))


def field_panel(ax, case, row, index, limit):
    xs, ys = case["xs"], case["ys"]
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    np.testing.assert_allclose(np.diff(xs), dx)
    np.testing.assert_allclose(np.diff(ys), dy)
    norm = colors.Normalize(0, case["error_max"]) if row == 2 else colors.Normalize(-limit, limit)
    image = ax.imshow(case["fields"][row][index], origin="lower",
                      extent=(xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2),
                      cmap="magma" if row == 2 else "RdBu_r", norm=norm, interpolation="nearest")
    ax.set_facecolor("#dedede")
    ax.add_patch(Circle((.2, .2), .05, facecolor="#777777", edgecolor="black", linewidth=.7))
    ax.set(xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]), ylabel="y")
    ax.tick_params(labelsize=8)
    return image


def render_case(case, output, limit, stride, fps):
    re = case["re"]
    m = case["metrics"]
    labels = ("Full-order simulation: vorticity", "29-mode ROM reconstruction: vorticity",
              "Velocity-vector error magnitude")
    heading = (f"Re {re}" + (" (held out)" if m["held_out"] else " (training Re)")
               + f" | scored fluctuation error {m['fluctuation_error_pct']:.2f}%")
    footer = (f"Mean-operator forecast, not an uncertainty band. Vorticity clipped at +/-{limit:g}; "
              "finite differences masked next to solid.\n"
              "Color limits fixed over time. Bottom panel uses velocity error, not vorticity error.")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.subplots_adjust(left=.09, right=.9, bottom=.15, top=.87, hspace=.5)
    images = []
    for row, ax in enumerate(axes):
        image = field_panel(ax, case, row, 0, limit)
        images.append(image)
        ax.set_title(labels[row], fontsize=11, loc="left")
        fig.colorbar(image, ax=ax, fraction=.025, pad=.025,
                     extend="neither" if row == 2 else "both")
    axes[-1].set_xlabel("x (solver coordinates)")
    title = fig.suptitle(heading, fontsize=14)
    fig.text(.5, .025, footer, ha="center", fontsize=9)

    def update(index):
        phase = ("held-out rollout" if m["held_out"] else
                 "training window" if case["t"][index] <= case["train_end"] + 1e-8 else "forecast")
        title.set_text(f"{heading}\nt = {case['t'][index]:.3f} | {phase}")
        for image, values in zip(images, case["fields"]):
            image.set_data(values[index])

    update(len(case["t"]) - 1)
    fig.savefig(output / f"fullfield_re{re}_final.png", dpi=160)
    frame_indices = np.unique(np.r_[np.arange(0, len(case["t"]), stride), len(case["t"]) - 1])
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-threads", "1", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(output / f"fullfield_re{re}.mp4"), dpi=120):
        for index in frame_indices:
            update(index)
            writer.grab_frame()
    plt.close(fig)
    return dict(frames=len(frame_indices), fps=fps, duration_seconds=len(frame_indices) / fps)


def render_comparison(cases, output, limit):
    fig, axes = plt.subplots(3, len(cases), figsize=(18, 6), squeeze=False)
    fig.subplots_adjust(left=.055, right=.98, bottom=.14, top=.83, hspace=.65, wspace=.28)
    for k, case in enumerate(cases):
        for row in range(3):
            ax = axes[row, k]
            image = field_panel(ax, case, row, -1, limit)
            fig.colorbar(image, ax=ax, fraction=.022, pad=.02,
                         extend="neither" if row == 2 else "both")
            if k == 0:
                ax.set_ylabel(("FOM vorticity", "ROM vorticity", "Velocity error")[row] + "\ny")
            if row == 2:
                ax.set_xlabel("x")
        m = case["metrics"]
        axes[0, k].set_title(f"Re {case['re']}" + (" (held out)" if m["held_out"] else " (training Re)")
                             + f"\nScored fluctuation error: {m['fluctuation_error_pct']:.2f}%", fontsize=11)
    fig.suptitle(f"Full-order fields versus 29-mode mean-operator reconstruction at t={cases[0]['t'][-1]:g}",
                 fontsize=15)
    training = cases[0]["metrics"]["score_time_interval"]
    held = cases[-1]["metrics"]["score_time_interval"]
    fig.text(.5, .025,
             f"Shared vorticity range +/-{limit:g}; values beyond limits saturate. "
             "Velocity-error scales differ by Re, but are fixed over time.\n"
             f"Re150/300 scored on ({training[0]:g},{training[1]:g}]; "
             f"held-out Re200 scored on ({held[0]:g},{held[1]:g}]. "
             "These are mean-operator fields, not posterior uncertainty bands.",
             ha="center", fontsize=9)
    fig.savefig(output / "fullfield_comparison.png", dpi=160)
    plt.close(fig)


def self_check():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "snapshots.npz"
        values = np.arange(60, dtype=np.float32).reshape(10, 6)
        np.savez_compressed(path, snapshots=values)
        np.testing.assert_array_equal(snapshot_window(path, 3, 7), values[3:7])
        np.testing.assert_array_equal(snapshot_window(path, 0, 10), values)
    xs, ys = np.linspace(0, 2, 11), np.linspace(0, 1, 7)
    x, y = np.meshgrid(xs, ys)
    mask = np.zeros(x.shape, dtype=bool)
    mask[3, 5] = True
    omega = vorticity(np.array([[-y, x]]), xs, ys, mask)
    np.testing.assert_allclose(omega[np.isfinite(omega)], 2, atol=1e-6)
    if not np.all(np.isnan(omega[0, [2, 3, 3, 3, 4], [5, 4, 5, 6, 5]])):
        raise AssertionError("Solid-adjacent derivative stencil was not masked")
    print("Compressed-window and vorticity checks passed")


def main(args):
    if args.check:
        self_check()
        return
    if not all((args.fit, args.fom_dir, args.output)):
        raise ValueError("--fit, --fom-dir and --output are required for rendering")
    if args.stride < 1 or args.fps < 1 or not np.isfinite(args.vorticity_limit) or args.vorticity_limit <= 0:
        raise ValueError("Stride, fps and vorticity limit must be positive")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required for streaming MP4 output")
    summary = json.loads((args.fit / "summary.json").read_text())
    request = json.loads((args.fit / "request.json").read_text())
    if request["config"]["num_modes"] != 29 or request["config"]["feature_shift"] != "none":
        raise ValueError("This report requires the unshifted 29-mode fit")
    if [int(row["re"]) for row in summary["targets"]] != [150, 300, 200]:
        raise ValueError("Expected the two training Re and held-out Re200, in saved target order")
    start, train, _ = request["window"]
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.fit / "posterior.npz") as data:
        basis = data["basis_entries"]
    cases = []
    with np.load(args.fit / "plotdata.npz") as plots:
        for k, row in enumerate(summary["targets"]):
            case = load_case(args.fit, args.fom_dir, basis, plots, row, k, start, train)
            case["metrics"]["movie"] = render_case(case, args.output, args.vorticity_limit, args.stride, args.fps)
            cases.append(case)
            print(json.dumps(case["metrics"]), flush=True)
    render_comparison(cases, args.output, args.vorticity_limit)
    report = dict(visualization="finite-difference vorticity from gridded velocity",
                  vorticity_limit=args.vorticity_limit, error_limits="per Re, fixed over time",
                  no_refitting=True, cases=[case["metrics"] for case in cases])
    (args.output / "fullfield_metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--fom-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--vorticity-limit", type=float, default=40.)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args())
