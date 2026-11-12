"""Reproduce the original forecasts, then distinguish work-budget limits from escape."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from core.weakform_opinf.pipeline import _ic_sigma
from stabilization import integrate


def evaluate(job):
    folder, output = job
    destination = output / f"{folder.name}.json"
    if destination.exists():
        return folder.name + ": already evaluated"
    request = json.loads((folder / "request.json").read_text())
    old = json.loads((folder / "summary.json").read_text())
    cfg = old["config"]
    with np.load(folder / "posterior.npz") as archive:
        keys = ["O_samples", "O_mean_rao_blackwell", "gp_lengthscale_samples",
                "gp_variance_samples", "gp_noise_samples"]
        keys += [f"{key}_{k}" for key in ("t", "t_samp", "snaps_comp") for k in range(3)]
        d = {key: archive[key] for key in keys}
    draws, mean = d["O_samples"], d["O_mean_rao_blackwell"]
    hypers = tuple(d[key].mean(0) for key in
                   ("gp_lengthscale_samples", "gp_variance_samples", "gp_noise_samples"))
    rng = np.random.default_rng(cfg["seed"])
    duration = 10 * request["window"][1]
    budget = 20000 * math.ceil(duration / request["window"][2])
    long_t = np.linspace(0, duration, 1001)
    result = dict(name=folder.name, max_rhs=budget, original_max_rhs=20000, duration=duration, targets=[])
    plots = dict(t=long_t)
    with np.load(folder / "plotdata.npz") as previous:
        for k, row in enumerate(old["targets"]):
            u = 200 / row["re"]
            th = tuple(z.mean(0) if row["held_out"] else z[k] for z in hypers)
            sig = _ic_sigma(d[f"t_samp_{k}"], *th, cfg["num_modes"], roundoff_nugget=True)
            q0s = d[f"snaps_comp_{k}"][:, 0] + sig[None] * rng.standard_normal((len(draws), cfg["num_modes"]))
            q0 = q0s.mean(0)
            cap = 2 * np.max(np.abs(d[f"snaps_comp_{k}"]))
            short, pred = integrate(mean, q0, u, d[f"t_{k}"], cap)
            assert short["bounded"] == row["means"]["rao_blackwell"]["bounded"], (folder.name, k, short)
            if short["bounded"]:
                np.testing.assert_allclose(pred, previous[f"rao_blackwell_{k}"], rtol=1e-8, atol=1e-8)
            original_mean, _ = integrate(mean, q0, u, long_t, cap)
            assert original_mean["bounded"] == row["mean_long"]["bounded"]
            assert original_mean["reason"] == row["mean_long"]["reason"]
            original = [integrate(O, initial, u, long_t, cap)[0]
                        for O, initial in zip(draws[:50], q0s[:50])]
            np.testing.assert_allclose(np.mean([s["bounded"] for s in original]), row["long_survival"], atol=1e-14)
            expanded_mean, pred = integrate(mean, q0, u, long_t, cap, max_rhs=budget)
            if expanded_mean["bounded"]:
                plots[f"mean_{k}"] = pred
            expanded = [integrate(O, initial, u, long_t, cap, max_rhs=budget)[0]
                        for O, initial in zip(draws[:50], q0s[:50])]
            failures = {}
            for status in expanded:
                if not status["bounded"]:
                    failures[status["reason"]] = failures.get(status["reason"], 0) + 1
            result["targets"].append(dict(
                re=row["re"], n_draws=50, survival=float(np.mean([s["bounded"] for s in expanded])),
                failures=failures, mean=expanded_mean, original_mean=original_mean,
                original_draw_status=original, expanded_draw_status=expanded,
                reproduced_original_short_mean_and_long_survival=True))
    np.savez_compressed(output / f"{folder.name}.npz", **plots)
    destination.write_text(json.dumps(result, indent=2))
    return f"{folder.name}: " + str([(x["re"], x["survival"], x["failures"]) for x in result["targets"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fits", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(p, args.output) for p in sorted(args.fits.iterdir()) if (p / "summary.json").exists()]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for message in pool.map(evaluate, jobs):
            print(message, flush=True)
