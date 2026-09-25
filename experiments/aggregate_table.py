"""
aggregate_table.py — the paper's cross-experiment comparison table (Euler, heat, 2D diffusion-reaction).

Reads saved predictions only; nothing is fitted. For every reported regime it needs
<experiment>/results/comparison/<regime>/04_unified.npz (production Bayesian OpInf) and 05_neural_ode.npz:

    (cd euler && python 04_unified.py && python 05_neural_ode.py)        # likewise heat and burgers_2d
    python aggregate_table.py

Metrics are those the runners print, in reduced (POD) coordinates, for the median over the stable posterior draws
(production) or ensemble members (Neural ODE) against the reduced full-order truth:

    train / forecast error   ||median - truth||_F / ||truth||_F over t <= / > the end of the training window
    coverage                 fraction of (mode, t > end of training) inside the empirical 5-95% band of the draws
    width                    mean width of that band over the forecast window (reduced units; compare within a regime)
    width ratio              Neural-ODE width over production width, the band-sharpness column of the paper's table
    stable                   percentage of draws or members whose ROM solve stays finite

Heat values average the five training trajectories; its held-out forcing is reported in separate rows. The error
is not the full-order error: it excludes the POD projection residual.

The production files store the reduced truth, the Neural-ODE files only their ensemble solves. Neural-ODE metrics are
therefore recomputed against the production truth, and the script stops unless they reproduce the train, forecast and
all-time coverage values that the Neural-ODE runner saved, which checks that both runs share the data, basis and
truth. Production metrics are recomputed the same way and checked against their saved values. Writes
results/aggregate/aggregate_table.{csv,json,tex}.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "aggregate"
# (experiment folder, system label, regime, regime label)
REGIMES = [
    ("euler", "Euler", "dense_low_noise", "dense, low noise"),
    ("euler", "Euler", "sparse_low_noise", "sparse, low noise"),
    ("euler", "Euler", "dense_high_noise", "dense, high noise"),
    ("heat", "Heat", "sparse_low_noise", "sparse, low noise"),
    ("heat", "Heat", "sparse_medium_noise", "sparse, medium noise"),
    ("heat", "Heat", "sparse_high_noise", "sparse, high noise"),
    ("burgers_2d", "2D diffusion-reaction", "dense_medium_noise", "dense, medium noise"),
]
METHODS = {"production": "04_unified.npz", "neural_ode": "05_neural_ode.npz"}
TOLERANCE = 1e-6  # relative; float32 ensemble solves and refitted bases differ at ~1e-8


def _relative(estimate, truth):
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def score(solves, truth, t_full, t_pred, span_end):
    """Runner metrics of one trajectory; solves is (draws, modes, len(t_pred))."""
    solves = np.asarray(solves, dtype=float)
    if len(solves) == 0:
        return None
    reference = interp1d(t_full, truth, kind="cubic", fill_value="extrapolate")(t_pred)
    median = np.median(solves, axis=0)
    fit, forecast = t_pred <= span_end, t_pred > span_end
    low, high = np.percentile(solves, 5, axis=0), np.percentile(solves, 95, axis=0)
    inside = (reference >= low) & (reference <= high)
    return {"train": _relative(median[:, fit], reference[:, fit]),
            "forecast": _relative(median[:, forecast], reference[:, forecast]),
            "coverage": float(np.mean(inside[:, forecast])), "coverage_all": float(np.mean(inside)),
            "width": float(np.mean((high - low)[:, forecast])), "width_all": float(np.mean(high - low))}


def _mean(scores):
    kept = [item for item in scores if item is not None]
    if not kept:
        return {key: float("nan") for key in ("train", "forecast", "coverage", "coverage_all", "width", "width_all")}
    return {key: float(np.mean([item[key] for item in kept])) for key in kept[0]}


def _check(label, recomputed, saved):
    for key, value in saved.items():
        if not np.isclose(recomputed[key], float(value), rtol=TOLERANCE, atol=TOLERANCE):
            raise ValueError(f"{label}: recomputed {key} {recomputed[key]:.12g} differs from the saved "
                             f"{float(value):.12g}; the two runs do not share truth and basis, or the file is stale.")


def regime_metrics(folder, regime):
    """Headline (training-trajectory) and, for heat, held-out metrics of both methods for one regime."""
    directory = HERE / folder / "results" / "comparison" / regime
    missing = [str(directory / name) for name in METHODS.values() if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError("Missing saved predictions (run the regime's 04/05 runners first):\n  "
                                + "\n  ".join(missing))
    production = np.load(directory / METHODS["production"], allow_pickle=False)
    node = np.load(directory / METHODS["neural_ode"], allow_pickle=False)
    if "ci_cov_ext" not in production.files:
        raise ValueError(f"{directory / METHODS['production']} predates forecast-window coverage; rerun 04_unified.py.")
    span_end = float(production["training_span"][1])
    multi = "n_ics" in production.files
    count = int(production["n_ics"]) if multi else 1
    trained = int(production["num_train_ics"]) if multi else 1
    key = (lambda name, i: f"{name}_{i}") if multi else (lambda name, i: name)
    if multi and not str(production["eval_labels"][trained]).lower().startswith("test"):
        raise ValueError(f"{directory}: expected the held-out trajectory after the {trained} training ones.")
    if (int(node["n_ics"]) if "n_ics" in node.files else 1) != count:
        raise ValueError(f"{directory}: the methods were evaluated on different numbers of trajectories.")
    t_pred = production[key("t_pred", 0)]
    if not np.allclose(node["t_pred"], t_pred, rtol=0, atol=1e-12):
        raise ValueError(f"{directory}: the methods were evaluated on different time grids.")
    truth = [(production[key("true_comp", i)], production[key("t_full", i)]) for i in range(count)]
    solves = {"production": [production[key("rom_solves", i)] for i in range(count)],
              "neural_ode": [node[f"rom_solves_{i}"] if multi else node["rom_solves"] for i in range(count)]}
    saved = {"production": {"train": production["train_error"], "forecast": production["pred_error"],
                            "coverage": production["ci_cov_ext"], "coverage_all": production["ci_coverage"],
                            "width_all": production["ci_width"]},
             "neural_ode": {"train": node["train_error"], "forecast": node["pred_error"],
                            "coverage_all": node["ci_coverage"], "width_all": node["ci_width"]}}
    stable = {"production": production["stability_pct"], "neural_ode": node["stability_pct"]}
    result = {}
    for method in METHODS:
        scores = [score(solves[method][i], *truth[i], t_pred, span_end) for i in range(count)]
        head = _mean(scores[:trained])
        _check(f"{folder}/{regime} {method}", head, saved[method])
        result[method] = {**head, "stable": float(stable[method]), "held_out": scores[trained] if multi else None}
    if multi:
        _check(f"{folder}/{regime} production held-out", result["production"]["held_out"],
               {"train": production["test_train_error"], "forecast": production["test_pred_error"],
                "coverage": production["test_ci_cov_ext"]})
    return result


def rows(results):
    """Flat table rows: every regime, then the heat held-out trajectory."""
    table = []
    for (folder, system, regime, label), result in zip(REGIMES, results):
        table.append({"system": system, "regime": label, "trajectories": "training", "key": f"{folder}/{regime}",
                      **{f"{method}_{name}": result[method][name] for method in METHODS
                         for name in ("stable", "train", "forecast", "coverage", "width")}})
    for (folder, system, regime, label), result in zip(REGIMES, results):
        if result["production"]["held_out"] is not None:
            table.append({"system": system, "regime": label, "trajectories": "held-out", "key": f"{folder}/{regime}",
                          **{f"{method}_{name}": result[method]["held_out"][name] for method in METHODS
                             for name in ("train", "forecast", "coverage", "width")}})
    for row in table:
        row["width_ratio"] = row["neural_ode_width"] / row["production_width"]
    return table


def _percent(value, digits):
    return "--" if value is None or not np.isfinite(value) else f"{100 * value:.{digits}f}"


def latex(table):
    """Table body rows: errors in % (two decimals), coverage in % and the Neural-ODE over production band-width
    ratio (one decimal); systems named once."""
    lines, last = [], None
    for row in table:
        name = row["system"].replace("-", "--")
        system = name if row["trajectories"] == "training" else f"{name}, held-out"
        if last is not None and system != last:
            lines.append("\\addlinespace")
        cells = ["" if system == last else system, row["regime"]]
        for method in METHODS:
            cells += [_percent(row[f"{method}_train"], 2), _percent(row[f"{method}_forecast"], 2),
                      _percent(row[f"{method}_coverage"], 1)]
        cells.append(f"{row['width_ratio']:.1f}" if np.isfinite(row["width_ratio"]) else "--")
        lines.append(" & ".join(cells) + " \\\\")
        last = system
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT, help=f"Output directory (default: {OUTPUT}).")
    args = parser.parse_args()
    results = [regime_metrics(folder, regime) for folder, _, regime, _ in REGIMES]
    table = rows(results)
    args.output.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(name for row in table for name in row))
    with open(args.output / "aggregate_table.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    (args.output / "aggregate_table.json").write_text(json.dumps(table, indent=2) + "\n")
    (args.output / "aggregate_table.tex").write_text(latex(table))
    header = f"{'system':<24s}{'regime':<22s}{'traj.':<10s}" + "".join(
        f"{method + ' ' + name:>15s}" for method in ("ours", "NODE") for name in ("train", "forecast", "cov."))
    print(header + f"{'width ratio':>13s}")
    for row in table:
        values = "".join(f"{_percent(row[f'{method}_{name}'], 2):>15s}" for method in METHODS
                         for name in ("train", "forecast", "coverage"))
        print(f"{row['system']:<24s}{row['regime']:<22s}{row['trajectories']:<10s}{values}"
              f"{row['width_ratio']:>13.2f}")
    stable = {f"{row['key']} {method}": row[f"{method}_stable"] for row in table if row["trajectories"] == "training"
              for method in METHODS if row[f"{method}_stable"] != 100}
    print("stable draws/members below 100%:", stable or "none")
    print(f"wrote {args.output}/aggregate_table.{{csv,json,tex}}")


if __name__ == "__main__":
    main()
