"""Training-selected deterministic baselines on the frozen cylinder comparison."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.linalg import svd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.cylinder.after_the_fact.stabilization import features, integrate

CASES = ("jac20", "jac20-noise43", "jac20-r29")
REYNOLDS = (150, 300, 200)
METHODS = ("opinf_fd", "sindy_fd", "opinf_gp", "sindy_gp")
REGULARIZED_METHODS = ("sindy_fd_regularized", "sindy_gp_regularized")
RIDGES = np.logspace(-7, 2, 10)
SPARSE_RIDGES = (1e-4, 1e-2, 1e-1)
THRESHOLDS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1., 3., 10.)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(fits, fom, output):
    from experiments.cylinder.after_the_fact.render_fullfield import snapshot_window

    output.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        folder = fits / case
        request = json.loads((folder / "request.json").read_text())
        original = json.loads((folder / "summary.json").read_text())
        long_check = json.loads((fits.parent / "long_check" / f"{case}.json").read_text())
        if request["window"] != [16., 1., 3.] or request["samples"] != 250:
            raise ValueError(f"Unexpected reference design: {case}")
        data, targets = {}, []
        with (np.load(folder / "posterior.npz") as posterior,
              np.load(folder / "operator_prior.npz") as prior,
              np.load(folder / "plotdata.npz") as plots):
            basis = posterior["basis_entries"]
            np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
            data["reference_operator"] = posterior["O_mean_rao_blackwell"]
            for k, re in enumerate(REYNOLDS):
                t, truth = plots[f"t_{k}"], plots[f"truth_{k}"]
                if k < 2:
                    data[f"obs_{k}"] = posterior[f"snaps_comp_{k}"]
                    data[f"t_obs_{k}"] = posterior[f"t_samp_{k}"]
                    data[f"gp_states_{k}"] = prior["anchor_states"][k]
                    data[f"gp_rates_{k}"] = prior["anchor_rates"][k]
                path = fom / f"medium_re{re}.npz"
                with np.load(path) as full:
                    indices = np.searchsorted(full["t"], 16 + t - 1e-8)
                    np.testing.assert_allclose(full["t"][indices], 16 + t, atol=1e-8)
                np.testing.assert_array_equal(np.diff(indices), 1)
                field = snapshot_window(path, int(indices[0]), int(indices[-1]) + 1).astype(float).T
                np.testing.assert_allclose(basis.T @ field, truth, rtol=1e-8, atol=1e-8)
                score = t > (1 if k < 2 else 0)
                residual = float(np.sum((field[:, score] - basis @ truth[:, score]) ** 2))
                scored = field[:, score]
                denominator = float(np.sum((scored - scored.mean(axis=1, keepdims=True)) ** 2))
                reference = plots[f"rao_blackwell_{k}"]
                error = np.sqrt((np.sum((reference[:, score] - truth[:, score]) ** 2) + residual) / denominator)
                old = original["targets"][k]
                np.testing.assert_allclose(error, old["means"]["rao_blackwell"]["error"], rtol=1e-7)
                np.testing.assert_allclose(np.sqrt(residual / denominator), old["floor"], rtol=1e-7)
                data.update({f"{name}_{k}": value for name, value in (
                    ("t", t), ("truth", truth), ("reference", reference),
                    ("q0", reference[:, 0]), ("cap", 2 * np.max(np.abs(posterior[f"snaps_comp_{k}"]))),
                    ("residual", residual), ("denominator", denominator))})
                targets.append(dict(re=re, held_out=k == 2, reference_error=float(error),
                                    floor=float(np.sqrt(residual / denominator)),
                                    reference_long=long_check["targets"][k]["mean"]))
                del field, scored
        digest = hashlib.sha256()
        for key in sorted(data):
            if key.startswith(("obs_", "t_obs_", "gp_states_", "gp_rates_")):
                digest.update(key.encode())
                digest.update(np.asarray(data[key]).tobytes())
        digest.update(basis.tobytes())
        np.savez_compressed(output / f"{case}.npz", **data)
        write_json(output / f"{case}.json", dict(
            case=case, modes=basis.shape[1], seed=request["config"]["seed"],
            training_basis_hash=digest.hexdigest(), targets=targets,
            source_files={name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                          for name in ("request.json", "operator_prior.npz", "plotdata.npz", "summary.json")}))
        print(f"prepared {case}: reproduced all reference errors and floors", flush=True)


def regression_data(training, route):
    F, Y = [], []
    for k, tr in enumerate(training):
        if route == "gp":
            states, rates = tr["gp_states"], tr["gp_rates"]
        elif route == "fd":
            states = tr["obs"]
            rates = np.gradient(states, tr["t_obs"], axis=1, edge_order=2)
        else:
            raise ValueError(f"Unknown derivative route {route}")
        F.append(features(states, 200 / REYNOLDS[k]))
        Y.append(rates.T)
    F, Y = np.vstack(F), np.vstack(Y)
    fscale, yscale = np.linalg.norm(F, axis=0), np.linalg.norm(Y, axis=0)
    if (not np.isfinite(F).all() or not np.isfinite(Y).all()
            or np.any(fscale <= 0) or np.any(yscale <= 0)):
        raise ValueError("Nonfinite data or zero feature/derivative normalization")
    return F / fscale, Y / yscale, fscale, yscale


def ridge_coefficients(design, response, alpha):
    U, s, Vh = svd(design, full_matrices=False)
    return Vh.T @ ((s / (s * s + alpha * alpha))[:, None] * (U.T @ response))


def sparse_coefficients(design, response, alpha, threshold, unbias=True):
    import pysindy as ps

    optimizer = ps.STLSQ(threshold=threshold, alpha=alpha * alpha,
                        max_iter=30, normalize_columns=False, unbias=unbias)
    optimizer.fit(design, response)
    return optimizer.coef_.T


def physical_operator(coefficients, fscale, yscale):
    return (coefficients * yscale[None] / fscale[:, None]).T


def training_score(operator, training):
    errors, statuses = [], []
    for k, tr in enumerate(training):
        observations = tr["obs"]
        status, prediction = integrate(operator, observations[:, 0], 200 / REYNOLDS[k],
                                       tr["t_obs"], 2 * np.max(np.abs(observations)))
        statuses.append(status)
        if status["bounded"]:
            scale = np.linalg.norm(observations - observations.mean(axis=1, keepdims=True))
            if scale <= 0:
                raise ValueError("Training fluctuation normalization must be positive")
            errors.append(float(np.linalg.norm(prediction - observations) / scale))
    return (float(np.mean(errors)) if len(errors) == len(training) else None), statuses


def fit_method(training, method):
    family, route, *variant = method.split("_")
    if method not in METHODS + REGULARIZED_METHODS:
        raise ValueError(f"Unknown method {method}")
    design, response, fscale, yscale = regression_data(training, route)
    parameters = ([dict(alpha=float(a), threshold=None) for a in RIDGES] if family == "opinf" else
                  [dict(alpha=a, threshold=t) for a in SPARSE_RIDGES for t in THRESHOLDS])
    candidates, best, best_operator = [], None, None
    for parameter in parameters:
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                coefficients = (ridge_coefficients(design, response, parameter["alpha"]) if family == "opinf"
                                else sparse_coefficients(design, response, **parameter, unbias=not variant))
            except np.linalg.LinAlgError as error:
                candidates.append(dict(**parameter, training_error=None, numerical_failure=str(error)))
                print(f"{method} {parameter}: numerical failure {error}", flush=True)
                continue
        if not np.isfinite(coefficients).all():
            raise FloatingPointError(f"Nonfinite coefficients: {method} {parameter}")
        operator = physical_operator(coefficients, fscale, yscale)
        error, statuses = training_score(operator, training)
        row = dict(**parameter, training_error=error, training_status=statuses,
                   nonzero=int(np.count_nonzero(operator)), total_coefficients=int(operator.size),
                   warnings=sorted(set(f"{w.category.__name__}: {w.message}" for w in caught)),
                   seconds=time.monotonic() - started)
        candidates.append(row)
        print(json.dumps(dict(method=method, **row)), flush=True)
        if error is not None and (best is None or error < best["training_error"]):
            best, best_operator = row, operator
    return best_operator, dict(method=method, selected=best, candidates=candidates)


def evaluate(operator, data, name, plots):
    targets = []
    for k, re in enumerate(REYNOLDS):
        t, truth = data[f"t_{k}"], data[f"truth_{k}"]
        short, prediction = integrate(operator, data[f"q0_{k}"], 200 / re, t, float(data[f"cap_{k}"]))
        error = None
        if short["bounded"]:
            if not np.isfinite(prediction).all():
                raise FloatingPointError("A completed prediction is nonfinite")
            score = t > (1 if k < 2 else 0)
            error = float(np.sqrt((np.sum((prediction[:, score] - truth[:, score]) ** 2)
                                   + data[f"residual_{k}"]) / data[f"denominator_{k}"]))
            plots[f"{name}_{k}"] = prediction
        long, long_prediction = integrate(operator, data[f"q0_{k}"], 200 / re, np.linspace(0, 10, 1001),
                                         float(data[f"cap_{k}"]), max_rhs=80000)
        if long["bounded"]:
            plots[f"{name}_long_{k}"] = long_prediction
        targets.append(dict(re=re, held_out=k == 2, short=short, error=error, long=long))
    return targets


def fit_case(inputs, case, output, source_commit, regularized_only=False):
    import importlib.metadata

    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError("Refusing to overwrite an existing completed comparator case")
    with np.load(inputs / f"{case}.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads((inputs / f"{case}.json").read_text())
    training = [{key: data[f"{key}_{k}"] for key in ("obs", "t_obs", "gp_states", "gp_rates")} for k in (0, 1)]
    plots = {key: value for key, value in data.items() if key.startswith(("truth_", "t_", "obs_", "reference_"))}
    models, operators = [], {}
    for method in (REGULARIZED_METHODS if regularized_only else METHODS):
        operator, report = fit_method(training, method)
        report["targets"] = None if operator is None else evaluate(operator, data, method, plots)
        if operator is not None:
            operators[method] = operator
        models.append(report)
        write_json(output / "progress.json", dict(**metadata, source_commit=source_commit, models=models))
    for k, re in enumerate(REYNOLDS):
        status, prediction = integrate(data["reference_operator"], data[f"q0_{k}"], 200 / re,
                                       data[f"t_{k}"], float(data[f"cap_{k}"]))
        if not status["bounded"]:
            raise ValueError(f"Saved reference no longer completes: {case}, Re{re}, {status}")
        np.testing.assert_allclose(prediction, data[f"reference_{k}"], rtol=1e-7, atol=1e-8)
    np.savez_compressed(output / "operators.npz", **operators)
    np.savez_compressed(output / "plotdata.npz", **plots)
    write_json(output / "summary.json", dict(
        **metadata, source_commit=source_commit, models=models, reproduced_reference_trajectories=True,
        versions={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "scikit-learn", "pysindy")}))


def report(results, output, regularized=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((results / case / "summary.json").read_text()) for case in CASES]
    if len({s["source_commit"] for s in summaries}) != 1:
        raise ValueError("Mixed comparator source revisions")
    if regularized is not None:
        followups = [json.loads((regularized / case / "summary.json").read_text()) for case in CASES]
        if len({s["source_commit"] for s in followups}) != 1:
            raise ValueError("Mixed follow-up source revisions")
        for summary, extra in zip(summaries, followups):
            if summary["training_basis_hash"] != extra["training_basis_hash"] or summary["case"] != extra["case"]:
                raise ValueError("Unmatched follow-up training data or basis")
            summary["regularized_source_commit"] = extra["source_commit"]
            summary["models"].extend(extra["models"])
    labels = {"reference": "Bayesian contraction", "opinf_fd": "OpInf / FD", "sindy_fd": "SINDy / FD",
              "opinf_gp": "OpInf / GP", "sindy_gp": "SINDy / GP",
              "sindy_fd_regularized": "SINDy / FD (ridge retained)",
              "sindy_gp_regularized": "SINDy / GP (ridge retained)"}
    colors = {"reference": "black", "opinf_fd": "#8c8c8c", "sindy_fd": "#c44e52",
              "opinf_gp": "#2878b5", "sindy_gp": "#e39c24",
              "sindy_fd_regularized": "#b86b93", "sindy_gp_regularized": "#7658a0"}
    methods = ("reference",) + METHODS + (() if regularized is None else REGULARIZED_METHODS)
    width = .75 / len(methods)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, summary in zip(axes, summaries):
        by_method = {m["method"]: m for m in summary["models"]}
        for j, method in enumerate(methods):
            for k, re in enumerate(REYNOLDS):
                reference = summary["targets"][k]
                if method == "reference":
                    error, short, long = reference["reference_error"], "complete", reference["reference_long"]
                    selection = None
                else:
                    model = by_method[method]
                    selection = model["selected"]
                    target = None if model["targets"] is None else model["targets"][k]
                    error = None if target is None else target["error"]
                    short = "selection failed" if target is None else target["short"]["reason"]
                    long = dict(bounded=False, reason="selection failed") if target is None else target["long"]
                rows.append(dict(case=summary["case"], modes=summary["modes"], seed=summary["seed"],
                                 method=method, re=re, held_out=k == 2, error=error,
                                 short_reason=short, long_bounded=long["bounded"], long_reason=long["reason"],
                                 alpha=None if selection is None else selection["alpha"],
                                 threshold=None if selection is None else selection["threshold"],
                                 nonzero=None if selection is None else selection["nonzero"]))
                x = k + (j-(len(methods)-1)/2)*width
                if error is not None:
                    ax.bar(x, 100*error, width=.95*width, color=colors[method],
                           label=labels[method] if k == 0 else None)
                else:
                    ax.text(x, 1.01, "FAIL", transform=ax.get_xaxis_transform(), rotation=90,
                            color=colors[method], fontsize=7, ha="center", va="bottom")
            if not any(r["error"] is not None for r in rows if r["case"] == summary["case"] and r["method"] == method):
                ax.bar([], [], color=colors[method], label=labels[method])
        for k, target in enumerate(summary["targets"]):
            ax.hlines(100*target["floor"], k-.4, k+.4, color="#16843d", linestyle="--", linewidth=1)
        ax.set_xticks(range(3), ["Re150", "Re300", "Re200\nheld out"])
        ax.set_title(f"r={summary['modes']}, seed {summary['seed']}")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Full-field fluctuation-relative forecast error (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Matched cylinder baselines; dashed green = POD floor; FAIL is not zero error")
    fig.tight_layout(rect=(0, .16, 1, .87))
    fig.savefig(output / "01_forecast_errors.png", dpi=160)
    plt.close(fig)

    for number, summary in enumerate(summaries, 2):
        with np.load(results / summary["case"] / "plotdata.npz") as archive:
            data = {key: archive[key] for key in archive.files}
        if regularized is not None:
            with np.load(regularized / summary["case"] / "plotdata.npz") as archive:
                for key in archive.files:
                    if key in data:
                        np.testing.assert_array_equal(data[key], archive[key])
                    else:
                        data[key] = archive[key]
        fig, axes = plt.subplots(3, 3, figsize=(13, 9))
        for k, re in enumerate(REYNOLDS):
            t = data[f"t_{k}"]
            for mode, ax in enumerate(axes[k]):
                ax.plot(t+16, data[f"truth_{k}"][mode], color="#16843d", lw=1.5, label="truth")
                for method in methods:
                    key = f"{method}_{k}"
                    if key in data:
                        ax.plot(t+16, data[key][mode], color=colors[method], lw=1,
                                linestyle="--" if method == "reference" else "-", label=labels[method])
                if k < 2:
                    ax.plot(data[f"t_obs_{k}"][::6]+16, data[f"obs_{k}"][mode, ::6], "k*", ms=3,
                            label="observations (subset)")
                    ax.axvline(17, color="gray", linestyle=":", lw=1)
                else:
                    ax.plot([16], data[f"reference_{k}"][mode, :1], "k*", ms=4)
                ax.set(xlabel="time", ylabel=f"Re{re}, POD {mode+1}")
            failed = [labels[m["method"]] + ": " + ("selection failed" if m["targets"] is None
                      else m["targets"][k]["short"]["reason"]) for m in summary["models"]
                      if m["targets"] is None or not m["targets"][k]["short"]["bounded"]]
            if failed:
                axes[k, 0].text(.02, .02, "\n".join(failed), transform=axes[k, 0].transAxes, fontsize=7)
        fig.suptitle(f"r={summary['modes']}, seed {summary['seed']}; incomplete trajectories are omitted and labelled")
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, .09, 1, .95))
        fig.savefig(output / f"{number:02d}_{summary['case']}_trajectories.png", dpi=160)
        plt.close(fig)
    write_json(output / "summary.json", dict(cases=summaries, rows=rows))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["Full-field fluctuation errors (%); selection uses training observations only.",
             "Deterministic boundedness is not posterior survival or uncertainty calibration."]
    for summary in summaries:
        lines.append(f"\n{summary['case']} (r={summary['modes']}, seed={summary['seed']}): Re150 / Re300 / Re200")
        for method in methods:
            items = [r for r in rows if r["case"] == summary["case"] and r["method"] == method]
            errors = ["FAIL: "+r["short_reason"] if r["error"] is None else f"{100*r['error']:.3f}" for r in items]
            lines.append(f"{labels[method]:22s} {' / '.join(errors)}; long bounded: "
                         + "/".join("yes" if r["long_bounded"] else "no" for r in items))
    text = "\n".join(lines) + "\n"
    (output / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--fits", type=Path, required=True)
    prep.add_argument("--fom", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--inputs", type=Path, required=True)
    fit.add_argument("--case", choices=CASES, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--source-commit", required=True)
    fit.add_argument("--regularized-only", action="store_true")
    show = commands.add_parser("report")
    show.add_argument("--results", type=Path, required=True)
    show.add_argument("--output", type=Path, required=True)
    show.add_argument("--regularized", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.fits, args.fom, args.output)
    elif args.command == "fit":
        fit_case(args.inputs, args.case, args.output, args.source_commit, args.regularized_only)
    else:
        report(args.results, args.output, args.regularized)
