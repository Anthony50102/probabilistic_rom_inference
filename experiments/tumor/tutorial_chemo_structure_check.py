"""Clean-data check of the chemotherapy ROM form on the tutorial-schedule task.

Fits the reported model form (c + A q + N alpha q, the `cAN` Galerkin structure) and the form with
the quadratic logistic block (`cAHN`) by plain least squares to the NOISE-FREE projected truth of
one acquisition, in integral form, then forecasts every dose arm from day 5. No noise, no Gaussian
processes, no Bayesian inference: it separates what the model form can represent from what the
training window determines. See experiments/TUMOR_TUTORIAL_SCHEDULE_CHEMO.md.

Usage (after 04_unified_benchmark.py has prepared the acquisition):
    python tutorial_chemo_structure_check.py
    python tutorial_chemo_structure_check.py --seed-directory \
        results/benchmarks/alternative-regimen-chemo/segmented_observed_none_r4/seed48
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

DEFAULT = Path(__file__).resolve().parent / "results/benchmarks/alternative-regimen-chemo/segmented_observed_none_r4/seed45"
ARMS = ("strength1p00", "strength0p50", "strength1p50", "strength2p00")
SUBSTEPS = 20


def features(q, alpha, operators, upper):
    columns = []
    if "c" in operators:
        columns.append(np.ones((1, q.shape[1])))
    if "A" in operators:
        columns.append(q)
    if "H" in operators:
        columns.append((q[:, None, :] * q[None, :, :])[upper[0], upper[1], :])
    if "N" in operators:
        columns.append(alpha[None, :] * q)
    return np.vstack(columns)


def integral_rows(reference, times, operators, start, end, upper):
    """Rows q(t_k+1) - q(t_k) = integral of the features, on a spline of the clean coefficients."""
    q = reference["true_comp"]
    inside = (times >= start - 1e-9) & (times <= end + 1e-9)
    t, qq = times[inside], q[:, inside]
    fine = np.linspace(t[0], t[-1], SUBSTEPS * (len(t) - 1) + 1)
    F = features(CubicSpline(t, qq, axis=1)(fine), np.interp(fine, reference["input_times"], reference["input_values"]),
                 operators, upper)
    rows = [np.trapezoid(F[:, k * SUBSTEPS:(k + 1) * SUBSTEPS + 1], fine[k * SUBSTEPS:(k + 1) * SUBSTEPS + 1], axis=1)
            for k in range(len(t) - 1)]
    return np.stack(rows), (qq[:, 1:] - qq[:, :-1]).T


def fit(references, times, operators, windows, upper):
    blocks = [integral_rows(references[arm], times, operators, start, end, upper) for arm, start, end in windows]
    X, Y = np.vstack([b[0] for b in blocks]), np.vstack([b[1] for b in blocks])
    scale = np.linalg.norm(X, axis=0)
    scale[scale == 0] = 1
    return (np.linalg.lstsq(X / scale, Y, rcond=None)[0] / scale[:, None]).T


def forecast(O, reference, times, operators, upper, start=5.):
    q = reference["true_comp"]
    first = int(np.argmin(abs(times - start)))

    def rhs(t, y):
        alpha = np.array([np.interp(t, reference["input_times"], reference["input_values"])])
        return O @ features(y[:, None], alpha, operators, upper)[:, 0]

    def blow_up(t, y):
        return 1e3 * np.abs(q).max() - np.abs(y).max()

    blow_up.terminal = True
    # LSODA: the training-window cAHN fit is stiff and blows up in finite time; an explicit solver crawls.
    solution = solve_ivp(rhs, (times[first], times[-1]), q[:, first], method="LSODA", t_eval=times[first:],
                         rtol=1e-8, atol=1e-8, max_step=.25, events=blow_up)
    if not solution.success or solution.y.shape[1] != len(times) - first:
        return None
    out = np.full_like(q, np.nan)
    out[:, first:] = solution.y
    return out


def error(prediction, reference, times, start, end):
    q, window = reference["true_comp"], (times >= start) & (times <= end)
    return 100 * np.linalg.norm(prediction[:, window] - q[:, window]) / np.linalg.norm(q[:, window])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-directory", type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    references = {arm: np.load(args.seed_directory / "data" / f"reference_{arm}.npz") for arm in ARMS}
    times = references[ARMS[0]]["t_pred"]
    upper = np.triu_indices(references[ARMS[0]]["true_comp"].shape[0])
    designs = {"training window, days 5-57": [(ARMS[0], 5., 57.)],
               "hindsight, all arms, days 5-119": [(arm, 5., 119.) for arm in ARMS]}
    print(f"Reduced relative error (%) of clean least-squares ROMs, forecast window days 57-119 ({args.seed_directory})")
    print(f"{'fitted on':<33}{'form':<6}" + "".join(f"{arm:>14}" for arm in ARMS))
    for label, windows in designs.items():
        for operators in ("cAN", "cAHN"):
            O = fit(references, times, operators, windows, upper)
            cells = []
            for arm in ARMS:
                prediction = forecast(O, references[arm], times, operators, upper)
                cells.append("unstable" if prediction is None else f"{error(prediction, references[arm], times, 57, 119):.1f}")
            print(f"{label:<33}{operators:<6}" + "".join(f"{cell:>14}" for cell in cells))


if __name__ == "__main__":
    main()
