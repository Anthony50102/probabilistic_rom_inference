"""Split the tutorial-schedule chemo forecasts into regrowth and drug kill.

Why does production forecast the 2x arm best? In the truth (TumorTwin: du/dt = d lap(u) + k u (1 - u) - alpha u,
diffusion conserving mass) the burden B obeys d ln B/dt = g(t) - alpha(t): the drug removes exactly alpha(t) per
cell per day, and the per-cell regrowth g(t) rises as the drug thins the tumour. Over a window, then,
    mean g = [ln B(end) - ln B(start)] / length + mean alpha.
For a `cAN` ROM (dq/dt = c + A q + alpha N q, burden w.q) the same split is
    regrowth  g_hat = w.(c + A q) / w.q,    kill per unit alpha  kappa_hat = -w.(N q) / w.q,
so its burden-rate error is a regrowth error (g_hat - g) plus a drug-strength error (1 - kappa_hat) alpha, which
scales with the dose. The ROM rates are evaluated on the true projected states, so the differences are operator
differences, not differences in where the forecasts have drifted.

Reports production's point operators and the clean least-squares `cAN` fits of tutorial_chemo_structure_check.py
(training window and hindsight). See experiments/TUMOR_TUTORIAL_SCHEDULE_CHEMO.md.

Usage (after 04_unified_benchmark.py has fitted the seeds; a few seconds):
    python tutorial_chemo_rate_decomposition.py
    python tutorial_chemo_rate_decomposition.py --seed-directory \
        results/benchmarks/alternative-regimen-chemo/segmented_observed_none_r4/seed45
"""
import argparse
from pathlib import Path

import numpy as np

import tutorial_chemo_structure_check as check

CASE = Path(__file__).resolve().parent / "results/benchmarks/alternative-regimen-chemo/segmented_observed_none_r4"
DEFAULT = [CASE / f"seed{seed}" for seed in (48, 49, 50)]
ARMS = ("strength0p50", "strength1p00", "strength1p50", "strength2p00")
WINDOWS = {"untreated 5-35": (5., 35.), "treated 35-57": (35., 57.), "forecast 57-119": (57., 119.)}


def window_mean(reference, values, start, end, weight_by_alpha=False):
    """Mean over [start, end] of values on t_pred, optionally times alpha, on a grid that keeps alpha's knots."""
    times, alpha_t, alpha_v = reference["t_pred"], reference["input_times"], reference["input_values"]
    inside = lambda t: t[(t > start) & (t < end)]  # noqa: E731
    t = np.unique(np.r_[start, inside(times), inside(alpha_t), end])
    f = np.interp(t, times, values)
    if weight_by_alpha:
        f = f * np.interp(t, alpha_t, alpha_v)
    return np.trapezoid(f, t) / (end - start)


def split_operators(O, r, layout):
    if layout == "production":  # c | A | B | N, with B zero for the cAN form
        assert not np.any(O[:, 1 + r]), "production fit is not the cAN form"
        return O[:, 0], O[:, 1:1 + r], O[:, 2 + r:]
    return O[:, 0], O[:, 1:1 + r], O[:, 1 + r:]  # least-squares cAN: c | A | N


def rates(operators, weights, offset, reference, start, end):
    """Window means (per day) of the true and ROM regrowth, ROM kill per unit alpha, alpha, and the two errors."""
    c, A, N = operators
    q = reference["true_comp"]
    burden = weights @ q + offset
    g_hat = weights @ (c[:, None] + A @ q) / burden
    kappa_hat = -(weights @ (N @ q)) / burden
    ones = np.ones_like(g_hat)
    alpha = window_mean(reference, ones, start, end, weight_by_alpha=True)
    ln_b = np.interp([start, end], reference["t_pred"], np.log(reference["truth_burden"]))
    g = (ln_b[1] - ln_b[0]) / (end - start) + alpha
    regrowth_error = window_mean(reference, g_hat, start, end) - g
    drug_error = window_mean(reference, ones - kappa_hat, start, end, weight_by_alpha=True)
    return g, g + regrowth_error, window_mean(reference, kappa_hat, start, end), alpha, regrowth_error, drug_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-directory", type=Path, nargs="+", default=DEFAULT)
    args = parser.parse_args(argv)
    for directory in args.seed_directory:
        references = {arm: np.load(directory / "data" / f"reference_{arm}.npz") for arm in ARMS}
        observations = np.load(directory / "data" / "observations.npz")
        weights, offset = observations["burden_weights"], float(observations["burden_offset"])
        times = references[ARMS[0]]["t_pred"]
        r = references[ARMS[0]]["true_comp"].shape[0]
        upper = np.triu_indices(r)
        fits = {"production": split_operators(np.load(directory / "production" / "operators.npz")["O_point"], r,
                                              "production"),
                "clean LS, training": split_operators(check.fit(references, times, "cAN", [("strength1p00", 5., 57.)],
                                                                upper), r, "ls"),
                "clean LS, hindsight": split_operators(check.fit(references, times, "cAN",
                                                                 [(arm, 5., 119.) for arm in ARMS], upper), r, "ls")}
        print(f"\nPer-cell burden rates, %/day, window means ({directory})")
        print("Truth: regrowth = d ln B/dt + alpha, kill 1 per unit alpha. ROM: on the true projected states.")
        print("regrowth err = ROM - truth; drug err = (1 - ROM kill per unit alpha) x alpha; net = their sum")
        print(f"{'fit':<21}{'dose':<14}{'window':<17}{'true regrowth':>14}{'ROM regrowth':>13}{'kill/alpha':>11}"
              f"{'alpha':>7}{'regrowth err':>13}{'drug err':>9}{'net':>7}")
        for label, operators in fits.items():
            for arm in ARMS:
                for window, (start, end) in WINDOWS.items():
                    g, g_hat, kappa, alpha, regrowth_error, drug_error = rates(operators, weights, offset,
                                                                               references[arm], start, end)
                    print(f"{label:<21}{arm:<14}{window:<17}{100 * g:14.2f}{100 * g_hat:13.2f}{kappa:11.2f}"
                          f"{100 * alpha:7.2f}{100 * regrowth_error:13.2f}{100 * drug_error:9.2f}"
                          f"{100 * (regrowth_error + drug_error):7.2f}")


if __name__ == "__main__":
    main()
