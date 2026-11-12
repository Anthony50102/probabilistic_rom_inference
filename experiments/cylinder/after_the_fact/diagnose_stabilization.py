"""After-the-fact GP/optimization diagnostics, never inputs to fitted priors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import jax
import jax.numpy as jnp
from scipy.linalg import null_space

from core.weakform_opinf import WeakFormConfig, gp
from stabilization import feature_jacobian, features


def diagnose(folder):
    summary = json.loads((folder / "summary.json").read_text())
    priors = json.loads((folder / "priors.json").read_text())
    cfg = WeakFormConfig(**summary["config"])
    with np.load(folder / "posterior.npz") as archive:
        d = {key: archive[key] for key in archive.files}
    draws, conditional = d["O_samples"], d["O_conditional_means"]
    mean = d["O_mean_rao_blackwell"]
    losses = d["losses"]
    fit_gp = priors["closure"]["data_only_gp"]
    result = dict(name=folder.name, steps=len(losses),
                  loss_500_block_means=[float(b.mean()) for b in np.array_split(losses, len(losses) // 500)],
                  loss_last_1000_sd=float(losses[-1000:].std()),
                  sample_mean_relative_difference=float(np.linalg.norm(draws.mean(0) - mean) / np.linalg.norm(mean)),
                  RB_MC_SE_relative=float(np.linalg.norm(conditional.std(0, ddof=1)) /
                                          (np.sqrt(len(draws)) * np.linalg.norm(mean))),
                  sample_MC_SE_relative=float(np.linalg.norm(draws.std(0, ddof=1)) /
                                              (np.sqrt(len(draws)) * np.linalg.norm(mean))),
                  conditional_variance_fraction_estimate=float(
                      np.sum((draws - conditional) ** 2) / np.sum((draws - mean) ** 2)),
                  targets=[])
    if "op_log_tau_samples" in d:
        result["block_multiplier_quantiles_5_50_95"] = np.percentile(
            np.exp(d["op_log_tau_samples"]), (5, 50, 95), axis=0).tolist()
    amp = np.sqrt(np.mean([np.var(d[f"snaps_comp_{k}"], axis=1) for k in (0, 1)], axis=0))
    for k, u in enumerate((4 / 3, 2 / 3)):
        tr = dict(t_sampled=d[f"t_samp_{k}"], snapshots_comp=d[f"snaps_comp_{k}"])
        grid = np.linspace(tr["t_sampled"][0], tr["t_sampled"][-1], cfg.num_eval_points)
        theta = tuple(np.median(d[key], axis=0)[k] for key in
                      ("gp_lengthscale_samples", "gp_variance_samples", "gp_noise_samples"))
        batch = jax.jit(gp.trajectory_gp_conditional(tr, cfg)(grid)[1])
        states, rates, *_ = map(np.asarray, batch(*map(jnp.asarray, theta), jnp.asarray(tr["snapshots_comp"])))
        map_theta = tuple(jnp.asarray(fit_gp[key][k]) for key in ("ell", "sig2", "nu"))
        map_states = np.asarray(batch(*map_theta, jnp.asarray(tr["snapshots_comp"]))[0])
        truth = np.array([np.interp(grid, d[f"t_{k}"], q) for q in d[f"true_comp_{k}"]])
        truth_rates = np.gradient(truth, grid, axis=1)
        fluct_norm = np.linalg.norm(truth - truth.mean(1, keepdims=True))
        normal_mean, normal_draw = [], []
        for j in np.linspace(2, len(grid) - 3, 12, dtype=int):
            tangent = rates[:, j] / amp
            basis = null_space(tangent[None])
            J = feature_jacobian(states[:, j], u)
            for operators, values in ((mean[None], normal_mean), (draws[:16], normal_draw)):
                for O in operators:
                    N = basis.T @ ((O @ J) * amp[None] / amp[:, None]) @ basis
                    values.append(np.linalg.eigvalsh((N + N.T) * .5)[-1])
        result["targets"].append(dict(
            re=(150, 300)[k],
            posterior_median_hypers=[x.tolist() for x in theta],
            data_GP_MAP_hypers=[np.asarray(x).tolist() for x in map_theta],
            gp_state_error=float(np.linalg.norm(states - truth) / fluct_norm),
            data_gp_state_error=float(np.linalg.norm(map_states - truth) / fluct_norm),
            gp_rate_error=float(np.linalg.norm(rates - truth_rates) / np.linalg.norm(truth_rates)),
            O_rate_residual_on_GP=float(np.linalg.norm(features(states, u) @ mean.T - rates.T) / np.linalg.norm(rates)),
            O_rate_residual_on_truth=float(np.linalg.norm(features(truth, u) @ mean.T - truth_rates.T) /
                                           np.linalg.norm(truth_rates)),
            normal_numerical_abscissa_mean_quantiles=np.percentile(normal_mean, (5, 50, 95)).tolist(),
            normal_numerical_abscissa_draw_quantiles=np.percentile(normal_draw, (5, 50, 95)).tolist()))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fits", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in sorted(args.fits.iterdir()):
        destination = args.output / f"{folder.name}.json"
        if folder.is_dir() and (folder / "summary.json").exists() and not destination.exists():
            if len(json.loads((folder / "summary.json").read_text())["targets"]) != 3:
                continue
            result = diagnose(folder)
            destination.write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
