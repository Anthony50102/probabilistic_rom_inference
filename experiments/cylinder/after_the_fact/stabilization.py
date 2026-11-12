"""Declared exploratory fixed-GP screens; not a replacement for joint Bayesian inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import qr, solve_triangular

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def features(q, u):
    a, b = np.tril_indices(len(q))
    return np.concatenate([np.ones((1, q.shape[1])), q, q[a] * q[b],
                           np.full((1, q.shape[1]), u), u * q]).T


def feature_jacobian(q, u):
    r = len(q)
    a, b = np.tril_indices(r)
    H = np.zeros((len(a), r))
    np.add.at(H, (np.arange(len(a)), a), q[b])
    np.add.at(H, (np.arange(len(a)), b), q[a])
    return np.vstack([np.zeros((1, r)), np.eye(r), H, np.zeros((1, r)), u * np.eye(r)])


def gaussian_fit(A, y, scale, prior_A=None, prior_y=None):
    """Whitened likelihood with a diagonal Gaussian, optionally a Gaussian Jacobian prior.

    Compute in prior-standardised coordinates, using QR rather than normal equations.
    The returned root satisfies covariance = root @ root.T.
    """
    m = A.shape[1]
    blocks, targets = [A * scale, np.eye(m)], [y, np.zeros(m)]
    if prior_A is not None:
        blocks.append(prior_A * scale)
        targets.append(prior_y)
    Z, rhs = np.vstack(blocks), np.concatenate(targets)
    Q, R = qr(Z, mode="economic")
    mu = scale * solve_triangular(R, Q.T @ rhs)
    root = scale[:, None] * solve_triangular(R, np.eye(m))
    return mu, root


def integrate(O, q0, u, t, cap, method="RK45", tol=1e-7):
    r = len(q0)
    a, b = np.tril_indices(r)
    H = O[:, 1 + r:1 + r + len(a)]
    L = O[:, 1:1 + r] + u * O[:, 2 + r + len(a):]
    c = O[:, 0] + u * O[:, 1 + r + len(a)]
    calls = 0

    def rhs(tt, q):
        nonlocal calls
        calls += 1
        if calls > 20000:
            raise RuntimeError("RHS evaluation budget exhausted")
        return c + L @ q + H @ (q[a] * q[b])

    def event(tt, q):
        return cap - np.max(np.abs(q))

    event.terminal = True
    event.direction = -1
    kwargs = {"jac": lambda tt, q: O @ feature_jacobian(q, u)} if method in ("Radau", "BDF") else {}
    try:
        sol = solve_ivp(rhs, (t[0], t[-1]), q0, t_eval=t, rtol=tol,
                        atol=tol * .01, events=event, method=method, **kwargs)
    except RuntimeError as err:
        if str(err) != "RHS evaluation budget exhausted":
            raise
        return dict(bounded=False, reason=str(err), calls=calls), None
    bounded = sol.success and len(sol.t) == len(t)
    stop = float(sol.t_events[0][0]) if len(sol.t_events[0]) else float(sol.t[-1])
    return dict(bounded=bool(bounded), stop=stop, calls=calls,
                reason="complete" if bounded else ("cap" if len(sol.t_events[0]) else sol.message)), sol.y


def prepare(saved, fom_dir, output):
    import jax
    import jax.numpy as jnp
    from core.weakform_opinf import gp, weakform
    from core.weakform_opinf.config import WeakFormConfig

    jax.config.update("jax_enable_x64", True)
    d = np.load(saved)
    r = d["O_samples"].shape[1]
    cfg = WeakFormConfig(num_modes=r, operators="cAHBN", precision="float64")
    grid = np.linspace(0, 1, cfg.num_eval_points)
    tf = {k: np.asarray(v) for k, v in weakform.build_test_functions(grid, cfg).items()}
    Q = np.hstack([d[f"snaps_comp_{k}"] for k in (0, 1)])
    center = Q.mean(1)
    amp = np.sqrt(np.mean([np.var(d[f"snaps_comp_{k}"], axis=1) for k in (0, 1)], axis=0))
    if np.any(amp <= 0):
        raise ValueError("Positive within-trajectory fluctuations are needed")
    a, b = np.tril_indices(r)
    degree = np.r_[0, np.ones(r), np.full(len(a), 2), 0, np.ones(r)]
    input_degree = np.r_[np.zeros(1 + r + len(a)), np.ones(1 + r)]
    S, U = np.sqrt(np.mean(Q ** 2)), np.sqrt((16 / 9 + 4 / 9) / 2)
    block_size = np.r_[1, np.full(r, r), np.full(len(a), len(a)), 1, np.full(r, r)]
    cache = dict(r=r, center=center, amp=amp, global_scale=S ** (1 - degree) * U ** (-input_degree),
                 state_scale=S, block_size=block_size, grid=grid, original_draw_mean=d["O_samples"].mean(0))
    design, targets, covD, covW, states, derivs, alphas = [], [], [], [], [], [], []
    for k, u in enumerate((4 / 3, 2 / 3)):
        tr = dict(t_sampled=d[f"t_samp_{k}"], snapshots_comp=d[f"snaps_comp_{k}"])
        th = tuple(np.median(d[key], axis=0)[k] for key in
                   ("gp_lengthscale_samples", "gp_variance_samples", "gp_noise_samples"))
        batch = jax.jit(gp.trajectory_gp_conditional(tr, cfg)(grid)[1])
        X, dz, Kz, Kx, _ = map(np.asarray, batch(*map(jnp.asarray, th), jnp.asarray(tr["snapshots_comp"])))
        df = gp.effective_dof(tr, cfg, *th)
        alpha = np.minimum(1, df / (len(grid) + len(tf["wpsi"])))
        ell, sig2, _ = th
        eps = np.finfo(float).eps
        vD = np.maximum(np.diagonal(Kz, axis1=1, axis2=2), 0) + (len(tr["t_sampled"]) * eps * sig2 / ell ** 2)[:, None]
        vW = np.maximum(np.einsum("kj,ijl,kl->ik", tf["wpsi_dot"], Kx, tf["wpsi_dot"]), 0)
        vW += len(tr["t_sampled"]) * eps * sig2[:, None] * np.sum(np.abs(tf["wpsi_dot"]), axis=1)[None] ** 2
        design.append((features(X, u), features(X - center[:, None], u)))
        targets.append(np.hstack([dz, -X @ tf["wpsi_dot"].T]))
        covD.append(vD / alpha[:, None])
        covW.append(vW / alpha[:, None] + 1e-8)
        states.append(X)
        derivs.append(dz)
        alphas.append(alpha)
    std = np.sqrt(np.hstack([np.hstack([vd, vw]) for vd, vw in zip(covD, covW)]))
    cache["Y"] = np.hstack(targets) / std
    for shifted in (False, True):
        F = np.vstack([z[int(shifted)] for z in design])
        A = np.vstack([np.vstack([z[int(shifted)], tf["wpsi"] @ z[int(shifted)]]) for z in design])
        cache[f"A_{int(shifted)}"] = A[None] / std[:, :, None]
        cache[f"F_{int(shifted)}"] = F
    cache.update(gp_states=np.stack(states), gp_derivatives=np.stack(derivs), alpha=np.stack(alphas))
    V = d["basis_entries"]
    for k, re in enumerate((150, 300, 200)):
        t, truth = d[f"t_pred_{k}"], d[f"true_comp_{k}"]
        ff = np.load(fom_dir / f"medium_re{re}.npz")
        inds = np.searchsorted(ff["t"], 16 + t - 1e-9)
        np.testing.assert_allclose(ff["t"][inds], 16 + t, atol=1e-8)
        X = ff["snapshots"][inds].T.astype(float)
        np.testing.assert_allclose(V.T @ X, truth, rtol=1e-8, atol=1e-8)
        fc = t > (1 if k < 2 else 0)
        cache[f"denom_{k}"] = np.sum((X[:, fc] - X[:, fc].mean(1, keepdims=True)) ** 2)
        cache[f"residual_{k}"] = np.sum((X[:, fc] - V @ truth[:, fc]) ** 2)
        cache[f"truth_{k}"], cache[f"t_{k}"] = truth, t
        cache[f"q0_{k}"] = d[f"state0_samples_{k}"].mean(0)
        cache[f"cap_{k}"] = 2 * np.max(np.abs(d[f"snaps_comp_{k}"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **cache)
    print(json.dumps(dict(r=r, alpha_min=float(cache["alpha"].min()), alpha_max=float(cache["alpha"].max()),
                         state_scale=float(S), fluct_range=[float(amp.min()), float(amp.max())])))


def cases(jacobian_global=False):
    if jacobian_global:
        return [dict(id=f"jacglobal{i:02d}", shift=False, prior="global", multiplier=1., power=1.,
                     damping=damp, jac_sd=sd)
                for i, (damp, sd) in enumerate((d, s) for d in (0., 1., 5., 20.) for s in (.3, 1., 3., 10.))]
    rows = []
    for shift in (False, True):
        for prior in ("global", "row", "balanced"):
            for mult in (.03, .1, .3, 1., 3., 10., 30.):
                rows.append(dict(shift=shift, prior=prior, multiplier=mult, power=1.))
        for power in (4., 16.):
            rows.append(dict(shift=shift, prior="global", multiplier=1., power=power))
    for damp in (0., 1., 5., 20.):
        for sd in (.3, 1., 3., 10.):
            rows.append(dict(shift=True, prior="row", multiplier=1., power=1., damping=damp, jac_sd=sd))
    for i, row in enumerate(rows):
        row["id"] = f"case{i:03d}"
    return rows


def jacobian_prior(cache, shift, damping, sd):
    r = int(cache["r"])
    amp = cache["amp"]
    rows, target = [], []
    for k, u in enumerate((4 / 3, 2 / 3)):
        X, dz = cache["gp_states"][k], cache["gp_derivatives"][k]
        for j in np.linspace(2, X.shape[1] - 3, 12, dtype=int):
            v = dz[:, j] / amp
            v /= np.linalg.norm(v)
            P = np.eye(r) - np.outer(v, v)
            z = X[:, j] - shift
            rows.append((feature_jacobian(z, u) * amp[None]) @ P)
            target.append(-damping * P)
    norm = sd * np.sqrt(len(rows))
    return np.hstack(rows).T / norm, np.hstack(target) / norm


def fit_case(cache, case, draws=32):
    start = time.time()
    r = int(cache["r"])
    shift = cache["center"] if case["shift"] else np.zeros(r)
    F, A, Y = cache[f"F_{int(case['shift'])}"], cache[f"A_{int(case['shift'])}"], cache["Y"]
    amp, gscale = cache["amp"], cache["global_scale"]
    if case["prior"] == "global":
        scales = np.broadcast_to(gscale, (r, len(gscale))).copy()
    elif case["prior"] == "row":
        scales = (amp / cache["state_scale"])[:, None] * gscale[None]
    else:
        rms = np.sqrt(np.mean(F ** 2, axis=0))
        scales = amp[:, None] / (rms * np.sqrt(cache["block_size"]))[None]
    scales *= case["multiplier"]
    extra_A, extra_y = None, None
    if "damping" in case:
        extra_A, extra_y = jacobian_prior(cache, shift, case["damping"], case["jac_sd"])
    mean, roots = [], []
    for i in range(r):
        mu, root = gaussian_fit(A[i] * np.sqrt(case["power"]), Y[i] * np.sqrt(case["power"]), scales[i],
                               None if extra_A is None else extra_A / amp[i],
                               None if extra_y is None else extra_y[i])
        mean.append(mu)
        roots.append(root)
    mean, roots = np.array(mean), np.array(roots)
    rng = np.random.default_rng(42)
    samples = mean[None] + np.einsum("imj,sij->sim", roots, rng.normal(size=(draws,) + mean.shape))
    out = dict(case, conditional_only=True, targets=[], analytic_mean_norm=float(np.linalg.norm(mean)))
    plot = dict(mean=mean, roots=roots, shift=shift, scales=scales)
    for k, u in enumerate((4 / 3, 2 / 3, 1.)):
        t, truth = cache[f"t_{k}"], cache[f"truth_{k}"]
        q0 = cache[f"q0_{k}"] - shift
        cap = float(cache[f"cap_{k}"]) + np.max(np.abs(shift))
        fc = t > (1 if k < 2 else 0)
        denom, residual = cache[f"denom_{k}"], cache[f"residual_{k}"]

        def error(sol):
            return float(np.sqrt((np.sum((sol[:, fc] - truth[:, fc]) ** 2) + residual) / denom))

        # The shifted-coordinate cap is only an early numerical guard. Apply the original-state cap below.
        report, sol = integrate(mean, q0, u, t, cap)
        row = dict(re=(150, 300, 200)[k], pm=report, floor=float(np.sqrt(residual / denom)))
        if report["bounded"]:
            sol = sol + shift[:, None]
            row["pm"]["bounded"] = bool(np.max(np.abs(sol)) <= cache[f"cap_{k}"])
            if not row["pm"]["bounded"]:
                row["pm"]["reason"] = "original_state_cap"
            if row["pm"]["bounded"]:
                row["pm_error"] = error(sol)
                plot[f"pm_{k}"] = sol
        sols, failures = [], {}
        for sample in samples:
            status, pred = integrate(sample, q0, u, t, cap)
            if status["bounded"] and np.max(np.abs(pred + shift[:, None])) <= cache[f"cap_{k}"]:
                sols.append(pred + shift[:, None])
            else:
                reason = "original_state_cap" if status["bounded"] else status["reason"]
                failures[reason] = failures.get(reason, 0) + 1
        row.update(n_draws=draws, n_bounded=len(sols), survival=len(sols) / draws, failures=failures)
        if sols:
            sols = np.array(sols)
            med = np.median(sols, axis=0)
            lo, hi = np.percentile(sols, (5, 95), axis=0)
            row["median_error"] = error(med)
            row["coverage_survivors"] = float(np.mean((truth[:, fc] >= lo[:, fc]) & (truth[:, fc] <= hi[:, fc])))
            row["width_over_fluct"] = float(np.linalg.norm((hi - lo)[:, fc]) /
                                             np.linalg.norm(truth[:, fc] - truth[:, fc].mean(1, keepdims=True)))
            plot.update({f"{name}_{k}": val for name, val in (("median", med), ("lo", lo), ("hi", hi))})
        if k < 2:
            X, dz = cache["gp_states"][k], cache["gp_derivatives"][k]
            rate = features(X - shift[:, None], u) @ mean.T
            row["gp_derivative_residual"] = float(np.linalg.norm(rate.T - dz) / np.linalg.norm(dz))
        out["targets"].append(row)
    out["seconds"] = time.time() - start
    return out, plot


def run_one(args):
    cache_path, outdir, case, draws = args
    path = outdir / f"{case['id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    with np.load(cache_path) as data:
        cache = {key: data[key] for key in data.files}
    out, plot = fit_case(cache, case, draws)
    np.savez_compressed(outdir / f"{case['id']}.npz", **plot)
    path.write_text(json.dumps(out, indent=2))
    print(case["id"], [(x["re"], x.get("pm_error"), x["survival"]) for x in out["targets"]], flush=True)
    return out


def check():
    from opinf.operators import QuadraticOperator
    rng = np.random.default_rng(7)
    q, u = rng.normal(size=(3, 5)), .7
    np.testing.assert_array_equal(features(q, u)[:, 4:10].T, QuadraticOperator.ckron(q))
    O = rng.normal(size=(3, 14))
    J = O[:, 1:4] + u * O[:, 11:] + QuadraticOperator(O[:, 4:10]).jacobian(q[:, 0])
    np.testing.assert_allclose(O @ feature_jacobian(q[:, 0], u), J, atol=1e-12)
    A, y, sd = rng.normal(size=(8, 4)), rng.normal(size=8), np.exp(rng.normal(size=4))
    B, z = rng.normal(size=(6, 4)), rng.normal(size=6)
    mu, root = gaussian_fit(A, y, sd, B, z)
    cov = np.linalg.inv(A.T @ A + np.diag(sd ** -2) + B.T @ B)
    np.testing.assert_allclose(mu, cov @ (A.T @ y + B.T @ z), atol=1e-12)
    np.testing.assert_allclose(root @ root.T, cov, atol=1e-12)
    print("Feature, Jacobian and Gaussian-prior checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    p = sub.add_parser("prepare")
    p.add_argument("saved", type=Path)
    p.add_argument("fom_dir", type=Path)
    p.add_argument("output", type=Path)
    p = sub.add_parser("sweep")
    p.add_argument("cache", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--draws", type=int, default=32)
    p.add_argument("--jacobian-global", action="store_true")
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "prepare":
        prepare(args.saved, args.fom_dir, args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(run_one, [(args.cache, args.output, case, args.draws)
                                             for case in cases(args.jacobian_global)]))
        (args.output / "summary.json").write_text(json.dumps(results, indent=2))
