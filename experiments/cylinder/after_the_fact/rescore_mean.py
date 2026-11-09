"""After-the-fact rescoring (session diagnostic, no refits): forecast with the POSTERIOR-MEAN operator instead of
the declared 'median of stable draws'. Reads each run's 04_unified.npz; writes rescore_mean.json next to it.
usage: python rescore_mean.py RUN_DIR [RUN_DIR ...]"""
import json, os, sys
import numpy as np, opinf
from scipy.integrate import solve_ivp
ckron = opinf.operators.QuadraticOperator.ckron
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
relf = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b - b.mean(1, keepdims=True)))

def period(t, x):
    x = x - x.mean(); up = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if len(up) < 2: return np.nan
    tc = t[up] - x[up] * (t[up + 1] - t[up]) / (x[up + 1] - x[up]); return float(np.mean(np.diff(tc)))

def integrate(Om, q0, t, cap):
    r = len(q0); c, A, H = Om[:, 0], Om[:, 1:1 + r], Om[:, 1 + r:]
    ev = lambda tt, q: np.max(np.abs(q)) - cap; ev.terminal = True
    s = solve_ivp(lambda tt, q: c + A @ q + H @ ckron(q), (t[0], t[-1]), q0, t_eval=t, rtol=1e-6, atol=1e-8, events=ev)
    return s.y if s.y.shape[1] == len(t) else None

for run in sys.argv[1:]:
    d = np.load(os.path.join(run, "04_unified.npz"))
    O = d["O_samples"].astype(float); t = d["t_pred"]; train = float(d["training_span"][1])
    V, X, Tc, Y = d["basis_entries"], d["true_states"], d["true_comp"], d["snaps_comp"]
    fit, fc = t <= train + 1e-12, t > train + 1e-12
    q0 = d["state0_samples"].mean(0)
    cap = 2 * np.max(np.abs(Y))
    out = dict(run=os.path.basename(run.rstrip("/")), r=int(d["num_modes"]), n_draws=len(O),
               declared_stable_pct=float(d["stability_pct"]))
    sol = integrate(O.mean(0), q0, t, cap)
    out["mean_op_bounded"] = sol is not None
    # Exact link reduced -> full order (orthonormal V, Tc = V^T X): full^2 = (reduced * ratio)^2 + floor^2.
    fl = lambda Z: np.linalg.norm(Z - Z.mean(1, keepdims=True))
    out["fluct_norm_ratio_reduced_to_full"] = float(fl(Tc[:, fc]) / fl(X[:, fc]))
    out["floor_full_order_fc_fluct"] = relf(V @ Tc[:, fc], X[:, fc])
    out["check_Tc_is_VtX"] = float(np.linalg.norm(V.T @ X - Tc) / np.linalg.norm(Tc))
    if sol is not None:
        lead = int(np.argmax(np.std(Tc[:, fc], 1)))
        out.update(mean_op_fc_fluct_reduced=float(np.linalg.norm(sol[:, fc] - Tc[:, fc]) / np.linalg.norm(Tc[:, fc] - Tc[:, fc].mean(1, keepdims=True))),
                   mean_op_train_fluct_reduced=float(np.linalg.norm(sol[:, fit] - Tc[:, fit]) / np.linalg.norm(Tc[:, fit] - Tc[:, fit].mean(1, keepdims=True))),
                   mean_op_full_order_fc_fluct=relf(V @ sol[:, fc], X[:, fc]),
                   mean_op_strouhal=0.1 / period(t[fc], sol[lead, fc]), true_strouhal=0.1 / period(t[fc], Tc[lead, fc]),
                   mean_op_amplitude=float(np.std(sol[lead, fc])), true_amplitude=float(np.std(Tc[lead, fc])))
        tl = np.arange(t[0], t[0] + 10 * (train - t[0]) + 1e-9, t[1] - t[0])
        out["mean_op_long_10x_bounded"] = integrate(O.mean(0), q0, tl, cap) is not None
        frames = np.array([np.argmin(np.abs(t - (train + f * (t[-1] - train)))) for f in (0.5, 1.0)])
        np.savez_compressed(os.path.join(run, "rescore_mean.npz"), t=t, mean=sol[:5],
                            frame_mean=(V @ sol[:, frames]).astype(np.float32))
    json.dump(out, open(os.path.join(run, "rescore_mean.json"), "w"), indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}))
