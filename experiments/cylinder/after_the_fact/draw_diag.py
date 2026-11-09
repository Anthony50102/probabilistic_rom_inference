"""Why do the transient-task draws blow up? (session diagnostic; reads one run's 04_unified.npz on the VM)

usage: python tr_diag.py RUN_DIR"""
import os
import sys

import numpy as np
import opinf
from scipy.integrate import solve_ivp

d = np.load(os.path.join(sys.argv[1], "04_unified.npz"))
O = d["O_samples"]
r = int(d["num_modes"])
t_pred, t_samp, Y, truth = d["t_pred"], d["t_samp"], d["snaps_comp"], d["true_comp"]
train = float(d["training_span"][1])
ckron = opinf.operators.QuadraticOperator.ckron
print(f"r={r}, O {O.shape}, train {train:.3f}, horizon {t_pred[-1]:.3f}")


def rhs_of(Om):
    c, A, H = Om[:, 0], Om[:, 1:1 + r], Om[:, 1 + r:]
    return (lambda t, q: c + A @ q + H @ ckron(q)), A, H


def run(Om, q0, label):
    f, A, H = rhs_of(Om)
    stop = lambda t, q: np.max(np.abs(q)) - 10 * np.max(np.abs(truth))
    stop.terminal = True
    sol = solve_ivp(f, (t_pred[0], t_pred[-1]), q0, t_eval=t_pred, method="RK45", rtol=1e-6, atol=1e-8, events=stop)
    n = sol.y.shape[1]
    fit = t_pred[:n] <= train
    e_tr = np.linalg.norm(sol.y[:, fit] - truth[:, :n][:, fit]) / np.linalg.norm(truth[:, :n][:, fit])
    ext = t_pred[:n] > train
    e_fc = np.linalg.norm(sol.y[:, ext] - truth[:, :n][:, ext]) / np.linalg.norm(truth[:, ~(t_pred <= train)] - truth[:, ~(t_pred <= train)].mean(1, keepdims=True)) if n == len(t_pred) else np.nan
    eig = np.linalg.eigvals(A)
    print(f"  {label:28s} reached t={t_pred[n - 1]:.3f} ({'OK' if n == len(t_pred) else 'BLOW-UP'}), "
          f"train err {e_tr:.3f}, fc fluct err {e_fc:.3f}, max Re eig(A) {eig.real.max():+.2f}, |H| {np.linalg.norm(H):.2e}")
    return sol


print("posterior median operator:")
run(np.median(O, axis=0), Y[:, 0], "median O")
print("posterior mean operator:")
run(O.mean(0), Y[:, 0], "mean O")
print("five draws:")
for i in np.linspace(0, len(O) - 1, 8, dtype=int):
    run(O[i], Y[:, 0], f"draw {i}")

# Reference: least squares on the NOISE-FREE reduced truth, training window only, derivatives from the 0.005 grid.
fit = t_pred <= train + 1e-9
Q = truth[:, fit]
dQ = np.gradient(Q, t_pred[fit], axis=1, edge_order=2)
D = np.vstack([np.ones(Q.shape[1]), Q, np.stack([ckron(q) for q in Q.T], axis=1)]).T
print("least squares on noise-free truth (training window only):")
for lam in (1e-8, 1e-4, 1e-2, 1, 1e2):
    Om = np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ dQ.T).T
    run(Om, truth[:, 0], f"clean LS lam={lam:g}")
print("least squares on noise-free truth over the WHOLE window (cheating; tells if cAH can represent it):")
Q = truth
dQ = np.gradient(Q, t_pred, axis=1, edge_order=2)
D = np.vstack([np.ones(Q.shape[1]), Q, np.stack([ckron(q) for q in Q.T], axis=1)]).T
for lam in (1e-8, 1e-4, 1e-2):
    Om = np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ dQ.T).T
    run(Om, truth[:, 0], f"whole-window LS lam={lam:g}")
