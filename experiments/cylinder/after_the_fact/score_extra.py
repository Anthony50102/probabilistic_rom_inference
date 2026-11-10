"""Declared diagnostics of the stability-prior round (DESIGN.md "Stability prior"), per run directory; runs next to the
data (VM). Reads 04_unified.npz, writes stability_extra.json:
  ep_violation_pm      ||C vec(PM*)|| / (||C||_2 ||PM*||_F), C built independently of core/constraints.py
  ep_violation_draws   max of the same over the draws
  trap_cert_pm         Q = I trapping certificate of PM*: min_m lambda_max(sym(A + J_H(m))) (< 0 = certified region)
  op_norm_pm, op_norm_draws_median   Frobenius norms (size of the fitted operators)
usage: python score_extra.py RUN_DIR [RUN_DIR ...]"""
import json, os, sys
import numpy as np
from scipy.optimize import minimize


def pairs(r):
    return [(j, i) for i in range(r) for j in range(i + 1)]


def ep_matrix(r):
    m = 1 + r + r * (r + 1) // 2
    col = {p: 1 + r + k for k, p in enumerate(pairs(r))}
    rows = []
    for a in range(r):
        for b in range(a, r):
            for c in range(b, r):
                row = np.zeros(r * m)
                for i in sorted({a, b, c}):
                    rest = [a, b, c]; rest.remove(i)
                    row[i * m + col[tuple(sorted(rest))]] += 1.0
                rows.append(row)
    return np.array(rows)


def jac_basis(r):
    P = pairs(r)
    D = np.zeros((r, len(P), r))
    for k, (j, i) in enumerate(P):
        D[i, k, j] += 1.0
        D[j, k, i] += 1.0
    return D


def trap_certificate(Om, r, D, m0):
    A, H = Om[:, 1:1 + r], Om[:, 1 + r:]
    f = lambda m: np.linalg.eigvalsh(0.5 * ((A + H @ np.tensordot(m, D, axes=1)) + (A + H @ np.tensordot(m, D, axes=1)).T))[-1]
    best = min((minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=4000 * r, xatol=1e-6, fatol=1e-9))
                for x0 in (m0, np.zeros(r))), key=lambda res: res.fun)
    return float(best.fun)


for run in sys.argv[1:]:
    d = np.load(os.path.join(run, "04_unified.npz"))
    O = d["O_samples"].astype(float); r = int(d["num_modes"]); n, _, m = O.shape
    C = ep_matrix(r); Cn = np.linalg.norm(C, 2)
    viol = lambda Om: float(np.linalg.norm(C @ Om.reshape(-1)) / (Cn * np.linalg.norm(Om)))
    PM = O.mean(0)
    out = dict(run=os.path.basename(run.rstrip("/")), r=r, n_draws=n,
               ep_violation_pm=viol(PM), ep_violation_draws=max(viol(o) for o in O),
               trap_cert_pm=trap_certificate(PM, r, jac_basis(r), d["snaps_comp"].mean(1)),
               op_norm_pm=float(np.linalg.norm(PM)), op_norm_draws_median=float(np.median(np.linalg.norm(O, axis=(1, 2)))))
    json.dump(out, open(os.path.join(run, "stability_extra.json"), "w"), indent=1)
    print(json.dumps(out))
