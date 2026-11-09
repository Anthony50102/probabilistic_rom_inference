"""Does centring fix the transient task? LS on noise-free truth in a centred POD basis (session diagnostic)."""
import os, sys
import numpy as np, opinf
from scipy.integrate import solve_ivp
d = np.load(os.path.join(sys.argv[1], "04_unified.npz"))
X, t, train = d["true_states"].astype(float), d["t_pred"], float(d["training_span"][1])
ckron = opinf.operators.QuadraticOperator.ckron
fit = t <= train + 1e-9
for centre in (False, True):
    shift = X[:, fit].mean(1, keepdims=True) if centre else 0
    U, s, _ = np.linalg.svd(X[:, fit] - shift, full_matrices=False)
    for r in (5, 9, 13):
        V = U[:, :r]
        Q = V.T @ (X - shift)
        floor = np.linalg.norm((V @ Q + shift - X)[:, ~fit]) / np.linalg.norm((X - X.mean(1, keepdims=True))[:, ~fit])
        for window, mask in (("train", fit), ("whole", np.ones_like(fit))):
            Qm, tm = Q[:, mask], t[mask]
            dQ = np.gradient(Qm, tm, axis=1, edge_order=2)
            D = np.vstack([np.ones(Qm.shape[1]), Qm, np.stack([ckron(q) for q in Qm.T], 1)]).T
            out = []
            for lam in (1e-6, 1e-3, 1e-1, 1e1):
                Om = np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ dQ.T).T
                c, A, H = Om[:, 0], Om[:, 1:1 + r], Om[:, 1 + r:]
                ev = lambda tt, q: np.max(np.abs(q)) - 10 * np.max(np.abs(Q)); ev.terminal = True
                sol = solve_ivp(lambda tt, q: c + A @ q + H @ ckron(q), (t[0], t[-1]), Q[:, 0], t_eval=t,
                                rtol=1e-6, atol=1e-8, events=ev)
                n = sol.y.shape[1]
                fc = ~fit[:n]
                err = (np.linalg.norm(sol.y[:, fc] - Q[:, :n][:, fc]) / np.linalg.norm(Q[:, ~fit] - Q[:, ~fit].mean(1, keepdims=True))
                       if n == len(t) else np.nan)
                out.append(f"lam {lam:g}: {'ok' if n == len(t) else 'blow@%.2f' % t[n-1]} {err:.2f}")
            print(f"centre={centre!s:5} r={r:2d} floorFl {floor:.3f} LS[{window}]  " + " | ".join(out))
