"""Configuration for the cylinder-wake (incompressible Navier–Stokes, DFG 2D-2, Re = 100) experiment.

The full-order data come from ``fom_dolfinx.py`` (FEniCSx): the velocity (u, v) sampled on a uniform 0.01 grid
(18,564 values per snapshot), saved every 0.005 time units from rest to T = 24. Incompressible Navier–Stokes is
quadratic in the velocity, so a Galerkin ROM on any linear basis has exactly the cAH structure
dq/dt = c + Aq + H[q ⊗ q]; the ROM here is learned from noisy snapshots only.

Tasks (fixed before any ROM fit, see DESIGN.md):
  lc  periodic vortex shedding: training window [t0, t0 + 1] with t0 = 16 (about three shedding periods),
      forecast to t0 + 3.
  tr  onset of shedding: the lift amplitude A_k of each cycle relative to the limit-cycle amplitude A_inf (median
      over t >= 16); t_b = first cycle with A_k >= 0.6 A_inf, t_a = last cycle before t_b with A_k <= 0.1 A_inf;
      training window [t_a, t_b], forecast to t_a + 3 (t_b - t_a).
"""

import os

import numpy as np
import opinf
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
FOM_PATH = os.environ.get("CYL_FOM", os.path.join(HERE, "data", "fom_medium.npz"))
LC_START = 16.0
LC_TRAIN, LC_TOTAL = 1.0, 3.0
TR_LOW, TR_HIGH, TR_TOTAL_FACTOR = 0.1, 0.6, 3.0
SAMPLES_PER_TIME = {"dense": 250, "sparse": 55}       # per 10 D/U (a 1.0 window for the cylinder): the Euler counts


class Basis(opinf.basis.PODBasis):
    """Uncentred POD basis that is never silently refitted (see the burgers_2d config)."""

    def fit(self, states):
        if self.entries is not None:
            return self
        return super().fit(states)


def load_fom(path=FOM_PATH):
    d = np.load(path)
    D = float(d["D"]) if "D" in d.files else 0.1
    return dict(t=d["t"], snapshots=d["snapshots"], t_steps=d["t_steps"], cl=d["cl"], cd=d["cd"],
                xs=d["xs"], ys=d["ys"], inside=d["inside"], D=D, time_scale=D / 0.1,
                geometry=str(d["geometry"]) if "geometry" in d.files else "cylinder",
                nu=float(d["nu"]) if "nu" in d.files else 1e-3)


def cycle_amplitudes(t_steps, cl):
    """Half peak-to-peak lift amplitude of each cycle, timed at the cycle's maximum."""
    up = np.flatnonzero((cl[:-1] < 0) & (cl[1:] >= 0))
    times, amps = [], []
    for a, b in zip(up[:-1], up[1:]):
        seg = cl[a:b + 1]
        times.append(t_steps[a + np.argmax(seg)])
        amps.append(0.5 * (seg.max() - seg.min()))
    return np.array(times), np.array(amps)


def task_window(task, fom):
    """(start, training length, total length) of a task, in the FOM's absolute time.

    lc is defined in convective units (D/U): for a geometry with reference diameter D the times scale by D / 0.1
    (the pinball, D = 0.05: train [8, 8.5], forecast to 9.5); the cylinder's window is unchanged."""
    if task == "lc":
        s = fom.get("time_scale", 1.0)
        return s * LC_START, s * LC_TRAIN, s * LC_TOTAL
    if task == "tr":
        times, amps = cycle_amplitudes(fom["t_steps"], fom["cl"])
        a_inf = np.median(amps[times >= fom.get("time_scale", 1.0) * LC_START])
        rel = amps / a_inf
        k_b = int(np.flatnonzero(rel >= TR_HIGH)[0])
        below = np.flatnonzero(rel[:k_b] <= TR_LOW)
        k_a = int(below[-1]) if len(below) else 0
        t_a, t_b = float(times[k_a]), float(times[k_b])
        return t_a, t_b - t_a, TR_TOTAL_FACTOR * (t_b - t_a)
    raise ValueError(f"unknown task {task!r}")


def window_states(fom, start, total):
    """Saved snapshots on [start, start + total] (relative times) and a cubic-in-time interpolant of them.

    The FOM saves every 0.005 (about 66 points per shedding period), so the interpolant's error at a training
    sample time is far below the noise levels used."""
    t = fom["t"]
    keep = (t >= start - 0.05) & (t <= start + total + 0.05)
    spline = CubicSpline(t[keep] - start, fom["snapshots"][keep].astype(float), axis=0)
    on = (t >= start - 1e-9) & (t <= start + total + 1e-9)
    return t[on] - start, fom["snapshots"][on].astype(float).T, spline


def noise(states, level, rng=np.random):
    """Pointwise Gaussian noise proportional to the state, as in the 2D diffusion–reaction benchmark."""
    if not level:
        return states
    return rng.normal(loc=states, scale=np.abs(states * level), size=states.shape)
