"""Heat ROM helper utilities."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def generate_rom_solves(
    operator_samples: np.ndarray,
    rom,
    q0: np.ndarray,
    time_eval: np.ndarray,
    input_func: Optional[Callable] = None,
    max_samples: int = 200,
    q0_samples: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate stable ROM solves from operator samples for one trajectory.

    If ``q0_samples`` (shape ``(len(operator_samples), len(q0))``) is given, each
    trajectory starts from its own initial condition instead of the shared
    ``q0`` — propagating initial-condition uncertainty into the band.
    """
    solves = []
    n = min(len(operator_samples), max_samples)
    for i in range(n):
        rom.model._extract_operators(np.array(operator_samples[i]))
        q0_i = q0 if q0_samples is None else np.asarray(q0_samples[i])
        try:
            if input_func is not None:
                rom.model.predict(state0=q0_i, t=time_eval, input_func=input_func)
            else:
                rom.model.predict(state0=q0_i, t=time_eval)
            result = rom.model.predict_result_
            if hasattr(result, "y"):
                sol = result.y
            elif hasattr(result, "ys"):
                sol = np.array(result.ys).T
            else:
                continue
            if sol.shape[1] == len(time_eval) and np.all(np.isfinite(sol)):
                solves.append(sol)
        except Exception:
            pass
    if solves:
        return np.array(solves)
    return np.empty((0, len(q0), len(time_eval)))
