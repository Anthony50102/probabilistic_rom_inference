"""
Plotting utilities for Cubic Heat equation experiments.

Provides visualization methods for multi-trajectory ROM predictions
with input support, matching the euler experiment plotting style.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple, Callable, Dict

from core.plotting import Plotter


# =============================================================================
# Standalone utility functions
# =============================================================================

def _generate_rom_solves(
    operator_samples: np.ndarray,
    rom,
    q0: np.ndarray,
    time_eval: np.ndarray,
    input_func: Optional[Callable] = None,
    max_samples: int = 200,
) -> np.ndarray:
    """
    Generate ROM solves from operator samples for a single trajectory.

    Parameters
    ----------
    operator_samples : np.ndarray, shape (num_samples, r, d)
        Posterior operator samples.
    rom : opinf.ROM
        ROM model.
    q0 : np.ndarray, shape (r,)
        Initial condition.
    time_eval : np.ndarray
        Time grid for ROM evaluation.
    input_func : callable, optional
        Input function u(t).
    max_samples : int
        Max number of operator samples to try.

    Returns
    -------
    np.ndarray, shape (n_stable, r, len(time_eval))
    """
    solves = []
    n = min(len(operator_samples), max_samples)
    for i in range(n):
        rom.model._extract_operators(np.array(operator_samples[i]))
        try:
            if input_func is not None:
                rom.model.predict(state0=q0, t=time_eval, input_func=input_func)
            else:
                rom.model.predict(state0=q0, t=time_eval)
            result = rom.model.predict_result_
            if hasattr(result, 'y'):
                sol = result.y
            elif hasattr(result, 'ys'):
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


def compute_trajectory_errors(
    rom_solves: np.ndarray,
    true_compressed: np.ndarray,
    time_eval: np.ndarray,
    time_true: np.ndarray,
    num_modes: int,
) -> List[float]:
    """
    Compute relative errors for ROM solves against interpolated truth.

    Parameters
    ----------
    rom_solves : np.ndarray, shape (n_stable, r, n_eval)
    true_compressed : np.ndarray, shape (r, n_full)
    time_eval : np.ndarray, shape (n_eval,)
    time_true : np.ndarray, shape (n_full,)
    num_modes : int

    Returns
    -------
    errors : list of float
        Relative error for each stable ROM solve.
    """
    truth_at_eval = np.array([
        np.interp(time_eval, time_true, true_compressed[i])
        for i in range(num_modes)
    ])
    errors = []
    for sol in rom_solves:
        err = np.linalg.norm(sol - truth_at_eval) / np.linalg.norm(truth_at_eval)
        errors.append(err)
    return errors


def plot_heat_grid_search(
    grid_search_result,
    snapshots_compressed: np.ndarray,
    time_sampled: np.ndarray,
    time_eval_training: np.ndarray,
    time_eval_prediction: np.ndarray,
    num_modes: int,
    input_func: Callable,
    time_full: Optional[np.ndarray] = None,
    true_states_compressed: Optional[np.ndarray] = None,
    training_span: Optional[Tuple[float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Plot all stable deterministic ROM solves from grid search (with input support).

    Uses operator_plot style: single column, modes as rows, purple
    median + 5-95% band from all stable solves, gray true trajectory,
    training span shading, best solve in blue.
    """
    from core.plotting import plot_deterministic_rom_solves
    return plot_deterministic_rom_solves(
        grid_search_result=grid_search_result,
        snapshots_compressed=snapshots_compressed,
        time_sampled=time_sampled,
        time_eval_training=time_eval_training,
        time_eval_prediction=time_eval_prediction,
        time_full=time_full,
        true_states_compressed=true_states_compressed,
        input_func=input_func,
        training_span=training_span,
        figsize=figsize,
    )


# =============================================================================
# HeatPlotter class
# =============================================================================

class HeatPlotter(Plotter):
    """Plotter for Cubic Heat equation experiments.

    Extends the base Plotter with multi-trajectory visualization methods
    for systems with external inputs (multiple ICs / input parameters).
    Styling matches the Euler experiment plotter for visual consistency.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # -----------------------------------------------------------------
    # Multi-trajectory plot (rows = trajectories, cols = modes)
    # -----------------------------------------------------------------
    def multi_trajectory_plot(
        self,
        operator_samples: np.ndarray,
        rom,
        trajectories: List[Dict],
        time_eval: np.ndarray,
        figsize: Optional[Tuple[float, float]] = None,
        max_num_samples: int = 200,
        training_span: Optional[Tuple[float, float]] = None,
    ):
        """
        Plot ROM predictions for multiple trajectories in one figure.

        Layout: one row per trajectory, one column per POD mode.
        Style matches euler's single-column operator_plot.

        Parameters
        ----------
        operator_samples : np.ndarray, shape (num_samples, r, d)
            Posterior operator samples.
        rom : opinf.ROM
            ROM model used for predictions.
        trajectories : list of dict
            Each dict contains:
            - 'q0': np.ndarray shape (r,), initial condition
            - 'input_func': callable, input function u(t)
            - 'snapshots': np.ndarray shape (r, n) or None, noisy data
            - 'time_snapshots': np.ndarray shape (n,) or None
            - 'true_compressed': np.ndarray shape (r, n_full) or None
            - 'time_true': np.ndarray shape (n_full,) or None
            - 'label': str, row label
        time_eval : np.ndarray
            Time points for ROM evaluation.
        figsize : tuple, optional
        max_num_samples : int
        training_span : tuple of (float, float), optional
            If given, shade the training time region with a light background.

        Returns
        -------
        fig, axes, all_rom_solves : list of np.ndarray per trajectory
        """
        n_traj = len(trajectories)
        n_modes = self.numPODmodes

        if figsize is None:
            figsize = (4 * n_modes, 2.5 * n_traj)

        fig, axes = plt.subplots(
            n_traj, n_modes, figsize=figsize,
            sharex=True, squeeze=False,
        )

        all_rom_solves = []

        for row, traj in enumerate(trajectories):
            rom_solves = _generate_rom_solves(
                operator_samples, rom, traj['q0'], time_eval,
                traj.get('input_func'), max_num_samples,
            )
            all_rom_solves.append(rom_solves)
            n_stable = len(rom_solves)
            n_tried = min(len(operator_samples), max_num_samples)
            label = traj.get('label', f'Trajectory {row + 1}')

            for col in range(n_modes):
                ax = axes[row, col]

                # Training span shading
                if training_span is not None:
                    ax.axvspan(training_span[0], training_span[1],
                               color='gray', alpha=0.10, zorder=0)

                # True solution (clean, full resolution)
                if traj.get('true_compressed') is not None and traj.get('time_true') is not None:
                    ax.plot(
                        traj['time_true'], traj['true_compressed'][col],
                        color='tab:gray', lw=2,
                        label='True solution' if (row == 0 and col == 0) else None,
                    )

                # Training data (noisy, subsampled)
                if traj.get('snapshots') is not None and traj.get('time_snapshots') is not None:
                    ax.plot(
                        traj['time_snapshots'], traj['snapshots'][col],
                        'k*', ms=5, zorder=5,
                        label='Training data' if (row == 0 and col == 0) else None,
                    )

                # ROM predictions
                if n_stable > 0:
                    ax.plot(
                        time_eval,
                        np.median(rom_solves[:, col, :], axis=0),
                        color='tab:purple', linestyle='--', alpha=0.9, lw=2,
                        label='ROM median' if (row == 0 and col == 0) else None,
                        zorder=0
                    )
                    ax.fill_between(
                        time_eval,
                        np.percentile(rom_solves[:, col, :], 5, axis=0),
                        np.percentile(rom_solves[:, col, :], 95, axis=0),
                        color='tab:purple', alpha=0.15,
                        label='ROM 5\u201395%' if (row == 0 and col == 0) else None,
                        zorder=0
                    )

                # Column titles (top row)
                if row == 0:
                    ax.set_title(f'Mode {col + 1}')
                # Row labels (left column)
                if col == 0:
                    ax.set_ylabel(f'{label}\n({n_stable}/{n_tried} stable)',
                                  fontsize=9)
                # X-axis label (bottom row)
                if row == n_traj - 1:
                    ax.set_xlabel('Time')

                # Fix y-axis to truth for cross-method comparison
                if traj.get('true_compressed') is not None:
                    from core.plotting import _ylim_from_truth
                    ax.set_ylim(*_ylim_from_truth(traj['true_compressed'][col]))

        # Single legend at the top
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center',
                       ncol=len(handles), fontsize=9,
                       bbox_to_anchor=(0.5, 1.02))

        fig.suptitle('ROM Predictions: All Trajectories', fontsize=14, y=1.05)
        fig.tight_layout()
        return fig, axes, all_rom_solves

    # -----------------------------------------------------------------
    # Single-trajectory plot (rows = modes, single column)
    # -----------------------------------------------------------------
    def single_trajectory_plot(
        self,
        operator_samples: np.ndarray,
        rom,
        q0: np.ndarray,
        time_eval: np.ndarray,
        input_func: Optional[Callable] = None,
        snapshots: Optional[np.ndarray] = None,
        time_snapshots: Optional[np.ndarray] = None,
        true_compressed: Optional[np.ndarray] = None,
        time_true: Optional[np.ndarray] = None,
        title: str = 'ROM Predictions',
        figsize: Optional[Tuple[float, float]] = None,
        max_num_samples: int = 200,
        training_span: Optional[Tuple[float, float]] = None,
    ):
        """
        Single-trajectory plot matching euler's single-column operator_plot.

        Modes as rows, single column.

        Parameters
        ----------
        operator_samples : np.ndarray, shape (num_samples, r, d)
        rom : opinf.ROM
        q0 : np.ndarray, shape (r,)
        time_eval : np.ndarray
        input_func : callable, optional
        snapshots : np.ndarray, optional, shape (r, n)
        time_snapshots : np.ndarray, optional
        true_compressed : np.ndarray, optional, shape (r, n_full)
        time_true : np.ndarray, optional
        title : str
        figsize : tuple, optional
        max_num_samples : int

        Returns
        -------
        fig, axes, rom_solves
        """
        n_modes = self.numPODmodes
        if figsize is None:
            figsize = (10, 2.5 * n_modes)

        rom_solves = _generate_rom_solves(
            operator_samples, rom, q0, time_eval,
            input_func, max_num_samples,
        )

        fig, axes = plt.subplots(n_modes, 1, figsize=figsize, sharex=True)
        if n_modes == 1:
            axes = [axes]

        n_stable = len(rom_solves)
        n_tried = min(len(operator_samples), max_num_samples)

        for i in range(n_modes):
            ax = axes[i]

            # Training span shading
            if training_span is not None:
                ax.axvspan(training_span[0], training_span[1],
                           color='gray', alpha=0.10, zorder=0)

            # True solution
            if true_compressed is not None and time_true is not None:
                ax.plot(time_true, true_compressed[i],
                        color='tab:gray', lw=2, label='True solution')

            # Training data
            if snapshots is not None and time_snapshots is not None:
                ax.plot(time_snapshots, snapshots[i],
                        'k*', ms=5, label='Training data', zorder=5)

            # ROM predictions
            if n_stable > 0:
                ax.plot(
                    time_eval,
                    np.median(rom_solves[:, i, :], axis=0),
                    color='tab:purple', linestyle='--', alpha=0.9, lw=2, label='ROM median',
                )
                ax.fill_between(
                    time_eval,
                    np.percentile(rom_solves[:, i, :], 5, axis=0),
                    np.percentile(rom_solves[:, i, :], 95, axis=0),
                    color='tab:purple', alpha=0.15, label='ROM 5\u201395%',
                )

            ax.set_ylabel(f'Mode {i + 1}')
            if i == 0:
                ax.legend(loc='upper right', fontsize=9)

            # Fix y-axis to truth for cross-method comparison
            if true_compressed is not None:
                from core.plotting import _ylim_from_truth
                ax.set_ylim(*_ylim_from_truth(true_compressed[i]))

        axes[-1].set_xlabel('Time')
        fig.suptitle(f'{title}  ({n_stable}/{n_tried} stable)', fontsize=14)
        fig.tight_layout()
        return fig, axes, rom_solves


# =============================================================================
# Operator matrix heatmap comparison
# =============================================================================

def plot_operator_comparison(
    deterministic_operator: np.ndarray,
    posterior_mean: np.ndarray,
    title: str = "Operator Comparison",
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Side-by-side heatmap comparison of operator matrices.

    Panels: deterministic (grid search) | posterior mean | difference.

    Parameters
    ----------
    deterministic_operator : np.ndarray, shape (r, d)
        Best operator from grid search.
    posterior_mean : np.ndarray, shape (r, d)
        Posterior mean operator from SVI or MCMC.
    title : str
        Figure suptitle.
    figsize : tuple, optional

    Returns
    -------
    fig, axes
    """
    diff = posterior_mean - deterministic_operator

    if figsize is None:
        r, d = deterministic_operator.shape
        figsize = (min(5 * 3, 18), max(r * 0.6, 3))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Shared colour scale for the two operator panels
    vmax_op = max(np.abs(deterministic_operator).max(), np.abs(posterior_mean).max())
    vmin_op = -vmax_op

    im0 = axes[0].imshow(deterministic_operator, aspect='auto', cmap='RdBu_r',
                          vmin=vmin_op, vmax=vmax_op)
    axes[0].set_title('Deterministic\n(grid search)')

    im1 = axes[1].imshow(posterior_mean, aspect='auto', cmap='RdBu_r',
                          vmin=vmin_op, vmax=vmax_op)
    axes[1].set_title('Posterior mean')

    # Difference panel with its own scale
    vmax_d = np.abs(diff).max()
    im2 = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r',
                          vmin=-vmax_d, vmax=vmax_d)
    axes[2].set_title('Difference\n(posterior − det.)')

    for ax in axes:
        ax.set_xlabel('Operator column')
        ax.set_ylabel('Mode')

    fig.colorbar(im1, ax=axes[:2].tolist(), shrink=0.8, label='Coefficient value')
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label='Difference')

    # Annotate Frobenius norms
    norm_det = np.linalg.norm(deterministic_operator)
    norm_post = np.linalg.norm(posterior_mean)
    norm_diff = np.linalg.norm(diff)
    fig.text(0.5, -0.02,
             f'‖Det‖_F = {norm_det:.3f}    ‖Post‖_F = {norm_post:.3f}    '
             f'‖Diff‖_F = {norm_diff:.3f}  ({100*norm_diff/norm_det:.1f}% of det.)',
             ha='center', fontsize=9, style='italic')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axes