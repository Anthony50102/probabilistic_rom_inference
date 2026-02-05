# utils.py
"""Utilities for logging, timing, loading, saving, and data generation."""

__all__ = [
    "summarize_experiment",
    "save_figure",
    "generate_trajectory",
]

import os
import platform
import subprocess
import numpy as np
import matplotlib.pyplot as plt

import opinf


# =============================================================================
# Experiment Summary
# =============================================================================
def summarize_experiment(
    training_span: tuple[float, float],
    num_samples: int,
    noiselevel: float,
    num_regression_points: int,
    numPODmodes: int,
    gp_regularizer: float = None,
    ndraws: int = None,
    figures_path: str = None,
):
    """Summarize the experimental setup.
    
    Parameters
    ----------
    training_span : tuple
        (start, end) time for training data
    num_samples : int
        Number of snapshots sampled
    noiselevel : float
        Noise level as a fraction (e.g., 0.01 for 1%)
    num_regression_points : int
        Number of points for GP regression
    numPODmodes : int
        Number of POD modes retained
    gp_regularizer : float, optional
        GP regularization parameter
    ndraws : int, optional
        Number of posterior draws
    figures_path : str, optional
        Path to save report
    """
    report = [
        "EXPERIMENTAL SCENARIO",
        f"Data: {num_samples:d} uniformly sampled snapshots "
        f"over {training_span[0]:.2f} ≤ t < {training_span[1]:.2f} "
        f"with {noiselevel:.2%} noise",
        f"Dimension: retaining {numPODmodes} POD modes",
        f"Training: using {num_regression_points:d} regression points",
    ]
    if gp_regularizer is not None:
        report.append(f"GP regularization: eta = {gp_regularizer:.2e}")
    if ndraws is not None:
        report.append(f"Posterior: {ndraws} draws")
    report_str = "\n".join(report)

    if figures_path is not None:
        os.makedirs(figures_path, exist_ok=True)
        with open(os.path.join(figures_path, "report.txt"), "w") as out:
            out.write(report_str)
    
    print("\n" + report_str + "\n")
    return report_str


# =============================================================================
# Figure Saving
# =============================================================================
def _open_file(file_path: str):
    """Open a file with the system default application."""
    if os.path.isfile(file_path):
        if platform.system() == "Darwin":  # MacOS
            subprocess.call(("open", file_path))
        elif platform.system() == "Windows":  # Windows
            os.startfile(file_path)
        else:  # Linux
            subprocess.call(("xdg-open", file_path))


def save_figure(figname: str, figures_path: str = "figures", andopen: bool = False, fig=None):
    """Save the current matplotlib figure.
    
    Parameters
    ----------
    figname : str
        Filename for the figure
    figures_path : str
        Directory to save to
    andopen : bool
        Whether to open the file after saving
    fig : matplotlib.figure.Figure, optional
        Figure to save (defaults to current figure)
    """
    if fig is None:
        fig = plt.gcf()
    
    os.makedirs(figures_path, exist_ok=True)
    save_path = os.path.join(figures_path, figname)

    with opinf.utils.TimedBlock(f"Saving {save_path}"):
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.001, dpi=250)
        plt.close(fig)

    if andopen:
        _open_file(save_path)


# =============================================================================
# Data Generation
# =============================================================================
def generate_trajectory(
    config,
    training_span: tuple[float, float],
    num_samples: int,
    noiselevel: float = 0.0,
):
    """Generate sparse, noisy data for a single trajectory.

    Parameters
    ----------
    config : module
        Configuration module with FullOrderModel, initial_conditions, time_domain
    training_span : (float, float)
        Time domain over which to sample solution data.
    num_samples : int > 0
        Number of snapshots to sample.
    noiselevel : float >= 0
        Percentage of noise applied to the sampled snapshots.

    Returns
    -------
    model : FullOrderModel
        The full-order model instance
    full_time_domain : ndarray
        Full time domain
    true_states : ndarray
        True solution states
    time_domain_sampled : ndarray
        Sampled time points
    snapshots_sampled : ndarray
        Noisy snapshots at sampled times
    """
    with opinf.utils.TimedBlock("generating training data"):
        # Initialize and solve the truth model over the full domain
        model = config.FullOrderModel()
        true_states = model.solve(
            config.initial_conditions,
            config.time_domain,
        )

        # Uniformly sample from the training span --> training time domain
        time_domain_sampled = np.sort(
            np.random.uniform(
                training_span[0],
                training_span[1],
                size=num_samples,
            )
        )
        time_domain_sampled[0] = training_span[0]
        time_domain_sampled[-1] = training_span[1]

        # Get noisy snapshots over the training time domain
        snapshots_sampled = model.noise(
            model.solve(config.initial_conditions, time_domain_sampled),
            noiselevel,
        )

        return (
            model,
            config.time_domain,
            true_states,
            time_domain_sampled,
            snapshots_sampled,
        )
