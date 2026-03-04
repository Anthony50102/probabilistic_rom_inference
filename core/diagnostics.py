# core/diagnostics.py
"""
Bayesian model diagnostics for Probabilistic ROM Inference.

Provides post-inference diagnostic tools to detect common pitfalls:
- Posterior correlation analysis
- Effective sample size (ESS)
- R-hat convergence diagnostic
- Divergence detection (MCMC)
- Prior-posterior overlap / sensitivity
- Trace plot visualization
- Rank plots for chain mixing

Works with both SVI and MCMC results from the bayesian_opinf module.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Dict, List, Tuple, Union, Callable
from dataclasses import dataclass, field


# =============================================================================
# Diagnostic Result Container
# =============================================================================

@dataclass
class DiagnosticReport:
    """Container for all diagnostic results with a summary printer."""

    # Correlation
    correlation_matrix: Optional[np.ndarray] = None
    high_correlations: Optional[List[Tuple[str, str, float]]] = None

    # ESS (per-parameter)
    ess: Optional[Dict[str, float]] = None
    ess_bulk: Optional[Dict[str, float]] = None
    ess_tail: Optional[Dict[str, float]] = None

    # R-hat (per-parameter)
    rhat: Optional[Dict[str, float]] = None

    # Divergences
    num_divergences: Optional[int] = None
    divergence_fraction: Optional[float] = None

    # Prior-posterior overlap
    prior_posterior_overlap: Optional[Dict[str, float]] = None

    # Warnings collected during analysis
    warnings: List[str] = field(default_factory=list)

    def summary(self, verbose: bool = True) -> str:
        """Print a human-readable summary of all diagnostics."""
        lines = []
        lines.append("=" * 64)
        lines.append("  BAYESIAN MODEL DIAGNOSTIC REPORT")
        lines.append("=" * 64)

        # --- Correlation ---
        if self.high_correlations is not None:
            lines.append("\n--- Posterior Correlation ---")
            if len(self.high_correlations) == 0:
                lines.append("  No high correlations detected (|r| > 0.9).")
            else:
                lines.append(f"  {len(self.high_correlations)} highly correlated pairs (|r| > 0.9):")
                for p1, p2, r in self.high_correlations[:10]:
                    lines.append(f"    {p1} <-> {p2}: r = {r:+.3f}")
                if len(self.high_correlations) > 10:
                    lines.append(f"    ... and {len(self.high_correlations) - 10} more")

        # --- ESS ---
        if self.ess is not None:
            lines.append("\n--- Effective Sample Size (ESS) ---")
            ess_vals = np.array(list(self.ess.values()))
            lines.append(f"  Min ESS:  {ess_vals.min():.1f}")
            lines.append(f"  Mean ESS: {ess_vals.mean():.1f}")
            lines.append(f"  Max ESS:  {ess_vals.max():.1f}")
            low_ess = {k: v for k, v in self.ess.items() if v < 100}
            if low_ess:
                lines.append(f"  WARNING: {len(low_ess)} parameters with ESS < 100:")
                for k, v in sorted(low_ess.items(), key=lambda x: x[1])[:5]:
                    lines.append(f"    {k}: ESS = {v:.1f}")

        # --- R-hat ---
        if self.rhat is not None:
            lines.append("\n--- R-hat Convergence ---")
            rhat_vals = np.array(list(self.rhat.values()))
            lines.append(f"  Min R-hat:  {rhat_vals.min():.4f}")
            lines.append(f"  Mean R-hat: {rhat_vals.mean():.4f}")
            lines.append(f"  Max R-hat:  {rhat_vals.max():.4f}")
            bad_rhat = {k: v for k, v in self.rhat.items() if v > 1.01}
            if bad_rhat:
                lines.append(f"  WARNING: {len(bad_rhat)} parameters with R-hat > 1.01:")
                for k, v in sorted(bad_rhat.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"    {k}: R-hat = {v:.4f}")
            else:
                lines.append("  All R-hat values <= 1.01. Chains appear converged.")

        # --- Divergences ---
        if self.num_divergences is not None:
            lines.append("\n--- MCMC Divergences ---")
            lines.append(f"  Divergent transitions: {self.num_divergences}")
            if self.divergence_fraction is not None:
                lines.append(f"  Fraction: {self.divergence_fraction:.2%}")
            if self.num_divergences > 0:
                lines.append("  WARNING: Divergences indicate the posterior may not be"
                             " well-explored.")

        # --- Prior-Posterior Overlap ---
        if self.prior_posterior_overlap is not None:
            lines.append("\n--- Prior-Posterior Overlap ---")
            for k, v in self.prior_posterior_overlap.items():
                status = "OK" if 0.1 < v < 0.9 else "CHECK"
                lines.append(f"  {k}: overlap = {v:.2%} [{status}]")

        # --- Collected warnings ---
        if self.warnings:
            lines.append("\n--- Summary Warnings ---")
            for i, w in enumerate(self.warnings, 1):
                lines.append(f"  {i}. {w}")

        lines.append("\n" + "=" * 64)
        text = "\n".join(lines)
        if verbose:
            print(text)
        return text


# =============================================================================
# Core Diagnostic Functions
# =============================================================================

def compute_posterior_correlation(
    samples: dict,
    param_name: str = "O",
    threshold: float = 0.9,
) -> Tuple[np.ndarray, List[str], List[Tuple[str, str, float]]]:
    """
    Compute correlation matrix of flattened posterior operator samples.

    Parameters
    ----------
    samples : dict
        Posterior samples dict (from SVI or MCMC).
    param_name : str
        Name of the operator parameter to analyze.
    threshold : float
        Absolute correlation threshold for flagging pairs.

    Returns
    -------
    corr_matrix : np.ndarray
        Full correlation matrix of the flattened operator entries.
    param_labels : list of str
        Labels for each flattened parameter (e.g., "O[0,0]").
    high_pairs : list of (str, str, float)
        Pairs with |correlation| > threshold.
    """
    O_samples = _extract_param(samples, param_name)

    # Flatten each sample to 1-D: (n_samples, n_params)
    n_samples = O_samples.shape[0]
    flat = O_samples.reshape(n_samples, -1)

    # Generate labels
    if O_samples.ndim == 3:
        rows, cols = O_samples.shape[1], O_samples.shape[2]
        labels = [f"{param_name}[{i},{j}]" for i in range(rows) for j in range(cols)]
    else:
        labels = [f"{param_name}[{i}]" for i in range(flat.shape[1])]

    corr = np.corrcoef(flat.T)

    # Find highly correlated pairs (upper triangle only)
    high_pairs = []
    n_params = corr.shape[0]
    for i in range(n_params):
        for j in range(i + 1, n_params):
            if abs(corr[i, j]) > threshold:
                high_pairs.append((labels[i], labels[j], float(corr[i, j])))

    # Sort by absolute correlation descending
    high_pairs.sort(key=lambda x: -abs(x[2]))
    return corr, labels, high_pairs


def compute_ess(
    samples: dict,
    param_name: str = "O",
) -> Dict[str, float]:
    """
    Compute effective sample size for each element of a parameter.

    Uses the initial monotone sequence estimator (Geyer 1992).
    For SVI samples (independent draws), ESS ≈ n_samples.

    Parameters
    ----------
    samples : dict
        Posterior samples dict.
    param_name : str
        Parameter name.

    Returns
    -------
    ess_dict : dict
        Mapping from parameter label to ESS value.
    """
    O_samples = _extract_param(samples, param_name)
    n_samples = O_samples.shape[0]
    flat = O_samples.reshape(n_samples, -1)

    if O_samples.ndim == 3:
        rows, cols = O_samples.shape[1], O_samples.shape[2]
        labels = [f"{param_name}[{i},{j}]" for i in range(rows) for j in range(cols)]
    else:
        labels = [f"{param_name}[{i}]" for i in range(flat.shape[1])]

    ess_dict = {}
    for idx, label in enumerate(labels):
        chain = flat[:, idx]
        ess_dict[label] = _ess_1d(chain)

    return ess_dict


def compute_rhat(
    samples_by_chain: list,
    param_name: str = "O",
) -> Dict[str, float]:
    """
    Compute split R-hat for each element of a parameter across chains.

    Parameters
    ----------
    samples_by_chain : list of dict
        List of sample dicts, one per chain. Each dict has key `param_name`.
    param_name : str
        Parameter name.

    Returns
    -------
    rhat_dict : dict
        Mapping from parameter label to R-hat value.
    """
    chains = []
    for chain_samples in samples_by_chain:
        O = _extract_param(chain_samples, param_name)
        n = O.shape[0]
        flat = O.reshape(n, -1)
        chains.append(flat)

    n_params = chains[0].shape[1]
    if chains[0].shape[1] != n_params:
        raise ValueError("All chains must have the same number of parameters.")

    # Generate labels from first chain
    O0 = _extract_param(samples_by_chain[0], param_name)
    if O0.ndim == 3:
        rows, cols = O0.shape[1], O0.shape[2]
        labels = [f"{param_name}[{i},{j}]" for i in range(rows) for j in range(cols)]
    else:
        labels = [f"{param_name}[{i}]" for i in range(n_params)]

    rhat_dict = {}
    for idx, label in enumerate(labels):
        per_chain = [c[:, idx] for c in chains]
        rhat_dict[label] = _rhat_1d(per_chain)

    return rhat_dict


def detect_divergences(mcmc_result) -> Tuple[int, float]:
    """
    Count divergent transitions from an MCMC result.

    Parameters
    ----------
    mcmc_result : MCMCResult or numpyro.infer.MCMC
        Result object from run_mcmc.

    Returns
    -------
    n_divergences : int
        Total number of divergent transitions.
    fraction : float
        Fraction of total transitions that diverged.
    """
    mcmc = getattr(mcmc_result, "mcmc", mcmc_result)

    try:
        extra_fields = mcmc.get_extra_fields()
        if "diverging" in extra_fields:
            div = np.asarray(extra_fields["diverging"])
            n_div = int(div.sum())
            total = div.size
            return n_div, n_div / total if total > 0 else 0.0
    except Exception:
        pass

    # Fallback: try to access via last_state
    try:
        info = mcmc.last_state
        if hasattr(info, "diverging"):
            n_div = int(np.asarray(info.diverging).sum())
            return n_div, 0.0  # can't compute fraction without total
    except Exception:
        pass

    return 0, 0.0


def compute_prior_posterior_overlap(
    samples: dict,
    prior_mean: np.ndarray,
    prior_std: float,
    param_name: str = "O",
    n_elements: int = 5,
) -> Dict[str, float]:
    """
    Estimate overlap between prior and posterior for a subset of parameters.

    Uses a simple histogram-based overlap coefficient. High overlap (> 0.9)
    suggests the data is not informative; very low overlap (< 0.05) suggests
    the prior may be misspecified or the model is overfit.

    Parameters
    ----------
    samples : dict
        Posterior samples dict.
    prior_mean : np.ndarray
        Prior mean for the operator (same shape as one sample).
    prior_std : float
        Prior standard deviation (scalar, isotropic).
    param_name : str
        Parameter name.
    n_elements : int
        Number of (randomly chosen) elements to analyze.

    Returns
    -------
    overlap_dict : dict
        Mapping from parameter label to overlap coefficient [0, 1].
    """
    O_samples = _extract_param(samples, param_name)
    n_samples = O_samples.shape[0]
    flat_samples = O_samples.reshape(n_samples, -1)
    flat_prior_mean = prior_mean.ravel()

    n_params = flat_samples.shape[1]
    indices = np.random.choice(n_params, size=min(n_elements, n_params), replace=False)

    if O_samples.ndim == 3:
        rows, cols = O_samples.shape[1], O_samples.shape[2]
        all_labels = [f"{param_name}[{i},{j}]" for i in range(rows) for j in range(cols)]
    else:
        all_labels = [f"{param_name}[{i}]" for i in range(n_params)]

    overlap_dict = {}
    for idx in indices:
        post_samples = flat_samples[:, idx]
        prior_mu = flat_prior_mean[idx]

        # Build histograms over shared range
        lo = min(post_samples.min(), prior_mu - 4 * prior_std)
        hi = max(post_samples.max(), prior_mu + 4 * prior_std)
        bins = np.linspace(lo, hi, 100)

        post_hist, _ = np.histogram(post_samples, bins=bins, density=True)
        prior_hist = _normal_pdf_binned(bins, prior_mu, prior_std)

        # Overlap coefficient: integral of min(p, q)
        bin_width = bins[1] - bins[0]
        overlap = np.sum(np.minimum(post_hist, prior_hist)) * bin_width
        overlap_dict[all_labels[idx]] = float(np.clip(overlap, 0, 1))

    return overlap_dict


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_correlation_matrix(
    corr_matrix: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Posterior Correlation Matrix",
    max_display: int = 30,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "RdBu_r",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the posterior correlation matrix as a heatmap.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Square correlation matrix.
    labels : list of str, optional
        Axis tick labels.
    title : str
        Plot title.
    max_display : int
        Maximum number of parameters to display (subsamples if larger).
    figsize : tuple, optional
        Figure size.
    cmap : str
        Matplotlib colormap.

    Returns
    -------
    fig, ax
    """
    n = corr_matrix.shape[0]
    if n > max_display:
        idx = np.linspace(0, n - 1, max_display, dtype=int)
        corr_sub = corr_matrix[np.ix_(idx, idx)]
        labels_sub = [labels[i] for i in idx] if labels else None
    else:
        corr_sub = corr_matrix
        labels_sub = labels

    if figsize is None:
        size = max(6, min(16, len(corr_sub) * 0.4))
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_sub, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

    if labels_sub and len(labels_sub) <= 30:
        ax.set_xticks(range(len(labels_sub)))
        ax.set_xticklabels(labels_sub, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels_sub)))
        ax.set_yticklabels(labels_sub, fontsize=7)

    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig, ax


def plot_ess(
    ess_dict: Dict[str, float],
    title: str = "Effective Sample Size per Parameter",
    figsize: Tuple[int, int] = (10, 4),
    threshold: float = 100,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar chart of ESS values with a threshold line.

    Parameters
    ----------
    ess_dict : dict
        Mapping from label to ESS.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    threshold : float
        Minimum acceptable ESS (drawn as horizontal line).

    Returns
    -------
    fig, ax
    """
    labels = list(ess_dict.keys())
    values = list(ess_dict.values())

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["tab:red" if v < threshold else "tab:blue" for v in values]
    ax.bar(range(len(values)), values, color=colors, width=0.8)
    ax.axhline(threshold, color="k", linestyle="--", linewidth=1, label=f"Threshold = {threshold}")
    ax.set_ylabel("ESS")
    ax.set_title(title)
    ax.legend()

    if len(labels) <= 30:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
    else:
        ax.set_xlabel("Parameter index")
    fig.tight_layout()
    return fig, ax


def plot_trace(
    samples: dict,
    param_name: str = "O",
    indices: Optional[List[Tuple[int, int]]] = None,
    n_random: int = 6,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Trace plots for selected operator elements.

    Shows the sampled values over iteration, useful for checking mixing
    and stationarity.

    Parameters
    ----------
    samples : dict
        Posterior samples.
    param_name : str
        Parameter name.
    indices : list of (row, col), optional
        Specific operator elements to plot. If None, picks randomly.
    n_random : int
        Number of random elements to plot (if indices is None).
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig, axes
    """
    O_samples = _extract_param(samples, param_name)
    n_samples = O_samples.shape[0]

    if indices is None:
        if O_samples.ndim == 3:
            rows, cols = O_samples.shape[1], O_samples.shape[2]
            all_idx = [(i, j) for i in range(rows) for j in range(cols)]
        else:
            all_idx = [(i,) for i in range(O_samples.shape[1])]
        chosen = np.random.choice(len(all_idx), size=min(n_random, len(all_idx)), replace=False)
        indices = [all_idx[c] for c in chosen]

    n_plots = len(indices)
    if figsize is None:
        figsize = (12, 2.5 * n_plots)

    fig, axes = plt.subplots(n_plots, 2, figsize=figsize, squeeze=False)

    for row_idx, idx in enumerate(indices):
        if len(idx) == 2:
            chain = O_samples[:, idx[0], idx[1]]
            label = f"{param_name}[{idx[0]},{idx[1]}]"
        else:
            chain = O_samples[:, idx[0]]
            label = f"{param_name}[{idx[0]}]"

        # Trace
        axes[row_idx, 0].plot(chain, linewidth=0.5, alpha=0.8)
        axes[row_idx, 0].set_ylabel(label)
        axes[row_idx, 0].set_xlabel("Sample")
        axes[row_idx, 0].set_title(f"Trace: {label}")

        # Histogram
        axes[row_idx, 1].hist(chain, bins=40, density=True, alpha=0.7, edgecolor="white")
        axes[row_idx, 1].set_xlabel(label)
        axes[row_idx, 1].set_ylabel("Density")
        axes[row_idx, 1].set_title(f"Marginal: {label}")

    fig.suptitle(f"Trace & Marginal Plots for {param_name}", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_rank(
    samples_by_chain: list,
    param_name: str = "O",
    n_elements: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Rank histogram plots for chain mixing assessment.

    For well-mixed chains, rank histograms should be approximately uniform.
    Skewed or U-shaped rank plots indicate poor mixing.

    Parameters
    ----------
    samples_by_chain : list of dict
        One sample dict per chain.
    param_name : str
        Parameter name.
    n_elements : int
        Number of random elements to plot.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig, axes
    """
    chains = []
    for chain_dict in samples_by_chain:
        O = _extract_param(chain_dict, param_name)
        chains.append(O.reshape(O.shape[0], -1))

    n_chains = len(chains)
    n_params = chains[0].shape[1]
    indices = np.random.choice(n_params, size=min(n_elements, n_params), replace=False)

    if figsize is None:
        figsize = (12, 3 * len(indices))

    fig, axes = plt.subplots(len(indices), 1, figsize=figsize, squeeze=False)

    for plot_idx, param_idx in enumerate(indices):
        # Pool all chains and compute ranks
        all_vals = np.concatenate([c[:, param_idx] for c in chains])
        ranks = np.argsort(np.argsort(all_vals))

        offset = 0
        for chain_idx, c in enumerate(chains):
            n = c.shape[0]
            chain_ranks = ranks[offset:offset + n]
            axes[plot_idx, 0].hist(
                chain_ranks, bins=20, alpha=0.5,
                label=f"Chain {chain_idx}", density=True
            )
            offset += n

        axes[plot_idx, 0].set_title(f"Rank plot: param {param_idx}")
        axes[plot_idx, 0].legend(fontsize=8)
        axes[plot_idx, 0].set_xlabel("Rank")
        axes[plot_idx, 0].set_ylabel("Density")

    fig.suptitle("Rank Histograms (uniform = good mixing)", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_prior_posterior(
    samples: dict,
    prior_mean: np.ndarray,
    prior_std: float,
    param_name: str = "O",
    n_elements: int = 6,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Overlay prior and posterior densities for selected parameters.

    Helps diagnose: prior too wide (data dominated), prior too narrow
    (prior dominated), or good calibration.

    Parameters
    ----------
    samples : dict
        Posterior samples.
    prior_mean : np.ndarray
        Prior operator mean.
    prior_std : float
        Prior standard deviation (isotropic).
    param_name : str
        Parameter name.
    n_elements : int
        Number of elements to plot.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig, axes
    """
    O_samples = _extract_param(samples, param_name)
    n_samples = O_samples.shape[0]
    flat_samples = O_samples.reshape(n_samples, -1)
    flat_prior = prior_mean.ravel()

    n_params = flat_samples.shape[1]
    indices = np.random.choice(n_params, size=min(n_elements, n_params), replace=False)

    n_plots = len(indices)
    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for plot_idx, param_idx in enumerate(indices):
        r, c = divmod(plot_idx, ncols)
        ax = axes[r, c]

        post = flat_samples[:, param_idx]
        mu_prior = flat_prior[param_idx]

        # Posterior histogram
        ax.hist(post, bins=40, density=True, alpha=0.6, color="steelblue",
                edgecolor="white", label="Posterior")

        # Prior curve
        x_range = np.linspace(mu_prior - 4 * prior_std, mu_prior + 4 * prior_std, 200)
        prior_pdf = _normal_pdf(x_range, mu_prior, prior_std)
        ax.plot(x_range, prior_pdf, "r-", linewidth=2, label="Prior")

        ax.axvline(mu_prior, color="r", linestyle=":", alpha=0.5)
        ax.axvline(post.mean(), color="steelblue", linestyle=":", alpha=0.7)
        ax.set_title(f"Param {param_idx}", fontsize=10)
        ax.legend(fontsize=8)

    # Hide empty subplots
    for idx in range(n_plots, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Prior vs Posterior", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig, axes


# =============================================================================
# Full Diagnostic Runner
# =============================================================================

def run_diagnostics(
    samples: dict,
    param_name: str = "O",
    prior_mean: Optional[np.ndarray] = None,
    prior_std: Optional[float] = None,
    mcmc_result=None,
    samples_by_chain: Optional[list] = None,
    correlation_threshold: float = 0.9,
    ess_threshold: float = 100,
    rhat_threshold: float = 1.01,
    verbose: bool = True,
    plot: bool = True,
    figsize_corr: Optional[Tuple[int, int]] = None,
) -> DiagnosticReport:
    """
    Run a full suite of Bayesian diagnostics and return a report.

    Parameters
    ----------
    samples : dict
        Posterior samples from SVI or MCMC.
    param_name : str
        Name of the operator parameter to diagnose.
    prior_mean : np.ndarray, optional
        Prior mean for prior-posterior comparison.
    prior_std : float, optional
        Prior std for prior-posterior comparison.
    mcmc_result : MCMCResult, optional
        MCMC result object (for divergence checking).
    samples_by_chain : list of dict, optional
        Per-chain samples (for R-hat and rank plots).
    correlation_threshold : float
        Threshold for flagging correlated pairs.
    ess_threshold : float
        Minimum acceptable ESS.
    rhat_threshold : float
        Maximum acceptable R-hat.
    verbose : bool
        Whether to print the summary report.
    plot : bool
        Whether to generate diagnostic plots.
    figsize_corr : tuple, optional
        Figure size for correlation matrix plot.

    Returns
    -------
    DiagnosticReport
        Full diagnostic report.
    """
    report = DiagnosticReport()

    # 1. Posterior correlation
    try:
        corr, labels, high_pairs = compute_posterior_correlation(
            samples, param_name, threshold=correlation_threshold
        )
        report.correlation_matrix = corr
        report.high_correlations = high_pairs
        if len(high_pairs) > 0:
            report.warnings.append(
                f"Found {len(high_pairs)} highly correlated parameter pairs "
                f"(|r| > {correlation_threshold}). Consider reparameterization."
            )
    except Exception as e:
        report.warnings.append(f"Could not compute correlation: {e}")

    # 2. ESS
    try:
        ess = compute_ess(samples, param_name)
        report.ess = ess
        low_count = sum(1 for v in ess.values() if v < ess_threshold)
        if low_count > 0:
            report.warnings.append(
                f"{low_count} parameters have ESS < {ess_threshold}. "
                f"Consider running longer chains or reparameterizing."
            )
    except Exception as e:
        report.warnings.append(f"Could not compute ESS: {e}")

    # 3. R-hat (multi-chain only)
    if samples_by_chain is not None and len(samples_by_chain) > 1:
        try:
            rhat = compute_rhat(samples_by_chain, param_name)
            report.rhat = rhat
            bad_count = sum(1 for v in rhat.values() if v > rhat_threshold)
            if bad_count > 0:
                report.warnings.append(
                    f"{bad_count} parameters have R-hat > {rhat_threshold}. "
                    f"Chains may not have converged."
                )
        except Exception as e:
            report.warnings.append(f"Could not compute R-hat: {e}")

    # 4. Divergences (MCMC only)
    if mcmc_result is not None:
        try:
            n_div, frac = detect_divergences(mcmc_result)
            report.num_divergences = n_div
            report.divergence_fraction = frac
            if n_div > 0:
                report.warnings.append(
                    f"{n_div} divergent transitions ({frac:.1%}). "
                    f"Try increasing target_accept_prob or reparameterizing."
                )
        except Exception as e:
            report.warnings.append(f"Could not detect divergences: {e}")

    # 5. Prior-posterior overlap
    if prior_mean is not None and prior_std is not None:
        try:
            overlap = compute_prior_posterior_overlap(
                samples, prior_mean, prior_std, param_name
            )
            report.prior_posterior_overlap = overlap
            for k, v in overlap.items():
                if v > 0.95:
                    report.warnings.append(
                        f"Prior-posterior overlap for {k} is {v:.0%} — "
                        f"data may not be informative for this parameter."
                    )
                elif v < 0.05:
                    report.warnings.append(
                        f"Prior-posterior overlap for {k} is {v:.0%} — "
                        f"prior may be misspecified (too narrow or wrong location)."
                    )
        except Exception as e:
            report.warnings.append(f"Could not compute prior-posterior overlap: {e}")

    # --- Plots ---
    if plot:
        if report.correlation_matrix is not None:
            plot_correlation_matrix(
                report.correlation_matrix, labels,
                title=f"Posterior Correlation: {param_name}",
                figsize=figsize_corr,
            )
            plt.show()

        if report.ess is not None:
            plot_ess(report.ess, title=f"ESS: {param_name}")
            plt.show()

        plot_trace(samples, param_name)
        plt.show()

        if prior_mean is not None and prior_std is not None:
            plot_prior_posterior(samples, prior_mean, prior_std, param_name)
            plt.show()

        if samples_by_chain is not None and len(samples_by_chain) > 1:
            plot_rank(samples_by_chain, param_name)
            plt.show()

    # --- Summary ---
    if verbose:
        report.summary(verbose=True)

    return report


# =============================================================================
# Private Helpers
# =============================================================================

def _extract_param(samples: dict, param_name: str) -> np.ndarray:
    """Robustly extract a parameter array from a samples dict."""
    # Exact match
    if param_name in samples:
        return np.asarray(samples[param_name])

    # AutoDelta / AutoNormal loc
    auto_loc = f"{param_name}_auto_loc"
    if auto_loc in samples:
        return np.asarray(samples[auto_loc])

    # Prefixed patterns
    for prefix in ["auto_", ""]:
        key = f"{prefix}{param_name}_auto_loc"
        if key in samples:
            return np.asarray(samples[key])

    # Fuzzy: any key containing param_name but not common exclusions
    exclude = {"ode", "constraint", "latent"}
    candidates = [
        k for k in samples
        if param_name in k and not any(ex in k.lower() for ex in exclude)
    ]
    if candidates:
        best = min(candidates, key=len)
        return np.asarray(samples[best])

    raise KeyError(
        f"Cannot find '{param_name}' in samples. "
        f"Available keys: {sorted(samples.keys())}"
    )


def _ess_1d(chain: np.ndarray) -> float:
    """
    Estimate effective sample size for a 1-D chain using the initial
    positive sequence estimator.
    """
    n = len(chain)
    if n < 4:
        return float(n)

    mean = chain.mean()
    var = chain.var(ddof=1)
    if var < 1e-30:
        return float(n)

    # Compute autocorrelations via FFT
    centered = chain - mean
    fft_vals = np.fft.fft(centered, n=2 * n)
    acf = np.fft.ifft(fft_vals * np.conj(fft_vals)).real[:n] / (n * var)

    # Initial positive sequence estimator (Geyer 1992)
    # Sum consecutive pairs of autocorrelations while they stay positive
    tau = 1.0
    for t in range(1, n // 2):
        rho_pair = acf[2 * t - 1] + acf[2 * t]
        if rho_pair < 0:
            break
        tau += 2 * rho_pair

    return max(1.0, n / tau)


def _rhat_1d(chains: List[np.ndarray]) -> float:
    """
    Compute split-Rhat for a single parameter across multiple chains.
    Each chain is split in half before computing.
    """
    # Split each chain in half
    split_chains = []
    for c in chains:
        mid = len(c) // 2
        if mid > 0:
            split_chains.append(c[:mid])
            split_chains.append(c[mid:])

    if len(split_chains) < 2:
        return 1.0

    m = len(split_chains)
    n = min(len(c) for c in split_chains)
    if n < 2:
        return 1.0

    # Truncate to same length
    split_chains = [c[:n] for c in split_chains]

    chain_means = np.array([c.mean() for c in split_chains])
    chain_vars = np.array([c.var(ddof=1) for c in split_chains])

    grand_mean = chain_means.mean()
    B = n * np.var(chain_means, ddof=1)
    W = np.mean(chain_vars)

    if W < 1e-30:
        return 1.0

    var_hat = (1 - 1 / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate normal PDF."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _normal_pdf_binned(bins: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate normal PDF at bin centers."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    return _normal_pdf(centers, mu, sigma)


# =============================================================================
# Stability Diagnostics
# =============================================================================

def _compute_operator_blocks(
    operators: str, num_modes: int, input_dim: int = 0,
) -> Dict[str, Tuple[int, int]]:
    """Return ``{block_name: (col_start, col_end)}`` for an OpInf operator string."""
    col = 0
    blocks = {}
    for ch in operators:
        if ch == "c":
            w = 1
        elif ch == "A":
            w = num_modes
        elif ch == "H":
            w = num_modes * (num_modes + 1) // 2
        elif ch == "G":
            # cubic
            w = num_modes * (num_modes + 1) * (num_modes + 2) // 6
        elif ch == "B":
            w = max(input_dim, 1)
        elif ch == "N":
            w = num_modes * max(input_dim, 1)
        else:
            raise ValueError(f"Unknown operator character '{ch}'")
        blocks[ch] = (col, col + w)
        col += w
    return blocks


@dataclass
class StabilityReport:
    """Container for stability diagnostic results."""

    # MAP / mean operator
    map_stable: Optional[bool] = None
    prior_stable: Optional[bool] = None

    # ELBO convergence
    elbo_converged: Optional[bool] = None
    elbo_final_slope: Optional[float] = None

    # Prior-posterior shift
    shift_norms_per_row: Optional[np.ndarray] = None
    shift_relative_per_block: Optional[Dict[str, float]] = None

    # Eigenvalue analysis (A block)
    prior_A_eigenvalues: Optional[np.ndarray] = None
    posterior_A_eigenvalues: Optional[np.ndarray] = None
    prior_A_max_real: Optional[float] = None
    posterior_A_max_real: Optional[float] = None

    # Perturbation sensitivity
    perturb_stable_fracs: Optional[Dict[str, float]] = None

    # Overall
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        lines.append("=" * 64)
        lines.append("  STABILITY DIAGNOSTIC REPORT")
        lines.append("=" * 64)

        # MAP / prior stability
        lines.append("\n--- Operator Stability ---")
        if self.prior_stable is not None:
            s = "STABLE" if self.prior_stable else "UNSTABLE"
            lines.append(f"  Prior operator:     {s}")
        if self.map_stable is not None:
            s = "STABLE" if self.map_stable else "UNSTABLE"
            lines.append(f"  Posterior MAP/mean: {s}")

        # ELBO
        if self.elbo_converged is not None:
            lines.append("\n--- ELBO Convergence ---")
            s = "converged" if self.elbo_converged else "NOT converged"
            lines.append(f"  Status: {s}  (final slope = {self.elbo_final_slope:.2e})")

        # Prior-posterior shift
        if self.shift_norms_per_row is not None:
            lines.append("\n--- Prior → Posterior Shift ---")
            for i, norm in enumerate(self.shift_norms_per_row):
                lines.append(f"  Mode {i}: ‖ΔO‖ = {norm:.4f}")
        if self.shift_relative_per_block:
            lines.append("  Per-block relative shift (‖Δblock‖/‖prior_block‖):")
            for name, val in self.shift_relative_per_block.items():
                lines.append(f"    {name}: {val:.4f}")

        # Eigenvalue analysis
        if self.prior_A_eigenvalues is not None:
            lines.append("\n--- Linear Operator (A) Eigenvalues ---")
            lines.append(f"  Prior  max Re(λ): {self.prior_A_max_real:+.6f}"
                         f"  {'(unstable!)' if self.prior_A_max_real > 0 else '(stable)'}")
            lines.append(f"  Post.  max Re(λ): {self.posterior_A_max_real:+.6f}"
                         f"  {'(unstable!)' if self.posterior_A_max_real > 0 else '(stable)'}")

        # Perturbation sensitivity
        if self.perturb_stable_fracs is not None:
            lines.append("\n--- Perturbation Sensitivity ---")
            for name, frac in self.perturb_stable_fracs.items():
                bar = "█" * int(frac * 20) + "░" * (20 - int(frac * 20))
                lines.append(f"  Perturb {name:>1}: {bar} {frac:.0%} stable")

        # Warnings
        if self.warnings:
            lines.append("\n--- ⚠  Warnings ---")
            for w in self.warnings:
                lines.append(f"  • {w}")

        text = "\n".join(lines)
        print(text)
        return text


def diagnose_stability(
    posterior_operator: np.ndarray,
    prior_operator: np.ndarray,
    rom,
    snapshots_compressed: np.ndarray,
    time_eval: np.ndarray,
    operators: str = "cAH",
    input_func: Optional[Callable] = None,
    input_dim: int = 0,
    losses: Optional[List[float]] = None,
    perturb_scale: float = 0.01,
    perturb_trials: int = 100,
    plot: bool = True,
    verbose: bool = True,
) -> StabilityReport:
    """
    Run stability diagnostics on a posterior operator.

    Parameters
    ----------
    posterior_operator : array (r, d)
        Posterior MAP/mean operator matrix.
    prior_operator : array (r, d)
        Prior operator matrix.
    rom : opinf ROM
        ROM object (must have ``model._extract_operators`` and ``model.predict``).
    snapshots_compressed : array (r, n_time)
        Compressed training snapshots (for initial condition).
    time_eval : array (n_time,)
        Time points for ROM integration.
    operators : str
        OpInf operator string (e.g. ``"cAH"``, ``"cAHBN"``).
    input_func : callable, optional
        Input function for systems with inputs.
    input_dim : int
        Input dimension (0 if no inputs).
    losses : list of float, optional
        SVI loss history for convergence check.
    perturb_scale : float
        Scale of Gaussian perturbation relative to each operator entry.
    perturb_trials : int
        Number of perturbation trials per block.
    plot : bool
        Whether to generate diagnostic plots.
    verbose : bool
        Whether to print the summary report.

    Returns
    -------
    StabilityReport
    """
    report = StabilityReport()
    num_modes = posterior_operator.shape[0]
    blocks = _compute_operator_blocks(operators, num_modes, input_dim)

    # ------------------------------------------------------------------
    # 1. MAP / prior stability
    # ------------------------------------------------------------------
    def _is_stable(O):
        rom.model._extract_operators(np.array(O))
        try:
            if input_func is not None:
                rom.model.predict(
                    state0=snapshots_compressed[:, 0],
                    t=time_eval,
                    input_func=input_func,
                )
            else:
                rom.model.predict(
                    state0=snapshots_compressed[:, 0],
                    t=time_eval,
                )
            return rom.model.predict_result_.y.shape[1] >= time_eval.size
        except Exception:
            return False

    report.prior_stable = _is_stable(prior_operator)
    report.map_stable = _is_stable(posterior_operator)

    if not report.prior_stable:
        report.warnings.append(
            "Prior operator is UNSTABLE — posterior has nowhere safe to start."
        )
    if not report.map_stable:
        report.warnings.append(
            "Posterior MAP/mean is UNSTABLE — SVI may not have converged "
            "or GAMMA/GAMMA2 need tuning."
        )

    # ------------------------------------------------------------------
    # 2. ELBO convergence
    # ------------------------------------------------------------------
    if losses is not None and len(losses) > 200:
        tail = np.array(losses[-200:])
        x = np.arange(len(tail), dtype=float)
        slope = np.polyfit(x, tail, 1)[0]
        report.elbo_final_slope = float(slope)
        report.elbo_converged = abs(slope) < 0.1 * np.std(tail)
        if not report.elbo_converged:
            report.warnings.append(
                f"ELBO may not have converged (final slope={slope:.2e})."
            )

    # ------------------------------------------------------------------
    # 3. Prior → posterior shift
    # ------------------------------------------------------------------
    diff = posterior_operator - prior_operator
    report.shift_norms_per_row = np.linalg.norm(diff, axis=1)

    rel = {}
    for name, (c0, c1) in blocks.items():
        prior_block_norm = np.linalg.norm(prior_operator[:, c0:c1])
        diff_block_norm = np.linalg.norm(diff[:, c0:c1])
        rel[name] = (
            diff_block_norm / prior_block_norm
            if prior_block_norm > 1e-12
            else float("inf")
        )
    report.shift_relative_per_block = rel

    biggest = max(rel, key=rel.get)
    if rel[biggest] > 1.0:
        report.warnings.append(
            f"Block '{biggest}' shifted more than 100% from prior "
            f"(relative shift = {rel[biggest]:.2f})."
        )

    # ------------------------------------------------------------------
    # 4. Eigenvalue analysis of A block
    # ------------------------------------------------------------------
    if "A" in blocks:
        c0, c1 = blocks["A"]
        A_prior = prior_operator[:, c0:c1]
        A_post = posterior_operator[:, c0:c1]
        report.prior_A_eigenvalues = np.linalg.eigvals(A_prior)
        report.posterior_A_eigenvalues = np.linalg.eigvals(A_post)
        report.prior_A_max_real = float(np.max(report.prior_A_eigenvalues.real))
        report.posterior_A_max_real = float(np.max(report.posterior_A_eigenvalues.real))

        if report.posterior_A_max_real > 0:
            report.warnings.append(
                f"Posterior A has eigenvalue with Re(λ)={report.posterior_A_max_real:+.4f} "
                f"— linear dynamics are unstable."
            )

    # ------------------------------------------------------------------
    # 5. Perturbation sensitivity per block
    # ------------------------------------------------------------------
    perturb_fracs = {}
    rng = np.random.RandomState(42)
    for name, (c0, c1) in blocks.items():
        stable_count = 0
        for _ in range(perturb_trials):
            O_pert = posterior_operator.copy()
            block = O_pert[:, c0:c1]
            scale = perturb_scale * (np.abs(block) + 1e-8)
            O_pert[:, c0:c1] += rng.randn(*block.shape) * scale
            if _is_stable(O_pert):
                stable_count += 1
        perturb_fracs[name] = stable_count / perturb_trials
    report.perturb_stable_fracs = perturb_fracs

    fragile = [n for n, f in perturb_fracs.items() if f < 0.5]
    if fragile:
        report.warnings.append(
            f"Blocks {fragile} are fragile (<50% stable under {perturb_scale:.0%} perturbation)."
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if plot:
        _plot_stability_diagnostics(
            report, posterior_operator, prior_operator, blocks,
            operators, losses,
        )

    if verbose:
        report.summary()

    return report


def _plot_stability_diagnostics(
    report: StabilityReport,
    posterior_operator: np.ndarray,
    prior_operator: np.ndarray,
    blocks: Dict[str, Tuple[int, int]],
    operators: str,
    losses: Optional[List[float]],
):
    """Generate stability diagnostic plots."""
    num_modes = posterior_operator.shape[0]
    has_eig = report.prior_A_eigenvalues is not None
    has_losses = losses is not None and len(losses) > 0
    ncols = 2 + int(has_eig) + int(has_losses)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    axes = np.atleast_1d(axes)
    ax_idx = 0

    # --- ELBO loss curve ---
    if has_losses:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(losses, color='tab:blue', lw=0.5)
        ax.set_xlabel("SVI step")
        ax.set_ylabel("ELBO loss")
        ax.set_title("ELBO Convergence")
        ax.set_yscale("symlog")

    # --- Prior-posterior shift heatmap ---
    ax = axes[ax_idx]; ax_idx += 1
    diff = posterior_operator - prior_operator
    vmax = np.max(np.abs(diff))
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Operator column")
    ax.set_ylabel("Mode")
    ax.set_yticks(range(num_modes))
    ax.set_yticklabels([f"Mode {i+1}" for i in range(num_modes)])
    ax.set_title("O_posterior − O_prior")
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Block separators
    for name, (c0, c1) in blocks.items():
        if c0 > 0:
            ax.axvline(c0 - 0.5, color="k", lw=0.8, ls="--")
        ax.text((c0 + c1) / 2 - 0.5, -0.7, name, ha="center", fontsize=9,
                fontweight="bold")

    # --- Eigenvalue plot ---
    if has_eig:
        ax = axes[ax_idx]; ax_idx += 1
        eig_pr = report.prior_A_eigenvalues
        eig_po = report.posterior_A_eigenvalues
        ax.scatter(eig_pr.real, eig_pr.imag, marker="o", s=80,
                   edgecolors="tab:blue", facecolors="none", label="Prior A", zorder=5)
        ax.scatter(eig_po.real, eig_po.imag, marker="x", s=80,
                   color="tab:red", label="Posterior A", zorder=5)
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title("A Eigenvalues")
        ax.legend(fontsize=8)

    # --- Perturbation bar chart ---
    if report.perturb_stable_fracs is not None:
        ax = axes[ax_idx]; ax_idx += 1
        names = list(report.perturb_stable_fracs.keys())
        fracs = [report.perturb_stable_fracs[n] for n in names]
        colors = ["tab:green" if f >= 0.8 else "tab:orange" if f >= 0.5 else "tab:red"
                  for f in fracs]
        ax.barh(names, fracs, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction stable")
        ax.set_title("Perturbation Sensitivity")
        ax.axvline(0.5, color="k", lw=0.5, ls="--")

    fig.suptitle("Stability Diagnostics", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
