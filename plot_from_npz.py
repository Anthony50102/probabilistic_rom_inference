#!/usr/bin/env python3
"""Plot ROM trajectories from saved .npz result files.

Usage:
    python plot_from_npz.py <path/to/result.npz> [save_dir]

Generates ROM trajectory plots from the arrays saved by 04_unified.py or
04_unified.py.  Single-IC experiments (Euler, Burgers, Tumor)
produce a modes-only grid; multi-IC experiments (Heat) produce an
ICs × modes grid matching the paper figure layout.

If save_dir is not given, saves next to the .npz file.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def _rom_trajectory_grid(all_rom_solves, t_pred, all_true_comp, t_full,
                         all_snaps_comp, all_t_samp, training_span,
                         num_modes, labels, title, save_path):
    """ICs × modes ROM trajectory grid (paper format).

    Parameters
    ----------
    all_rom_solves : list of ndarray (n_stable, num_modes, T_pred) per IC
    all_true_comp  : list of ndarray (num_modes, T_full) per IC
    all_snaps_comp : list of ndarray (num_modes, T_samp) per IC
    all_t_samp     : list of ndarray (T_samp,) per IC
    labels         : list of str, one per IC row
    """
    n_ics = len(all_rom_solves)
    has_any = any(len(s) > 0 for s in all_rom_solves)
    if not has_any:
        print(f"  ⚠ No stable solves — skipping {save_path}")
        return

    fig, axes = plt.subplots(
        n_ics, num_modes,
        figsize=(4 * num_modes, 2.5 * n_ics),
        sharex=True, squeeze=False,
    )

    for row in range(n_ics):
        rom_arr = np.asarray(all_rom_solves[row])
        n_stable = rom_arr.shape[0] if rom_arr.ndim == 3 else 0

        true_interp = interp1d(t_full, all_true_comp[row],
                               kind='cubic', fill_value='extrapolate')
        true_at_pred = true_interp(t_pred)

        for col in range(num_modes):
            ax = axes[row, col]
            ax.axvspan(float(training_span[0]), float(training_span[1]),
                       color='gray', alpha=0.10, zorder=0)

            ax.plot(t_pred, true_at_pred[col], color='tab:gray', lw=2,
                    label='True' if (row == 0 and col == 0) else None)
            ax.plot(all_t_samp[row], all_snaps_comp[row][col],
                    'k*', ms=4, zorder=5,
                    label='Data' if (row == 0 and col == 0) else None)

            if n_stable > 0:
                ax.plot(t_pred,
                        np.median(rom_arr[:, col, :], axis=0),
                        color='tab:purple', ls='--', lw=2, alpha=0.9,
                        label='Median' if (row == 0 and col == 0) else None)
                ax.fill_between(
                    t_pred,
                    np.percentile(rom_arr[:, col, :], 5, axis=0),
                    np.percentile(rom_arr[:, col, :], 95, axis=0),
                    color='tab:purple', alpha=0.15,
                    label='90% CI' if (row == 0 and col == 0) else None)

            ax.axvline(float(training_span[1]), color='k', ls=':', lw=0.8,
                       alpha=0.5)

            yvals = true_at_pred[col]
            ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
            pad = max(abs(ymax - ymin) * 0.3, 1e-6)
            ax.set_ylim(ymin - pad, ymax + pad)

            if row == 0:
                ax.set_title(f'Mode {col + 1}')
            if col == 0:
                ax.set_ylabel(labels[row], fontsize=8)
            if row == n_ics - 1:
                ax.set_xlabel('Time')

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc='upper center',
                   ncol=len(handles), fontsize=9,
                   bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {save_path}")
    plt.close(fig)


def _loss_plot(losses, title, save_path):
    """Two-panel loss convergence plot (full + last 50%)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(losses, lw=0.8, color='tab:blue')
    ax[0].set_xlabel('SVI Iteration')
    ax[0].set_ylabel('ELBO Loss')
    ax[0].set_title('Loss Convergence')
    ax[0].grid(True, alpha=0.3)
    half = len(losses) // 2
    ax[1].plot(range(half, len(losses)), losses[half:],
               lw=0.8, color='tab:blue')
    ax[1].set_xlabel('SVI Iteration')
    ax[1].set_ylabel('ELBO Loss')
    ax[1].set_title('Loss (last 50%)')
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {save_path}")
    plt.close(fig)


def _rom_notebook_plot(rom_solves, t_pred, true_comp, t_full, snaps_comp,
                       t_samp, training_span, num_modes, n_stable, n_total,
                       title, save_path):
    """Single-panel full-span ROM trajectory (notebook style)."""
    rom_arr = np.asarray(rom_solves)
    if rom_arr.ndim < 3 or rom_arr.shape[0] == 0:
        return
    rom_med = np.median(rom_arr, axis=0)
    rom_q05 = np.percentile(rom_arr, 5, axis=0)
    rom_q95 = np.percentile(rom_arr, 95, axis=0)

    true_interp = interp1d(t_full, true_comp, kind='cubic',
                           fill_value='extrapolate')
    true_at_pred = true_interp(t_pred)
    train_end = float(training_span[1])

    fig, ax = plt.subplots(num_modes, 1, figsize=(10, 2.5 * num_modes),
                           sharex=True)
    if num_modes == 1:
        ax = [ax]
    for i in range(num_modes):
        ax[i].axvspan(float(training_span[0]), train_end, color='gray',
                      alpha=0.10, zorder=0)
        ax[i].plot(t_pred, true_at_pred[i], color='tab:gray', lw=2,
                   label='True solution')
        ax[i].plot(t_samp, snaps_comp[i], 'k*', ms=5,
                   label='Training data', zorder=5)
        ax[i].plot(t_pred, rom_med[i], color='tab:purple', linestyle='--',
                   alpha=0.9, lw=2, label='ROM median')
        ax[i].fill_between(t_pred, rom_q05[i], rom_q95[i],
                           color='tab:purple', alpha=0.15, label='ROM 5–95%')
        ax[i].axvline(train_end, color='k', ls=':', lw=0.8, alpha=0.5)
        ax[i].set_ylabel(f'Mode {i+1}')
        yvals = true_at_pred[i]
        ymin, ymax = np.nanmin(yvals), np.nanmax(yvals)
        pad = max(abs(ymax - ymin) * 0.3, 1e-6)
        ax[i].set_ylim(ymin - pad, ymax + pad)
    ax[-1].set_xlabel('Time')
    handles, labels = ax[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center',
                   ncol=len(handles), fontsize=10, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {save_path}")
    plt.close(fig)


def _operator_traces_plot(O_samples, title, save_path, n_random=6):
    """Trace plots for random operator entries."""
    n_post, num_modes, m = O_samples.shape
    rng = np.random.RandomState(42)
    n_entries = min(n_random, num_modes * m)
    indices = rng.choice(num_modes * m, size=n_entries, replace=False)

    fig, axes = plt.subplots(n_entries, 1, figsize=(10, 2 * n_entries),
                             sharex=True)
    if n_entries == 1:
        axes = [axes]
    for k, idx in enumerate(indices):
        i, j = divmod(idx, m)
        vals = O_samples[:, i, j]
        axes[k].plot(vals, lw=0.5, color='tab:blue')
        axes[k].set_ylabel(f'O[{i},{j}]', fontsize=8)
        axes[k].axhline(np.mean(vals), color='tab:red', ls='--', lw=1,
                        alpha=0.7)
    axes[-1].set_xlabel('Sample index')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {save_path}")
    plt.close(fig)


class _MinimalBasis:
    """Lightweight basis for compress/decompress using saved POD vectors."""
    def __init__(self, entries, shift=None, physical_dim=None):
        self.entries = entries  # (n_dof, r)
        self.shift = shift
        self.physical_dim = physical_dim

    def compress(self, states):
        if self.shift is not None and self.entries.shape[0] == 2 * states.shape[0]:
            lifted = np.concatenate((states, states**2), axis=0)
            lifted = lifted - self.shift[:, None]
            return self.entries.T @ lifted
        return self.entries.T @ states

    def decompress(self, compressed):
        states = self.entries @ compressed
        if self.shift is not None:
            states = states + self.shift[:, None]
            if self.physical_dim is not None and states.shape[0] == 2 * self.physical_dim:
                return np.split(states, 2, axis=0)[0]
        return states


def _full_order_error_plot(rom_solves, basis_entries, true_states, t_full,
                           t_pred, training_span, title, save_path,
                           basis_shift=None):
    """Full-order prediction error plot (ROM vs projection error)."""
    rom_arr = np.asarray(rom_solves)
    if rom_arr.ndim < 3 or rom_arr.shape[0] == 0:
        return

    basis = _MinimalBasis(
        basis_entries,
        shift=basis_shift,
        physical_dim=true_states.shape[0],
    )
    true_interp = interp1d(t_full, true_states, axis=1,
                           kind='linear', fill_value='extrapolate')
    true_at_eval = true_interp(t_pred)

    # Projection error (basis limit)
    true_proj = basis.decompress(basis.compress(true_at_eval))
    norm_truth = np.maximum(np.linalg.norm(true_at_eval, axis=0), 1e-10)
    proj_err = np.linalg.norm(true_at_eval - true_proj, axis=0) / norm_truth

    # ROM errors
    rom_errors = []
    for i in range(rom_arr.shape[0]):
        rom_full = basis.decompress(rom_arr[i])
        rom_errors.append(
            np.linalg.norm(true_at_eval - rom_full, axis=0) / norm_truth)
    rom_errors = np.array(rom_errors)

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1]})
    ts = float(training_span[1])

    # ROM error
    axes[0].axvspan(float(training_span[0]), ts, color='gray', alpha=0.10)
    axes[0].plot(t_pred, np.median(rom_errors, axis=0), color='tab:purple',
                 ls='--', lw=2, label='ROM error (median)')
    axes[0].fill_between(t_pred,
                         np.percentile(rom_errors, 5, axis=0),
                         np.percentile(rom_errors, 95, axis=0),
                         color='tab:purple', alpha=0.15, label='ROM error (5–95%)')
    axes[0].set_ylabel('Relative Error')
    axes[0].set_title('ROM Prediction Error')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].set_yscale('log')

    # Projection error
    axes[1].axvspan(float(training_span[0]), ts, color='gray', alpha=0.10)
    axes[1].plot(t_pred, proj_err, 'k--', lw=2,
                 label='Projection error (basis limit)')
    axes[1].set_ylabel('Relative Error')
    axes[1].set_title('Projection Error (Basis Limit)')
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].set_yscale('log')

    # Difference
    axes[2].axvspan(float(training_span[0]), ts, color='gray', alpha=0.10)
    diff = np.maximum(np.median(rom_errors, axis=0) - proj_err, 1e-16)
    axes[2].plot(t_pred, diff, 'tab:purple', lw=2,
                 label='ROM error − projection error')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Relative Error')
    axes[2].set_title('Excess ROM Error')
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].set_yscale('log')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  📊 Saved: {save_path}")
    plt.close(fig)


def plot_from_npz(npz_path, save_dir=None):
    """Load .npz and generate plots."""
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.keys())

    if save_dir is None:
        save_dir = os.path.dirname(npz_path)
    os.makedirs(save_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(npz_path))[0]
    schema_name = os.path.basename(os.path.dirname(npz_path))
    if base == "04_unified":
        base = f"04_{schema_name}"
    t_pred = d['t_pred']
    num_modes = int(d['num_modes'])
    training_span = d['training_span']

    has_single_ic = 'true_comp' in keys and 'rom_solves' in keys
    has_multi_ic = 'n_ics' in keys and 'true_comp_0' in keys

    if not has_single_ic and not has_multi_ic:
        print(f"  ⚠ {npz_path} missing truth data (run with updated save code)")
        return

    # Single-IC: wrap in lists and use the same grid plotter (1 row)
    if has_single_ic:
        _rom_trajectory_grid(
            all_rom_solves=[d['rom_solves']],
            t_pred=t_pred, t_full=d['t_full'],
            all_true_comp=[d['true_comp']],
            all_snaps_comp=[d['snaps_comp']],
            all_t_samp=[d['t_samp']],
            training_span=training_span, num_modes=num_modes,
            labels=[base],
            title=f'Bayesian OpInf — {base}',
            save_path=os.path.join(save_dir, f'{base}_rom_trajectories.png'),
        )

    # Multi-IC: collect per-IC arrays into lists
    elif has_multi_ic:
        n_ics = int(d['n_ics'])
        t_full = d['t_full'] if 't_full' in keys else t_pred
        all_rom, all_true, all_snaps, all_ts, labels = [], [], [], [], []
        for ic in range(n_ics):
            rk = f'rom_solves_{ic}'
            if rk not in keys:
                continue
            all_rom.append(d[rk])
            all_true.append(d[f'true_comp_{ic}'])
            all_snaps.append(d[f'snaps_comp_{ic}'])
            all_ts.append(d[f't_samp_{ic}'])
            try:
                labels.append(str(d['eval_labels'][ic]))
            except Exception:
                labels.append(f'IC {ic}')

        _rom_trajectory_grid(
            all_rom_solves=all_rom, t_pred=t_pred, t_full=t_full,
            all_true_comp=all_true, all_snaps_comp=all_snaps,
            all_t_samp=all_ts, training_span=training_span,
            num_modes=num_modes, labels=labels,
            title=f'Bayesian OpInf — {base}',
            save_path=os.path.join(save_dir, f'{base}_rom_trajectories.png'),
        )

    # Loss convergence plot
    if 'losses' in keys:
        _loss_plot(d['losses'], base,
                   os.path.join(save_dir, f'{base}_loss.png'))

    # ROM notebook plot (single-panel full-span, single-IC only)
    if has_single_ic:
        rom_arr = np.asarray(d['rom_solves'])
        n_stable = rom_arr.shape[0] if rom_arr.ndim == 3 else 0
        n_total = int(d.get('n_total', n_stable))
        _rom_notebook_plot(
            rom_solves=d['rom_solves'], t_pred=t_pred,
            true_comp=d['true_comp'], t_full=d['t_full'],
            snaps_comp=d['snaps_comp'], t_samp=d['t_samp'],
            training_span=training_span, num_modes=num_modes,
            n_stable=n_stable, n_total=n_total,
            title=f'Bayesian OpInf — {base}',
            save_path=os.path.join(save_dir, f'{base}_rom_notebook.png'),
        )

    # Operator traces plot
    if 'O_samples' in keys:
        _operator_traces_plot(
            O_samples=d['O_samples'],
            title=f'Operator Traces — {base}',
            save_path=os.path.join(save_dir, f'{base}_operator_traces.png'),
        )

    # Full-order error plot (single-IC, needs basis + full-order truth)
    if has_single_ic and 'basis_entries' in keys and 'true_states' in keys:
        _full_order_error_plot(
            rom_solves=d['rom_solves'],
            basis_entries=d['basis_entries'],
            true_states=d['true_states'],
            t_full=d['t_full'], t_pred=t_pred,
            training_span=training_span,
            title=f'Full-Order Error — {base}',
            save_path=os.path.join(save_dir, f'{base}_full_order_error.png'),
            basis_shift=d['basis_shift'] if 'basis_shift' in keys else None,
        )
    elif has_multi_ic and 'basis_entries' in keys and 'true_states_0' in keys:
        _full_order_error_plot(
            rom_solves=d['rom_solves_0'],
            basis_entries=d['basis_entries'],
            true_states=d['true_states_0'],
            t_full=d['t_full'] if 't_full' in keys else t_pred,
            t_pred=t_pred,
            training_span=training_span,
            title=f'Full-Order Error (IC 0) — {base}',
            save_path=os.path.join(save_dir, f'{base}_full_order_error.png'),
            basis_shift=d['basis_shift'] if 'basis_shift' in keys else None,
        )

    # Print summary metrics
    for k in ['train_error', 'pred_error', 'stability_pct', 'ci_coverage']:
        if k in keys:
            print(f"  {k}: {float(d[k]):.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    npz_path = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_from_npz(npz_path, save_dir)
