#!/usr/bin/env python
"""Generate full-order model data using TumorTwin's reaction-diffusion solver.

Runs a pre-treatment (no chemo/radiation) tumor growth simulation using
real TNBC patient initial conditions derived from MRI ADC imaging data.
Saves snapshot data as NPZ for subsequent ROM learning.

The reaction-diffusion PDE is:
    du/dt = d ∇²u + k u (1 - u/θ)

where:
    u = tumor cell density (cellularity), normalized to [0, 1]
    d = diffusion coefficient (mm²/day)
    k = proliferation rate (1/day)
    θ = carrying capacity

Usage:
    conda run -n prob_rom python generate_fom_data.py
    conda run -n prob_rom python generate_fom_data.py --patient TNBC_demo_002
    conda run -n prob_rom python generate_fom_data.py --days 120 --k 0.035
"""

import os
import sys
import time
import argparse
import numpy as np

# TumorTwin imports
TUMORTWIN_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'TumorTwin'
)
sys.path.insert(0, os.path.abspath(TUMORTWIN_DIR))

import torch
from pathlib import Path
from datetime import timedelta

from tumortwin.types import TNBCPatientData, CropSettings, CropTarget
from tumortwin.preprocessing import ADC_to_cellularity
from tumortwin.models import ReactionDiffusion3D
from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.utils import daterange


def generate_fom_snapshots(
    patient_id="TNBC_demo_001",
    total_days=90.0,
    dt_save=0.5,
    dt_solve=0.5,
    k=0.025,
    d=0.05,
    theta=1.0,
):
    """Run TumorTwin forward solve and return snapshot data.

    Parameters
    ----------
    patient_id : str
        TNBC demo patient identifier.
    total_days : float
        Total simulation time in days.
    dt_save : float
        Interval between saved snapshots (days).
    dt_solve : float
        ODE solver internal step size (days).
    k, d, theta : float
        Reaction-diffusion parameters.

    Returns
    -------
    dict with keys: snapshots, times_days, grid_shape, breast_mask,
        tumor_mask_t0, spacing, patient_id, k, d, theta, solve_time
    """
    data_dir = Path(os.path.abspath(TUMORTWIN_DIR)) / 'input_files' / patient_id

    patient_data = TNBCPatientData.from_file(
        data_dir / f'{patient_id}.json',
        image_dir=data_dir,
        crop_settings=CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10),
    )

    # Initial condition: ADC MRI → cellularity map
    u0_img = ADC_to_cellularity(
        patient_data.visits[0].adc_image,
        patient_data.visits[0].roi_enhance_image,
    )
    u0 = torch.from_numpy(u0_img.array).float()

    # Model: pure growth, NO treatment
    model = ReactionDiffusion3D(
        k=torch.tensor(k, requires_grad=False),
        d=torch.tensor(d, requires_grad=False),
        theta=torch.tensor(theta, requires_grad=False),
        patient_data=patient_data,
        initial_time=patient_data.visits[0].time,
        chemotherapy_specifications=None,
        radiotherapy_specification=None,
        require_grad=False,
    )

    solver = TorchDiffEqSolver(
        model,
        TorchDiffEqSolverOptions(
            step_size=timedelta(days=dt_solve),
            method='rk4',
            device=torch.device('cpu'),
            use_adjoint=False,
        ),
    )

    # Generate timepoints
    t0 = patient_data.visits[0].time
    timepoints = daterange(t0, t0 + timedelta(days=total_days), timedelta(days=dt_save))

    print(f"  Patient: {patient_id}")
    print(f"  Grid shape: {tuple(u0.shape)}")
    print(f"  Total DOFs: {u0.numel():,}")
    print(f"  Active tumor voxels (t=0): {(u0 > 0).sum().item():,}")
    print(f"  Breast mask voxels: {(patient_data.breastmask_image.array > 0).sum():,}")
    print(f"  Parameters: k={k}, d={d}, θ={theta}")
    print(f"  Simulation: {total_days} days, {len(timepoints)} snapshots")
    print(f"  Solving...")

    t_start = time.time()
    times, solutions = solver.solve(timepoints=timepoints, u_initial=u0)
    solve_time = time.time() - t_start
    print(f"  Done in {solve_time:.1f}s")

    # Convert to numpy snapshot matrix: (n_dof, n_time)
    # TumorTwin solver returns times as torch.Tensor of relative days
    # and solutions as a torch.Tensor or list of tensors
    grid_shape = tuple(u0.shape)
    n_dof = u0.numel()

    times_tensor = times if isinstance(times, torch.Tensor) else torch.stack(times)
    times_days = times_tensor.detach().cpu().numpy().astype(np.float64)

    if isinstance(solutions, torch.Tensor) and solutions.dim() > 3:
        # Stacked tensor: (n_time, nx, ny, nz)
        n_time = solutions.shape[0]
        snapshots = solutions.detach().cpu().numpy().reshape(n_time, -1).T.astype(np.float32)
    else:
        # List of tensors
        sol_list = solutions if isinstance(solutions, list) else [solutions[i] for i in range(len(solutions))]
        n_time = len(sol_list)
        snapshots = np.zeros((n_dof, n_time), dtype=np.float32)
        for i, sol in enumerate(sol_list):
            snapshots[:, i] = sol.detach().cpu().numpy().ravel()

    return dict(
        snapshots=snapshots,
        times_days=times_days,
        grid_shape=np.array(grid_shape),
        breast_mask=patient_data.breastmask_image.array.astype(bool),
        tumor_mask_t0=(u0_img.array > 0),
        spacing=np.array([
            float(u0_img.spacing.x),
            float(u0_img.spacing.y),
            float(u0_img.spacing.z),
        ]),
        patient_id=patient_id,
        k=k,
        d=d,
        theta=theta,
        solve_time=solve_time,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate TumorTwin FOM snapshot data for ROM learning."
    )
    parser.add_argument('--patient', default='TNBC_demo_001',
                        help='Patient ID (default: TNBC_demo_001)')
    parser.add_argument('--days', type=float, default=90.0,
                        help='Total simulation days (default: 90)')
    parser.add_argument('--dt-save', type=float, default=0.5,
                        help='Snapshot save interval in days (default: 0.5)')
    parser.add_argument('--dt-solve', type=float, default=0.5,
                        help='Solver step size in days (default: 0.5)')
    parser.add_argument('--k', type=float, default=0.025,
                        help='Proliferation rate (default: 0.025)')
    parser.add_argument('--d', type=float, default=0.05,
                        help='Diffusion coefficient (default: 0.05)')
    parser.add_argument('--theta', type=float, default=1.0,
                        help='Carrying capacity (default: 1.0)')
    parser.add_argument('--output', default=None,
                        help='Output path (default: data/<patient>_fom.npz)')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"TumorTwin FOM Data Generation")
    print(f"{'='*60}")

    result = generate_fom_snapshots(
        patient_id=args.patient,
        total_days=args.days,
        dt_save=args.dt_save,
        dt_solve=args.dt_solve,
        k=args.k,
        d=args.d,
        theta=args.theta,
    )

    # Save
    out_path = args.output or os.path.join(
        os.path.dirname(__file__), 'data', f'{args.patient}_fom.npz'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(out_path, **result)

    file_size = os.path.getsize(out_path) / 1e6
    snaps = result['snapshots']
    print(f"\n{'='*60}")
    print(f"Saved: {out_path} ({file_size:.1f} MB)")
    print(f"  Snapshot matrix: {snaps.shape} ({snaps.dtype})")
    print(f"  Time span: {result['times_days'][0]:.1f} – {result['times_days'][-1]:.1f} days")
    print(f"  Tumor volume (t=0): {snaps[:, 0].sum():.0f}")
    print(f"  Tumor volume (t=end): {snaps[:, -1].sum():.0f}")
    print(f"  Max cellularity: {snaps.max():.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
