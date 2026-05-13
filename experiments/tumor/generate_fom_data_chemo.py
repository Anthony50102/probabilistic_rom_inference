#!/usr/bin/env python
"""Generate FOM tumor growth data WITH chemotherapy.

Same TumorTwin reaction-diffusion solver as `generate_fom_data.py`, but with
TNBC chemotherapy (taxol) applied per the TNBC_demo_001 protocol:

    du/dt = d ∇²u + k u (1 − u/θ) − sensitivity · α(t) · u

where α(t) is the smooth exponential-decay drug concentration (sum of
exponentially decaying pulses at the dose times).

Defaults (matching `tutorials/TNBC_Demo.ipynb`):
    k = 0.025, d = 0.05, θ = 1.0, sensitivity = 0.2, decay_rate = 0.7
    schedule = 12 weekly doses (taxol) from patient JSON
    dose_scale = 1.0    (multiplies every dose; for ablation: 0.5, 1.5)

The output NPZ stores both the snapshots and the chemotherapy schedule
(in days-since-IC) so the ROM training scripts can rebuild α(t) without
re-loading the patient JSON.

Usage:
    python generate_fom_data_chemo.py
    python generate_fom_data_chemo.py --dose-scale 0.5
    python generate_fom_data_chemo.py --k 0.05 --d 0.025  # aggressive growth
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch

TUMORTWIN_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'TumorTwin'
)
sys.path.insert(0, os.path.abspath(TUMORTWIN_DIR))

from tumortwin.types import (
    TNBCPatientData, CropSettings, CropTarget, ChemotherapySpecification,
)
from tumortwin.preprocessing import ADC_to_cellularity
from tumortwin.models import ReactionDiffusion3D
from tumortwin.solvers import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.utils import daterange, days_since_first


def generate_fom_chemo(
    patient_id="TNBC_demo_001",
    total_days=120.0,
    dt_save=0.5,
    dt_solve=0.5,
    k=0.025,
    d=0.05,
    theta=1.0,
    sensitivity=0.2,
    decay_rate=0.7,
    dose_scale=1.0,
    dose_days_override=None,
):
    """Run TumorTwin forward solve with chemo and return snapshots + schedule.

    Returns a dict ready to save as NPZ.
    """
    data_dir = Path(os.path.abspath(TUMORTWIN_DIR)) / 'input_files' / patient_id
    patient_data = TNBCPatientData.from_file(
        data_dir / f'{patient_id}.json',
        image_dir=data_dir,
        crop_settings=CropSettings(crop_to=CropTarget.ROI_ENHANCE, padding=10),
    )

    u0_img = ADC_to_cellularity(
        patient_data.visits[0].adc_image,
        patient_data.visits[0].roi_enhance_image,
    )
    u0 = torch.from_numpy(u0_img.array).float()

    t0 = patient_data.visits[0].time
    if dose_days_override is not None:
        # Build a custom schedule from days-since-t0; take dose magnitudes
        # from the patient JSON in the same order (cycled if shorter).
        json_amts = [float(c.dose) for c in patient_data.chemotherapy]
        if len(json_amts) == 0:
            json_amts = [1.0]
        dose_times = [t0 + timedelta(days=float(dd)) for dd in dose_days_override]
        dose_amts = [
            json_amts[i % len(json_amts)]
            for i in range(len(dose_times))
        ]
    else:
        dose_times = [c.time for c in patient_data.chemotherapy]
        dose_amts = [float(c.dose) for c in patient_data.chemotherapy]
    dose_days = np.array(
        [days_since_first(dt, t0) for dt in dose_times], dtype=np.float64
    )

    # Apply dose_scale via sensitivity rather than doses: TumorTwin's
    # ChemotherapySpecification pydantic validator normalizes `doses` to max=1
    # inside the spec, which would silently kill any scaling we put on doses.
    # The chemo cell death rate is `sensitivity · Σ dose_i · exp(-decay·Δt)`,
    # so scaling sensitivity is mathematically identical to scaling every
    # dose, but survives normalization.
    effective_sensitivity = sensitivity * dose_scale

    ct = ChemotherapySpecification(
        sensitivity=effective_sensitivity,
        decay_rate=decay_rate,
        times=dose_times,
        doses=dose_amts,
    )
    # Pydantic validator normalizes doses to max=1 — pull the actual stored
    # values back out for record-keeping.
    stored_doses = np.array(ct.doses, dtype=np.float64)

    model = ReactionDiffusion3D(
        k=torch.tensor(k, requires_grad=False),
        d=torch.tensor(d, requires_grad=False),
        theta=torch.tensor(theta, requires_grad=False),
        patient_data=patient_data,
        initial_time=t0,
        chemotherapy_specifications=[ct],
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
    timepoints = daterange(t0, t0 + timedelta(days=total_days),
                           timedelta(days=dt_save))

    print(f"  Patient: {patient_id}")
    print(f"  Grid: {tuple(u0.shape)}, DOFs: {u0.numel():,}")
    print(f"  Active tumor voxels (t=0): {(u0 > 0).sum().item():,}")
    print(f"  Params: k={k} d={d} θ={theta} sens={sensitivity} "
          f"decay={decay_rate} dose_scale={dose_scale}")
    print(f"  {len(dose_days)} doses at days "
          f"{', '.join(f'{x:.0f}' for x in dose_days)}")
    print(f"  Simulation: {total_days} d, {len(timepoints)} snapshots")
    print(f"  Solving...")

    t_start = time.time()
    times, solutions = solver.solve(timepoints=timepoints, u_initial=u0)
    solve_time = time.time() - t_start
    print(f"  Done in {solve_time:.1f}s")

    times_tensor = times if isinstance(times, torch.Tensor) else torch.stack(times)
    times_days = times_tensor.detach().cpu().numpy().astype(np.float64)

    n_dof = u0.numel()
    if isinstance(solutions, torch.Tensor) and solutions.dim() > 3:
        n_time = solutions.shape[0]
        snapshots = (solutions.detach().cpu().numpy()
                     .reshape(n_time, -1).T.astype(np.float32))
    else:
        sol_list = solutions if isinstance(solutions, list) else [
            solutions[i] for i in range(len(solutions))
        ]
        n_time = len(sol_list)
        snapshots = np.zeros((n_dof, n_time), dtype=np.float32)
        for i, sol in enumerate(sol_list):
            snapshots[:, i] = sol.detach().cpu().numpy().ravel()

    return dict(
        snapshots=snapshots,
        times_days=times_days,
        grid_shape=np.array(u0.shape),
        breast_mask=patient_data.breastmask_image.array.astype(bool),
        tumor_mask_t0=(u0_img.array > 0),
        spacing=np.array([
            float(u0_img.spacing.x),
            float(u0_img.spacing.y),
            float(u0_img.spacing.z),
        ]),
        patient_id=patient_id,
        k=k, d=d, theta=theta,
        sensitivity=sensitivity,
        decay_rate=decay_rate,
        dose_scale=dose_scale,
        chemo_dose_days=dose_days,
        chemo_doses=stored_doses,
        solve_time=solve_time,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--patient', default='TNBC_demo_001')
    p.add_argument('--days', type=float, default=120.0)
    p.add_argument('--dt-save', type=float, default=0.5)
    p.add_argument('--dt-solve', type=float, default=0.5)
    p.add_argument('--k', type=float, default=0.025)
    p.add_argument('--d', type=float, default=0.05)
    p.add_argument('--theta', type=float, default=1.0)
    p.add_argument('--sensitivity', type=float, default=0.2)
    p.add_argument('--decay-rate', type=float, default=0.7)
    p.add_argument('--dose-scale', type=float, default=1.0,
                   help="Multiplier applied to every dose (e.g. 0.5, 1.5).")
    p.add_argument('--dose-days', type=str, default=None,
                   help="Comma-separated custom dose schedule "
                        "(days since IC), e.g. '20,40,60,80,100'. "
                        "Overrides the patient-JSON schedule.")
    p.add_argument('--tag', default=None,
                   help="Suffix appended to the output file name.")
    p.add_argument('--output', default=None)
    args = p.parse_args()

    dose_days_override = None
    if args.dose_days is not None:
        dose_days_override = [float(x) for x in args.dose_days.split(',')]

    print('=' * 60)
    print('TumorTwin FOM Data Generation (with chemo)')
    print('=' * 60)
    result = generate_fom_chemo(
        patient_id=args.patient,
        total_days=args.days,
        dt_save=args.dt_save,
        dt_solve=args.dt_solve,
        k=args.k, d=args.d, theta=args.theta,
        sensitivity=args.sensitivity,
        decay_rate=args.decay_rate,
        dose_scale=args.dose_scale,
        dose_days_override=dose_days_override,
    )

    if args.output is not None:
        out_path = args.output
    else:
        tag = args.tag if args.tag is not None else f"dose{args.dose_scale:g}"
        out_path = os.path.join(
            os.path.dirname(__file__), 'data',
            f'{args.patient}_fom_chemo_{tag}.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **result)

    fsize = os.path.getsize(out_path) / 1e6
    snaps = result['snapshots']
    print(f"\n{'=' * 60}")
    print(f"Saved: {out_path} ({fsize:.1f} MB)")
    print(f"  Snapshots: {snaps.shape}  range=[{snaps.min():.4f}, {snaps.max():.4f}]")
    print(f"  Tumor volume t=0: {snaps[:, 0].sum():.0f}, "
          f"t=end: {snaps[:, -1].sum():.0f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
