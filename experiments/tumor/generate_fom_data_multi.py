#!/usr/bin/env python
"""Generate multi-trajectory FOM data for ROM learning across (k, d) pairs.

Runs TumorTwin's ReactionDiffusion3D for multiple (k, d) parameter
combinations on the SAME patient anatomy (TNBC_demo_001). This enables
training a ROM that generalises across parameter space.

Each trajectory is saved as a separate NPZ file. A manifest JSON file
records all parameter sets and their file paths.

Usage:
    conda run -n prob_rom python generate_fom_data_multi.py
"""

import os
import sys
import json
import time
import numpy as np

# Reuse the single-trajectory generator
sys.path.insert(0, os.path.dirname(__file__))
from generate_fom_data import generate_fom_snapshots

# ── Parameter sets ──────────────────────────────────────────────────────────
TRAINING_PARAMS = [
    (0.01, 0.03),   # slow growth, low diffusion
    (0.02, 0.08),   # moderate growth, high diffusion
    (0.03, 0.05),   # middle of range
    (0.04, 0.02),   # fast growth, low diffusion
    (0.05, 0.10),   # fast growth, high diffusion
]
TEST_PARAMS = (0.025, 0.06)  # unseen interpolation point

PATIENT_ID = "TNBC_demo_001"
TOTAL_DAYS = 90.0
THETA = 1.0

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def npz_filename(k, d):
    return f"{PATIENT_ID}_k{k:.3f}_d{d:.3f}_fom.npz"


def generate_all():
    os.makedirs(DATA_DIR, exist_ok=True)

    all_params = TRAINING_PARAMS + [TEST_PARAMS]
    manifest = {"patient_id": PATIENT_ID, "theta": THETA,
                "total_days": TOTAL_DAYS, "training": [], "test": None}

    wall_start = time.time()

    for idx, (k, d) in enumerate(all_params):
        is_test = (k, d) == TEST_PARAMS
        label = "TEST" if is_test else f"TRAIN {idx+1}/{len(TRAINING_PARAMS)}"
        fname = npz_filename(k, d)
        out_path = os.path.join(DATA_DIR, fname)

        print(f"\n{'='*60}")
        print(f"[{label}]  k={k}, d={d}")
        print(f"{'='*60}")

        result = generate_fom_snapshots(
            patient_id=PATIENT_ID,
            total_days=TOTAL_DAYS,
            k=k,
            d=d,
            theta=THETA,
        )

        np.savez_compressed(out_path, **result)
        file_mb = os.path.getsize(out_path) / 1e6
        snaps = result["snapshots"]

        print(f"  Saved: {fname} ({file_mb:.1f} MB)")
        print(f"  Shape: {snaps.shape}, tumor vol t=0: {snaps[:, 0].sum():.0f}, "
              f"t=end: {snaps[:, -1].sum():.0f}")

        entry = {"k": k, "d": d, "file": fname, "path": out_path,
                 "size_mb": round(file_mb, 2)}
        if is_test:
            manifest["test"] = entry
        else:
            manifest["training"].append(entry)

    # Save manifest
    manifest_path = os.path.join(DATA_DIR, "multi_traj_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - wall_start
    print(f"\n{'='*60}")
    print(f"All done — {len(all_params)} trajectories in {total_time:.1f}s")
    print(f"Manifest: {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    generate_all()
