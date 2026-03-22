#!/usr/bin/env python3
"""
Test moderate data regimes to find where 04 (Conditional Integral) 
outperforms 05 (Neural ODE) by the widest margin.

Regimes:
  1. NUM_SAMPLES=30, NOISE_LEVEL=0.05  (halved samples + moderate noise)
  2. NUM_SAMPLES=40, NOISE_LEVEL=0.08  (moderate samples + high noise)
  3. NUM_SAMPLES=60, NOISE_LEVEL=0.10  (same samples + very high noise)

Safety: checks for running processes, re-reads files before each edit,
        restores original SCHEMAS after all tests.
"""

import os
import re
import sys
import time
import subprocess
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_04 = os.path.join(SCRIPT_DIR, "04_conditional_integral.py")
FILE_05 = os.path.join(SCRIPT_DIR, "05_neural_ode.py")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "comparison", "dense_medium_noise")
PYTHON = sys.executable

ORIGINAL_SCHEMAS = {
    "NUM_SAMPLES": 60,
    "NOISE_LEVEL": 0.03,
}

REGIMES = [
    {"name": "regime1_halved_moderate", "NUM_SAMPLES": 30, "NOISE_LEVEL": 0.05},
    {"name": "regime2_moderate_high",   "NUM_SAMPLES": 40, "NOISE_LEVEL": 0.08},
    {"name": "regime3_same_veryhigh",   "NUM_SAMPLES": 60, "NOISE_LEVEL": 0.10},
]

# Regex to match the SCHEMAS block (works for both files)
SCHEMAS_PATTERN = re.compile(
    r'(SCHEMAS\s*=\s*\[\s*\{[^}]*?"NUM_SAMPLES":\s*)\d+(\s*,\s*"NOISE_LEVEL":\s*)[\d.]+',
    re.DOTALL,
)


def check_no_running_processes():
    """Abort if 04 or 05 scripts are currently running."""
    result = subprocess.run(
        ["ps", "aux"], capture_output=True, text=True
    )
    for keyword in ["04_conditional_integral", "05_neural_ode"]:
        if keyword in result.stdout and "test_regimes" not in result.stdout.split(keyword)[0].split('\n')[-1]:
            # More careful check: exclude our own process
            for line in result.stdout.splitlines():
                if keyword in line and "test_regimes" not in line and "grep" not in line:
                    print(f"⚠️  Found running process: {line.strip()}")
                    return False
    return True


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def set_schemas(filepath, num_samples, noise_level):
    """Edit the SCHEMAS block in the given file to use new values."""
    content = read_file(filepath)
    
    new_content, count = SCHEMAS_PATTERN.subn(
        rf'\g<1>{num_samples}\g<2>{noise_level}',
        content,
    )
    if count == 0:
        raise RuntimeError(f"Could not find SCHEMAS pattern in {filepath}")
    
    write_file(filepath, new_content)
    
    # Verify
    verify = read_file(filepath)
    if f'"NUM_SAMPLES": {num_samples}' not in verify:
        raise RuntimeError(f"Verification failed: NUM_SAMPLES not set to {num_samples}")
    if f'"NOISE_LEVEL": {noise_level}' not in verify:
        raise RuntimeError(f"Verification failed: NOISE_LEVEL not set to {noise_level}")
    
    print(f"  ✅ Set {os.path.basename(filepath)}: samples={num_samples}, noise={noise_level}")


def restore_originals():
    """Restore both files to original SCHEMAS."""
    print("\n🔄 Restoring original SCHEMAS (samples=60, noise=0.03)...")
    for filepath in [FILE_04, FILE_05]:
        set_schemas(filepath, ORIGINAL_SCHEMAS["NUM_SAMPLES"], ORIGINAL_SCHEMAS["NOISE_LEVEL"])
    print("✅ Originals restored.\n")


def run_script(script_path, label):
    """Run a Python script and return success/failure."""
    print(f"  🚀 Running {os.path.basename(script_path)} for {label}...")
    start = time.time()
    result = subprocess.run(
        [PYTHON, script_path],
        capture_output=True, text=True,
        cwd=SCRIPT_DIR,
        timeout=600,  # 10 min max per script
    )
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"  ❌ FAILED ({elapsed:.0f}s)")
        print(f"  stderr: {result.stderr[-500:]}")
        return False
    
    print(f"  ✅ Done ({elapsed:.0f}s)")
    return True


def load_results(method_name):
    """Load results from .npz file."""
    path = os.path.join(RESULTS_DIR, f"{method_name}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "train_error": float(data["train_error"]),
        "pred_error": float(data["pred_error"]),
        "stability_pct": float(data["stability_pct"]),
        "runtime": float(data["runtime"]),
    }


def main():
    print("=" * 80)
    print("REGIME TESTING — Moderate Adjustments")
    print("Finding where 04 (Conditional Integral) > 05 (Neural ODE)")
    print("=" * 80)
    
    # Safety check
    if not check_no_running_processes():
        print("❌ Other experiments are running. Aborting.")
        sys.exit(1)
    
    all_results = {}
    
    try:
        for i, regime in enumerate(REGIMES):
            ns, nl = regime["NUM_SAMPLES"], regime["NOISE_LEVEL"]
            label = f"Regime {i+1}: samples={ns}, noise={nl}"
            print(f"\n{'─' * 70}")
            print(f"📊 {label}")
            print(f"{'─' * 70}")
            
            # Check again before each regime
            if not check_no_running_processes():
                print("⚠️  Other process detected, waiting 30s...")
                time.sleep(30)
                if not check_no_running_processes():
                    print("❌ Still running. Skipping this regime.")
                    continue
            
            # Edit both files
            set_schemas(FILE_04, ns, nl)
            set_schemas(FILE_05, ns, nl)
            
            regime_results = {"label": label, "ns": ns, "nl": nl}
            
            # Run 04
            ok_04 = run_script(FILE_04, label)
            if ok_04:
                regime_results["04"] = load_results("04_conditional_integral")
            else:
                regime_results["04"] = None
            
            # Run 05
            ok_05 = run_script(FILE_05, label)
            if ok_05:
                regime_results["05"] = load_results("05_neural_ode")
            else:
                regime_results["05"] = None
            
            all_results[regime["name"]] = regime_results
    
    finally:
        # ALWAYS restore originals
        restore_originals()
    
    # ── Print results table ──────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("RESULTS SUMMARY")
    print("=" * 110)
    header = (f"{'Regime':<35s} │ {'04 Train%':>9s} {'04 Pred%':>9s} {'04 Stab':>7s} │ "
              f"{'05 Train%':>9s} {'05 Pred%':>9s} {'05 Stab':>7s} │ {'Gap(pred)':>10s}")
    print(header)
    print("─" * 110)
    
    # Also include baseline for comparison
    print(f"{'Baseline (n=60, σ=0.03)':<35s} │ {'0.97':>9s} {'3.11':>9s} {'100%':>7s} │ "
          f"{'1.95':>9s} {'6.51':>9s} {'100%':>7s} │ {'3.40pp':>10s}")
    
    best_gap = -1
    best_regime = None
    
    for name, r in all_results.items():
        label = f"n={r['ns']}, σ={r['nl']}"
        
        if r.get("04") and r.get("05"):
            t04 = r["04"]["train_error"] * 100
            p04 = r["04"]["pred_error"] * 100
            s04 = r["04"]["stability_pct"]
            t05 = r["05"]["train_error"] * 100
            p05 = r["05"]["pred_error"] * 100
            s05 = r["05"]["stability_pct"]
            gap = p05 - p04
            
            if gap > best_gap:
                best_gap = gap
                best_regime = label
            
            print(f"{label:<35s} │ {t04:>8.2f}% {p04:>8.2f}% {s04:>6.0f}% │ "
                  f"{t05:>8.2f}% {p05:>8.2f}% {s05:>6.0f}% │ {gap:>9.2f}pp")
        else:
            fail_04 = "FAIL" if not r.get("04") else "ok"
            fail_05 = "FAIL" if not r.get("05") else "ok"
            print(f"{label:<35s} │ {'—':>9s} {'—':>9s} {'—':>7s} │ "
                  f"{'—':>9s} {'—':>9s} {'—':>7s} │ 04={fail_04}, 05={fail_05}")
    
    print("─" * 110)
    if best_regime:
        print(f"\n🏆 Recommended regime: {best_regime}  (prediction gap = {best_gap:.2f}pp)")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
