"""Noise sweep: compare 04 vs 02 at multiple noise levels."""
import subprocess, sys, os, re, numpy as np, json, textwrap

PYTHON = "/Users/anthonypoole/miniconda3/envs/prob_rom_jax_opinf/bin/python"
BASE = os.path.dirname(os.path.abspath(__file__))
FILE_02 = os.path.join(BASE, "02_two_stage_svi.py")
FILE_04 = os.path.join(BASE, "04_conditional_integral.py")
RESULTS_DIR = os.path.join(BASE, "results", "comparison")

NOISE_LEVELS = [0.02, 0.04, 0.06, 0.07]

ORIGINAL_SCHEMAS = textwrap.dedent('''\
SCHEMAS = [
    {
        "name": "sparse_low_noise",
        "label": "Sparse data, low noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.01,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_medium_noise",
        "label": "Sparse data, medium noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.03,
        "NUM_EVAL_POINTS": 100,
    },
    {
        "name": "sparse_high_noise",
        "label": "Sparse data, high noise",
        "NUM_SAMPLES": 20,
        "NOISE_LEVEL": 0.05,
        "NUM_EVAL_POINTS": 100,
    },
]''')

TEST_SCHEMA_TEMPLATE = '''\
SCHEMAS = [
    {{"name": "test_noise", "label": "Test", "NUM_SAMPLES": 20, "NOISE_LEVEL": {noise}, "NUM_EVAL_POINTS": 100}},
]'''

def replace_schemas(filepath, new_schemas_text):
    with open(filepath, 'r') as f:
        content = f.read()
    # Match SCHEMAS = [ ... ] block
    pattern = r'SCHEMAS\s*=\s*\[.*?\n\]'
    new_content = re.sub(pattern, new_schemas_text, content, count=1, flags=re.DOTALL)
    with open(filepath, 'w') as f:
        f.write(new_content)

def run_method(script_path, label):
    print(f"  Running {label}...", flush=True)
    result = subprocess.run(
        [PYTHON, script_path],
        capture_output=True, text=True, cwd=BASE, timeout=1200
    )
    if result.returncode != 0:
        print(f"  ERROR in {label}:")
        print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        return False
    print(f"  {label} completed.", flush=True)
    return True

def load_results(method_name):
    path = os.path.join(RESULTS_DIR, "test_noise", f"{method_name}.npz")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "train_error": float(data["train_error"]),
        "pred_error": float(data["pred_error"]),
        "stability_pct": float(data["stability_pct"]),
    }

def main():
    all_results = {}

    for noise in NOISE_LEVELS:
        pct = int(noise * 100)
        print(f"\n{'='*60}")
        print(f"Testing noise = {pct}%")
        print(f"{'='*60}", flush=True)

        schema_text = TEST_SCHEMA_TEMPLATE.format(noise=noise)
        replace_schemas(FILE_04, schema_text)
        replace_schemas(FILE_02, schema_text)

        results_04 = None
        results_02 = None

        if run_method(FILE_04, "04_conditional_integral"):
            results_04 = load_results("04_conditional_integral")
            if results_04:
                print(f"  04: train={results_04['train_error']*100:.2f}%, pred={results_04['pred_error']*100:.2f}%, stability={results_04['stability_pct']:.1f}%")

        if run_method(FILE_02, "02_two_stage_svi"):
            results_02 = load_results("02_two_stage_svi")
            if results_02:
                print(f"  02: train={results_02['train_error']*100:.2f}%, pred={results_02['pred_error']*100:.2f}%, stability={results_02['stability_pct']:.1f}%")

        all_results[pct] = {"04": results_04, "02": results_02}

    # Restore original schemas
    print(f"\n{'='*60}")
    print("Restoring original SCHEMAS...")
    replace_schemas(FILE_04, ORIGINAL_SCHEMAS)
    replace_schemas(FILE_02, ORIGINAL_SCHEMAS)
    print("Done.")

    # Print summary
    print(f"\n{'='*60}")
    print("FULL RESULTS TABLE")
    print(f"{'='*60}")
    
    # Known results
    known = {
        1: {"04": 5.58, "02": 4.86},
        3: {"04": 7.35, "02": 9.98},
        5: {"04": 10.47, "02": 9.66},
    }

    print(f"{'Noise%':>6} | {'04 Train':>9} | {'04 Pred':>9} | {'04 Stab':>8} | {'02 Train':>9} | {'02 Pred':>9} | {'02 Stab':>8} | {'04 wins?':>8} | {'Delta':>8}")
    print("-" * 100)

    for pct in [1, 2, 3, 4, 5, 6, 7]:
        if pct in known and pct not in all_results:
            p04 = known[pct]["04"]
            p02 = known[pct]["02"]
            wins = "YES" if p04 < p02 else "no"
            delta = p02 - p04
            print(f"{pct:>5}% | {'(known)':>9} | {p04:>8.2f}% | {'(known)':>8} | {'(known)':>9} | {p02:>8.2f}% | {'(known)':>8} | {wins:>8} | {delta:>+7.2f}pp")
        elif pct in all_results:
            r = all_results[pct]
            if r["04"] and r["02"]:
                t04 = r["04"]["train_error"] * 100
                p04 = r["04"]["pred_error"] * 100
                s04 = r["04"]["stability_pct"]
                t02 = r["02"]["train_error"] * 100
                p02 = r["02"]["pred_error"] * 100
                s02 = r["02"]["stability_pct"]
                wins = "YES" if p04 < p02 else "no"
                delta = p02 - p04
                print(f"{pct:>5}% | {t04:>8.2f}% | {p04:>8.2f}% | {s04:>7.1f}% | {t02:>8.2f}% | {p02:>8.2f}% | {s02:>7.1f}% | {wins:>8} | {delta:>+7.2f}pp")
            else:
                print(f"{pct:>5}% | FAILED")
        else:
            print(f"{pct:>5}% | no data")

    # JSON dump for easy parsing
    print(f"\n--- RAW RESULTS JSON ---")
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = v
    print(json.dumps(serializable, indent=2, default=str))

if __name__ == "__main__":
    main()
