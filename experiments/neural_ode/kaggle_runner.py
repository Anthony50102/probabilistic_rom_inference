# %% [markdown]
# # Neural ODE Architecture Sweep — Kaggle Runner
#
# Runs the Euler and Heat neural ODE architecture sweep scripts from the repo.
# Clone this repo into Kaggle, then run this notebook.
#
# **Expected runtime**: ~2-3 hours on T4 GPU

# %%
# Install JAX neural ODE dependencies
!pip install -q equinox diffrax optax

# %%
import os, subprocess, zipfile, shutil

REPO_ROOT = '/kaggle/input'  # adjust if repo is cloned elsewhere
# Auto-detect repo root
for d in ['/kaggle/working/thesis_claude/probabilistic_rom_inference',
          '/kaggle/input/thesis_claude/probabilistic_rom_inference',
          '../probabilistic_rom_inference',
          'probabilistic_rom_inference']:
    if os.path.isdir(d):
        REPO_ROOT = d
        break
print(f"Repo root: {REPO_ROOT}")

SWEEP_DIR = os.path.join(REPO_ROOT, 'experiments', 'neural_ode')
assert os.path.isfile(os.path.join(SWEEP_DIR, 'euler_neural_ode.py')), \
    f"Can't find euler_neural_ode.py in {SWEEP_DIR}"

# %%
# Run Euler sweep
print("=" * 70)
print("Running Euler architecture sweep...")
print("=" * 70)
result = subprocess.run(
    ['python', 'euler_neural_ode.py'],
    cwd=SWEEP_DIR,
    capture_output=False,
)
print(f"Euler exit code: {result.returncode}")

# %%
# Run Heat sweep
print("=" * 70)
print("Running Heat architecture sweep...")
print("=" * 70)
result = subprocess.run(
    ['python', 'heat_neural_ode.py'],
    cwd=SWEEP_DIR,
    capture_output=False,
)
print(f"Heat exit code: {result.returncode}")

# %%
# Zip all results
FIGURES_DIR = os.path.join(SWEEP_DIR, 'figures')
ZIP_PATH = '/kaggle/working/neural_ode_sweep_results.zip'

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(FIGURES_DIR):
        for f in files:
            filepath = os.path.join(root, f)
            arcname = os.path.join('figures', f)
            zf.write(filepath, arcname)

n_files = len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])
print(f"\n📦 Zipped {n_files} figures to: {ZIP_PATH}")
print("Download from the Output tab on the right →")
