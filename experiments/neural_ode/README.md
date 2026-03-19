# Neural ODE Architecture Sweep

Systematic comparison of neural ODE architectures for learning PDE dynamics in
reduced coordinates. Tests whether neural ODEs can match Bayesian OpInf given
fair hyperparameter tuning.

## Experiments

| Script | PDE System | Architectures | Steps | LR Schedule |
|--------|-----------|---------------|-------|-------------|
| `euler_neural_ode.py` | Compressible Euler | 2×64, 3×128, 4×256 | 5000 | Cosine annealing |
| `heat_neural_ode.py` | Cubic Heat | 2×64, 3×128, 4×256 | 5000 | Cosine annealing |

## Data Regimes (same as 04_conditional_integral.py)

**Euler:**
1. Dense low noise: 250 samples, 1% noise
2. Sparse low noise: 55 samples, 3% noise
3. Dense high noise: 250 samples, 10% noise

**Heat:**
1. Dense low noise: 65 samples/IC, 1% noise, 5 training ICs + 1 test
2. Sparse medium noise: 20 samples/IC, 5% noise
3. Dense high noise: 65 samples/IC, 10% noise

## Usage

### Local (full repo)
```bash
cd probabilistic_rom_inference/experiments/neural_ode
conda activate prob_rom_jax_opinf
python euler_neural_ode.py
python heat_neural_ode.py
```

### Kaggle
1. Upload this directory as a Kaggle dataset
2. Run `python euler_neural_ode.py --save-data` locally first to generate data .npz files
3. In Kaggle notebook, set `DATA_DIR` to the dataset path and run

## Key Design Choices

- **No GP denoising**: Neural ODEs train on raw noisy POD-compressed data (fair comparison)
- **Ensemble UQ**: 20 members per architecture for uncertainty quantification
- **Cosine LR**: Warm start at lr_max, cosine decay to 0 over training
- **Same POD basis**: Fitted once on training data, shared across all methods
- **Same evaluation**: Relative L2 error (train/pred), 90% CI coverage, stability %

## Output

Figures are saved to `figures/` with naming: `{euler,heat}_{arch}_{regime}_*.png`
Summary comparison tables are printed to stdout.
