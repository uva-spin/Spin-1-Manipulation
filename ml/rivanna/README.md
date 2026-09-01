# Rivanna ML Training

Train and evaluate per-bin MLP models on the Rivanna GPU cluster (or locally).

Each spectral bin gets its own small neural network that predicts **I+** and **I−** (or P and Q) at that bin, given manipulation parameters and the local Ps value.

## Files

| File | Purpose |
|------|---------|
| `single_bin.py` | Train one bin's MLP |
| `test-binning.py` | Evaluate the combined 500-bin model |
| `combine_single_bin_models.py` | Merge per-bin `.pth` files (also at `ml/combine_single_bin_models.py`) |
| `train_single_bin_array.slurm` | SLURM GPU array — one bin per task |
| `combine_single_bin_models.slurm` | SLURM job to merge checkpoints |

## Train one bin locally

```bash
python ml/rivanna/single_bin.py \
  --bin-idx 172 \
  --data-dir Data_Creation/rivanna/data/combined_train_all \
  --output-dir ml/models/single_bin
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--bin-idx` | (required) | Spectral bin index (0–499) |
| `--data-dir` | `combined_train_all` | Directory with `train_bin_XXXX.npz` files |
| `--data-file` | None | Explicit NPZ path (overrides `--data-dir` lookup) |
| `--output-dir` | `single_bin_models` | Where to write `.pth` and metrics JSON |
| `--feature-clip-z` | 0.0 | Optional z-score clip on Ps before training |
| `--train-polarization-fraction` | 0.8 | Fraction of distinct p0 values used for training |

### Model details

- **Features:** `gamma_rf`, `n_steps`, `ps` (configured via `FEATURE_SET` in the script)
- **Targets:** `iplus`, `iminus` (configured via `TARGET_MODE = "iplus_iminus"`)
- **Architecture:** 4-layer MLP trunk (256 hidden units) with separate heads for each target
- **Split:** Train/holdout by polarization `p0`; holdout is further split into val/test

Output per bin:

```
ml/models/single_bin/
├── binning_model_bin_172.pth
└── binning_model_bin_172_metrics.json
```

## Train all bins on the cluster

From the repo root:

```bash
DATA_DIR=Data_Creation/rivanna/data/combined_train_all \
OUTPUT_DIR=ml/models/single_bin \
TRAIN_SCRIPT=ml/rivanna/single_bin.py \
  sbatch ml/rivanna/train_single_bin_array.slurm
```

Throttle concurrent jobs:

```bash
sbatch --array=125-374%32 ml/rivanna/train_single_bin_array.slurm
```

The SLURM script auto-detects `single_bin.py` in the working directory or at `ml/rivanna/single_bin.py`. It runs inside an Apptainer PyTorch container on Rivanna (`gpu:a6000`).

## Combine bin checkpoints

After all 500 bins are trained:

```bash
python ml/combine_single_bin_models.py \
  --model-dir ml/models/single_bin \
  --output ml/models/combined_bin_model.pth \
  --num-bins 500 --strict
```

Or on the cluster:

```bash
MODEL_DIR=ml/models/single_bin \
  sbatch ml/rivanna/combine_single_bin_models.slurm
```

The combined file contains all 500 state dicts plus per-bin normalization statistics (`X_mean`, `X_std`, output means/stds).

## Evaluate

Edit the constants at the top of `test-binning.py`:

```python
MODEL_PATH = "ml/models/combined_bin_model.pth"
TEST_FILE = "data/spectra.npz"          # or path to test spectra NPZ
OUTPUT_DIR = "results/test_binning"
```

Then run:

```bash
python ml/rivanna/test-binning.py
```

### What evaluation produces

- `test_statistics.json` — L1, median RPE for P, Q, I+, I−
- `median_rpe_per_bin.csv` — per-bin error breakdown
- `residuals_heatmap_iplus.png`, `residuals_heatmap_iminus.png`
- `lineshape_examples.png` — side-by-side true vs predicted spectra

The test file should be a `spectra.npz` with shape `(N, 2, num_bins)` for I+/I−, plus `p0`, `applied_power`, `n_steps`, `center_bin`, and `source` arrays. By default only ssRF rows (`source == 0`) are evaluated.

## Typical end-to-end timing

| Step | Where | Approximate scale |
|------|-------|-------------------|
| Data generation | `Data_Creation/rivanna/` SLURM arrays | Hours (500 bins × many polarizations) |
| Per-bin training | `train_single_bin_array.slurm` | ~1–3 h per bin on GPU; run as array |
| Combine | `combine_single_bin_models.py` | Minutes |
| Evaluation | `test-binning.py` | Minutes |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No training NPZ for bin N` | That bin has no data — check `combined_train_all/train_bin_NNNN.npz` exists |
| `need >= 2 distinct p0 values` | Regenerate data with a wider polarization grid |
| Import error for `single_bin` | Run from repo root, or `cd ml/rivanna` |
| Missing `CONTAINERDIR` on cluster | `module load apptainer pytorch/2.9.0` before submitting |
