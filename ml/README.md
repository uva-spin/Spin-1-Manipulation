# Machine Learning

Models and reinforcement-learning agents for predicting NMR spectra after manipulation and for optimizing burn policies.

## Model types

| Script | What it learns | Input data |
|--------|----------------|------------|
| [`rivanna/single_bin.py`](rivanna/README.md) | One small MLP **per spectral bin** | `train_bin_XXXX.npz` from data pipeline |
| `spectrum_pq.py` | Full-spectrum P/Q prediction | `spectra.npz` (N × 2 × 500) |
| `dae.py` | Denoising autoencoder on Ps | `spectra.npz` |
| `binning_model.py` | All 500 bins in one script (local) | Combined NPZ directory |

## Reinforcement learning / optimization

These scripts search for burn sequences that maximize tensor polarization Q:

| Script | Method | Physics backend |
|--------|--------|-----------------|
| `dqn.py` | Double DQN | `Data_Creation/rivanna` ssrf_realtime |
| `sarsa.py` | Tabular SARSA | rivanna ssrf_realtime |
| `q-learning.py` | Tabular Q-learning | Legacy lookup-table mapper |
| `opt_q.py` | Greedy incremental RF search | `physics/ssrf_realtime` |

### Example: train a DQN burn policy

```bash
python ml/dqn.py --episodes 100 --polarization 0.45 --max-burns 200
```

### Example: greedy Q optimizer (single burn demo)

```bash
python ml/opt_q.py
```

Plots and traces are written to `results/current/binwise_incremental_realtime/`.

## Per-bin MLP pipeline (recommended)

This is the main production workflow. See [`rivanna/README.md`](rivanna/README.md) for details.

```bash
# 1. Train one bin locally
python ml/rivanna/single_bin.py \
  --bin-idx 208 \
  --data-dir Data_Creation/rivanna/data/combined_train_all \
  --output-dir ml/models/single_bin

# 2. Combine all 500 bin checkpoints
python ml/combine_single_bin_models.py \
  --model-dir ml/models/single_bin \
  --output ml/models/combined_bin_model.pth \
  --num-bins 500 --strict

# 3. Evaluate
python ml/rivanna/test-binning.py
```

## Full-spectrum P/Q model

```bash
# Generate training spectra
python Data_Creation/create_dae_voigt_burn_spectra.py --quick

# Train
python ml/spectrum_pq.py \
  --spectra Data_Creation/dae_voigt_burn_spectra/spectra.npz \
  --out-dir ml/spectrum_pq_results
```

## Denoising autoencoder

```bash
python ml/dae.py \
  --spectra Data_Creation/dae_voigt_burn_spectra/spectra.npz \
  --noise-std 0.1
```

## Tabular Q-learning (legacy mapper)

```bash
python ml/q-learning.py --episodes 100 --polarization 0.45
```

Requires a pre-built lookup table from `Data_Creation/lookup_table.py`.

## Dependencies

- **PyTorch** — all neural models
- **numpy, pandas, matplotlib** — data I/O and plotting
- **tqdm** — progress bars in RL scripts

Install from the repo root:

```bash
pip install numpy scipy matplotlib pandas torch tqdm
```

## Directory guide

```
ml/
├── rivanna/                  Cluster training + evaluation (start here)
├── combine_single_bin_models.py   Merge per-bin checkpoints
├── spectrum_pq.py            Full-spectrum model
├── dae.py                    Denoising autoencoder
├── dqn.py, sarsa.py, q-learning.py, opt_q.py   RL / optimization
├── binning_model.py          Monolithic local trainer (all bins)
└── models/                   Checkpoint output (gitignored)
```

## Evaluation metrics

Scripts report **L1 loss**, **R²**, and **median RPE** (relative percent error) on P, Q, I+, and I−. See `ml/rivanna/test-binning.py` for per-bin evaluation with heatmaps and example lineshape plots.
