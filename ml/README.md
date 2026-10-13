# Machine Learning

Full-spectrum models and RL agents for NMR spectra after ssRF / AFP manipulation.
Training data comes from [`Data_Creation/create_data.py`](../Data_Creation/README.md)
(`spectra.npz`, shape `N × 2 × 500` for I+ / I−).

Shared helpers live in [`utils/`](utils/) (`load_npz`, dataset prep, checkpoints).
Default spectra path: `ml/data/spectra_combined.npz` (override with `--spectra` where available).

## Full-spectrum P/Q models

Each script predicts integrated **P_total** / **Q_total** from the manipulated spectrum
(plus power profile / step context where applicable):

| Script | Architecture |
|--------|----------------|
| `lstm.py` | BiLSTM over frequency bins |
| `gru.py` | GRU sequence model |
| `cnn.py` | 1D CNN |
| `mlp.py` | MLP on flattened spectrum |
| `transformer.py` | Transformer encoder |
| `inception.py` | Inception-style 1D CNN |
| `old_model.py` | Legacy baseline |
| `dae.py` | Denoising autoencoder on Ps |
| `eval_model.py` | Load a checkpoint and evaluate on spectra |

### Example

```bash
# Generate spectra
python Data_Creation/create_data.py --ssrf --unmanipulated

# Train (from ml/, or with PYTHONPATH=ml)
python ml/lstm.py
python ml/gru.py
python ml/cnn.py
python ml/eval_model.py --spectra Data_Creation/spectra_data/spectra.npz
```

```bash
python ml/dae.py --spectra Data_Creation/spectra_data/spectra.npz --noise-std 0.1
```

## Analysis / plots

Diagnostic plotting under [`analysis/`](analysis/):

| Script | Purpose |
|--------|---------|
| `pq_report.py` | Shared eval/report helpers used by trainers |
| `plot_loss_landscapes.py` | Loss landscape visualizations |
| `plot_model_example_comparisons.py` | Pred vs true example spectra |
| `plot_pred_vs_true_by_snr.py` | Predictions stratified by SNR |
| `plot_pred_vs_true_outliers.py` | Outlier examples |
| `plot_val_loss_overlay.py` | Overlay validation loss curves |
| `plot_bias_variance_vs_snr.py` | Bias / variance vs SNR |
| `weight_histo.py` | Weight histograms |

```bash
python ml/analysis/plot_loss_landscapes.py
python ml/analysis/plot_model_example_comparisons.py
```

## Reinforcement learning / optimization

Burn-policy search to maximize tensor polarization Q:

| Script | Method | Physics |
|--------|--------|---------|
| `dqn.py` | Double DQN | `physics.rf` |
| `sarsa.py` | Tabular SARSA | `physics.rf` + `spectrum_gen` bridge |
| `opt_q.py` | Greedy incremental RF search | `physics.rf` |
| `rf_profile_rl.py` | RF profile RL | `physics.rf` |
| `q-learning.py` | Tabular Q-learning | Legacy lookup table |

```bash
python ml/dqn.py --episodes 100 --polarization 0.45 --max-burns 200
python ml/opt_q.py
python ml/sarsa.py
```

`q-learning.py` needs a lookup table from `Data_Creation/lookup_table.py` (if present).

## Dependencies

```bash
pip install numpy scipy matplotlib pandas torch tqdm
```

## Directory guide

```
ml/
├── lstm.py, gru.py, cnn.py, mlp.py, transformer.py, inception.py
├── dae.py, old_model.py, eval_model.py
├── dqn.py, sarsa.py, opt_q.py, rf_profile_rl.py, q-learning.py
├── utils/          Shared load_npz / training helpers
├── analysis/       Plots and pq_report
├── data/           Local spectra NPZ (gitignored)
└── results/        Checkpoints and plots (gitignored)
```

## Metrics

Trainers report **L1**, **R²**, and **median RPE** on P/Q (and related spectra where applicable) via `analysis/pq_report.py`.
