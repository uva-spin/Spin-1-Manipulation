# Machine learning

Spectrum models predict `P_total` and `Q_total` from a manipulated deuteron spectrum. The spectra come from [`Data_Creation/create_data.py`](../Data_Creation/README.md).

Shared loading, splits, and normalization are in [`utils/helpers.py`](utils/helpers.py). Checkpoints and plots go under `ml/results/` (gitignored).

## Train on a new NPZ

Every spectrum model takes `--spectra`. If you omit it, the model loads `ml/data/spectra.npz` (`DEFAULT_SPECTRA_PATH` in `utils/constants.py`). From the repo root:

```bash
python Data_Creation/create_data.py --quick --no-plots
python ml/lstm.py --spectra Data_Creation/spectra_data/spectra.npz
```

`lstm.py` trains for 500 epochs and writes `ml/results/lstm/lstm_result_ssrf_afp_combined_v7/lstm_best.pth` plus `history.json` and the P/Q report. `python ml/lstm.py --spectra <path> --test-only` reloads that checkpoint and evaluates again. `gru.py` is the same pattern, with its own results directory `ml/results/gru/gru_result_ssrf_afp_combined_v4/`. The LSTM and GRU scripts accept only `--spectra` and `--test-only`; their epochs, batch size, and learning rate are constants in the files and in `utils/constants.py`.

The other spectrum models take `--spectra`, `--epochs`, `--max-samples`, and `--out-dir`, and they can be started from the repo root:

```bash
python ml/cnn.py \
  --spectra Data_Creation/spectra_data/spectra.npz \
  --epochs 30 \
  --max-samples 20000
```

| Script | Model | Results directory |
|--------|--------|-------------------|
| `lstm.py` | Bidirectional LSTM. Takes `--spectra` and `--test-only` | `results/lstm/lstm_result_ssrf_afp_combined_v7/` |
| `gru.py` | Bidirectional GRU. Takes `--spectra` and `--test-only` | `results/gru/gru_result_ssrf_afp_combined_v4/` |
| `cnn.py` | Residual 1D CNN | `results/cnn/cnn_pq_results_v3/` |
| `mlp.py` | Residual MLP on the flattened spectrum | `results/mlp/mlp_pq_results_v4/` |
| `transformer.py` | Transformer encoder | `results/transformer/transformer_pq_results_v3/` |
| `inception.py` | Inception-style 1D CNN | `results/inception/inception_pq_results_v1/` |
| `old_model.py` | Older Inception / residual / SE CNN | `results/old_model/old_model_pq_results_v1/` |
| `eval_model.py` | Load a checkpoint and score it | `--out-dir` you pass |

`cnn.py`, `mlp.py`, `transformer.py`, `inception.py`, and `old_model.py` also accept `--batch-size`, `--lr`, `--noise-std`, `--n-examples`, `--checkpoint`, and `--test-only`.

### Inputs

`prepare_datasets` builds three channels per frequency bin:

1. **Ps** = I+ + I−, z-scored on the training split
2. **power_profile**, divided by 10
3. **n_steps**, z-scored and broadcast across bins

Splits are by `run_id` (15% of trajectories for validation, 15% for test). Ps gets Gaussian noise with std `NOISE_STD` (0.01) before the z-score. Reports from `analysis/pq_report.py` include L1, R², and median relative percent error on P and Q.

```bash
python ml/eval_model.py --model lstm \
  --checkpoint ml/results/lstm/lstm_result_ssrf_afp_combined_v7/lstm_best.pth \
  --spectra ml/data/spectra.npz \
  --out-dir ml/results/lstm/eval_once
```

## Slurm (Rivanna)

Submit `train.slurm` from `ml/`, the same directory as the model scripts. Apptainer runs with that directory as its working directory (`--pwd`), so the model is invoked as `<model>.py` and the default spectra path is `data/spectra.npz` relative to `ml/`.

```bash
cd ml
MODEL=lstm sbatch train.slurm
MODEL=cnn sbatch train.slurm -- --epochs 30
SPECTRA=data/spectra.npz MODEL=gru sbatch train.slurm
```

`MODEL` selects `<model>.py`. `SPECTRA` defaults to `data/spectra.npz`. Arguments after `--` are passed to the model script, so only use flags that script accepts (the LSTM and GRU scripts accept `--test-only`; the CNN-family scripts also accept `--epochs`, `--out-dir`, and similar). Logs go to `ml/slurm_logs/`.

## Denoising autoencoder

`dae.py` trains on Ps only. It does not predict P and Q. Pass the same style of NPZ:

```bash
python ml/dae.py \
  --spectra Data_Creation/spectra_data/spectra.npz \
  --noise-std 0.1
```

If you omit `--spectra`, it looks for `Data_Creation/dae_voigt_burn_spectra/spectra.npz` and then `ml/dae_voigt_burn_spectra/spectra.npz`. `--no-scale-01` keeps the network output in raw Ps units instead of scaling each row to [0, 1].

## Plots

[`analysis/`](analysis/README.md) holds the report helpers and standalone plots. `pq_report.py` is imported by the trainers. The `plot_*.py` scripts read checkpoints under `ml/results/`. From the repo root:

```bash
python ml/analysis/plot_loss_landscapes.py --models lstm --n-samples 256
python ml/analysis/plot_val_loss_overlay.py
```

## Reinforcement learning

These scripts roll out `physics.rf` themselves. They do not read `spectra.npz`.

| Script | Method | How to run |
|--------|--------|------------|
| `dqn.py` | Double DQN on a 701-bin ideal-bin profile | `python ml/dqn.py --episodes 100 --polarization 0.45` |
| `sarsa.py` | Tabular SARSA | `python ml/sarsa.py` (fixed episode count and polarization grid in the file) |
| `opt_q.py` | Greedy search over burn bins | `python ml/opt_q.py` |
| `rf_profile_rl.py` | RF-profile policy | `python ml/rf_profile_rl.py` |

`q-learning.py` imports `physics.lineshape.ssRFMapper` and expects `Data_Creation/lookup_table.pkl`. Neither is in this tree, so that script does not run here.

DQN writes under `results/current/dqn/` at the repo root, not under `ml/results/`.

## Dependencies

```bash
pip install numpy scipy matplotlib pandas torch tqdm
```
