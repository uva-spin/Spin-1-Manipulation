# Spin-1 Manipulation

Simulates deuteron (spin-1) solid-state NMR and trains models that read a manipulated spectrum and predict vector polarization **P** and tensor polarization **Q**.

Training spectra have **500 frequency bins** on R ∈ [−6, 6]. Each bin stores two branch intensities: **I+** (+1↔0) and **I−** (0↔−1). Ps = I+ + I−. Each saved event also stores `P_total` and `Q_total`.

Two manipulations show up in the data:

- **ssRF** (semi-saturated radiofrequency): a local RF burn
- **AFP** (adiabatic fast passage): an adiabatic sweep

## Layout

| Path | Role |
|------|------|
| [`physics/`](physics/README.md) | Rate-equation model (`physics.rf`) and the analytic lineshape |
| [`Data_Creation/`](Data_Creation/README.md) | Writes labeled spectra to `spectra.npz` |
| [`ml/`](ml/README.md) | Trains spectrum→P/Q models and a few RL burn policies |

`physics.rf` defaults to **701 bins** on R ∈ [−3, 3]. The NPZ files used for training are the 500-bin spectra from `Data_Creation/`.

## Setup

Python 3.10 or newer. From the repo root:

```bash
pip install numpy scipy matplotlib pandas torch tqdm pytest
```

CPU PyTorch is enough for a small training run.

## Create data, then train

### 1. Write a small NPZ

```bash
python Data_Creation/create_data.py --quick --no-plots
```

`--quick` uses P = 0.30 and 0.50, three burn centers per polarization, and short burn and relax grids. It includes ssRF, per-bin AFP, and the optimal RF profile. The file is `Data_Creation/spectra_data/spectra.npz`.

With no mode flags and without `--quick`, the script generates **ssRF only**, on the grids at the top of [`Data_Creation/create_data.py`](Data_Creation/create_data.py): burn steps 0–400 and relax steps 0–8000, step 1. That grid is large. Cap it, for example:

```bash
python Data_Creation/create_data.py --ssrf --afp --unmanipulated \
  --max-burn-steps 40 --burn-steps-step 5 \
  --max-relax-steps 40 --relax-steps-step 5 \
  --no-plots
```

### 2. Train

All spectrum models take `--spectra`. If you omit it, they load `ml/data/spectra.npz`.

```bash
python ml/lstm.py --spectra Data_Creation/spectra_data/spectra.npz
```

The default run is 500 epochs (`NUM_EPOCHS` in `ml/utils/constants.py`). The checkpoint and plots go to `ml/results/lstm/lstm_result_ssrf_afp_combined_v7/`.

A short run of another model, from the repo root:

```bash
python ml/cnn.py \
  --spectra Data_Creation/spectra_data/spectra.npz \
  --epochs 5 \
  --max-samples 2000
```

Each of these models reads three values per frequency bin:

1. Ps, z-scored on the training split
2. per-bin RF power profile, divided by 10
3. `n_steps`, z-scored and repeated on every bin

Train, validation, and test splits follow `run_id`, so frames from one trajectory stay in one split. Dataset prep adds Gaussian noise to Ps (`NOISE_STD = 0.01` in `ml/utils/constants.py`).

More flags, NPZ fields, and the other scripts are in [`Data_Creation/README.md`](Data_Creation/README.md) and [`ml/README.md`](ml/README.md).

## Tests

```bash
python -m pytest Data_Creation/spectrum_gen/tests -q
cd physics/rf && pytest -q
```

## Glossary

| Term | Meaning |
|------|---------|
| Bin | One frequency index |
| I+ / I− | Intensities of the +1↔0 and 0↔−1 transitions |
| Ps | I+ + I− |
| P / Q | Vector / tensor polarization |
| p0 | Initial polarization |
| γ_rf | ssRF burn strength |
| source | `0` ssRF, `1` AFP, `2` unmanipulated, `3` optimal profile, `4` AFP profile, `5` ssRF+AFP combined |
