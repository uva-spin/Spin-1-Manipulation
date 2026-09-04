# Spin-1-Manipulation

Simulations and machine-learning tools for manipulating **spin-1** (deuteron) solid-state NMR targets. The project focuses on two manipulation types:

- **ssRF** (semi-saturated radiofrequency) — local RF burns that redistribute polarization without full adiabatic inversion ([Clement et al.](https://www.sciencedirect.com/science/article/pii/S0168900223001675))
- **AFP** (adiabatic fast passage) — adiabatic sweeps that flip populations across a frequency window (see Abragam, *Principles of Nuclear Magnetism*)

The typical workflow is: **simulate physics → generate training data → train models → evaluate or optimize burn policies**.

## Repository layout

| Directory | What it does |
|-----------|--------------|
| [`physics/`](physics/README.md) | NMR lineshape math and the realtime spin-1 rate-equation model |
| [`Data_Creation/`](Data_Creation/README.md) | Scripts that generate labeled spectra for ML training |
| [`Data_Creation/rivanna/`](Data_Creation/rivanna/README.md) | **Primary data pipeline** — self-contained, cluster-ready |
| [`ml/`](ml/README.md) | Per-bin MLPs, full-spectrum models, RL burn policies, Seq2Seq models |
| [`ml/rivanna/`](ml/rivanna/README.md) | Cluster training and evaluation of models on supercomputing cluster |
| [`analysis/`](analysis/README.md) | Small visualization utilities (optional) |

## Quick start

### 1. Install dependencies

From the repo root:

```bash
pip install numpy scipy matplotlib pandas torch tqdm pytest
```

For the data-generation package only, you can use the smaller list in [`Data_Creation/rivanna/requirements.txt`](Data_Creation/rivanna/requirements.txt).

You need **Python 3.10+** and a working **PyTorch** install (CPU is fine for small tests; GPU helps for training).

### 2. Verify the physics model

```bash
cd physics/ssrf_realtime
pytest -q
```

### 3. Generate a small amount of training data (local smoke test)

```bash
cd Data_Creation/rivanna
python -m pytest tests/test_smoke.py -q
python generate_bins.py --mode all --smoke --bin-idx 208
python combine_all_train.py --strict
```

This writes per-bin NPZ files under `Data_Creation/rivanna/data/combined_train_all/`.

### 4. Train one spectral bin locally

```bash
python ml/rivanna/single_bin.py \
  --bin-idx 208 \
  --data-dir Data_Creation/rivanna/data/combined_train_all \
  --output-dir ml/models/single_bin
```

### 5. Evaluate (after training all bins and combining — see [`ml/rivanna/README.md`](ml/rivanna/README.md))

Edit the `MODEL_PATH` and `TEST_FILE` constants at the top of `ml/rivanna/test-binning.py`, then:

```bash
python ml/rivanna/test-binning.py
```

## End-to-end pipeline (production)

```
fit_params.json  →  equilibrium lineshape
        ↓
ssRF / AFP / unmanipulated trajectories  (SLURM arrays)
        ↓
combine_all_train.py  →  train_bin_0000.npz … train_bin_0499.npz
        ↓
single_bin.py × 500  →  binning_model_bin_*.pth
        ↓
combine_single_bin_models.py  →  combined_bin_model.pth
        ↓
test-binning.py  →  metrics and plots
```

On the Rivanna cluster, submit the SLURM scripts in `Data_Creation/rivanna/` and `ml/rivanna/` instead of running the Python workers by hand. See the subdirectory READMEs for exact commands.

## Other workflows

| Goal | Where to start |
|------|----------------|
| Full-spectrum P/Q prediction | [`ml/README.md`](ml/README.md) → `spectrum_pq.py` |
| Denoising autoencoder on Ps | [`ml/README.md`](ml/README.md) → `dae.py` |
| RL burn policy (DQN) | `python ml/dqn.py --episodes 100 --polarization 0.45` |
| Greedy incremental Q optimizer | `python ml/opt_q.py` |
| Legacy lookup-table for previous iteration of predicting P and Q (doesn't work in regions where I- > I+ initially) | [`Data_Creation/lookup_table.py`](Data_Creation/lookup_table.py) |

## Glossary

| Term | Meaning |
|------|---------|
| **Bin** | One frequency index in the 500-point spectrum (bins 0–499) |
| **Burn center** | The bin where RF is applied (`center_bin`) |
| **I+ / I−** | Intensity from +1↔0 and 0↔−1 transitions at each bin |
| **Ps** | Scalar signal: `Ps = I+ + I−` |
| **P** | Vector polarization (per-bin: `I+ + I−`; integrated over the spectrum for totals) |
| **Q** | Tensor polarization: `Q = I+ − I−` (primary optimization target for burns) |
| **p0** | Initial polarization before any manipulation |
| **γ_rf** (`gamma_rf`) | ssRF burn strength / applied RF power |
| **n_steps** | Number of integration steps in a burn trajectory |
| **source** | Event type in training data: `0` = ssRF, `1` = AFP, `2` = unmanipulated |
| **Shard** | Partial NPZ from one SLURM array task (one burn-center bin) |
| **RPE** | Relative percent error — common evaluation metric |

## Notes

- Large artifacts (`*.npz`, `*.pth`, `results/`, `data/`) are gitignored. Generate them locally or on the cluster.
- **`Data_Creation/rivanna/`** and **`ml/rivanna/`** are the production paths.
