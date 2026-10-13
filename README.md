# Spin-1-Manipulation

Simulations and machine-learning tools for manipulating **spin-1** (deuteron) solid-state NMR targets:

- **ssRF** (semi-saturated radiofrequency) — local RF burns
- **AFP** (adiabatic fast passage) — adiabatic sweeps

## Repository layout

| Directory | What it does |
|-----------|--------------|
| [`physics/`](physics/README.md) | NMR lineshape math and the realtime spin-1 rate-equation model |
| [`Data_Creation/`](Data_Creation/README.md) | Full-spectrum training data (`create_data.py` + `spectrum_gen/`) |
| [`ml/`](ml/README.md) | Full-spectrum models, RL burn policies, Seq2Seq models |
| [`analysis/`](analysis/README.md) | Small visualization utilities (optional) |

## Quick start

```bash
pip install numpy scipy matplotlib pandas torch tqdm pytest
```

Python **3.10+**; PyTorch CPU is fine for small tests.

```bash
# Physics
cd physics/rf && pytest -q

# Full-spectrum sample data
python Data_Creation/create_data.py --quick
python -m pytest Data_Creation/spectrum_gen/tests -q
```

Edit **USER PARAMETERS** in [`Data_Creation/create_data.py`](Data_Creation/create_data.py) (scalar / `(lo,hi)` / list RF knobs). Output: `Data_Creation/spectra_data/spectra.npz`.

## Other workflows

| Goal | Where to start |
|------|----------------|
| Full-spectrum P/Q prediction | [`ml/README.md`](ml/README.md) → `spectrum_pq.py` |
| Denoising autoencoder on Ps | [`ml/README.md`](ml/README.md) → `dae.py` |
| RL burn policy (DQN) | `python ml/dqn.py --episodes 100 --polarization 0.45` |
| Greedy incremental Q optimizer | `python ml/opt_q.py` |

## Glossary

| Term | Meaning |
|------|---------|
| **Bin** | One frequency index in the 500-point spectrum |
| **I+ / I−** | Intensities from +1↔0 and 0↔−1 transitions |
| **P / Q** | Vector / tensor polarization |
| **p0** | Initial polarization |
| **γ_rf** | ssRF burn strength |
| **source** | `0` ssRF, `1` AFP, `2` unmanipulated, `3` profile, `4` AFP Profile, `5` combined |
