# Rivanna Data Pipeline

ssRF / AFP / unmanipulated training data from a Dulya-fitted equilibrium lineshape. Physics comes from [`physics.rf`](../../physics/rf/README.md) at the repo root — submit from this directory with the repo on `PYTHONPATH` (the SLURM scripts do that).

## What to run

| Step | Local | Cluster |
|------|--------|---------|
| Unmanipulated rows | `python generate_bins.py --mode unmanipulated` | `sbatch unmanipulated_bin_array.slurm` |
| ssRF shards (one burn-center bin) | `python generate_bins.py --mode ssrf --bin-idx 208` | `sbatch ssrf_traj_array.slurm` |
| AFP shards | `python generate_bins.py --mode afp --bin-idx 208` | `sbatch afp_traj_array.slurm` |
| Merge → `train_bin_XXXX.npz` | `python combine_all_train.py --strict` | `sbatch combine_all_train.slurm` |

Smoke test (one bin, coarse P grid):

```bash
python -m pytest tests/test_smoke.py -q
python generate_bins.py --mode all --smoke --bin-idx 208
python combine_all_train.py --strict
```

Optional: `plot_physics_demo.py` (diagnostic PNGs) and `create_sample_single_bin_data.py` (tiny train/test set).

## Layout

| Path | Role |
|------|------|
| `fit_params.json` | Frozen Dulya lineshape fit |
| `model_bridge.py` | Dulya equilibrium → `physics.rf` `Spin1Model` |
| `ssrf_bin_traj.py` / `afp_bin_traj.py` / `unmanipulated_bin_lineshape.py` | Workers |
| `combine_all_train.py` | Merge shards → per-bin training NPZs |
| `generate_bins.py` | Local dispatcher for the three workers |
| `*.slurm` | Cluster array + combine jobs |

## Data directories

```
data/
├── ssrf_shards/          One NPZ per burn-center bin
├── afp_shards/
├── unmanip_train/
├── combined_train_all/   train_bin_XXXX.npz  ← ML training
└── plots/                Optional diagnostics
```

## Training NPZ schema

Each `combined_train_all/train_bin_XXXX.npz` row is one event observed at bin XXXX:

| Field | Description |
|-------|-------------|
| `p0` | Initial vector polarization |
| `gamma_rf` | ssRF burn strength (0 for AFP / unmanipulated) |
| `n_steps` | Integration steps in the burn |
| `center_bin` | Burn-center bin |
| `source` | `0` ssRF, `1` AFP, `2` unmanipulated |
| `ps`, `iplus`, `iminus` | Intensities at this bin |
| `P`, `Q` | CC-calibrated per-bin targets |
| `is_mirror` | Mirror partner of a burn |

## Cluster notes

Submit from this directory. Scripts assume Rivanna-style `apptainer` with `module load apptainer pytorch/2.9.0`. The container entrypoint is already Python — pass scripts directly (`apptainer run … ssrf_bin_traj.py`), do not prefix with `python`. Account: `spinquest_standard`. Override paths with `DATA_DIR`, `SHARD_DIR`, `P_MIN`, `P_MAX`. Array jobs bind the repo root and set `PYTHONPATH` so `physics.rf` imports.

## Constants (`common.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `NUM_BINS` | 500 | Spectrum size |
| `F_MIN`, `F_MAX` | −6, 6 | Frequency range |
| `BURN_R_MIN`, `BURN_R_MAX` | −3, 3 | Valid burn-center window |
| `P_MIN`, `P_MAX`, `P_STEP` | −0.9, 0.9, 0.05 | Polarization grid |

Next: train per-bin models — [`ml/rivanna/README.md`](../../ml/rivanna/README.md).
