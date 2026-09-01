# Rivanna Data Pipeline

Self-contained package for generating spin-1 ssRF / AFP training data from a Dulya-fitted equilibrium lineshape. You can copy this entire folder to a cluster and run everything from inside it — no parent-repo imports are required.

## Layout

| Path | Role |
|------|------|
| `fit_params.json` | Frozen Dulya lineshape fit parameters |
| `dulya_kernel.py`, `lineshape.py` | Equilibrium lineshape from the Dulya fit |
| `ssrf_realtime/` | Vendored copy of the physics rate-equation model |
| `common.py` | Shared constants (500 bins, polarization grid, burn window, paths) |
| `model_bridge.py` | Bridge Dulya equilibrium → `Spin1Model` burns |
| `ssrf_bin_traj.py` | ssRF trajectory worker (one burn-center bin) |
| `afp_bin_traj.py` | AFP flip + relaxation worker |
| `unmanipulated_bin_lineshape.py` | Unburned equilibrium rows per bin |
| `combine_all_train.py` | Merge all sources → `train_bin_XXXX.npz` |
| `generate_bins.py` | Dispatcher for all three modes (local runs) |
| `*.slurm` | SLURM array and combine jobs |

## Data directories

After generation, data lives under `data/`:

```
data/
├── ssrf_shards/          Raw ssRF trajectory shards (one per burn-center bin)
├── afp_shards/           Raw AFP trajectory shards
├── unmanip_train/        Unmanipulated equilibrium rows
├── combined_train_all/   Final per-bin training NPZs  ← used by ML training
└── plots/                Optional diagnostic plots
```

## Training NPZ schema

Each `combined_train_all/train_bin_XXXX.npz` contains one row per simulated event **as observed at bin XXXX**:

| Field | Type | Description |
|-------|------|-------------|
| `p0` | float | Initial vector polarization |
| `gamma_rf` | float | ssRF burn strength (0 for unmanipulated / AFP rows) |
| `n_steps` | int | Integration steps applied in the burn |
| `center_bin` | int | Burn-center bin index |
| `source` | int | `0` = ssRF, `1` = AFP, `2` = unmanipulated |
| `ps` | float | Scalar signal `I+ + I−` at this bin |
| `iplus`, `iminus` | float | Branch intensities at this bin |
| `P`, `Q` | float | CC-calibrated per-bin polarization targets |
| `is_mirror` | bool | Whether this row is the mirror partner of a burn |

## Local usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Smoke test:

```bash
cd Data_Creation/rivanna
python -m pytest tests/test_smoke.py -q
```

Generate data for one bin (fast local test):

```bash
python generate_bins.py --mode all --smoke --bin-idx 208
```

Generate one mode at a time:

```bash
python generate_bins.py --mode unmanipulated
python generate_bins.py --mode ssrf --bin-idx 172
python generate_bins.py --mode afp --bin-idx 172
```

Combine shards into training files:

```bash
python combine_all_train.py --strict
```

## Cluster usage (Rivanna)

Submit from this directory:

```bash
cd /path/to/Data_Creation/rivanna

sbatch unmanipulated_bin_array.slurm
sbatch ssrf_traj_array.slurm
sbatch afp_traj_array.slurm
sbatch combine_all_train.slurm
```

The ssRF and AFP jobs are SLURM arrays — one task per burn-center bin in the window R ∈ (−3, 3). After all shards finish, `combine_all_train.slurm` merges them.

### SLURM notes

- Scripts assume Rivanna-style `apptainer` with `module load apptainer pytorch/2.9.0`.
- The container entrypoint is already Python — pass scripts directly:
  `apptainer run … ssrf_bin_traj.py` (do **not** prefix with `python`).
- Account: `spinquest_standard`. Override paths with environment variables like `DATA_DIR`, `SHARD_DIR`, `P_MIN`, `P_MAX`.

## Constants (from `common.py`)

| Constant | Value | Meaning |
|----------|-------|---------|
| `NUM_BINS` | 500 | Spectrum size |
| `F_MIN`, `F_MAX` | −6, 6 | Full frequency range |
| `BURN_R_MIN`, `BURN_R_MAX` | −3, 3 | Valid burn-center window |
| `P_MIN`, `P_MAX`, `P_STEP` | −0.9, 0.9, 0.05 | Polarization grid |

## Next step

Train per-bin models on the combined NPZ files. See [`ml/rivanna/README.md`](../../ml/rivanna/README.md).
