# Data Creation

Scripts that generate labeled NMR spectra for machine-learning training and evaluation.

## Two paths

| Path | When to use |
|------|-------------|
| **[`rivanna/`](rivanna/README.md)** | **Start here.** Self-contained, cluster-ready pipeline. Generates `train_bin_XXXX.npz` files for per-bin MLP training. |
| Scripts in this directory (root) | Legacy and experimental. Some require modules that are no longer in the repo. |

## Primary pipeline (`rivanna/`)

The rivanna package simulates ssRF burns, AFP flips, and unmanipulated equilibrium lineshapes, then merges them into per-bin training files.

```bash
cd Data_Creation/rivanna

# Local smoke test
python -m pytest tests/test_smoke.py -q
python generate_bins.py --mode all --smoke --bin-idx 208
python combine_all_train.py --strict
```

Output: `data/combined_train_all/train_bin_0000.npz` through `train_bin_0499.npz`.

See [`rivanna/README.md`](rivanna/README.md) for the full cluster workflow and NPZ schema.

## Other scripts in this directory

### Full-spectrum data (for `spectrum_pq.py` / `dae.py`)

```bash
python Data_Creation/create_dae_voigt_burn_spectra.py --quick
```

Writes `Data_Creation/dae_voigt_burn_spectra/spectra.npz` with shape `(N, 2, 500)` for I+/I−.

### Test / sample events

```bash
python Data_Creation/create_sample_manipulation_events.py --quick
python Data_Creation/generate_vector_lineshape_test_data.py
```

These create small NPZ files for debugging model I/O.

### Legacy lookup-table pipeline

| Script | Purpose |
|--------|---------|
| `lookup_table.py` | Build polarization → lineshape lookup table |
| `burn_lookup_table.py` | Burn-augmented lookup via realtime dynamics |
| `ssRFData.py`, `ssRFData_mc.py` | MC training data via lookup mapper |

These depend on `physics/afp.py` and related legacy modules. Use the rivanna pipeline unless you specifically need the lookup-table approach.

## Data flow (rivanna)

```
fit_params.json
    → equilibrium lineshape (Dulya kernel)
    → ssRF shards  (one SLURM task per burn-center bin)
    → AFP shards
    → unmanipulated bin rows
    → combine_all_train.py
    → train_bin_XXXX.npz  (one file per observation bin)
```

Each `train_bin_XXXX.npz` row describes one simulated event seen **at spectral bin XXXX**, with fields like `p0`, `gamma_rf`, `n_steps`, `ps`, `iplus`, `iminus`, `P`, `Q`, and `source`.

## Where the data goes next

Per-bin NPZ files feed into [`ml/rivanna/single_bin.py`](../ml/rivanna/README.md). Full-spectrum NPZ files feed into [`ml/spectrum_pq.py`](../ml/README.md) or [`ml/dae.py`](../ml/README.md).
