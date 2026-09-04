# Data Creation

Labeled NMR spectra for machine-learning training. **Start in [`rivanna/`](rivanna/README.md).**

## Per-bin training data (main path)

This is what [`ml/rivanna/single_bin.py`](../ml/rivanna/README.md) trains on.

```bash
cd Data_Creation/rivanna
python -m pytest tests/test_smoke.py -q
python generate_bins.py --mode all --smoke --bin-idx 208
python combine_all_train.py --strict
```

Output: `rivanna/data/combined_train_all/train_bin_0000.npz` … `train_bin_0499.npz`.

On the cluster, submit from `rivanna/`:

```bash
sbatch unmanipulated_bin_array.slurm
sbatch ssrf_traj_array.slurm
sbatch afp_traj_array.slurm
sbatch combine_all_train.slurm
```

```
fit_params.json
    → equilibrium lineshape
    → ssRF / AFP / unmanipulated shards
    → combine_all_train.py
    → train_bin_XXXX.npz
```

Each row is one simulated event **as seen at spectral bin XXXX** (`p0`, `gamma_rf`, `n_steps`, `ps`, `iplus`, `iminus`, `P`, `Q`, `source`).

## Full-spectrum data (optional)

For [`ml/spectrum_pq.py`](../ml/README.md) and [`ml/dae.py`](../ml/README.md):

```bash
python Data_Creation/create_dae_voigt_burn_spectra.py --quick
```

Writes `Data_Creation/dae_voigt_burn_spectra/spectra.npz` with shape `(N, 2, 500)` (I+ / I−).

## Tabular Q-learning lookup (optional)

[`lookup_table.py`](lookup_table.py) builds `lookup_table.pkl` for [`ml/q-learning.py`](../ml/q-learning.py). Not used by the per-bin pipeline.
