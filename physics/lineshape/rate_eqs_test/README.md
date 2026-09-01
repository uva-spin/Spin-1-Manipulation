# rate_eqs_test

Research scripts for spin-1 rate-equation experiments: per-bin trajectory generation, gamma optimization, Dulya lineshape fitting, and data merging.

**For new work, prefer [`Data_Creation/rivanna/`](../../../Data_Creation/rivanna/README.md).** Much of the functionality here has been superseded by that pipeline, but these scripts remain for physics R&D.

## What's here

| Script | Purpose |
|--------|---------|
| `ssrf_bin_traj.py`, `afp_bin_traj.py` | Per-bin MC trajectories |
| `rate_eqs_test_ssrf_all_bins_gamma_opt*.py` | Find minimum `gamma_rf` to null local Q |
| `fit.py`, `fit_multisite.py`, `fit-dulya.py` | Fit experimental Dulya lineshapes |
| `combine_ssrf_afp_train.py` | Merge ssRF and AFP training shards |
| `ssrf_afp.py` | Interactive ssRF + AFP preview (edit constants at top) |

## SLURM arrays

Submit from this directory on the cluster:

```bash
sbatch ssrf_traj_array.slurm
sbatch afp_traj_array.slurm
sbatch gamma_opt_array.slurm
sbatch gamma_opt_combine.slurm
```

## Caveats

Some scripts import modules that may not be present in the current repo (for example `physics/afp.py`, `physics/Lineshape.py`). Check imports before running. The rivanna pipeline does not have these dependencies.

## Tests

Physics unit tests live in [`physics/ssrf_realtime/`](../ssrf_realtime/README.md):

```bash
cd physics/ssrf_realtime && pytest -q
```
