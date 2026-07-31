# dulya_fit_v2

Self-contained Dulya-fit MC data generation for spin-1 ssRF / AFP trajectories.

Copy this entire folder to a cluster; run everything from inside it. No parent-repo
imports are required (`dulya_fit/`, `physics/`, etc. are not used).

## Layout

| Path | Role |
|------|------|
| `fit_params.json` | Frozen Dulya lineshape parameters |
| `dulya_kernel.py` / `lineshape.py` | Equilibrium lineshape |
| `ssrf_realtime_v2/` | Vendored physics model |
| `ssrf_bin_traj.py` | ssRF discrete `(P, γ, n_steps)` worker |
| `afp_bin_traj.py` | AFP + relaxation worker |
| `unmanipulated_bin_lineshape.py` | Unburned equilibrium per bin |
| `*.slurm` | Array / combine jobs (submit from this dir) |

## Local smoke

```bash
cd dulya_fit_v2
python -m pytest tests/test_smoke.py -q
python generate_bins.py --mode all --smoke --bin-idx 208
```

## Cluster

```bash
cd /path/to/dulya_fit_v2
sbatch ssrf_traj_array.slurm
sbatch afp_traj_array.slurm
sbatch unmanipulated_bin_array.slurm
```

Python deps: see `requirements.txt` (numpy, scipy, matplotlib). SLURM scripts
assume Rivanna-style `apptainer` + `pytorch/2.9.0`; the container entrypoint is
already Python, so scripts are passed as `apptainer run … ssrf_bin_traj.py`
(do **not** prefix with `python`).
