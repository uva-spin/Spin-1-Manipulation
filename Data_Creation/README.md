# Data creation

`create_data.py` writes labeled full-spectrum NMR events for the models in [`ml/`](../ml/README.md). Helpers live in [`spectrum_gen/`](spectrum_gen/).

Run the commands below from the repo root.

## Small file

```bash
python Data_Creation/create_data.py --quick --no-plots
```

Output: `Data_Creation/spectra_data/spectra.npz`, shape `(N, 2, 500)` for I+ and I−. Omit `--no-plots` to also write example figures under `Data_Creation/spectra_data/plots/`.

`--quick` fixes the grid: P ∈ {0.30, 0.50}, three centers per P, burn and relax steps `0, 5, …, 100`. Modes are ssRF, per-bin AFP, and the optimal profile.

Copy that file before training `lstm.py` or `gru.py`. Those two scripts only read `ml/data/spectra.npz`:

```bash
mkdir -p ml/data
cp Data_Creation/spectra_data/spectra.npz ml/data/spectra.npz
cd ml && python lstm.py
```

## Modes

If you pass no mode flag, the script generates **ssRF only**. If you pass any mode flag, every mode you did not name stays off. `--quick` is the exception above, and it only applies when you pass no mode flag.

| Flag | What it adds |
|------|----------------|
| `--ssrf` / `--no-ssrf` | Selective ssRF burns |
| `--afp` / `--no-afp` | Per-bin AFP |
| `--afp-relax` | Relaxation steps after AFP. Implies `--afp`. `--no-afp --afp-relax` is an error |
| `--afp-profile` | One Q<0 AFP-profile sweep, plus relaxation |
| `--profile` / `--no-profile` | Optimal RF profile |
| `--unmanipulated` / `--no-unmanipulated` | Equilibrium spectra, no RF |
| `--ssrf-afp-combined` | Events that apply ssRF and AFP in one trajectory. Not implied by `--ssrf --afp` |

`--combined-max-centers` and `--combined-max-relax-steps` (default 500) cap the combined mode. Combined relax steps are also capped by `--max-relax-steps`.

## Grids and RF knobs

Polarization and step counts are CLI flags. Defaults are the constants at the top of `create_data.py`:

| Flag | Default |
|------|---------|
| `--p-min`, `--p-max`, `--p-step` | 0.2, 0.6, 0.025 |
| `--unmanip-p-step` | 0.00005 |
| `--min-burn-steps`, `--max-burn-steps`, `--burn-steps-step` | 0, 400, 1 |
| `--min-relax-steps`, `--max-relax-steps`, `--relax-steps-step` | 0, 8000, 1 |

The default burn and relax grids are large. For a first training set, raise the step and lower the max, as in the root README.

`GAMMA_RF`, `AFP_WINDOW`, `GAUSSIAN_FWHM_R`, `LORENTZIAN_FWHM_R`, and `DT` are edited in the USER PARAMETERS block:

- a number is fixed
- `(lo, hi)` is sampled once per selective ssRF burn and held for that burn
- a list is iterated (Cartesian product of the list knobs)

`--gamma-rf`, `--afp-window`, and `--dt` override those three knobs with a single number. Optimal-profile ssRF uses its designed per-bin rates and ignores `GAMMA_RF`.

`--afp-bin-stride N` keeps every Nth Q<0 bin in AFP-only events. `0` keeps every affected bin.

Other flags: `--output`, `--plot-dir`, `--seed` (default 42).

## Equilibrium lineshape

The default equilibrium is the analytic Pake spectrum. A frozen Dulya fit replaces it for every equilibrium spectrum:

```bash
python Data_Creation/create_data.py \
  --fit-params Data_Creation/fit_params.json \
  --unmanipulated --no-plots
```

`FIT_PARAMS` at the top of `create_data.py` is the same switch when you leave `--fit-params` unset.

## NPZ fields

| Field | Contents |
|-------|----------|
| `spectra` | `(N, 2, 500)` I+ / I− |
| `P_total`, `Q_total` | Target polarizations |
| `power_profile` | `(N, 500)` per-bin RF envelope |
| `rf_dose` | `(N, 500)` cumulative per-bin RF dose (∫ γ dt, excluding the AFP envelope) |
| `applied_power` | Peak γ for the event |
| `n_steps` | Step count saved for this frame |
| `burn_steps`, `relax_steps` | Burn-phase and relax-phase counts |
| `run_id` | Trajectory id. Trainers split on this |
| `p0`, `center_bin`, `source` | Initial polarization, burn center, manipulation code |
| `combo_scenario`, `combo_layout` | Combined-mode labels. Unused modes store a sentinel |
| `meta_json` | JSON string of the run settings and source counts |

`source` codes: `0` ssRF, `1` AFP, `2` unmanipulated, `3` optimal profile, `4` AFP profile, `5` ssRF+AFP combined.

## Tests

```bash
python -m pytest Data_Creation/spectrum_gen/tests -q
```
