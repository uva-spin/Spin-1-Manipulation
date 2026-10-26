# LabVIEW Dulya / ssRF Sim Node

Python API for a LabVIEW Python Node that simulates a Dulya lineshape and applies ssRF / AFP / optimal-profile manipulations in real time. State lives in **module globals** for the lifetime of the LabVIEW Python session.

## Setup

1. Point the LabVIEW Python Node (or session) at this folder, or ensure the repo root `Spin-1-Manipulation` is on `sys.path`.
2. Use a Python environment that can import `numpy`, `scipy`, and this repo’s `physics` / `Data_Creation` packages (e.g. the project conda env).
3. Optionally supply a Dulya fit JSON (`Data_Creation/fit_params.json`). Leave the path empty (default) to use the analytic Pake lineshape, same as `create_data.py` without `--fit-params`.

## Call order

1. **Once per session:** `InitSim(fit_params_path='', P0, Q0=-1, n_bins=500)`
2. **Loop / events:** `main(cmd, dt, burn_R, gamma_rf, n_steps, afp_mode, efficiency, rf_mode)`

`InitSim` and every `main` call return the same tuple for stable wiring:

```text
(freq, intensity, P, Q, t, status)
```

- `freq`, `intensity` — 1D arrays for an XY graph (`intensity = I+ + I-`)
- `P`, `Q`, `t` — scalars
- `status` — string (`"ok"` or an error message; on error the last good spectrum is returned when possible)

## Commands (`cmd`)

| cmd | Name | Effect |
|-----|------|--------|
| 0 | get_spectrum | Return current spectrum (no dynamics) |
| 1 | step | Advance recovery/diffusion: `n_steps` macro-steps with step size `dt` (if `dt<=0`, model `dt` is used); RF off |
| 2 | ssrf | Voigt (default) or single-bin burn at `burn_R` for `n_steps` |
| 3 | afp | Instantaneous AFP (`afp_mode`: 0 = all Q&lt;0, 1 = selective window around `burn_R`) |
| 4 | ssrf_profile | Design + apply optimal ssRF profile on current state (can take seconds) |
| 5 | afp_profile | AFP over the full Q&lt;0 profile |
| 6 | reset | Rebuild equilibrium from the last `InitSim` parameters |

### `InitSim` fit path

| `fit_params_path` | Equilibrium |
|-------------------|-------------|
| `''` / empty (default) | Analytic Pake (`create_data.py` default) |
| path to fit JSON | Frozen Dulya fit seated via `use_fit_params` |

### Useful arguments

| Arg | Used by | Notes |
|-----|---------|--------|
| `dt` | step | Macro-step size |
| `burn_R` | ssrf, afp (selective) | Physical R of burn / window center |
| `gamma_rf` | ssrf | Peak RF rate |
| `n_steps` | step, ssrf, ssrf_profile | Number of Euler macro-steps |
| `afp_mode` | afp | 0 = Q&lt;0, 1 = selective |
| `efficiency` | afp, afp_profile | AFP efficiency ∈ [0, 1] |
| `rf_mode` | ssrf | 0 = physical Voigt, 1 = single-bin |

Unused args for a given `cmd` may be left at defaults.

## Suggested LabVIEW wiring

1. **Init** frame: call `InitSim` with desired `P0`; leave `fit_params_path` empty unless you want a Dulya fit.
2. **Timed loop:** call `main(1, dt, …)` each tick for continuous recovery; wire `freq`/`intensity` to an XY Graph.
3. **Buttons / events:** call `main(2|3|4|5, …)` with burn/AFP parameters when the operator applies a manipulation.
4. Display `P`, `Q`, `t`, and `status` on indicators.

Persistence is in Python module globals (`model`, spectrum cache). You do **not** need to pass population arrays through LabVIEW Global Variables.

## Example (REPL)

```python
from ml.labview_sim.labview_sim import InitSim, main

# Default: analytic Pake (same as create_data without --fit-params)
InitSim('', 0.45)
# Optional Dulya fit:
# InitSim("Data_Creation/fit_params.json", 0.45)

freq, intensity, P, Q, t, status = main(2, burn_R=0.4, gamma_rf=2.0, n_steps=20)
freq, intensity, P, Q, t, status = main(1, dt=0.0015, n_steps=10)
```

## Relation to ML node

The existing `ml/labview_model` package (`InitModel` / `main` → P, Q inference) is separate. This package drives the **physics simulator**, not the LSTM.
