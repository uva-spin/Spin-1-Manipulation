# physics.rf

Realtime spin-1 rate equations for ssRF burns, AFP, recovery, and diffusion on a Pake powder lineshape.

`Data_Creation/spectrum_gen/` uses this package to write training spectra. `ml/dqn.py`, `ml/sarsa.py`, and `ml/opt_q.py` step the same model while they search for a burn policy.

## What a step does

You start from branch intensities I+ and I−. Each call to `step_once` updates internal spin populations while RF is on or off, then you read intensities back with `physical_intensities()`. With `dnp_enabled=False` (the default), RF is what changes vector and tensor polarization during a burn.

## `Spin1Params` (`model.py`)

| Field | Default | Meaning |
|-------|---------|---------|
| `n_bins` | 701 | Frequency bins |
| `r_min`, `r_max` | −3, 3 | Frequency window |
| `p0` | 0.45 | Initial vector polarization |
| `dt` | 0.0015 | Integration timestep |
| `steps` | 50 | Steps per burn macro-step |
| `gamma_rf` | 2.0 | RF burn strength |
| `dnp_enabled` | False | Dynamic nuclear polarization reservoir |

Training NPZ files from `Data_Creation` use 500 bins on R ∈ [−6, 6]. Pass `n_bins`, `r_min`, and `r_max` when you want that grid. The snippet below does.

## Typical call

```python
from physics.rf import (
    Spin1Model,
    Spin1Params,
    build_model_for_intensities,
    configure_single_bin_ssrf,
)

params = Spin1Params(n_bins=500, r_min=-6.0, r_max=6.0, p0=0.45, gamma_rf=10.0, steps=20)
model = build_model_for_intensities(iplus, iminus, params=params, p0=0.45)
configure_single_bin_ssrf(model, bin_idx=172, gamma_rf=10.0)

for _ in range(params.steps):
    model.step_once(dt=params.dt, rf_on=True, dnp_on=False)

iplus_new, iminus_new, _ = model.physical_intensities()
```

## `rate_equations_realtime.py`

| Function | Purpose |
|----------|---------|
| `build_model_for_intensities` | Build a `Spin1Model` from I+ and I− arrays |
| `configure_single_bin_ssrf` | Aim RF at one bin |
| `configure_physical_voigt_ssrf` | Install a physical-R Voigt RF profile |
| `burn_preserves_ps_sign` | True when a burn step did not flip the sign of Ps |
| `burn_preserves_branch_order` | True when I+ stays at or above I− |

`ml/opt_q.py` uses the two checks while it searches burn parameters.

## Tests

```bash
cd physics/rf
pytest -q
```
