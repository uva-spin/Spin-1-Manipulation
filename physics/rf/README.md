# physics.rf

Realtime spin-1 rate-equation model for ssRF burns, AFP, recovery, and diffusion on a 500-bin Pake powder lineshape.

## What it simulates

Given initial branch intensities **I+** and **I−** across 500 frequency bins, the model steps forward in time while RF is applied at a chosen burn center. Each step updates internal spin populations, then converts back to physical intensities.

With DNP disabled (the usual case in this repo), RF is the main mechanism that changes vector and tensor polarization.

## Key types

### `Spin1Params` (`model.py`)

Configuration dataclass. Important fields:

| Field | Typical value | Meaning |
|-------|---------------|---------|
| `n_bins` | 500 | Number of frequency bins |
| `r_min`, `r_max` | −3, 3 | Burn window in normalized frequency |
| `p0` | 0.45 | Initial vector polarization |
| `dt` | 0.015 | Integration timestep |
| `steps` | 20 | Steps per burn macro-step |
| `gamma_rf` | varies | RF burn strength |
| `dnp_enabled` | False | Dynamic nuclear polarization reservoir |

### `Spin1Model` (`model.py`)

The simulation object. Typical usage:

```python
from physics.rf import (
    Spin1Model,
    Spin1Params,
    build_model_for_intensities,
    configure_single_bin_ssrf,
)

params = Spin1Params(n_bins=500, p0=0.45, gamma_rf=10.0, steps=20)
model = build_model_for_intensities(iplus, iminus, params=params, p0=0.45)
configure_single_bin_ssrf(model, bin_idx=172, gamma_rf=10.0)

for _ in range(params.steps):
    model.step_once(dt=params.dt, rf_on=True, dnp_on=False)

iplus_new, iminus_new, _ = model.physical_intensities()
```

## Helper module: `rate_equations_realtime.py`

| Function | Purpose |
|----------|---------|
| `build_model_for_intensities` | Create a `Spin1Model` from I+/I− arrays |
| `configure_single_bin_ssrf` | Point RF at one bin with a given `gamma_rf` |
| `configure_physical_voigt_ssrf` | Install a physical-R Voigt RF profile |
| `burn_preserves_ps_sign` | Check that a burn step did not flip Ps sign |
| `burn_preserves_branch_order` | Check that I+ ≥ I− ordering is preserved |

These guards are used by `ml/opt_q.py` when searching for valid burn parameters.

## Tests

```bash
cd physics/rf
pytest -q
```

`pytest.ini` sets `pythonpath = ../..` so imports resolve from the repo root.

## Used by

- `Data_Creation/rivanna/` — training data generation (`from physics.rf import …`)
- `ml/dqn.py`, `ml/sarsa.py` — reinforcement-learning burn policies
- `ml/opt_q.py` — greedy incremental Q optimization
