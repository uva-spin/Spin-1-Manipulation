# Physics

NMR lineshape calculations and the realtime spin-1 rate-equation model used throughout the repo.

## What's here

```
physics/
├── lineshape/           Analytic Pake powder lineshape, legacy mappers, R&D scripts
│   ├── Lineshape.py     Core equilibrium lineshape: GenerateVectorLineshape(P, f)
│   ├── ssRFMapper.py    Legacy lookup-table ssRF mapper
│   └── rate_eqs_test/   Research scripts (gamma optimization, fitting, SLURM arrays)
└── ssrf_realtime/       Primary realtime physics package (see its README)
```

## Core concepts

A deuteron (spin-1) spectrum is stored as **500 frequency bins**. At each bin the signal splits into two branch intensities:

- **I+** — contribution from the +1 ↔ 0 transition
- **I−** — contribution from the 0 ↔ −1 transition

From these:

```
Ps = I+ + I−        (scalar signal)
Q  = I+ − I−        (local tensor polarization at this bin)
P  = integrated Ps  (vector polarization over the spectrum)
```

Manipulations (ssRF burns, AFP flips) change I+ and I− over time. The realtime model integrates spin-population rate equations under RF, recovery, and optional diffusion.

## `ssrf_realtime/` — main simulation package

This is the physics engine used by `Data_Creation/rivanna/` and several ML scripts (`dqn.py`, `sarsa.py`, `opt_q.py`).

Key entry points:

| Module | Purpose |
|--------|---------|
| `model.py` | `Spin1Model`, `Spin1Params` — integrate burns step-by-step |
| `rate_equations_realtime.py` | Build models from intensities, configure single-bin ssRF |
| `voigt_burn_physics.py` | Voigt RF profile in physical R-space |

See [`ssrf_realtime/README.md`](ssrf_realtime/README.md) for parameters and usage.

### Run tests

```bash
cd physics/ssrf_realtime
pytest -q
```

## `lineshape/Lineshape.py` — equilibrium spectrum

Generate an unburned powder lineshape at polarization `P` over frequency grid `f`:

```python
from physics.lineshape.Lineshape import GenerateVectorLineshape

Ps, Iplus, Iminus = GenerateVectorLineshape(0.45, frequency_grid)
```

Used by `ml/opt_q.py` and several data-generation scripts.

## `lineshape/rate_eqs_test/` — research scripts

Older R&D code for per-bin trajectory generation, gamma optimization, and fitting experimental Dulya lineshapes. Much of this functionality has been superseded by `Data_Creation/rivanna/`, but these scripts remain useful for physics experiments.

Contains its own SLURM arrays (`ssrf_traj_array.slurm`, `gamma_opt_array.slurm`, etc.). Run from that directory on the cluster.

**Note:** Some scripts here import modules that may not exist in the current tree (for example `physics/afp.py`). Check imports before running.

## Relationship to data generation

`Data_Creation/rivanna/` vendors a copy of `ssrf_realtime/` so the cluster pipeline is self-contained. Changes to the physics model should be made in `physics/ssrf_realtime/` and then copied or synced to `Data_Creation/rivanna/ssrf_realtime/` if the pipelines should stay aligned.
