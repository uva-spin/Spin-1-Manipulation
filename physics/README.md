# Physics

Lineshape helpers and the spin-1 rate-equation model. Training spectra are produced by [`Data_Creation/`](../Data_Creation/README.md), which calls into `physics.rf`. The reinforcement-learning scripts `ml/dqn.py`, `ml/sarsa.py`, and `ml/opt_q.py` call it directly.

```
physics/
├── lineshape/     Analytic Pake powder lineshape
└── rf/            Realtime rate-equation package (see rf/README.md)
```

## Spectrum quantities

A deuteron spectrum is a set of frequency bins. Each bin has two branch intensities:

- **I+** from the +1 ↔ 0 transition
- **I−** from the 0 ↔ −1 transition

```
Ps = I+ + I−
Q  = I+ − I−     at one bin
P  = integrated signal over the spectrum
```

ssRF burns and AFP sweeps change I+ and I−. The realtime model steps spin populations under RF, recovery, and optional diffusion.

`Data_Creation` saves **500 bins** on R ∈ [−6, 6]. `Spin1Params` in `physics/rf` defaults to **701 bins** on R ∈ [−3, 3]. Those are different grids. The spectrum→P/Q trainers use the 500-bin NPZ.

## `lineshape/Lineshape.py`

Unburned powder lineshape at polarization `P`:

```python
from physics.lineshape.Lineshape import GenerateVectorLineshape

Ps, Iplus, Iminus = GenerateVectorLineshape(0.45, frequency_grid)
```

## `rf/`

See [`rf/README.md`](rf/README.md) for `Spin1Model`, `Spin1Params`, and the burn helpers.

```bash
cd physics/rf
pytest -q
```

`pytest.ini` sets `pythonpath = ../..` so imports resolve from the repo root.
