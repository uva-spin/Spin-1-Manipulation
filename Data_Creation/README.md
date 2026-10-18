# Data Creation

Full-spectrum labeled NMR spectra for DAE / spectrum training.

## Quick start

```bash
python Data_Creation/create_data.py --quick
python Data_Creation/create_data.py --ssrf --unmanipulated
python Data_Creation/create_data.py --ssrf-afp-combined --no-ssrf --no-afp --max-burn-steps 50 --max-relax-steps 50
```

Edit **USER PARAMETERS** at the top of [`create_data.py`](create_data.py) for grids and flexible RF knobs:

- scalar → fixed value
- `(lo, hi)` → sample once per burn bin (constant through that burn)
- `[...]` → iterate (Cartesian product of list knobs)

Helpers live in [`spectrum_gen/`](spectrum_gen/).

Output: `Data_Creation/spectra_data/spectra.npz` with shape `(N, 2, 500)` (I+ / I−).  
Sources: `0=ssRF`, `1=AFP`, `2=unmanipulated`, `3=optimal profile`, `4=AFP Profile`, `5=ssRF+AFP combined`.

```bash
python -m pytest Data_Creation/spectrum_gen/tests -q
```
