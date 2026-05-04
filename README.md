# Spin-1-Manipulation

Repository for generating spin-1 lineshape data, applying ssRF burn effects, and
training/evaluating compact ML models on those spectra.

## Project Layout

- `Data_Creation/`: data generation and burn simulation scripts
  - `lookup_table.py`: builds lookup table used for burn-to-(I+, I-) mapping
  - `ssRFMapper.py`: applies ssRF burn profile to `Ps`, `Iplus`, `Iminus`
  - `ssRFData.py`: produces burned training/testing datasets
- `physics/lineshape/`: physics-oriented reference implementations and experiments
- `ml/`: model training and inference scripts
  - `rivanna/`: per-bin SLURM-compatible training scripts
  - `binning_model.py`: combined bin model training/evaluation flow
  - `dae.py`: denoising autoencoder pipeline and evaluation plots
- `analysis/`: one-off analysis and visualization scripts
- `diagrams/`: supporting visual assets

## Typical Workflow

1. Build lookup data with `Data_Creation/lookup_table.py`.
2. Generate burned datasets with `Data_Creation/ssRFData.py`.
3. Train models from `ml/` (single-bin, combined, or DAE).
4. Use `analysis/` scripts for diagnostics and figures.

## Notes

- Scripts are primarily research/experiment oriented and may emit local result
  artifacts when run.
- Keep reusable core logic in mapper/model modules; keep plotting and one-off
  experiments in `analysis/` or `physics/`.