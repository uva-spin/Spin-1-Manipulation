# Voigt burn-profile update

## Preserved

- Equal-width physical-`R` population grid.
- Analytic broadened Pake-doublet spectrum.
- Signed initial vector polarization and optional tensor initialization.
- Common RF power for the two overlapping absorption branches.
- Full direct/mirror population bookkeeping.
- Population-dependent packet-pair spin diffusion.
- DNP source, saturation control, optional `T1`, and two-plot PyQt GUI.

## Replaced

The ideal one-bin RF selector was replaced by a bin-integrated Voigt transition-rate field.  The same physical profile is sampled by the `+ <-> 0` and `0 <-> -` transitions at their respective physical frequencies.

## Added

- Gaussian RF FWHM control.
- Lorentzian RF FWHM control.
- Strongest-bin and continuous-peak normalization modes.
- Exact SciPy Voigt evaluation with a pseudo-Voigt fallback.
- Gauss-Legendre integration over finite simulation bins.
- RF-profile overlay on the top GUI plot.
- Live profile-width and active-bin diagnostics.
- Tests for full three-level RF dynamics, broad-profile mirror behavior, center-versus-wing saturation, and grid convergence.

## Limiting compatibility

Setting both RF widths to zero restores the previous one-bin operator exactly.  The previous stored RF/DNP trajectory remains a regression test.

## Important interpretation

The Voigt function is an **applied rate profile**, not a hole shape imposed on the signal.  The resulting hole and mirror response evolve from the populations and can depart from Voigt shape at finite burn time or when diffusion, DNP, and `T1` compete with RF.
