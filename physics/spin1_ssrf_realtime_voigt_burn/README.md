# Spin-1 real-time ss-RF simulator with a Voigt burn-rate profile

This package extends the previously validated physical-`R`-bin simulator by replacing the ideal one-bin RF selector with an **adjustable Voigt transition-rate profile**.  The established population state, analytic Pake-doublet lineshape, DNP source, population-dependent diffusion kernel, direct/mirror population bookkeeping, and two-plot PyQt interface are retained.

The key modeling rule is:

```text
The Voigt function is the spatially sampled RF rate field.
It is not a hole subtracted from the displayed spectrum.
```

Every bin evolves through the full three-level population rate equations.  The direct holes, mirror responses, finite-power saturation, diffusion refill, DNP competition, and long-time line shape therefore emerge from the populations.

## RF-rate model

For packet `k`, the `+ <-> 0` transition is observed at physical frequency `R = x[k]`, while the same packet's `0 <-> -` transition is observed at `R = -x[k]`.  A single RF source centered at `R_b` produces one common physical profile

```text
v(R - R_b; Gaussian FWHM, Lorentzian FWHM).
```

The continuous Voigt is integrated over every finite simulation bin:

```text
vbar[j] = (1/dR) integral_bin_j v(R-R_b) dR.
```

The per-packet equalization rates are

```text
Gamma_plus[k]  = Gamma_RF * w[k] * vbar(x[k]-R_b)
Gamma_minus[k] = Gamma_RF * w[k] * vbar(-x[k]-R_b)
```

where `w[k]` is the pre-existing optional Pake-density/coupling factor.  Both branches see the same physical RF field; they sample it at their own transition frequencies.

The RF contribution is calculated directly from all three populations:

```text
dn_plus/dt  = -Gamma_plus  (n_plus-n_zero)
dn_zero/dt  = +Gamma_plus  (n_plus-n_zero)
              -Gamma_minus (n_zero-n_minus)
dn_minus/dt = +Gamma_minus (n_zero-n_minus)
```

No direct-hole/mirror-area rule is used in the runtime model.  In an isolated narrow-bin RF test, the familiar one-half mirror relation appears algebraically as a limiting result of these equations.  With a finite Voigt profile, both transitions of a packet may be driven at once, and diffusion, DNP, and `T1` may act simultaneously, so the observed response is whatever the coupled ODEs predict.

## Voigt parameters

The GUI exposes:

| Control | Meaning |
|---|---|
| `physical R` | RF carrier center in dimensionless physical `R`. |
| `common Gamma_RF` | Common center equalization-rate scale. |
| `Gaussian FWHM dR` | Inhomogeneous/generator-smearing part of the RF rate profile. |
| `Lorentzian FWHM dR` | Homogeneous/power-broadening part of the RF rate profile. |
| `Gamma_RF meaning` | Either normalize the strongest finite-bin average to one or retain the continuous profile's peak normalization. |

Two profile normalizations are available:

- **strongest bin** (`center_bin`, default): the maximum bin-averaged profile value is one.  `Gamma_RF` is therefore the strongest bin-averaged equalization scale and remains intuitive as bin count changes.
- **continuous peak** (`continuous_peak`): the underlying continuous Voigt has `V(0)=1`.  This is useful for grid-convergence and hardware-profile studies; the finite center-bin average may be below one if the profile is narrower than a bin.

Setting both RF FWHM values to zero restores the exact previous one-bin RF operator.  A stored regression test confirms that the old RF trajectory is reproduced to machine precision in this limit.

The top plot displays the normalized Voigt rate profile on a secondary axis.  This overlay is diagnostic only; the spectrum itself is reconstructed from the evolving populations.

## Do not confuse the three widths

The software now contains three independent spectral widths:

1. `line_gamma`: broadening in the static analytic Pake-doublet reference line;
2. `rf_gaussian_fwhm_R` and `rf_lorentzian_fwhm_R`: the applied ss-RF rate profile;
3. `zq_width_R`: the zero-quantum spectral-overlap width in the diffusion kernel.

They may be related by the apparatus and material, but they are not interchangeable parameters and should be calibrated separately.

## Finite-power behavior

The imposed **rate profile** remains Voigt, but the resulting hole generally does not.  For a shallow, RF-only burn, the initial change approximately follows the local Voigt rate multiplied by the pre-burn line intensity.  At longer times, the center saturates before the wings.  Diffusion and DNP further reshape the feature.  This is intentional and is a central advantage of applying the profile to the rate equations rather than subtracting a prescribed Voigt hole.

## Preserved population-dependent spin diffusion

The recovery operator remains the physically constrained packet-pair model from the previous working package.  For example,

```text
J_plus(i,j) = K(i,j) [p0(i) p+(j) - p+(i) p0(j)]
J_minus(i,j)= K(i,j) [p-(i) p0(j) - p0(i) p-(j)]
```

with packet capacity, zero-quantum spectral overlap, and optional orientation correlation in `K(i,j)`.  Internal diffusion conserves total vector polarization, while RF is a vector-polarization sink and DNP is an external source.

## DNP and T1

DNP still drives each packet toward the user-selected signed saturation polarization.  It may operate simultaneously with the finite-width RF profile and diffusion.  `T1` remains an optional independent relaxation toward `P_eq`.

## Recommended calibration sequence

1. Fit the unmanipulated analytic Pake line (`line_gamma`, asymmetry, display scale).
2. Turn diffusion, DNP, and `T1` off and measure shallow short burns at several RF centers.
3. Fit the Gaussian and Lorentzian RF widths to the initial spectral derivative or very short-time hole.
4. Calibrate `Gamma_RF` from the central short-time depletion rate.
5. Increase burn time to test power broadening and center saturation.
6. Re-enable diffusion and fit `K0` and `zq_width_R` to post-burn recovery.
7. Re-enable DNP and fit the DNP source and microwave-dependent diffusion factor.

The RF profile should ultimately be fitted to data obtained with the actual burn coil, matching network, generator, amplifier, and target material.

## Run the GUI

```bash
cd spin1_ssrf_realtime_voigt_burn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run_app.py
```

On Windows:

```text
.venv\Scripts\activate
```

The GUI tries PyQt6, PySide6, and PyQt5 in that order.

## Tests and headless validation

```bash
pip install -r requirements.txt
pytest -q
MPLBACKEND=Agg python3 examples/headless_demo.py
```

The test suite verifies:

- exact previous RF behavior when both Voigt widths are zero;
- one common physical profile for both absorption branches;
- finite-bin integration of the Voigt profile;
- packet-population conservation under RF;
- the complete three-level RF ODE;
- absence of a hard-coded mirror-response ratio for a broad profile;
- faster center depletion than wing depletion;
- grid convergence of the continuous-peak profile integral;
- all prior diffusion, DNP, signed-polarization, and conservation tests.

## Implementation notes

- Exact Voigt evaluation uses `scipy.special.voigt_profile`.
- A normalized pseudo-Voigt fallback is provided for environments without SciPy.
- Gauss-Legendre quadrature averages the profile over each finite `R` bin.
- Profile arrays are cached and automatically invalidated when center, widths, normalization, or grid settings change.
- The explicit Euler integrator still requires conservative choices of `dt` when `Gamma_RF`, diffusion, or DNP rates are large.
