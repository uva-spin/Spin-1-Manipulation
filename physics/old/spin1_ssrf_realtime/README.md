# Spin-1 ss-RF real-time simulation package, v11

This version keeps the v9 realistic analytic Pake-doublet lineshape and the ideal-bin ss-RF population bookkeeping, then adds signed initial polarization and optional DNP build dynamics.

## Main v11 changes

- Added a prominent live readout field for the total vector polarization `P(t)`, displayed both as a dimensionless value and a percent.
- Reduced the default GUI height and moved the controls into a scrollable side panel so the two-plot window fits better on a laptop screen.
- Initial vector polarization `P0` can be set anywhere in the signed range, for example `+0.10`, `+0.58`, or `-0.45`.
- The displayed spectrum is signed: negative vector polarization produces an inverted signed NMR absorption signal.
- Added a DNP ON/OFF button.
- Added a DNP saturation setting `P_DNP_sat` and finite DNP build rate `dnp_rate`.
- DNP drives the line toward the selected saturation polarization and does not maximize beyond that setting.
- With DNP off, ss-RF equalization removes vector polarization. Spin diffusion/recovery then redistributes the remaining area and relaxes the line toward the Boltzmann-shaped state for the current reduced `P(t)`, not the original pre-burn line.
- The lower plot still shows only the two direct burn-location intensities:

  \[
  I^+(R_{\rm RF},t), \qquad I^-(R_{\rm RF},t).
  \]

## Run the GUI

```bash
cd spin1_ssrf_realtime_sim_v11
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 run_app.py
```

The GUI attempts PyQt6, then PySide6, then PyQt5.

## Run the headless demo and tests

```bash
cd spin1_ssrf_realtime_sim_v11
pip install numpy matplotlib pytest
MPLBACKEND=Agg python3 examples/headless_demo.py
pytest -q
```

The demo writes:

- `outputs/v11_two_dynamic_plots_no_dnp.png`
- `outputs/v11_no_dnp_total_area_loss.png`
- `outputs/v11_dnp_build_to_saturation.png`
- `outputs/v11_negative_initial_polarization.png`
- `outputs/v11_lineshape_validation_against_plot_signal.png`
- `outputs/v11_no_dnp_burn_recovery.csv`

## Modeling note

The DNP term is a first-order phenomenological build term:

\[
\dot n_{m,k}\big|_{\rm DNP}
= \Gamma_{\rm DNP}\left[n^{\rm sat}_{m,k}(P_{\rm DNP}^{\rm sat})-n_{m,k}\right].
\]

The internal redistribution terms use the current integrated vector polarization `P(t)` as the reference. This means that with DNP off, RF-created area loss is not restored. The hole can fill and the recoil peak can relax, but the final smooth line has less total vector area than the initial line.

## Current limits

The RF operator is still an ideal one-bin operator. The realistic finite-width RF hole profile should later be added as a separate Voigt kernel. The DNP and spin-diffusion/recovery rates are phenomenological and should be fit to measured burn/recovery data.
