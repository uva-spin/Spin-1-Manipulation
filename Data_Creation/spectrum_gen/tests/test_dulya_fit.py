"""Fitted Dulya equilibrium is opt-in; the default stays analytic Pake."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from Data_Creation.spectrum_gen.bridge import build_equilibrium_spin1_model, full_spectrum_intensities, level_pq
from Data_Creation.spectrum_gen.constants import DT, F_MAX, F_MIN, RF_GAUSSIAN_FWHM_R, RF_LORENTZIAN_FWHM_R
from Data_Creation.spectrum_gen.dulya import active_shape, dulya_lineshape, use_fit_params
from Data_Creation.spectrum_gen.eq import equilibrium_lineshape
from Data_Creation.spectrum_gen.ssrf_traj import run_one_polarization

FIT = Path(__file__).resolve().parents[2] / 'fit_params.json'


def test_missing_fit_file_is_an_error():
    with pytest.raises(FileNotFoundError):
        use_fit_params(FIT.parent / 'no-such-fit.json')
    use_fit_params(None)


def test_default_lineshape_ignores_fit_until_installed():
    use_fit_params(None)
    freq = np.linspace(F_MIN, F_MAX, 40)
    _, iplus_pake, _ = equilibrium_lineshape(0.4, freq)
    use_fit_params(FIT)
    try:
        _, iplus, iminus = equilibrium_lineshape(0.4, freq)
        assert not np.allclose(iplus, iplus_pake)
        model = build_equilibrium_spin1_model(
            polarization=0.4,
            num_bins=freq.size,
            dt=0.0015,
            r_min=F_MIN,
            r_max=F_MAX,
            legacy_spectral_recovery=False,
        )
        ip_m, im_m, _ = full_spectrum_intensities(model)
        assert np.allclose(ip_m, iplus, rtol=1e-5, atol=1e-6)
        assert np.allclose(im_m, iminus, rtol=1e-5, atol=1e-6)
        assert np.max(np.abs(ip_m + im_m)) < 0.2
        _, ip_fit, im_fit = dulya_lineshape(0.4, freq, active_shape())
        fit_scale = 0.4 / float(np.sum(ip_fit + im_fit))
        assert np.allclose(ip_m, ip_fit * fit_scale, rtol=1e-5, atol=1e-8)
        assert np.allclose(im_m, im_fit * fit_scale, rtol=1e-5, atol=1e-8)
        assert float(np.sum(ip_m + im_m)) == pytest.approx(0.4, abs=1e-6)
        p_eq, _q_eq = level_pq(model)
        assert p_eq == pytest.approx(0.4, abs=1e-6)
        assert model.display_cal == pytest.approx(model.dR)
    finally:
        use_fit_params(None)
    _, iplus_again, _ = equilibrium_lineshape(0.4, freq)
    assert np.allclose(iplus_again, iplus_pake)


def test_ssrf_on_fitted_lineshape_raises_tensor_polarization():
    """Weak-horn ssRF lowers P and raises Q while the signal stays on the fit scale."""
    use_fit_params(FIT)
    try:
        freq = np.linspace(F_MIN, F_MAX, 200)
        center = int(np.argmin(np.abs(freq + 1.0)))
        traj = run_one_polarization(
            center,
            0.45,
            num_bins=freq.size,
            gamma_rf=5.0,
            n_steps=40,
            dt=DT,
            gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R,
            lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R,
            capture_spectrum=True,
            legacy_spectral_recovery=False,
        )
        assert not traj['skipped']
        signal = traj['iplus_full'] + traj['iminus_full']
        assert float(np.max(signal[0])) < 0.05
        assert traj['p_final'] < traj['p_initial']
        assert traj['q_final'] > traj['q_initial']
        assert traj['p_initial'] == pytest.approx(0.45, abs=1e-6)
    finally:
        use_fit_params(None)
