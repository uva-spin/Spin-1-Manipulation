"""DAE generator uses ssRF-beta physics and independent profile / unmanip events."""
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "Data_Creation"
_RIVANNA = _DATA / "rivanna"
for path in (_ROOT, _DATA, _RIVANNA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from create_data import (
    F_MAX,
    F_MIN,
    NUM_BINS,
    PROFILE_CENTER_BIN,
    UNMANIP_CENTER_BIN,
    _resolve_modes,
    generate_spectra,
)
from physics.rf.lineshape import boltzmann_Q
from common import SOURCE_AFP, SOURCE_PROFILE, SOURCE_SSRF, SOURCE_UNMANIP
from physics.rf.optimal_profile import OptimizerSettings


def test_resolve_modes_profile_is_independent():
    # (ssrf, afp, afp_relax, afp_profile, profile, unmanipulated)
    assert _resolve_modes(
        ssrf=None, afp=None, afp_relax=False, afp_profile=False, profile=None, unmanipulated=None, quick=False,
    ) == (True, False, False, False, False, False)
    assert _resolve_modes(
        ssrf=None, afp=None, afp_relax=False, afp_profile=False, profile=None, unmanipulated=None, quick=True,
    ) == (True, True, False, False, True, False)
    assert _resolve_modes(
        ssrf=None, afp=None, afp_relax=False, afp_profile=False, profile=True, unmanipulated=None, quick=False,
    ) == (False, False, False, False, True, False)
    assert _resolve_modes(
        ssrf=True, afp=None, afp_relax=False, afp_profile=False, profile=True, unmanipulated=None, quick=False,
    ) == (True, False, False, False, True, False)
    assert _resolve_modes(
        ssrf=None, afp=None, afp_relax=False, afp_profile=False, profile=None, unmanipulated=True, quick=False,
    ) == (False, False, False, False, False, True)
    assert _resolve_modes(
        ssrf=True, afp=None, afp_relax=False, afp_profile=False, profile=None, unmanipulated=True, quick=False,
    ) == (True, False, False, False, False, True)


def test_generate_spectra_keeps_profile_events_separate():
    settings = OptimizerSettings(
        duration_samples=3,
        duration_refinements=0,
        max_iterations=2,
        starts=1,
        max_wall_seconds=20.0,
        search_dt=0.01,
    )
    data = generate_spectra(
        do_ssrf=True,
        do_afp=False,
        do_profile=True,
        p_values=np.asarray([0.45]),
        min_burn_steps=0,
        max_burn_steps=2,
        burn_steps_step=1,
        max_centers_per_p=1,
        num_bins=21,
        r_min=-3.0,
        r_max=3.0,
        dt=0.02,
        profile_settings=settings,
        gamma_rf=2.0,
    )
    sources = np.unique(data["source"])
    assert SOURCE_SSRF in sources
    assert SOURCE_PROFILE in sources
    assert SOURCE_AFP not in sources
    assert SOURCE_UNMANIP not in sources
    assert data["spectra"].shape[-1] == 21
    profile = data["source"] == SOURCE_PROFILE
    ssrf = data["source"] == SOURCE_SSRF
    assert np.all(data["center_bin"][profile] == PROFILE_CENTER_BIN)
    assert np.all(data["center_bin"][ssrf] >= 0)
    assert data["power_profile"].shape == (data["spectra"].shape[0], 21)
    ssrf_prof = data["power_profile"][ssrf]
    ssrf_centers = data["center_bin"][ssrf]
    peak_at_center = ssrf_prof[np.arange(ssrf_prof.shape[0]), ssrf_centers]
    np.testing.assert_allclose(peak_at_center, 2.0, atol=0.05)
    assert np.any(ssrf_prof.std(axis=1) > 0.1)
    profile_prof = data["power_profile"][profile]
    assert profile_prof.shape[1] == 21
    np.testing.assert_allclose(profile_prof.max(axis=1), data["applied_power"][profile], atol=1e-5)
    assert set(data["n_steps"][ssrf].tolist()) <= {0, 1, 2}
    assert 0 in set(data["n_steps"][profile].tolist())
    np.testing.assert_allclose(data["P_total"][ssrf][data["n_steps"][ssrf] == 0][0], 0.45, atol=5e-3)
    np.testing.assert_allclose(data["P_total"][profile][data["n_steps"][profile] == 0][0], 0.45, atol=5e-3)
    np.testing.assert_allclose(
        data["Q_total"][ssrf][data["n_steps"][ssrf] == 0][0],
        boltzmann_Q(0.45),
        atol=5e-3,
    )


def test_generate_unmanipulated_events():
    data = generate_spectra(
        do_ssrf=False,
        do_afp=False,
        do_unmanipulated=True,
        p_values=np.asarray([0.3, 0.45]),
        num_bins=21,
        r_min=-3.0,
        r_max=3.0,
        dt=0.02,
    )
    assert data["spectra"].shape == (2, 2, 21)
    assert np.all(data["source"] == SOURCE_UNMANIP)
    assert np.all(data["n_steps"] == 0)
    assert np.all(data["applied_power"] == 0.0)
    assert np.all(data["center_bin"] == UNMANIP_CENTER_BIN)
    assert np.all(data["power_profile"] == 0.0)
    np.testing.assert_allclose(data["P_total"], [0.3, 0.45], atol=5e-3)
    np.testing.assert_allclose(data["Q_total"], [boltzmann_Q(0.3), boltzmann_Q(0.45)], atol=5e-3)


def test_default_spectrum_grid_is_500_bins_pm6():
    assert NUM_BINS == 500
    assert F_MIN == pytest.approx(-6.0)
    assert F_MAX == pytest.approx(6.0)


def test_burn_centers_are_q_negative_inside_inner_window():
    from burn_selection import equilibrium_q_profile
    from common import BURN_BIN_CHOICES, BURN_R_MAX, BURN_R_MIN, EXCLUDED_MANIPULATION_BURN_BINS
    from create_data import _burn_window_bins, q_negative_bins_for_p0

    burn_window = _burn_window_bins()
    expected = np.asarray(
        [b for b in BURN_BIN_CHOICES if b not in EXCLUDED_MANIPULATION_BURN_BINS],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(burn_window, expected)
    freq = np.linspace(F_MIN, F_MAX, NUM_BINS)
    assert np.all(freq[burn_window] > BURN_R_MIN)
    assert np.all(freq[burn_window] < BURN_R_MAX)
    p0 = 0.45
    q = equilibrium_q_profile(p0, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX)
    centers = q_negative_bins_for_p0(p0, burn_window)
    assert centers.size > 0
    assert np.all(np.isin(centers, burn_window))
    assert np.all(q[centers] < 0.0)
    assert np.any(q[burn_window] >= 0.0)


def test_ssrf_voigt_power_profile_peaks_at_center():
    from create_data import ssrf_voigt_power_profile, zero_power_profile

    zeros = zero_power_profile(21)
    assert zeros.shape == (21,)
    assert np.all(zeros == 0.0)
    prof = ssrf_voigt_power_profile(7, 10.0, num_bins=21, r_min=-3.0, r_max=3.0)
    assert prof.shape == (21,)
    assert int(np.argmax(prof)) == 7
    np.testing.assert_allclose(prof[7], 10.0, atol=0.05)
    assert prof.min() >= 0.0


def test_polarization_grid_is_denser_than_v4_npz():
    from create_data import P_STEP, UNMANIP_P_STEP, polarization_grid

    values = polarization_grid(0.2, 0.6, P_STEP)
    assert P_STEP == pytest.approx(0.025)
    assert values.size >= 16
    np.testing.assert_allclose(values[0], 0.2, atol=1e-12)
    np.testing.assert_allclose(values[-1], 0.6, atol=1e-12)
    assert UNMANIP_P_STEP < P_STEP
    unmanip = polarization_grid(0.2, 0.6, UNMANIP_P_STEP)
    assert unmanip.size > values.size


def test_unmanipulated_uses_finer_p_step_than_ssrf():
    data = generate_spectra(
        do_ssrf=True,
        do_afp=False,
        do_unmanipulated=True,
        p_min=0.2,
        p_max=0.3,
        p_step=0.05,
        unmanip_p_step=0.025,
        min_burn_steps=0,
        max_burn_steps=0,
        burn_steps_step=1,
        max_centers_per_p=1,
        num_bins=21,
        r_min=-3.0,
        r_max=3.0,
        dt=0.02,
        gamma_rf=2.0,
    )
    ssrf = data['source'] == SOURCE_SSRF
    unmanip = data['source'] == SOURCE_UNMANIP
    # ssRF: one center × one step × 3 polarizations (0.20, 0.25, 0.30)
    assert len(np.unique(np.round(data['p0'][ssrf], 5))) == 3
    # unmanipulated: finer grid → 5 polarizations (0.20 .. 0.30 step 0.025)
    assert len(np.unique(np.round(data['p0'][unmanip], 5))) == 5
    assert unmanip.sum() == 5
