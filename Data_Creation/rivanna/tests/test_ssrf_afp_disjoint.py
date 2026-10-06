"""Non-overlapping selective ssRF and AFP windows in combined trajectories."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = Path(__file__).resolve().parents[1]
if str(_V2) not in sys.path:
    sys.path.insert(0, str(_V2))

from common import (
    COMBO_LAYOUT_PROFILE_AFP_SELECTIVE,
    COMBO_LAYOUT_PROFILE_PROFILE,
    COMBO_LAYOUT_SELECTIVE_SELECTIVE,
    COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE,
    COMBO_SCENARIO_BOTH_SELECTIVE,
    COMBO_SCENARIO_SSRF_FIRST_REGION,
    F_MAX,
    F_MIN,
    NUM_BINS,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
)
from ssrf_afp_combined_traj import (
    afp_influence_bins,
    combined_row_clock,
    run_combined_polarization,
    selective_manipulation_overlap,
    split_q_negative_regions,
    ssrf_influence_bins,
)

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from Data_Creation.create_data import (
    _burn_window_bins,
    _combo_max_relax,
    _combined_center_pairs,
    q_negative_bins_for_p0,
)
from burn_selection import equilibrium_q_profile


def _all_qneg_bins(p0):
    burn_window = _burn_window_bins()
    frequency = np.linspace(F_MIN, F_MAX, NUM_BINS)
    centers = q_negative_bins_for_p0(p0, burn_window)
    regions = split_q_negative_regions(centers, frequency)
    if len(regions) < 2:
        pytest.skip("P has fewer than two Q<0 regions")
    return np.concatenate(regions)


def _two_regions(p0):
    burn_window = _burn_window_bins()
    frequency = np.linspace(F_MIN, F_MAX, NUM_BINS)
    centers = q_negative_bins_for_p0(p0, burn_window)
    regions = split_q_negative_regions(centers, frequency)
    if len(regions) < 2:
        pytest.skip("P has fewer than two Q<0 regions")
    return regions[0], regions[1]


def test_selective_overlap_detects_shared_bins():
    ssrf_center = 211
    afp_center = 211
    assert selective_manipulation_overlap(
        ssrf_center,
        afp_center,
        num_bins=NUM_BINS,
        r_min=F_MIN,
        r_max=F_MAX,
        afp_window=8,
        gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R,
        lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R,
    )


@pytest.mark.parametrize("p0", [0.3, 0.5])
def test_combined_selective_afp_avoids_ssrf_influence(p0):
    region = _all_qneg_bins(p0)
    traj = run_combined_polarization(
        p0,
        scenario=COMBO_SCENARIO_BOTH_SELECTIVE,
        layout=COMBO_LAYOUT_SELECTIVE_SELECTIVE,
        ssrf_region=region,
        afp_region=region,
        gamma_rf=10.0,
        max_burn=100,
        max_relax=0,
        afp_window=8,
        profile_settings={},
        max_centers_per_region=3,
    )
    if traj.get("skipped"):
        pytest.skip(traj.get("reason", "skipped"))
    ssrf_center = int(traj["ssrf_center"])
    ssrf_block = ssrf_influence_bins(
        ssrf_center,
        num_bins=NUM_BINS,
        r_min=F_MIN,
        r_max=F_MAX,
    )
    afp_frame = int(traj["afp_frame"])
    ssrf_start = int(traj["ssrf_start"])
    n_ssrf = int(traj["n_ssrf_frames"])
    assert afp_frame < ssrf_start
    afp_prof = np.asarray(traj["power_profiles"][afp_frame])
    ssrf_prof = np.asarray(traj["power_profiles"][ssrf_start])
    assert np.max(afp_prof) > 0.0
    assert np.max(np.abs(ssrf_prof)) > 0.0
    assert not np.any((afp_prof > 0.0) & (np.abs(ssrf_prof) > 1e-8))
    for frame in range(ssrf_start, ssrf_start + n_ssrf):
        assert not np.any((afp_prof > 0.0) & (np.abs(traj["power_profiles"][frame]) > 1e-8))
    afp_center = int(traj["afp_center"])
    assert not (afp_influence_bins(afp_center, num_bins=NUM_BINS, afp_window=8) & ssrf_block)


def test_combined_row_clock_matches_separate_streams():
    assert combined_row_clock(0, 1, 10.0, relax_start=41) == (0, 0.0)
    assert combined_row_clock(1, 1, 10.0, relax_start=41) == (0, 10.0)
    assert combined_row_clock(40, 1, 10.0, relax_start=41) == (39, 10.0)
    assert combined_row_clock(41, 1, 10.0, relax_start=41) == (0, 0.0)
    assert combined_row_clock(44, 1, 10.0, relax_start=41) == (3, 0.0)


def test_combo_max_relax_caps_afp_profile_length():
    assert _combo_max_relax(8000, 400, None) == 400
    assert _combo_max_relax(8000, 400, 50) == 50
    assert _combo_max_relax(100, 400, None) == 100


def test_combined_center_pairs_are_bounded_not_cartesian():
    region_a, region_b = _two_regions(0.2)
    q = equilibrium_q_profile(0.2)
    cap = 5
    sel = _combined_center_pairs(
        COMBO_LAYOUT_SELECTIVE_SELECTIVE,
        region_a,
        region_b,
        q,
        max_centers=cap,
        scenario=COMBO_SCENARIO_SSRF_FIRST_REGION,
    )
    assert 1 <= len(sel) <= cap
    # Must not explode to |A|×|B|.
    assert len(sel) < region_a.size * region_b.size
    assert len(_combined_center_pairs(
        COMBO_LAYOUT_PROFILE_PROFILE,
        region_a,
        region_b,
        q,
        max_centers=cap,
        scenario=COMBO_SCENARIO_SSRF_FIRST_REGION,
    )) == 1
    afp_sel = _combined_center_pairs(
        COMBO_LAYOUT_PROFILE_AFP_SELECTIVE,
        region_a,
        region_b,
        q,
        max_centers=cap,
        scenario=COMBO_SCENARIO_SSRF_FIRST_REGION,
    )
    assert len(afp_sel) <= cap
    ssrf_sel = _combined_center_pairs(
        COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE,
        region_a,
        region_b,
        q,
        max_centers=cap,
        scenario=COMBO_SCENARIO_SSRF_FIRST_REGION,
    )
    assert len(ssrf_sel) <= cap
    assert _combined_center_pairs(
        COMBO_LAYOUT_SELECTIVE_SELECTIVE,
        region_a,
        region_b,
        q,
        max_centers=cap,
        scenario=COMBO_SCENARIO_BOTH_SELECTIVE,
    ) == [(None, None)]
