"""
Generate full-spectrum manipulated lineshapes for DAE training.

Edit the USER PARAMETERS block below, then run this script. Helpers live in
``Data_Creation/spectrum_gen/``.

Flexible knobs (``GAMMA_RF``, ``AFP_WINDOW``, ``GAUSSIAN_FWHM_R``,
``LORENTZIAN_FWHM_R``, ``DT``):
  - scalar: use that value
  - tuple ``(lo, hi)``: sample uniformly once per burn bin (held for every step of that burn)
  - list ``[...]``: iterate (Cartesian product across list knobs)

Examples (from repo root):
  python Data_Creation/create_data.py --quick
  python Data_Creation/create_data.py --ssrf --afp
  python Data_Creation/create_data.py --unmanipulated
  python Data_Creation/create_data.py --ssrf-afp-combined --no-ssrf --no-afp
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import sys
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.rf.optimal_profile import DATA_GEN_SETTINGS, QUICK_SETTINGS  # noqa: E402

from Data_Creation.spectrum_gen import (  # noqa: E402
    _burn_window_bins,
    _combo_max_relax,
    _combined_center_pairs,
    _resolve_modes,
    _sanitize_saved_arrays,
    describe_param,
    generate_spectra,
    q_negative_bins_for_p0,
    save_example_plots,
)
from Data_Creation.spectrum_gen.constants import (  # noqa: E402
    BURN_R_MAX,
    BURN_R_MIN,
    COMBO_LAYOUT_PROFILE_AFP_SELECTIVE,
    COMBO_LAYOUT_PROFILE_PROFILE,
    COMBO_LAYOUT_SELECTIVE_SELECTIVE,
    COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE,
    COMBO_SCENARIO_AFP_FIRST_REGION,
    COMBO_SCENARIO_BOTH_SELECTIVE,
    COMBO_SCENARIO_SSRF_FIRST_REGION,
    F_MAX,
    F_MIN,
    NUM_BINS,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    SOURCE_AFP,
    SOURCE_AFP_PROFILE,
    SOURCE_PROFILE,
    SOURCE_SSRF,
    SOURCE_SSRF_AFP,
    SOURCE_UNMANIP,
)
from Data_Creation.spectrum_gen.events import _combo_centers_cap  # noqa: E402

# ---------------------------------------------------------------------------
# USER PARAMETERS — edit these to create new sample data
# ---------------------------------------------------------------------------
P_MIN = 0.2
P_MAX = 0.6
P_STEP = 0.025
UNMANIP_P_STEP = 0.00005

# scalar | (lo, hi) sample once per selective ssRF burn | list of values to iterate.
# Optimal-profile ssRF keeps the designed per-bin rates (not this knob).
GAMMA_RF = (5, 10.0)
# GAUSSIAN_FWHM_R = (0.03, 3*0.03)
# LORENTZIAN_FWHM_R = (0.015, 3*0.015)
GAUSSIAN_FWHM_R = 0.03
LORENTZIAN_FWHM_R = 0.015
AFP_WINDOW = 8
# AFP-only: 0 keeps every Q<0 bin; n>0 keeps every nth affected bin
AFP_BIN_STRIDE = 0
# DT = (0.0015*0.8, 0.0015*1.2)
DT = 0.0015
MIN_BURN_STEPS = 0
MAX_BURN_STEPS = 400
BURN_STEPS_STEP = 1
MIN_RELAX_STEPS = 0
MAX_RELAX_STEPS = 8000
RELAX_STEPS_STEP = 1

DEFAULT_OUTPUT = SCRIPT_DIR / 'spectra_data' / 'spectra.npz'
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'spectra_data' / 'plots'
DEFAULT_SEED = 42

__all__ = [
    'AFP_WINDOW',
    'DT',
    'GAUSSIAN_FWHM_R',
    'GAMMA_RF',
    'LORENTZIAN_FWHM_R',
    '_burn_window_bins',
    '_combo_max_relax',
    '_combined_center_pairs',
    'generate_spectra',
    'main',
    'parse_args',
    'q_negative_bins_for_p0',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate full-spectrum ssRF / AFP / profile / unmanipulated lineshapes. '
            'Edit USER PARAMETERS for flexible GAMMA_RF / AFP_WINDOW / FWHM / DT '
            '(scalar, range tuple, or list). CLI --gamma-rf / --afp-window / --dt are optional scalar overrides.'
        )
    )
    parser.add_argument('--ssrf', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--afp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--afp-relax', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--afp-profile', action='store_true')
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--unmanipulated', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--ssrf-afp-combined', action='store_true')
    parser.add_argument('--combined-max-centers', type=int, default=None)
    parser.add_argument('--combined-max-relax-steps', type=int, default=500)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--plot-dir', type=Path, default=None)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--p-min', type=float, default=P_MIN)
    parser.add_argument('--p-max', type=float, default=P_MAX)
    parser.add_argument('--p-step', type=float, default=P_STEP)
    parser.add_argument('--unmanip-p-step', type=float, default=UNMANIP_P_STEP)
    parser.add_argument('--min-burn-steps', type=int, default=MIN_BURN_STEPS)
    parser.add_argument('--max-burn-steps', type=int, default=MAX_BURN_STEPS)
    parser.add_argument('--burn-steps-step', type=int, default=BURN_STEPS_STEP)
    parser.add_argument('--min-relax-steps', type=int, default=MIN_RELAX_STEPS)
    parser.add_argument('--max-relax-steps', type=int, default=MAX_RELAX_STEPS)
    parser.add_argument('--relax-steps-step', type=int, default=RELAX_STEPS_STEP)
    parser.add_argument('--gamma-rf', type=float, default=None)
    parser.add_argument('--afp-window', type=int, default=None)
    parser.add_argument(
        '--afp-bin-stride',
        type=int,
        default=AFP_BIN_STRIDE,
        help='AFP-only: keep every nth Q<0 bin. 0 keeps every affected bin.',
    )
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--quick', action='store_true')
    return parser.parse_args()


def main():
    faulthandler.enable()
    args = parse_args()
    try:
        do_ssrf, do_afp, do_afp_relax, do_afp_profile, do_profile, do_unmanipulated = _resolve_modes(
            ssrf=args.ssrf,
            afp=args.afp,
            afp_relax=args.afp_relax,
            afp_profile=args.afp_profile,
            profile=args.profile,
            unmanipulated=args.unmanipulated,
            quick=args.quick,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    gamma_rf = args.gamma_rf if args.gamma_rf is not None else GAMMA_RF
    afp_window = args.afp_window if args.afp_window is not None else AFP_WINDOW
    gaussian_fwhm_R = GAUSSIAN_FWHM_R
    lorentzian_fwhm_R = LORENTZIAN_FWHM_R
    dt = args.dt if args.dt is not None else DT

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_settings = QUICK_SETTINGS if args.quick else DATA_GEN_SETTINGS
    combined_max_centers = args.combined_max_centers
    combined_max_relax_steps = args.combined_max_relax_steps

    if args.quick:
        p_values = np.asarray([0.3, 0.5])
        min_burn_steps, max_burn_steps, burn_steps_step = 0, 100, 5
        min_relax_steps, max_relax_steps, relax_steps_step = 0, 100, 5
        max_centers_per_p = 3
        if combined_max_centers is None:
            combined_max_centers = max_centers_per_p
        print('Quick smoke: P={0.30, 0.50}, 3 centers/P, short step grids', flush=True)
    else:
        p_values = None
        min_burn_steps, max_burn_steps, burn_steps_step = args.min_burn_steps, args.max_burn_steps, args.burn_steps_step
        min_relax_steps, max_relax_steps, relax_steps_step = args.min_relax_steps, args.max_relax_steps, args.relax_steps_step
        max_centers_per_p = None

    do_ssrf_afp_combined = bool(args.ssrf_afp_combined)
    modes = [m for m, on in (
        ('ssRF', do_ssrf),
        ('AFP+relax' if do_afp_relax else 'AFP', do_afp),
        ('AFP Profile', do_afp_profile),
        ('profile', do_profile),
        ('unmanipulated', do_unmanipulated),
        ('ssRF+AFP combined', do_ssrf_afp_combined),
    ) if on]

    print(
        f"Generating {'+'.join(modes) or 'nothing'} | P [{args.p_min}, {args.p_max}] "
        f'step={args.p_step} | gamma_rf={describe_param(gamma_rf)} | '
        f'AFP window={describe_param(afp_window)} | '
        f'FWHM G={describe_param(gaussian_fwhm_R)} L={describe_param(lorentzian_fwhm_R)} | '
        f'dt={describe_param(dt)} n={NUM_BINS} R∈[{F_MIN},{F_MAX}] burn R∈({BURN_R_MIN},{BURN_R_MAX}) | seed={args.seed}',
        flush=True,
    )
    if do_afp:
        afp_stride = int(args.afp_bin_stride)
        afp_bins = 'every affected bin' if afp_stride <= 0 else f'every {afp_stride}th affected bin'
        print(f'  AFP centers: {afp_bins}', flush=True)
    if do_ssrf_afp_combined:
        print(
            f'  combined: centers/region <= {_combo_centers_cap(max_centers_per_p, combined_max_centers)}; '
            f'relax <= {_combo_max_relax(max_relax_steps, max_burn_steps, combined_max_relax_steps)}',
            flush=True,
        )

    profile_rates = {}
    try:
        data = generate_spectra(
            do_ssrf=do_ssrf,
            do_afp=do_afp,
            do_afp_relax=do_afp_relax,
            do_afp_profile=do_afp_profile,
            do_profile=do_profile,
            do_unmanipulated=do_unmanipulated,
            do_ssrf_afp_combined=do_ssrf_afp_combined,
            p_min=args.p_min,
            p_max=args.p_max,
            p_step=args.p_step,
            unmanip_p_step=args.unmanip_p_step,
            min_burn_steps=min_burn_steps,
            max_burn_steps=max_burn_steps,
            burn_steps_step=burn_steps_step,
            min_relax_steps=min_relax_steps,
            max_relax_steps=max_relax_steps,
            relax_steps_step=relax_steps_step,
            gamma_rf=gamma_rf,
            afp_window=afp_window,
            afp_bin_stride=args.afp_bin_stride,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
            dt=dt,
            max_centers_per_p=max_centers_per_p,
            combined_max_centers=combined_max_centers,
            combined_max_relax_steps=combined_max_relax_steps,
            p_values=p_values,
            profile_settings=profile_settings,
            profile_rates_out=profile_rates,
            seed=args.seed,
        )
    except MemoryError:
        print('ERROR: out of memory while generating spectra.', file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    except Exception:
        print('ERROR: create_data.py failed:', file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1) from None

    data = _sanitize_saved_arrays(data)
    source = data['source']
    metadata = {
        'dataset': 'spectra',
        'source_codes': {
            'ssRF': int(SOURCE_SSRF),
            'AFP': int(SOURCE_AFP),
            'unmanipulated': int(SOURCE_UNMANIP),
            'optimal profile': int(SOURCE_PROFILE),
            'AFP Profile': int(SOURCE_AFP_PROFILE),
            'ssRF+AFP combined': int(SOURCE_SSRF_AFP),
        },
        'combo_scenario_codes': {
            'ssRF_first_Qneg_region': int(COMBO_SCENARIO_SSRF_FIRST_REGION),
            'AFP_first_Qneg_region': int(COMBO_SCENARIO_AFP_FIRST_REGION),
            'both_selective_all_Qneg': int(COMBO_SCENARIO_BOTH_SELECTIVE),
        },
        'combo_layout_codes': {
            'profile_profile': int(COMBO_LAYOUT_PROFILE_PROFILE),
            'profile_afp_selective': int(COMBO_LAYOUT_PROFILE_AFP_SELECTIVE),
            'ssrf_selective_profile': int(COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE),
            'selective_selective': int(COMBO_LAYOUT_SELECTIVE_SELECTIVE),
        },
        'n_events': int(source.size),
        'n_by_source': {
            name: int(np.sum(source == code))
            for name, code in (
                ('ssRF', SOURCE_SSRF),
                ('AFP', SOURCE_AFP),
                ('unmanipulated', SOURCE_UNMANIP),
                ('optimal profile', SOURCE_PROFILE),
                ('AFP Profile', SOURCE_AFP_PROFILE),
                ('ssRF+AFP combined', SOURCE_SSRF_AFP),
            )
        },
        'gamma_rf': describe_param(gamma_rf),
        'afp_window': describe_param(afp_window),
        'gaussian_fwhm_R': describe_param(gaussian_fwhm_R),
        'lorentzian_fwhm_R': describe_param(lorentzian_fwhm_R),
        'dt': describe_param(dt),
        'seed': int(args.seed),
    }
    data['meta_json'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(output, **data)
    n = data['spectra'].shape[0]
    print(
        f"Saved N={n} spectra={data['spectra'].shape} -> {output} | "
        f"by source: {metadata['n_by_source']}",
        flush=True,
    )
    if not args.no_plots and n > 0:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else output.parent / 'plots'
        paths = save_example_plots(data, plot_dir, profile_rates=profile_rates)
        print(f'Wrote {len(paths)} plots -> {plot_dir}', flush=True)


if __name__ == '__main__':
    main()
