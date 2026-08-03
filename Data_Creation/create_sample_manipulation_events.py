"""
Create individual ssRF / AFP / unmanipulated manipulation events for testing.

Each event is one final full-spectrum manipulation saved to its own NPZ file
(see ``manipulation_event_io.py``). Physics matches the dulya_fit_v4 training
pipeline (Dulya equilibrium + ssrf_realtime_v2 burns / AFP + relaxation).

Examples (from repo root):
  python Data_Creation/create_sample_manipulation_events.py
  python Data_Creation/create_sample_manipulation_events.py --output Data_Creation/sample_manipulation_events
  python Data_Creation/create_sample_manipulation_events.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DULYA_V4 = SCRIPT_DIR / "dulya_fit_v4"
DULYA_V2 = SCRIPT_DIR / "dulya_fit_v2"
DULYA_PKG = DULYA_V4 if DULYA_V4.is_dir() else DULYA_V2

if str(DULYA_PKG) not in sys.path:
    sys.path.insert(0, str(DULYA_PKG))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from afp_bin_traj import run_one_polarization as run_afp_event  # noqa: E402
from common import (  # noqa: E402
    AFP_N_RELAX,
    BURN_BIN_CHOICES,
    F_MAX,
    F_MIN,
    NUM_BINS,
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    burn_steps_grid,
    gamma_rf_grid,
)
from manipulation_event_io import (  # noqa: E402
    save_manipulation_event,
    write_manifest,
)
from model_bridge import mirror_bin_idx  # noqa: E402
from ssrf_bin_traj import (  # noqa: E402
    run_one_polarization as run_ssrf_event,
    run_unmanipulated_polarization,
)

DEFAULT_OUTPUT = SCRIPT_DIR / "sample_manipulation_events"

DEFAULT_P_VALUES = (0.25, 0.45, 0.65)
# Spread across burn window R in (-3, 3) (bins ~125-374).
DEFAULT_SSRF_BINS = (150, 250, 350)
DEFAULT_AFP_BINS = (150, 220, 290, 350)
# Tiny positive-P smoke set for quick local/cluster eval.
QUICK_P_VALUES = (0.35, 0.55)
QUICK_SSRF_BINS = (190,220)
QUICK_AFP_BINS = (190,220)


def _frequency_axis(num_bins: int = NUM_BINS) -> np.ndarray:
    return np.linspace(float(F_MIN), float(F_MAX), int(num_bins), dtype=np.float32)


def _final_spectrum_from_traj(traj: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iplus = np.asarray(traj["ip_spectrum"], dtype=np.float32)
    iminus = np.asarray(traj["im_spectrum"], dtype=np.float32)
    ps = iplus + iminus
    return ps, iplus, iminus


def _save_ssrf_event(
    output_dir: Path,
    *,
    bin_idx: int,
    p0: float,
    gamma_rf: float,
    burn_steps: int,
    frequency: np.ndarray,
) -> Path | None:
    traj = run_ssrf_event(
        int(bin_idx),
        float(p0),
        gamma_rf=float(gamma_rf),
        n_steps=int(burn_steps),
        capture_spectrum=True,
    )
    if bool(traj.get("skipped", False)):
        print(f"  skip ssRF p0={p0:.3f} bin={bin_idx} gamma={gamma_rf} steps={burn_steps}", flush=True)
        return None

    ps, iplus, iminus = _final_spectrum_from_traj(traj)
    mirror = mirror_bin_idx(NUM_BINS, int(bin_idx))
    event_id = f"ssrf_p{p0:.3f}_g{gamma_rf:.2g}_b{burn_steps}_bin{bin_idx}"
    out = output_dir / "ssrf" / f"{event_id}.npz"
    save_manipulation_event(
        out,
        ps=ps,
        iplus=iplus,
        iminus=iminus,
        frequency=frequency,
        p0=float(p0),
        source=SOURCE_SSRF,
        manipulation_mode="ssrf",
        center_bin=int(bin_idx),
        step=int(burn_steps),
        burn_steps=int(burn_steps),
        gamma_rf=float(gamma_rf),
        event_id=event_id,
        extra_meta={
            "mirror_bin": int(mirror),
            "p_initial": float(traj.get("p_initial", p0)),
            "q_initial": float(traj.get("q_initial", 0.0)),
            "p_final": float(traj.get("p_final", p0)),
            "q_final": float(traj.get("q_final", 0.0)),
            "rf_mode": str(traj.get("rf_mode", "")),
            "physics": "dulya_fit_v2/ssrf_bin_traj.run_one_polarization",
        },
    )
    print(f"  wrote {out.relative_to(output_dir.parent)}", flush=True)
    return out


def _save_afp_event(
    output_dir: Path,
    *,
    bin_idx: int,
    p0: float,
    frequency: np.ndarray,
    n_relax: int,
) -> Path | None:
    traj = run_afp_event(
        int(bin_idx),
        float(p0),
        n_relax=int(n_relax),
        capture_spectrum=True,
    )
    if bool(traj.get("skipped", False)):
        print(f"  skip AFP p0={p0:.3f} bin={bin_idx}", flush=True)
        return None

    ps, iplus, iminus = _final_spectrum_from_traj(traj)
    mirror = mirror_bin_idx(NUM_BINS, int(bin_idx))
    event_id = f"afp_p{p0:.3f}_bin{bin_idx}"
    out = output_dir / "afp" / f"{event_id}.npz"
    save_manipulation_event(
        out,
        ps=ps,
        iplus=iplus,
        iminus=iminus,
        frequency=frequency,
        p0=float(p0),
        source=SOURCE_AFP,
        manipulation_mode="afp",
        center_bin=int(bin_idx),
        step=int(n_relax),
        burn_steps=-1,
        gamma_rf=float("nan"),
        event_id=event_id,
        extra_meta={
            "mirror_bin": int(mirror),
            "afp_subset": [int(x) for x in traj.get("afp_subset", [])],
            "n_relax_steps": int(n_relax),
            "p_initial": float(traj.get("p_initial", p0)),
            "q_initial": float(traj.get("q_initial", 0.0)),
            "p_final": float(traj.get("p_final", p0)),
            "q_final": float(traj.get("q_final", 0.0)),
            "physics": "dulya_fit_v2/afp_bin_traj.run_one_polarization",
        },
    )
    print(f"  wrote {out.relative_to(output_dir.parent)}", flush=True)
    return out


def _save_unmanip_event(
    output_dir: Path,
    *,
    p0: float,
    frequency: np.ndarray,
) -> Path:
    traj = run_unmanipulated_polarization(float(p0))
    ps = np.asarray(traj["ps_full"][0], dtype=np.float32)
    iplus = np.asarray(traj["iplus_full"][0], dtype=np.float32)
    iminus = np.asarray(traj["iminus_full"][0], dtype=np.float32)
    event_id = f"unmanip_p{p0:.3f}"
    out = output_dir / "unmanip" / f"{event_id}.npz"
    save_manipulation_event(
        out,
        ps=ps,
        iplus=iplus,
        iminus=iminus,
        frequency=frequency,
        p0=float(p0),
        source=SOURCE_UNMANIP,
        manipulation_mode="unmanip",
        center_bin=int(traj.get("center_bin", NUM_BINS // 2)),
        step=0,
        burn_steps=0,
        gamma_rf=float("nan"),
        event_id=event_id,
        extra_meta={
            "physics": "dulya_fit_v2/ssrf_bin_traj.run_unmanipulated_polarization",
        },
    )
    print(f"  wrote {out.relative_to(output_dir.parent)}", flush=True)
    return out


def create_sample_events(
    output_dir: Path,
    *,
    p_values: tuple[float, ...],
    ssrf_bins: tuple[int, ...],
    afp_bins: tuple[int, ...],
    gamma_values: tuple[float, ...] | None = None,
    burn_steps_values: tuple[int, ...] | None = None,
    n_relax: int = AFP_N_RELAX,
) -> list[Path]:
    output_dir = Path(output_dir)
    frequency = _frequency_axis()

    # Default: every other grid point -> gamma {5.0, 10.0}, steps {20, 60, 100}.
    gamma_values = tuple(float(g) for g in (gamma_values or gamma_rf_grid()[::2]))
    burn_steps_values = tuple(int(n) for n in (burn_steps_values or burn_steps_grid()[::2]))

    saved: list[Path] = []
    print("Generating ssRF events...", flush=True)
    for p0 in p_values:
        for bin_idx in ssrf_bins:
            if int(bin_idx) not in set(BURN_BIN_CHOICES.tolist()):
                print(f"  warning: bin {bin_idx} outside default burn R window", flush=True)
            for gamma_rf in gamma_values:
                for burn_steps in burn_steps_values:
                    path = _save_ssrf_event(
                        output_dir,
                        bin_idx=int(bin_idx),
                        p0=float(p0),
                        gamma_rf=float(gamma_rf),
                        burn_steps=int(burn_steps),
                        frequency=frequency,
                    )
                    if path is not None:
                        saved.append(path)

    print("Generating AFP events...", flush=True)
    for p0 in p_values:
        for bin_idx in afp_bins:
            path = _save_afp_event(
                output_dir,
                bin_idx=int(bin_idx),
                p0=float(p0),
                frequency=frequency,
                n_relax=int(n_relax),
            )
            if path is not None:
                saved.append(path)

    print("Generating unmanipulated events...", flush=True)
    for p0 in p_values:
        saved.append(_save_unmanip_event(output_dir, p0=float(p0), frequency=frequency))

    manifest = write_manifest(output_dir, saved)
    print(f"Wrote {len(saved)} events and {manifest}", flush=True)
    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create individual manipulation event NPZs for testing")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Small set: 2 polarizations, 1 ssRF bin, 2 AFP bins, one gamma/burn combo each",
    )
    p.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        default=None,
        help="Initial polarizations (default: preset grid)",
    )
    p.add_argument(
        "--ssrf-bins",
        type=int,
        nargs="+",
        default=None,
        help="ssRF burn center bins (default: preset spread over burn window)",
    )
    p.add_argument(
        "--afp-bins",
        type=int,
        nargs="+",
        default=None,
        help="AFP center bins (default: preset spread)",
    )
    p.add_argument(
        "--n-relax",
        type=int,
        default=AFP_N_RELAX,
        help="AFP relaxation macro-steps (default: training AFP_N_RELAX)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.quick:
        p_values = tuple(args.p_values or QUICK_P_VALUES)
        ssrf_bins = tuple(args.ssrf_bins or QUICK_SSRF_BINS)
        afp_bins = tuple(args.afp_bins or QUICK_AFP_BINS)
        gamma_values = (7.5,)
        burn_steps_values = (60,)
    else:
        p_values = tuple(args.p_values or DEFAULT_P_VALUES)
        ssrf_bins = tuple(args.ssrf_bins or DEFAULT_SSRF_BINS)
        afp_bins = tuple(args.afp_bins or DEFAULT_AFP_BINS)
        gamma_values = None
        burn_steps_values = None

    create_sample_events(
        args.output,
        p_values=p_values,
        ssrf_bins=ssrf_bins,
        afp_bins=afp_bins,
        gamma_values=gamma_values,
        burn_steps_values=burn_steps_values,
        n_relax=int(args.n_relax),
    )


if __name__ == "__main__":
    main()
