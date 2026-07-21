"""
Per-bin ssRF burn trajectories for SLURM array jobs.

Each task burns one R-bin from an unburned lineshape with a Voigt RF profile
(±5 bins), marching until the mirrored intensity amplitude starts decreasing —
the turnover after maximum semi-saturating RF, when relaxation begins restoring
mirror populations. That decreasing step is discarded and the burn ends. Saves
(ps, iplus, iminus) and amplitude |ps| at both the burn bin and its mirror at
every timestep, for each initial vector polarization.

After all shards exist, ``--organize`` routes samples into one training file per
spectral bin (for 500 independent models): burn-bin observations go to the burn
bin's file, mirror-bin observations go to the mirror bin's file. There is no
single combined NPZ.

Usage:
  python ssrf_bin_traj.py --bin-idx 172
  python ssrf_bin_traj.py --organize --shard-dir ssrf_shards --output-dir ssrf_train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.special import wofz

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime.model import Spin1Model, Spin1Params

NUM_BINS = 500
DT = 0.005
GAMMA_RF = 100.0
# Voigt RF envelope (bin units): Gaussian sigma + Lorentzian HWHM, truncated to ±HALF_WIDTH.
SIGMA_BINS = 2.0
VOIGT_GAMMA_BINS = 1.0
HALF_WIDTH = 5
MAX_STEPS = 5000
P_MIN = -0.70
P_MAX = 0.70
P_STEP = 0.05
PS_ABS_MIN = 1e-12
MIRROR_AMP_EPS = 1e-15
MIRROR_AMP_RTOL = 1e-6
MAX_GDT = 0.05
MAX_NSUB = 20

DEFAULT_SHARD_DIR = Path(__file__).resolve().parent / "ssrf_shards"
DEFAULT_TRAIN_DIR = Path(__file__).resolve().parent / "ssrf_train"


def mirror_bin_idx(n_bins: int, bin_idx: int) -> int:
    return int(n_bins) - 1 - int(bin_idx)


def ssrf_touched_bins(n_bins: int, subset: list[int] | np.ndarray) -> list[int]:
    """Packet/intensity bins ssRF changes: each burn index i also updates mirror(i)."""
    touched: set[int] = set()
    for i in subset:
        touched.add(int(i))
        touched.add(mirror_bin_idx(n_bins, int(i)))
    return sorted(touched)


def commit_touched_bins_only(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_sim: np.ndarray,
    iminus_sim: np.ndarray,
    touched: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep baseline intensities except on RF-touched bins (burn ∪ mirrors)."""
    out_ip = np.asarray(iplus, dtype=float).copy()
    out_im = np.asarray(iminus, dtype=float).copy()
    ip_sim = np.asarray(iplus_sim, dtype=float)
    im_sim = np.asarray(iminus_sim, dtype=float)
    for k in touched:
        out_ip[k] = float(ip_sim[k])
        out_im[k] = float(im_sim[k])
    return out_ip, out_im


def polarization_grid(p_min: float, p_max: float, p_step: float) -> np.ndarray:
    return np.arange(float(p_min), float(p_max) + 1e-12, float(p_step), dtype=float)


def _voigt_kernel(x: np.ndarray, x0: float, sigma: float, lorentz_gamma: float) -> np.ndarray:
    """Discretized Voigt (Faddeeva), same form as ``ssRFMapper._voigt_profile``."""
    sigma = max(float(sigma), 1e-12)
    x_norm = (np.asarray(x, dtype=float) - float(x0)) / (sigma * np.sqrt(2.0))
    z = x_norm + 1j * (float(lorentz_gamma) / (sigma * np.sqrt(2.0)))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def make_voigt_rf_profile(
    n_bins: int,
    center: int,
    gamma_rf: float,
    *,
    sigma: float = SIGMA_BINS,
    lorentz_gamma: float = VOIGT_GAMMA_BINS,
    half_width: int = HALF_WIDTH,
) -> tuple[np.ndarray, list[int]]:
    """
    Voigt RF envelope peaked at ``center``, truncated to ±half_width bins.

    Peak is normalized to ``gamma_rf``. Support bins become ``ssrf_subset_indices``.
    """
    profile = np.zeros(int(n_bins), dtype=float)
    c = int(center)
    lo = max(0, c - int(half_width))
    hi = min(int(n_bins) - 1, c + int(half_width))
    xs = np.arange(lo, hi + 1, dtype=float)
    kernel = _voigt_kernel(xs, float(c), float(sigma), float(lorentz_gamma))
    peak = float(np.max(kernel)) if kernel.size else 0.0
    support: list[int] = list(range(lo, hi + 1))
    if peak > 0.0:
        profile[lo : hi + 1] = float(gamma_rf) * (kernel / peak)
    return profile, support


def freeze_rf_profile(model: Spin1Model, profile: np.ndarray) -> Callable[[], None]:
    """Keep ``params.rf_profile`` fixed; ``ssrf_burn`` always calls ``set_rf_profile``."""
    frozen = np.asarray(profile, dtype=float).copy()
    model.params.rf_profile = frozen.copy()

    def _frozen_set_rf_profile() -> None:
        model.params.rf_profile = frozen.copy()

    model.set_rf_profile = _frozen_set_rf_profile  # type: ignore[method-assign]
    return _frozen_set_rf_profile


def intensities_at_bins(
    model: Spin1Model, bin_idx: int, mirror_idx: int
) -> tuple[float, float, float, float, float, float]:
    ip, im, _ = model.physical_intensities()
    iplus = float(ip[bin_idx])
    iminus = float(im[bin_idx])
    iplus_m = float(ip[mirror_idx])
    iminus_m = float(im[mirror_idx])
    return iplus, iminus, iplus + iminus, iplus_m, iminus_m, iplus_m + iminus_m


def mirror_amplitude(ps_m: float) -> float:
    """Scalar amplitude of the mirrored bin intensity (Ps at mirror)."""
    return abs(float(ps_m))


def mirror_amplitude_decreased(
    ps_m: float,
    ps_m_prev: float,
    *,
    atol: float = MIRROR_AMP_EPS,
    rtol: float = MIRROR_AMP_RTOL,
) -> bool:
    """True when mirrored amplitude fell relative to the previous kept step."""
    cur = mirror_amplitude(ps_m)
    prev = mirror_amplitude(ps_m_prev)
    return cur < prev - max(float(atol), float(rtol) * prev)


def euler_n_sub(gamma_rf: float, dt: float) -> tuple[int, float]:
    """Substep count so |gamma| * dt_sub <= MAX_GDT (stable Euler)."""
    g = abs(float(gamma_rf))
    dt_f = float(dt)
    if g <= 0.0 or dt_f <= 0.0:
        return 1, dt_f
    n_sub = min(max(1, int(np.ceil(g * dt_f / MAX_GDT))), int(MAX_NSUB))
    return n_sub, dt_f / float(n_sub)


def run_one_polarization(
    bin_idx: int,
    polarization: float,
    *,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    gamma_rf: float = GAMMA_RF,
    sigma_bins: float = SIGMA_BINS,
    voigt_gamma_bins: float = VOIGT_GAMMA_BINS,
    half_width: int = HALF_WIDTH,
    max_steps: int = MAX_STEPS,
) -> dict:
    f = np.linspace(-3.0, 3.0, int(num_bins))
    _, iplus0, iminus0 = GenerateVectorLineshape(float(polarization), f)
    iplus0 = np.asarray(iplus0, dtype=float)
    iminus0 = np.asarray(iminus0, dtype=float)
    mirror_idx = mirror_bin_idx(int(num_bins), bin_idx)

    ps0 = float(iplus0[bin_idx] + iminus0[bin_idx])
    if abs(ps0) < PS_ABS_MIN:
        return {
            "polarization": float(polarization),
            "skipped": True,
            "n_steps": 0,
            "ps": np.zeros(0, dtype=float),
            "iplus": np.zeros(0, dtype=float),
            "iminus": np.zeros(0, dtype=float),
            "ps_m": np.zeros(0, dtype=float),
            "iplus_m": np.zeros(0, dtype=float),
            "iminus_m": np.zeros(0, dtype=float),
            "ps0": ps0,
            "stop_reason": "skipped_tiny_ps0",
            "support": [],
        }

    profile, support = make_voigt_rf_profile(
        int(num_bins),
        bin_idx,
        float(gamma_rf),
        sigma=float(sigma_bins),
        lorentz_gamma=float(voigt_gamma_bins),
        half_width=int(half_width),
    )

    params = Spin1Params(
        n_bins=int(num_bins),
        r_min=-3.0,
        r_max=3.0,
        p0=float(polarization),
        q0=0.0,
        p_dnp_sat=float(polarization),
        dnp_enabled=False,
        rf_enabled=True,
        relax_enabled=True,
        afp_enabled=False,
        gamma_rf=float(gamma_rf),
        dt=float(dt),
        ssrf_subset_indices=[int(i) for i in support],
        rf_burn_R=float(f[bin_idx]),
        initial_polarization=float(polarization),
    )
    model = Spin1Model(params, initial_polarization=float(polarization))
    model.load_from_physical_intensities(iplus0, iminus0)
    freeze_rf_profile(model, profile)
    touched = ssrf_touched_bins(int(num_bins), support)
    # Restrict Euler + relaxation to RF-touched packets (support ∪ mirrors),
    # matching ssrf_afp commit-local behavior without per-step spectrum rewrites.
    model._active_idx = np.asarray(touched, dtype=int) if touched else None

    ps_t: list[float] = []
    ip_t: list[float] = []
    im_t: list[float] = []
    ps_m_t: list[float] = []
    ip_m_t: list[float] = []
    im_m_t: list[float] = []

    ip, im, ps, ip_m, im_m, ps_m = intensities_at_bins(model, bin_idx, mirror_idx)
    ps_t.append(ps)
    ip_t.append(ip)
    im_t.append(im)
    ps_m_t.append(ps_m)
    ip_m_t.append(ip_m)
    im_m_t.append(im_m)

    n_sub, dt_sub = euler_n_sub(float(gamma_rf), float(dt))
    steps_done = 0
    stop_reason = "max_steps"
    while steps_done < int(max_steps):
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
        steps_done += 1
        ip, im, ps, ip_m, im_m, ps_m = intensities_at_bins(model, bin_idx, mirror_idx)
        # Mirror turnover: max semi-saturating RF; discard this step and end.
        if mirror_amplitude_decreased(ps_m, ps_m_t[-1]):
            stop_reason = "mirror_decrease"
            break
        ps_t.append(ps)
        ip_t.append(ip)
        im_t.append(im)
        ps_m_t.append(ps_m)
        ip_m_t.append(ip_m)
        im_m_t.append(im_m)

    return {
        "polarization": float(polarization),
        "skipped": False,
        "n_steps": len(ps_t),
        "ps": np.asarray(ps_t, dtype=float),
        "iplus": np.asarray(ip_t, dtype=float),
        "iminus": np.asarray(im_t, dtype=float),
        "ps_m": np.asarray(ps_m_t, dtype=float),
        "iplus_m": np.asarray(ip_m_t, dtype=float),
        "iminus_m": np.asarray(im_m_t, dtype=float),
        "ps0": ps0,
        "stop_reason": stop_reason,
        "support": support,
    }


def run_one_bin(
    bin_idx: int,
    *,
    p_values: np.ndarray,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    gamma_rf: float = GAMMA_RF,
    sigma_bins: float = SIGMA_BINS,
    voigt_gamma_bins: float = VOIGT_GAMMA_BINS,
    half_width: int = HALF_WIDTH,
    max_steps: int = MAX_STEPS,
) -> dict:
    bin_idx = int(bin_idx)
    if bin_idx < 0 or bin_idx >= int(num_bins):
        raise ValueError(f"bin_idx={bin_idx} out of range for num_bins={num_bins}")

    mirror_idx = mirror_bin_idx(int(num_bins), bin_idx)
    p_values = np.asarray(p_values, dtype=float)
    n_p = int(p_values.size)
    t_max = int(max_steps) + 1

    n_steps = np.zeros(n_p, dtype=np.int32)
    skipped = np.zeros(n_p, dtype=bool)
    ps = np.full((n_p, t_max), np.nan, dtype=float)
    iplus = np.full((n_p, t_max), np.nan, dtype=float)
    iminus = np.full((n_p, t_max), np.nan, dtype=float)
    ps_m = np.full((n_p, t_max), np.nan, dtype=float)
    iplus_m = np.full((n_p, t_max), np.nan, dtype=float)
    iminus_m = np.full((n_p, t_max), np.nan, dtype=float)

    for j, p0 in enumerate(p_values):
        print(f"  P={p0:+.3f} ({j + 1}/{n_p})", flush=True)
        traj = run_one_polarization(
            bin_idx,
            float(p0),
            num_bins=num_bins,
            dt=dt,
            gamma_rf=gamma_rf,
            sigma_bins=sigma_bins,
            voigt_gamma_bins=voigt_gamma_bins,
            half_width=half_width,
            max_steps=max_steps,
        )
        skipped[j] = bool(traj["skipped"])
        n = int(traj["n_steps"])
        n_steps[j] = n
        if n <= 0:
            continue
        ps[j, :n] = traj["ps"]
        iplus[j, :n] = traj["iplus"]
        iminus[j, :n] = traj["iminus"]
        ps_m[j, :n] = traj["ps_m"]
        iplus_m[j, :n] = traj["iplus_m"]
        iminus_m[j, :n] = traj["iminus_m"]

    f = np.linspace(-3.0, 3.0, int(num_bins))
    return {
        "bin_idx": bin_idx,
        "mirror_idx": mirror_idx,
        "R": float(f[bin_idx]),
        "num_bins": int(num_bins),
        "dt": float(dt),
        "gamma_rf": float(gamma_rf),
        "sigma_bins": float(sigma_bins),
        "voigt_gamma_bins": float(voigt_gamma_bins),
        "half_width": int(half_width),
        "max_steps": int(max_steps),
        "p_values": p_values,
        "n_steps": n_steps,
        "skipped": skipped,
        "ps": ps,
        "iplus": iplus,
        "iminus": iminus,
        "ps_m": ps_m,
        "iplus_m": iplus_m,
        "iminus_m": iminus_m,
    }


def shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_bin_{int(bin_idx):04d}.npz"


def save_shard(result: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "bin_idx": int(result["bin_idx"]),
        "mirror_idx": int(result["mirror_idx"]),
        "R": float(result["R"]),
        "num_bins": int(result["num_bins"]),
        "dt": float(result["dt"]),
        "gamma_rf": float(result["gamma_rf"]),
        "sigma_bins": float(result["sigma_bins"]),
        "voigt_gamma_bins": float(result.get("voigt_gamma_bins", VOIGT_GAMMA_BINS)),
        "half_width": int(result["half_width"]),
        "max_steps": int(result["max_steps"]),
        "dataset": "ssrf_bin_traj",
    }
    ps = np.asarray(result["ps"], dtype=float)
    ps_m = np.asarray(result["ps_m"], dtype=float)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            p_values=np.asarray(result["p_values"], dtype=float),
            n_steps=np.asarray(result["n_steps"], dtype=np.int32),
            skipped=np.asarray(result["skipped"], dtype=bool),
            # Burn-bin observables
            ps=ps,
            iplus=np.asarray(result["iplus"], dtype=float),
            iminus=np.asarray(result["iminus"], dtype=float),
            amp=np.abs(ps),
            # Mirror-bin observables
            ps_m=ps_m,
            iplus_m=np.asarray(result["iplus_m"], dtype=float),
            iminus_m=np.asarray(result["iminus_m"], dtype=float),
            amp_m=np.abs(ps_m),
            bin_idx=np.asarray(int(result["bin_idx"]), dtype=np.int32),
            mirror_idx=np.asarray(int(result["mirror_idx"]), dtype=np.int32),
            dt=np.asarray(float(result["dt"]), dtype=float),
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def load_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        ps = np.asarray(data["ps"], dtype=float)
        ps_m = np.asarray(data["ps_m"], dtype=float)
        amp = np.asarray(data["amp"], dtype=float) if "amp" in data.files else np.abs(ps)
        amp_m = (
            np.asarray(data["amp_m"], dtype=float) if "amp_m" in data.files else np.abs(ps_m)
        )
        return {
            **meta,
            "p_values": np.asarray(data["p_values"], dtype=float),
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
            "ps": ps,
            "iplus": np.asarray(data["iplus"], dtype=float),
            "iminus": np.asarray(data["iminus"], dtype=float),
            "amp": amp,
            "ps_m": ps_m,
            "iplus_m": np.asarray(data["iplus_m"], dtype=float),
            "iminus_m": np.asarray(data["iminus_m"], dtype=float),
            "amp_m": amp_m,
        }


def train_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_train_bin_{int(bin_idx):04d}.npz"


def _empty_bin_bags(num_bins: int) -> list[dict[str, list[np.ndarray]]]:
    keys = (
        "p0",
        "step",
        "burn_bin",
        "is_mirror",
        "ps",
        "iplus",
        "iminus",
        "amp",
    )
    return [{k: [] for k in keys} for _ in range(int(num_bins))]


def _append_samples(
    bag: dict[str, list[np.ndarray]],
    *,
    p0: float,
    n: int,
    burn_bin: int,
    is_mirror: bool,
    ps: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    amp: np.ndarray,
) -> None:
    bag["p0"].append(np.full(n, float(p0), dtype=float))
    bag["step"].append(np.arange(n, dtype=np.int32))
    bag["burn_bin"].append(np.full(n, int(burn_bin), dtype=np.int32))
    bag["is_mirror"].append(np.full(n, bool(is_mirror), dtype=bool))
    bag["ps"].append(np.asarray(ps, dtype=float))
    bag["iplus"].append(np.asarray(iplus, dtype=float))
    bag["iminus"].append(np.asarray(iminus, dtype=float))
    bag["amp"].append(np.asarray(amp, dtype=float))


def _finalize_bag(bag: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    if not bag["ps"]:
        return {
            "p0": np.zeros(0, dtype=float),
            "step": np.zeros(0, dtype=np.int32),
            "burn_bin": np.zeros(0, dtype=np.int32),
            "is_mirror": np.zeros(0, dtype=bool),
            "ps": np.zeros(0, dtype=float),
            "iplus": np.zeros(0, dtype=float),
            "iminus": np.zeros(0, dtype=float),
            "amp": np.zeros(0, dtype=float),
        }
    return {k: np.concatenate(v) for k, v in bag.items()}


def save_train_bin(
    bin_idx: int,
    arrays: dict[str, np.ndarray],
    path: Path,
    *,
    n_missing: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(np.asarray(arrays["ps"]).size)
    meta = {
        "bin_idx": int(bin_idx),
        "n_samples": n_samples,
        "n_missing_shards": int(n_missing),
        "dataset": "ssrf_train_bin",
        "fields": "ps,iplus,iminus,amp at this bin; burn_bin=RF center; is_mirror",
    }
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            bin_idx=np.asarray(int(bin_idx), dtype=np.int32),
            p0=np.asarray(arrays["p0"], dtype=float),
            step=np.asarray(arrays["step"], dtype=np.int32),
            burn_bin=np.asarray(arrays["burn_bin"], dtype=np.int32),
            is_mirror=np.asarray(arrays["is_mirror"], dtype=bool),
            ps=np.asarray(arrays["ps"], dtype=float),
            iplus=np.asarray(arrays["iplus"], dtype=float),
            iminus=np.asarray(arrays["iminus"], dtype=float),
            amp=np.asarray(arrays["amp"], dtype=float),
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def organize_shards(
    shard_dir: Path,
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    strict: bool = True,
) -> dict:
    """
    Route shard samples into one training NPZ per spectral bin.

    Each burn shard records amplitudes at burn bin *and* mirror bin. Those
    observations are filed under their respective bin indices so each of the
    ``num_bins`` models can train independently.
    """
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing: list[int] = []
    bags = _empty_bin_bags(num_bins)

    for burn_bin in range(int(num_bins)):
        path = shard_path(shard_dir, burn_bin)
        if not path.is_file():
            missing.append(burn_bin)
            continue
        shard = load_shard(path)
        b = int(shard["bin_idx"])
        m = int(shard["mirror_idx"])
        p_values = np.asarray(shard["p_values"], dtype=float)
        n_steps = np.asarray(shard["n_steps"], dtype=np.int32)
        for j, p0 in enumerate(p_values):
            n = int(n_steps[j])
            if n <= 0:
                continue
            _append_samples(
                bags[b],
                p0=float(p0),
                n=n,
                burn_bin=b,
                is_mirror=False,
                ps=shard["ps"][j, :n],
                iplus=shard["iplus"][j, :n],
                iminus=shard["iminus"][j, :n],
                amp=shard["amp"][j, :n],
            )
            if m != b:
                _append_samples(
                    bags[m],
                    p0=float(p0),
                    n=n,
                    burn_bin=b,
                    is_mirror=True,
                    ps=shard["ps_m"][j, :n],
                    iplus=shard["iplus_m"][j, :n],
                    iminus=shard["iminus_m"][j, :n],
                    amp=shard["amp_m"][j, :n],
                )

    if missing and strict:
        raise FileNotFoundError(
            f"Missing {len(missing)} shard(s) under {shard_dir}; "
            f"first missing bin_idx={missing[0]}"
        )
    if missing:
        print(f"WARNING: missing {len(missing)} shards; continuing", flush=True)

    samples_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    for bin_idx in range(int(num_bins)):
        arrays = _finalize_bag(bags[bin_idx])
        samples_per_bin[bin_idx] = int(arrays["ps"].size)
        save_train_bin(
            bin_idx,
            arrays,
            train_bin_path(output_dir, bin_idx),
            n_missing=len(missing),
        )

    return {
        "output_dir": str(output_dir),
        "samples_per_bin": samples_per_bin,
        "n_samples": int(samples_per_bin.sum()),
        "n_missing": len(missing),
        "dataset": "ssrf_train_bin",
    }


def _resolve_bin_idx(cli_bin_idx: int | None) -> int | None:
    if cli_bin_idx is not None:
        return int(cli_bin_idx)
    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_idx is not None and str(env_idx).strip() != "":
        return int(env_idx)
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-bin ssRF Voigt burn trajectory worker / per-bin organizer"
    )
    p.add_argument("--bin-idx", type=int, default=None)
    p.add_argument(
        "--organize",
        "--combine",
        dest="organize",
        action="store_true",
        help="Organize shards into one training NPZ per bin (alias: --combine)",
    )
    p.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directory for per-bin training NPZs (organize mode)",
    )
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--p-min", type=float, default=P_MIN)
    p.add_argument("--p-max", type=float, default=P_MAX)
    p.add_argument("--p-step", type=float, default=P_STEP)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--gamma-rf", type=float, default=GAMMA_RF)
    p.add_argument("--sigma-bins", type=float, default=SIGMA_BINS)
    p.add_argument("--voigt-gamma-bins", type=float, default=VOIGT_GAMMA_BINS)
    p.add_argument("--half-width", type=int, default=HALF_WIDTH)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--skip-if-exists", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.organize:
        result = organize_shards(
            args.shard_dir,
            args.output_dir,
            num_bins=args.num_bins,
            strict=bool(args.strict),
        )
        print(
            f"Organized {result['n_samples']} samples from {args.shard_dir} -> "
            f"{args.output_dir} ({args.num_bins} bin files; "
            f"missing={result.get('n_missing', 0)})",
            flush=True,
        )
        return

    bin_idx = _resolve_bin_idx(args.bin_idx)
    if bin_idx is None:
        raise SystemExit(
            "Provide --bin-idx <int>, or set SLURM_ARRAY_TASK_ID, or pass --organize"
        )

    out = shard_path(args.shard_dir, bin_idx)
    if args.skip_if_exists and out.is_file():
        print(f"Skipping existing shard {out}", flush=True)
        return

    p_values = polarization_grid(args.p_min, args.p_max, args.p_step)
    print(
        f"bin_idx={bin_idx}  n_P={p_values.size}  "
        f"P=[{args.p_min},{args.p_max}] step={args.p_step}  "
        f"dt={args.dt}  gamma={args.gamma_rf}  "
        f"sigma={args.sigma_bins}  voigt_gamma={args.voigt_gamma_bins}  +/-{args.half_width}  "
        f"max_steps={args.max_steps}  stop=mirror_turnover",
        flush=True,
    )
    result = run_one_bin(
        bin_idx,
        p_values=p_values,
        num_bins=args.num_bins,
        dt=args.dt,
        gamma_rf=args.gamma_rf,
        sigma_bins=args.sigma_bins,
        voigt_gamma_bins=args.voigt_gamma_bins,
        half_width=args.half_width,
        max_steps=args.max_steps,
    )
    save_shard(result, out)
    print(
        f"Wrote {out}  mirror={result['mirror_idx']}  "
        f"mean_steps={float(np.mean(result['n_steps'])):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
