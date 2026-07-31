"""
Global full-spectrum model: Ps[500] -> (I+[500], I-[500]) via alpha-parameterization.

Inference path (Ps-only):
  1. Normalize Ps per sample (divide by max abs or total area).
  2. Optionally estimate P from spectral area and append delta_Ps = Ps - Ps_eq(P).
  3. Run SpectrumSplitNet to predict alpha[500] in (0, 1).
  4. Reconstruct I+ = alpha * Ps, I- = (1 - alpha) * Ps (exact I+ + I- = Ps).

Training:
  python ml/spectrum_split_model.py train \\
      --data Data_Creation/dulya_fit_v2/data/spectrum_train \\
      --output ml/models/spectrum_split_model.pth

Evaluation:
  python ml/test_spectrum_split.py \\
      --model ml/models/spectrum_split_model.pth \\
      --data Data_Creation/dulya_fit_v2/data/spectrum_train
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

_SCRIPT_DIR = Path(__file__).resolve().parent
# Local layout: ml/spectrum_split_model.py → repo root is parents[1].
# Cluster layout: script often lives next to dulya_fit_v4/ (e.g. RF/).
_REPO_CANDIDATES = (
    _SCRIPT_DIR.parent,  # .../Spin-1-Manipulation
    _SCRIPT_DIR,  # .../RF or similar
)
REPO_ROOT = next(
    (
        root
        for root in _REPO_CANDIDATES
        if (root / "Data_Creation" / "dulya_fit_v4").is_dir()
        or (root / "dulya_fit_v4").is_dir()
        or (root / "Data_Creation" / "dulya_fit_v2").is_dir()
        or (root / "dulya_fit_v2").is_dir()
    ),
    _SCRIPT_DIR.parent,
)

def _resolve_dulya_pkg() -> Path:
    for root in (REPO_ROOT, _SCRIPT_DIR):
        for name in ("dulya_fit_v4", "dulya_fit_v2"):
            for candidate in (root / "Data_Creation" / name, root / name):
                if candidate.is_dir():
                    return candidate
    raise ModuleNotFoundError(
        "Could not locate dulya_fit_v4/v2 (need bin_io.py). "
        f"Looked under {REPO_ROOT} and {_SCRIPT_DIR}."
    )


DULYA = _resolve_dulya_pkg()
if str(DULYA) not in sys.path:
    sys.path.insert(0, str(DULYA))

from bin_io import SPECTRUM_TRAIN_MANIFEST_NAME
from bin_setup import equilibrium_lineshape, get_shape_params, spin1_scale_factors
from common import F_MAX, F_MIN, NUM_BINS
from lineshape import GenerateDulyaLineshape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

DEFAULT_DATA = "spectrum_train.npz"
DEFAULT_OUTPUT = "models/spectrum_split_model.pth"

TRAIN_POLARIZATION_FRACTION = 0.8
NUM_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25
MIN_DELTA = 1e-6
ALPHA_BOUND_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.001
MASKED_ALPHA_LOSS_WEIGHT = 0.01
Q_LOSS_WEIGHT = 0.01
ALPHA_MASK_REL_THRESHOLD = 1e-3

_AMP_CALIBRATION_CACHE: dict[tuple, dict[str, np.ndarray | float | int]] = {}


def build_amp_integration_calibration(
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
    p_min: float = -0.9,
    p_max: float = 0.9,
    p_step: float = 0.05,
) -> dict[str, np.ndarray | float | int]:
    """Equilibrium map from naive amp integration ``sum/(amp*n)`` to true ``P``."""
    shape = shape_params if shape_params is not None else get_shape_params()
    amp = float(shape["amp"])
    n_bins = int(num_bins)
    f = np.linspace(float(F_MIN), float(F_MAX), n_bins)
    p_input: list[float] = []
    p_amp_naive: list[float] = []
    for p0 in np.arange(float(p_min), float(p_max) + 1e-12, float(p_step)):
        if abs(float(p0)) < 1e-12:
            continue
        _, ip, im = GenerateDulyaLineshape(float(p0), f, shape)
        p_sum = float(np.sum(ip + im))
        q_sum = float(np.sum(ip - im))
        denom = amp * float(n_bins)
        if abs(denom) < 1e-30:
            continue
        p_amp = p_sum / denom
        p_input.append(float(p0))
        p_amp_naive.append(float(p_amp))
    return {
        "p_input": np.asarray(p_input, dtype=float),
        "p_amp_naive": np.asarray(p_amp_naive, dtype=float),
        "amp": amp,
        "n_bins": n_bins,
    }


def _get_amp_integration_calibration(
    *,
    num_bins: int,
    shape_params: dict[str, float] | None,
) -> dict[str, np.ndarray | float | int]:
    shape = shape_params if shape_params is not None else get_shape_params()
    key = (int(num_bins), round(float(shape["amp"]), 12))
    cached = _AMP_CALIBRATION_CACHE.get(key)
    if cached is not None:
        return cached
    cal = build_amp_integration_calibration(num_bins=int(num_bins), shape_params=shape)
    _AMP_CALIBRATION_CACHE[key] = cal
    return cal


def integrated_polarizations_calibrated(
    iplus: np.ndarray,
    iminus: np.ndarray,
    *,
    num_bins: int | None = None,
    shape_params: dict[str, float] | None = None,
    post_correct: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lineshape-integrated vector P and tensor Q with Dulya ``amp`` normalization.

    Fit-scale stored spectra integrate as ``sum(I±) / (amp * n_bins)``; optional
    equilibrium post-correction inverts the nonlinear amp map for ``P`` and scales
    ``Q`` by the same factor (see ``Data_Creation/dulya_fit/polarization.py``).
    """
    ip = np.asarray(iplus, dtype=float)
    im = np.asarray(iminus, dtype=float)
    if ip.ndim == 1:
        ip = ip.reshape(1, -1)
        im = im.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    n_bins = int(num_bins if num_bins is not None else ip.shape[1])
    shape = shape_params if shape_params is not None else get_shape_params()
    amp = float(shape["amp"])
    denom = amp * float(n_bins)
    if abs(denom) < 1e-30:
        raise ValueError("amp * num_bins normalization factor is zero")

    p_sum = np.sum(ip + im, axis=1)
    q_sum = np.sum(ip - im, axis=1)
    p_naive = p_sum / denom
    q_naive = q_sum / denom

    if post_correct:
        cal = _get_amp_integration_calibration(num_bins=n_bins, shape_params=shape)
        p_axis = np.asarray(cal["p_input"], dtype=float)
        p_amp_axis = np.asarray(cal["p_amp_naive"], dtype=float)
        p_corr = np.interp(p_naive, p_amp_axis, p_axis)
        ratio = np.where(np.abs(p_naive) > 1e-30, p_corr / p_naive, 1.0)
        q_corr = q_naive * ratio
        p_out, q_out = p_corr, q_corr
    else:
        p_out, q_out = p_naive, q_naive

    if squeeze:
        return np.asarray(p_out[0], dtype=float), np.asarray(q_out[0], dtype=float)
    return np.asarray(p_out, dtype=float), np.asarray(q_out, dtype=float)


@dataclass
class SpectrumSample:
    ps: np.ndarray
    iplus: np.ndarray
    iminus: np.ndarray
    p0: float
    source: int = 0


def estimate_P_from_area(
    ps: np.ndarray,
    p0_hint: float | None = None,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> float:
    """Estimate vector polarization P from total spectral area (spin1 scaling)."""
    ps_arr = np.asarray(ps, dtype=float).reshape(-1)
    area = float(np.sum(ps_arr))
    if abs(area) < 1e-30:
        return float(p0_hint or 0.0)

    shape = shape_params if shape_params is not None else get_shape_params()
    f = np.linspace(float(F_MIN), float(F_MAX), int(num_bins))
    p_grid = np.linspace(-0.9, 0.9, 37, dtype=float)
    best_p = float(p0_hint or 0.0)
    best_err = float("inf")
    for p_try in p_grid:
        _, ip_eq, im_eq = equilibrium_lineshape(float(p_try), f, shape)
        to_spin1, _ = spin1_scale_factors(float(p_try), ip_eq, im_eq)
        area_eq = float(np.sum((ip_eq + im_eq) * to_spin1))
        err = abs(area - area_eq)
        if err < best_err:
            best_err = err
            best_p = float(p_try)
    return best_p


def compute_ps_eq(
    p_est: float,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dulya equilibrium Ps, I+, I- at fit scale."""
    f = np.linspace(float(F_MIN), float(F_MAX), int(num_bins))
    return GenerateDulyaLineshape(
        float(p_est),
        f,
        shape_params if shape_params is not None else get_shape_params(),
    )


def alpha_from_intensities(
    ps: np.ndarray, iplus: np.ndarray, iminus: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    ps_arr = np.asarray(ps, dtype=float)
    denom = np.maximum(np.abs(ps_arr), eps)
    return np.clip(np.asarray(iplus, dtype=float) / denom, 0.0, 1.0)


def normalize_ps(ps: np.ndarray, mode: str = "max") -> tuple[np.ndarray, float]:
    ps_arr = np.asarray(ps, dtype=float)
    if mode == "area":
        scale = float(np.sum(np.abs(ps_arr)))
    else:
        scale = float(np.max(np.abs(ps_arr)))
    if scale < 1e-30:
        scale = 1.0
    return ps_arr / scale, scale


class ConvBlock1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectrumSplitNet(nn.Module):
    """1D U-Net predicting per-bin alpha in (0, 1)."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, num_bins: int = NUM_BINS):
        super().__init__()
        self.num_bins = int(num_bins)
        self.padded_len = int(2 ** int(np.ceil(np.log2(max(self.num_bins, 2)))))
        c = int(base_channels)
        self.enc1 = ConvBlock1d(in_channels, c)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1d(c, c * 2)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock1d(c * 2, c * 4)
        self.pool3 = nn.MaxPool1d(2)

        self.bottleneck = ConvBlock1d(c * 4, c * 8)

        self.up3 = nn.ConvTranspose1d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock1d(c * 8, c * 4)
        self.up2 = nn.ConvTranspose1d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1d(c * 4, c * 2)
        self.up1 = nn.ConvTranspose1d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1d(c * 2, c)

        self.head = nn.Conv1d(c, 1, kernel_size=1)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if length >= self.padded_len:
            return x[..., : self.padded_len]
        pad = self.padded_len - length
        return torch.nn.functional.pad(x, (0, pad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape[-1] != e3.shape[-1]:
            d3 = torch.nn.functional.interpolate(d3, size=e3.shape[-1], mode="linear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = torch.nn.functional.interpolate(d2, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = torch.nn.functional.interpolate(d1, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        alpha = torch.sigmoid(self.head(d1)).squeeze(1)
        return alpha[..., : self.num_bins]


def build_input_channels(
    ps_batch: np.ndarray,
    *,
    use_residual: bool,
    shape_params: dict[str, float] | None = None,
) -> np.ndarray:
    """Build (B, C, L) network input from Ps spectra without true-P hints."""
    ps_batch = np.asarray(ps_batch, dtype=np.float32)
    b, length = ps_batch.shape
    channels = [ps_batch]
    if use_residual:
        delta = np.zeros_like(ps_batch, dtype=np.float32)
        shape = shape_params if shape_params is not None else get_shape_params()
        for i in range(b):
            p_est = estimate_P_from_area(ps_batch[i], None, num_bins=length, shape_params=shape)
            ps_eq, _, _ = compute_ps_eq(p_est, num_bins=length, shape_params=shape)
            ps_norm, _ = normalize_ps(ps_batch[i])
            eq_norm, _ = normalize_ps(ps_eq)
            delta[i] = (ps_norm - eq_norm).astype(np.float32)
        channels.append(delta)
    stacked = np.stack(channels, axis=1)
    return stacked


def physics_loss(
    alpha: torch.Tensor,
    ps: torch.Tensor,
    iplus_true: torch.Tensor,
    iminus_true: torch.Tensor,
    *,
    alpha_true: torch.Tensor | None = None,
    alpha_bound_weight: float = ALPHA_BOUND_WEIGHT,
    smoothness_weight: float = SMOOTHNESS_WEIGHT,
    masked_alpha_weight: float = MASKED_ALPHA_LOSS_WEIGHT,
    q_loss_weight: float = Q_LOSS_WEIGHT,
    alpha_mask_rel_threshold: float = ALPHA_MASK_REL_THRESHOLD,
) -> torch.Tensor:
    """Optimize per-bin intensities, signal-region alpha, and integrated Q."""
    iplus_pred = alpha * ps
    iminus_pred = (1.0 - alpha) * ps
    l1 = torch.mean(torch.abs(iplus_pred - iplus_true)) + torch.mean(
        torch.abs(iminus_pred - iminus_true)
    )

    if alpha_true is None:
        alpha_true = torch.where(
            torch.abs(ps) > 1e-12,
            iplus_true / torch.clamp(torch.abs(ps), min=1e-12),
            torch.zeros_like(ps),
        )
    peak = torch.amax(torch.abs(ps), dim=1, keepdim=True)
    alpha_mask = torch.abs(ps) >= float(alpha_mask_rel_threshold) * torch.clamp(
        peak, min=1e-12
    )
    alpha_mask_f = alpha_mask.to(dtype=alpha.dtype)
    masked_alpha = torch.sum(torch.abs(alpha - alpha_true) * alpha_mask_f) / torch.clamp(
        torch.sum(alpha_mask_f), min=1.0
    )

    q_pred = torch.sum(iplus_pred - iminus_pred, dim=1)
    q_true = torch.sum(iplus_true - iminus_true, dim=1)
    signal_area = torch.sum(torch.abs(ps), dim=1)
    q_loss = torch.mean(
        torch.abs(q_pred - q_true) / torch.clamp(signal_area, min=1e-12)
    )

    bound = torch.mean(torch.relu(-alpha) + torch.relu(alpha - 1.0))
    if alpha.shape[1] > 1:
        smooth = torch.mean(torch.abs(alpha[:, 1:] - alpha[:, :-1]))
    else:
        smooth = torch.zeros((), device=alpha.device)
    return (
        l1
        + float(masked_alpha_weight) * masked_alpha
        + float(q_loss_weight) * q_loss
        + float(alpha_bound_weight) * bound
        + float(smoothness_weight) * smooth
    )


@dataclass
class InferenceResult:
    iplus: np.ndarray
    iminus: np.ndarray
    alpha: np.ndarray
    p_est: float


class SpectrumSplitModel:
    """Wrapper for trained SpectrumSplitNet with Ps-only inference."""

    def __init__(
        self,
        net: SpectrumSplitNet,
        *,
        use_residual: bool = True,
        normalize_mode: str = "max",
        shape_params: dict[str, float] | None = None,
        num_bins: int = NUM_BINS,
    ):
        self.net = net.eval()
        self.use_residual = bool(use_residual)
        self.normalize_mode = str(normalize_mode)
        self.shape_params = shape_params if shape_params is not None else get_shape_params()
        self.num_bins = int(num_bins)

    @torch.no_grad()
    def predict_ps(
        self,
        ps: np.ndarray,
        *,
        p0_hint: float | None = None,
        device: torch.device | None = None,
    ) -> InferenceResult:
        """Split one Ps spectrum; ``p0_hint`` is retained for API compatibility only."""
        dev = device or next(self.net.parameters()).device
        ps_arr = np.asarray(ps, dtype=np.float32).reshape(1, -1)
        ps_norm, _ = normalize_ps(ps_arr[0], mode=self.normalize_mode)
        p_est = estimate_P_from_area(
            ps_arr[0],
            None,
            num_bins=self.num_bins,
            shape_params=self.shape_params,
        )
        x_np = build_input_channels(
            ps_norm.reshape(1, -1),
            use_residual=self.use_residual,
            shape_params=self.shape_params,
        )
        x = torch.from_numpy(x_np).float().to(dev)
        alpha = self.net(x).cpu().numpy()[0]
        ps_phys = ps_arr[0]
        iplus = alpha * ps_phys
        iminus = (1.0 - alpha) * ps_phys
        return InferenceResult(iplus=iplus, iminus=iminus, alpha=alpha, p_est=p_est)

    @torch.no_grad()
    def predict_batch(
        self,
        ps_batch: np.ndarray,
        *,
        p0_batch: np.ndarray | None = None,
        device: torch.device | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split Ps spectra; ``p0_batch`` is retained for API compatibility only."""
        dev = device or next(self.net.parameters()).device
        ps_batch = np.asarray(ps_batch, dtype=np.float32)
        b = ps_batch.shape[0]
        ps_norm = np.zeros_like(ps_batch)
        for i in range(b):
            ps_norm[i], _ = normalize_ps(ps_batch[i], mode=self.normalize_mode)
        x_np = build_input_channels(
            ps_norm,
            use_residual=self.use_residual,
            shape_params=self.shape_params,
        )
        x = torch.from_numpy(x_np).float().to(dev)
        alpha = self.net(x).cpu().numpy()
        iplus = alpha * ps_batch
        iminus = (1.0 - alpha) * ps_batch
        return iplus, iminus


def _build_sample(
    ps: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    *,
    use_residual: bool,
    normalize_mode: str,
    shape_params: dict[str, float],
) -> dict[str, torch.Tensor]:
    ps_norm, scale = normalize_ps(ps, mode=normalize_mode)
    alpha = alpha_from_intensities(ps, iplus, iminus).astype(np.float32)
    x_np = build_input_channels(
        ps_norm.reshape(1, -1),
        use_residual=use_residual,
        shape_params=shape_params,
    )[0]
    return {
        "x": torch.from_numpy(x_np).float(),
        "ps": torch.from_numpy(ps_norm).float(),
        "alpha": torch.from_numpy(alpha).float(),
        "iplus": torch.from_numpy(np.asarray(iplus, dtype=np.float32) / max(scale, 1e-30)).float(),
        "iminus": torch.from_numpy(np.asarray(iminus, dtype=np.float32) / max(scale, 1e-30)).float(),
        "scale": torch.tensor(float(scale), dtype=torch.float32),
    }


def is_sharded_source(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() or path.name == SPECTRUM_TRAIN_MANIFEST_NAME


def discover_spectrum_shards(path: Path) -> list[Path]:
    """Resolve --data into shard NPZ paths (single file, manifest, or glob)."""
    path = Path(path)
    if path.is_file() and path.name != SPECTRUM_TRAIN_MANIFEST_NAME:
        return [path]

    manifest_path = path / SPECTRUM_TRAIN_MANIFEST_NAME if path.is_dir() else path
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        shard_dir = manifest_path.parent
        return [shard_dir / name for name in manifest["shard_files"]]

    if path.is_dir():
        shards = sorted(path.glob("spectrum_train_*.npz"))
        if shards:
            return shards

    raise FileNotFoundError(f"No spectrum shards found at {path}")


def read_light_columns(shard_paths: list[Path]) -> dict[str, np.ndarray]:
    """Load p0/source (and shard row indices) without reading ps/iplus/iminus."""
    p0_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    shard_id_parts: list[np.ndarray] = []
    local_idx_parts: list[np.ndarray] = []
    for shard_id, path in enumerate(shard_paths):
        with np.load(path, allow_pickle=False) as npz:
            p0 = np.asarray(npz["p0"], dtype=np.float32)
            source = (
                np.asarray(npz["source"], dtype=np.uint8)
                if "source" in npz.files
                else np.zeros(p0.shape[0], dtype=np.uint8)
            )
        n = int(p0.shape[0])
        p0_parts.append(p0)
        source_parts.append(source)
        shard_id_parts.append(np.full(n, shard_id, dtype=np.int32))
        local_idx_parts.append(np.arange(n, dtype=np.int64))
    return {
        "p0": np.concatenate(p0_parts),
        "source": np.concatenate(source_parts),
        "shard_id": np.concatenate(shard_id_parts),
        "local_idx": np.concatenate(local_idx_parts),
    }


def row_mask_by_shard(
    shard_id: np.ndarray,
    local_idx: np.ndarray,
    shard_row_counts: list[int],
    selected: np.ndarray,
) -> dict[int, np.ndarray]:
    sel_shard_id = shard_id[selected]
    sel_local_idx = local_idx[selected]
    masks: dict[int, np.ndarray] = {}
    for sid, count in enumerate(shard_row_counts):
        rows = sel_local_idx[sel_shard_id == sid]
        if rows.size == 0:
            continue
        mask = np.zeros(int(count), dtype=bool)
        mask[rows] = True
        masks[sid] = mask
    return masks


class ShardedSpectrumDataset(data.IterableDataset):
    """Iterable dataset over sharded spectrum_train NPZs."""

    def __init__(
        self,
        shard_paths: list[Path],
        row_mask_by_shard: dict[int, np.ndarray],
        *,
        use_residual: bool,
        normalize_mode: str = "max",
        shape_params: dict[str, float] | None = None,
        shuffle: bool = True,
        seed: int = SEED,
    ):
        self.shard_paths = list(shard_paths)
        self.row_mask_by_shard = row_mask_by_shard
        self.use_residual = bool(use_residual)
        self.normalize_mode = str(normalize_mode)
        self.shape_params = shape_params if shape_params is not None else get_shape_params()
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return int(sum(int(mask.sum()) for mask in self.row_mask_by_shard.values()))

    def __iter__(self):
        worker = data.get_worker_info()
        shard_ids = list(self.row_mask_by_shard.keys())
        self._epoch += 1
        if self.shuffle:
            np.random.default_rng(self.seed + self._epoch).shuffle(shard_ids)
        if worker is not None:
            shard_ids = shard_ids[worker.id :: worker.num_workers]

        for shard_id in shard_ids:
            mask = self.row_mask_by_shard[shard_id]
            with np.load(self.shard_paths[shard_id], allow_pickle=False) as npz:
                ps = np.asarray(npz["ps"], dtype=np.float32)[mask]
                iplus = np.asarray(npz["iplus"], dtype=np.float32)[mask]
                iminus = np.asarray(npz["iminus"], dtype=np.float32)[mask]

            order = np.arange(ps.shape[0])
            if self.shuffle:
                np.random.default_rng(self.seed + self._epoch + shard_id).shuffle(order)

            for j in order:
                yield _build_sample(
                    ps[j],
                    iplus[j],
                    iminus[j],
                    use_residual=self.use_residual,
                    normalize_mode=self.normalize_mode,
                    shape_params=self.shape_params,
                )


def load_spectrum_npz(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        out = {k: np.asarray(data[k]) for k in data.files if k != "meta_json"}
        if "meta_json" in data.files:
            out["meta_json"] = np.asarray(data["meta_json"])
    for req in ("ps", "iplus", "iminus", "p0"):
        if req not in out:
            raise KeyError(f"{path}: missing required field {req!r}")
    return out


class SpectrumDataset(data.Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: np.ndarray,
        *,
        use_residual: bool,
        normalize_mode: str = "max",
        shape_params: dict[str, float] | None = None,
    ):
        self.ps = np.asarray(arrays["ps"], dtype=np.float32)
        self.iplus = np.asarray(arrays["iplus"], dtype=np.float32)
        self.iminus = np.asarray(arrays["iminus"], dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.use_residual = bool(use_residual)
        self.normalize_mode = str(normalize_mode)
        self.shape_params = shape_params if shape_params is not None else get_shape_params()

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        j = int(self.indices[idx])
        return _build_sample(
            self.ps[j],
            self.iplus[j],
            self.iminus[j],
            use_residual=self.use_residual,
            normalize_mode=self.normalize_mode,
            shape_params=self.shape_params,
        )


def polarization_train_holdout_split(p0: np.ndarray, fraction: float = TRAIN_POLARIZATION_FRACTION) -> tuple[np.ndarray, np.ndarray]:
    p0 = np.asarray(p0, dtype=float)
    uniq = np.unique(np.round(p0, 4))
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    n_train = max(1, int(round(float(fraction) * uniq.size)))
    train_vals = set(uniq[:n_train].tolist())
    train_mask = np.array([round(float(v), 4) in train_vals for v in p0], dtype=bool)
    holdout_mask = ~train_mask
    return np.flatnonzero(train_mask), np.flatnonzero(holdout_mask)


def train_model(
    data_path: Path,
    output_path: Path,
    *,
    use_residual: bool = True,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    masked_alpha_weight: float = MASKED_ALPHA_LOSS_WEIGHT,
    q_loss_weight: float = Q_LOSS_WEIGHT,
    alpha_mask_rel_threshold: float = ALPHA_MASK_REL_THRESHOLD,
) -> dict[str, Any]:
    data_path = Path(data_path)
    in_channels = 2 if use_residual else 1

    if is_sharded_source(data_path):
        shard_paths = discover_spectrum_shards(data_path)
        light = read_light_columns(shard_paths)
        shard_row_counts = np.bincount(light["shard_id"], minlength=len(shard_paths)).tolist()
        train_sel, holdout_sel = polarization_train_holdout_split(light["p0"])
        train_masks = row_mask_by_shard(light["shard_id"], light["local_idx"], shard_row_counts, train_sel)
        holdout_masks = row_mask_by_shard(light["shard_id"], light["local_idx"], shard_row_counts, holdout_sel)

        with np.load(shard_paths[0], allow_pickle=False) as npz:
            num_bins = int(npz["ps"].shape[1])
        net = SpectrumSplitNet(in_channels=in_channels, num_bins=num_bins).to(DEVICE)

        train_ds = ShardedSpectrumDataset(shard_paths, train_masks, use_residual=use_residual, shuffle=True)
        holdout_ds = (
            ShardedSpectrumDataset(shard_paths, holdout_masks, use_residual=use_residual, shuffle=False)
            if holdout_masks
            else None
        )
        train_loader = data.DataLoader(train_ds, batch_size=batch_size)
        holdout_loader = data.DataLoader(holdout_ds, batch_size=batch_size) if holdout_ds is not None else None
        n_train, n_holdout = len(train_ds), (len(holdout_ds) if holdout_ds is not None else 0)
        train_idx, holdout_idx = None, None
    else:
        arrays = load_spectrum_npz(data_path)
        train_sel, holdout_sel = polarization_train_holdout_split(arrays["p0"])
        num_bins = int(arrays["ps"].shape[1])
        net = SpectrumSplitNet(in_channels=in_channels, num_bins=num_bins).to(DEVICE)

        train_ds = SpectrumDataset(arrays, train_sel, use_residual=use_residual)
        holdout_ds = SpectrumDataset(arrays, holdout_sel, use_residual=use_residual) if holdout_sel.size else None
        train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        holdout_loader = (
            data.DataLoader(holdout_ds, batch_size=batch_size, shuffle=False) if holdout_ds else None
        )
        n_train, n_holdout = int(train_sel.size), int(holdout_sel.size)
        train_idx, holdout_idx = train_sel, holdout_sel

    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(int(num_epochs)):
        net.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(DEVICE)
            ps = batch["ps"].to(DEVICE)
            alpha_true = batch["alpha"].to(DEVICE)
            alpha_pred = net(x)
            loss = physics_loss(
                alpha_pred,
                ps,
                alpha_true * ps,
                (1.0 - alpha_true) * ps,
                alpha_true=alpha_true,
                masked_alpha_weight=masked_alpha_weight,
                q_loss_weight=q_loss_weight,
                alpha_mask_rel_threshold=alpha_mask_rel_threshold,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item())
            n_batches += 1

        train_loss /= max(n_batches, 1)
        val_loss = train_loss
        if holdout_loader is not None:
            net.eval()
            val_acc = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in holdout_loader:
                    x = batch["x"].to(DEVICE)
                    ps = batch["ps"].to(DEVICE)
                    alpha_true = batch["alpha"].to(DEVICE)
                    alpha_pred = net(x)
                    val_acc += float(
                        physics_loss(
                            alpha_pred,
                            ps,
                            alpha_true * ps,
                            (1.0 - alpha_true) * ps,
                            alpha_true=alpha_true,
                            masked_alpha_weight=masked_alpha_weight,
                            q_loss_weight=q_loss_weight,
                            alpha_mask_rel_threshold=alpha_mask_rel_threshold,
                        ).item()
                    )
                    val_n += 1
            val_loss = val_acc / max(val_n, 1)
            scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"epoch {epoch + 1}/{num_epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}",
                flush=True,
            )

        if val_loss + MIN_DELTA < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stop at epoch {epoch + 1}", flush=True)
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "in_channels": in_channels,
        "use_residual": use_residual,
        "num_bins": num_bins,
        "normalize_mode": "max",
        "shape_params": get_shape_params(),
        "masked_alpha_loss_weight": float(masked_alpha_weight),
        "q_loss_weight": float(q_loss_weight),
        "alpha_mask_rel_threshold": float(alpha_mask_rel_threshold),
        "train_indices": train_idx,
        "holdout_indices": holdout_idx,
        "best_val_loss": best_loss,
    }
    torch.save(payload, output_path)
    return {"output": str(output_path), "best_val_loss": best_loss, "n_train": n_train, "n_holdout": n_holdout}


def load_trained_model(path: Path, device: torch.device | None = None) -> SpectrumSplitModel:
    dev = device or DEVICE
    payload = torch.load(Path(path), map_location=dev, weights_only=False)
    in_channels = int(payload.get("in_channels", 2 if payload.get("use_residual", True) else 1))
    net = SpectrumSplitNet(
        in_channels=in_channels,
        num_bins=int(payload.get("num_bins", NUM_BINS)),
    )
    net.load_state_dict(payload["state_dict"])
    net.to(dev)
    return SpectrumSplitModel(
        net,
        use_residual=bool(payload.get("use_residual", True)),
        normalize_mode=str(payload.get("normalize_mode", "max")),
        shape_params=payload.get("shape_params"),
        num_bins=int(payload.get("num_bins", NUM_BINS)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train/evaluate global spectrum split model")
    sub = p.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train SpectrumSplitNet")
    train_p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Single spectrum_train.npz, or a sharded output directory/manifest",
    )
    train_p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train_p.add_argument("--no-residual", action="store_true", help="Disable delta_Ps channel")
    train_p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    train_p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train_p.add_argument("--lr", type=float, default=LEARNING_RATE)
    train_p.add_argument(
        "--masked-alpha-weight",
        type=float,
        default=MASKED_ALPHA_LOSS_WEIGHT,
        help="Weight for alpha L1 over bins with appreciable Ps",
    )
    train_p.add_argument(
        "--q-loss-weight",
        type=float,
        default=Q_LOSS_WEIGHT,
        help="Weight for area-normalized integrated-Q absolute error",
    )
    train_p.add_argument(
        "--alpha-mask-threshold",
        type=float,
        default=ALPHA_MASK_REL_THRESHOLD,
        help="Include bins where |Ps| >= threshold * max(|Ps|)",
    )

    infer_p = sub.add_parser("infer", help="Run Ps-only inference on one NPZ row")
    infer_p.add_argument("--model", type=Path, default=DEFAULT_OUTPUT)
    infer_p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    infer_p.add_argument("--index", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "train":
        result = train_model(
            args.data,
            args.output,
            use_residual=not bool(args.no_residual),
            num_epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.lr),
            masked_alpha_weight=float(args.masked_alpha_weight),
            q_loss_weight=float(args.q_loss_weight),
            alpha_mask_rel_threshold=float(args.alpha_mask_threshold),
        )
        print(json.dumps(result, indent=2), flush=True)
        return

    if args.command == "infer":
        model = load_trained_model(args.model, DEVICE)
        arrays = load_spectrum_npz(args.data)
        idx = int(args.index)
        res = model.predict_ps(arrays["ps"][idx], device=DEVICE)
        true_ip = arrays["iplus"][idx]
        true_im = arrays["iminus"][idx]
        l1_ip = float(np.mean(np.abs(res.iplus - true_ip)))
        l1_im = float(np.mean(np.abs(res.iminus - true_im)))
        print(
            json.dumps(
                {
                    "index": idx,
                    "p_est": res.p_est,
                    "p0": float(arrays["p0"][idx]),
                    "L1_Iplus": l1_ip,
                    "L1_Iminus": l1_im,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    raise SystemExit(f"Unknown command {args.command!r}")


if __name__ == "__main__":
    main()
