"""Publication-ready loss landscapes for spectrum→P/Q models.

Samples a 2-D filter-normalized random plane around each trained checkpoint
(Li et al., NeurIPS 2018) and evaluates the same relative P+Q loss used in
training. Writes contour + optional 3-D surface plots (PNG/PDF) and caches
the loss grid as NPZ for replotting.

Usage:
  python ml/analysis/plot_loss_landscapes.py
  python ml/analysis/plot_loss_landscapes.py --resolution 31 --n-samples 1024 --surface
  python ml/analysis/plot_loss_landscapes.py --models lstm,cnn --range 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

ANALYSIS_DIR = Path(__file__).resolve().parent
ML_DIR = ANALYSIS_DIR.parent
RESULTS_DIR = ML_DIR / "results"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import cnn as cnn_pq
import lstm as lstm_pq
import mlp as mlp_pq
import old_model as old_pq
import transformer as transformer_pq

# Latest result run per architecture (matches weight_histo / val-loss overlay).
MODELS = [
    ("lstm/lstm_result_v9", "lstm_best.pth", "LSTM", "#2ca02c", "lstm"),
    ("cnn/cnn_pq_results_v4", "cnn_pq_best.pth", "CNN", "#d62728", "cnn"),
    ("mlp/mlp_pq_results_v3", "mlp_pq_best.pth", "MLP", "#1f77b4", "mlp"),
    ("transformer/transformer_pq_results_v2", "transformer_pq_best.pth", "Transformer", "#ff7f0e", "transformer"),
    ("old_model/old_model_pq_results_v1", "old_model_pq_best.pth", "Old Polarization Model", "#9467bd", "old"),
]

CHECKPOINT_LOADERS = {
    "lstm": lstm_pq.load_checkpoint,
    "cnn": cnn_pq.load_checkpoint,
    "mlp": mlp_pq.load_checkpoint,
    "transformer": transformer_pq.load_checkpoint,
    "old": old_pq.load_checkpoint,
}

DEFAULT_SPECTRA = ML_DIR / "data" / "spectra_v6.npz"
DEFAULT_OUT_DIR = ANALYSIS_DIR / "loss_landscapes_pub"


def apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "figure.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.formatter.use_mathtext": True,
    })


def _style_ax(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", width=1.35, length=5.0, pad=3)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.6)
    ax.set_facecolor("white")


def _slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def is_bias_or_bn(name: str, param: torch.Tensor) -> bool:
    """Bias / BatchNorm / LayerNorm / 1-D scale parameters (Li et al. ignore set)."""
    key = name.lower()
    if param.ndim <= 1:
        return True
    if key.endswith(".bias") or ".bias" in key:
        return True
    if "batchnorm" in key or "layernorm" in key or "groupnorm" in key:
        return True
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return True
    return False


def _normalize_filter(direction: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Scale each filter (leading axis) so ||d_i||_F = ||w_i||_F."""
    out = direction.clone()
    # Conv*: (out_c, ...), Linear/LSTM: (out_features, in_features[, ...])
    flat_d = out.view(out.shape[0], -1)
    flat_w = weight.view(weight.shape[0], -1)
    d_norm = flat_d.norm(dim=1).clamp_min(1e-10)
    w_norm = flat_w.norm(dim=1)
    flat_d.mul_(w_norm.unsqueeze(1) / d_norm.unsqueeze(1))
    return out


def make_random_directions(
    model: nn.Module,
    *,
    seed: int,
    ignore_biasbn: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Two independent filter-normalized Gaussian directions over parameters."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    def _one() -> dict[str, torch.Tensor]:
        direction: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            raw = torch.randn(param.shape, dtype=torch.float32, generator=rng)
            if ignore_biasbn and is_bias_or_bn(name, param):
                direction[name] = torch.zeros_like(param, dtype=torch.float32, device="cpu")
                continue
            if param.ndim >= 2:
                direction[name] = _normalize_filter(raw, param.detach().float().cpu())
            else:
                w_norm = float(param.detach().float().norm().item())
                d_norm = float(raw.norm().item())
                scale = w_norm / d_norm if d_norm > 1e-10 else 0.0
                direction[name] = raw * scale
        return direction

    return _one(), _one()


def snapshot_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


@torch.no_grad()
def set_params_from_plane(
    model: nn.Module,
    base: dict[str, torch.Tensor],
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    alpha: float,
    beta: float,
) -> None:
    for name, param in model.named_parameters():
        param.copy_(
            base[name].to(device=param.device, dtype=param.dtype)
            + float(alpha) * d1[name].to(device=param.device, dtype=param.dtype)
            + float(beta) * d2[name].to(device=param.device, dtype=param.dtype)
        )


def build_eval_batch(
    spectra_path: Path,
    ckpt_stats: dict,
    *,
    n_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed (x, y_p_z, y_q_z) batch using checkpoint normalization stats."""
    arrays = lstm_pq.load_lstm_npz(spectra_path)
    spectra = arrays["spectra"]
    n = int(spectra.shape[0])
    n_bins = int(spectra.shape[2])
    if "power_profile" in arrays and arrays["power_profile"] is not None:
        power_profile = arrays["power_profile"]
    else:
        power_profile = lstm_pq.resolve_power_profile(arrays)
    n_steps = np.asarray(arrays["n_steps"]).reshape(-1)
    y_p = np.asarray(arrays["P_total"], dtype=np.float32).reshape(-1)
    y_q = np.asarray(arrays["Q_total"], dtype=np.float32).reshape(-1)

    rng = np.random.default_rng(seed)
    take = min(int(n_samples), n)
    idx = np.sort(rng.choice(n, size=take, replace=False))

    ps = np.asarray(spectra[idx, 0, :] + spectra[idx, 1, :], dtype=np.float32)
    noise_std = float(ckpt_stats.get("noise_std", lstm_pq.NOISE_STD))
    if noise_std > 0.0:
        ps = ps + rng.normal(0.0, noise_std, size=ps.shape).astype(np.float32)
    pwr = np.asarray(power_profile[idx], dtype=np.float32)
    steps = n_steps[idx].astype(np.float32, copy=False)

    ps_n = (ps - float(ckpt_stats["ps_mean"])) / float(ckpt_stats["ps_std"])
    pwr_n = (pwr - float(ckpt_stats["pwr_mean"])) / float(ckpt_stats["pwr_std"])
    steps_n = (steps - float(ckpt_stats["steps_mean"])) / float(ckpt_stats["steps_std"])
    steps_seq = np.broadcast_to(steps_n.reshape(-1, 1), (take, n_bins))
    x = np.stack([ps_n, pwr_n, steps_seq], axis=-1).astype(np.float32)

    y_p_z = (y_p[idx] - float(ckpt_stats["P_mean"])) / float(ckpt_stats["P_std"])
    y_q_z = (y_q[idx] - float(ckpt_stats["Q_mean"])) / float(ckpt_stats["Q_std"])

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y_p_z.astype(np.float32)).to(device),
        torch.from_numpy(y_q_z.astype(np.float32)).to(device),
    )


@torch.no_grad()
def eval_pq_loss(
    model: nn.Module,
    x: torch.Tensor,
    y_p: torch.Tensor,
    y_q: torch.Tensor,
    stats: dict,
    *,
    batch_size: int,
) -> float:
    model.eval()
    p_mean = float(stats["P_mean"])
    p_std = float(stats["P_std"])
    q_mean = float(stats["Q_mean"])
    q_std = float(stats["Q_std"])
    total = 0.0
    n_batches = 0
    n = x.shape[0]
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        pred_p, pred_q = model(x[sl])
        loss_p = lstm_pq.relative_weighted_loss(pred_p, y_p[sl], mean=p_mean, std=p_std)
        loss_q = lstm_pq.relative_weighted_loss(pred_q, y_q[sl], mean=q_mean, std=q_std)
        total += float((loss_p + loss_q).item())
        n_batches += 1
    return total / max(n_batches, 1)


def compute_landscape(
    model: nn.Module,
    base: dict[str, torch.Tensor],
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    x: torch.Tensor,
    y_p: torch.Tensor,
    y_q: torch.Tensor,
    stats: dict,
    *,
    alphas: np.ndarray,
    betas: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    loss_grid = np.full((betas.size, alphas.size), np.nan, dtype=np.float64)
    n_pts = int(alphas.size * betas.size)
    t0 = time.perf_counter()
    done = 0
    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            set_params_from_plane(model, base, d1, d2, float(alpha), float(beta))
            loss_grid[i, j] = eval_pq_loss(
                model, x, y_p, y_q, stats, batch_size=batch_size,
            )
            done += 1
            if done == 1 or done % max(1, n_pts // 10) == 0 or done == n_pts:
                elapsed = time.perf_counter() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n_pts - done) / max(rate, 1e-6)
                print(
                    f"    grid {done}/{n_pts}  loss={loss_grid[i, j]:.6g}  "
                    f"{rate:.1f} pts/s  ETA {eta:.0f}s",
                    flush=True,
                )
    # Restore the trained weights.
    set_params_from_plane(model, base, d1, d2, 0.0, 0.0)
    return loss_grid


def _loss_norm(loss: np.ndarray, *, log_scale: bool):
    finite = loss[np.isfinite(loss) & (loss > 0)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0), False
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if log_scale and vmax > vmin and vmin > 0:
        # Avoid LogNorm collapse when the surface is nearly flat.
        if vmax / vmin > 1.05:
            return LogNorm(vmin=vmin, vmax=vmax), True
    return Normalize(vmin=vmin, vmax=vmax), False


def plot_contour(
    alphas: np.ndarray,
    betas: np.ndarray,
    loss: np.ndarray,
    *,
    model_label: str,
    out_stem: Path,
    n_levels: int = 30,
    log_scale: bool = True,
) -> list[Path]:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    A, B = np.meshgrid(alphas, betas)
    norm, used_log = _loss_norm(loss, log_scale=log_scale)
    levels = n_levels
    if used_log:
        finite = loss[np.isfinite(loss) & (loss > 0)]
        levels = np.geomspace(float(np.min(finite)), float(np.max(finite)), n_levels)

    cf = ax.contourf(A, B, loss, levels=levels, cmap="magma", norm=norm)
    cs = ax.contour(
        A, B, loss, levels=levels, colors="0.25", linewidths=0.45, alpha=0.55, norm=norm,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")
    ax.plot(0.0, 0.0, marker="*", markersize=14, color="white",
            markeredgecolor="black", markeredgewidth=0.9, zorder=5)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Relative loss ($P+Q$)" + (" [log]" if used_log else ""))
    cbar.ax.tick_params(width=1.2, length=4)
    ax.set_xlabel(r"Filter-normalized direction $\alpha$")
    ax.set_ylabel(r"Filter-normalized direction $\beta$")
    ax.set_title(rf"{model_label}: loss landscape", pad=8)
    ax.set_aspect("equal", adjustable="box")
    _style_ax(ax)
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    saved: list[Path] = []
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("png", 300), ("pdf", None)):
        path = out_stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.05}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def plot_surface(
    alphas: np.ndarray,
    betas: np.ndarray,
    loss: np.ndarray,
    *,
    model_label: str,
    out_stem: Path,
    log_scale: bool = True,
) -> list[Path]:
    apply_pub_style()
    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    A, B = np.meshgrid(alphas, betas)
    z = np.array(loss, dtype=np.float64, copy=True)
    z_label = r"Relative loss ($P+Q$)"
    if log_scale:
        positive = z[np.isfinite(z) & (z > 0)]
        if positive.size and float(np.max(positive) / np.min(positive)) > 1.05:
            z = np.log10(np.clip(z, np.min(positive), None))
            z_label = r"$\log_{10}$ relative loss ($P+Q$)"

    norm, _ = _loss_norm(np.asarray(loss), log_scale=False)
    surf = ax.plot_surface(
        A, B, z,
        cmap="magma",
        norm=Normalize(vmin=float(np.nanmin(z)), vmax=float(np.nanmax(z))),
        linewidth=0.15,
        antialiased=True,
        edgecolor="0.35",
        alpha=0.95,
    )
    # Mark the trained point.
    z0 = float(z[np.abs(betas).argmin(), np.abs(alphas).argmin()])
    ax.scatter([0.0], [0.0], [z0], color="white", edgecolors="black",
               s=60, depthshade=False, zorder=10)
    ax.set_xlabel(r"$\alpha$", labelpad=8)
    ax.set_ylabel(r"$\beta$", labelpad=8)
    ax.set_zlabel(z_label, labelpad=8)
    ax.set_title(rf"{model_label}: loss surface", pad=10)
    ax.view_init(elev=28, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label=z_label)
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    saved: list[Path] = []
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("png", 300), ("pdf", None)):
        path = out_stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.08}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def plot_overlay_panel(
    results: list[dict],
    *,
    out_stem: Path,
    n_levels: int = 20,
    log_scale: bool = True,
) -> list[Path]:
    """Side-by-side contour panel for all models (shared α/β extent)."""
    if not results:
        return []
    apply_pub_style()
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.4 * ncols, 3.9 * nrows),
        squeeze=False,
    )
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    for ax, row in zip(axes.ravel(), results):
        alphas = row["alphas"]
        betas = row["betas"]
        loss = row["loss"]
        A, B = np.meshgrid(alphas, betas)
        norm, used_log = _loss_norm(loss, log_scale=log_scale)
        levels = n_levels
        if used_log:
            finite = loss[np.isfinite(loss) & (loss > 0)]
            levels = np.geomspace(float(np.min(finite)), float(np.max(finite)), n_levels)
        cf = ax.contourf(A, B, loss, levels=levels, cmap="magma", norm=norm)
        ax.contour(A, B, loss, levels=levels, colors="0.3", linewidths=0.35, alpha=0.5, norm=norm)
        ax.plot(0.0, 0.0, marker="*", markersize=11, color="white",
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        ax.set_title(row["label"], pad=6)
        ax.set_aspect("equal", adjustable="box")
        _style_ax(ax)
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
        cbar.ax.tick_params(labelsize=9, width=1.0, length=3.5)

    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(r"$\alpha$")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(r"$\beta$")

    fig.suptitle("Filter-normalized loss landscapes", y=0.995, fontsize=15)
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    saved: list[Path] = []
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("png", 300), ("pdf", None)):
        path = out_stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.06}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def save_grid_npz(
    path: Path,
    *,
    alphas: np.ndarray,
    betas: np.ndarray,
    loss: np.ndarray,
    meta: dict,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        alphas=alphas.astype(np.float64),
        betas=betas.astype(np.float64),
        loss=loss.astype(np.float64),
        meta_json=np.asarray(json.dumps(meta)),
    )
    return path


def load_grid_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        alphas = np.asarray(z["alphas"], dtype=np.float64)
        betas = np.asarray(z["betas"], dtype=np.float64)
        loss = np.asarray(z["loss"], dtype=np.float64)
        meta = json.loads(str(z["meta_json"]))
    return alphas, betas, loss, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Directory for plots and cached grids",
    )
    parser.add_argument(
        "--spectra", type=Path, default=DEFAULT_SPECTRA,
        help="Spectra NPZ or memmap directory used for loss evaluation",
    )
    parser.add_argument(
        "--models", type=str, default="all",
        help="Comma-separated model keys: lstm,cnn,mlp,transformer,old  (default: all)",
    )
    parser.add_argument("--resolution", type=int, default=41,
                        help="Grid points per axis (odd preferred so 0 is included)")
    parser.add_argument("--range", type=float, default=1.0,
                        help="Half-width of α/β axis (filter-normalized units)")
    parser.add_argument("--n-samples", type=int, default=2048,
                        help="Number of spectra used to estimate loss at each grid point")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0, help="Direction + subsample seed")
    parser.add_argument("--data-seed", type=int, default=42, help="Spectra subsample seed")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument(
        "--no-ignore-biasbn", action="store_true",
        help="Include bias/BN directions (default: ignore, matching Li et al.)",
    )
    parser.add_argument("--surface", action="store_true", help="Also write 3-D surface plots")
    parser.add_argument("--no-overlay", action="store_true", help="Skip multi-model panel")
    parser.add_argument("--no-log", action="store_true", help="Use linear color scale")
    parser.add_argument(
        "--replot-only", action="store_true",
        help="Skip computation; rebuild plots from cached NPZ grids in --out-dir",
    )
    parser.add_argument(
        "--also-results-dir", action="store_true",
        help="Also copy plots into each model's results directory",
    )
    return parser.parse_args()


def selected_models(spec: str) -> list[tuple]:
    if spec.strip().lower() == "all":
        return list(MODELS)
    want = {s.strip().lower() for s in spec.split(",") if s.strip()}
    key_aliases = {
        "lstm": "lstm",
        "cnn": "cnn",
        "mlp": "mlp",
        "transformer": "transformer",
        "old": "old",
        "old_model": "old",
        "old_polarization_model": "old",
    }
    mapped = {key_aliases.get(k, k) for k in want}
    out = [row for row in MODELS if row[4] in mapped]
    missing = mapped - {row[4] for row in out}
    if missing:
        raise ValueError(f"Unknown model key(s): {sorted(missing)}. Choose from {sorted(CHECKPOINT_LOADERS)}")
    return out


def main() -> None:
    args = parse_args()
    apply_pub_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    models = selected_models(args.models)
    log_scale = not args.no_log
    ignore_biasbn = not args.no_ignore_biasbn

    # Symmetric odd grid so (0, 0) lands on a sample.
    n = max(3, int(args.resolution))
    if n % 2 == 0:
        n += 1
    half = float(args.range)
    alphas = np.linspace(-half, half, n, dtype=np.float64)
    betas = np.linspace(-half, half, n, dtype=np.float64)

    panel_rows: list[dict] = []

    for run_dir, ckpt_name, label, _color, loader_key in models:
        slug = _slug(label)
        grid_path = args.out_dir / f"{slug}_loss_landscape.npz"
        contour_stem = args.out_dir / f"{slug}_loss_landscape"
        surface_stem = args.out_dir / f"{slug}_loss_surface"

        if args.replot_only:
            if not grid_path.is_file():
                raise FileNotFoundError(f"Missing cached grid for replot: {grid_path}")
            alphas_r, betas_r, loss, meta = load_grid_npz(grid_path)
            print(f"{label}: replot from {grid_path}", flush=True)
        else:
            ckpt_path = RESULTS_DIR / run_dir / ckpt_name
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
            print(f"{label}: loading {ckpt_path} on {device}", flush=True)
            loader = CHECKPOINT_LOADERS[loader_key]
            model, ckpt = loader(ckpt_path, device=device)
            model.eval()
            stats = ckpt["stats"]

            print(f"  building eval batch (n={args.n_samples}) from {args.spectra}", flush=True)
            x, y_p, y_q = build_eval_batch(
                args.spectra, stats,
                n_samples=args.n_samples, seed=args.data_seed, device=device,
            )
            base = snapshot_params(model)
            # Distinct direction seed per architecture so planes are independent.
            dir_seed = int(args.seed) + 17 * sum(ord(c) for c in loader_key)
            d1, d2 = make_random_directions(
                model, seed=dir_seed, ignore_biasbn=ignore_biasbn,
            )
            center_loss = eval_pq_loss(
                model, x, y_p, y_q, stats, batch_size=args.batch_size,
            )
            print(f"  center loss={center_loss:.6g}  grid={n}x{n}  range=±{half}", flush=True)
            loss = compute_landscape(
                model, base, d1, d2, x, y_p, y_q, stats,
                alphas=alphas, betas=betas, batch_size=args.batch_size,
            )
            alphas_r, betas_r = alphas, betas
            meta = {
                "model": label,
                "loader_key": loader_key,
                "checkpoint": str(ckpt_path),
                "spectra": str(args.spectra),
                "n_samples": int(args.n_samples),
                "resolution": int(n),
                "range": float(half),
                "seed": int(args.seed),
                "data_seed": int(args.data_seed),
                "dir_seed": int(dir_seed),
                "ignore_biasbn": bool(ignore_biasbn),
                "center_loss": float(center_loss),
                "min_loss": float(np.nanmin(loss)),
                "max_loss": float(np.nanmax(loss)),
                "device": str(device),
            }
            save_grid_npz(grid_path, alphas=alphas_r, betas=betas_r, loss=loss, meta=meta)
            print(f"  Saved {grid_path}", flush=True)
            del model, x, y_p, y_q
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for path in plot_contour(
            alphas_r, betas_r, loss,
            model_label=label, out_stem=contour_stem, log_scale=log_scale,
        ):
            print(f"  Saved {path}", flush=True)

        if args.surface:
            for path in plot_surface(
                alphas_r, betas_r, loss,
                model_label=label, out_stem=surface_stem, log_scale=log_scale,
            ):
                print(f"  Saved {path}", flush=True)

        if args.also_results_dir:
            results_dir = RESULTS_DIR / run_dir
            for path in plot_contour(
                alphas_r, betas_r, loss,
                model_label=label,
                out_stem=results_dir / "loss_landscape",
                log_scale=log_scale,
            ):
                print(f"  Saved {path}", flush=True)

        panel_rows.append({
            "label": label,
            "alphas": alphas_r,
            "betas": betas_r,
            "loss": loss,
        })

    if not args.no_overlay and len(panel_rows) > 1:
        for path in plot_overlay_panel(
            panel_rows,
            out_stem=args.out_dir / "loss_landscapes_overlay",
            log_scale=log_scale,
        ):
            print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
