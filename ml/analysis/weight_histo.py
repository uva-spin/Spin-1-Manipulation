"""Publication-ready weight / bias / activation / layer histograms for PQ models."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

# Latest result run per architecture (matches plot_val_loss_overlay.py).
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

SKIP_ACTIVATION_TYPES = (
    nn.Dropout,
    nn.Identity,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveMaxPool1d,
)


def apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "figure.titlesize": 16,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    if hasattr(ckpt, "state_dict"):
        return ckpt.state_dict()
    raise TypeError(f"Unrecognized checkpoint format in {path}")


def weight_params(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    """Return trainable weight matrices/kernels (exclude 1-D BN/LN scales)."""
    return [
        (name, param)
        for name, param in state_dict.items()
        if "weight" in name and param.ndim >= 2
    ]


def bias_params(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    """Return bias vectors (Linear/Conv/LSTM/BN/LN), excluding BN running stats."""
    out: list[tuple[str, torch.Tensor]] = []
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor) or param.ndim == 0:
            continue
        key = name.lower()
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        if key.endswith(".bias") or key.endswith("_bias") or ".bias_" in key or key.startswith("bias"):
            out.append((name, param))
            continue
        # LSTM packed biases: encoder.bias_ih_l0, encoder.bias_hh_l0_reverse, ...
        if re.search(r"(^|\.)bias_(ih|hh)_l\d+", key):
            out.append((name, param))
    return out


def layer_group_name(param_name: str) -> str:
    """Map a parameter name onto its architectural layer / block."""
    m = re.match(r"^encoder\.(?:weight|bias)_(?:ih|hh)_l(\d+)(_reverse)?", param_name)
    if m:
        return f"encoder L{m.group(1)}" + (" rev" if m.group(2) else "")

    m = re.match(r"^(encoder\.layers\.\d+)", param_name)
    if m:
        return m.group(1)

    m = re.match(r"^(encoder\.\d+)", param_name)
    if m:
        return m.group(1)

    m = re.match(r"^(residual_blocks\.\d+)", param_name)
    if m:
        return m.group(1)

    for prefix in (
        "inception_block",
        "se_block",
        "token_embed",
        "encoder.norm",
        "fc",
        "head_p",
        "head_q",
    ):
        if param_name == prefix or param_name.startswith(prefix + "."):
            return prefix

    parts = param_name.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def layer_params(state_dict: dict[str, torch.Tensor]) -> list[tuple[str, torch.Tensor]]:
    """Concatenate learnable tensors belonging to the same architectural layer."""
    groups: dict[str, list[torch.Tensor]] = {}
    order: list[str] = []
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor) or param.ndim == 0:
            continue
        key = name.lower()
        if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
            continue
        if not param.is_floating_point():
            continue
        group = layer_group_name(name)
        if group not in groups:
            groups[group] = []
            order.append(group)
        groups[group].append(param.detach().float().reshape(-1).cpu())
    return [(group, torch.cat(chunks)) for group, chunks in ((g, groups[g]) for g in order)]


def pretty_param_name(name: str) -> str:
    label = re.sub(r"(\.weight|_weight|\.bias|_bias)$", "", name)
    replacements = [
        (r"^encoder\.weight_ih_l(\d+)(_reverse)?$", r"encoder ih L\1\2"),
        (r"^encoder\.weight_hh_l(\d+)(_reverse)?$", r"encoder hh L\1\2"),
        (r"^encoder\.bias_ih_l(\d+)(_reverse)?$", r"encoder ih bias L\1\2"),
        (r"^encoder\.bias_hh_l(\d+)(_reverse)?$", r"encoder hh bias L\1\2"),
        (r"^encoder L(\d+)( rev)?$", r"encoder L\1\2"),
        (r"_reverse$", " rev"),
        (r"^encoder\.layers\.(\d+)\.self_attn\.in_proj$", r"enc L\1 attn in"),
        (r"^encoder\.layers\.(\d+)\.self_attn\.out_proj$", r"enc L\1 attn out"),
        (r"^encoder\.layers\.(\d+)\.linear1$", r"enc L\1 FFN up"),
        (r"^encoder\.layers\.(\d+)\.linear2$", r"enc L\1 FFN down"),
        (r"^encoder\.layers\.(\d+)\.norm1$", r"enc L\1 norm1"),
        (r"^encoder\.layers\.(\d+)\.norm2$", r"enc L\1 norm2"),
        (r"^encoder\.layers\.(\d+)$", r"enc L\1"),
        (r"^encoder\.norm$", "enc final norm"),
        (r"^token_embed$", "token embed"),
        (r"^encoder\.(\d+)\.conv1$", r"res\1 conv1"),
        (r"^encoder\.(\d+)\.conv2$", r"res\1 conv2"),
        (r"^encoder\.(\d+)\.bn1$", r"res\1 bn1"),
        (r"^encoder\.(\d+)\.bn2$", r"res\1 bn2"),
        (r"^encoder\.(\d+)\.skip$", r"enc\1 skip"),
        (r"^encoder\.(\d+)\.fc1$", r"block\1 fc1"),
        (r"^encoder\.(\d+)\.fc2$", r"block\1 fc2"),
        (r"^encoder\.(\d+)$", r"block\1"),
        (r"^inception_block\.branch(\d+)\.(\d+)$", r"incept b\1.\2"),
        (r"^inception_block$", "inception"),
        (r"^residual_blocks\.(\d+)\.conv(\d+)$", r"res\1 conv\2"),
        (r"^residual_blocks\.(\d+)$", r"res\1"),
        (r"^se_block\.fc(\d+)$", r"SE fc\1"),
        (r"^se_block$", "SE"),
        (r"^fc$", "fc"),
        (r"^head_p$", r"$P$ head"),
        (r"^head_q$", r"$Q$ head"),
    ]
    for pattern, repl in replacements:
        label = re.sub(pattern, repl, label)
    return label.replace("_", " ")


def split_body_and_heads(
    params: list[tuple[str, torch.Tensor]],
) -> tuple[
    list[tuple[str, torch.Tensor]],
    tuple[str, torch.Tensor] | None,
    tuple[str, torch.Tensor] | None,
]:
    body: list[tuple[str, torch.Tensor]] = []
    head_p = head_q = None
    for name, param in params:
        key = name.lower()
        if key == "head_p" or key.startswith("head_p.") or key.startswith("head_p "):
            head_p = (name, param)
        elif key == "head_q" or key.startswith("head_q.") or key.startswith("head_q "):
            head_q = (name, param)
        else:
            body.append((name, param))
    return body, head_p, head_q


def _to_numpy(param: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(param, np.ndarray):
        return np.asarray(param, dtype=np.float64).reshape(-1)
    return param.detach().float().cpu().numpy().reshape(-1)


def _value_range(arrays: list[np.ndarray], *, q: float = 0.005) -> tuple[float, float]:
    stacked = np.concatenate(arrays)
    lo, hi = np.quantile(stacked, [q, 1.0 - q])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(stacked.min()), float(stacked.max())
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _file_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return slug or "param"


def _save_figure(fig: plt.Figure, out_stem: Path, *, pad_inches: float = 0.15) -> list[Path]:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    path = out_stem.with_suffix(".png")
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )
    plt.close(fig)
    return [path]


def _style_2d_hist_ax(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", width=1.1, length=4)
    ax.grid(True, which="major", linestyle="--", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.3)


def plot_param_histograms(
    params: list[tuple[str, torch.Tensor | np.ndarray]],
    *,
    model_label: str,
    color: str,
    out_dir: Path,
    value_label: str,
    kind_title: str,
    bins: int = 60,
) -> list[Path]:
    """Write one PNG per tensor into ``out_dir``."""
    if not params:
        raise ValueError(f"No {kind_title} tensors found for {model_label}")

    saved: list[Path] = []
    for name, param in params:
        values = _to_numpy(param)
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        fig.patch.set_facecolor("white")
        ax.hist(
            values,
            bins=bins,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.35,
            density=False,
        )
        ax.set_title(f"{model_label} {pretty_param_name(name)}", fontsize=12, pad=6)
        ax.set_xlabel(value_label)
        ax.set_ylabel("Count")
        _style_2d_hist_ax(ax)
        mu = float(values.mean())
        sigma = float(values.std())
        ax.text(
            0.98,
            0.95,
            rf"$\mu={mu:.3f}$" + "\n" + rf"$\sigma={sigma:.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            linespacing=1.25,
        )
        fig.tight_layout()
        saved.extend(_save_figure(fig, out_dir / _file_slug(name)))
    return saved


def _style_3d_ax(ax) -> None:
    ax.view_init(elev=18, azim=-58)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.55))
    ax.xaxis.pane.set_edgecolor("0.75")
    ax.yaxis.pane.set_edgecolor("0.75")
    ax.zaxis.pane.set_edgecolor("0.75")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.45)
    ax.tick_params(axis="both", which="major", labelsize=9, pad=1)


def _draw_3d_layer_ridges(
    ax,
    layers: list[tuple[str, np.ndarray]],
    *,
    color: str,
    bins: int,
    density: bool = True,
    value_label: str = "Weight value",
) -> None:
    if not layers:
        ax.set_axis_off()
        return

    arrays = [arr for _, arr in layers]
    vmin, vmax = _value_range(arrays)
    bin_edges = np.linspace(vmin, vmax, bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rgb = _hex_to_rgb(color)

    dens_rows = [
        np.histogram(weights, bins=bin_edges, density=density)[0]
        for _, weights in layers
    ]
    dens = np.asarray(dens_rows, dtype=float)
    z_max = float(dens.max()) if dens.size else 1.0
    if z_max <= 0:
        z_max = 1.0

    for i in range(len(layers) - 1, -1, -1):
        hist = dens[i]
        alpha = 0.22 + 0.50 * (i / max(len(layers) - 1, 1))
        face = [
            (centers[0], float(i), 0.0),
            *zip(centers, np.full_like(centers, float(i)), hist),
            (centers[-1], float(i), 0.0),
        ]
        poly = Poly3DCollection(
            [face],
            facecolors=[(*rgb, alpha)],
            edgecolors=[(*rgb, min(alpha + 0.30, 0.95))],
            linewidths=0.55,
        )
        ax.add_collection3d(poly)
        ax.plot(centers, np.full_like(centers, float(i)), hist, color=color, lw=1.35, zorder=10)

    n_layers = len(layers)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(-0.7, n_layers - 0.3)
    ax.set_zlim(0.0, z_max * 1.10)
    ax.set_xlabel(value_label, labelpad=10, fontsize=12)
    ax.set_ylabel("Layer", labelpad=12, fontsize=12)
    ax.set_zlabel("Density" if density else "Count", labelpad=8, fontsize=12)
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([str(i + 1) for i in range(n_layers)], fontsize=8)
    _style_3d_ax(ax)


def _draw_head_hist(
    ax: plt.Axes,
    head: tuple[str, torch.Tensor] | None,
    *,
    color: str,
    bins: int,
    title: str,
    value_label: str = "Weight value",
    xlim: tuple[float, float] | None = None,
) -> None:
    ax.set_title(title, fontsize=13, pad=6)
    if head is None:
        ax.text(0.5, 0.5, "not found", ha="center", va="center", transform=ax.transAxes)
        _style_2d_hist_ax(ax)
        return

    weights = _to_numpy(head[1])
    n_bins = min(bins, max(16, int(np.sqrt(weights.size)) * 2))
    hist_range = xlim if xlim is not None else _value_range([weights])
    counts, edges = np.histogram(weights, bins=n_bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(edges[1] - edges[0])

    ax.bar(
        centers,
        counts,
        width=width * 0.92,
        align="center",
        color=color,
        alpha=0.80,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )

    ax.set_xlabel(value_label, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    if xlim is not None:
        ax.set_xlim(*xlim)
    _style_2d_hist_ax(ax)
    ax.set_facecolor("#fafafa")

    mu = float(weights.mean())
    sigma = float(weights.std())
    ax.text(
        0.97,
        0.94,
        rf"$n={weights.size}$"
        + "\n"
        + rf"$\mu={mu:.3f}$"
        + "\n"
        + rf"$\sigma={sigma:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        linespacing=1.3,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.85",
            "linewidth": 0.8,
            "alpha": 0.92,
        },
    )


def plot_param_histograms_3d(
    params: list[tuple[str, torch.Tensor | np.ndarray]],
    *,
    model_label: str,
    color: str,
    out_dir: Path,
    value_label: str,
    kind_title: str,
    bins: int = 60,
) -> list[Path]:
    """3D body-layer waterfall with polished 2D $P$/$Q$ head histograms."""
    if not params:
        raise ValueError(f"No {kind_title} tensors found for {model_label}")

    tensor_params = [
        (name, param if isinstance(param, torch.Tensor) else torch.from_numpy(np.asarray(param)))
        for name, param in params
    ]
    body, head_p, head_q = split_body_and_heads(tensor_params)
    body_layers = [(name, _to_numpy(param)) for name, param in body]

    head_arrays = []
    if head_p is not None:
        head_arrays.append(_to_numpy(head_p[1]))
    if head_q is not None:
        head_arrays.append(_to_numpy(head_q[1]))
    head_xlim = _value_range(head_arrays) if head_arrays else None

    saved: list[Path] = []
    if body_layers:
        fig = plt.figure(figsize=(9.0, 6.4))
        fig.patch.set_facecolor("white")
        ax_body = fig.add_subplot(111, projection="3d")
        _draw_3d_layer_ridges(
            ax_body,
            body_layers,
            color=color,
            bins=bins,
            density=True,
            value_label=value_label,
        )
        ax_body.set_title(f"{model_label} {kind_title} body layers", fontsize=14, pad=10)
        fig.tight_layout()
        saved.extend(_save_figure(fig, out_dir / "body_layers"))

    for stem, head, title in (
        ("p_head", head_p, r"$P$ head"),
        ("q_head", head_q, r"$Q$ head"),
    ):
        if head is None:
            continue
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        fig.patch.set_facecolor("white")
        _draw_head_hist(
            ax,
            head,
            color=color,
            bins=bins,
            title=f"{model_label} {kind_title}: {title}",
            value_label=value_label,
            xlim=head_xlim,
        )
        fig.tight_layout()
        saved.extend(_save_figure(fig, out_dir / stem))
    return saved


def activation_hook_targets(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Select mid-level modules whose outputs are useful activation histograms."""
    targets: list[tuple[str, nn.Module]] = []
    seen: set[str] = set()

    def add(name: str, module: nn.Module) -> None:
        if name in seen or isinstance(module, SKIP_ACTIVATION_TYPES):
            return
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.TransformerEncoder)):
            return
        seen.add(name)
        targets.append((name, module))

    for name, module in model.named_children():
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            for i, child in enumerate(module):
                add(f"{name}.{i}", child)
        elif isinstance(module, nn.TransformerEncoder):
            for i, layer in enumerate(module.layers):
                add(f"{name}.layers.{i}", layer)
            if module.norm is not None:
                add(f"{name}.norm", module.norm)
        else:
            add(name, module)
    return targets


def _tensor_from_module_output(out) -> torch.Tensor | None:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    return None


def collect_layer_activations(
    model: nn.Module,
    batch: torch.Tensor,
    *,
    max_values: int = 250_000,
    seed: int = 0,
) -> list[tuple[str, np.ndarray]]:
    """Run one forward pass and collect flattened activations per hooked layer."""
    model.eval()
    captured: dict[str, list[np.ndarray]] = {}
    handles = []
    rng = np.random.default_rng(seed)

    def make_hook(name: str):
        def _hook(_module, _inp, out):
            tensor = _tensor_from_module_output(out)
            if tensor is None:
                return
            values = tensor.detach().float().reshape(-1).cpu().numpy()
            if values.size > max_values:
                idx = rng.choice(values.size, size=max_values, replace=False)
                values = values[idx]
            captured.setdefault(name, []).append(values)

        return _hook

    for name, module in activation_hook_targets(model):
        handles.append(module.register_forward_hook(make_hook(name)))

    device = next(model.parameters()).device
    with torch.no_grad():
        model(batch.to(device))

    for handle in handles:
        handle.remove()

    return [
        (name, np.concatenate(chunks))
        for name, chunks in captured.items()
        if chunks
    ]


def build_activation_batch(
    spectra_path: Path,
    ckpt_stats: dict,
    *,
    n_samples: int = 512,
    seed: int = 42,
) -> torch.Tensor:
    """Pack a small (N, bins, 3) batch normalized with checkpoint stats."""
    arrays = lstm_pq.load_lstm_npz(spectra_path)
    spectra = np.asarray(arrays["spectra"])
    ps = np.asarray(spectra[:, 0, :] + spectra[:, 1, :], dtype=np.float32, copy=True)
    power_profile = lstm_pq.resolve_power_profile(arrays)
    n_steps = np.asarray(arrays["n_steps"]).reshape(-1)
    n = ps.shape[0]
    rng = np.random.default_rng(seed)
    noise_std = float(ckpt_stats.get("noise_std", lstm_pq.NOISE_STD))
    if noise_std > 0.0:
        ps = ps + rng.normal(0.0, noise_std, size=ps.shape).astype(np.float32)

    take = min(int(n_samples), n)
    idx = rng.choice(n, size=take, replace=False)
    t = ps.shape[1]
    ps_n = (ps[idx] - float(ckpt_stats["ps_mean"])) / float(ckpt_stats["ps_std"])
    pwr_n = (power_profile[idx] - float(ckpt_stats["pwr_mean"])) / float(ckpt_stats["pwr_std"])
    steps_n = (n_steps[idx] - float(ckpt_stats["steps_mean"])) / float(ckpt_stats["steps_std"])
    steps_seq = np.broadcast_to(steps_n.reshape(-1, 1), (take, t))
    x = np.stack([ps_n, pwr_n, steps_seq], axis=-1).astype(np.float32)
    return torch.from_numpy(x)


def _plot_kind(
    params: list[tuple[str, torch.Tensor | np.ndarray]],
    *,
    model_label: str,
    color: str,
    out_dir: Path,
    slug: str,
    kind: str,
    value_label: str,
    bins: int,
    skip_2d: bool,
    skip_3d: bool,
    also_results_dir: Path | None,
) -> None:
    if not params:
        print(f"  skip {kind}: none found")
        return
    print(f"  {kind}: {len(params)} tensors/layers")

    targets = [out_dir / slug]
    if also_results_dir is not None:
        targets.append(also_results_dir / "histograms")

    for dest in targets:
        if not skip_2d:
            for path in plot_param_histograms(
                params,
                model_label=model_label,
                color=color,
                out_dir=dest / kind,
                value_label=value_label,
                kind_title=kind,
                bins=bins,
            ):
                print(f"  Saved {path}")
        if not skip_3d:
            for path in plot_param_histograms_3d(
                params,
                model_label=model_label,
                color=color,
                out_dir=dest / f"{kind}_3d",
                value_label=value_label,
                kind_title=kind,
                bins=bins,
            ):
                print(f"  Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ANALYSIS_DIR / "weight_histograms_pub",
        help="Directory for publication plots (default: ml/analysis/weight_histograms_pub)",
    )
    parser.add_argument("--bins", type=int, default=60, help="Histogram bins")
    parser.add_argument(
        "--spectra",
        type=Path,
        default=ML_DIR / "data" / "spectra_v6.npz",
        help="Spectra NPZ used for layer-activation histograms",
    )
    parser.add_argument(
        "--activation-samples",
        type=int,
        default=512,
        help="Number of spectra used when collecting activations",
    )
    parser.add_argument(
        "--also-results-dir",
        action="store_true",
        help="Also write copies into each model's results directory",
    )
    parser.add_argument("--skip-2d", action="store_true", help="Skip per-tensor 2D grids")
    parser.add_argument("--skip-3d", action="store_true", help="Skip 3D layer-waterfall plots")
    parser.add_argument("--skip-weights", action="store_true")
    parser.add_argument("--skip-biases", action="store_true")
    parser.add_argument("--skip-activations", action="store_true")
    parser.add_argument("--skip-layers", action="store_true")
    args = parser.parse_args()

    apply_pub_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for legacy in args.out_dir.glob("*_histograms*"):
        if legacy.is_file():
            legacy.unlink()

    for run_dir, ckpt_name, label, color, loader_key in MODELS:
        ckpt_path = RESULTS_DIR / run_dir / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        state_dict = load_state_dict(ckpt_path)
        slug = label.lower().replace(" ", "_")
        results_dir = RESULTS_DIR / run_dir if args.also_results_dir else None
        print(f"{label}: checkpoint {ckpt_path}")

        kinds: list[tuple[str, str, list]] = []
        if not args.skip_weights:
            kinds.append(("weight", "Weight value", weight_params(state_dict)))
        if not args.skip_biases:
            kinds.append(("bias", "Bias value", bias_params(state_dict)))
        if not args.skip_layers:
            kinds.append(("layer", "Parameter value", layer_params(state_dict)))

        if not args.skip_activations:
            loader = CHECKPOINT_LOADERS[loader_key]
            model, ckpt = loader(ckpt_path, device=torch.device("cpu"))
            batch = build_activation_batch(
                args.spectra,
                ckpt["stats"],
                n_samples=args.activation_samples,
            )
            activations = collect_layer_activations(model, batch)
            kinds.append(("activation", "Activation value", activations))
            del model

        for kind, value_label, params in kinds:
            _plot_kind(
                params,
                model_label=label,
                color=color,
                out_dir=args.out_dir,
                slug=slug,
                kind=kind,
                value_label=value_label,
                bins=args.bins,
                skip_2d=args.skip_2d,
                skip_3d=args.skip_3d,
                also_results_dir=results_dir,
            )


if __name__ == "__main__":
    main()
