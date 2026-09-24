"""Deuteron lineshape, one AFP sweep on Q < 0, then relaxation to the pre-AFP state.

The equilibrium vector lineshape is the model's Boltzmann Pake doublet.
Local tensor signal is Q(R) = I+(R) - I-(R). A single AFP sweep is applied to
those bins (and, through the model, their mirrors). If a bin and its mirror
are both Q < 0, only the more negative bin is swept so the exchange is not
applied twice.

Relaxation then returns the lineshape to the packet from before the sweep, so
both vector polarization P and tensor polarization Q go back to their initial
values.

Run from the repo root:

  python physics/plot_afp_qneg_sweep.py
  python physics/plot_afp_qneg_sweep.py --p 0.45
  python physics/plot_afp_qneg_sweep.py --no-diagnostic

Also writes three publication panels (before AFP, post-AFP, after relaxation)
under physics/output/afp_qneg_panels/.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.rf.model import Spin1Model, Spin1Params

N_BINS = 500
R_MIN = -3.0
R_MAX = 3.0
DEFAULT_P = 0.45
DT = 0.002
MAX_RELAX_STEPS = 8000
RELAX_TOL = 1e-4
DEFAULT_OUT = Path(__file__).resolve().parent / "output" / "afp_qneg_sweep.png"
DEFAULT_PUB_DIR = Path(__file__).resolve().parent / "output" / "afp_qneg_panels"

# Publication panel styling
PUB_FIGSIZE = (6.0, 4.2)
PUB_DPI = 300
PUB_LABEL_SIZE = 16
PUB_TICK_SIZE = 14
PUB_LEGEND_SIZE = 13
PUB_LINEWIDTH = 2.2
PUB_SPINE_WIDTH = 1.75
PUB_GRID_ALPHA = 0.35
PUB_GRID_LW = 0.7


def q_negative_sweep_indices(q):
    """Bins with Q < 0, keeping at most one index from each mirror pair."""
    q = np.asarray(q)
    neg = np.flatnonzero(q < 0.0)
    if neg.size == 0:
        return []
    n = q.size
    chosen = []
    blocked = set()
    for i in neg[np.argsort(q[neg])]:
        i = int(i)
        mirror = n - 1 - i
        if i in blocked:
            continue
        chosen.append(i)
        blocked.add(i)
        blocked.add(mirror)
    return sorted(chosen)


def contiguous_spans(indices, frequency):
    """Inclusive (R_lo, R_hi) spans of contiguous bin indices."""
    if len(indices) == 0:
        return []
    indices = np.asarray(indices, dtype=int)
    spans = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        i = int(raw)
        if i == prev + 1:
            prev = i
            continue
        spans.append((float(frequency[start]), float(frequency[prev])))
        start = prev = i
    spans.append((float(frequency[start]), float(frequency[prev])))
    return spans


def shade_sweep_regions(ax, spans):
    labeled = False
    for (lo, hi) in spans:
        ax.axvspan(
            lo,
            hi,
            color="tab:red",
            alpha=0.12,
            label="AFP sweep (initial Q < 0)" if not labeled else None,
            zorder=0,
        )
        labeled = True


def simulate(polarization, n_bins=N_BINS):
    model = Spin1Model(
        Spin1Params(
            p0=float(polarization),
            n_bins=n_bins,
            r_min=R_MIN,
            r_max=R_MAX,
            rf_enabled=False,
            dnp_enabled=False,
            afp_enabled=False,
            afp_efficiency=1.0,
            afp_preserve_intensity_area=True,
        )
    )
    frequency = np.asarray(model.Rplus)
    (iplus0, iminus0, _) = model.physical_intensities()
    iplus0 = np.asarray(iplus0, dtype=float)
    iminus0 = np.asarray(iminus0, dtype=float)
    sweep = q_negative_sweep_indices(iplus0 - iminus0)
    model.params.afp_subset_indices = sweep
    before = model.polarizations()
    model.afp_sweep()
    post = model.polarizations()
    (iplus1, iminus1, _) = model.physical_intensities()
    model.params.relax_enabled = True
    model.params.diffusion_scale = 0.0
    model.params.capacity_rate_power = 0.0
    model.params.d_same_plus0 = 1.0
    model.params.d_same_0minus = 1.0
    model.params.d_spec_plus0 = 4.0
    model.params.d_spec_0minus = 4.0
    model.params.dt = DT
    model._active_idx = None
    (p_target, q_target) = model.install_recovery_to_pre_afp_state()
    p_trace = [float(post["P"])]
    q_trace = [float(post["Q"])]
    steps = 0
    for steps in range(1, MAX_RELAX_STEPS + 1):
        model.step_once(dt=DT, rf_on=False, dnp_on=False, copy=False)
        pol = model.polarizations()
        p_trace.append(float(pol["P"]))
        q_trace.append(float(pol["Q"]))
        if abs(p_trace[-1] - p_target) < RELAX_TOL and abs(q_trace[-1] - q_target) < RELAX_TOL:
            break
    relaxed = model.polarizations()
    (iplus2, iminus2, _) = model.physical_intensities()
    return {
        "frequency": frequency,
        "iplus0": iplus0,
        "iminus0": iminus0,
        "iplus1": np.asarray(iplus1),
        "iminus1": np.asarray(iminus1),
        "iplus2": np.asarray(iplus2),
        "iminus2": np.asarray(iminus2),
        "sweep": sweep,
        "spans": contiguous_spans(sweep, frequency),
        "P_before": float(before["P"]),
        "Q_before": float(before["Q"]),
        "P_after": float(post["P"]),
        "Q_after": float(post["Q"]),
        "P_relaxed": float(relaxed["P"]),
        "Q_relaxed": float(relaxed["Q"]),
        "P_target": p_target,
        "Q_target": q_target,
        "p_trace": np.asarray(p_trace),
        "q_trace": np.asarray(q_trace),
        "n_relax": steps,
        "time": np.arange(steps + 1) * DT,
    }


def _fmt_delta(value):
    if abs(value) < 5e-5:
        return "0.0000"
    sign = "+" if value >= 0.0 else ""
    return f"{sign}{value:.4f}"


def _style_publication_axis(ax):
    """Black outline, inward ticks, grid, and publication-sized labels."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(PUB_SPINE_WIDTH)
    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=7,
        width=1.25,
        labelsize=PUB_TICK_SIZE,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3.5,
        width=0.9,
    )
    ax.grid(
        True,
        which="major",
        color="0.55",
        linestyle="-",
        linewidth=PUB_GRID_LW,
        alpha=PUB_GRID_ALPHA,
        zorder=0,
    )
    ax.grid(
        True,
        which="minor",
        color="0.75",
        linestyle=":",
        linewidth=0.5,
        alpha=0.45,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.xaxis.label.set_size(PUB_LABEL_SIZE)
    ax.yaxis.label.set_size(PUB_LABEL_SIZE)


def _shade_sweep_regions_pub(ax, spans):
    labeled = False
    for (lo, hi) in spans:
        ax.axvspan(
            lo,
            hi,
            color="0.72",
            alpha=0.32,
            label="AFP sweep" if not labeled else None,
            zorder=0,
            linewidth=0,
        )
        labeled = True


def _ylims_shared(*arrays, pin_zero_floor=False):
    """Shared y-limits covering all arrays (same scale on every panel in a set)."""
    stacked = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    y_max = float(np.nanmax(stacked))
    y_min = float(np.nanmin(stacked))
    if pin_zero_floor and y_min >= -1e-6:
        raw_lo = 0.0
    else:
        raw_lo = float(y_min)
    headroom_hi = 1.08 * max(y_max, 1e-12)
    headroom_lo = 1.08 * abs(raw_lo) if raw_lo < 0.0 else 0.0
    span = max(headroom_hi, headroom_lo, 1e-12)
    if span <= 1.0:
        step = 0.1
    elif span <= 2.0:
        step = 0.25
    else:
        step = 0.5
    hi = float(np.ceil(headroom_hi / step) * step)
    lo = 0.0 if raw_lo >= 0.0 else float(np.floor(raw_lo / step) * step)
    return (lo, hi)


def _write_panel_info(result, out_dir, intensity_panels, q_panels, ylim_i, ylim_q):
    """Write panel captions / P,Q metadata that used to live in plot titles."""
    lines = [
        "AFP Q<0 sweep — publication panel metadata",
        f"initial P0 = {result['P_before']:.6f}",
        f"shared intensity y-limits: [{ylim_i[0]:.6g}, {ylim_i[1]:.6g}]",
        f"shared Q(R) y-limits: [{ylim_q[0]:.6g}, {ylim_q[1]:.6g}]",
        "intensity curves: I+, I-, and total lineshape P = I+ + I-",
        "Q curves: local Q(R) = I+(R) - I-(R)",
        f"Q<0 sweep bins: {len(result['sweep'])} in {len(result['spans'])} region(s)",
    ]
    for (lo, hi) in result["spans"]:
        lines.append(f"  R in [{lo:.3f}, {hi:.3f}]")
    lines.append("")
    lines.append("Intensity panels:")
    for panel in intensity_panels:
        lines.append(
            f"  {panel['stem']}.png  —  {panel['label']}"
            f"  (P={panel['P']:.6f}, Q={panel['Q']:.6f})"
        )
    lines.append("")
    lines.append("Q(R) panels:")
    for panel in q_panels:
        lines.append(
            f"  {panel['stem']}.png  —  {panel['label']}"
            f"  (P={panel['P']:.6f}, Q={panel['Q']:.6f})"
        )
    lines.extend(
        [
            "",
            f"vector P: {result['P_before']:.6f} -> post-AFP {result['P_after']:.6f}"
            f" -> relaxed {result['P_relaxed']:.6f}  (target P={result['P_target']:.6f})",
            f"tensor Q: {result['Q_before']:.6f} -> post-AFP {result['Q_after']:.6f}"
            f" -> relaxed {result['Q_relaxed']:.6f}  (target Q={result['Q_target']:.6f})",
            f"relaxation steps: {result['n_relax']}  dt={DT}",
        ]
    )
    info_path = Path(out_dir) / "panel_info.txt"
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return info_path


def _save_styled_panel(ax, *, xlim, ylim, out_path):
    _style_publication_axis(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim)
    ax.set_autoscale_on(False)
    out_path = Path(out_path)
    fig = ax.figure
    fig.savefig(out_path, dpi=PUB_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_publication_panels(result, out_dir):
    """Publication intensity and Q(R) panels for before / post-AFP / relaxed."""
    f = result["frequency"]
    xlim = (float(f[0]), float(f[-1]))
    intensity_panels = [
        {
            "stem": "01_before_afp",
            "label": "Before AFP",
            "iplus": result["iplus0"],
            "iminus": result["iminus0"],
            "shade": False,
            "P": result["P_before"],
            "Q": result["Q_before"],
        },
        {
            "stem": "02_after_afp",
            "label": "Immediately after AFP flip",
            "iplus": result["iplus1"],
            "iminus": result["iminus1"],
            "shade": True,
            "P": result["P_after"],
            "Q": result["Q_after"],
        },
        {
            "stem": "03_after_relaxation",
            "label": "After relaxation",
            "iplus": result["iplus2"],
            "iminus": result["iminus2"],
            "shade": False,
            "P": result["P_relaxed"],
            "Q": result["Q_relaxed"],
        },
    ]
    for panel in intensity_panels:
        ip = np.asarray(panel["iplus"], dtype=float)
        im = np.asarray(panel["iminus"], dtype=float)
        panel["ptot"] = ip + im
        panel["q_r"] = ip - im

    q_panels = [
        {
            "stem": f"{panel['stem']}_q",
            "label": panel["label"],
            "q_r": panel["q_r"],
            "shade": panel["shade"],
            "P": panel["P"],
            "Q": panel["Q"],
        }
        for panel in intensity_panels
    ]

    ylim_i = _ylims_shared(
        *(
            arr
            for panel in intensity_panels
            for arr in (panel["iplus"], panel["iminus"], panel["ptot"])
        ),
        pin_zero_floor=True,
    )
    ylim_q = _ylims_shared(*(panel["q_r"] for panel in q_panels), pin_zero_floor=False)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for panel in intensity_panels:
        fig, ax = plt.subplots(figsize=PUB_FIGSIZE, layout="constrained")
        if panel["shade"]:
            _shade_sweep_regions_pub(ax, result["spans"])
        ax.plot(
            f,
            panel["ptot"],
            color="black",
            lw=PUB_LINEWIDTH + 0.3,
            solid_capstyle="round",
            label=r"$P = I_+ + I_-$",
            zorder=4,
        )
        ax.plot(
            f,
            panel["iplus"],
            color="#c0392b",
            lw=PUB_LINEWIDTH,
            solid_capstyle="round",
            label=r"$I_+$",
            zorder=3,
        )
        ax.plot(
            f,
            panel["iminus"],
            color="#1f4e79",
            lw=PUB_LINEWIDTH,
            solid_capstyle="round",
            label=r"$I_-$",
            zorder=3,
        )
        ax.set_xlabel(r"$R$")
        ax.set_ylabel("Intensity (arb. units)")
        saved.append(_save_styled_panel(ax, xlim=xlim, ylim=ylim_i, out_path=out_dir / f"{panel['stem']}.png"))

    for panel in q_panels:
        fig, ax = plt.subplots(figsize=PUB_FIGSIZE, layout="constrained")
        if panel["shade"]:
            _shade_sweep_regions_pub(ax, result["spans"])
        ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
        ax.plot(
            f,
            panel["q_r"],
            color="#d97706",
            lw=PUB_LINEWIDTH + 0.2,
            solid_capstyle="round",
            label=r"$Q(R)=I_+-I_-$",
            zorder=3,
        )
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$Q(R)$ (arb. units)")
        saved.append(_save_styled_panel(ax, xlim=xlim, ylim=ylim_q, out_path=out_dir / f"{panel['stem']}.png"))

    info_path = _write_panel_info(
        result, out_dir, intensity_panels, q_panels, ylim_i, ylim_q
    )
    saved.append(info_path)
    return saved


def plot_result(result, out_path):
    f = result["frequency"]
    ip0, im0 = result["iplus0"], result["iminus0"]
    ip1, im1 = result["iplus1"], result["iminus1"]
    ip2, im2 = result["iplus2"], result["iminus2"]
    q0, q1, q2 = ip0 - im0, ip1 - im1, ip2 - im2
    d_ps = (ip2 + im2) - (ip1 + im1)
    d_q = q2 - q1

    fig = plt.figure(figsize=(11.2, 8.6), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=(3.2, 1.35), height_ratios=(1.15, 1.0, 1.0))
    ax_line = fig.add_subplot(grid[0, 0])
    ax_pol = fig.add_subplot(grid[0, 1])
    ax_q = fig.add_subplot(grid[1, :], sharex=ax_line)
    ax_d = fig.add_subplot(grid[2, :], sharex=ax_line)

    shade_sweep_regions(ax_line, result["spans"])
    ax_line.plot(f, ip0, color="tab:red", ls="--", lw=1.0, alpha=0.7, label=r"before $I_+$")
    ax_line.plot(f, im0, color="tab:blue", ls="--", lw=1.0, alpha=0.7, label=r"before $I_-$")
    ax_line.plot(f, ip1, color="tab:red", ls=":", lw=1.1, alpha=0.9, label=r"post-AFP $I_+$")
    ax_line.plot(f, im1, color="tab:blue", ls=":", lw=1.1, alpha=0.9, label=r"post-AFP $I_-$")
    ax_line.plot(f, ip2, color="tab:red", lw=1.5, label=r"relaxed $I_+$")
    ax_line.plot(f, im2, color="tab:blue", lw=1.5, label=r"relaxed $I_-$")
    ax_line.set_ylabel("Intensity (arb.)")
    ax_line.set_title(
        rf"AFP on initial $Q<0$, then relax back to the pre-AFP state"
        rf"    ($P_0={result['P_before']:.3f}$, {len(result['sweep'])} bins)"
    )
    ax_line.grid(True, alpha=0.3)
    # ax_line.legend(loc="upper left", fontsize=7, ncols=2)

    t = result["time"]
    ax_pol.plot(t, result["p_trace"], color="black", lw=1.5, label=r"vector $P$")
    ax_pol.plot(t, result["q_trace"], color="tab:orange", lw=1.5, label=r"tensor $Q$")
    ax_pol.axhline(result["P_target"], color="black", ls="--", lw=0.9, label=rf"$P_{{\mathrm{{initial}}}}={result['P_target']:.3f}$")
    ax_pol.axhline(result["Q_target"], color="tab:orange", ls="--", lw=0.9, label=rf"$Q_{{\mathrm{{initial}}}}={result['Q_target']:.3f}$")
    ax_pol.set_xlabel("Relax time")
    ax_pol.set_ylabel("Polarization")
    ax_pol.set_title("return to pre-AFP $P$ and $Q$")
    ax_pol.grid(True, alpha=0.3)
    # ax_pol.legend(fontsize=7, loc="best")

    shade_sweep_regions(ax_q, result["spans"])
    ax_q.plot(f, q0, color="0.45", ls="--", lw=1.1, label=r"before $Q(R)$")
    ax_q.plot(f, q1, color="tab:orange", ls=":", lw=1.2, label=r"post-AFP $Q(R)$")
    ax_q.plot(f, q2, color="tab:orange", lw=1.5, label=r"relaxed $Q(R)$")
    ax_q.axhline(0.0, color="0.3", lw=0.8)
    ax_q.set_ylabel(r"Local $Q$")
    ax_q.grid(True, alpha=0.3)
    # ax_q.legend(loc="upper left", fontsize=8)

    shade_sweep_regions(ax_d, result["spans"])
    ax_d.plot(f, d_ps, color="black", lw=1.4, label=r"$\Delta P_s$ (relaxed $-$ post-AFP)")
    ax_d.plot(f, d_q, color="tab:orange", lw=1.4, label=r"$\Delta Q(R)$ (relaxed $-$ post-AFP)")
    ax_d.axhline(0.0, color="0.3", lw=0.8)
    ax_d.set_xlabel(r"$R$")
    ax_d.set_ylabel("Change during relaxation")
    ax_d.set_title(
        rf"integrated during relaxation:  "
        rf"$\Delta P={_fmt_delta(result['P_relaxed'] - result['P_after'])}$,  "
        rf"$\Delta Q={_fmt_delta(result['Q_relaxed'] - result['Q_after'])}$"
    )
    ax_d.grid(True, alpha=0.3)
    # ax_d.legend(loc="upper left", fontsize=8)
    ax_d.set_xlim(float(f[0]), float(f[-1]))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=DEFAULT_P, help="initial vector polarization")
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pub-dir",
        type=Path,
        default=DEFAULT_PUB_DIR,
        help="directory for three separate publication lineshape panels",
    )
    parser.add_argument(
        "--no-diagnostic",
        action="store_true",
        help="skip the multi-panel diagnostic figure",
    )
    args = parser.parse_args(argv)
    result = simulate(args.p, n_bins=args.n_bins)
    if not args.no_diagnostic:
        out_path = plot_result(result, args.out)
        print(f"Saved diagnostic: {out_path}")
    pub_paths = plot_publication_panels(result, args.pub_dir)
    print(f"Q<0 sweep bins: {len(result['sweep'])} in {len(result['spans'])} region(s)")
    for (lo, hi) in result["spans"]:
        print(f"  R in [{lo:.3f}, {hi:.3f}]")
    print(f"vector P: {result['P_before']:.6f} -> post-AFP {result['P_after']:.6f} -> relaxed {result['P_relaxed']:.6f}  (target={result['P_target']:.6f})")
    print(f"tensor Q: {result['Q_before']:.6f} -> post-AFP {result['Q_after']:.6f} -> relaxed {result['Q_relaxed']:.6f}  (target={result['Q_target']:.6f})")
    print(f"relaxation steps: {result['n_relax']}  dt={DT}")
    for path in pub_paths:
        print(f"Saved panel: {path}")


if __name__ == "__main__":
    main()
