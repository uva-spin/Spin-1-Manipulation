"""
Efficient training data generation for ssRF burn correction.

Generates lineshapes over a range of polarizations (like binning.py), applies
a single burn at different locations per event to create many training instances,
and uses the lookup table only for mapping burns (not as the event source).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path

from ssRFMapper import AFP, ssRFMapper


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = REPO_ROOT / "results" / "current" / "data_creation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Lineshape functions matching binning.py (same as lookup table source)
g = 0.05
s = 0.04
bigy = np.sqrt(3 - s)


def _lineshape(x, eps):
    def cosal(x, eps):
        return (1 - eps * x - s) / bigxsquare(x, eps)

    def bigxsquare(x, eps):
        return np.sqrt(g**2 + (1 - eps * x - s)**2)

    def mult_term(x, eps):
        return 1 / (2 * np.pi * np.sqrt(bigxsquare(x, eps)))

    def cosaltwo(x, eps):
        return np.sqrt((1 + cosal(x, eps)) / 2)

    def sinaltwo(x, eps):
        return np.sqrt((1 - cosal(x, eps)) / 2)

    def termone(x, eps):
        return np.pi / 2 + np.arctan((bigy**2 - bigxsquare(x, eps)) / (2 * bigy * np.sqrt(bigxsquare(x, eps)) * sinaltwo(x, eps)))

    def termtwo(x, eps):
        return np.log((bigy**2 + bigxsquare(x, eps) + 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)) /
                      (bigy**2 + bigxsquare(x, eps) - 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)))

    def icurve(x, eps):
        return mult_term(x, eps) * (2 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))

    return icurve(x, eps) / 100


def _generate_vector_lineshape(P, x):
    """Generate Ps, Iplus, Iminus for polarization P. Matches binning.py."""
    r = (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)
    if P > 0:
        Iplus = r * _lineshape(x, 1)
        Iminus = _lineshape(x, -1)
    else:
        r = 1 / r
        Iplus = -_lineshape(x, 1)
        Iminus = -r * _lineshape(x, -1)
    deltaP = (P / np.sum(Iplus + Iminus))
    Iplus *= deltaP
    Iminus *= deltaP
    signal = Iplus + Iminus
    return signal, Iplus, Iminus


def _ensure_lookup_table(path: Path, f_arr: np.ndarray) -> None:
    """Build a compact Ps→I± lookup table if the full ``lookup_table.pkl`` is absent."""
    if path.exists():
        return
    print(f"No {path.name} found; building a minimal table for this run (subset of P)...")
    polarizations = np.concatenate([
        np.linspace(-0.65, -0.05, 800),
        np.linspace(0.05, 0.65, 800),
    ])
    rows = []
    for P in tqdm.tqdm(polarizations, desc="Minimal lookup"):
        signal, Iplus, Iminus = _generate_vector_lineshape(P, f_arr)
        rows.append({
            'P': P,
            'Ps': signal,
            'Iplus': Iplus,
            'Iminus': Iminus,
        })
    pd.DataFrame(rows).to_pickle(path)
    print(f"Wrote {path}")


def _save_population_burn_figure(
    f_arr: np.ndarray,
    samples: list,
    output_path: Path,
    suptitle_extra: str = "",
    dpi: int = 200,
) -> None:
    """n rows × 2 cols: [level populations | lineshape: Ps and I± unburned vs burned]."""
    n = len(samples)
    if n == 0:
        return
    row_h = 3.35
    fig_h = max(4.0, row_h * n + 0.9)
    fig_w = 11.8
    fig, axes = plt.subplots(
        n, 2,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=False,
        gridspec_kw={'width_ratios': [1.15, 1.05], 'wspace': 0.28, 'hspace': 0.32},
        layout='constrained',
    )
    if n == 1:
        axes = np.asarray(axes).reshape(1, -1)
    title_fs = 10
    leg_fs = 7.5

    def _burn_vlines(ax, sample):
        ax.axvline(sample['x0'], color='green', alpha=0.45, linestyle=':', linewidth=1)
        ax.axvline(-sample['x0'], color='purple', alpha=0.45, linestyle=':', linewidth=1)
        x0_2 = sample.get('x0_second', np.nan)
        if pd.notna(x0_2):
            ax.axvline(float(x0_2), color='darkorange', alpha=0.45, linestyle=':', linewidth=1)
            ax.axvline(-float(x0_2), color='coral', alpha=0.45, linestyle=':', linewidth=1)

    for row, sample in enumerate(samples):
        ax_pop = axes[row, 0]
        ax_sig = axes[row, 1]

        ax_pop.plot(f_arr, sample['rho_plus_unburned'], color='tab:red', linestyle='--', alpha=0.55, linewidth=1.0)
        ax_pop.plot(f_arr, sample['rho_zero_unburned'], color='tab:gray', linestyle='--', alpha=0.55, linewidth=1.0)
        ax_pop.plot(f_arr, sample['rho_minus_unburned'], color='tab:blue', linestyle='--', alpha=0.55, linewidth=1.0)
        ax_pop.plot(f_arr, sample['rho_plus'], color='tab:red', linestyle='-', linewidth=1.25, label=r'$\rho_+$')
        ax_pop.plot(f_arr, sample['rho_zero'], color='tab:gray', linestyle='-', linewidth=1.25, label=r'$\rho_0$')
        ax_pop.plot(f_arr, sample['rho_minus'], color='tab:blue', linestyle='-', linewidth=1.25, label=r'$\rho_-$')
        _burn_vlines(ax_pop, sample)
        ax_pop.set_ylabel('population (norm.)')
        ax_pop.grid(True, alpha=0.3)
        # ax_pop.legend(loc='upper right', fontsize=leg_fs, title='solid=burned, dashed=unburned')
        ax_pop.set_title(
            f"P = {sample['P']:.3f}, x0 = {sample['x0']:.2f}, amp = {sample['amp']:.2e}",
            fontsize=title_fs,
        )

        ax_sig.plot(f_arr, sample['Ps_unburned'], color='gray', alpha=0.6, linestyle='--', linewidth=1.0, label=r'$P_s$ unburned')
        ax_sig.plot(f_arr, sample['Ps'], color='black', linestyle='-', linewidth=1.25, label=r'$P_s$ burned')
        ax_sig.plot(f_arr, sample['Iplus_unburned'], color='tab:red', alpha=0.4, linestyle='--', linewidth=1.0, label=r'$I_+$ unburned')
        ax_sig.plot(f_arr, sample['Iplus'], color='tab:red', linestyle='-', linewidth=1.25, alpha=0.9, label=r'$I_+$ burned')
        ax_sig.plot(f_arr, sample['Iminus_unburned'], color='tab:blue', alpha=0.4, linestyle='--', linewidth=1.0, label=r'$I_-$ unburned')
        ax_sig.plot(f_arr, sample['Iminus'], color='tab:blue', linestyle='-', linewidth=1.25, alpha=0.9, label=r'$I_-$ burned')
        _burn_vlines(ax_sig, sample)
        ax_sig.set_ylabel(r'$P_s$, $I_\pm$')
        ax_sig.grid(True, alpha=0.3)
        # ax_sig.legend(loc='upper right', fontsize=leg_fs, ncol=1, title='solid=burned, dashed=unburned')
        ax_sig.set_title('lineshape', fontsize=title_fs)

    for ax in axes[-1, :]:
        ax.set_xlabel('frequency')

    title = 'Spin-1 populations and lineshape vs frequency (ss-RF burn)'
    if suptitle_extra:
        title = f"{title} - {suptitle_extra}"
    fig.suptitle(title, fontsize=13)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


# --- Config ---
num_bins = 249
f = np.linspace(-3, 3, num_bins)
sigma = 0.16
gamma = 0.05

# Polarization range (sample fewer than full lookup table)
P_min, P_max = -0.7, 0.7
num_polarizations = 20  # Many events for large training set
P_values = np.concatenate([
    np.linspace(P_min, -0.1, num_polarizations // 2),
    np.linspace(0.1, P_max, num_polarizations // 2)
])

# Burn locations: sample x0 from valid ranges (avoid center)
x0_negative = np.linspace(-1.5, -0.3, 50)
x0_positive = np.linspace(0.3, 1.5, 50)
burn_locations = np.concatenate([x0_negative, x0_positive])

# Amp range for training
amp_min, amp_max = 5e-4, 5e-1
# Chance to inject a second burn after the mandatory first burn
second_burn_probability = 0.99

# Set True to collect samples and generate grid_plot.png
do_viz = True
# population_burn_example.png: n rows × 2 cols (populations | Ps + I±) per example
num_population_side_by_side_examples = 8

# --- Load lookup table (for mapping burns only) ---
lookup_path = SCRIPT_DIR / "lookup_table.pkl"
_ensure_lookup_table(lookup_path, f)
mapping_data = pd.read_pickle(lookup_path)

# Build lookup tables once (shared across all mappers)
base_mapper = ssRFMapper(f, sigma, gamma, x0=0.0, amp=1e-2)
base_mapper.compute_lookup_tables(mapping_data)

# --- Generate base lineshapes over polarization range ---
print("Generating base lineshapes over polarization range...")
base_events = []
for P in tqdm.tqdm(P_values, desc="Lineshapes"):
    signal, Iplus, Iminus = _generate_vector_lineshape(P, f)
    Ps = signal.copy()
    Qs = Iplus - Iminus
    true_Ps = np.sum(Ps)
    true_Qs = np.sum(Qs)
    base_events.append({
        'P': P,
        'Ps': Ps,
        'Iplus': Iplus.copy(),
        'Iminus': Iminus.copy(),
        'true_Ps': true_Ps,
        'true_Qs': true_Qs,
    })

# --- Apply burns at multiple locations per event ---
print("Generating training samples with random burn injection...")
training_rows = []
collected_for_viz = []
# Reuse single mapper, update x0/amp each call (avoids repeated object creation)
mapper = ssRFMapper(f, sigma, gamma, x0=0.0, amp=1e-4)
mapper.signal_to_iplus_lookup = base_mapper.signal_to_iplus_lookup
mapper.signal_to_iminus_lookup = base_mapper.signal_to_iminus_lookup

viz_locations = [
    burn_locations[0],   # far left (~-0.95)
    burn_locations[12],  # mid-left (~-0.84)
    burn_locations[24],  # near center-left (~-0.73)
    burn_locations[50],  # far right (~0.7)
    burn_locations[62],  # mid-right (~0.81)
    burn_locations[74],  # near center-right (~0.92)
]

# One viz sample per listed event (spread across P) so plots are produced every run.
# Primary burn x0 is drawn from viz_locations for these rows only (still random amp / second burn).
_ne_vis = min(max(1, num_population_side_by_side_examples), len(base_events))
viz_event_indices = tuple(sorted({
    int(round(float(i)))
    for i in np.linspace(0, max(1, len(base_events)) - 1, _ne_vis)
}))
viz_event_indices_set = frozenset(viz_event_indices)

for evt_idx, evt in enumerate(tqdm.tqdm(base_events, desc="Events")):
    Ps_base = evt['Ps'].copy()
    Iplus_base = evt['Iplus'].copy()
    Iminus_base = evt['Iminus'].copy()
    true_Qs = evt['true_Qs']

    # Always inject one burn, then optionally inject a second burn.
    inject_burn = True
    if do_viz and evt_idx in viz_event_indices_set:
        x0 = np.random.choice(viz_locations)
    else:
        x0 = np.random.choice(burn_locations)
    amp = np.random.uniform(amp_min, amp_max)
    inject_second_burn = np.random.rand() < second_burn_probability
    x0_second = np.random.choice(burn_locations) if inject_second_burn else np.nan
    amp_second = np.random.uniform(amp_min, amp_max) if inject_second_burn else 0.0

    Ps_burned = Ps_base.copy()
    Iplus_burned = Iplus_base.copy()
    Iminus_burned = Iminus_base.copy()

    mapper.x0 = x0
    mapper.amp = amp
    _, _, _, rho_plus, rho_zero, rho_minus = mapper.apply_ssRF(
        Ps_burned, Iplus_burned, Iminus_burned, return_burn_info=False
    )

    if inject_second_burn:
        mapper.x0 = x0_second
        mapper.amp = amp_second
        _, _, _, rho_plus, rho_zero, rho_minus = mapper.apply_ssRF(
            Ps_burned, Iplus_burned, Iminus_burned, return_burn_info=False
        )

    # Collect one sample per viz-tracked event (guaranteed when do_viz).
    if do_viz and evt_idx in viz_event_indices_set:
        rp0, rz0, rm0 = AFP.intensities_to_populations(
            np.asarray(Iplus_base, dtype=float),
            np.asarray(Iminus_base, dtype=float),
        )
        collected_for_viz.append({
            'P': evt['P'],
            'x0': x0,
            'amp': amp,
            'x0_second': x0_second,
            'amp_second': amp_second,
            'Ps_unburned': Ps_base.copy(),
            'Iplus_unburned': Iplus_base.copy(),
            'Iminus_unburned': Iminus_base.copy(),
            'Ps': Ps_burned.copy(),
            'Iplus': Iplus_burned.copy(),
            'Iminus': Iminus_burned.copy(),
            'rho_plus_unburned': rp0.copy(),
            'rho_zero_unburned': rz0.copy(),
            'rho_minus_unburned': rm0.copy(),
            'rho_plus': rho_plus.copy(),
            'rho_zero': rho_zero.copy(),
            'rho_minus': rho_minus.copy(),
        })

    true_Ps_burned = np.sum(Ps_burned)
    training_rows.append({
        'P': evt['P'],
        'burn_injected': inject_burn,
        'x0': x0,
        'amp': amp,
        'second_burn_injected': inject_second_burn,
        'x0_second': x0_second,
        'amp_second': amp_second,
        'Ps': Ps_burned,
        'Iminus': Iminus_burned,
        'Iplus': Iplus_burned,
        'rho_plus': rho_plus,
        'rho_zero': rho_zero,
        'rho_minus': rho_minus,
        'true_Ps': true_Ps_burned,
        'true_Qs': true_Qs,
    })

# --- Save training data ---
training_data = pd.DataFrame(training_rows)
testing_output_path = OUTPUT_DIR / "testing_data.pkl"
training_data.to_pickle(testing_output_path)
print(f"\nSaved {len(training_data)} training samples to {testing_output_path}")
print(f"  Base events: {len(base_events)}")
print("  Burn injection probability per event: 1.0")
print(f"  Second burn probability per event: {second_burn_probability}")
if do_viz:
    if collected_for_viz:
        print(f"  Visualization: {len(collected_for_viz)} samples (evt indices {viz_event_indices})")
    else:
        print("  Visualization: no samples collected (set do_viz or check base_events).")

# --- Optional: grid visualization (commented out for large-scale data generation) ---
if collected_for_viz:
    grid_size = int(np.ceil(np.sqrt(len(collected_for_viz))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = np.atleast_2d(axes)
    axes = axes.flatten()

    for idx, data in enumerate(collected_for_viz):
        if idx < len(axes):
            axes[idx].plot(f, data['Ps_unburned'], color='gray', alpha=0.5, linestyle='--', linewidth=0.8)
            axes[idx].plot(f, data['Iplus_unburned'], color='red', alpha=0.2, linestyle='--', linewidth=1)
            axes[idx].plot(f, data['Iminus_unburned'], color='blue', alpha=0.2, linestyle='--', linewidth=1)
            axes[idx].plot(f, data['Ps'], alpha=1.0, linewidth=1, color='black')
            axes[idx].plot(f, data['Iplus'], alpha=0.3, linestyle='-', linewidth=2, color='red')
            axes[idx].plot(f, data['Iminus'], alpha=0.3, linestyle='-', linewidth=2, color='blue')
            primary_burn_x = data['x0']
            mirrored_burn_x = -data['x0']
            axes[idx].axvline(primary_burn_x, color='green', alpha=0.5, linestyle=':', linewidth=1)
            axes[idx].axvline(mirrored_burn_x, color='purple', alpha=0.5, linestyle=':', linewidth=1)
            axes[idx].set_title(
                f"P = {data['P']:.3f}, x0 = {data['x0']:.2f}, amp = {data['amp']:.2e}, "
                f"x0_2 = {data['x0_second']:.2f}, amp_2 = {data['amp_second']:.2e}",
                fontsize=10
            )
            axes[idx].grid(True, alpha=0.3)

    for idx in range(len(collected_for_viz), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Burns at Several Locations (x0) Across Polarizations\n'
                 'P: negative, ~0, positive  |  x0 from ~-0.95 to ~0.92', fontsize=16)
    plt.tight_layout()
    grid_plot_path = OUTPUT_DIR / "grid_plot.png"
    plt.savefig(grid_plot_path, dpi=600)
    plt.close()
    print(f"Saved {grid_plot_path}")

if do_viz and collected_for_viz:
    pop_sources = collected_for_viz
    n_ex = min(len(pop_sources), max(1, int(num_population_side_by_side_examples)))
    chunk = pop_sources[:n_ex]
    extra = f"{len(chunk)} example(s) of {len(pop_sources)} collected"
    _save_population_burn_figure(
        f, chunk, OUTPUT_DIR / "population_burn_example.png", suptitle_extra=extra, dpi=200
    )
