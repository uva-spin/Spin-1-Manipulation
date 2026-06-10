import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QDoubleSpinBox,
    QGroupBox,
    QGridLayout,
    QPushButton,
)
from PyQt5.QtCore import QTimer

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.Lineshape import Baseline, GenerateVectorLineshape
from physics.lineshape.ssRFMapper_sim import ssRFMapper
from scipy.special import wofz

# Lookup-table mapping (legacy — GUI uses population-based ssRFMapper_sim instead):
# import pandas as pd
# from paths import LOOKUP_TABLE_PATH

# Scales population transfer rate so interactive burns are visible (direct Ps edits
# subtract absolute amounts; population transfers are proportional to ρ).
BURN_POWER_SCALE = 100.0


def normalized_voigt(
    x: np.ndarray, x0: float, sigma: float, gamma: float
) -> np.ndarray:
    """Normalized Voigt profile (integral = 1) for interactive burning."""
    x_norm = (x - x0) / (sigma * np.sqrt(2))
    z = x_norm + 1j * (gamma / (sigma * np.sqrt(2)))
    profile = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

    if len(profile) > 5:
        kernel = np.array([0.05, 0.1, 0.15, 0.4, 0.15, 0.1, 0.05])
        profile = np.convolve(profile, kernel, mode="same")

    integral = np.trapezoid(profile, x)
    if integral > 0:
        return profile / integral
    return profile

class InteractivePlot(FigureCanvas):
    """Interactive matplotlib plot."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Main plot 
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('R', fontsize=12, fontweight='bold', color='#2C3E50')
        self.ax.set_ylabel('Signal Amplitude', fontsize=12, fontweight='bold', color='#2C3E50')
        self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#BDC3C7')
        self.ax.set_facecolor('#FAFAFA')
        self.ax.set_xlim(-3, 3)
        self.ax.set_xticks(np.arange(-3, 3.4, 0.4))
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#7F8C8D')
        self.ax.spines['bottom'].set_color('#7F8C8D')
        self.ax.tick_params(colors='#2C3E50', labelsize=10)
        
        # Data storage
        self.frequency = None
        self.original_signal = None
        self.current_signal = None
        self.burned_signal = None
        self.original_baseline = None
        self.current_baseline = None
        self.original_iplus = None
        self.current_iplus = None
        self.original_iminus = None
        self.current_iminus = None
        self.original_iplus_lower = None
        self.original_iminus_lower = None
        self.current_signal_lower = None
        self.burning_active = False
        self.burn_center = None
        self.burn_range = None
        self.center_freq = 0.0 

        
        # Plot lines — reference (dashed) and current (solid)
        self.line_original, = self.ax.plot(
            [], [], color="#2E86AB", linewidth=2, label="Reference Signal",
            alpha=0.7, antialiased=True, linestyle="--",
        )
        self.line_iplus_original, = self.ax.plot(
            [], [], color="#C73E1D", linewidth=1.5, label="Reference I+",
            alpha=0.5, antialiased=True, linestyle="--",
        )
        self.line_iminus_original, = self.ax.plot(
            [], [], color="#7209B7", linewidth=1.5, label="Reference I-",
            alpha=0.5, antialiased=True, linestyle="--",
        )
        self.line_current, = self.ax.plot(
            [], [], color="#A23B72", linewidth=3, label="Current Signal", antialiased=True,
        )
        self.line_baseline, = self.ax.plot(
            [], [], color="#F18F01", linewidth=2, label="Baseline", alpha=0.7, antialiased=True,
        )
        self.line_iplus, = self.ax.plot(
            [], [], color="#C73E1D", linewidth=2, label="I+", alpha=0.9, antialiased=True,
        )
        self.line_iminus, = self.ax.plot(
            [], [], color="#7209B7", linewidth=2, label="I-", alpha=0.9, antialiased=True,
        )
        self.voigt_sigma = 0.01 / 3
        self.voigt_gamma = 0.01 / 6
        
        # Mouse interaction
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Burning parameters
        # self.burn_center_freq = 32.68  # MHz
        self.burn_center_freq = 0.0  # MHz
        self.burn_accumulator = 0.0  # For gradual burn buildup
        self.burn_smoothing_factor = 0.1  # Controls how quickly burn builds up
        
        # Timer for continuous burning with smoother updates
        self.burn_timer = QTimer()
        self.burn_timer.timeout.connect(self.apply_burn_continuously)
        self.burn_timer.setInterval(50)  # 50ms intervals for smoother burning
        
        self.mapper = None
        self._ref_rho_plus = None
        self._ref_rho_zero = None
        self._ref_rho_minus = None
        self._burn_ref_iplus = None
        self._burn_ref_iminus = None
        self._burn_ref_baseline = None
        self._reset_signal = None
        self._reset_iplus = None
        self._reset_iminus = None
        self._reset_baseline = None
        self._mapper_output_scale = 1.0
        self._fixed_ylim = None
        
        self.ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                      framealpha=0.9, fontsize=10, edgecolor='#BDC3C7')
        self.fig.tight_layout()
        
        # Set figure background
        self.fig.patch.set_facecolor('#F8F9FA')
    
    def set_signal_data(
        self,
        frequency: np.ndarray,
        signal: np.ndarray,
        baseline: np.ndarray,
        iplus: np.ndarray,
        iminus: np.ndarray,
        iplus_lower: np.ndarray,
        iminus_lower: np.ndarray,
        signal_lower: np.ndarray,
        *,
        reset_original: bool = True,
    ):
        """Set the signal data to plot.

        When ``reset_original`` is True (parameter change / initial load), snapshot
        the unburned reference traces. During burns only ``current_*`` should change.
        """
        self.frequency = np.asarray(frequency, dtype=float)
        signal = np.asarray(signal, dtype=float)
        baseline = np.asarray(baseline, dtype=float)
        iplus = np.asarray(iplus, dtype=float)
        iminus = np.asarray(iminus, dtype=float)
        iplus_lower = np.asarray(iplus_lower, dtype=float)
        iminus_lower = np.asarray(iminus_lower, dtype=float)
        signal_lower = np.asarray(signal_lower, dtype=float)

        if reset_original:
            # Dashed traces are a fixed backend-reference lineshape that is not
            # modified by burns.
            self.original_signal = signal_lower.copy()
            self.original_baseline = baseline.copy()
            self.original_iplus = iplus_lower.copy()
            self.original_iminus = iminus_lower.copy()
            self.original_iplus_lower = iplus_lower.copy()
            self.original_iminus_lower = iminus_lower.copy()
            self._burn_ref_iplus = iplus.copy()
            self._burn_ref_iminus = iminus.copy()
            self._burn_ref_baseline = baseline.copy()
            self._reset_signal = signal.copy()
            self._reset_iplus = iplus.copy()
            self._reset_iminus = iminus.copy()
            self._reset_baseline = baseline.copy()
            self._snapshot_burn_reference()
            self._fixed_ylim = self._compute_ylim_from_original()

        self.current_signal = signal.copy()
        self.burned_signal = np.zeros_like(signal)
        self.current_baseline = baseline.copy()
        self.current_iplus = iplus.copy()
        self.current_iminus = iminus.copy()
        self.current_iplus_lower = iplus_lower.copy()
        self.current_iminus_lower = iminus_lower.copy()
        self.current_signal_lower = signal_lower.copy()
        self.update_plot()

        if hasattr(self, "polarization_update_callback"):
            self.polarization_update_callback()

    def _snapshot_burn_reference(self):
        """Cache unburned spin-1 populations and round-trip output scale."""
        if (
            self.mapper is None
            or self._burn_ref_iplus is None
            or self._burn_ref_iminus is None
            or self._burn_ref_baseline is None
        ):
            self._ref_rho_plus = None
            self._ref_rho_zero = None
            self._ref_rho_minus = None
            self._mapper_output_scale = 1.0
            return

        iplus = (self._burn_ref_iplus - self._burn_ref_baseline).astype(float)
        iminus = (self._burn_ref_iminus - self._burn_ref_baseline).astype(float)
        rho_plus, rho_zero, rho_minus = self.mapper._load_state_from_intensities(
            iplus, iminus
        )
        self._ref_rho_plus = rho_plus.copy()
        self._ref_rho_zero = rho_zero.copy()
        self._ref_rho_minus = rho_minus.copy()

        # The AFP population-intensity round trip is not exactly unity. Calibrate once
        # so the first burn step does not artificially jump the entire lineshape up.
        _, iplus_rt, iminus_rt = self.mapper._populations_to_outputs(
            rho_plus, rho_zero, rho_minus
        )
        in_sum = float(np.sum(iplus + iminus))
        out_sum = float(np.sum(iplus_rt + iminus_rt))
        if abs(out_sum) > np.finfo(float).eps:
            self._mapper_output_scale = in_sum / out_sum
        else:
            self._mapper_output_scale = 1.0

    def _compute_ylim_from_original(self):
        """Y limits anchored to the unmanipulated traces (stable while burning)."""
        if self.original_signal is None:
            return None
        traces = [
            self.original_signal,
            self.original_iplus,
            self.original_iminus,
            self.current_signal,
            self.current_iplus,
            self.current_iminus,
        ]
        if self.original_baseline is not None and np.any(self.original_baseline):
            traces.append(self.original_baseline)
        traces = [np.asarray(t, dtype=float) for t in traces if t is not None]
        if not traces:
            return None
        y = np.concatenate(traces)
        ymin, ymax = float(np.min(y)), float(np.max(y))
        if ymax <= ymin:
            ymax = ymin + 1.0
        margin = (ymax - ymin) * 0.08
        return ymin - margin, ymax + margin
    
    def update_plot(self):
        """Update the plot with current data with optimized performance."""
        if self.frequency is not None and self.current_signal is not None:
            # Pass copies so matplotlib cannot alias our stored arrays.
            self.line_original.set_data(self.frequency, self.original_signal.copy())
            self.line_iplus_original.set_data(self.frequency, self.original_iplus.copy())
            self.line_iminus_original.set_data(self.frequency, self.original_iminus.copy())
            self.line_current.set_data(self.frequency, self.current_signal.copy())
            self.line_baseline.set_data(self.frequency, self.current_baseline.copy())
            self.line_iplus.set_data(self.frequency, self.current_iplus.copy())
            self.line_iminus.set_data(self.frequency, self.current_iminus.copy())

            if self._fixed_ylim is not None:
                self.ax.set_ylim(self._fixed_ylim)
            elif not self.burning_active:
                self.ax.relim()
                self.ax.autoscale_view()

            if self.burning_active:
                self.draw_idle()
            else:
                self.draw()
    
    
    def calculate_burn_effect(self, center_freq: float):
        """
        Voigt-shaped burn with gradual buildup (same shape/threshold as legacy GUI).

        Returns per-bin strength weights (≥ 0) and the active bin indices.
        """
        voigt_profile = normalized_voigt(
            self.frequency, center_freq, self.voigt_sigma, self.voigt_gamma
        )

        smooth_accumulator = self.burn_accumulator ** 0.7
        burn_threshold = np.max(voigt_profile) * 0.1
        significant_burn_mask = np.where(voigt_profile > burn_threshold)[0]
        strengths = voigt_profile * smooth_accumulator

        return strengths, significant_burn_mask

    def apply_burn(self, center_freq: float):
        """Apply ss-RF burn via difference-proportional intensity decay.

        Each visited bin follows growth then ss-RF decay (in that order), the
        ss-RF driving the two populations toward each other with power
        proportional to the user power constant and the population difference.
        Working directly in intensity space preserves the absolute lineshape
        scale (no population round-trip renormalization).
        """
        if (
            self.frequency is None
            or self.current_signal is None
            or self.mapper is None
            or self.current_baseline is None
        ):
            return

        strengths, burn_indices = self.calculate_burn_effect(center_freq)
        if len(burn_indices) == 0:
            return

        iplus = (self.current_iplus - self.current_baseline).astype(float)
        iminus = (self.current_iminus - self.current_baseline).astype(float)

        if (
            self._burn_ref_iplus is not None
            and self._burn_ref_iminus is not None
            and self._burn_ref_baseline is not None
        ):
            ref_iplus = (self._burn_ref_iplus - self._burn_ref_baseline).astype(float)
            ref_iminus = (self._burn_ref_iminus - self._burn_ref_baseline).astype(float)
        else:
            ref_iplus = iplus.copy()
            ref_iminus = iminus.copy()

        # One GUI timer tick ≡ one simulation step; buildup is in ``strengths``.
        dt = 1.0

        for freq_bin in burn_indices:
            strength = float(strengths[freq_bin])
            if strength <= 0.0:
                continue
            self.mapper._step_bin(
                iplus,
                iminus,
                ref_iplus,
                ref_iminus,
                int(freq_bin),
                strength,
                dt,
            )

        ps = iplus + iminus
        self.current_signal = np.asarray(ps + self.current_baseline, dtype=float)
        self.current_iplus = np.asarray(iplus + self.current_baseline, dtype=float)
        self.current_iminus = np.asarray(iminus + self.current_baseline, dtype=float)

        self.update_plot()
        self.polarization_update_callback()

    # Lookup-table burn path (legacy):
    # def apply_burn_lookup(self, center_freq: float):
    #     burn_effect, significant_burn_mask = self.calculate_burn_effect_legacy(...)
    #     self.current_signal[significant_burn_mask] += burn_effect
    #     ...
    #     iplus, iminus = self.mapper.map_signal_to_iplus_iminus(burned_lineshape)
    
    def apply_burn_continuously(self):
        if self.burning_active and self.burn_center is not None:
            self.burn_accumulator = min(1.0, self.burn_accumulator + self.burn_smoothing_factor)
            self.apply_burn(self.burn_center)
    
    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax or event.button != 1:  # Left mouse button
            return
        
        if self.frequency is None:
            return
        
        freq = event.xdata
        if freq is None:
            return
        
        if freq < self.frequency.min() or freq > self.frequency.max():
            return
        
        self.burning_active = True
        self.burn_center = freq
        self.burn_accumulator = 0.0 
        self.burn_timer.start()
        
        self.update_plot()
    
    def on_release(self, event):
        if event.button == 1:  # Left mouse button
            self.burning_active = False
            self.burn_center = None
            self.burn_accumulator = 0.0 
            self.burn_timer.stop()
        
            self.update_plot()
    
    def on_motion(self, event):
        if not self.burning_active or event.inaxes != self.ax:
            return
        
        if self.frequency is None:
            return
        
        # Update burn center to follow mouse
        freq = event.xdata
        if freq is not None and self.frequency.min() <= freq <= self.frequency.max():
            self.burn_center = freq
            self.update_plot()
    
    def reset_signal(self):
        if self._reset_signal is not None:
            self.current_signal = self._reset_signal.copy()
            self.burned_signal = np.zeros_like(self._reset_signal)
            self.current_baseline = self._reset_baseline.copy()
            self.current_iplus = self._reset_iplus.copy()
            self.current_iminus = self._reset_iminus.copy()
            
            self.burn_accumulator = 0.0
            self.burning_active = False
            self.burn_center = None
            # self.baseline_constrained_bins = None
            # self.burn_applied_bins = None
            
            
            self.update_plot()
            
            self.polarization_update_callback()
    
    def set_burn_parameters(self, intensity: float, width: float):
        self.burn_intensity = intensity
        self.burn_width = width
        self.voigt_sigma = width / 3
        self.voigt_gamma = width / 6

class TensorSymGUI(QMainWindow):
    """Main GUI window for TensorSym application."""
    
    def __init__(self, withBaseline: bool = True):
        super().__init__()
        self.setWindowTitle("TensorSym - Deuterium Signal Analysis")
        self.setGeometry(100, 100, 1600, 1000)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F8F9FA;
            }
            QWidget {
                background-color: #F8F9FA;
                color: #2C3E50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #2C3E50;
                font-weight: 500;
            }
        """)
        
        self.withBaseline = withBaseline
        self.frequency = np.linspace(-3, 3, 249)
        self.lower_polarization = 0.25

        self.setup_ui()
        self._init_mapper()
        self.update_signal()
        self.update_polarization_display()

    def _init_mapper(self):
        """Initialize population-based ssRFMapper (no lookup table)."""
        try:
            burn_intensity = self.burn_intensity_spinbox.value()
            burn_width = self.burn_width_spinbox.value()
            self.mapper = ssRFMapper(
                self.frequency,
                sigma=burn_width / 3,
                gamma=burn_width / 6,
                x0=0.0,
                amp=burn_intensity,
                power_constant=burn_intensity * BURN_POWER_SCALE,
                center_freq=0.0,
            )
            self.plot.mapper = self.mapper
            self.plot._snapshot_burn_reference()

            # Lookup-table mapping (legacy):
            # lookup_df = pd.read_pickle(LOOKUP_TABLE_PATH)
            # self.mapper.compute_lookup_tables(lookup_df)
            # print(f"Loaded lookup table from {LOOKUP_TABLE_PATH}")
        except Exception as exc:
            print(f"Failed to initialize ssRFMapper: {exc}")
            self.mapper = None
            self.plot.mapper = None
        
    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Right panel for plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # Interactive plot
        self.plot = InteractivePlot(plot_widget, width=10, height=8)
        self.plot.polarization_update_callback = self.update_polarization_display
        plot_layout.addWidget(self.plot)
        
        # # Plot controls
        # plot_controls = self.create_plot_controls()
        # plot_layout.addWidget(plot_controls)
        
        main_layout.addWidget(plot_widget, 3)
    
    def create_control_panel(self):
        """Create the control panel with parameter controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Deuteron parameters
        deut_group = QGroupBox("Deuteron Parameters")
        deut_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #2E86AB;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        deut_layout = QGridLayout(deut_group)
        
        deut_layout.addWidget(QLabel("Polarization (P):"), 0, 0)
        self.polarization_spinbox = QDoubleSpinBox()
        self.polarization_spinbox.setRange(-0.99, 0.99)
        self.polarization_spinbox.setSingleStep(0.05)
        self.polarization_spinbox.setValue(0.6)
        self.polarization_spinbox.setDecimals(3)
        self.polarization_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #2E86AB;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.polarization_spinbox.valueChanged.connect(self.update_signal)
        deut_layout.addWidget(self.polarization_spinbox, 0, 1)
        
        layout.addWidget(deut_group)
        
        # Baseline parameters
        base_group = QGroupBox("Baseline Parameters")
        base_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #F18F01;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        base_layout = QGridLayout(base_group)
        
        # Voltage
        base_layout.addWidget(QLabel("Voltage (U):"), 0, 0)
        self.U_spinbox = QDoubleSpinBox()
        self.U_spinbox.setRange(0.1, 10.0)
        self.U_spinbox.setSingleStep(0.1)
        self.U_spinbox.setValue(2.4283)
        self.U_spinbox.setDecimals(6)
        self.U_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #F18F01;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.U_spinbox.valueChanged.connect(self.update_signal)
        base_layout.addWidget(self.U_spinbox, 0, 1)
        
        # Cknob
        base_layout.addWidget(QLabel("Cknob:"), 1, 0)
        self.Cknob_spinbox = QDoubleSpinBox()
        self.Cknob_spinbox.setRange(0.0001, 10.0)
        self.Cknob_spinbox.setSingleStep(0.001)
        self.Cknob_spinbox.setValue(0.199000)
        self.Cknob_spinbox.setDecimals(6)
        self.Cknob_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #F18F01;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.Cknob_spinbox.valueChanged.connect(self.update_signal)
        base_layout.addWidget(self.Cknob_spinbox, 1, 1)
        
        # Eta
        base_layout.addWidget(QLabel("Eta (η):"), 2, 0)
        self.eta_spinbox = QDoubleSpinBox()
        self.eta_spinbox.setRange(0.00001, 0.1)
        self.eta_spinbox.setSingleStep(0.001)
        self.eta_spinbox.setValue(0.0104)
        self.eta_spinbox.setDecimals(8)
        self.eta_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #F18F01;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.eta_spinbox.valueChanged.connect(self.update_signal)
        base_layout.addWidget(self.eta_spinbox, 2, 1)
        
        # Trim
        base_layout.addWidget(QLabel("Trim:"), 3, 0)
        self.trim_spinbox = QDoubleSpinBox()
        # self.trim_spinbox.setRange(0.01, 36.0)
        self.trim_spinbox.setSingleStep(0.01)
        self.trim_spinbox.setValue(3)
        self.trim_spinbox.setDecimals(4)
        self.trim_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #F18F01;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.trim_spinbox.valueChanged.connect(self.update_signal)
        base_layout.addWidget(self.trim_spinbox, 3, 1)
        
        # Cstray
        base_layout.addWidget(QLabel("Cstray:"), 4, 0)
        self.Cstray_spinbox = QDoubleSpinBox()
        self.Cstray_spinbox.setRange(1e-25, 1e-15)
        # self.Cstray_spinbox.setSingleStep(1e-20)
        self.Cstray_spinbox.setValue(0.00000000000000000020)
        self.Cstray_spinbox.setDecimals(22)
        self.Cstray_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #F18F01;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.Cstray_spinbox.valueChanged.connect(self.update_signal)
        base_layout.addWidget(self.Cstray_spinbox, 4, 1)
        
        # Phi constant
        base_layout.addWidget(QLabel("Phi Constant:"), 5, 0)
        self.phi_const_spinbox = QDoubleSpinBox()
        self.phi_const_spinbox.setRange(-180.0, 180.0)
        self.phi_const_spinbox.setSingleStep(1.0)
        self.phi_const_spinbox.setValue(6.19)
        self.phi_const_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #F18F01;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.phi_const_spinbox.valueChanged.connect(self.update_signal)
        base_layout.addWidget(self.phi_const_spinbox, 5, 1)
        
        # DC offset
        base_layout.addWidget(QLabel("DC Offset:"), 6, 0)
        self.DC_offset_spinbox = QDoubleSpinBox()
        self.DC_offset_spinbox.setRange(-10.0, 10.0)
        self.DC_offset_spinbox.setSingleStep(0.1)
        self.DC_offset_spinbox.setValue(0.0)
        self.DC_offset_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #F18F01;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.DC_offset_spinbox.valueChanged.connect(self.update_signal)
        base_layout.addWidget(self.DC_offset_spinbox, 6, 1)
        
        layout.addWidget(base_group)
        
        # Voigt function parameters
        voigt_group = QGroupBox("Voigt Burn Parameters")
        voigt_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #06FFA5;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        voigt_layout = QGridLayout(voigt_group)
        
        # Burn intensity
        voigt_layout.addWidget(QLabel("Burn Intensity:"), 0, 0)
        self.burn_intensity_spinbox = QDoubleSpinBox()
        self.burn_intensity_spinbox.setRange(0.000001, 0.01)
        self.burn_intensity_spinbox.setSingleStep(0.0005)
        self.burn_intensity_spinbox.setValue(0.0005)
        self.burn_intensity_spinbox.setDecimals(6)
        self.burn_intensity_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #06FFA5;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.burn_intensity_spinbox.valueChanged.connect(self.update_burn_parameters)
        voigt_layout.addWidget(self.burn_intensity_spinbox, 0, 1)
        
        # Burn width
        voigt_layout.addWidget(QLabel("Burn Width (MHz):"), 1, 0)
        self.burn_width_spinbox = QDoubleSpinBox()
        self.burn_width_spinbox.setRange(0.0001, 1.0)
        self.burn_width_spinbox.setSingleStep(0.05)
        self.burn_width_spinbox.setValue(0.1)
        self.burn_width_spinbox.setDecimals(6)
        self.burn_width_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #06FFA5;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.burn_width_spinbox.valueChanged.connect(self.update_burn_parameters)
        voigt_layout.addWidget(self.burn_width_spinbox, 1, 1)
        
        layout.addWidget(voigt_group)
        
        # Calibration and Polarization Display
        calc_group = QGroupBox("Calculated Polarization")
        calc_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid purple;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        calc_layout = QGridLayout(calc_group)
        
        # Calibration constant
        calc_layout.addWidget(QLabel("Calibration Constant (C):"), 0, 0)
        self.calibration_spinbox = QDoubleSpinBox()
        self.calibration_spinbox.setRange(0.001, 1000.0)
        self.calibration_spinbox.setSingleStep(0.1)
        self.calibration_spinbox.setValue(1.0)
        self.calibration_spinbox.setDecimals(6)
        self.calibration_spinbox.setStyleSheet("""
            QDoubleSpinBox {
                border: 2px solid #7209B7;
                border-radius: 4px;
                padding: 4px;
                background-color: #F8F9FA;
            }
        """)
        self.calibration_spinbox.valueChanged.connect(self.update_polarization_display)
        calc_layout.addWidget(self.calibration_spinbox, 0, 1)
        
        # Total area display
        calc_layout.addWidget(QLabel("Total Area (I+ + I-):"), 1, 0)
        self.area_label = QLabel("0.000")
        self.area_label.setStyleSheet("""
            QLabel {
                border: 2px solid #7209B7;
                border-radius: 4px;
                padding: 4px;
                background-color: #F0F0F0;
                font-weight: bold;
                color: #2C3E50;
            }
        """)
        calc_layout.addWidget(self.area_label, 1, 1)
        
        # Calculated vector polarization display
        calc_layout.addWidget(QLabel("Calculated Vector Polarization:"), 2, 0)
        self.vector_polarization_calc_label = QLabel("0.000")
        self.vector_polarization_calc_label.setStyleSheet("""
            QLabel {
                border: 2px solid #7209B7;
                border-radius: 4px;
                padding: 4px;
                background-color: #E8F4F8;
                font-weight: bold;
                color: #2C3E50;
                font-size: 14px;
            }
        """)
        calc_layout.addWidget(self.vector_polarization_calc_label, 2, 1)
        
        layout.addWidget(calc_group)
        
        # Calculated tensor display
        calc_layout.addWidget(QLabel("Calculated Tensor Polarization:"), 3, 0)
        self.tensor_polarization_calc_label = QLabel("0.000")
        self.tensor_polarization_calc_label.setStyleSheet("""
            QLabel {
                border: 2px solid #7209B7;
                border-radius: 4px;
                padding: 4px;
                background-color: #F0F0F0;
                font-weight: bold;
                color: #2C3E50;
            }
        """)
        calc_layout.addWidget(self.tensor_polarization_calc_label, 3, 1)
        
        layout.addWidget(calc_group)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.reset_button = QPushButton("Reset Signal")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #A23B72;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #8B2F5F;
            }
            QPushButton:pressed {
                background-color: #6B1F3F;
            }
        """)
        self.reset_button.clicked.connect(self.reset_signal)
        button_layout.addWidget(self.reset_button)
        
        self.save_button = QPushButton("Save Plot")
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #2E86AB;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1E6B8B;
            }
            QPushButton:pressed {
                background-color: #0E4B6B;
            }
        """)
        self.save_button.clicked.connect(self.save_plot)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_plot_controls(self):
        """Create plot control widgets."""
        controls = QWidget()
        controls.setStyleSheet("""
            QWidget {
                background-color: #E8F4F8;
                border: 2px solid #2E86AB;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        layout = QHBoxLayout(controls)
    
    def update_signal(self):
        """Update the signal with current parameters."""
        polarization = self.polarization_spinbox.value()
        lineshape_data, iplus_data, iminus_data = GenerateVectorLineshape(
            polarization, self.frequency
        )
        _, iplus_lower_data, iminus_lower_data = GenerateVectorLineshape(
            self.lower_polarization, self.frequency
        )
        signal_lower_data = iplus_lower_data + iminus_lower_data

        if self.withBaseline:
            baseline_data = Baseline(
                self.frequency,
                self.U_spinbox.value(),
                self.Cknob_spinbox.value(),
                self.eta_spinbox.value(),
                self.trim_spinbox.value(),
                self.Cstray_spinbox.value(),
                self.phi_const_spinbox.value(),
                self.DC_offset_spinbox.value(),
            )
        else:
            baseline_data = np.zeros_like(lineshape_data)

        signal_data = baseline_data + lineshape_data

        self.plot.set_burn_parameters(self.burn_intensity_spinbox.value(), self.burn_width_spinbox.value())
        
        # Update plot
        self.plot.set_signal_data(self.frequency, signal_data, baseline_data, iplus_data + baseline_data, iminus_data + baseline_data, iplus_lower_data, iminus_lower_data, signal_lower_data)
        
        # Update polarization display
        self.update_polarization_display()
    
    def calculate_total_area(self):
        """Calculate the total area of I+ + I- components."""
        if self.plot.current_iplus is not None and self.plot.current_iminus is not None:
            # Calculate area using trapezoidal integration
            # total_signal = self.plot.current_iplus + self.plot.current_iminus
            # area = np.trapezoid(total_signal, self.plot.frequency)
            # return area
            total_signal = np.sum(self.plot.current_iplus + self.plot.current_iminus)
            return total_signal
        return 0.0
    
    def update_polarization_display(self):
        """Update the calculated polarization display."""
        # Calculate total area
        total_area = self.calculate_total_area()
        
        # Update area display
        self.area_label.setText(f"{total_area:.6f}")
        
        # calibration_constant = self.calibration_spinbox.value()
        

        if self.plot.current_iplus is not None and self.plot.current_iminus is not None:
            iplus = self.plot.current_iplus - self.plot.current_baseline
            iminus = self.plot.current_iminus - self.plot.current_baseline
            calculated_vector_polarization = np.sum(iplus + iminus)
            calculated_tensor_polarization = np.sum(iplus - iminus)
        else:
            calculated_vector_polarization = 0.0
            calculated_tensor_polarization = 0.0

        # Lookup-table polarization estimate (legacy):
        # if self.plot.mapper is not None:
        #     iplus_results, iminus_results = self.plot.mapper.map_signal_to_iplus_iminus(
        #         burned_deuteron_signal
        #     )
        #     calculated_vector_polarization = np.sum(iplus_results + iminus_results)
        #     calculated_tensor_polarization = np.sum(iplus_results - iminus_results)
        
        
        # Update polarization display
        self.vector_polarization_calc_label.setText(f"{calculated_vector_polarization:.4f}")
        self.tensor_polarization_calc_label.setText(f"{calculated_tensor_polarization:.4f}")
        
        # Force GUI update
        self.vector_polarization_calc_label.repaint()
        self.tensor_polarization_calc_label.repaint()
        self.area_label.repaint()
    
    def update_burn_parameters(self):
        """Update burn parameters."""
        intensity = self.burn_intensity_spinbox.value()
        width = self.burn_width_spinbox.value()
        self.plot.set_burn_parameters(intensity, width)
        if self.mapper is not None:
            self.mapper.amp = intensity
            self.mapper.power_constant = intensity * BURN_POWER_SCALE
            self.mapper.sigma = width / 3
            self.mapper.gamma = width / 6
    
    def reset_signal(self):
        """Reset signal to original state."""
        self.plot.reset_signal()
        # Update polarization display after reset
        self.update_polarization_display()
    
    def save_plot(self):
        """Save the current plot."""
        filename = f"tensorsym_plot_P_{self.polarization_spinbox.value():.3f}.png"
        self.plot.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
        print(f"Plot saved as {filename}")

def main():
    """Main function to run the application."""
    withBaseline = False
    app = QApplication(sys.argv)
    window = TensorSymGUI(withBaseline)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
