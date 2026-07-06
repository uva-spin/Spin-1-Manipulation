"""Compact PyQt GUI for the spin-1 ss-RF real-time simulation, v14.

Two live plots are shown:
1. Current overlapping Pake-doublet spectrum with the RF and mirror bins.
2. I+(R_RF,t) and I-(R_RF,t) at the selected burn location.

The recovery controls are base scales.  The model computes the actual local
same-theta and left/right neighbor recovery rates from the selected R and the
instantaneous populations; these values are displayed in the text readout.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

import numpy as np

from .model import Spin1Model, Spin1Params


def _load_qt():
    errors = []
    for package in ("PyQt6", "PySide6", "PyQt5"):
        try:
            if package == "PyQt6":
                from PyQt6 import QtCore, QtWidgets  # type: ignore
            elif package == "PySide6":
                from PySide6 import QtCore, QtWidgets  # type: ignore
            else:
                from PyQt5 import QtCore, QtWidgets  # type: ignore
            return QtCore, QtWidgets, package
        except Exception as exc:  # pragma: no cover
            errors.append(f"{package}: {exc}")
    raise RuntimeError("No Qt binding was found. Install PyQt6, PySide6, or PyQt5.\n" + "\n".join(errors))


QtCore, QtWidgets, QT_BINDING = _load_qt()

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # type: ignore
from matplotlib.figure import Figure


class Spin1RealtimeWindow(QtWidgets.QMainWindow):
    def __init__(self, params: Optional[Spin1Params] = None):
        super().__init__()
        self.setWindowTitle("Spin-1 ss-RF real-time simulator v14")
        self.model = Spin1Model(params or Spin1Params())
        self.steps_per_tick = 12
        self.paused = False
        self.trace_t0 = self.model.t
        self.trace_t: list[float] = []
        self.trace_Ip_R: list[float] = []
        self.trace_Im_R: list[float] = []
        self.trace_max_points = 4500

        self._build_ui()
        self._init_plots()
        self._start_new_trace(record_now=True)
        self._update_plots()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(35)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _spin_box(self, label: str, value: float, lo: float, hi: float, step: float, decimals: int, callback: Callable[[float], None]):
        row = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(label)
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(value)
        box.valueChanged.connect(callback)
        row.addWidget(lab)
        row.addWidget(box)
        return row, box

    def _int_box(self, label: str, value: int, lo: int, hi: int, step: int, callback: Callable[[int], None]):
        row = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(label)
        box = QtWidgets.QSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setValue(value)
        box.valueChanged.connect(callback)
        row.addWidget(lab)
        row.addWidget(box)
        return row, box

    def _readonly_value(self, label: str):
        row = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel(label)
        lab.setMinimumWidth(120)
        box = QtWidgets.QLineEdit()
        box.setReadOnly(True)
        box.setMinimumWidth(92)
        box.setStyleSheet("font-family: monospace; font-size: 11px; padding: 2px;")
        row.addWidget(lab)
        row.addWidget(box)
        return row, box

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main = QtWidgets.QHBoxLayout(central)
        main.setContentsMargins(5, 5, 5, 5)
        main.setSpacing(6)

        self.fig = Figure(figsize=(7.4, 3.8), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_spectrum_click)
        main.addWidget(self.canvas, stretch=5)

        side_widget = QtWidgets.QWidget()
        side_widget.setMaximumWidth(300)
        side_outer = QtWidgets.QVBoxLayout(side_widget)
        side_outer.setContentsMargins(0, 0, 0, 0)
        side_outer.setSpacing(5)
        main.addWidget(side_widget, stretch=0)

        pol_box = QtWidgets.QGroupBox("Live polarization")
        pol_layout = QtWidgets.QVBoxLayout(pol_box)
        pol_layout.setContentsMargins(6, 6, 6, 6)
        self.p_readout = QtWidgets.QLineEdit()
        self.p_readout.setReadOnly(True)
        self.p_readout.setStyleSheet("font-weight: bold; font-size: 15px; padding: 3px;")
        self.q_readout = QtWidgets.QLabel()
        self.q_readout.setStyleSheet("font-size: 11px;")
        pol_layout.addWidget(self.p_readout)
        pol_layout.addWidget(self.q_readout)
        side_outer.addWidget(pol_box, stretch=0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        try:
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        except Exception:  # pragma: no cover
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_widget = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(controls_widget)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(4)
        scroll.setWidget(controls_widget)
        side_outer.addWidget(scroll, stretch=1)

        run_box = QtWidgets.QGroupBox("Run")
        run_layout = QtWidgets.QVBoxLayout(run_box)
        self.rf_button = QtWidgets.QPushButton()
        self.rf_button.setCheckable(True)
        self.rf_button.setChecked(bool(self.model.params.rf_enabled))
        self.rf_button.toggled.connect(self._toggle_rf)
        self._update_rf_button()
        self.dnp_button = QtWidgets.QPushButton()
        self.dnp_button.setCheckable(True)
        self.dnp_button.setChecked(bool(self.model.params.dnp_enabled))
        self.dnp_button.toggled.connect(self._toggle_dnp)
        self._update_dnp_button()
        self.pause_button = QtWidgets.QPushButton("Pause simulation")
        self.pause_button.clicked.connect(self._toggle_pause)
        reset_button = QtWidgets.QPushButton("Reset populations")
        reset_button.clicked.connect(self._reset_model)
        restart_button = QtWidgets.QPushButton("Restart lower trace")
        restart_button.clicked.connect(lambda: self._start_new_trace(record_now=True))
        for w in (self.rf_button, self.dnp_button, self.pause_button, reset_button, restart_button):
            run_layout.addWidget(w)
        side.addWidget(run_box)

        rf_box = QtWidgets.QGroupBox("RF bin")
        rf_layout = QtWidgets.QVBoxLayout(rf_box)
        row, self.R_box = self._spin_box("physical R", self.model.params.rf_burn_R, -2.95, 2.95, 0.01, 3, self._set_R)
        rf_layout.addLayout(row)
        row, self.gamma_box = self._spin_box("common Γ_RF", self.model.params.gamma_rf, 0.0, 50.0, 0.1, 3, self._set_gamma_rf)
        rf_layout.addLayout(row)
        hint = QtWidgets.QLabel("Tip: click the top plot to choose R.")
        hint.setWordWrap(True)
        rf_layout.addWidget(hint)
        side.addWidget(rf_box)

        live_rates_box = QtWidgets.QGroupBox("Recovery rates at selected R (live)")
        live_rates_layout = QtWidgets.QVBoxLayout(live_rates_box)
        self.rate_readouts = {}
        live_rows = [
            ("sameθ I+ +↔0", "same_plus"),
            ("sameθ I- 0↔-", "same_minus"),
            ("neighbor I+ L/R", "neigh_plus"),
            ("neighbor I- L/R", "neigh_minus"),
            ("RF-off dI+/dt", "refill_plus"),
            ("RF-off dI-/dt", "refill_minus"),
        ]
        for label, key in live_rows:
            row, box = self._readonly_value(label)
            live_rates_layout.addLayout(row)
            self.rate_readouts[key] = box
        live_note = QtWidgets.QLabel("These are computed values. They change with physical R and with the burned populations.")
        live_note.setWordWrap(True)
        live_note.setStyleSheet("font-size: 9px;")
        live_rates_layout.addWidget(live_note)
        side.addWidget(live_rates_box)

        dnp_box = QtWidgets.QGroupBox("DNP")
        dnp_layout = QtWidgets.QVBoxLayout(dnp_box)
        row, self.p_dnp_sat_box = self._spin_box("P saturation", self.model.params.p_dnp_sat, -0.99, 0.99, 0.01, 3, self._set_p_dnp_sat)
        dnp_layout.addLayout(row)
        row, self.dnp_rate_box = self._spin_box("DNP build rate", self.model.params.dnp_rate, 0.0, 20.0, 0.01, 4, self._set_dnp_rate)
        dnp_layout.addLayout(row)
        side.addWidget(dnp_box)

        line_box = QtWidgets.QGroupBox("Lineshape")
        line_layout = QtWidgets.QVBoxLayout(line_box)
        row, self.gamma_line_box = self._spin_box("Γ broadening", self.model.params.line_gamma, 0.001, 0.5, 0.005, 4, self._set_line_gamma_reset)
        line_layout.addLayout(row)
        row, self.asym_box = self._spin_box("η cos2φ", self.model.params.line_asym, -0.5, 0.5, 0.005, 4, self._set_line_asym_reset)
        line_layout.addLayout(row)
        row, self.display_scale_box = self._spin_box("plot scale", self.model.params.display_scale, 0.01, 20.0, 0.05, 3, self._set_display_scale_reset)
        line_layout.addLayout(row)
        side.addWidget(line_box)

        diff_box = QtWidgets.QGroupBox("Material scale knobs (fixed)")
        diff_layout = QtWidgets.QVBoxLayout(diff_box)
        controls = [
            ("base same-θ +↔0", self.model.params.d_same_plus0, 0.0, 50.0, 0.05, 3, self._set_d_same_plus0),
            ("base same-θ 0↔-", self.model.params.d_same_0minus, 0.0, 50.0, 0.05, 3, self._set_d_same_0minus),
            ("base neighbor +↔0", self.model.params.d_spec_plus0, 0.0, 100.0, 0.25, 3, self._set_d_spec_plus0),
            ("base neighbor 0↔-", self.model.params.d_spec_0minus, 0.0, 100.0, 0.25, 3, self._set_d_spec_0minus),
            ("mirror drive", self.model.params.same_theta_mirror_gain, 0.0, 10.0, 0.1, 3, self._set_same_theta_mirror_gain),
            ("population weighting", self.model.params.population_availability, 0.0, 1.0, 0.05, 3, self._set_population_availability),
            ("T2 width ΔR", self.model.params.t2_width_R, 0.001, 0.5, 0.005, 4, self._set_t2_width),
            ("R-dep power", self.model.params.recovery_position_power, 0.0, 4.0, 0.1, 3, self._set_recovery_position_power),
            ("R-dep floor", self.model.params.recovery_position_floor, 0.0, 1.0, 0.01, 3, self._set_recovery_position_floor),
        ]
        for label, value, lo, hi, step, dec, cb in controls:
            row, _box = self._spin_box(label, value, lo, hi, step, dec, cb)
            diff_layout.addLayout(row)
        note = QtWidgets.QLabel("These knobs are fixed material-scale multipliers. The live recovery rates shown above are recomputed from R and current populations.")
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 9px;")
        diff_layout.addWidget(note)
        side.addWidget(diff_box)

        numerics = QtWidgets.QGroupBox("Initial / numeric")
        num_layout = QtWidgets.QVBoxLayout(numerics)
        num_controls = [
            ("initial P", self.model.params.p0, -0.99, 0.99, 0.01, 3, self._set_p0_reset),
            ("T1 rate", self.model.params.t1_rate, 0.0, 10.0, 0.01, 4, self._set_t1_rate),
            ("T1 P_eq", self.model.params.t1_p_eq, -0.99, 0.99, 0.01, 3, self._set_t1_p_eq),
            ("dt", self.model.params.dt, 1e-5, 0.05, 0.0005, 5, self._set_dt),
            ("plot noise", self.model.params.noise_sigma, 0.0, 0.02, 0.0001, 5, self._set_noise),
        ]
        for label, value, lo, hi, step, dec, cb in num_controls:
            row, _box = self._spin_box(label, value, lo, hi, step, dec, cb)
            num_layout.addLayout(row)
        row, self.steps_box = self._int_box("steps/tick", self.steps_per_tick, 1, 1000, 1, self._set_steps)
        num_layout.addLayout(row)
        side.addWidget(numerics)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumWidth(255)
        self.info_label.setStyleSheet("font-size: 10px;")
        side.addWidget(self.info_label)
        side.addStretch(1)

    def _init_plots(self) -> None:
        self.ax_spec = self.fig.add_subplot(2, 1, 1)
        self.ax_trace = self.fig.add_subplot(2, 1, 2)
        self.line_Ip, = self.ax_spec.plot([], [], drawstyle="steps-mid", label="I+(R)")
        self.line_Im, = self.ax_spec.plot([], [], drawstyle="steps-mid", label="I-(R)")
        self.line_total, = self.ax_spec.plot([], [], linewidth=1.3, label="total")
        self.burn_line = self.ax_spec.axvline(self.model.params.rf_burn_R, linestyle="--", linewidth=1.1, label="RF bin")
        self.mirror_line = self.ax_spec.axvline(-self.model.params.rf_burn_R, linestyle=":", linewidth=1.1, label="mirror")
        self.point_Ip, = self.ax_spec.plot([], [], marker="o", linestyle="None", markersize=5, zorder=7, label="I+ direct")
        self.point_Im, = self.ax_spec.plot([], [], marker="s", linestyle="None", markersize=5, zorder=7, label="I- direct")
        self.point_Ip_m, = self.ax_spec.plot([], [], marker="^", linestyle="None", markersize=5, zorder=7, label="I+ mirror")
        self.point_Im_m, = self.ax_spec.plot([], [], marker="v", linestyle="None", markersize=5, zorder=7, label="I- mirror")
        self.bar_Ip_R, = self.ax_spec.plot([], [], linewidth=4.0, color=self.line_Ip.get_color(), zorder=6, label="_nolegend_")
        self.bar_Im_R, = self.ax_spec.plot([], [], linewidth=4.0, color=self.line_Im.get_color(), zorder=6, label="_nolegend_")
        self.bar_Ip_mR, = self.ax_spec.plot([], [], linewidth=4.0, color=self.line_Ip.get_color(), zorder=6, label="_nolegend_")
        self.bar_Im_mR, = self.ax_spec.plot([], [], linewidth=4.0, color=self.line_Im.get_color(), zorder=6, label="_nolegend_")
        self.ax_spec.set_xlabel("physical R")
        self.ax_spec.set_ylabel("intensity [arb.]")
        self.ax_spec.set_title("Overlapping Pake-doublet components")
        self.ax_spec.legend(loc="upper left", ncols=4, fontsize=7)
        self.trace_line_Ip_R, = self.ax_trace.plot([], [], label="I+(R_RF,t)")
        self.trace_line_Im_R, = self.ax_trace.plot([], [], label="I-(R_RF,t)")
        self.ax_trace.set_xlabel("time since this R was selected [arb.]")
        self.ax_trace.set_ylabel("intensity at RF bin [arb.]")
        self.ax_trace.set_title("Burn-location intensities from selected initial values")
        self.ax_trace.legend(loc="best", fontsize=8)

    def rf_is_on(self) -> bool:
        return bool(self.rf_button.isChecked())

    def _update_rf_button(self) -> None:
        self.rf_button.setText("RF ON" if self.rf_button.isChecked() else "RF OFF")
        self.rf_button.setStyleSheet("font-weight: bold; padding: 6px;")

    def _toggle_rf(self, checked: bool) -> None:
        self.model.set_rf_enabled(bool(checked))
        self._update_rf_button()
        self._record_trace_point()
        self._update_plots()

    def dnp_is_on(self) -> bool:
        return bool(self.dnp_button.isChecked())

    def _update_dnp_button(self) -> None:
        self.dnp_button.setText("DNP ON" if self.dnp_button.isChecked() else "DNP OFF")
        self.dnp_button.setStyleSheet("font-weight: bold; padding: 6px;")

    def _toggle_dnp(self, checked: bool) -> None:
        self.model.set_dnp_enabled(bool(checked))
        self._update_dnp_button()
        self._record_trace_point()
        self._update_plots()

    def _set_R(self, value: float) -> None:
        self.model.params.rf_burn_R = float(value)
        self._start_new_trace(record_now=True)
        self._update_plots()

    def _on_spectrum_click(self, event) -> None:
        if event.inaxes is not self.ax_spec or event.xdata is None:
            return
        self.R_box.setValue(max(-2.95, min(2.95, float(event.xdata))))

    def _set_gamma_rf(self, value: float) -> None: self.model.params.gamma_rf = float(value)
    def _set_d_same_plus0(self, value: float) -> None: self.model.params.d_same_plus0 = float(value)
    def _set_d_same_0minus(self, value: float) -> None: self.model.params.d_same_0minus = float(value)
    def _set_d_spec_plus0(self, value: float) -> None: self.model.params.d_spec_plus0 = float(value)
    def _set_d_spec_0minus(self, value: float) -> None: self.model.params.d_spec_0minus = float(value)
    def _set_same_theta_mirror_gain(self, value: float) -> None: self.model.params.same_theta_mirror_gain = float(value)
    def _set_population_availability(self, value: float) -> None: self.model.params.population_availability = float(value)
    def _set_t2_width(self, value: float) -> None: self.model.params.t2_width_R = float(value)
    def _set_recovery_position_power(self, value: float) -> None: self.model.params.recovery_position_power = float(value)
    def _set_recovery_position_floor(self, value: float) -> None: self.model.params.recovery_position_floor = float(value)
    def _set_t1_rate(self, value: float) -> None: self.model.params.t1_rate = float(value)
    def _set_t1_p_eq(self, value: float) -> None: self.model.params.t1_p_eq = float(value)
    def _set_p_dnp_sat(self, value: float) -> None: self.model.params.p_dnp_sat = float(value)
    def _set_dnp_rate(self, value: float) -> None: self.model.params.dnp_rate = float(value)
    def _set_dt(self, value: float) -> None: self.model.params.dt = float(value)
    def _set_steps(self, value: int) -> None: self.steps_per_tick = int(value)
    def _set_noise(self, value: float) -> None: self.model.params.noise_sigma = float(value)

    def _set_p0_reset(self, value: float) -> None:
        self.model.params.p0 = float(value)
        self._reset_model()

    def _set_line_gamma_reset(self, value: float) -> None:
        self.model.params.line_gamma = float(value)
        self._reset_model()

    def _set_line_asym_reset(self, value: float) -> None:
        self.model.params.line_asym = float(value)
        self._reset_model()

    def _set_display_scale_reset(self, value: float) -> None:
        self.model.params.display_scale = float(value)
        self._reset_model()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.setText("Run simulation" if self.paused else "Pause simulation")

    def _reset_model(self) -> None:
        self.model.reset()
        self.model.set_rf_enabled(self.rf_is_on())
        self.model.set_dnp_enabled(self.dnp_is_on())
        self._start_new_trace(record_now=True)
        self._update_plots()

    def _start_new_trace(self, record_now: bool = False) -> None:
        self.trace_t0 = self.model.t
        self.trace_t.clear()
        self.trace_Ip_R.clear()
        self.trace_Im_R.clear()
        if record_now:
            self._record_trace_point()

    def _tick(self) -> None:
        if not self.paused:
            self.model.set_rf_enabled(self.rf_is_on())
            self.model.set_dnp_enabled(self.dnp_is_on())
            self.model.step(n_steps=self.steps_per_tick)
            self._record_trace_point()
        self._update_plots()

    def _record_trace_point(self) -> None:
        vals = self.model.local_intensities(self.model.params.rf_burn_R)
        self.trace_t.append(float(self.model.t - self.trace_t0))
        self.trace_Ip_R.append(vals["Iplus"])
        self.trace_Im_R.append(vals["Iminus"])
        if len(self.trace_t) > self.trace_max_points:
            del self.trace_t[:-self.trace_max_points]
            del self.trace_Ip_R[:-self.trace_max_points]
            del self.trace_Im_R[:-self.trace_max_points]

    def _finite_values(self, *series: list[float]) -> np.ndarray:
        vals: list[float] = []
        for s in series:
            vals.extend(float(x) for x in s if np.isfinite(x))
        return np.array(vals, dtype=float)

    def _set_marker(self, artist, x: float, y: float) -> None:
        artist.set_data([x], [y]) if np.isfinite(y) else artist.set_data([], [])

    def _set_bar(self, artist, x: float, y0: float, y1: float) -> None:
        artist.set_data([x, x], [y0, y1]) if np.isfinite(y0) and np.isfinite(y1) and abs(y1 - y0) > 1e-18 else artist.set_data([], [])

    def _update_plots(self) -> None:
        Rp, Ip_step, Rm, Im_step = self.model.packet_spectrum(noise_sigma=self.model.params.noise_sigma)
        R, _Ip, _Im, total = self.model.spectrum(noise_sigma=self.model.params.noise_sigma)
        Rb = self.model.params.rf_burn_R
        vals = self.model.response_values(Rb)
        self.line_Ip.set_data(Rp, Ip_step)
        self.line_Im.set_data(Rm, Im_step)
        self.line_total.set_data(R, total)
        self.burn_line.set_xdata([Rb, Rb])
        self.mirror_line.set_xdata([-Rb, -Rb])
        self._set_marker(self.point_Ip, Rb, vals["Iplus_R"])
        self._set_marker(self.point_Im, Rb, vals["Iminus_R"])
        self._set_marker(self.point_Ip_m, -Rb, vals["Iplus_minusR"])
        self._set_marker(self.point_Im_m, -Rb, vals["Iminus_minusR"])
        self._set_bar(self.bar_Ip_R, Rb, vals["Iplus_R_ref"], vals["Iplus_R"])
        self._set_bar(self.bar_Im_R, Rb, vals["Iminus_R_ref"], vals["Iminus_R"])
        self._set_bar(self.bar_Ip_mR, -Rb, vals["Iplus_minusR_ref"], vals["Iplus_minusR"])
        self._set_bar(self.bar_Im_mR, -Rb, vals["Iminus_minusR_ref"], vals["Iminus_minusR"])
        self.ax_spec.relim(); self.ax_spec.autoscale_view(); self.ax_spec.set_xlim(-3.05, 3.05)
        self.trace_line_Ip_R.set_data(self.trace_t, self.trace_Ip_R)
        self.trace_line_Im_R.set_data(self.trace_t, self.trace_Im_R)
        finite_y = self._finite_values(self.trace_Ip_R, self.trace_Im_R)
        if finite_y.size:
            y_min, y_max = float(np.min(finite_y)), float(np.max(finite_y))
            span = max(y_max - y_min, 0.03 * max(abs(y_max), abs(y_min), 1e-12))
            self.ax_trace.set_ylim(y_min - 0.12 * span, y_max + 0.12 * span)
        if self.trace_t:
            self.ax_trace.set_xlim(0.0, max(0.5, self.trace_t[-1] + 0.05))
        self.ax_trace.set_title(f"Burn-location intensities from selected initial values   [R={Rb:.3f}, RF {'ON' if self.rf_is_on() else 'OFF'}, DNP {'ON' if self.dnp_is_on() else 'OFF'}]")
        self._update_info_label()
        self.canvas.draw_idle()

    def _update_live_rate_readouts(self) -> None:
        rates = self.model.local_recovery_rates()

        def fmt(x: float) -> str:
            return "n/a" if x is None or not np.isfinite(x) else f"{x:.3e}"

        plus_l = rates["Iplus_neighbor_left"]
        plus_r = rates["Iplus_neighbor_right"]
        minus_l = rates["Iminus_neighbor_left"]
        minus_r = rates["Iminus_neighbor_right"]
        values = {
            "same_plus": fmt(rates["Iplus_same_theta"]),
            "same_minus": fmt(rates["Iminus_same_theta"]),
            "neigh_plus": f"{fmt(plus_l)} / {fmt(plus_r)}",
            "neigh_minus": f"{fmt(minus_l)} / {fmt(minus_r)}",
            "refill_plus": fmt(rates["Iplus_refill_dt_no_rf"]),
            "refill_minus": fmt(rates["Iminus_refill_dt_no_rf"]),
        }
        for key, value in values.items():
            if hasattr(self, "rate_readouts") and key in self.rate_readouts:
                self.rate_readouts[key].setText(value)

    def _update_info_label(self) -> None:
        self._update_live_rate_readouts()
        vals = self.model.response_values()
        pol = self.model.polarizations()
        areas = self.model.branch_areas()
        rates = self.model.local_recovery_rates()

        def fmt(x: float) -> str:
            return "n/a" if x is None or not np.isfinite(x) else f"{x:.3e}"

        self.p_readout.setText(f"P(t) = {pol['P']:+.5f}   ({100.0 * pol['P']:+.2f}%)")
        self.q_readout.setText(f"Q(t)={pol['Q']:+.5f}   Q_B[P]={pol['Q_boltz_at_P']:+.5f}   area∝{areas['A_total']:+.3e}")
        txt = (
            f"t={self.model.t:.3f}   RF {'ON' if self.rf_is_on() else 'OFF'}   DNP {'ON' if self.dnp_is_on() else 'OFF'}\n"
            f"R={self.model.params.rf_burn_R:.3f}   Γ_RF={self.model.params.gamma_rf:.3e}\n"
            f"I+(R)={fmt(vals['Iplus_R'])}, Δ={fmt(vals['dIplus_R'])}\n"
            f"I-(R)={fmt(vals['Iminus_R'])}, Δ={fmt(vals['dIminus_R'])}\n"
            f"mirror ΔI+={fmt(vals['dIplus_minusR'])}, ΔI-={fmt(vals['dIminus_minusR'])}\n"
            f"deviation factors I+/I-={fmt(rates['Iplus_deviation'])}/{fmt(rates['Iminus_deviation'])}"
        )
        self.info_label.setText(txt)


def main(argv: Optional[list[str]] = None) -> int:
    app = QtWidgets.QApplication(sys.argv if argv is None else argv)
    win = Spin1RealtimeWindow()
    win.resize(960, 500)
    win.show()
    return int(app.exec() if hasattr(app, "exec") else app.exec_())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
