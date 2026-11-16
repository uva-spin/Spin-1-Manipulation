"""LabVIEW entry: Ps spectrum to a generator list sweep.

Call ``InitModel(model_path)`` once per Python session, then

    main(ps, address, f_min, f_max)

``ps`` is the 500-point NMR spectrum. ``address`` is an IP, ``host:port``,
or a VISA ``TCPIP::host::...`` resource (raw SCPI port 5025 unless the
resource gives a numeric port). ``f_min`` and ``f_max`` are the generator
frequencies at the first and last point.

Returns ``(frequency, amplitude, duration)``. Amplitudes are the model
rates written to ``SOUR:LIST:POW``, not calibrated dBm.
"""
import socket
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from utils.constants import DEVICE

SCPI_PORT = 5025

actor = None
checkpoint = None
device = DEVICE


class Actor(nn.Module):
    """Same network as ``ml/rf_profile_rl.py``."""

    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, obs):
        return self.net(obs).clamp(0.0, 1.0)


def InitModel(model_path):
    """Load the power-profile actor once per LabVIEW Python session."""
    global actor, checkpoint, device
    device = DEVICE
    try:
        loaded = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(model_path, map_location=device)
    net = Actor(int(loaded["obs_dim"]), int(loaded["action_dim"])).to(device)
    net.load_state_dict(loaded["state_dict"])
    net.eval()
    actor = net
    checkpoint = loaded


def parse_address(address, default_port=SCPI_PORT):
    text = str(address).strip()
    if not text:
        raise ValueError("Generator address is empty")
    if text.upper().startswith("TCPIP"):
        parts = text.split("::")
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"Unrecognized VISA address {address!r}")
        port = default_port
        if len(parts) >= 3 and parts[2].isdigit():
            port = int(parts[2])
        return parts[1], port
    if text.count(":") == 1:
        host, port_text = text.rsplit(":", 1)
        if port_text.isdigit():
            return host, int(port_text)
    return text, default_port


def profile_from_ps(ps, f_min, f_max):
    if actor is None or checkpoint is None:
        raise RuntimeError("Call InitModel before main")
    spectrum = np.asarray(ps, dtype=np.float32).reshape(-1)
    n_bins = int(checkpoint["obs_dim"])
    if spectrum.size != n_bins:
        raise ValueError(f"Ps must have length {n_bins}, got {spectrum.size}")
    f0 = float(f_min)
    f1 = float(f_max)
    if not f0 < f1:
        raise ValueError("Require f_min < f_max")
    with torch.no_grad():
        x = torch.from_numpy(spectrum.reshape(1, -1)).to(device)
        action = actor(x).squeeze(0).detach().cpu().numpy()
    u_norm = np.clip(action[:-1], 0.0, 1.0)
    t_norm = float(np.clip(action[-1], 0.0, 1.0))
    amplitude = u_norm * float(checkpoint["u_max"])
    duration = float(checkpoint["t_min"]) + t_norm * (
        float(checkpoint["t_max"]) - float(checkpoint["t_min"])
    )
    frequency = np.linspace(f0, f1, amplitude.size)
    return frequency, amplitude, duration


def scpi_commands(frequency, amplitude, duration):
    freq = ",".join(format(float(v), ".8g") for v in frequency)
    amp = ",".join(format(float(v), ".8g") for v in amplitude)
    dwell = ",".join(format(float(duration), ".8g") for _ in range(frequency.size))
    return (
        "SOUR:FREQ:MODE LIST",
        f"SOUR:LIST:FREQ {freq}",
        f"SOUR:LIST:POW {amp}",
        f"SOUR:LIST:DWEL {dwell}",
        "*OPC?",
    )


def send_scpi(address, commands, timeout=10.0):
    host, port = parse_address(address)
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        reply = ""
        for command in commands:
            sock.sendall((command + "\n").encode("ascii"))
            if command.endswith("?"):
                reply = _read_line(sock)
        if reply.strip() != "1":
            raise RuntimeError(f"Generator at {host}:{port} rejected the list ({reply!r})")
    return host, port


def _read_line(sock):
    chunks = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    return b"".join(chunks).decode("ascii", errors="replace").strip()


def main(ps, address, f_min, f_max):
    """Infer the profile and send it to the generator.

    Returns ``(frequency, amplitude, duration)``.
    """
    frequency, amplitude, duration = profile_from_ps(ps, f_min, f_max)
    send_scpi(address, scpi_commands(frequency, amplitude, duration))
    return (
        np.ascontiguousarray(frequency, dtype=np.float64),
        np.ascontiguousarray(amplitude, dtype=np.float64),
        float(duration),
    )
