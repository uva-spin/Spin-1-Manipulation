"""Learn the theoretical ssRF power profile from a Ps spectrum.

Each training spectrum is labeled by ``design_optimal_profile`` (the same
L-BFGS search the optimal-profile events use). The actor is fit on those
spectra with Gaussian noise added to Ps, and the target stays the clean
optimal profile. Pass ``--f-min`` and ``--f-max`` for the lab frequency axis.
After training, ``--ps spectrum.npy`` writes the power profile.
"""
import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.rf.optimal_profile import DATA_GEN_SETTINGS, design_optimal_profile
from physics.rf.profile_control import (
    DEFAULT_P0,
    RFProfileEnv,
    decode_unit_action,
    play_program,
    q_shaped_manual_rates,
    rates_and_duration_to_program,
)

N_FREQ = 500
PHYSICS_R_MIN = -6.0
PHYSICS_R_MAX = 6.0
NOISE_STD = 0.02
NOISE_DRAWS = 8

OUTPUT_DIR = REPO_ROOT / "results" / "current" / "rf_profile_rl_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """Ps spectrum to a unit-interval power profile (rates, then duration)."""

    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def raw(self, obs):
        return self.net(obs)

    def forward(self, obs):
        return self.raw(obs).clamp(0.0, 1.0)


def ps_observation(env):
    """Total Ps the actor sees."""
    _ip, _im, ps = env.model.physical_intensities()
    return np.asarray(ps, dtype=np.float32)


def save_actor(actor, path, env):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs_dim": int(env.n_bins),
            "action_dim": int(env.action_dim),
            "u_max": float(env.u_max),
            "t_min": float(env.t_min),
            "t_max": float(env.t_max),
            "state_dict": actor.state_dict(),
        },
        path,
    )


def load_actor(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    actor = Actor(int(ckpt["obs_dim"]), int(ckpt["action_dim"])).to(DEVICE)
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor, ckpt


def load_ps(path, n_bins):
    path = Path(path)
    raw = np.load(path) if path.suffix == ".npy" else np.genfromtxt(path, delimiter=",")
    ps = np.asarray(raw, dtype=np.float64)
    if path.suffix != ".npy":
        ps = ps[np.isfinite(ps)]
    ps = ps.reshape(-1)
    if ps.size != int(n_bins):
        raise ValueError(f"Ps must have length {n_bins}, got {ps.size} from {path}")
    return ps.astype(np.float32)


def program_from_action(action, ckpt, args):
    rates, duration = decode_unit_action(
        action, u_max=ckpt["u_max"], t_min=ckpt["t_min"], t_max=ckpt["t_max"]
    )
    return rates_and_duration_to_program(
        rates,
        duration,
        r_min=args.r_min,
        r_max=args.r_max,
        name="RL profile",
    )


def teacher_settings(env):
    return replace(
        DATA_GEN_SETTINGS,
        max_power=float(env.u_max),
        min_duration=float(env.t_min),
        max_duration=float(env.t_max),
    )


def polarization_grid(args):
    if args.p_min is None or args.p_max is None or args.p_min == args.p_max:
        lo, hi = 0.2, 0.6
    elif args.p_min < args.p_max:
        lo, hi = float(args.p_min), float(args.p_max)
    else:
        lo, hi = float(args.p_max), float(args.p_min)
    grid = np.linspace(lo, hi, max(int(args.episodes), 1))
    grid = np.unique(np.round(np.concatenate([grid, [float(args.polarization)]]), 6))
    return grid


def action_from_design(design, env):
    u_norm = np.clip(np.asarray(design.rates, dtype=np.float64) / float(env.u_max), 0.0, 1.0)
    if float(design.duration) <= 0.0 or not np.any(u_norm > 0.0):
        return np.zeros(env.action_dim, dtype=np.float32)
    span = float(env.t_max) - float(env.t_min)
    t_norm = float(np.clip((float(design.duration) - float(env.t_min)) / span, 0.0, 1.0))
    return np.concatenate([u_norm, [t_norm]]).astype(np.float32)


def collect_targets(env, polarizations, settings):
    obs, actions, teacher_dq = [], [], []
    for i, p0 in enumerate(polarizations):
        env.reset(float(p0))
        obs.append(ps_observation(env))
        design = design_optimal_profile(env.model, settings=settings)
        actions.append(action_from_design(design, env))
        dq = float(design.q_final - design.q_initial)
        teacher_dq.append(dq)
        print(
            f"optimal {i + 1}/{len(polarizations)}  P={float(p0):.3f}  "
            f"dQ={dq:.5e}  T={design.duration:.4g}  status={design.status}",
            flush=True,
        )
    return obs, actions, teacher_dq


def fit_actor(actor, obs, actions, *, lr, steps, noise_std):
    """Fit clean and noisy Ps to the same optimal action."""
    x_clean = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=DEVICE)
    y = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    best_err = float("inf")
    best_state = None
    std = float(noise_std)
    for step in range(int(steps)):
        pred = actor.raw(x_clean)
        loss = F.mse_loss(pred, y)
        if std > 0.0:
            noise = torch.randn(NOISE_DRAWS, *x_clean.shape, device=DEVICE) * std
            pred_n = actor.raw((x_clean.unsqueeze(0) + noise).reshape(-1, x_clean.shape[-1]))
            y_n = y.unsqueeze(0).expand(NOISE_DRAWS, -1, -1).reshape(-1, y.shape[-1])
            loss = loss + F.mse_loss(pred_n, y_n)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 != 0 and step + 1 != int(steps):
            continue
        with torch.no_grad():
            err = float((actor.forward(x_clean) - y).abs().max())
        if err < best_err:
            best_err = err
            best_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
        if best_err <= 1e-4:
            break
    if best_state is not None:
        actor.load_state_dict(best_state)
    return best_err


def predict_action(actor, ps):
    with torch.no_grad():
        x = torch.as_tensor(ps, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        return actor(x).squeeze(0).cpu().numpy()


def play_action(env, action, polarization):
    env.reset(float(polarization))
    _, reward, _, info = env.step(action)
    return reward, info, env.last_program


def median_noisy_reward(actor, env, ps, polarization, noise_std, draws=NOISE_DRAWS, seed=0):
    """ΔQ when the measured Ps is noisy and the spin state is not."""
    rng = np.random.default_rng(seed)
    clean = np.asarray(ps, dtype=np.float32)
    rewards = []
    for _ in range(int(draws)):
        noisy = clean + rng.normal(0.0, float(noise_std), size=clean.shape).astype(np.float32)
        reward, _, _ = play_action(env, predict_action(actor, noisy), polarization)
        rewards.append(float(reward))
    return float(np.median(rewards))


def run_baseline(env, kind, polarization):
    env.reset(polarization)
    ip, im, _ = env.model.physical_intensities()
    if kind == "zero":
        action = np.zeros(env.action_dim, dtype=float)
        _, reward, _, info = env.step(action)
        return reward, info, env.last_program
    if kind == "q_shaped":
        rates = q_shaped_manual_rates(ip, im, gamma_rf=2.0)
        duration = 1.0
        program = rates_and_duration_to_program(
            rates,
            duration,
            r_min=env.r_min,
            r_max=env.r_max,
            mask=env.mask,
            name="Q-shaped manual",
        )
        delta_q, q_final, p_final, empty = play_program(env.model, program)
        info = {"empty": empty, "Q": q_final, "P": p_final, "delta_Q": 0.0 if empty else float(delta_q)}
        return (0.0 if empty else float(delta_q)), info, program
    raise ValueError(kind)


def plot_training(polarizations, optimal_dq, actor_dq, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(polarizations, optimal_dq, color="tab:orange", marker="o", label="optimal profile")
    ax.plot(polarizations, actor_dq, color="tab:blue", marker=".", label="actor")
    ax.set_xlabel("polarization")
    ax.set_ylabel(r"reward $\Delta Q$")
    ax.set_title("Actor vs theoretical optimal profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def power_profile_arrays(program, f_min, f_max):
    """Dense amplitude at each lab frequency. Length is the physics bin count."""
    amplitude = np.asarray(program.compile().envelope, dtype=float)
    frequency = np.linspace(float(f_min), float(f_max), amplitude.size)
    duration = 0.0
    if program.profiles and program.profiles[0].pulses:
        duration = float(program.profiles[0].pulses[0].duration)
    return frequency, amplitude, duration


def scpi_power_profile(frequency, amplitude, duration):
    """LIST sweep: one frequency, one amplitude, one dwell per point.

    Amplitudes are the learned rates, not calibrated dBm.
    """
    freq = ",".join(format(float(v), ".8g") for v in frequency)
    amp = ",".join(format(float(v), ".8g") for v in amplitude)
    dwell = ",".join(format(float(duration), ".8g") for _ in range(frequency.size))
    return (
        "SOUR:FREQ:MODE LIST\n"
        f"SOUR:LIST:FREQ {freq}\n"
        f"SOUR:LIST:POW {amp}\n"
        f"SOUR:LIST:DWEL {dwell}\n"
    )


def save_power_profile(directory, program, f_min, f_max):
    frequency, amplitude, duration = power_profile_arrays(program, f_min, f_max)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "power_profile.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency", "amplitude"])
        writer.writerows(zip(frequency.tolist(), amplitude.tolist()))
    (directory / "power_profile.scpi").write_text(
        scpi_power_profile(frequency, amplitude, duration),
        encoding="utf-8",
    )
    return frequency, amplitude, duration


def rates_from_action(action, env):
    rates, _duration = decode_unit_action(
        action, u_max=env.u_max, t_min=env.t_min, t_max=env.t_max
    )
    return np.asarray(rates, dtype=float)


def plot_noisy_case(frequency, clean_ps, noisy_ps, optimal, inferred, path, title):
    """Noisy Ps on top, inferred power over the clean-spectrum optimum below."""
    f = np.asarray(frequency, dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(f, clean_ps, color="0.25", lw=1.4, label="clean Ps")
    axes[0].plot(f, noisy_ps, color="tab:red", lw=0.8, alpha=0.85, label="noisy Ps")
    axes[0].set_ylabel("Ps")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].step(f, optimal, where="mid", color="tab:orange", lw=2.0, label="optimal, clean Ps")
    axes[1].step(f, inferred, where="mid", color="tab:blue", lw=1.2, label="inferred, noisy Ps")
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("frequency")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_noisy_examples(actor, env, polarizations, obs, actions, args, out_dir, n=4):
    """A few noisy spectra and the profile inferred from each, over the clean optimum."""
    if float(args.noise_std) <= 0.0:
        return []
    polarizations = np.asarray(polarizations, dtype=float)
    count = min(int(n), int(polarizations.size))
    rows = np.unique(np.rint(np.linspace(0, polarizations.size - 1, count)).astype(int))
    match = int(np.argmin(np.abs(polarizations - float(args.polarization))))
    if match not in rows:
        rows = np.unique(np.append(rows, match))
    frequency = np.linspace(float(args.f_min), float(args.f_max), env.n_bins)
    rng = np.random.default_rng(0)
    paths = []
    for i in rows:
        clean = np.asarray(obs[int(i)], dtype=np.float64)
        noisy = clean + rng.normal(0.0, float(args.noise_std), size=clean.shape)
        inferred = rates_from_action(predict_action(actor, noisy.astype(np.float32)), env)
        optimal = rates_from_action(actions[int(i)], env)
        p0 = float(polarizations[int(i)])
        path = out_dir / f"noisy_profile_P{p0:.3f}.png"
        plot_noisy_case(
            frequency,
            clean,
            noisy,
            optimal,
            inferred,
            path,
            f"P={p0:.3f}  noise std={float(args.noise_std):.3g}",
        )
        paths.append(path)
    return paths


def plot_profile(env, program, path, title, frequency, before=None, after=None):
    f = np.asarray(frequency, dtype=float)
    if after is None:
        ip1, im1, _ = env.model.physical_intensities()
    else:
        ip1, im1 = after
    rates = program.compile().envelope
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    if before is not None:
        ip0, im0 = before
        axes[0].step(f, ip0, color="tab:red", linestyle="--", alpha=0.6, label=r"$I_+$ before")
        axes[0].step(f, im0, color="tab:blue", linestyle="--", alpha=0.6, label=r"$I_-$ before")
        axes[0].step(f, ip0 - im0, color="tab:green", linestyle="--", alpha=0.6, label=r"$Q$ before")
    axes[0].step(f, ip1, color="tab:red", label=r"$I_+$ after")
    axes[0].step(f, im1, color="tab:blue", label=r"$I_-$ after")
    axes[0].step(f, ip1 - im1, color="tab:green", label=r"$Q$ after")
    axes[0].set_ylabel("intensity")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[1].step(f, rates, color="C2", where="mid")
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("frequency")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_env(args, polarization):
    return RFProfileEnv(
        n_bins=args.n_bins,
        r_min=args.r_min,
        r_max=args.r_max,
        u_max=args.u_max,
        polarization=polarization,
    )


def train(args):
    env = make_env(args, args.polarization)
    polarizations = polarization_grid(args)
    obs, actions, teacher_dq = collect_targets(env, polarizations, teacher_settings(env))
    actor = Actor(env.n_bins, env.action_dim).to(DEVICE)
    if args.noise_std < 0.0:
        raise ValueError("Require --noise-std >= 0")
    fit_err = fit_actor(
        actor, obs, actions, lr=args.lr, steps=args.updates, noise_std=args.noise_std
    )
    print(
        f"fit max |action error| = {fit_err:.3e}  noise std = {args.noise_std:.4g}",
        flush=True,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_actor(actor, out_dir / "actor.pt", env)

    actor_dq = []
    for p0, ps in zip(polarizations, obs):
        reward, _, _ = play_action(env, predict_action(actor, ps), p0)
        actor_dq.append(float(reward))
    plot_training(polarizations, teacher_dq, actor_dq, out_dir / "reward_curve.png")

    match = int(np.argmin(np.abs(polarizations - float(args.polarization))))
    env.reset(args.polarization)
    ip0, im0, _ = env.model.physical_intensities()
    ps = ps_observation(env)
    _, reward, _, info = env.step(predict_action(actor, ps))
    ip1, im1, _ = env.model.physical_intensities()
    program = env.last_program
    teacher_reward, teacher_info, _ = play_action(env, actions[match], args.polarization)
    shortfall = float(teacher_reward) - float(reward)
    if shortfall > max(1e-3, 0.02 * abs(float(teacher_reward))):
        raise RuntimeError(
            f"Actor ΔQ {reward:.6e} is short of the optimal profile {teacher_reward:.6e}"
        )
    noisy_reward = None
    if args.noise_std > 0.0:
        noisy_reward = median_noisy_reward(
            actor, env, ps, args.polarization, args.noise_std
        )
        noisy_shortfall = float(teacher_reward) - float(noisy_reward)
        if noisy_shortfall > max(1e-3, 0.02 * abs(float(teacher_reward))):
            raise RuntimeError(
                f"Noisy Ps ΔQ {noisy_reward:.6e} is short of the optimal profile {teacher_reward:.6e}"
            )
    program.save(out_dir / "learned_program.json")
    frequency, amplitude, _duration = save_power_profile(
        out_dir, program, args.f_min, args.f_max
    )
    plot_profile(
        env,
        program,
        out_dir / f"learned_profile_P{args.polarization:.2f}.png",
        f"Optimal profile  P={args.polarization:.3f}  ΔQ={reward:.4e}  Q={info['Q']:.4e}",
        frequency,
        before=(ip0, im0),
        after=(ip1, im1),
    )
    noisy_paths = save_noisy_examples(
        actor, env, polarizations, obs, actions, args, out_dir
    )
    for path in noisy_paths:
        print(f"Saved {path}", flush=True)

    print("Baselines at the same polarization:", flush=True)
    for kind in ("zero", "q_shaped"):
        b_reward, b_info, _ = run_baseline(
            make_env(args, args.polarization),
            kind,
            args.polarization,
        )
        print(
            f"  {kind:9s}  reward={b_reward:.5e}  Q={b_info['Q']:.5e}",
            flush=True,
        )
    print(
        f"  optimal   reward={teacher_reward:.5e}  Q={teacher_info['Q']:.5e}",
        flush=True,
    )
    print(f"  actor     reward={reward:.5e}  Q={info['Q']:.5e}", flush=True)
    if noisy_reward is not None:
        print(
            f"  actor+noise reward={noisy_reward:.5e}  std={args.noise_std:.4g}",
            flush=True,
        )
    print(
        f"Power profile: {amplitude.size} points, "
        f"frequency [{frequency[0]:.6g}, {frequency[-1]:.6g}]",
        flush=True,
    )
    print(f"Saved artifacts to {out_dir}", flush=True)
    print(f"Infer: python ml/rf_profile_rl.py --ps spectrum.npy --actor {out_dir / 'actor.pt'}", flush=True)
    return actor_dq, program


def infer(args):
    actor_path = Path(args.actor) if args.actor else Path(args.out_dir) / "actor.pt"
    if not actor_path.is_file():
        raise FileNotFoundError(f"No actor checkpoint at {actor_path}. Train first.")
    actor, ckpt = load_actor(actor_path)
    ps = load_ps(args.ps, ckpt["obs_dim"])
    action = predict_action(actor, ps)
    program = program_from_action(action, ckpt, args)
    out_dir = Path(args.out_dir)
    frequency, amplitude, duration = save_power_profile(out_dir, program, args.f_min, args.f_max)
    program.save(out_dir / "inferred_program.json")
    print(
        f"Ps length {ps.size} -> {amplitude.size} amplitudes, "
        f"frequency [{frequency[0]:.6g}, {frequency[-1]:.6g}], dwell {duration:.4g}",
        flush=True,
    )
    print(f"Saved {out_dir / 'power_profile.csv'} and {out_dir / 'power_profile.scpi'}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fit an ssRF power profile to the theoretical optimum")
    p.add_argument("--episodes", type=int, default=9, help="Polarizations labeled by the optimal-profile solver")
    p.add_argument("--polarization", type=float, default=DEFAULT_P0)
    p.add_argument("--p-min", type=float, default=None)
    p.add_argument("--p-max", type=float, default=None)
    p.add_argument("--n-bins", type=int, default=N_FREQ)
    p.add_argument("--r-min", type=float, default=PHYSICS_R_MIN)
    p.add_argument("--r-max", type=float, default=PHYSICS_R_MAX)
    p.add_argument("--f-min", type=float, default=PHYSICS_R_MIN)
    p.add_argument("--f-max", type=float, default=PHYSICS_R_MAX)
    p.add_argument("--u-max", type=float, default=20.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--updates", type=int, default=8000, help="Adam steps fitting Ps to the optimal profile")
    p.add_argument("--noise-std", type=float, default=NOISE_STD, help="Gaussian std added to Ps during training")
    p.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--ps", type=str, default=None, help="1-D Ps spectrum (.npy or text). Skips training.")
    p.add_argument("--actor", type=str, default=None, help="Actor checkpoint. Default: <out-dir>/actor.pt")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.f_min < args.f_max:
        raise ValueError("Require --f-min < --f-max")
    if args.ps:
        infer(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
