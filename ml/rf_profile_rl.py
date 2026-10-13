"""One-shot SAC: infer an ideal-bin RF PulseProgram from a spin-1 lineshape."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.rf.profile_control import (
    DEFAULT_N_BINS,
    DEFAULT_P0,
    RFProfileEnv,
    q_shaped_manual_rates,
    rates_and_duration_to_program,
    play_program,
)

OUTPUT_DIR = REPO_ROOT / "results" / "current" / "rf_profile_rl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity=20000):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=DEVICE),
            torch.as_tensor(self.actions[idx], device=DEVICE),
            torch.as_tensor(self.rewards[idx], device=DEVICE),
        )


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        action = torch.sigmoid(u)
        log_prob = dist.log_prob(u) - F.softplus(-u) - F.softplus(u)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def act_deterministic(self, obs):
        mean, _ = self.forward(obs)
        return torch.sigmoid(mean)


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class SACAgent:
    def __init__(self, obs_dim, action_dim, *, lr=3e-4, gamma=0.0, tau=0.005, alpha=0.05):
        self.actor = Actor(obs_dim, action_dim).to(DEVICE)
        self.q1 = Critic(obs_dim, action_dim).to(DEVICE)
        self.q2 = Critic(obs_dim, action_dim).to(DEVICE)
        self.q1_t = Critic(obs_dim, action_dim).to(DEVICE)
        self.q2_t = Critic(obs_dim, action_dim).to(DEVICE)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha = float(alpha)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        x = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if deterministic:
            action = self.actor.act_deterministic(x)
        else:
            action, _ = self.actor.sample(x)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch):
        obs, actions, rewards = batch
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(obs)
            q_t = torch.min(self.q1_t(obs, next_action), self.q2_t(obs, next_action))
            target = rewards + self.gamma * (q_t - self.alpha * next_logp)
        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), 1.0
        )
        self.q_opt.step()

        new_action, logp = self.actor.sample(obs)
        q_pi = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = (self.alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        with torch.no_grad():
            for src, dst in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(src.parameters(), dst.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return float(q_loss.item()), float(actor_loss.item())


def sample_polarization(p_min, p_max, fallback):
    if p_min is None or p_max is None or p_min == p_max:
        return float(fallback)
    lo, hi = (float(p_min), float(p_max)) if p_min < p_max else (float(p_max), float(p_min))
    return float(np.random.uniform(lo, hi))


def run_baseline(env, kind, polarization):
    env.reset(polarization)
    ip, im, _ = env.model.physical_intensities()
    if kind == "zero":
        action = np.zeros(env.action_dim, dtype=float)
        action[-1] = 0.0
        _, reward, _, info = env.step(action)
        return reward, info, env.last_program
    if kind == "random":
        action = np.random.rand(env.action_dim)
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


def plot_training(rewards, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, color="tab:blue")
    ax.set_xlabel("episode")
    ax.set_ylabel(r"reward $\Delta Q$")
    ax.set_title("RF-profile SAC")
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profile(env, program, path, title, before=None, after=None):
    f = env.model.Rplus
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
    axes[1].stem(f, rates, linefmt="C2-", markerfmt="C2o", basefmt="k-")
    axes[1].set_ylabel(r"$U(R)$")
    axes[1].set_xlabel("R")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(args):
    env = RFProfileEnv(
        n_bins=args.n_bins,
        u_max=args.u_max,
        polarization=args.polarization,
    )
    agent = SACAgent(env.observation_dim, env.action_dim, lr=args.lr, alpha=args.alpha)
    replay = ReplayBuffer(env.observation_dim, env.action_dim, capacity=args.replay)
    rewards = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        p0 = sample_polarization(args.p_min, args.p_max, args.polarization)
        obs = env.reset(p0)
        action = agent.act(obs, deterministic=False)
        _, reward, _, info = env.step(action)
        replay.add(obs, action, reward)
        rewards.append(reward)
        if replay.size >= args.batch and (ep + 1) % args.update_every == 0:
            for _ in range(args.updates):
                agent.update(replay.sample(args.batch))
        if (ep + 1) % max(1, args.episodes // 10) == 0 or ep == 0:
            print(
                f"episode {ep + 1}/{args.episodes}  P={p0:.3f}  "
                f"reward={reward:.5e}  Q={info['Q']:.5e}  empty={info['empty']}",
                flush=True,
            )

    plot_training(rewards, out_dir / "reward_curve.png")
    obs = env.reset(args.polarization)
    ip0, im0, _ = env.model.physical_intensities()
    action = agent.act(obs, deterministic=True)
    _, reward, _, info = env.step(action)
    ip1, im1, _ = env.model.physical_intensities()
    program = env.last_program
    program.save(out_dir / "learned_program.json")
    plot_profile(
        env,
        program,
        out_dir / f"learned_profile_P{args.polarization:.2f}.png",
        f"SAC profile  P={args.polarization:.3f}  ΔQ={reward:.4e}  Q={info['Q']:.4e}",
        before=(ip0, im0),
        after=(ip1, im1),
    )

    print("Baselines at the same polarization:", flush=True)
    for kind in ("zero", "q_shaped", "random"):
        b_reward, b_info, _ = run_baseline(
            RFProfileEnv(n_bins=args.n_bins, u_max=args.u_max, polarization=args.polarization),
            kind,
            args.polarization,
        )
        print(
            f"  {kind:9s}  reward={b_reward:.5e}  Q={b_info['Q']:.5e}",
            flush=True,
        )
    print(f"  sac       reward={reward:.5e}  Q={info['Q']:.5e}", flush=True)
    print(f"Saved artifacts to {out_dir}", flush=True)
    return rewards, program


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="One-shot SAC for ideal-bin RF profiles")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--polarization", type=float, default=DEFAULT_P0)
    p.add_argument("--p-min", type=float, default=None)
    p.add_argument("--p-max", type=float, default=None)
    p.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    p.add_argument("--u-max", type=float, default=20.0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--replay", type=int, default=8000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--updates", type=int, default=4)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
