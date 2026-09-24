"""
Double DQN for the ssRF-beta ideal-bin RF profile.

The environment is a 701-bin Boltzmann Pake doublet on R∈[-3,3]. Actions fill
a simultaneous PulseProgram (per-bin rate U_j ∈ [0, 20] and shared duration
T ∈ [0.01, 2]) which is replayed through IdealBinModel. Reward is the change in
population tensor polarization Q = n_+ - 2 n_0 + n_-, matching ssRF-beta
``polarizations()``.
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.rf.profile_control import ProfileBuildEnv, U_MAX, T_MIN, T_MAX, DEFAULT_N_BINS

OUTPUT_DIR = REPO_ROOT / "results" / "current" / "dqn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPISODES = 100
POLARIZATION = 0.45
MAX_BURNS = 200
HIDDEN_DIM = 256
REPLAY_SIZE = 10000
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
TARGET_UPDATE_INTERVAL = 100
LEARN_START = 256
MAX_GRAD_NORM = 1.0


@dataclass
class BurnConfig:
    num_bins: int = DEFAULT_N_BINS
    f_min: float = -3.0
    f_max: float = 3.0
    u_max: float = U_MAX
    n_u_bins: int = 11
    t_min: float = T_MIN
    t_max: float = T_MAX
    n_t_bins: int = 7
    duration: float = 1.0
    max_burns: int = MAX_BURNS
    enforce_full_spectrum: bool = False
    candidate_mode: str = "union"

    def make_env(self, polarization=POLARIZATION):
        return ProfileBuildEnv(
            n_bins=self.num_bins,
            r_min=self.f_min,
            r_max=self.f_max,
            u_max=self.u_max,
            n_u_bins=self.n_u_bins,
            t_min=self.t_min,
            t_max=self.t_max,
            n_t_bins=self.n_t_bins,
            duration=self.duration,
            max_steps=self.max_burns,
            candidate_mode=self.candidate_mode,
            polarization=polarization,
            enforce_full_spectrum=self.enforce_full_spectrum,
        )


class ReplayBuffer:
    """Fixed-capacity circular buffer of DQN transitions."""

    def __init__(self, capacity, state_dim, n_actions, seed=42):
        self.capacity = capacity
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.ones((self.capacity, self.n_actions), dtype=np.bool_)

    def add(self, state, action, reward, next_state, done, next_mask):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        if next_mask is None:
            self.next_masks[self.ptr] = True
        else:
            self.next_masks[self.ptr] = np.asarray(next_mask, dtype=bool)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
            "next_masks": self.next_masks[idx],
        }


class QNetwork(nn.Module):
    """MLP mapping lineshape observations to per-action Q-values."""

    def __init__(self, state_dim, n_actions, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """Double DQN agent for ssRF-beta ideal-bin RF profiles."""

    def __init__(
        self,
        state_dim,
        n_actions,
        *,
        hidden_dim=HIDDEN_DIM,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_END,
        buffer_size=REPLAY_SIZE,
        batch_size=BATCH_SIZE,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        learn_start=LEARN_START,
        device=None,
        seed=SEED,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.learn_start = learn_start
        self.device = device if device is not None else DEVICE
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(state_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.target = QNetwork(state_dim, n_actions, hidden_dim=hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, state_dim, n_actions, seed=seed)
        self.learn_steps = 0

    def set_epsilon(self, epsilon):
        self.epsilon = np.clip(epsilon, 0.0, 1.0)

    def select_action(self, state, explore=True, action_mask=None):
        mask = np.ones(self.n_actions, dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool)
        valid_actions = np.flatnonzero(mask)
        if valid_actions.size == 0:
            return 0
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state_t).squeeze(0).detach().cpu().numpy()
        q_values[~mask] = -np.inf
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done, next_mask):
        self.buffer.add(state, action, reward, next_state, done, next_mask)

    def learn(self):
        if self.buffer.size < max(self.learn_start, self.batch_size):
            return None
        batch = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)
        q_values = self.online(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_online = self.online(next_states)
            next_q_online = next_q_online.masked_fill(~next_masks, -1000000000.0)
            next_actions = torch.argmax(next_q_online, dim=1)
            next_q_target = self.target(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            no_valid = ~next_masks.any(dim=1)
            next_q = torch.where(no_valid, torch.zeros_like(next_q), next_q)
            td_target = rewards + self.gamma * (1.0 - dones) * next_q
        loss = F.smooth_l1_loss(q_sa, td_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()


def epsilon_schedule(episode, total_episodes, eps_start=EPSILON_START, eps_end=EPSILON_END):
    if total_episodes <= 1:
        return eps_end
    progress = episode / (total_episodes - 1)
    return eps_start + (eps_end - eps_start) * progress


def train(config, episodes=EPISODES, polarizations=None, seed=SEED, eps_start=EPSILON_START, eps_end=EPSILON_END, lr=LEARNING_RATE, device=None):
    if polarizations is None:
        polarizations = np.linspace(0.4, 0.5, 20)
    env = config.make_env(float(polarizations[0]))
    agent = DQNAgent(env.observation_dim, env.n_actions, lr=lr, seed=seed, device=device)
    rng = np.random.default_rng(seed)
    history = []
    losses = []
    for ep in tqdm.tqdm(range(episodes), desc="Training DQN agent"):
        agent.set_epsilon(epsilon_schedule(ep, episodes, eps_start=eps_start, eps_end=eps_end))
        p = rng.choice(polarizations)
        state = env.reset(p)
        ep_return = 0.0
        for _ in range(config.max_burns):
            action_mask = env.valid_action_mask()
            if not action_mask.any():
                break
            action = agent.select_action(state, explore=True, action_mask=action_mask)
            next_state, reward, done, _ = env.step(action)
            next_mask = None if done else env.valid_action_mask()
            agent.remember(state, action, reward, next_state, done, next_mask)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            ep_return += reward
            state = next_state
            if done:
                break
        history.append(ep_return)
    stats = {
        "p_lo": np.min(polarizations),
        "p_hi": np.max(polarizations),
        "final_q_gain": env.current_q - env.initial_q,
        "episode_returns": np.asarray(history),
        "losses": np.asarray(losses),
        "physics_model": "physics.rf.IdealBinModel",
        "n_actions": env.n_actions,
        "observation_dim": env.observation_dim,
    }
    return agent, env, stats


def greedy_episode(env, agent, polarization):
    state = env.reset(polarization)
    n_unburned = env._n0.copy()
    env.model.n = n_unburned
    iplus_unburned, iminus_unburned, _ = env.model.physical_intensities()
    env.model.n = n_unburned.copy()
    trace = [{"step": 0, "q": env.initial_q, "P": env.current_p, "action": None}]
    for step in range(env.max_steps):
        action_mask = env.valid_action_mask()
        if not action_mask.any():
            break
        action = agent.select_action(state, explore=False, action_mask=action_mask)
        next_state, reward, done, info = env.step(action)
        trace.append(
            {
                "step": step + 1,
                "action": action,
                "kind": info.get("kind"),
                "bin_idx": info.get("bin_idx", -1),
                "f": info.get("f", 0.0),
                "rate": info.get("rate", 0.0),
                "duration": info.get("duration", env.duration),
                "reward": reward,
                "q": info["q"],
                "P": info.get("P", env.current_p),
                "q_gain": info["q_gain"],
            }
        )
        state = next_state
        if done:
            break
    iplus, iminus, _ = env.model.physical_intensities()
    return {
        "polarization": polarization,
        "initial_q": env.initial_q,
        "final_q": env.current_q,
        "initial_p": env._p0,
        "final_p": env.current_p,
        "trace": trace,
        "iplus_unburned": np.asarray(iplus_unburned),
        "iminus_unburned": np.asarray(iminus_unburned),
        "iplus": np.asarray(iplus),
        "iminus": np.asarray(iminus),
        "rates": np.asarray(env.rates).copy(),
        "duration": float(env.duration),
        "f": np.asarray(env.f).copy(),
        "program": env.last_program,
    }


def plot_training_returns(returns, output_path):
    window = min(200, max(1, len(returns) // 20))
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(returns, alpha=0.25, linewidth=0.8, label="episode return")
    ax.plot(
        np.arange(window - 1, window - 1 + len(smoothed)),
        smoothed,
        color="C1",
        linewidth=2,
        label=f"{window}-ep moving avg",
    )
    ax.set_xlabel("episode")
    ax.set_ylabel(r"sum of $\Delta Q$ rewards")
    ax.set_title("DQN: ssRF-beta ideal-bin RF profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_losses(losses, output_path):
    if losses.size == 0:
        return
    window = min(200, max(1, len(losses) // 20))
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.25, linewidth=0.8, label="TD loss")
    ax.plot(
        np.arange(window - 1, window - 1 + len(smoothed)),
        smoothed,
        color="C1",
        linewidth=2,
        label=f"{window}-step moving avg",
    )
    ax.set_xlabel("gradient step")
    ax.set_ylabel("smooth L1 loss")
    ax.set_title("DQN temporal-difference loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_greedy_burns(result, output_path):
    f = result["f"]
    iplus = result["iplus"]
    iminus = result["iminus"]
    iplus0 = result["iplus_unburned"]
    iminus0 = result["iminus_unburned"]
    rates = result["rates"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].step(f, iplus0, color="tab:red", linestyle="--", alpha=0.55, label=r"$I_+$ before")
    axes[0].step(f, iminus0, color="tab:blue", linestyle="--", alpha=0.55, label=r"$I_-$ before")
    axes[0].step(f, iplus0 - iminus0, color="tab:green", linestyle="--", alpha=0.55, label=r"$I_+-I_-$ before")
    axes[0].step(f, iplus, color="tab:red", label=r"$I_+$ after")
    axes[0].step(f, iminus, color="tab:blue", label=r"$I_-$ after")
    axes[0].step(f, iplus - iminus, color="tab:green", label=r"$I_+-I_-$ after")
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[1].step(f, iplus0 - iminus0, color="tab:purple", linestyle="--", alpha=0.55, label="spectral Q before")
    axes[1].step(f, iplus - iminus, color="tab:purple", label="spectral Q after")
    axes[1].set_ylabel("spectral Q")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)
    axes[2].stem(f, rates, linefmt="C2-", markerfmt="C2o", basefmt="k-")
    axes[2].set_ylabel(r"$U(R)$")
    axes[2].set_xlabel("R")
    axes[2].grid(True, alpha=0.3)
    delta_q = result["final_q"] - result["initial_q"]
    title = (
        f"P={result['polarization']:.3f}  "
        f"Q={result['initial_q']:.4f}→{result['final_q']:.4f}  "
        f"ΔQ={delta_q:+.4f}  T={result['duration']:.3f}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Double DQN for ssRF-beta ideal-bin RF profiles")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--polarization", type=float, default=POLARIZATION)
    parser.add_argument("--max-burns", type=int, default=MAX_BURNS)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--u-max", type=float, default=U_MAX)
    parser.add_argument("--n-u-bins", type=int, default=11)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument(
        "--full-spectrum",
        action="store_true",
        help="Scan bins in R order; action chooses U only. Default is free-bin selection.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    config = BurnConfig(
        num_bins=args.n_bins,
        max_burns=args.max_burns,
        u_max=args.u_max,
        n_u_bins=args.n_u_bins,
        duration=args.duration,
        enforce_full_spectrum=args.full_spectrum,
    )
    polarizations = np.linspace(0.4, 0.5, 20)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_probe = config.make_env(args.polarization)
    print(
        f"Training Double DQN on ssRF-beta IdealBinModel "
        f"(n_bins={config.num_bins}, U_max={config.u_max}, T={config.duration}, "
        f"device={DEVICE}, n_actions={env_probe.n_actions}, candidate={config.candidate_mode})...",
        flush=True,
    )
    agent, env, stats = train(
        config,
        episodes=args.episodes,
        polarizations=polarizations,
        seed=args.seed,
        lr=args.lr,
        device=DEVICE,
    )
    ckpt_path = out_dir / "dqn_policy.pt"
    torch.save(
        {
            "online": agent.online.state_dict(),
            "target": agent.target.state_dict(),
            "observation_dim": env.observation_dim,
            "n_actions": env.n_actions,
            "hidden_dim": HIDDEN_DIM,
            "config": {
                "num_bins": config.num_bins,
                "max_burns": config.max_burns,
                "n_u_bins": config.n_u_bins,
                "u_max": config.u_max,
                "duration": config.duration,
                "enforce_full_spectrum": config.enforce_full_spectrum,
            },
            "stats": {
                "p_lo": stats["p_lo"],
                "p_hi": stats["p_hi"],
                "n_actions": stats["n_actions"],
                "observation_dim": stats["observation_dim"],
            },
        },
        ckpt_path,
    )
    np.save(out_dir / "episode_returns.npy", stats["episode_returns"])
    plot_training_returns(stats["episode_returns"], out_dir / "training_returns.png")
    plot_losses(stats["losses"], out_dir / "td_loss.png")
    eval_p = args.polarization
    greedy = greedy_episode(env, agent, eval_p)
    plot_greedy_burns(greedy, out_dir / f"greedy_policy_P{eval_p:.2f}.png")
    if greedy["program"] is not None:
        greedy["program"].save(out_dir / "learned_program.json")
    print(f"Greedy policy at P={eval_p * 100:.2f}%  (population Q, ssRF-beta):")
    for row in greedy["trace"]:
        if row["action"] is None:
            print(f"  start: Q={row['q']:.5f}  P={row['P']:.5f}")
        elif row.get("kind") == "duration":
            print(
                f"  T {row['step']}: duration={row['duration']:.4f}, "
                f"reward={row['reward']:.5f}, Q={row['q']:.5f}"
            )
        elif row.get("rate", 0.0) <= 0.0:
            print(
                f"  skip {row['step']}: bin={row['bin_idx']}, R={row['f']:.3f}, Q={row['q']:.5f}"
            )
        else:
            print(
                f"  U {row['step']}: bin={row['bin_idx']}, R={row['f']:.3f}, "
                f"U={row['rate']:.4e}, T={row['duration']:.4f}, "
                f"reward={row['reward']:.5f}, Q={row['q']:.5f}"
            )
    print(
        f"  total population ΔQ: {greedy['final_q'] - greedy['initial_q']:+.5f}  "
        f"P: {greedy['initial_p']:.5f}→{greedy['final_p']:.5f}  T={greedy['duration']:.4f}"
    )
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
