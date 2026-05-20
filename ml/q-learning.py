"""
Tabular Q-learning for sequential ssRF burns on a tensor lineshape.

Maximizes integrated Q polarization  Q = sum(I_+ - I_-)  by choosing burn center x0
and amplitude (shared Voigt gamma, sigma per burn). Actions are discretized;
state is (Q-bin, burn-step).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssRFMapper import ssRFMapper


def q_polarization(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


@dataclass
class BurnConfig:
    num_bins: int = 249
    f_min: float = -3.0
    f_max: float = 3.0
    sigma: float = 0.16
    gamma: float = 0.05
    # Burn center avoids line center (mirrored burn is applied in mapper).
    x0_values: tuple[float, ...] = ()
    amp_min: float = 5e-5
    amp_max: float = 5e-3
    n_amp_bins: int = 1
    max_burns: int = 2
    n_q_bins: int = 64
    lookup_path: Path | None = None

    def __post_init__(self) -> None:
        self.x0_values = tuple(np.linspace(-2, 2, self.num_bins))
        # if not self.x0_values:
        #     neg = np.linspace(-2, -0.05, 36)
        #     pos = np.linspace(0.05, 2, 36)
        #     self.x0_values = tuple(float(x) for x in np.concatenate([neg, pos]))
        if self.lookup_path is None:
            self.lookup_path = REPO_ROOT / "Data_Creation" / "lookup_table.pkl"

    @property
    def f(self) -> np.ndarray:
        return np.linspace(self.f_min, self.f_max, self.num_bins)

    @property
    def amp_values(self) -> np.ndarray:
        return np.linspace(self.amp_min, self.amp_max, self.n_amp_bins)

    @property
    def n_actions(self) -> int:
        return len(self.x0_values) * self.n_amp_bins

    @property
    def n_states(self) -> int:
        return self.n_q_bins * self.max_burns


class TensorBurnEnv:

    def __init__(self, config: BurnConfig, mapper: ssRFMapper):
        self.config = config
        self.mapper = mapper
        self.f = config.f
        self._q_lo: float = 0.0
        self._q_hi: float = 1.0
        self.reset(polarization=0.5)

    def _action_to_burn(self, action: int) -> tuple[float, float]:
        n_x0 = len(self.config.x0_values)
        x0_idx = action // self.config.n_amp_bins
        amp_idx = action % self.config.n_amp_bins
        return self.config.x0_values[x0_idx], float(self.config.amp_values[amp_idx])

    def _q_to_bin(self, q: float) -> int:
        if not np.isfinite(q):
            return 0
        span = self._q_hi - self._q_lo
        if span <= 0:
            return 0
        idx = int((q - self._q_lo) / span * self.config.n_q_bins)
        return int(np.clip(idx, 0, self.config.n_q_bins - 1))

    def _state_index(self, q_bin: int, step: int) -> int:
        step = int(np.clip(step, 0, self.config.max_burns - 1))
        return q_bin + step * self.config.n_q_bins

    def state_index(self) -> int:
        return self._state_index(self._q_bin, self._step)

    def set_q_bounds(self, q_lo: float, q_hi: float) -> None:
        self._q_lo = q_lo
        self._q_hi = q_hi

    def reset(self, polarization: float) -> int:
        self._polarization = float(polarization)
        _, iplus, iminus = GenerateVectorLineshape(self._polarization, self.f)
        self._iplus = np.asarray(iplus, dtype=float)
        self._iminus = np.asarray(iminus, dtype=float)
        self._ps = self._iplus + self._iminus
        self._q0 = q_polarization(self._iplus, self._iminus)
        self._q = self._q0
        self._step = 0
        self._q_bin = self._q_to_bin(self._q)
        return self.state_index()

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        if self._step >= self.config.max_burns:
            return self.state_index(), 0.0, True, {"noop": True}

        q_before = self._q
        x0, amp = self._action_to_burn(action)
        ps = self._ps.copy()
        iplus = self._iplus.copy()
        iminus = self._iminus.copy()

        self.mapper.x0 = x0
        self.mapper.amp = amp
        self.mapper.apply_ssRF(ps, iplus, iminus)

        self._ps = ps
        self._iplus = iplus
        self._iminus = iminus
        self._q = q_polarization(iplus, iminus)
        reward = self._q - q_before

        self._step += 1
        self._q_bin = self._q_to_bin(self._q)
        done = self._step >= self.config.max_burns
        info = {
            "x0": x0,
            "amp": amp,
            "q": self._q,
            "q_gain": self._q - self._q0,
        }
        return self.state_index(), reward, done, info

    @property
    def current_q(self) -> float:
        return self._q

    @property
    def initial_q(self) -> float:
        return self._q0


def load_mapper(config: BurnConfig) -> ssRFMapper:
    lookup_path = Path(config.lookup_path)
    if not lookup_path.is_file():
        raise FileNotFoundError(
            f"Lookup table not found at {lookup_path}. "
            "Run Data_Creation/lookup_table.py first."
        )
    mapping_data = pd.read_pickle(lookup_path)
    mapper = ssRFMapper(config.f, config.sigma, config.gamma, x0=0.0, amp=1e-3)
    mapper.compute_lookup_tables(mapping_data)
    return mapper


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1, ### learning rate
        gamma: float = 0.95, ### discount factor
        epsilon: float = 1.0, ### exploration probability
        seed: int | None = 42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions)) ### chose a random action
        return int(np.argmax(self.q_table[state])) ### chose the action with the highest Q-value

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        best_next = 0.0 if done else np.max(self.q_table[next_state]) ### the best next action

        self.q_table[state, action] += self.alpha * (reward + self.gamma * best_next - self.q_table[state, action])

    def decay_epsilon(self, factor: float = 0.995, min_eps: float = 0.05) -> None:
        self.epsilon = max(min_eps, self.epsilon * factor)


def estimate_q_bounds(
    env: TensorBurnEnv,
    polarizations: np.ndarray,
    n_random_actions: int = 30,
    seed: int = 0,
) -> tuple[float, float]:
    """Sample random burn sequences to set Q discretization limits."""
    rng = np.random.default_rng(seed)
    qs: list[float] = []
    for p in polarizations:
        env.reset(p)
        qs.append(env.initial_q)
        for _ in range(env.config.max_burns):
            action = int(rng.integers(0, env.config.n_actions))
            _, _, done, info = env.step(action)
            qs.append(info["q"])
            if done:
                break
    margin = 0.05 * (max(qs) - min(qs) + 1e-9)
    return min(qs) - margin, max(qs) + margin


def train(
    config: BurnConfig,
    episodes: int = 5000,
    polarizations: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[QLearningAgent, TensorBurnEnv, dict]:

    mapper = load_mapper(config)
    env = TensorBurnEnv(config, mapper)
    q_lo, q_hi = estimate_q_bounds(env, polarizations, seed=seed)
    env.set_q_bounds(q_lo, q_hi)

    agent = QLearningAgent(
        config.n_states, config.n_actions, seed=seed
    )
    rng = np.random.default_rng(seed)
    history: list[float] = []

    for ep in tqdm.tqdm(range(episodes), desc="Training Q-learning agent"):
        p = float(rng.choice(polarizations))
        state = env.reset(p)
        ep_return = 0.0

        for _ in range(config.max_burns):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_return += reward
            state = next_state
            if done:
                break

        history.append(ep_return)
        if (ep + 1) % 100 == 0:
            agent.decay_epsilon() ### reduce exploration probability

    stats = {
        "q_lo": q_lo,
        "q_hi": q_hi,
        "final_q_gain": env.current_q - env.initial_q,
        "episode_returns": np.asarray(history),
    }

    return agent, env, stats


### -- Testing on selected polarization -- ###

def greedy_episode(
    env: TensorBurnEnv, 
    agent: QLearningAgent, 
    polarization: float
) -> dict:
    state = env.reset(polarization)
    iplus_unburned = env._iplus.copy()
    iminus_unburned = env._iminus.copy()
    trace: list[dict] = [{"step": 0, "q": env.initial_q, "action": None}]

    for step in range(env.config.max_burns):
        action = agent.select_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        trace.append(
            {
                "step": step + 1,
                "action": action,
                "x0": info["x0"],
                "amp": info["amp"],
                "reward": reward,
                "q": info["q"],
                "q_gain": info["q_gain"],
            }
        )
        state = next_state
        if done:
            break

    return {
        "polarization": polarization,
        "initial_q": env.initial_q,
        "final_q": env.current_q,
        "trace": trace,
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus": env._iplus.copy(),
        "iminus": env._iminus.copy(),
        "f": env.f.copy(),
    }


def plot_training_returns(returns: np.ndarray, output_path: Path) -> None:
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
    ax.set_ylabel("sum of Q rewards")
    ax.set_title("Q-learning: ssRF burn policy training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_greedy_burns(result: dict, output_path: Path) -> None:
    f = result["f"]
    iplus = result["iplus"]
    iminus = result["iminus"]
    iplus0 = result["iplus_unburned"]
    iminus0 = result["iminus_unburned"]
    q_profile = iplus - iminus
    q_profile0 = iplus0 - iminus0

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(
        f, iplus0 + iminus0, color="black", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$P_s$ (unburned)",
    )
    axes[0].plot(
        f, iplus0, color="tab:red", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$I_+$ (unburned)",
    )
    axes[0].plot(
        f, iminus0, color="tab:blue", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$I_-$ (unburned)",
    )
    axes[0].plot(f, iplus + iminus, label=r"$P_s = I_+ + I_-$", color="black")
    axes[0].plot(f, iplus, label=r"$I_+$", color="tab:red")
    axes[0].plot(f, iminus, label=r"$I_-$", color="tab:blue")
    for row in result["trace"][1:]:
        axes[0].axvline(row["x0"], color="green", alpha=0.35, linestyle=":")
        axes[0].axvline(-row["x0"], color="purple", alpha=0.25, linestyle=":")
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        f, q_profile0, color="tab:purple", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$Q$ (unburned)",
    )
    axes[1].plot(f, q_profile, color="tab:purple", label=r"$Q = I_+ - I_-$")
    axes[1].set_xlabel("frequency")
    axes[1].set_ylabel("Q profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    title = (
        f"P={result['polarization']:.3f}  "
        f"Q: {result['initial_q']:.4f} → {result['final_q']:.4f}  "
        f"(Δ={result['final_q'] - result['initial_q']:.4f})"
    )
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_q_table(q_table: np.ndarray, output_path: Path) -> None:
    q_min, q_max = float(np.min(q_table)), float(np.max(q_table))
    span = q_max - q_min
    normalized = (q_table - q_min) / span if span > 0 else np.zeros_like(q_table)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(normalized, cmap="Spectral", aspect="auto", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, label="normalized Q")
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f"{q_min:.3g}", f"{(q_min + q_max) / 2:.3g}", f"{q_max:.3g}"])
    ax.set_xlabel("action")
    ax.set_ylabel("state")
    ax.set_title("Q-table (min–max normalized)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-learning for ssRF tensor Q burns")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--polarization", type=float, default=0.45)
    parser.add_argument("--max-burns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "current" / "q_learning",
    )
    args = parser.parse_args()

    config = BurnConfig(max_burns=args.max_burns)
    polarizations = np.concatenate([np.linspace(-0.5, -0.05, 12), np.linspace(0.05, 0.5, 12)])
    polarizations = np.unique(polarizations)

    print("Training Q-learning agent for tensor Q polarization...")
    agent, env, stats = train(
        config, 
        episodes=args.episodes, 
        polarizations=polarizations, 
        seed=args.seed
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "q_table.npy", agent.q_table)
    plot_training_returns(stats["episode_returns"], out_dir / "training_returns.png")

    eval_p = args.polarization
    greedy = greedy_episode(env, agent, eval_p)
    plot_greedy_burns(greedy, out_dir / f"greedy_policy_P{eval_p:.2f}.png")

    print(f"Q bounds for discretization: [{stats['q_lo']:.4f}, {stats['q_hi']:.4f}]")
    print(f"Greedy policy at P={eval_p*100:.2f}%:")
    for row in greedy["trace"]:
        if row["action"] is None:
            print(f"  start: Q={row['q']*100:.5f}%")
        else:
            print(
                f"  burn {row['step']}: x0={row['x0']:.3f}, amp={row['amp']:.4e}, "
                f"reward={row['reward']:.5f}, Q={row['q']*100:.5f}%"
            )
    print(f"  total Q gain: {(greedy['final_q'] - greedy['initial_q'])*100:.5f}%")
    print(f"Saved artifacts to {out_dir}")

    plot_q_table(agent.q_table, out_dir / "q_table.png")


if __name__ == "__main__":
    main()
