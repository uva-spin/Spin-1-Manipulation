"""
Tabular double Q-learning for sequential bin-wise ssRF burns on a tensor lineshape.

Maximizes integrated Q polarization  Q = sum(I_+ - I_-)  by choosing a frequency
bin and burn amplitude (per-bin Ps decrease, no Voigt profile). Actions are
discretized; state is (Q-bin, burn-step). Episodes sample a tensor polarization
P; lineshape is regenerated on reset(P). Each x-bin may be burned at most once
per episode.
"""

from __future__ import annotations

import argparse
import pickle
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
from physics.lineshape.ssRFMapper import ssRFMapper


def q_polarization(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


@dataclass
class BurnConfig:
    num_bins: int = 249
    f_min: float = -3.0
    f_max: float = 3.0
    sigma: float = 0.16 
    gamma: float = 0.05
    amp_min: float = 0.0
    amp_max: float = 5e-3
    n_amp_bins: int = 165
    max_burns: int = 10
    enforce_full_spectrum: bool = True
    n_q_bins: int = 20
    x_values: np.ndarray | None = None
    lookup_path: Path | None = None

    def __post_init__(self) -> None:
        if self.lookup_path is None:
            self.lookup_path = REPO_ROOT / "Data_Creation" / "lookup_table.pkl"

        if self.x_values is None or len(self.x_values) == 0:
            self.x_values = np.linspace(-2, 2, 165)
        if self.enforce_full_spectrum:
            self.max_burns = min(self.max_burns, len(self.x_values))

    @property
    def f(self) -> np.ndarray:
        return np.linspace(self.f_min, self.f_max, self.num_bins)

    @property
    def amp_values(self) -> np.ndarray:
        """Discrete burn amplitudes; action 0 is always skip (no burn)."""
        if self.n_amp_bins <= 1:
            return np.array([0.0])
        if self.amp_min <= 0.0:
            positive = np.linspace(
                self.amp_max / (self.n_amp_bins - 1),
                self.amp_max,
                self.n_amp_bins - 1,
            )
        else:
            positive = np.linspace(
                self.amp_min,
                self.amp_max,
                self.n_amp_bins - 1,
            )
        return np.concatenate(([0.0], positive))

    @property
    def n_actions(self) -> int:
        if self.enforce_full_spectrum:
            return self.n_amp_bins
        return len(self.x_values) * self.n_amp_bins

    @property
    def n_states(self) -> int:
        return self.n_q_bins * self.max_burns


class TensorBurnEnv:

    def __init__(self, config: BurnConfig, mapper: ssRFMapper):
        self.config = config
        self.mapper = mapper
        self.f = config.f
        self._polarization: float = 0.45
        self._q_lo: float = 0.0
        self._q_hi: float = 1.0
        self._used_x_bins: set[int] = set()
        self.reset()

    def _x_to_freq_bin(self, x: float) -> int:
        return int(np.argmin(np.abs(self.f - x)))

    def _action_to_burn(self, action: int) -> tuple[int, float, float]:
        if self.config.enforce_full_spectrum:
            # In full-spectrum mode, each step maps to one x-bin and action picks amplitude.
            x_idx = int(self._step)
            amp_idx = int(action)
        else:
            x_idx = int(action // self.config.n_amp_bins)
            amp_idx = int(action % self.config.n_amp_bins)
        x = float(self.config.x_values[x_idx])
        amp = float(self.config.amp_values[amp_idx])
        return x_idx, x, amp

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
        stride_q = self.config.n_q_bins
        return int(q_bin) + int(step) * stride_q

    def state_index(self) -> int:
        return self._state_index(self._q_bin, self._step)

    def set_q_bounds(self, q_lo: float, q_hi: float) -> None:
        self._q_lo = q_lo
        self._q_hi = q_hi

    def valid_action_mask(self) -> np.ndarray:
        """Return valid actions under the current action encoding."""
        if self.config.enforce_full_spectrum:
            if self._step >= self.config.max_burns:
                return np.zeros(self.config.n_actions, dtype=bool)
            # All amplitudes are valid, including zero-amplitude (skip burn at this bin).
            return np.ones(self.config.n_actions, dtype=bool)

        # Free-bin mode: allow only actions whose x-bin has not been burned this episode.
        n_x = len(self.config.x_values)
        unused_x = np.ones(n_x, dtype=bool)
        if self._used_x_bins:
            used_idx = np.fromiter(self._used_x_bins, dtype=int)
            unused_x[used_idx] = False
        return np.repeat(unused_x, self.config.n_amp_bins)

    def reset(self, polarization: float | None = None) -> int:
        if polarization is not None:
            self._polarization = float(polarization)
        _, iplus, iminus = GenerateVectorLineshape(self._polarization, self.f)
        self._iplus = np.asarray(iplus, dtype=float)
        self._iminus = np.asarray(iminus, dtype=float)
        self._ps = self._iplus + self._iminus
        self._q0 = q_polarization(self._iplus, self._iminus)
        self._q = self._q0
        self._step = 0
        self._used_x_bins = set()
        self._q_bin = self._q 
        # self._q_bin = self._q_to_bin(self._q)
        return self.state_index()

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        if self._step >= self.config.max_burns:
            return self.state_index(), 0.0, True, {"noop": True}

        q_before = self._q
        x_idx, x, amp = self._action_to_burn(action)
        if (not self.config.enforce_full_spectrum) and (x_idx in self._used_x_bins):
            return self.state_index(), -1e-6, False, {
                "repeated_x_bin": True,
                "bin_idx": x_idx,
                "x": x,
                "f": x,
                "amp": amp,
                "q": self._q,
                "q_gain": self._q - self._q0,
            }

        if not self.config.enforce_full_spectrum:
            self._used_x_bins.add(x_idx)
        freq_bin_idx = self._x_to_freq_bin(x)

        if amp <= 0.0:
            self._step += 1
            self._q_bin = self._q
            done = self._step >= self.config.max_burns
            return self.state_index(), 0.0, done, {
                "skipped": True,
                "bin_idx": x_idx,
                "x": x,
                "f": x,
                "freq_bin_idx": freq_bin_idx,
                "amp": amp,
                "q": self._q,
                "q_gain": self._q - self._q0,
            }

        ps = self._ps.copy()
        iplus = self._iplus.copy()
        iminus = self._iminus.copy()

        self.mapper.apply_bin_burn(ps, iplus, iminus, freq_bin_idx, amp)

        self._ps = ps
        self._iplus = iplus
        self._iminus = iminus
        self._q = q_polarization(iplus, iminus)
        reward = self._q - q_before

        self._step += 1
        # self._q_bin = self._q_to_bin(self._q)
        self._q_bin = self._q 
        done = self._step >= self.config.max_burns
        info = {
            "bin_idx": x_idx,
            "x": x,
            "f": x,
            "freq_bin_idx": freq_bin_idx,
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


def _read_lookup_pickle(path: Path):
    """Load lookup pickle with NumPy 1.x / 2.x module-path compatibility."""
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        module_remap = {
            "numpy._core": "numpy.core",
            "numpy._core.multiarray": "numpy.core.multiarray",
            "numpy._core.numeric": "numpy.core.numeric",
        }

        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                return super().find_class(module_remap.get(module, module), name)

        with path.open("rb") as handle:
            return _CompatUnpickler(handle).load()


def load_mapper(config: BurnConfig) -> ssRFMapper:
    lookup_path = Path(config.lookup_path)
    if not lookup_path.is_file():
        raise FileNotFoundError(
            f"Lookup table not found at {lookup_path}. "
            "Run Data_Creation/lookup_table.py first."
        )
    mapping_data = _read_lookup_pickle(lookup_path)
    mapper = ssRFMapper(config.f, config.sigma, config.gamma, x0=0.0, amp=1e-3)
    mapper.compute_lookup_tables(mapping_data)
    return mapper


class QLearningAgent:
    """Double Q-learning with optimistic initialization and action masking."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.15,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        optimistic_init: float = 0.02,
        seed: int | None = 42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        init = float(optimistic_init)
        self.q_table_a = np.full((n_states, n_actions), init, dtype=np.float64)
        self.q_table_b = np.full((n_states, n_actions), init, dtype=np.float64)

    @property
    def q_table(self) -> np.ndarray:
        """Ensemble Q-values used for greedy policy and visualization."""
        return 0.5 * (self.q_table_a + self.q_table_b)

    def _masked_q_row(self, state: int, action_mask: np.ndarray | None) -> np.ndarray:
        q_values = self.q_table[state].copy()
        if action_mask is not None:
            q_values[~np.asarray(action_mask, dtype=bool)] = -np.inf
        return q_values

    def select_action(
        self,
        state: int,
        explore: bool = True,
        action_mask: np.ndarray | None = None,
    ) -> int:
        mask = (
            np.ones(self.n_actions, dtype=bool)
            if action_mask is None
            else np.asarray(action_mask, dtype=bool)
        )
        valid_actions = np.flatnonzero(mask)
        if valid_actions.size == 0:
            return 0

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))

        q_values = self._masked_q_row(state, mask)
        return int(np.argmax(q_values))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        if self.rng.random() < 0.5:
            target = self.q_table_a
            eval_table = self.q_table_b
        else:
            target = self.q_table_b
            eval_table = self.q_table_a

        if done:
            best_next = 0.0
        else:
            next_values = target[next_state].copy()
            if next_action_mask is not None:
                mask = np.asarray(next_action_mask, dtype=bool)
                if not np.any(mask):
                    best_next = 0.0
                else:
                    next_values[~mask] = -np.inf
                    greedy_action = int(np.argmax(next_values))
                    best_next = float(eval_table[next_state, greedy_action])
            else:
                greedy_action = int(np.argmax(next_values))
                best_next = float(eval_table[next_state, greedy_action])

        td_target = reward + self.gamma * best_next
        target[state, action] += self.alpha * (
            td_target - target[state, action]
        )

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = float(np.clip(epsilon, 0.0, 1.0))


def estimate_q_bounds(
    env: TensorBurnEnv,
    polarizations: np.ndarray,
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


def epsilon_schedule(
    episode: int,
    total_episodes: int,
    eps_start: float = 1.0,
    eps_end: float = 0.02,
) -> float:
    if total_episodes <= 1:
        return eps_end
    progress = episode / (total_episodes - 1)
    return eps_start + (eps_end - eps_start) * progress


def train(
    config: BurnConfig,
    episodes: int = 5000,
    polarizations: np.ndarray | None = None,
    seed: int = 0,
    eps_start: float = 1.0,
    eps_end: float = 0.02,
) -> tuple[QLearningAgent, TensorBurnEnv, dict]:
    if polarizations is None:
        polarizations = np.linspace(0.40, 0.50, 20)

    mapper = load_mapper(config)
    env = TensorBurnEnv(config, mapper)
    q_lo, q_hi = estimate_q_bounds(env, polarizations, seed=seed)
    env.set_q_bounds(q_lo, q_hi)

    agent = QLearningAgent(config.n_states, config.n_actions, seed=seed)
    rng = np.random.default_rng(seed)
    history: list[float] = []

    for ep in tqdm.tqdm(range(episodes), desc="Training Q-learning agent"):
        agent.set_epsilon(
            epsilon_schedule(ep, episodes, eps_start=eps_start, eps_end=eps_end)
        )
        p = float(rng.choice(polarizations))
        state = env.reset(p)
        ep_return = 0.0

        for _ in range(config.max_burns):
            action_mask = env.valid_action_mask()
            if not action_mask.any():
                break
            action = agent.select_action(
                state, explore=True, action_mask=action_mask
            )
            next_state, reward, done, _ = env.step(action)
            next_mask = None if done else env.valid_action_mask()
            agent.update(
                state, action, reward, next_state, done, next_action_mask=next_mask
            )
            ep_return += reward
            state = next_state
            if done:
                break

        history.append(ep_return)

    stats = {
        "q_lo": q_lo,
        "q_hi": q_hi,
        "p_lo": float(np.min(polarizations)),
        "p_hi": float(np.max(polarizations)),
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
        action_mask = env.valid_action_mask()
        if not action_mask.any():
            break
        action = agent.select_action(
            state, explore=False, action_mask=action_mask
        )
        next_state, reward, done, info = env.step(action)
        trace.append(
            {
                "step": step + 1,
                "action": action,
                "bin_idx": info["bin_idx"],
                "f": info["f"],
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
    ax.set_title("Double Q-learning: bin-wise ssRF burn policy")
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
    axes[0].step(
        f, iplus0 + iminus0, color="black", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$P_s$ (unburned)",
    )
    axes[0].step(
        f, iplus0, color="tab:red", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$I_+$ (unburned)",
    )
    axes[0].step(
        f, iminus0, color="tab:blue", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$I_-$ (unburned)",
    )
    axes[0].step(f, iplus + iminus, label=r"$P_s = I_+ + I_-$", color="black")
    axes[0].step(f, iplus, label=r"$I_+$", color="tab:red")
    axes[0].step(f, iminus, label=r"$I_-$", color="tab:blue")
    for row in result["trace"][1:]:
        axes[0].axvline(row["f"], color="green", alpha=0.35, linestyle=":")
        axes[0].axvline(-row["f"], color="purple", alpha=0.25, linestyle=":")
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].step(
        f, q_profile0, color="tab:purple", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$Q$ (unburned)",
    )
    axes[1].step(f, q_profile, color="tab:purple", label=r"$Q = I_+ - I_-$")
    axes[1].set_xlabel("frequency")
    axes[1].set_ylabel("Q profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    delta_q = result["final_q"] - result["initial_q"]
    title = (
        f"P={result['polarization']:.3f}  "
        f"initial vector polarization: {result['initial_q']:.4f} → {result['final_q']:.4f}  "
        f"change in vector polarization: {delta_q:+.4f}"
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
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--polarization", type=float, default=0.45)
    parser.add_argument("--max-burns", type=int, default=165)
    parser.add_argument(
        "--free-bin-selection",
        action="store_true",
        help=(
            "If set, action chooses both bin and amplitude (legacy mode). "
            "Default is full-spectrum mode: one step per bin, action chooses amplitude only."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "current" / "q_learning",
    )
    args = parser.parse_args()

    config = BurnConfig(
        max_burns=args.max_burns,
        enforce_full_spectrum=not args.free_bin_selection,
    )
    # polarizations = np.concatenate([np.linspace(-0.5, -0.05, 12), np.linspace(0.05, 0.5, 12)])
    # polarizations = np.unique(polarizations)
    polarizations = np.linspace(0.40,0.50,20)

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

    print(
        f"Q bounds: [{stats['q_lo']:.4f}, {stats['q_hi']:.4f}]  "
        f"P bounds: [{stats['p_lo']:.4f}, {stats['p_hi']:.4f}]"
    )
    print(f"Greedy policy at P={eval_p*100:.2f}%:")
    for row in greedy["trace"]:
        if row["action"] is None:
            print(f"  start: Q={row['q']*100:.5f}%")
        else:
            print(
                f"  burn {row['step']}: bin={row['bin_idx']}, f={row['f']:.3f}, "
                f"amp={row['amp']:.4e}, reward={row['reward']:.5f}, Q={row['q']*100:.5f}%"
            )
    print(f"  total Q gain: {(greedy['final_q'] - greedy['initial_q'])*100:.5f}%")
    print(f"Saved artifacts to {out_dir}")

    plot_q_table(agent.q_table, out_dir / "q_table.png")


if __name__ == "__main__":
    main()
