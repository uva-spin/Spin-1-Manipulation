"""
Double DQN for sequential bin-wise ssRF burns using Data_Creation/rivanna ssrf_realtime.

Maximizes integrated tensor polarization  Q = sum(I_+ - I_-)  by choosing a
frequency bin and gamma_rf burn strength. The Q-network sees the current
lineshape (I_+, I_-), integrated Q, burn step, and which x-bins are already
used. Burns are only applied at bins whose initial spectral Q = I_+ - I_- is
negative (or theta-summed Q when q_filter_use_theta); other bins may only be
skipped.
"""
import argparse
import copy
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
RIVANNA = REPO_ROOT / 'Data_Creation' / 'rivanna'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RIVANNA) not in sys.path:
    sys.path.insert(0, str(RIVANNA))
from common import DIFFUSION_SCALE, DT, RF_GAUSSIAN_FWHM_R, RF_LORENTZIAN_FWHM_R, RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN
from model_bridge import build_spin1_model, burn_commit_touched_bins, commit_touched_bins_only, configure_ssrf_burn, euler_n_sub
from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.rf.conversions import physical_intensities_to_packet_n
from physics.rf.model import Spin1Model
from physics.rf.rate_equations_realtime import _value_crosses_zero, burn_preserves_ps_sign
OUTPUT_DIR = REPO_ROOT / 'results' / 'current' / 'dqn'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
EPISODES = 100
POLARIZATION = 0.45
MAX_BURNS = 200
FREE_BIN_SELECTION = False
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

def q_polarization(iplus, iminus):
    return np.sum(iplus - iminus)

def q_at_bin(iplus, iminus, bin_idx):
    iplus_theta = iplus[bin_idx] + iplus[len(iplus) - bin_idx - 1]
    iminus_theta = iminus[bin_idx] + iminus[len(iminus) - bin_idx - 1]
    return iplus_theta - iminus_theta

def spectral_q_at_bin(iplus, iminus, bin_idx):
    return iplus[bin_idx] - iminus[bin_idx]

def model_spectrum(model):
    (ip, im, _) = model.physical_intensities()
    return (np.asarray(model.Rplus).copy(), np.asarray(ip), np.asarray(im))

def clone_model(model):
    trial = Spin1Model(copy.deepcopy(model.params))
    trial.n = np.asarray(model.n).copy()
    trial.n_ref = np.asarray(model.n_ref).copy()
    trial.n_initial = np.asarray(model.n_initial).copy()
    trial.t = model.t
    trial.display_cal = model.display_cal
    trial._populations_from_intensities = model._populations_from_intensities
    trial._recovery_boltzmann_P = model._recovery_boltzmann_P
    trial._force_boltzmann_recovery = model._force_boltzmann_recovery
    trial._active_idx = None if model._active_idx is None else np.asarray(model._active_idx).copy()
    trial.n_plus = model.n_plus
    trial.n_zero = model.n_zero
    trial.n_minus = model.n_minus
    trial.n_plus_initial = model.n_plus_initial
    trial.n_zero_initial = model.n_zero_initial
    trial.n_minus_initial = model.n_minus_initial
    return trial

def sync_model_from_spectrum(model, iplus, iminus, *, n_ref, recovery_P):
    model.n = physical_intensities_to_packet_n(np.asarray(iplus), np.asarray(iminus), model.mu, display_cal=model.display_cal, dR=model.dR)
    model.n_ref = np.asarray(n_ref).copy()
    model._populations_from_intensities = True
    model.set_recovery_boltzmann_P(recovery_P)
    model._sync_level_populations(capture_initial=False)
    model.params.rf_enabled = False
    model.params.gamma_rf = 0.0
    model.t = 0.0

def commit_burn_to_spectrum(iplus, iminus, iplus_sim, iminus_sim, bin_idx, *, rf_mode, f, gaussian_fwhm_R, lorentzian_fwhm_R):
    touched = burn_commit_touched_bins(len(iplus), bin_idx, rf_mode=rf_mode, R=f, gaussian_fwhm_R=gaussian_fwhm_R, lorentzian_fwhm_R=lorentzian_fwhm_R, iplus=iplus, iminus=iminus, iplus_sim=iplus_sim, iminus_sim=iminus_sim)
    return commit_touched_bins_only(iplus, iminus, iplus_sim, iminus_sim, touched)

def apply_spin1_burn(model, bin_idx, gamma_rf, n_steps, *, rf_mode, gaussian_fwhm_R, lorentzian_fwhm_R):
    if gamma_rf <= 0.0 or n_steps <= 0:
        return None
    (f_before, iplus_before, iminus_before) = model_spectrum(model)
    bin_idx = bin_idx
    touched = burn_commit_touched_bins(len(iplus_before), bin_idx, rf_mode=rf_mode, R=f_before, gaussian_fwhm_R=gaussian_fwhm_R, lorentzian_fwhm_R=lorentzian_fwhm_R)
    ps_before = iplus_before[bin_idx] + iminus_before[bin_idx]
    burned = clone_model(model)
    configure_ssrf_burn(burned, bin_idx, gamma_rf, rf_mode=rf_mode, gaussian_fwhm_R=gaussian_fwhm_R, lorentzian_fwhm_R=lorentzian_fwhm_R)
    burned.params.rf_burn_R = f_before[bin_idx]
    burned.params.rf_enabled = True
    burned.params.dnp_enabled = False
    ip_prev = {k: iplus_before[k] for k in touched}
    im_prev = {k: iminus_before[k] for k in touched}
    (n_sub, dt_sub) = euler_n_sub(gamma_rf, burned.params.dt)
    steps_done = 0
    for _ in range(n_steps):
        state_before = burned.n.copy()
        for _ in range(n_sub):
            burned.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
        (_, ip_step, im_step) = model_spectrum(burned)
        sign_ok = True
        for idx in touched:
            for (b, a) in ((ip_prev[idx], ip_step[idx]), (im_prev[idx], im_step[idx]), (ip_prev[idx] + im_prev[idx], ip_step[idx] + im_step[idx])):
                if _value_crosses_zero(b, a):
                    sign_ok = False
                    break
            if not sign_ok:
                break
        if not sign_ok:
            burned.n = state_before
            break
        for idx in touched:
            ip_prev[idx] = ip_step[idx]
            im_prev[idx] = im_step[idx]
        steps_done += 1
    if steps_done == 0:
        return None
    (_, iplus_after, iminus_after) = model_spectrum(burned)
    ps_after = iplus_after[bin_idx] + iminus_after[bin_idx]
    if ps_after == ps_before:
        return None
    if not burn_preserves_ps_sign(iplus_before, iminus_before, iplus_after, iminus_after, bin_idx):
        return None
    return burned

@dataclass
class BurnConfig:
    num_bins: int = 249
    f_min: float = -3.0
    f_max: float = 3.0
    dt: float = DT
    burn_steps: int = 800
    gamma_min: float = 0.0
    gamma_max: float = 50.0
    n_gamma_bins: int = 50
    max_burns: int = MAX_BURNS
    enforce_full_spectrum: bool = FREE_BIN_SELECTION
    only_negative_initial_q: bool = True
    q_filter_use_theta: bool = False
    n_q_bins: int = MAX_BURNS
    x_values: np.ndarray | None = None
    rf_mode: str = RF_MODE_PHYSICAL_VOIGT
    diffusion_scale: float = DIFFUSION_SCALE
    gaussian_fwhm_R: float = RF_GAUSSIAN_FWHM_R
    lorentzian_fwhm_R: float = RF_LORENTZIAN_FWHM_R

    def __post_init__(self):
        if self.x_values is None or len(self.x_values) == 0:
            self.x_values = np.linspace(-2, 2, 165)
        if self.enforce_full_spectrum:
            self.max_burns = min(self.max_burns, len(self.x_values))
        if self.rf_mode not in (RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN):
            raise ValueError(f'unknown rf_mode={self.rf_mode!r}')

    @property
    def f(self):
        return np.linspace(self.f_min, self.f_max, self.num_bins)

    @property
    def gamma_values(self):
        """Discrete gamma_rf strengths; action 0 is always skip (no burn)."""
        if self.n_gamma_bins <= 1:
            return np.array([0.0])
        positive = np.linspace(self.gamma_max / (self.n_gamma_bins - 1), self.gamma_max, self.n_gamma_bins - 1)
        positive = positive[positive > 0.0]
        return np.concatenate(([0.0], positive))

    @property
    def n_actions(self):
        if self.enforce_full_spectrum:
            return self.n_gamma_bins
        return len(self.x_values) * self.n_gamma_bins

    @property
    def n_states(self):
        return self.n_q_bins * self.max_burns

def build_spin1_from_polarization(config, polarization):
    (_, iplus, iminus) = GenerateVectorLineshape(polarization, config.f)
    return build_spin1_model(np.asarray(iplus), np.asarray(iminus), polarization=polarization, num_bins=config.num_bins, dt=config.dt, rf_enabled=False, relax_enabled=True, diffusion_scale=config.diffusion_scale, rf_gaussian_fwhm_R=config.gaussian_fwhm_R, rf_lorentzian_fwhm_R=config.lorentzian_fwhm_R, r_min=config.f_min, r_max=config.f_max)

class Spin1BurnEnv:
    """RL environment for sequential ssRF burns with ssrf_realtime physics."""

    def __init__(self, config):
        self.config = config
        self.f = config.f
        self._polarization = 0.45
        self._used_x_bins = set()
        self._model = None
        self._iplus = np.zeros(config.num_bins)
        self._iminus = np.zeros(config.num_bins)
        self._iplus0 = np.zeros(config.num_bins)
        self._iminus0 = np.zeros(config.num_bins)
        self._n_ref = np.zeros(config.num_bins)
        self._recovery_P = 0.0
        self._q0 = 0.0
        self._q = 0.0
        self._step = 0
        self.reset()

    @property
    def observation_dim(self):
        return 2 * self.config.num_bins + 2 + len(self.config.x_values)

    def observation(self):
        n_x = len(self.config.x_values)
        used = np.zeros(n_x)
        for idx in self._used_x_bins:
            if 0 <= idx < n_x:
                used[idx] = 1.0
        step_frac = self._step / max(self.config.max_burns, 1)
        return np.concatenate([np.asarray(self._iplus), np.asarray(self._iminus), np.array([self._q, step_frac]), used]).astype(np.float32)

    def _x_to_freq_bin(self, x):
        return np.argmin(np.abs(self.f - x))

    def _action_to_burn(self, action):
        if self.config.enforce_full_spectrum:
            x_idx = self._step
            gamma_idx = action
        else:
            x_idx = action // self.config.n_gamma_bins
            gamma_idx = action % self.config.n_gamma_bins
        x = self.config.x_values[x_idx]
        gamma_rf = self.config.gamma_values[gamma_idx]
        return (x_idx, x, gamma_rf)

    def _initial_q_at_x_idx(self, x_idx):
        x = self.config.x_values[x_idx]
        freq_bin_idx = self._x_to_freq_bin(x)
        if self.config.q_filter_use_theta:
            return q_at_bin(self._iplus0, self._iminus0, freq_bin_idx)
        return spectral_q_at_bin(self._iplus0, self._iminus0, freq_bin_idx)

    def _burn_allowed_at_x_idx(self, x_idx):
        if not self.config.only_negative_initial_q:
            return True
        return self._initial_q_at_x_idx(x_idx) < 0.0

    def valid_action_mask(self):
        if self.config.enforce_full_spectrum:
            if self._step >= self.config.max_burns:
                return np.zeros(self.config.n_actions, dtype=bool)
            mask = np.ones(self.config.n_actions, dtype=bool)
            if not self._burn_allowed_at_x_idx(self._step):
                mask[1:] = False
            return mask
        n_x = len(self.config.x_values)
        n_gamma = self.config.n_gamma_bins
        mask = np.zeros(n_x * n_gamma, dtype=bool)
        for x_idx in range(n_x):
            if x_idx in self._used_x_bins:
                continue
            start = x_idx * n_gamma
            mask[start] = True
            if self._burn_allowed_at_x_idx(x_idx):
                mask[start + 1:start + n_gamma] = True
        return mask

    def reset(self, polarization=None):
        if polarization is not None:
            self._polarization = polarization
        self._model = build_spin1_from_polarization(self.config, self._polarization)
        self._model.set_recovery_boltzmann_P(self._model.n_plus - self._model.n_minus)
        self._n_ref = self._model.n_ref.copy()
        self._recovery_P = self._model.n_plus - self._model.n_minus
        (_, self._iplus, self._iminus) = model_spectrum(self._model)
        self._iplus0 = self._iplus.copy()
        self._iminus0 = self._iminus.copy()
        self._q0 = q_polarization(self._iplus, self._iminus)
        self._q = self._q0
        self._step = 0
        self._used_x_bins = set()
        return self.observation()

    def step(self, action):
        (x_idx, x, gamma_rf) = self._action_to_burn(action)
        if not self.config.enforce_full_spectrum and x_idx in self._used_x_bins:
            return (self.observation(), -1e-06, False, {'repeated_x_bin': True, 'bin_idx': x_idx, 'x': x, 'f': x, 'gamma_rf': gamma_rf, 'q': self._q, 'q_gain': self._q - self._q0})
        if not self.config.enforce_full_spectrum:
            self._used_x_bins.add(x_idx)
        freq_bin_idx = self._x_to_freq_bin(x)
        q_before = self._q
        if gamma_rf <= 0.0:
            self._step += 1
            done = self._step >= self.config.max_burns
            return (self.observation(), 0.0, done, {'skipped': True, 'bin_idx': x_idx, 'x': x, 'f': x, 'freq_bin_idx': freq_bin_idx, 'gamma_rf': gamma_rf, 'q': self._q, 'q_gain': self._q - self._q0})
        if not self._burn_allowed_at_x_idx(x_idx):
            self._step += 1
            done = self._step >= self.config.max_burns
            return (self.observation(), -1e-06, done, {'positive_initial_q': True, 'bin_idx': x_idx, 'x': x, 'f': x, 'freq_bin_idx': freq_bin_idx, 'gamma_rf': gamma_rf, 'initial_q_bin': self._initial_q_at_x_idx(x_idx), 'q': self._q, 'q_gain': self._q - self._q0})
        assert self._model is not None
        burned = apply_spin1_burn(self._model, freq_bin_idx, gamma_rf, self.config.burn_steps, rf_mode=self.config.rf_mode, gaussian_fwhm_R=self.config.gaussian_fwhm_R, lorentzian_fwhm_R=self.config.lorentzian_fwhm_R)
        if burned is None:
            self._step += 1
            done = self._step >= self.config.max_burns
            return (self.observation(), -1e-06, done, {'failed_burn': True, 'bin_idx': x_idx, 'x': x, 'f': x, 'freq_bin_idx': freq_bin_idx, 'gamma_rf': gamma_rf, 'q': self._q, 'q_gain': self._q - self._q0})
        (_, ip_sim, im_sim) = model_spectrum(burned)
        (ip_new, im_new) = commit_burn_to_spectrum(self._iplus, self._iminus, ip_sim, im_sim, freq_bin_idx, rf_mode=self.config.rf_mode, f=self.f, gaussian_fwhm_R=self.config.gaussian_fwhm_R, lorentzian_fwhm_R=self.config.lorentzian_fwhm_R)
        sync_model_from_spectrum(self._model, ip_new, im_new, n_ref=self._n_ref, recovery_P=self._recovery_P)
        self._iplus = ip_new
        self._iminus = im_new
        self._q = q_polarization(self._iplus, self._iminus)
        reward = self._q - q_before
        self._step += 1
        done = self._step >= self.config.max_burns
        info = {'bin_idx': x_idx, 'x': x, 'f': x, 'freq_bin_idx': freq_bin_idx, 'gamma_rf': gamma_rf, 'q': self._q, 'q_gain': self._q - self._q0}
        return (self.observation(), reward, done, info)

    @property
    def current_q(self):
        return self._q

    @property
    def initial_q(self):
        return self._q0

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
        return {'states': self.states[idx], 'actions': self.actions[idx], 'rewards': self.rewards[idx], 'next_states': self.next_states[idx], 'dones': self.dones[idx], 'next_masks': self.next_masks[idx]}

class QNetwork(nn.Module):
    """MLP mapping lineshape observations to per-action Q-values."""

    def __init__(self, state_dim, n_actions, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_actions))
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
    """Double DQN agent for sequential ssRF tensor-enhancement burns."""

    def __init__(self, state_dim, n_actions, *, hidden_dim=HIDDEN_DIM, lr=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON_START, epsilon_min=EPSILON_END, buffer_size=REPLAY_SIZE, batch_size=BATCH_SIZE, target_update_interval=TARGET_UPDATE_INTERVAL, learn_start=LEARN_START, device=None, seed=SEED):
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
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done, next_mask):
        self.buffer.add(state, action, reward, next_state, done, next_mask)

    def learn(self):
        if self.buffer.size < max(self.learn_start, self.batch_size):
            return None
        batch = self.buffer.sample(self.batch_size)
        states = torch.as_tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch['actions'], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch['dones'], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch['next_masks'], dtype=torch.bool, device=self.device)
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
    env = Spin1BurnEnv(config)
    agent = DQNAgent(env.observation_dim, config.n_actions, lr=lr, seed=seed, device=device)
    rng = np.random.default_rng(seed)
    history = []
    losses = []
    for ep in tqdm.tqdm(range(episodes), desc='Training DQN agent'):
        agent.set_epsilon(epsilon_schedule(ep, episodes, eps_start=eps_start, eps_end=eps_end))
        p = rng.choice(polarizations)
        state = env.reset(p)
        ep_return = 0.0
        for _ in range(config.max_burns):
            action_mask = env.valid_action_mask()
            if not action_mask.any():
                break
            action = agent.select_action(state, explore=True, action_mask=action_mask)
            (next_state, reward, done, _) = env.step(action)
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
    stats = {'p_lo': np.min(polarizations), 'p_hi': np.max(polarizations), 'final_q_gain': env.current_q - env.initial_q, 'episode_returns': np.asarray(history), 'losses': np.asarray(losses), 'physics_model': 'ssrf_realtime', 'rf_mode': config.rf_mode, 'n_actions': config.n_actions, 'observation_dim': env.observation_dim}
    return (agent, env, stats)

def greedy_episode(env, agent, polarization):
    state = env.reset(polarization)
    iplus_unburned = env._iplus.copy()
    iminus_unburned = env._iminus.copy()
    trace = [{'step': 0, 'q': env.initial_q, 'action': None}]
    for step in range(env.config.max_burns):
        action_mask = env.valid_action_mask()
        if not action_mask.any():
            break
        action = agent.select_action(state, explore=False, action_mask=action_mask)
        (next_state, reward, done, info) = env.step(action)
        trace.append({'step': step + 1, 'action': action, 'bin_idx': info['bin_idx'], 'f': info['f'], 'gamma_rf': info['gamma_rf'], 'reward': reward, 'q': info['q'], 'q_gain': info['q_gain']})
        state = next_state
        if done:
            break
    return {'polarization': polarization, 'initial_q': env.initial_q, 'final_q': env.current_q, 'trace': trace, 'iplus_unburned': iplus_unburned, 'iminus_unburned': iminus_unburned, 'iplus': env._iplus.copy(), 'iminus': env._iminus.copy(), 'f': env.f.copy()}

def plot_training_returns(returns, output_path):
    window = min(200, max(1, len(returns) // 20))
    smoothed = np.convolve(returns, np.ones(window) / window, mode='valid')
    (fig, ax) = plt.subplots(figsize=(8, 4))
    ax.plot(returns, alpha=0.25, linewidth=0.8, label='episode return')
    ax.plot(np.arange(window - 1, window - 1 + len(smoothed)), smoothed, color='C1', linewidth=2, label=f'{window}-ep moving avg')
    ax.set_xlabel('episode')
    ax.set_ylabel('sum of Q rewards')
    ax.set_title('DQN: ssrf_realtime bin-wise ssRF burn policy')
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
    smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
    (fig, ax) = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.25, linewidth=0.8, label='TD loss')
    ax.plot(np.arange(window - 1, window - 1 + len(smoothed)), smoothed, color='C1', linewidth=2, label=f'{window}-step moving avg')
    ax.set_xlabel('gradient step')
    ax.set_ylabel('smooth L1 loss')
    ax.set_title('DQN temporal-difference loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_greedy_burns(result, output_path):
    f = result['f']
    iplus = result['iplus']
    iminus = result['iminus']
    iplus0 = result['iplus_unburned']
    iminus0 = result['iminus_unburned']
    q_profile = iplus - iminus
    q_profile0 = iplus0 - iminus0
    (fig, axes) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].step(f, iplus0 + iminus0, color='black', linestyle='--', alpha=0.55, linewidth=1.0, label='$P_s$ (unburned)')
    axes[0].step(f, iplus0, color='tab:red', linestyle='--', alpha=0.55, linewidth=1.0, label='$I_+$ (unburned)')
    axes[0].step(f, iminus0, color='tab:blue', linestyle='--', alpha=0.55, linewidth=1.0, label='$I_-$ (unburned)')
    axes[0].step(f, iplus + iminus, label='$P_s = I_+ + I_-$', color='black')
    axes[0].step(f, iplus, label='$I_+$', color='tab:red')
    axes[0].step(f, iminus, label='$I_-$', color='tab:blue')
    for row in result['trace'][1:]:
        if row.get('gamma_rf', 0.0) > 0.0:
            axes[0].axvline(row['f'], color='green', alpha=0.35, linestyle=':')
            axes[0].axvline(-row['f'], color='purple', alpha=0.25, linestyle=':')
    axes[0].set_ylabel('intensity')
    axes[0].legend(loc='upper right', fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].step(f, q_profile0, color='tab:purple', linestyle='--', alpha=0.55, linewidth=1.0, label='$Q$ (unburned)')
    axes[1].step(f, q_profile, color='tab:purple', label='$Q = I_+ - I_-$')
    axes[1].set_xlabel('frequency')
    axes[1].set_ylabel('Q profile')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    delta_q = result['final_q'] - result['initial_q']
    title = f"P={result['polarization']:.3f}  initial vector polarization: {result['initial_q']:.4f} → {result['final_q']:.4f}  change in vector polarization: {delta_q:+.4f}"
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Double DQN for ssRF tensor (Q) enhancement')
    parser.add_argument('--episodes', type=int, default=EPISODES)
    parser.add_argument('--polarization', type=float, default=POLARIZATION)
    parser.add_argument('--max-burns', type=int, default=MAX_BURNS)
    parser.add_argument('--full-spectrum', action='store_true', help='Scan bins in order; action chooses gamma_rf only. Default is free-bin selection.')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    config = BurnConfig(max_burns=args.max_burns, enforce_full_spectrum=args.full_spectrum)
    polarizations = np.linspace(0.4, 0.5, 20)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Training Double DQN with rivanna ssrf_realtime ({config.rf_mode}, device={DEVICE}, n_actions={config.n_actions}, only_Q<0 bins={config.only_negative_initial_q}, theta_Q_filter={config.q_filter_use_theta})...')
    (agent, env, stats) = train(config, episodes=args.episodes, polarizations=polarizations, seed=args.seed, lr=args.lr, device=DEVICE)
    ckpt_path = out_dir / 'dqn_policy.pt'
    torch.save({'online': agent.online.state_dict(), 'target': agent.target.state_dict(), 'observation_dim': env.observation_dim, 'n_actions': config.n_actions, 'hidden_dim': HIDDEN_DIM, 'config': {'num_bins': config.num_bins, 'max_burns': config.max_burns, 'n_gamma_bins': config.n_gamma_bins, 'enforce_full_spectrum': config.enforce_full_spectrum, 'rf_mode': config.rf_mode}, 'stats': {'p_lo': stats['p_lo'], 'p_hi': stats['p_hi'], 'n_actions': stats['n_actions'], 'observation_dim': stats['observation_dim']}}, ckpt_path)
    np.save(out_dir / 'episode_returns.npy', stats['episode_returns'])
    plot_training_returns(stats['episode_returns'], out_dir / 'training_returns.png')
    plot_losses(stats['losses'], out_dir / 'td_loss.png')
    eval_p = args.polarization
    greedy = greedy_episode(env, agent, eval_p)
    plot_greedy_burns(greedy, out_dir / f'greedy_policy_P{eval_p:.2f}.png')
    print(f'Greedy policy at P={eval_p * 100:.2f}%:')
    for row in greedy['trace']:
        if row['action'] is None:
            print(f"  start: Q={row['q'] * 100:.5f}%")
        elif row.get('gamma_rf', 0.0) <= 0.0:
            print(f"  skip {row['step']}: bin={row['bin_idx']}, f={row['f']:.3f}, Q={row['q'] * 100:.5f}%")
        else:
            print(f"  burn {row['step']}: bin={row['bin_idx']}, f={row['f']:.3f}, gamma_rf={row['gamma_rf']:.4e}, reward={row['reward']:.5f}, Q={row['q'] * 100:.5f}%")
    print(f"  total Q gain: {(greedy['final_q'] - greedy['initial_q']) * 100:.5f}%")
    print(f'Saved artifacts to {out_dir}')
if __name__ == '__main__':
    main()
