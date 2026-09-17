"""
Tabular SARSA for sequential bin-wise ssRF burns using Data_Creation/rivanna ssrf_realtime.

Maximizes integrated Q polarization  Q = sum(I_+ - I_-)  by choosing a frequency
bin and gamma_rf burn strength. Burns are only applied at bins whose initial
spectral Q = I_+ - I_- is negative (or theta-summed Q when q_filter_use_theta);
other bins may only be skipped.
Burns integrate the shared rate-equation model (physical Voigt RF + spectral
recovery) from Data_Creation/rivanna.
"""
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tqdm
REPO_ROOT = Path(__file__).resolve().parents[1]
RIVANNA = REPO_ROOT / 'Data_Creation' / 'rivanna'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RIVANNA) not in sys.path:
    sys.path.insert(0, str(RIVANNA))
from common import DIFFUSION_SCALE, RF_GAUSSIAN_FWHM_R, RF_LORENTZIAN_FWHM_R, RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN
from model_bridge import build_spin1_model, burn_commit_touched_bins, commit_touched_bins_only, configure_ssrf_burn, euler_n_sub
from physics.lineshape.Lineshape import GenerateVectorLineshape
from ssrf_realtime.conversions import physical_intensities_to_packet_n
from ssrf_realtime.model import Spin1Model
from ssrf_realtime.rate_equations_realtime import _value_crosses_zero, burn_preserves_ps_sign
OUTPUT_DIR = REPO_ROOT / 'results' / 'current' / 'sarsa'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SARSA_DT = 0.015
SEED = 42
EPISODES = 100
POLARIZATION = 0.45
MAX_BURNS = 20
FREE_BIN_SELECTION = False

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
    num_bins = 249
    f_min = -3.0
    f_max = 3.0
    dt = SARSA_DT
    burn_steps = 100
    gamma_min = 0.0
    gamma_max = 50.0
    n_gamma_bins = 10
    max_burns = MAX_BURNS
    enforce_full_spectrum = FREE_BIN_SELECTION
    only_negative_initial_q = True
    q_filter_use_theta = False
    n_q_bins = MAX_BURNS
    x_values = None
    rf_mode = RF_MODE_PHYSICAL_VOIGT
    diffusion_scale = DIFFUSION_SCALE
    gaussian_fwhm_R = RF_GAUSSIAN_FWHM_R
    lorentzian_fwhm_R = RF_LORENTZIAN_FWHM_R

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
        self._q_lo = 0.0
        self._q_hi = 1.0
        self._used_x_bins = set()
        self._model = None
        self._iplus = np.zeros(config.num_bins)
        self._iminus = np.zeros(config.num_bins)
        self._iplus0 = np.zeros(config.num_bins)
        self._iminus0 = np.zeros(config.num_bins)
        self._n_ref = np.zeros(config.num_bins)
        self._recovery_P = 0.0
        self.reset()

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

    def _q_to_bin(self, q):
        if not np.isfinite(q):
            return 0
        span = self._q_hi - self._q_lo
        if span <= 0:
            return 0
        idx = (q - self._q_lo) / span * self.config.n_q_bins
        return np.clip(idx, 0, self.config.n_q_bins - 1)

    def _state_index(self, q_bin, step):
        step = np.clip(step, 0, self.config.max_burns - 1)
        return q_bin + step * self.config.n_q_bins

    def state_index(self):
        return self._state_index(self._q_bin, self._step)

    def set_q_bounds(self, q_lo, q_hi):
        self._q_lo = q_lo
        self._q_hi = q_hi

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
        self._q_bin = self._q_to_bin(self._q0)
        return self.state_index()

    def step(self, action):
        (x_idx, x, gamma_rf) = self._action_to_burn(action)
        if not self.config.enforce_full_spectrum and x_idx in self._used_x_bins:
            return (self.state_index(), -1e-06, False, {'repeated_x_bin': True, 'bin_idx': x_idx, 'x': x, 'f': x, 'gamma_rf': gamma_rf, 'q': self._q, 'q_gain': self._q - self._q0})
        if not self.config.enforce_full_spectrum:
            self._used_x_bins.add(x_idx)
        freq_bin_idx = self._x_to_freq_bin(x)
        if gamma_rf <= 0.0:
            self._step += 1
            self._q_bin = self._q_to_bin(self._q)
            done = self._step >= self.config.max_burns
            return (self.state_index(), 0.0, done, {'skipped': True, 'bin_idx': x_idx, 'x': x, 'f': x, 'freq_bin_idx': freq_bin_idx, 'gamma_rf': gamma_rf, 'q': self._q, 'q_gain': self._q - self._q0})
        if not self._burn_allowed_at_x_idx(x_idx):
            self._step += 1
            self._q_bin = self._q_to_bin(self._q)
            done = self._step >= self.config.max_burns
            return (self.state_index(), -1e-06, done, {'positive_initial_q': True, 'bin_idx': x_idx, 'x': x, 'f': x, 'freq_bin_idx': freq_bin_idx, 'gamma_rf': gamma_rf, 'initial_q_bin': self._initial_q_at_x_idx(x_idx), 'q': self._q, 'q_gain': self._q - self._q0})
        assert self._model is not None
        burned = apply_spin1_burn(self._model, freq_bin_idx, gamma_rf, self.config.burn_steps, rf_mode=self.config.rf_mode, gaussian_fwhm_R=self.config.gaussian_fwhm_R, lorentzian_fwhm_R=self.config.lorentzian_fwhm_R)
        if burned is None:
            self._step += 1
            self._q_bin = self._q_to_bin(self._q)
            done = self._step >= self.config.max_burns
            return (self.state_index(), -1e-06, done, {'failed_burn': True, 'bin_idx': x_idx, 'x': x, 'f': x, 'freq_bin_idx': freq_bin_idx, 'gamma_rf': gamma_rf, 'q': self._q, 'q_gain': self._q - self._q0})
        (_, ip_sim, im_sim) = model_spectrum(burned)
        (ip_new, im_new) = commit_burn_to_spectrum(self._iplus, self._iminus, ip_sim, im_sim, freq_bin_idx, rf_mode=self.config.rf_mode, f=self.f, gaussian_fwhm_R=self.config.gaussian_fwhm_R, lorentzian_fwhm_R=self.config.lorentzian_fwhm_R)
        sync_model_from_spectrum(self._model, ip_new, im_new, n_ref=self._n_ref, recovery_P=self._recovery_P)
        self._iplus = ip_new
        self._iminus = im_new
        self._q = q_polarization(self._iplus, self._iminus)
        reward = self._q - self._q0
        self._step += 1
        self._q_bin = self._q_to_bin(self._q)
        done = self._step >= self.config.max_burns
        info = {'bin_idx': x_idx, 'x': x, 'f': x, 'freq_bin_idx': freq_bin_idx, 'gamma_rf': gamma_rf, 'q': self._q, 'q_gain': self._q - self._q0}
        return (self.state_index(), reward, done, info)

    @property
    def current_q(self):
        return self._q

    @property
    def initial_q(self):
        return self._q0

class SARSAAgent:
    """SARSA agent for sequential bin-wise ssRF burns on a Spin1 lineshape."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=1.0, seed=42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = np.full((n_states, n_actions), 0.0)

    def _masked_q_row(self, state, action_mask):
        q_values = self.q_table[state].copy()
        if action_mask is not None:
            q_values[~np.asarray(action_mask, dtype=bool)] = -np.inf
        return q_values

    def select_action(self, state, explore=True, action_mask=None):
        mask = np.ones(self.n_actions, dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool)
        valid_actions = np.flatnonzero(mask)
        if valid_actions.size == 0:
            return 0
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        q_values = self._masked_q_row(state, mask)
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action, done):
        if done or next_action is None:
            next_q = 0.0
        else:
            next_q = self.q_table[next_state, next_action]
        td_target = reward + self.gamma * next_q
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

def estimate_q_bounds(env, polarizations, seed=0):
    rng = np.random.default_rng(seed)
    qs = []
    for p in polarizations:
        env.reset(p)
        qs.append(env.initial_q)
        for _ in range(env.config.max_burns):
            action = rng.integers(0, env.config.n_actions)
            (_, _, done, info) = env.step(action)
            qs.append(info['q'])
            if done:
                break
    margin = 0.05 * (max(qs) - min(qs) + 1e-09)
    return (min(qs) - margin, max(qs) + margin)

def epsilon_schedule(episode, total_episodes, eps_start=1.0, eps_end=0.01):
    if total_episodes <= 1:
        return eps_end
    progress = episode / (total_episodes - 1)
    return eps_start + (eps_end - eps_start) * progress

def train(config, episodes=5000, polarizations=None, seed=0, eps_start=1.0, eps_end=0.02):
    if polarizations is None:
        polarizations = np.linspace(0.4, 0.5, 20)
    env = Spin1BurnEnv(config)
    (q_lo, q_hi) = estimate_q_bounds(env, polarizations, seed=seed)
    env.set_q_bounds(q_lo, q_hi)
    agent = SARSAAgent(config.n_states, config.n_actions, seed=seed)
    rng = np.random.default_rng(seed)
    history = []
    for ep in tqdm.tqdm(range(episodes), desc='Training SARSA agent'):
        agent.set_epsilon(epsilon_schedule(ep, episodes, eps_start=eps_start, eps_end=eps_end))
        p = rng.choice(polarizations)
        state = env.reset(p)
        ep_return = 0.0
        action_mask = env.valid_action_mask()
        if not action_mask.any():
            history.append(ep_return)
            continue
        action = agent.select_action(state, explore=True, action_mask=action_mask)
        for _ in range(config.max_burns):
            (next_state, reward, done, _) = env.step(action)
            ep_return += reward
            if done:
                agent.update(state, action, reward, next_state, None, done)
                break
            next_mask = env.valid_action_mask()
            if not next_mask.any():
                agent.update(state, action, reward, next_state, None, True)
                break
            next_action = agent.select_action(next_state, explore=True, action_mask=next_mask)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
        history.append(ep_return)
    stats = {'q_lo': q_lo, 'q_hi': q_hi, 'p_lo': np.min(polarizations), 'p_hi': np.max(polarizations), 'final_q_gain': env.current_q - env.initial_q, 'episode_returns': np.asarray(history), 'physics_model': 'ssrf_realtime', 'rf_mode': config.rf_mode}
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
    ax.set_title('SARSA: ssrf_realtime bin-wise ssRF burn policy')
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

def plot_q_table(q_table, output_path):
    (q_min, q_max) = (np.min(q_table), np.max(q_table))
    span = q_max - q_min
    normalized = (q_table - q_min) / span if span > 0 else np.zeros_like(q_table)
    (fig, ax) = plt.subplots(figsize=(8, 4))
    im = ax.imshow(normalized, cmap='Spectral', aspect='auto', vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, label='normalized Q')
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels([f'{q_min:.3g}', f'{(q_min + q_max) / 2:.3g}', f'{q_max:.3g}'])
    ax.set_xlabel('action')
    ax.set_ylabel('state')
    ax.set_title('SARSA-table (min–max normalized)')
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    config = BurnConfig(max_burns=MAX_BURNS, enforce_full_spectrum=FREE_BIN_SELECTION)
    polarizations = np.linspace(0.4, 0.5, 20)
    print(f'Training SARSA agent with rivanna ssrf_realtime ({config.rf_mode}, only_Q<0 bins={config.only_negative_initial_q}, theta_Q_filter={config.q_filter_use_theta})...')
    (agent, env, stats) = train(config, episodes=EPISODES, polarizations=polarizations, seed=SEED)
    np.save(OUTPUT_DIR / 'sarsa_table.npy', agent.q_table)
    plot_training_returns(stats['episode_returns'], OUTPUT_DIR / 'training_returns.png')
    eval_p = POLARIZATION
    greedy = greedy_episode(env, agent, eval_p)
    plot_greedy_burns(greedy, OUTPUT_DIR / f'greedy_policy_P{eval_p:.2f}.png')
    print(f'Greedy policy at P={eval_p * 100:.2f}%:')
    for row in greedy['trace']:
        if row['action'] is None:
            print(f"  start: Q={row['q'] * 100:.5f}%")
        elif row.get('gamma_rf', 0.0) <= 0.0:
            print(f"  skip {row['step']}: bin={row['bin_idx']}, f={row['f']:.3f}, Q={row['q'] * 100:.5f}%")
        else:
            print(f"  burn {row['step']}: bin={row['bin_idx']}, f={row['f']:.3f}, gamma_rf={row['gamma_rf']:.4e}, reward={row['reward']:.5f}, Q={row['q'] * 100:.5f}%")
    print(f"  total Q gain: {(greedy['final_q'] - greedy['initial_q']) * 100:.5f}%")
    print(f'Saved artifacts to {OUTPUT_DIR}')
    plot_q_table(agent.q_table, OUTPUT_DIR / 'sarsa_table.png')
if __name__ == '__main__':
    main()
