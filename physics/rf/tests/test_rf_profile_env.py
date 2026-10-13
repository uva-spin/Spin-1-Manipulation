"""Env codec: unit action -> PulseProgram, masked bins stay off (no torch, no ssRF-beta)."""
import ast
from pathlib import Path
import numpy as np
import pytest
from physics.rf.profile_control import (
    RFProfileEnv,
    ProfileBuildEnv,
    decode_unit_action,
    rates_and_duration_to_program,
    q_shaped_manual_rates,
)


def test_decode_unit_action_scales_rates_and_duration():
    n = 11
    action = np.zeros(n + 1)
    action[3] = 1.0
    action[-1] = 0.0
    rates, duration = decode_unit_action(action, u_max=20.0, t_min=0.01, t_max=2.0)
    assert rates.shape == (n,)
    assert rates[3] == pytest.approx(20.0)
    assert duration == pytest.approx(0.01)
    action[-1] = 1.0
    _, duration = decode_unit_action(action, u_max=20.0, t_min=0.01, t_max=2.0)
    assert duration == pytest.approx(2.0)


def test_action_to_program_zeros_masked_bins():
    env = RFProfileEnv(n_bins=21, polarization=0.45)
    action = np.ones(env.action_dim)
    program = env.action_to_program(action)
    assert program.n_bins == 21
    np.testing.assert_allclose(program.grid, env.model.Rplus, atol=1e-12)
    rates = np.zeros(21)
    for pulse in program.profiles[0].pulses:
        rates[pulse.bin_index] = pulse.rate
        assert env.mask[pulse.bin_index]
    assert np.all(rates[~env.mask] == 0.0)


def test_env_step_finite_delta_q():
    env = RFProfileEnv(n_bins=21, polarization=0.45)
    action = np.zeros(env.action_dim)
    action[np.where(env.mask)[0][:3]] = 0.4
    action[-1] = 0.2
    obs, reward, done, info = env.step(action)
    assert done is True
    assert obs.shape == (env.observation_dim,)
    assert np.isfinite(reward)
    assert np.isfinite(info["Q"])
    assert env.last_program is not None


def test_empty_action_gives_zero_reward():
    env = RFProfileEnv(n_bins=21, polarization=0.45)
    action = np.zeros(env.action_dim)
    _, reward, _, info = env.step(action)
    assert info["empty"] is True
    assert reward == 0.0


def test_env_modules_do_not_import_ssrf_beta():
    files = [
        Path(__file__).resolve().parents[1] / "profile_control.py",
        Path(__file__).resolve().parents[1] / "optimal_profile.py",
        Path(__file__).resolve().parents[3] / "ml" / "rf_profile_rl.py",
    ]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                assert "ssRF-beta" not in name
                assert "ssrf-beta" not in name.lower().replace("_", "-")
                assert "ssrf_realtime" not in name


def test_q_shaped_manual_rates_nonnegative():
    env = RFProfileEnv(n_bins=21, polarization=0.45)
    ip, im, _ = env.model.physical_intensities()
    rates = q_shaped_manual_rates(ip, im, gamma_rf=2.0)
    assert rates.shape == ip.shape
    assert np.all(rates >= 0.0)
    program = rates_and_duration_to_program(rates, 0.2, r_min=-3.0, r_max=3.0, mask=env.mask)
    assert program.n_bins == 21


def test_profile_build_env_uses_population_q_and_simultaneous_program():
    env = ProfileBuildEnv(n_bins=21, polarization=0.45, max_steps=4, n_u_bins=5, n_t_bins=3)
    obs = env.reset(0.45)
    assert obs.shape == (env.observation_dim,)
    pol = env.model.polarizations()
    assert env.initial_q == pytest.approx(pol["Q"])
    assert env.current_p == pytest.approx(pol["P"])
    mask = env.valid_action_mask()
    assert mask[: env.n_duration_actions].all()
    candidates = np.flatnonzero(env.mask)
    assert candidates.size > 0
    n_u = env.u_values.size
    bin_idx = int(candidates[0])
    u_idx = n_u - 1
    action = env.n_duration_actions + bin_idx * n_u + u_idx
    obs2, reward, done, info = env.step(action)
    assert np.isfinite(reward)
    assert info["rate"] > 0.0
    assert env.last_program is not None
    assert env.last_program.n_bins == 21
    starts = {pulse.start for pulse in env.last_program.profiles[0].pulses}
    assert starts == {0.0} or len(starts) == 0
    assert env.rates[bin_idx] == pytest.approx(env.u_values[u_idx])
    assert np.all(env.rates[~env.mask] == 0.0)
