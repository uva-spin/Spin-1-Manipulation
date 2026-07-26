"""Smoke tests for dulya_fit_v2 per-bin pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = Path(__file__).resolve().parents[1]
if str(_V2) not in sys.path:
    sys.path.insert(0, str(_V2))

from afp_bin_traj import run_one_bin as run_afp_bin
from afp_bin_traj import run_one_polarization as run_afp_one
from bin_io import (
    load_ssrf_shard,
    organize_ssrf_shards,
    save_ssrf_shard,
    ssrf_shard_path,
    ssrf_train_bin_path,
)
from bin_setup import equilibrium_lineshape, get_shape_params
from common import (
    DEMO_BURN_BIN,
    DEMO_P,
    DT,
    F_MAX,
    F_MIN,
    NUM_BINS,
    PHYSICS_MODEL,
    RF_MODE,
    RF_MODE_PHYSICAL_VOIGT,
    RF_MODE_SINGLE_BIN,
)
from model_bridge import build_spin1_model, configure_ssrf_burn
from ssrf_bin_traj import run_one_bin as run_ssrf_bin
from ssrf_bin_traj import run_one_polarization as run_ssrf_one
from ssrf_realtime_v2.rate_equations_realtime import (
    voigt_burn_recovery_param_snapshot,
)
from unmanipulated_bin_lineshape import run_one_bin as run_unmanip_bin

SMOKE_BIN = 208
SMOKE_P = np.array([-0.3, 0.0, 0.3], dtype=float)


@pytest.fixture()
def smoke_dirs(tmp_path: Path) -> dict[str, Path]:
    dirs = {
        "ssrf_shards": tmp_path / "ssrf_shards",
        "ssrf_train": tmp_path / "ssrf_train",
        "plots": tmp_path / "plots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def test_package_is_self_contained() -> None:
    """No imports from sibling dulya_fit or repo-root physics packages."""
    import common
    import lineshape
    import model_bridge

    assert Path(common.FIT_PARAMS_PATH).resolve().parent == _V2.resolve()
    assert Path(common.FIT_PARAMS_PATH).is_file()
    assert hasattr(lineshape, "GenerateDulyaLineshape")
    assert model_bridge.Spin1Model.__module__.startswith("ssrf_realtime_v2")


def test_equilibrium_lineshape_finite() -> None:
    shape = get_shape_params()
    f = np.linspace(F_MIN, F_MAX, NUM_BINS)
    ps, ip, im = equilibrium_lineshape(0.45, f, shape)
    assert ps.shape == (NUM_BINS,)
    assert np.all(np.isfinite(ps))
    assert np.all(np.isfinite(ip))
    assert np.all(np.isfinite(im))
    assert float(np.max(np.abs(ps))) > 0.0


def test_unmanipulated_bin_smoke(tmp_path: Path) -> None:
    out = run_unmanip_bin(
        SMOKE_BIN,
        output_dir=tmp_path / "unmanip",
        num_bins=NUM_BINS,
        p_min=-0.3,
        p_max=0.3,
        p_step=0.3,
        skip_if_exists=False,
        shape_params=get_shape_params(),
    )
    assert Path(out).is_file()


def test_ssrf_shard_recovery_and_organize(smoke_dirs: dict[str, Path]) -> None:
    result = run_ssrf_bin(
        SMOKE_BIN,
        p_values=SMOKE_P,
        gamma_values=np.array([1.0, 2.0], dtype=float),
        steps_values=np.array([20, 40], dtype=np.int32),
        num_bins=NUM_BINS,
        rf_mode=RF_MODE_PHYSICAL_VOIGT,
    )
    shard_path = ssrf_shard_path(smoke_dirs["ssrf_shards"], SMOKE_BIN)
    save_ssrf_shard(result, shard_path)
    shard = load_ssrf_shard(shard_path)

    assert shard["physics_model"] == PHYSICS_MODEL
    assert shard["rf_mode"] == RF_MODE_PHYSICAL_VOIGT
    assert shard["sampling"] == "p_x_gamma_x_n_steps"
    assert int(shard["p_values"].size) == 12
    assert set(np.asarray(shard["gamma_rf"]).tolist()) == {1.0, 2.0}
    assert set(np.asarray(shard["burn_steps"]).tolist()) == {20, 40}

    for j in range(int(shard["p_values"].size)):
        if bool(shard["skipped"][j]):
            continue
        n = int(shard["n_steps"][j])
        assert n == int(shard["burn_steps"][j]) + 1
        ip = shard["iplus"][j, :n]
        im = shard["iminus"][j, :n]
        ps = shard["ps"][j, :n]
        assert np.all(np.isfinite(ps))
        assert np.max(np.abs(ps)) > 0.0
        assert not np.allclose(ip[0], ip[-1]) or not np.allclose(im[0], im[-1])

    org = organize_ssrf_shards(
        smoke_dirs["ssrf_shards"],
        smoke_dirs["ssrf_train"],
        num_bins=NUM_BINS,
        strict=False,
    )
    assert org["n_samples"] > 0

    train_path = ssrf_train_bin_path(smoke_dirs["ssrf_train"], SMOKE_BIN)
    assert train_path.is_file()
    with np.load(train_path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        expected = {
            "p0",
            "step",
            "gamma_rf",
            "burn_steps",
            "burn_bin",
            "is_mirror",
            "ps",
            "iplus",
            "iminus",
            "amp",
        }
        assert expected.issubset(set(data.files) | {"meta_json", "bin_idx"})
        assert meta["physics_model"] == PHYSICS_MODEL
        assert int(data["ps"].size) > 0


def test_afp_bin_smoke(tmp_path: Path) -> None:
    result = run_afp_bin(
        SMOKE_BIN,
        p_values=SMOKE_P,
        num_bins=NUM_BINS,
        n_relax=40,
    )
    assert int(result["bin_idx"]) == SMOKE_BIN
    assert result["n_steps"].shape[0] == SMOKE_P.size


def test_voigt_and_single_bin_share_recovery_params() -> None:
    f = np.linspace(F_MIN, F_MAX, NUM_BINS)
    shape = get_shape_params()
    _, ip, im = equilibrium_lineshape(DEMO_P, f, shape)
    voigt = build_spin1_model(
        ip,
        im,
        polarization=DEMO_P,
        num_bins=NUM_BINS,
        dt=DT,
        r_min=F_MIN,
        r_max=F_MAX,
    )
    single = build_spin1_model(
        ip,
        im,
        polarization=DEMO_P,
        num_bins=NUM_BINS,
        dt=DT,
        r_min=F_MIN,
        r_max=F_MAX,
    )
    configure_ssrf_burn(voigt, DEMO_BURN_BIN, 2.0, rf_mode=RF_MODE_PHYSICAL_VOIGT)
    configure_ssrf_burn(single, DEMO_BURN_BIN, 2.0, rf_mode=RF_MODE_SINGLE_BIN)
    assert voigt_burn_recovery_param_snapshot(voigt) == voigt_burn_recovery_param_snapshot(
        single
    )
    assert voigt._uses_physical_voigt_rf()
    assert not single._uses_physical_voigt_rf()
    assert RF_MODE == RF_MODE_PHYSICAL_VOIGT


def test_demo_polarization_ssrf_and_afp_effects() -> None:
    """P=0.48 Dulya equilibrium: ssRF and AFP both perturb the burn/center bin."""
    ssrf = run_ssrf_one(
        DEMO_BURN_BIN,
        DEMO_P,
        gamma_rf=2.0,
        n_steps=120,
        rf_mode=RF_MODE_PHYSICAL_VOIGT,
        capture_spectrum=True,
    )
    assert not ssrf["skipped"]
    assert int(ssrf["n_steps"]) > 5
    assert ssrf["rf_mode"] == RF_MODE_PHYSICAL_VOIGT
    assert ssrf["ip_spectrum"] is not None
    assert abs(ssrf["iplus"][-1] - ssrf["iplus"][0]) + abs(
        ssrf["iminus"][-1] - ssrf["iminus"][0]
    ) > 1e-8

    afp = run_afp_one(
        DEMO_BURN_BIN,
        DEMO_P,
        n_relax=2000,
        capture_spectrum=True,
    )
    assert int(afp["n_steps"]) > 1
    assert afp["ip_spectrum"] is not None
    d_ip = float(np.max(np.abs(afp["ip_spectrum"] - afp["ip_spectrum0"])))
    d_im = float(np.max(np.abs(afp["im_spectrum"] - afp["im_spectrum0"])))
    assert d_ip + d_im > 1e-8
    assert float(np.std(afp["ps"])) > 1e-8


def test_plot_physics_demo_writes_pngs(smoke_dirs: dict[str, Path]) -> None:
    from plot_physics_demo import (
        plot_afp_and_relaxation,
        plot_ssrf_burn_and_trajectory,
    )

    out = smoke_dirs["plots"]
    p1 = plot_ssrf_burn_and_trajectory(
        polarization=DEMO_P,
        burn_bin=DEMO_BURN_BIN,
        gamma_rf=2.0,
        max_steps=80,
        rf_mode=RF_MODE_PHYSICAL_VOIGT,
        out_dir=out,
    )
    p2 = plot_afp_and_relaxation(
        polarization=DEMO_P,
        burn_bin=DEMO_BURN_BIN,
        n_relax=40,
        out_dir=out,
    )
    assert p1.is_file() and p1.stat().st_size > 1000
    assert p2.is_file() and p2.stat().st_size > 1000
