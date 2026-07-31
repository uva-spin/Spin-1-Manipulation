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
from afp_bin_traj import run_one_bin_spectrum as run_afp_spectrum_bin
from afp_bin_traj import run_one_polarization as run_afp_one
from bin_io import (
    afp_shard_path,
    afp_spectrum_shard_path,
    load_ssrf_shard,
    organize_ssrf_shards,
    save_afp_shard,
    save_afp_spectrum_shard,
    save_ssrf_shard,
    save_ssrf_spectrum_shard,
    ssrf_shard_path,
    ssrf_spectrum_shard_path,
    ssrf_train_bin_path,
    validate_ps_iplus_iminus,
)
from combine_bin_events_simple import combine_bin_events
from combine_spectrum_rows import combine_spectrum_row_dirs
from combine_spectrum_train import combine_spectrum_shards
from flatten_spectrum_rows import flatten_one_bin, flatten_unmanipulated_rows
from bin_setup import equilibrium_lineshape, generate_unmanipulated_cube, get_shape_params
from common import (
    AFP_N_RELAX,
    BURN_BIN_ARRAY_END,
    BURN_BIN_ARRAY_START,
    BURN_BIN_CHOICES,
    BURN_R_MAX,
    BURN_R_MIN,
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
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    SPECTRUM_DENSE_BIN_STRIDE,
    burn_steps_grid,
    effective_afp_step_subsample,
    gamma_rf_grid,
    is_burn_bin,
    is_dense_spectrum_bin,
)
from model_bridge import build_spin1_model, configure_ssrf_burn
from ssrf_bin_traj import run_one_bin as run_ssrf_bin
from ssrf_bin_traj import run_one_bin_spectrum as run_ssrf_spectrum_bin
from ssrf_bin_traj import run_one_polarization as run_ssrf_one
from ssrf_realtime_v2.rate_equations_realtime import (
    voigt_burn_recovery_param_snapshot,
)
from unmanipulated_bin_lineshape import run_one_bin as run_unmanip_bin
from unmanipulated_bin_lineshape import save_unmanip_bin, unmanip_bin_path

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


def test_spectrum_train_grid_defaults() -> None:
    g = gamma_rf_grid()
    s = burn_steps_grid()
    np.testing.assert_allclose(g, np.array([5.0, 7.5, 10.0], dtype=float))
    np.testing.assert_array_equal(s, np.array([20, 40, 60, 80, 100], dtype=np.int32))
    assert BURN_R_MIN == -3.0 and BURN_R_MAX == 3.0
    assert int(BURN_BIN_CHOICES.size) == 250
    assert BURN_BIN_ARRAY_START == 125 and BURN_BIN_ARRAY_END == 374
    assert AFP_N_RELAX == 0
    assert effective_afp_step_subsample(0, 50) == 1
    assert effective_afp_step_subsample(5000, 50) == 50
    assert is_burn_bin(250)
    assert not is_burn_bin(0)
    # First burn bin is dense; next is MC for stride 5.
    assert is_dense_spectrum_bin(int(BURN_BIN_CHOICES[0]), stride=SPECTRUM_DENSE_BIN_STRIDE)
    assert not is_dense_spectrum_bin(int(BURN_BIN_CHOICES[1]), stride=SPECTRUM_DENSE_BIN_STRIDE)


def test_afp_instant_flip_keeps_single_spectrum_step() -> None:
    traj = run_afp_one(3, 0.3, n_relax=0, capture_spectrum=True, num_bins=32)
    assert int(traj["n_steps"]) == 1
    assert traj["ps_full"] is not None
    assert np.asarray(traj["ps_full"]).shape == (1, 32)
    spec = run_afp_spectrum_bin(
        3,
        p_values=SMOKE_P[:1],
        num_bins=32,
        n_relax=0,
        step_subsample=50,
        unmanip_fraction=0.0,
    )
    assert int(spec["step_subsample"]) == 1
    assert int(spec["n_steps"][0]) == 1


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


def test_ssrf_full_spectrum_capture() -> None:
    traj = run_ssrf_one(
        SMOKE_BIN,
        0.3,
        gamma_rf=1.5,
        n_steps=10,
        capture_spectrum=True,
    )
    assert not traj["skipped"]
    ps_full = np.asarray(traj["ps_full"], dtype=float)
    assert ps_full.shape == (int(traj["n_steps"]), NUM_BINS)
    ip_full = np.asarray(traj["iplus_full"], dtype=float)
    im_full = np.asarray(traj["iminus_full"], dtype=float)
    assert np.max(np.abs(ip_full + im_full - ps_full)) < 1e-8


def test_afp_full_spectrum_capture() -> None:
    traj = run_afp_one(
        SMOKE_BIN,
        0.3,
        n_relax=20,
        capture_spectrum=True,
    )
    ps_full = np.asarray(traj["ps_full"], dtype=float)
    assert ps_full.shape == (int(traj["n_steps"]), NUM_BINS)
    assert np.all(np.isfinite(ps_full))


def test_combine_spectrum_train_smoke(tmp_path: Path) -> None:
    n_bins = 8
    smoke_bin = 3
    ssrf_dir = tmp_path / "ssrf_spec"
    afp_dir = tmp_path / "afp_spec"
    unmanip_dir = tmp_path / "unmanip"
    ssrf_dir.mkdir()
    afp_dir.mkdir()
    unmanip_dir.mkdir()

    ssrf_res = run_ssrf_spectrum_bin(
        smoke_bin,
        p_values=SMOKE_P[:1],
        gamma_values=np.array([1.0], dtype=float),
        steps_values=np.array([10], dtype=np.int32),
        num_bins=n_bins,
        unmanip_fraction=0.0,
    )
    save_ssrf_spectrum_shard(ssrf_res, ssrf_shard_path(ssrf_dir, smoke_bin))

    afp_res = run_afp_spectrum_bin(
        smoke_bin,
        p_values=SMOKE_P[:1],
        num_bins=n_bins,
        n_relax=20,
        unmanip_fraction=0.0,
    )
    save_afp_spectrum_shard(afp_res, afp_shard_path(afp_dir, smoke_bin))

    shape = get_shape_params()
    cube = generate_unmanipulated_cube(
        num_bins=n_bins,
        p_min=float(SMOKE_P.min()),
        p_max=float(SMOKE_P.max()),
        p_step=0.3,
        shape_params=shape,
    )
    for bin_idx in range(n_bins):
        save_unmanip_bin(
            bin_idx,
            p_values=cube["p_values"],
            ps=cube["ps"][:, bin_idx],
            iplus=cube["iplus"][:, bin_idx],
            iminus=cube["iminus"][:, bin_idx],
            amp=cube["amp"][:, bin_idx],
            R=float(cube["R"][bin_idx]),
            path=unmanip_bin_path(unmanip_dir, bin_idx),
            p_min=float(SMOKE_P.min()),
            p_max=float(SMOKE_P.max()),
            p_step=0.3,
            num_bins=n_bins,
            shape_params=shape,
        )

    out = tmp_path / "spectrum_train.npz"
    result = combine_spectrum_shards(
        ssrf_dir,
        afp_dir,
        out,
        unmanip_dir=unmanip_dir,
        num_bins=n_bins,
        strict=False,
    )
    assert result["n_samples"] > 0
    assert result["n_unmanip"] > 0
    assert Path(out).is_file()
    with np.load(out, allow_pickle=False) as data:
        source = np.asarray(data["source"], dtype=np.uint8)
        ps = np.asarray(data["ps"], dtype=float)
        assert np.any(source == SOURCE_UNMANIP)
        assert ps.shape[1] == n_bins


def test_combine_spectrum_train_streaming_matches_single_file(tmp_path: Path) -> None:
    n_bins = 8
    smoke_bin = 3
    ssrf_dir = tmp_path / "ssrf_spec"
    afp_dir = tmp_path / "afp_spec"
    unmanip_dir = tmp_path / "unmanip"
    ssrf_dir.mkdir()
    afp_dir.mkdir()
    unmanip_dir.mkdir()

    ssrf_res = run_ssrf_spectrum_bin(
        smoke_bin,
        p_values=SMOKE_P[:1],
        gamma_values=np.array([1.0], dtype=float),
        steps_values=np.array([10], dtype=np.int32),
        num_bins=n_bins,
        unmanip_fraction=0.0,
    )
    save_ssrf_spectrum_shard(ssrf_res, ssrf_shard_path(ssrf_dir, smoke_bin))

    afp_res = run_afp_spectrum_bin(
        smoke_bin,
        p_values=SMOKE_P[:1],
        num_bins=n_bins,
        n_relax=20,
        unmanip_fraction=0.0,
    )
    save_afp_spectrum_shard(afp_res, afp_shard_path(afp_dir, smoke_bin))

    shape = get_shape_params()
    cube = generate_unmanipulated_cube(
        num_bins=n_bins,
        p_min=float(SMOKE_P.min()),
        p_max=float(SMOKE_P.max()),
        p_step=0.3,
        shape_params=shape,
    )
    for bin_idx in range(n_bins):
        save_unmanip_bin(
            bin_idx,
            p_values=cube["p_values"],
            ps=cube["ps"][:, bin_idx],
            iplus=cube["iplus"][:, bin_idx],
            iminus=cube["iminus"][:, bin_idx],
            amp=cube["amp"][:, bin_idx],
            R=float(cube["R"][bin_idx]),
            path=unmanip_bin_path(unmanip_dir, bin_idx),
            p_min=float(SMOKE_P.min()),
            p_max=float(SMOKE_P.max()),
            p_step=0.3,
            num_bins=n_bins,
            shape_params=shape,
        )

    single_out = tmp_path / "single" / "spectrum_train.npz"
    single_result = combine_spectrum_shards(
        ssrf_dir,
        afp_dir,
        single_out,
        unmanip_dir=unmanip_dir,
        num_bins=n_bins,
        strict=False,
        shard_size=0,
    )

    shard_out_dir = tmp_path / "sharded"
    shard_out_dir.mkdir()
    shard_result = combine_spectrum_shards(
        ssrf_dir,
        afp_dir,
        shard_out_dir / "spectrum_train.npz",
        unmanip_dir=unmanip_dir,
        num_bins=n_bins,
        strict=False,
        shard_size=3,
    )

    assert shard_result["n_samples"] == single_result["n_samples"]
    assert shard_result["n_ssrf"] == single_result["n_ssrf"]
    assert shard_result["n_afp"] == single_result["n_afp"]
    assert shard_result["n_unmanip"] == single_result["n_unmanip"]
    assert shard_result["n_filtered_empty"] == single_result["n_filtered_empty"]

    manifest_path = shard_out_dir / "spectrum_train_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["n_samples"] == single_result["n_samples"]
    assert sum(manifest["shard_row_counts"]) == manifest["n_samples"]
    assert len(manifest["shard_files"]) == manifest["n_shards"]
    for shard_file, count in zip(manifest["shard_files"], manifest["shard_row_counts"]):
        assert (shard_out_dir / shard_file).is_file()
        assert count <= 3

    with np.load(single_out, allow_pickle=False) as data:
        single_ps = np.asarray(data["ps"], dtype=float)
        single_source = np.asarray(data["source"], dtype=np.uint8)

    sharded_ps_parts = []
    sharded_source_parts = []
    for shard_file in manifest["shard_files"]:
        with np.load(shard_out_dir / shard_file, allow_pickle=False) as data:
            sharded_ps_parts.append(np.asarray(data["ps"], dtype=float))
            sharded_source_parts.append(np.asarray(data["source"], dtype=np.uint8))
    sharded_ps = np.concatenate(sharded_ps_parts, axis=0)
    sharded_source = np.concatenate(sharded_source_parts, axis=0)

    assert sharded_ps.shape == single_ps.shape
    np.testing.assert_allclose(np.sort(single_ps.sum(axis=1)), np.sort(sharded_ps.sum(axis=1)))
    assert sorted(single_source.tolist()) == sorted(sharded_source.tolist())


def test_combine_spectrum_rows_smoke(tmp_path: Path) -> None:
    n_bins = 8
    smoke_bin = 3
    ssrf_dir = tmp_path / "ssrf_spec"
    afp_dir = tmp_path / "afp_spec"
    ssrf_rows_dir = tmp_path / "ssrf_rows"
    afp_rows_dir = tmp_path / "afp_rows"
    unmanip_rows_dir = tmp_path / "unmanip_rows"
    unmanip_dir = tmp_path / "unmanip"
    for d in (ssrf_dir, afp_dir, ssrf_rows_dir, afp_rows_dir, unmanip_rows_dir, unmanip_dir):
        d.mkdir()

    ssrf_res = run_ssrf_spectrum_bin(
        smoke_bin,
        p_values=SMOKE_P[:1],
        gamma_values=np.array([1.0], dtype=float),
        steps_values=np.array([10], dtype=np.int32),
        num_bins=n_bins,
        unmanip_fraction=0.0,
    )
    save_ssrf_spectrum_shard(ssrf_res, ssrf_spectrum_shard_path(ssrf_dir, smoke_bin))

    afp_res = run_afp_spectrum_bin(
        smoke_bin,
        p_values=SMOKE_P[:1],
        num_bins=n_bins,
        n_relax=20,
        unmanip_fraction=0.0,
    )
    save_afp_spectrum_shard(afp_res, afp_spectrum_shard_path(afp_dir, smoke_bin))

    shape = get_shape_params()
    cube = generate_unmanipulated_cube(
        num_bins=n_bins,
        p_min=float(SMOKE_P.min()),
        p_max=float(SMOKE_P.max()),
        p_step=0.3,
        shape_params=shape,
    )
    for bin_idx in range(n_bins):
        save_unmanip_bin(
            bin_idx,
            p_values=cube["p_values"],
            ps=cube["ps"][:, bin_idx],
            iplus=cube["iplus"][:, bin_idx],
            iminus=cube["iminus"][:, bin_idx],
            amp=cube["amp"][:, bin_idx],
            R=float(cube["R"][bin_idx]),
            path=unmanip_bin_path(unmanip_dir, bin_idx),
            p_min=float(SMOKE_P.min()),
            p_max=float(SMOKE_P.max()),
            p_step=0.3,
            num_bins=n_bins,
            shape_params=shape,
        )

    flatten_one_bin("ssrf", smoke_bin, shard_dir=ssrf_dir, output_dir=ssrf_rows_dir)
    flatten_one_bin("afp", smoke_bin, shard_dir=afp_dir, output_dir=afp_rows_dir)
    flatten_unmanipulated_rows(
        unmanip_dir=unmanip_dir,
        output_dir=unmanip_rows_dir,
        num_bins=n_bins,
        strict=False,
    )

    direct_out = tmp_path / "direct" / "spectrum_train.npz"
    direct_result = combine_spectrum_shards(
        ssrf_dir,
        afp_dir,
        direct_out,
        unmanip_dir=unmanip_dir,
        num_bins=n_bins,
        strict=False,
    )

    rows_out = tmp_path / "rows" / "spectrum_train.npz"
    rows_result = combine_spectrum_row_dirs(
        ssrf_rows_dir,
        afp_rows_dir,
        rows_out,
        unmanip_rows_dir=unmanip_rows_dir,
        num_bins=n_bins,
        strict=False,
    )

    assert rows_result["n_ssrf"] == direct_result["n_ssrf"]
    assert rows_result["n_afp"] == direct_result["n_afp"]
    assert rows_result["n_unmanip"] == direct_result["n_unmanip"]
    assert rows_result["n_samples"] == direct_result["n_samples"]
    assert Path(rows_out).is_file()


def test_combine_bin_events_simple_smoke(tmp_path: Path) -> None:
    """Per-bin trajectory shards sharing one combo grid -> stacked into dense num_bins-wide rows."""
    n_bins = 8
    bin_a, bin_b = 2, 3
    other_bins = [b for b in range(n_bins) if b not in (bin_a, bin_b)]
    ssrf_dir = tmp_path / "ssrf_shards"
    afp_dir = tmp_path / "afp_shards"
    unmanip_dir = tmp_path / "unmanip"
    for d in (ssrf_dir, afp_dir, unmanip_dir):
        d.mkdir()

    ssrf_shards = {}
    afp_shards = {}
    for b in (bin_a, bin_b):
        ssrf_res = run_ssrf_bin(
            b,
            p_values=SMOKE_P,
            gamma_values=np.array([1.0], dtype=float),
            steps_values=np.array([5], dtype=np.int32),
            num_bins=n_bins,
            rf_mode=RF_MODE_PHYSICAL_VOIGT,
        )
        save_ssrf_shard(ssrf_res, ssrf_shard_path(ssrf_dir, b))
        ssrf_shards[b] = ssrf_res

        afp_res = run_afp_bin(
            b,
            p_values=SMOKE_P,
            num_bins=n_bins,
            n_relax=5,
        )
        save_afp_shard(afp_res, afp_shard_path(afp_dir, b))
        afp_shards[b] = afp_res

    shape = get_shape_params()
    cube = generate_unmanipulated_cube(
        num_bins=n_bins,
        p_min=float(SMOKE_P.min()),
        p_max=float(SMOKE_P.max()),
        p_step=0.3,
        shape_params=shape,
    )
    for bin_idx in range(n_bins):
        save_unmanip_bin(
            bin_idx,
            p_values=cube["p_values"],
            ps=cube["ps"][:, bin_idx],
            iplus=cube["iplus"][:, bin_idx],
            iminus=cube["iminus"][:, bin_idx],
            amp=cube["amp"][:, bin_idx],
            R=float(cube["R"][bin_idx]),
            path=unmanip_bin_path(unmanip_dir, bin_idx),
            p_min=float(SMOKE_P.min()),
            p_max=float(SMOKE_P.max()),
            p_step=0.3,
            num_bins=n_bins,
            shape_params=shape,
        )

    out = tmp_path / "combined" / "spectrum_train.npz"
    result = combine_bin_events(
        ssrf_dir,
        afp_dir,
        out,
        unmanip_dir=unmanip_dir,
        num_bins=n_bins,
        strict=False,
    )
    assert Path(result["output"]).is_file()
    assert result["n_samples"] > 0
    assert result["n_ssrf"] > 0
    assert result["n_afp"] > 0
    assert result["n_unmanip"] == cube["p_values"].size
    # bin_a and bin_b have shards; the other 6 bins are missing (non-strict).
    assert result["n_missing_ssrf"] == len(other_bins)
    assert result["n_missing_afp"] == len(other_bins)
    assert result["n_missing_unmanip"] == 0

    with np.load(result["output"], allow_pickle=False) as data:
        ps = np.asarray(data["ps"])
        iplus = np.asarray(data["iplus"])
        iminus = np.asarray(data["iminus"])
        source = np.asarray(data["source"])
        center_bin = np.asarray(data["center_bin"])

    assert ps.shape == (result["n_samples"], n_bins)
    # Stacking, not per-event embedding: center_bin is a sentinel (-1) since a
    # reconstructed spectrum row isn't tied to one physical burn location.
    assert np.all(center_bin[source != SOURCE_UNMANIP] == -1)

    # Missing bins (no shard at all) must be exactly 0 for every ssRF/AFP row.
    ssrf_mask = source == SOURCE_SSRF
    afp_mask = source == SOURCE_AFP
    assert np.all(ps[np.ix_(ssrf_mask, other_bins)] == 0.0)
    assert np.all(ps[np.ix_(afp_mask, other_bins)] == 0.0)

    # bin_a's own column should reproduce bin_a's directly-computed ssRF trajectory values.
    direct_ps = ssrf_shards[bin_a]["ps"][~ssrf_shards[bin_a]["skipped"]]
    direct_vals = sorted(v for row in direct_ps for v in row.tolist())
    got_vals = sorted(v for v in ps[ssrf_mask, bin_a].tolist() if v != 0.0)
    assert got_vals == pytest.approx(direct_vals, rel=1e-4)

    # Every stored ps/iplus/iminus triple must stay physically consistent
    # (ps == iplus + iminus), proving the three arrays never got misaligned
    # while stacking bins / concatenating sources -- combine_bin_events()
    # already asserts this internally (validate=True by default).
    np.testing.assert_allclose(ps, iplus + iminus, atol=1e-3, rtol=1e-3)
    # bin_a's own iplus/iminus columns must also reproduce the source shard exactly.
    direct_iplus = sorted(
        v for row in ssrf_shards[bin_a]["iplus"][~ssrf_shards[bin_a]["skipped"]] for v in row.tolist()
    )
    got_iplus = sorted(v for v in iplus[ssrf_mask, bin_a].tolist() if v != 0.0)
    assert got_iplus == pytest.approx(direct_iplus, rel=1e-4)


def test_validate_ps_iplus_iminus_catches_misalignment() -> None:
    """validate_ps_iplus_iminus must pass on consistent rows and fail loudly on scrambled ones."""
    n, nb = 5, 4
    rng = np.random.default_rng(0)
    iplus = rng.normal(size=(n, nb))
    iminus = rng.normal(size=(n, nb))
    ps = iplus + iminus
    rows = {"ps": ps, "iplus": iplus, "iminus": iminus}
    validate_ps_iplus_iminus(rows, label="consistent")

    scrambled = {"ps": np.roll(ps, shift=1, axis=0), "iplus": iplus, "iminus": iminus}
    with pytest.raises(ValueError, match="ps != iplus \\+ iminus"):
        validate_ps_iplus_iminus(scrambled, label="scrambled")


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
