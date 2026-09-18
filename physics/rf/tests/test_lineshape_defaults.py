"""Simulated lineshape uses ssRF-beta analytic Pake defaults, not Dulya fit_params."""
import ast
import sys
from pathlib import Path
import numpy as np
from physics.rf.model import Spin1Params

REPO_ROOT = Path(__file__).resolve().parents[3]
RIVANNA = REPO_ROOT / "Data_Creation" / "rivanna"


def test_spin1_params_match_ssrf_beta_lineshape_defaults():
    p = Spin1Params()
    assert p.n_bins == 701
    assert p.r_min == -3.0
    assert p.r_max == 3.0
    assert p.line_gamma == 0.05
    assert p.line_asym == 0.04


def test_no_ssrf_beta_imports():
    import physics.rf.pulse_program as pp
    import physics.rf.ideal_model as im
    import physics.rf.profile_control as pc
    import physics.rf.model as model
    sys.path.insert(0, str(REPO_ROOT))
    import ml.rf_profile_rl as rl
    for mod in (pp, im, pc, model, rl):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif node.module:
                    names = [node.module]
                for name in names:
                    assert "ssRF-beta" not in name
                    assert "ssrf-beta" not in name.lower().replace("_", "-")


def test_opt_q_equilibrium_does_not_load_fit_params():
    sys.path.insert(0, str(REPO_ROOT))
    import ml.opt_q as opt_q
    f = np.linspace(opt_q.F_MIN, opt_q.F_MAX, opt_q.NUM_BINS)
    ip, im = opt_q.equilibrium_spin1_intensities(0.45, f)
    assert ip.shape == (701,)
    assert im.shape == (701,)
    assert np.all(np.isfinite(ip))
    assert opt_q.NUM_BINS == 701
    assert opt_q.F_MIN == -3.0
    assert opt_q.F_MAX == 3.0


def test_bin_setup_equilibrium_is_pake_default(monkeypatch):
    sys.path.insert(0, str(RIVANNA))
    import bin_setup

    assert bin_setup.LINESHape_MODEL == "pake_default"
    assert bin_setup.PAKE_GAMMA == 0.05
    assert bin_setup.PAKE_ASYM == 0.04

    def boom():
        raise AssertionError("get_shape_params / fit_params.json must not be used")

    monkeypatch.setattr(bin_setup, "get_shape_params", boom)
    f = np.linspace(-3.0, 3.0, 51)
    ps, ip, im = bin_setup.equilibrium_lineshape(0.45, f)
    assert ip.shape == f.shape
    assert np.all(np.isfinite(ps))
