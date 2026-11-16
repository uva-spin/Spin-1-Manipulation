"""Simulated lineshape uses ssRF-beta analytic Pake defaults, not Dulya fit_params."""
import ast
import sys
from pathlib import Path
import numpy as np
import pytest
from physics.rf.lineshape import boltzmann_Q
from physics.rf.model import Spin1Model, Spin1Params

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_spin1_params_match_ssrf_beta_lineshape_defaults():
    p = Spin1Params()
    assert p.n_bins == 701
    assert p.r_min == -3.0
    assert p.r_max == 3.0
    assert p.line_gamma == 0.05
    assert p.line_asym == 0.04
    assert not hasattr(p, "display_cal")
    assert p.diffusion_enabled is True
    assert p.diffusion_scale == 5.0
    assert p.diffusion_overlap == "lorentzian"
    assert p.cross_branch_ratio == 1.0
    assert p.double_quantum_ratio == 0.10


def test_population_pq_matches_ssrf_beta():
    m = Spin1Model(Spin1Params(p0=0.45, q0=None, n_bins=101))
    pol = m.polarizations()
    assert pol["P"] == pytest.approx(0.45, rel=1e-8, abs=1e-8)
    assert pol["Q"] == pytest.approx(boltzmann_Q(0.45), rel=1e-8, abs=1e-8)
    assert not hasattr(m, "display_cal")
    ip, im, total = m.physical_intensities()
    assert float(np.sum(total)) == pytest.approx(0.45, rel=1e-8, abs=1e-8)
    assert float(np.sum(ip - im)) == pytest.approx(boltzmann_Q(0.45), rel=1e-8, abs=1e-8)


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
