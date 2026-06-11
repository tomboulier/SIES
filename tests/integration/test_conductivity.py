"""Integration tests of the conductivity forward and inverse problems."""

import numpy as np
import pytest

from sies.acquisition import Coincided, Concentric, ViewMode
from sies.asymptotics import contrast, theoretical_cgpt
from sies.pde import ConductivityR2
from sies.shapes import Ellipse, Flower


@pytest.fixture
def pde(full_view_cfg):
    target = (Flower(1.0, 1.0, 256) * 0.5) + np.array([0.2, 0.1])
    return ConductivityR2(target, 3.0, 0.0, full_view_cfg)


def test_constructor_validation(full_view_cfg, disk):
    with pytest.raises(ValueError, match="Conductivity"):
        ConductivityR2(disk, 1.0, 0.0, full_view_cfg)
    with pytest.raises(ValueError, match="Permittivity"):
        ConductivityR2(disk, 3.0, -1.0, full_view_cfg)
    with pytest.raises(ValueError, match="each inclusion"):
        ConductivityR2([disk, disk + np.array([5.0, 0.0])], 3.0, 0.0, full_view_cfg)
    with pytest.raises(ValueError, match="number of boundary points"):
        small = Ellipse(1.0, 1.0, 64) + np.array([5.0, 0.0])
        ConductivityR2([disk, small], [3.0, 3.0], [0.0, 0.0], full_view_cfg)


def test_msr_reciprocity(pde):
    # With coincided sources and receivers the MSR matrix is symmetric.
    data = pde.simulate_data(0.0)
    msr = data.msr[0]
    np.testing.assert_allclose(msr, msr.T, atol=1e-12)
    assert data.freqs == [0.0]


def test_msr_decays_with_distance(disk):
    # The perturbation decays like 1/|x| away from the inclusion.
    near = ConductivityR2(disk, 3.0, 0.0, Coincided(np.zeros(2), 2.0, 20))
    far = ConductivityR2(disk, 3.0, 0.0, Coincided(np.zeros(2), 20.0, 20))
    near_msr = near.simulate_data(0.0).msr[0]
    far_msr = far.simulate_data(0.0).msr[0]
    assert np.abs(far_msr).max() < 1e-2 * np.abs(near_msr).max()


def test_multifrequency_data_is_complex(disk, full_view_cfg):
    pde = ConductivityR2(disk, 3.0, 1.0, full_view_cfg)
    data = pde.simulate_data([0.0, 0.5])
    assert len(data.msr) == 2
    assert not np.iscomplexobj(data.msr[0])
    assert np.iscomplexobj(data.msr[1])


def test_reconstruct_cgpt_noiseless(pde):
    # Least-squares reconstruction recovers the theoretical CGPT.
    order = 3
    data = pde.simulate_data(0.0)
    result = pde.reconstruct_cgpt(data.msr[0], order)

    expected = theoretical_cgpt(pde.inclusions[0], contrast(3.0), order).real
    np.testing.assert_allclose(result.cgpt[0], expected, atol=1e-10)
    # The residual is dominated by the truncation of the orders > 3
    # still present in the data, not by the solver.
    assert result.relative_residual[0] < 0.05


def test_reconstruct_cgpt_analytic_equals_pinv(pde):
    order = 3
    data = pde.simulate_data(0.0)
    lsq = pde.reconstruct_cgpt(data.msr[0], order)
    analytic = pde.reconstruct_cgpt_analytic(data.msr[0], order)
    np.testing.assert_allclose(analytic.cgpt[0], lsq.cgpt[0], rtol=1e-6, atol=1e-10)


def test_reconstruct_accepts_list(pde):
    data = pde.simulate_data([0.0, 0.0])
    result = pde.reconstruct_cgpt_analytic(data.msr, 2)
    assert len(result.cgpt) == 2
    np.testing.assert_allclose(result.cgpt[0], result.cgpt[1])


def test_analytic_requires_equispaced(disk):
    view = ViewMode(nb_arcs=2, aperture=np.pi / 4)
    cfg = Concentric(np.zeros(2), 2.0, 10, 2.0, 10, view)
    pde = ConductivityR2(disk, 3.0, 0.0, cfg)
    data = pde.simulate_data(0.0)
    with pytest.raises(ValueError, match="equispaced"):
        pde.reconstruct_cgpt_analytic(data.msr[0], 2)


def test_analytic_caps_order(pde):
    data = pde.simulate_data(0.0)
    result = pde.reconstruct_cgpt_analytic(data.msr[0], 1000)
    # 50 sources: maximum resolvable order is 24
    assert result.cgpt[0].shape == (48, 48)


def test_grouped_operator_not_implemented(disk):
    view = ViewMode(nb_arcs=2, aperture=np.pi / 4)
    cfg = Concentric(np.zeros(2), 2.0, 10, 2.0, 10, view, grouped=True)
    pde = ConductivityR2(disk, 3.0, 0.0, cfg)
    with pytest.raises(NotImplementedError):
        pde.make_linear_operator(2)


def test_add_white_noise_levels(pde, rng):
    data = pde.simulate_data(0.0)
    noisy = pde.add_white_noise(data, 0.1, rng)
    assert len(noisy.msr_noisy) == 1
    assert noisy.noise_sigma[0] > 0
    relative = np.linalg.norm(noisy.msr_noisy[0] - data.msr[0]) / np.linalg.norm(data.msr[0])
    assert 0.05 < relative < 0.2


def test_add_white_noise_complex(disk, full_view_cfg, rng):
    pde = ConductivityR2(disk, 3.0, 1.0, full_view_cfg)
    data = pde.simulate_data(0.5)
    noisy = pde.add_white_noise(data, 0.1, rng)
    assert np.iscomplexobj(noisy.msr_noisy[0])


def test_plot(pde):
    ax = pde.plot()
    assert len(ax.lines) >= 4
