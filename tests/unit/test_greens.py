"""Unit tests of the 2D Green's function and its derivatives."""

import numpy as np
import pytest

from sies.greens import green2d, green2d_dn, green2d_grad


def test_green2d_values():
    x = np.array([[1.0], [0.0]])
    y = np.array([[0.0], [0.0]])
    # G(x) = log|x| / (2 pi); |x - y| = 1 so G = 0
    assert green2d(x, y) == pytest.approx(0.0)

    x = np.array([[np.e], [0.0]])
    assert green2d(x, y)[0, 0] == pytest.approx(1 / (2 * np.pi))


def test_green2d_symmetry():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5))
    y = rng.standard_normal((2, 7)) + 10
    np.testing.assert_allclose(green2d(x, y), green2d(y, x).T)


def test_green2d_grad_matches_finite_differences():
    x = np.array([[0.7], [-0.3]])
    y = np.array([[2.0], [1.0]])
    eps = 1e-6
    dx = np.array([[eps], [0.0]])
    dy = np.array([[0.0], [eps]])

    gx, gy = green2d_grad(x, y)
    fd_x = (green2d(x + dx, y) - green2d(x - dx, y)) / (2 * eps)
    fd_y = (green2d(x + dy, y) - green2d(x - dy, y)) / (2 * eps)

    assert gx[0, 0] == pytest.approx(fd_x[0, 0], rel=1e-6)
    assert gy[0, 0] == pytest.approx(fd_y[0, 0], rel=1e-6)


def test_green2d_dn_on_circle():
    # On a circle of radius R, <y - x, nu_y> = R(1 - cos) and |y - x|^2
    # = 2 R^2 (1 - cos) for x, y both on the circle... use an off-circle
    # source instead: compare against the directional finite difference.
    theta = 2 * np.pi * np.arange(8) / 8
    boundary = np.vstack([np.cos(theta), np.sin(theta)])
    normal = boundary.copy()  # unit circle: outward normal is radial
    source = np.array([[3.0], [1.0]])

    gn = green2d_dn(source, boundary, normal)
    assert gn.shape == (1, 8)

    eps = 1e-6
    fd = (green2d(source, boundary + eps * normal) - green2d(source, boundary - eps * normal)) / (
        2 * eps
    )
    np.testing.assert_allclose(gn, fd, rtol=1e-6, atol=1e-12)


def test_green2d_rejects_bad_shapes():
    with pytest.raises(ValueError, match="shape"):
        green2d(np.zeros((3, 4)), np.zeros((2, 4)))
