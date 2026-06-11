"""Unit tests of the layer potential operators."""

import numpy as np
import pytest

from sies.operators import KStar, SingleLayer, SingleLayerNormalDerivative
from sies.shapes import Ellipse


def test_kstar_constant_density_on_disk(disk):
    # On any boundary, K_D^*[1] integrates to 1/2 of the boundary measure;
    # on a disk the kernel is constant 1/(4 pi R) so each row sums to 1/2.
    kstar = KStar(disk)
    row_sums = kstar.matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.5, atol=1e-3)


def test_kstar_call_applies_matrix(disk):
    kstar = KStar(disk)
    density = np.ones(disk.nb_points)
    np.testing.assert_allclose(kstar(density), kstar.matrix @ density)


def test_single_layer_harmonicity(disk):
    # S_D[f] is harmonic away from D: check the mean value property of
    # the potential on a small circle far from the inclusion.
    density = np.cos(disk.theta)
    center = np.array([[5.0], [0.0]])
    theta = 2 * np.pi * np.arange(64) / 64
    circle = center + 0.1 * np.vstack([np.cos(theta), np.sin(theta)])

    mean_on_circle = SingleLayer.evaluate(disk, density, circle).mean()
    at_center = SingleLayer.evaluate(disk, density, center)[0]
    assert mean_on_circle == pytest.approx(at_center, abs=1e-10)


def test_single_layer_of_disk_constant_density(disk):
    # For the unit density on a circle of radius R: S[1](x) = R log|x| for
    # |x| > R (Newtonian potential of a circle).
    radius = disk.diameter / 2
    x = np.array([[2.0], [1.0]])
    expected = radius * np.log(np.linalg.norm(x))
    value = SingleLayer.evaluate(disk, np.ones(disk.nb_points), x)[0]
    assert value == pytest.approx(expected, rel=1e-6)


def test_single_layer_gradient_matches_finite_differences(disk):
    density = np.sin(disk.theta)
    x = np.array([[1.5], [0.5]])
    eps = 1e-6
    dx = np.array([[eps], [0.0]])
    dy = np.array([[0.0], [eps]])

    grad = SingleLayer.evaluate_gradient(disk, density, x)
    fd_x = (
        SingleLayer.evaluate(disk, density, x + dx) - SingleLayer.evaluate(disk, density, x - dx)
    ) / (2 * eps)
    fd_y = (
        SingleLayer.evaluate(disk, density, x + dy) - SingleLayer.evaluate(disk, density, x - dy)
    ) / (2 * eps)

    assert grad[0, 0] == pytest.approx(fd_x[0], rel=1e-5)
    assert grad[1, 0] == pytest.approx(fd_y[0], rel=1e-5)


def test_single_layer_on_own_boundary(disk):
    # On a circle of radius R, S[1] = R log R on the boundary: this
    # exercises the analytic treatment of the singular diagonal.
    operator = SingleLayer(disk)
    radius = disk.diameter / 2
    values = operator(np.ones(disk.nb_points))
    np.testing.assert_allclose(values, radius * np.log(radius), rtol=1e-3)


def test_single_layer_two_boundaries_matrix(disk):
    other = Ellipse(1.0, 0.5, 64) + np.array([5.0, 0.0])
    operator = SingleLayer(disk, other)
    assert operator.matrix.shape == (64, disk.nb_points)
    # Same as evaluating the kernel away from the diagonal
    density = np.ones(disk.nb_points)
    np.testing.assert_allclose(operator(density), SingleLayer.evaluate(disk, density, other.points))


def test_normal_derivative_rejects_same_boundary(disk):
    with pytest.raises(ValueError, match="jump"):
        SingleLayerNormalDerivative(disk, disk)


def test_normal_derivative_kernel_shape(disk):
    other = Ellipse(1.0, 0.5, 32) + np.array([5.0, 0.0])
    operator = SingleLayerNormalDerivative(disk, other)
    assert operator.matrix.shape == (32, disk.nb_points)
    assert operator.domain is disk
    assert operator.image is other
