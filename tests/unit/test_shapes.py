"""Unit tests of the shape classes and their geometry."""

import numpy as np
import pytest

from sies.shapes import Banana, C2Boundary, Ellipse, Flower, Rectangle, Triangle

ALL_SHAPES = [
    Ellipse(1.0, 0.5, 256),
    Flower(1.0, 1.0, 256),
    Flower(1.0, 0.8, 256, nb_petals=4, epsilon=0.2, pertb=2),
    Triangle(1.0, np.pi / 3, 256),
    Rectangle(2.0, 1.0, 256),
    Banana(0.8, 0.2, [0.0, 0.0], [0.0, 10.0], 256),
]


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda s: s.name)
def test_normals_are_unit_and_orthogonal_to_tangents(shape):
    np.testing.assert_allclose(np.linalg.norm(shape.normal, axis=0), 1.0, atol=1e-12)
    inner = np.sum(shape.normal * shape.tvec, axis=0)
    np.testing.assert_allclose(inner, 0.0, atol=1e-10)


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda s: s.name)
def test_normals_point_outward(shape):
    # Points slightly along the normal must be farther from the center.
    outside = shape.points + 1e-3 * shape.normal
    d_out = np.linalg.norm(outside - shape.center_of_mass[:, None], axis=0)
    d_on = np.linalg.norm(shape.points - shape.center_of_mass[:, None], axis=0)
    assert np.mean(d_out > d_on) > 0.99


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda s: s.name)
def test_sampling_is_valid(shape):
    assert C2Boundary.check_sampling(shape.points)


def test_ellipse_area_and_perimeter():
    ellipse = Ellipse(1.0, 0.5, 512)
    assert ellipse.area == pytest.approx(np.pi * 0.5, rel=1e-6)
    # Ramanujan approximation of the ellipse perimeter
    a, b = 1.0, 0.5
    h = ((a - b) / (a + b)) ** 2
    perimeter = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
    assert ellipse.perimeter == pytest.approx(perimeter, rel=1e-5)


def test_rectangle_area():
    assert Rectangle(2.0, 1.0, 512).area == pytest.approx(2.0, rel=2e-2)


def test_triangle_area():
    side, angle = 1.0, np.pi / 3
    exact = 0.5 * side**2 * np.sin(angle)
    assert Triangle(side, angle, 512).area == pytest.approx(exact, rel=2e-2)


def test_ellipse_rejects_swapped_axes():
    with pytest.raises(ValueError, match="semi-major"):
        Ellipse(0.5, 1.0, 64)


def test_banana_rejects_swapped_axes():
    with pytest.raises(ValueError, match="semi-major"):
        Banana(0.2, 0.8, [0, 0], [0, 10], 64)


def test_flower_rejects_bad_exponent():
    with pytest.raises(ValueError, match="positive integer"):
        Flower(1.0, 1.0, 64, pertb=0)


def test_translation_moves_center():
    ellipse = Ellipse(1.0, 0.5, 128)
    moved = ellipse + np.array([1.0, 2.0])
    np.testing.assert_allclose(moved.center_of_mass, [1.0, 2.0])
    back = moved - np.array([1.0, 2.0])
    np.testing.assert_allclose(back.points, ellipse.points, atol=1e-14)
    # The original object is unchanged.
    np.testing.assert_allclose(ellipse.center_of_mass, [0.0, 0.0])


def test_scaling_scales_geometry_and_attributes():
    ellipse = Ellipse(1.0, 0.5, 128)
    scaled = ellipse * 2.0
    assert scaled.area == pytest.approx(4 * ellipse.area, rel=1e-10)
    assert scaled.axis_a == pytest.approx(2.0)
    assert scaled.axis_b == pytest.approx(1.0)
    assert (3.0 * Rectangle(2.0, 1.0, 128)).width == pytest.approx(6.0)
    assert (2.0 * Triangle(1.0, np.pi / 4, 128)).side == pytest.approx(2.0)


def test_scaling_rejects_nonpositive_factor():
    with pytest.raises(ValueError, match="positive"):
        Ellipse(1.0, 0.5, 64) * (-1.0)


def test_rotation_preserves_area_and_tracks_angle():
    ellipse = Ellipse(1.0, 0.5, 128)
    rotated = ellipse.rotate(np.pi / 3)
    assert rotated.area == pytest.approx(ellipse.area, rel=1e-12)
    assert rotated.phi == pytest.approx(np.pi / 3)
    # Rotating the unit normal field keeps it unit
    np.testing.assert_allclose(np.linalg.norm(rotated.normal, axis=0), 1.0, atol=1e-12)


def test_principal_direction_of_elongated_ellipse():
    ellipse = Ellipse(2.0, 0.5, 256)
    direction = ellipse.principal_direction
    assert direction[0] == pytest.approx(1.0, abs=1e-6)


def test_principal_direction_of_vertical_shape():
    # A near-vertical principal axis must not blow up (no arctan of a
    # division by zero) and the sign convention keeps it deterministic.
    tall = Ellipse(2.0, 0.5, 256).rotate(np.pi / 2)
    direction = tall.principal_direction
    assert abs(direction[1]) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    first_nonzero = direction[0] if abs(direction[0]) > 1e-12 else direction[1]
    assert first_nonzero > 0


def test_is_inside_and_disjoint():
    small = Ellipse(1.0, 1.0, 64) * 0.5
    far = small + np.array([10.0, 0.0])
    assert small.is_inside([0.1, 0.1])
    assert not small.is_inside([5.0, 5.0])
    assert small.is_disjoint(far)
    assert not small.is_disjoint(small * 1.5)


def test_diameter_and_box():
    ellipse = Ellipse(1.0, 0.5, 256)
    assert ellipse.diameter == pytest.approx(2.0, rel=1e-4)
    width, height = ellipse.box
    assert width == pytest.approx(2.0, rel=1e-4)
    assert height == pytest.approx(1.0, rel=1e-4)


def test_constructor_validates_shapes():
    good = np.zeros((2, 8))
    with pytest.raises(ValueError, match=r"\(2, n\)"):
        C2Boundary(np.zeros((3, 8)), good, good, good)
    with pytest.raises(ValueError, match="same shape"):
        C2Boundary(good, good, good, np.zeros((2, 9)))


def test_repr_and_plot():
    ellipse = Ellipse(1.0, 0.5, 64)
    assert "Ellipse" in repr(ellipse)
    ax = ellipse.plot()
    assert len(ax.lines) == 1


def test_cpoints_theta_sigma_consistency():
    square = Rectangle(1.0, 1.0, 128)
    assert square.name == "Square"
    assert square.nb_points == 128
    assert square.cpoints.shape == (128,)
    assert square.theta[0] == 0.0
    assert square.sigma.shape == (128,)
    # Corner smoothing shortens the perimeter slightly.
    assert square.perimeter == pytest.approx(4.0, rel=8e-2)


def test_circle_named_circle():
    assert Ellipse(1.0, 1.0, 64).name == "Circle"


def test_stokes_center_of_mass_of_translated_disk():
    disk = Ellipse(1.0, 1.0, 256) + np.array([2.0, -1.0])
    com = C2Boundary.stokes_center_of_mass(disk.points, disk.tvec, disk.normal)
    np.testing.assert_allclose(com, [2.0, -1.0], atol=1e-10)
