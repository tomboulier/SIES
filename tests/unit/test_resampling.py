"""Unit tests of the curve resampling utilities."""

import numpy as np
import pytest

from sies.shapes import resample_curve


def _circle(n):
    theta = 2 * np.pi * np.arange(n) / n
    return np.vstack([np.cos(theta), np.sin(theta)]), theta


def test_resample_circle_recovers_geometry():
    points0, theta0 = _circle(128)
    points, tvec, avec, normal = resample_curve(points0, theta0, 64)

    np.testing.assert_allclose(np.linalg.norm(points, axis=0), 1.0, atol=1e-6)
    # tangent of unit circle has unit norm, acceleration points inward
    np.testing.assert_allclose(np.linalg.norm(tvec, axis=0), 1.0, atol=1e-4)
    np.testing.assert_allclose(avec, -points, atol=1e-3)
    np.testing.assert_allclose(normal, points, atol=1e-6)


def test_resample_with_box_rescales():
    points0, theta0 = _circle(128)
    points, *_ = resample_curve(points0, theta0, 128, box=(4.0, 2.0))
    assert points[0].max() - points[0].min() == pytest.approx(4.0, rel=1e-3)
    assert points[1].max() - points[1].min() == pytest.approx(2.0, rel=1e-3)


def test_resample_with_downsampling_smooths():
    points0, theta0 = _circle(256)
    points, *_ = resample_curve(points0, theta0, 256, downsample=4)
    np.testing.assert_allclose(np.linalg.norm(points, axis=0), 1.0, atol=1e-4)


def test_resample_rejects_bad_downsampling():
    points0, theta0 = _circle(32)
    with pytest.raises(ValueError, match="positive"):
        resample_curve(points0, theta0, 32, downsample=0)


def test_resample_rejects_degenerate_curve():
    theta0 = 2 * np.pi * np.arange(32) / 32
    flat = np.vstack([np.cos(theta0), np.zeros(32)])
    with pytest.raises(ValueError, match="extent"):
        resample_curve(flat, theta0, 32, box=(2.0, 1.0))
