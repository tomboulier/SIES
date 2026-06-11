"""Integration tests: numerical CGPT vs closed forms and invariances."""

import numpy as np
import pytest

from sies.asymptotics import (
    contrast,
    disk_cgpt,
    ellipse_cgpt,
    make_block_matrix,
    theoretical_cgpt,
    transform_cgpt,
)
from sies.shapes import Flower


def test_contrast_values():
    assert contrast(3.0) == pytest.approx(1.0)
    np.testing.assert_allclose(contrast([3.0, 2.0]), [1.0, 1.5])
    lam = contrast(3.0, 1.0, freq=0.5)
    assert np.iscomplexobj(lam)


def test_contrast_validation():
    with pytest.raises(ValueError, match="Conductivity"):
        contrast(1.0)
    with pytest.raises(ValueError, match="Conductivity"):
        contrast(-2.0)
    with pytest.raises(ValueError, match="Frequency"):
        contrast(3.0, 1.0, freq=-1.0)


def test_cgpt_matches_exact_disk(disk):
    order = 3
    computed = theoretical_cgpt(disk, contrast(3.0), order)
    exact = disk_cgpt(order, disk.diameter / 2, 3.0)
    np.testing.assert_allclose(computed.real, exact, atol=1e-8)


def test_cgpt_matches_exact_ellipse(ellipse):
    order = 4
    computed = theoretical_cgpt(ellipse, contrast(5.0), order)
    exact = ellipse_cgpt(order, 1.0, 0.5, 5.0)
    np.testing.assert_allclose(computed.real, exact, atol=1e-8)


def test_cgpt_insulating_contrast(disk):
    # cnd < 1 gives a negative-definite first-order CGPT (polarization
    # tensor) for an insulating inclusion.
    computed = theoretical_cgpt(disk, contrast(0.2), 1)
    eigenvalues = np.linalg.eigvalsh(computed.real)
    assert np.all(eigenvalues < 0)


def test_cgpt_extreme_contrast_uses_augmented_system(disk):
    # lambda = 1/2 corresponds to a perfectly conducting inclusion
    # (cnd -> infinity): the augmented system must remain solvable and
    # match the large-conductivity limit of the disk formula.
    computed = theoretical_cgpt(disk, 0.5, 2)
    exact = disk_cgpt(2, disk.diameter / 2, 1e12)
    np.testing.assert_allclose(computed.real, exact, atol=1e-6)


def test_cgpt_of_two_disjoint_inclusions(disk):
    # CGPT of a union of two distant disks: close to the sum of the CGPTs
    # of the translated individual disks (interaction is weak).
    left = disk + np.array([-3.0, 0.0])
    right = disk + np.array([3.0, 0.0])
    lam = contrast([4.0, 4.0])
    combined = theoretical_cgpt([left, right], lam, 1)

    single = theoretical_cgpt(disk, contrast(4.0), 1)
    np.testing.assert_allclose(combined.real, 2 * single.real, rtol=1e-2, atol=1e-10)


def test_overlapping_inclusions_rejected(disk):
    with pytest.raises(ValueError, match="disjoint"):
        make_block_matrix([disk, disk + np.array([0.1, 0.0])])


def test_missing_contrast_rejected(disk):
    far = disk + np.array([5.0, 0.0])
    with pytest.raises(ValueError, match="each inclusion"):
        theoretical_cgpt([disk, far], contrast(3.0), 2)


def test_cgpt_symmetry(ellipse):
    computed = theoretical_cgpt(ellipse, contrast(3.0), 4).real
    np.testing.assert_allclose(computed, computed.T, atol=1e-8)


def test_cgpt_transform_consistency():
    # The CGPT of a transformed shape equals the transformed CGPT: the
    # translation matrix is lower triangular so the relation is exact at
    # every truncation order.
    order = 3
    flower = Flower(1.0, 1.0, 256)
    lam = contrast(3.0)
    base = theoretical_cgpt(flower, lam, order).real

    translation = np.array([0.4, -0.3])
    scaling, rotation = 0.8, 0.5
    moved = (flower.rotate(rotation) * scaling) + translation
    target = theoretical_cgpt(moved, lam, order).real

    predicted = transform_cgpt(base, translation, scaling, rotation)
    np.testing.assert_allclose(predicted, target, rtol=1e-5, atol=1e-8)


def test_block_matrix_reuse_across_frequencies(ellipse):
    order = 2
    blocks = make_block_matrix([ellipse])
    lam = contrast(3.0, 1.0, freq=0.25)
    with_reuse = theoretical_cgpt(ellipse, lam, order, block_matrix=blocks)
    without = theoretical_cgpt(ellipse, lam, order)
    np.testing.assert_allclose(with_reuse, without)
    assert np.iscomplexobj(with_reuse)
