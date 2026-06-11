"""Unit tests of the CGPT transformation rules."""

import numpy as np
import pytest

from sies.asymptotics import (
    cgpt_to_complex,
    complex_to_cgpt,
    split_cgpt,
    transform_ccgpt,
    transform_ccgpt_inverse,
    transform_cgpt,
)
from sies.asymptotics.transforms import join_cgpt


@pytest.fixture
def cgpt(rng):
    order = 3
    matrix = rng.standard_normal((2 * order, 2 * order))
    return (matrix + matrix.T) / 2  # CGPT matrices are symmetric


def test_split_join_roundtrip(cgpt):
    np.testing.assert_allclose(join_cgpt(*split_cgpt(cgpt)), cgpt)


def test_complex_conversion_roundtrip(cgpt):
    n1, n2 = cgpt_to_complex(cgpt)
    np.testing.assert_allclose(complex_to_cgpt(n1, n2), cgpt, atol=1e-12)


def test_complex_cgpt_symmetries(cgpt):
    n1, n2 = cgpt_to_complex(cgpt)
    np.testing.assert_allclose(n1, n1.T, atol=1e-12)  # N1 symmetric
    np.testing.assert_allclose(n2, n2.conj().T, atol=1e-12)  # N2 hermitian


def test_transform_inverse_roundtrip(cgpt):
    n1, n2 = cgpt_to_complex(cgpt)
    translation, scaling, rotation = 0.3 - 0.7j, 1.7, 0.4

    z1, z2 = transform_ccgpt(n1, n2, translation, scaling, rotation)
    back1, back2 = transform_ccgpt_inverse(z1, z2, translation, scaling, rotation)
    np.testing.assert_allclose(back1, n1, atol=1e-10)
    np.testing.assert_allclose(back2, n2, atol=1e-10)


def test_identity_transform(cgpt):
    np.testing.assert_allclose(transform_cgpt(cgpt, 0.0, 1.0, 0.0), cgpt, atol=1e-12)


def test_translation_as_vector_equals_complex(cgpt):
    moved_complex = transform_cgpt(cgpt, 0.5 + 0.25j)
    moved_vector = transform_cgpt(cgpt, np.array([0.5, 0.25]))
    np.testing.assert_allclose(moved_complex, moved_vector)


def test_scaling_acts_diagonally(cgpt):
    # Under pure scaling, the entry (m, n) of N1, N2 scales as s^(m + n).
    n1, n2 = cgpt_to_complex(cgpt)
    scaling = 1.3
    z1, z2 = transform_ccgpt(n1, n2, 0.0, scaling, 0.0)
    order = n1.shape[0]
    powers = scaling ** (np.arange(1, order + 1)[:, None] + np.arange(1, order + 1)[None, :])
    np.testing.assert_allclose(z1, n1 * powers, atol=1e-12)
    np.testing.assert_allclose(z2, n2 * powers, atol=1e-12)
