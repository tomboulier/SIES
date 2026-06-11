"""Integration tests of the invariant shape descriptors."""

import numpy as np
import pytest

from sies.asymptotics import contrast, theoretical_cgpt
from sies.dictionary import ShapeDescriptor, ShapeDictionary, match_descriptor
from sies.shapes import Ellipse, Flower, Rectangle, Triangle


@pytest.fixture(scope="module")
def flower_cgpt():
    return theoretical_cgpt(Flower(1.0, 1.0, 256), contrast(3.0), 4).real


def test_descriptor_invariance(flower_cgpt):
    # Descriptors are invariant under translation, rotation and scaling
    # of the underlying shape.
    base = ShapeDescriptor.from_cgpt(flower_cgpt)

    moved = (Flower(1.0, 1.0, 256).rotate(0.7) * 1.6) + np.array([0.5, -0.2])
    moved_cgpt = theoretical_cgpt(moved, contrast(3.0), 4).real
    transformed = ShapeDescriptor.from_cgpt(moved_cgpt)

    assert base.distance(transformed) < 1e-4
    np.testing.assert_allclose(base.i1, transformed.i1, atol=1e-5)
    np.testing.assert_allclose(base.i2, transformed.i2, atol=1e-5)


def test_descriptor_separates_shapes(flower_cgpt):
    flower = ShapeDescriptor.from_cgpt(flower_cgpt)
    ellipse_cgpt = theoretical_cgpt(Ellipse(1.0, 0.5, 256), contrast(3.0), 4).real
    ellipse = ShapeDescriptor.from_cgpt(ellipse_cgpt)
    assert flower.distance(ellipse) > 0.1


def test_descriptor_order_and_truncated_distance(flower_cgpt):
    descriptor = ShapeDescriptor.from_cgpt(flower_cgpt)
    assert descriptor.order == 4
    assert descriptor.distance(descriptor) == 0.0
    assert descriptor.distance(descriptor, order=2) == 0.0


def test_dictionary_build_and_match():
    shapes = [
        Ellipse(1.0, 0.5, 256),
        Flower(1.0, 1.0, 256),
        Triangle(1.0, np.pi / 3, 256),
        Rectangle(2.0, 1.0, 256),
    ]
    dico = ShapeDictionary.build(shapes, cnd=3.0, order=4)
    assert dico.names == ["Ellipse", "Flower", "Triangle", "Rectangle"]

    # Each dictionary element matches itself best.
    for k, descriptor in enumerate(dico.descriptors):
        errors, ranking = dico.match(descriptor)
        assert ranking[0] == k
        assert errors[k] == pytest.approx(0.0, abs=1e-12)
        assert dico.identify(descriptor) == dico.names[k]


def test_match_descriptor_function(flower_cgpt):
    query = ShapeDescriptor.from_cgpt(flower_cgpt)
    items = [query, ShapeDescriptor(i1=query.i1 + 1, i2=query.i2)]
    errors, ranking = match_descriptor(query, items)
    assert ranking[0] == 0
    assert errors[1] > errors[0]


def test_degenerate_cgpt_rejected():
    with pytest.raises(ValueError, match="zero"):
        ShapeDescriptor.from_cgpt(np.zeros((8, 8)))


def test_empty_dictionary_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ShapeDictionary.build([], cnd=3.0)
