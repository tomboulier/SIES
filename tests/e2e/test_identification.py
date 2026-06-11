"""End-to-end test: dictionary identification from noisy measurements.

Reproduces the experiment of ``demo_dico_matching.m``: an unknown
target (a transformed dictionary element) is illuminated by point
sources; the CGPT is reconstructed from the noisy MSR matrix and the
shape is identified in the dictionary by invariant descriptors.
"""

import numpy as np
import pytest

from sies.acquisition import Coincided
from sies.dictionary import ShapeDescriptor, ShapeDictionary
from sies.pde import ConductivityR2
from sies.shapes import Ellipse, Flower, Rectangle, Triangle

NB_POINTS = 256
CND = 3.0


@pytest.fixture(scope="module")
def dictionary():
    shapes = [
        Ellipse(1.0, 0.5, NB_POINTS),
        Flower(1.0, 1.0, NB_POINTS),
        Triangle(1.0, np.pi / 3, NB_POINTS),
        Rectangle(1.0, 1.0, NB_POINTS),
        Rectangle(2.0, 1.0, NB_POINTS) * 0.5,
    ]
    return ShapeDictionary.build(shapes, cnd=CND, order=5)


@pytest.mark.e2e
@pytest.mark.parametrize("target_index", range(5))
def test_identification_under_noise(dictionary, target_index):
    rng = np.random.default_rng(100 + target_index)

    # The unknown target: a rotated, scaled and translated dictionary shape.
    target = dictionary.shapes[target_index]
    target = (target.rotate(0.2 * np.pi) * 0.75) + np.array([0.25, 0.25])

    cfg = Coincided(np.zeros(2), 1.5, 100)
    pde = ConductivityR2(target, CND, 0.0, cfg)

    data = pde.simulate_data(0.0)
    noisy = pde.add_white_noise(data, 0.05, rng)

    result = pde.reconstruct_cgpt_analytic(noisy.msr_noisy[0], 5)
    descriptor = ShapeDescriptor.from_cgpt(result.cgpt[0])

    errors, ranking = dictionary.match(descriptor, order=3)
    assert ranking[0] == target_index, (
        f"expected {dictionary.names[target_index]}, "
        f"identified {dictionary.names[ranking[0]]} (errors: {errors})"
    )
