"""Unit tests of the noise utilities."""

import numpy as np
import pytest

from sies.utils import add_white_noise


def test_noise_level_calibration(rng):
    data = rng.standard_normal((50, 200)) * 3.0
    noisy, sigma = add_white_noise(data, 0.1, rng)

    noise = noisy - data
    empirical = np.linalg.norm(noise) / np.sqrt(noise.size)
    assert empirical == pytest.approx(sigma, rel=0.05)
    assert sigma == pytest.approx(np.linalg.norm(data) / np.sqrt(data.size) * 0.1, rel=1e-12)


def test_per_row_mode_scales_with_row_energy(rng):
    data = np.vstack([np.full(1000, 10.0), np.full(1000, 0.1)])
    noisy, _ = add_white_noise(data, 0.1, rng, per_row=True)
    noise = noisy - data
    # Row noise is proportional to row energy: ratio about 100.
    ratio = np.std(noise[0]) / np.std(noise[1])
    assert 50 < ratio < 200


def test_global_mode(rng):
    data = rng.standard_normal((20, 20))
    noisy, sigma = add_white_noise(data, 0.2, rng, per_row=False)
    noise = noisy - data
    assert np.std(noise) == pytest.approx(sigma, rel=0.2)


def test_zero_level_returns_data(rng):
    data = rng.standard_normal((5, 5))
    noisy, sigma = add_white_noise(data, 0.0, rng)
    np.testing.assert_allclose(noisy, data)
    assert sigma == 0.0


def test_default_rng_used_when_omitted():
    noisy, _ = add_white_noise(np.ones((4, 4)), 0.1)
    assert noisy.shape == (4, 4)
