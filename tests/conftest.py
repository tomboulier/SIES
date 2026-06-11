"""Shared fixtures for the SIES test suite."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from sies.acquisition import Coincided
from sies.shapes import Ellipse


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


@pytest.fixture
def disk():
    """A disk of radius 0.5 with 256 boundary points."""
    return Ellipse(1.0, 1.0, 256) * 0.5


@pytest.fixture
def ellipse():
    """An ellipse of semi-axes (1, 0.5) with 256 boundary points."""
    return Ellipse(1.0, 0.5, 256)


@pytest.fixture
def full_view_cfg():
    """A coincided full-view circular configuration with 50 sources."""
    return Coincided(np.zeros(2), 1.5, 50)
