"""Small numerical utilities shared across the library."""

import numpy as np
from numpy.typing import NDArray

__all__ = ["add_white_noise"]


def add_white_noise(
    data: NDArray,
    level: float,
    rng: np.random.Generator | None = None,
    per_row: bool = True,
) -> tuple[NDArray, float]:
    """Add white Gaussian noise to a real data matrix.

    The noise standard deviation is calibrated relative to the energy of
    the signal: a `level` of ``0.1`` corresponds to 10% noise.

    Parameters
    ----------
    data : ndarray, shape (m, n)
        Input data matrix (real).
    level : float
        Noise level, relative to the root mean square of the data.
    rng : numpy.random.Generator, optional
        Random generator, for reproducibility.
    per_row : bool, default True
        If True, each row is corrupted independently with a noise level
        proportional to its own energy.

    Returns
    -------
    noisy : ndarray, shape (m, n)
        The noisy data.
    sigma : float
        Global standard deviation of the added noise,
        ``norm(data) / sqrt(data.size) * level``.

    Raises
    ------
    ValueError
        If `level` is negative or `data` is not a 2D array.
    """
    rng = rng or np.random.default_rng()
    data = np.asarray(data)
    if level < 0:
        raise ValueError("Noise level must be nonnegative.")
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")

    if per_row:
        scale = np.linalg.norm(data, axis=1, keepdims=True) / np.sqrt(data.shape[1])
    else:
        scale = np.linalg.norm(data) / np.sqrt(data.size)
    noisy = data + rng.standard_normal(data.shape) * scale * level

    sigma = float(np.linalg.norm(data) / np.sqrt(data.size) * level)
    return noisy, sigma
