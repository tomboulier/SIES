r"""Green's function of the 2D Laplacian and its derivatives.

The fundamental solution of the Laplace equation in two dimensions is

$$G(x) = \frac{1}{2\pi} \log|x|.$$

All functions in this module are vectorized: point clouds are given as
arrays of shape ``(2, n)`` and the result is a matrix indexed by the two
point clouds.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["green2d", "green2d_dn", "green2d_grad"]


def _check_points(*arrays: NDArray) -> None:
    """Validate that every input is a 2-row coordinate array.

    Parameters
    ----------
    *arrays : ndarray
        Arrays to validate.

    Raises
    ------
    ValueError
        If an array does not have exactly two rows.
    """
    for arr in arrays:
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("Coordinate arrays must have shape (2, n).")


def green2d(x: NDArray, y: NDArray) -> NDArray:
    """Evaluate the 2D Green's function on two point clouds.

    Parameters
    ----------
    x : ndarray, shape (2, m)
        First point cloud.
    y : ndarray, shape (2, n)
        Second point cloud. Must be disjoint from `x` (the kernel is
        singular on the diagonal).

    Returns
    -------
    ndarray, shape (m, n)
        Matrix ``G`` with ``G[i, j] = log(|x_i - y_j|) / (2 pi)``.
    """
    _check_points(x, y)
    dx = x[0][:, np.newaxis] - y[0][np.newaxis, :]
    dy = x[1][:, np.newaxis] - y[1][np.newaxis, :]
    return np.log(dx**2 + dy**2) / (4 * np.pi)


def green2d_grad(x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """Evaluate the gradient of the 2D Green's function.

    Parameters
    ----------
    x : ndarray, shape (2, m)
        First point cloud.
    y : ndarray, shape (2, n)
        Second point cloud.

    Returns
    -------
    gx : ndarray, shape (m, n)
        Partial derivative with respect to the first coordinate,
        evaluated at ``x_i - y_j``.
    gy : ndarray, shape (m, n)
        Partial derivative with respect to the second coordinate,
        evaluated at ``x_i - y_j``.
    """
    _check_points(x, y)
    dx = x[0][:, np.newaxis] - y[0][np.newaxis, :]
    dy = x[1][:, np.newaxis] - y[1][np.newaxis, :]
    dist2 = dx**2 + dy**2
    return dx / (2 * np.pi * dist2), dy / (2 * np.pi * dist2)


def green2d_dn(x: NDArray, y: NDArray, normal: NDArray) -> NDArray:
    r"""Evaluate the normal derivative of the Green's function on a boundary.

    For each source point $x_i$ and boundary point $y_j$ with
    outward normal $\nu_j$, compute
    $\langle \nabla G(y_j - x_i), \nu_j \rangle$.

    Parameters
    ----------
    x : ndarray, shape (2, m)
        Evaluation (source) points.
    y : ndarray, shape (2, n)
        Boundary points.
    normal : ndarray, shape (2, n)
        Outward unit normal vectors at the boundary points.

    Returns
    -------
    ndarray, shape (m, n)
        The normal derivative matrix.
    """
    _check_points(x, y, normal)
    gx, gy = green2d_grad(y, x)
    gn = normal[0][:, np.newaxis] * gx + normal[1][:, np.newaxis] * gy
    return gn.T
