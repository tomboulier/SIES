"""Resampling of closed curves by periodic cubic splines.

These utilities reconstruct the tangent, acceleration and outward normal
vectors of a closed curve from sampled points. Corner singularities can
be smoothed out by down-sampling the curve before the spline fit.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

__all__ = ["resample_curve"]


def _periodic_spline(theta: NDArray, values: NDArray) -> CubicSpline:
    r"""Fit a periodic cubic spline through points of a closed curve.

    Parameters
    ----------
    theta : ndarray, shape (n,)
        Strictly increasing parameter values in ``[0, 2 pi)``. The first
        and last samples must *not* coincide (the curve is not tied off).
    values : ndarray, shape (2, n)
        Curve coordinates at the parameter values.

    Returns
    -------
    scipy.interpolate.CubicSpline
        Periodic spline of period $2\pi$.
    """
    theta_ext = np.append(theta, theta[0] + 2 * np.pi)
    values_ext = np.column_stack([values, values[:, 0]])
    return CubicSpline(theta_ext, values_ext, axis=1, bc_type="periodic")


def resample_curve(
    points: NDArray,
    theta: NDArray,
    nb_points: int,
    box: tuple[float, float] | None = None,
    downsample: int = 1,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Resample a closed curve and compute its differential quantities.

    The curve is optionally rescaled to fit a bounding box, down-sampled
    (which smooths out corner singularities), then re-interpolated on a
    uniform parameter grid with a periodic cubic spline.

    Parameters
    ----------
    points : ndarray, shape (2, n)
        Coordinates of the curve samples, not tied off.
    theta : ndarray, shape (n,)
        Parameterization of the samples in ``[0, 2 pi)``.
    nb_points : int
        Number of points of the resampled curve.
    box : tuple of float, optional
        If given, target size ``(width, height)`` of the bounding box of
        the curve.
    downsample : int, default 1
        Down-sampling factor applied before interpolation; values larger
        than one smooth out corners.

    Returns
    -------
    points : ndarray, shape (2, nb_points)
        Resampled curve.
    tvec : ndarray, shape (2, nb_points)
        Tangent vectors (first derivative).
    avec : ndarray, shape (2, nb_points)
        Acceleration vectors (second derivative).
    normal : ndarray, shape (2, nb_points)
        Outward unit normal vectors.
    """
    downsample = int(np.ceil(downsample))
    if downsample < 1:
        raise ValueError("Down-sampling factor must be positive.")

    if box is not None:
        xmin, xmax = points[0].min(), points[0].max()
        ymin, ymax = points[1].min(), points[1].max()
        if np.isclose(xmax, xmin) or np.isclose(ymax, ymin):
            raise ValueError("Curve must have nonzero extent in both axes for box rescaling.")
        center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        scale = np.array([box[0] / (xmax - xmin), box[1] / (ymax - ymin)])
        points = (points - center[:, np.newaxis]) * scale[:, np.newaxis]

    spline = _periodic_spline(theta[::downsample], points[:, ::downsample])

    theta_new = 2 * np.pi * np.arange(nb_points) / nb_points
    new_points = spline(theta_new)
    tvec = spline(theta_new, 1)
    avec = spline(theta_new, 2)

    normal = np.vstack([tvec[1], -tvec[0]])
    normal = normal / np.linalg.norm(normal, axis=0)
    return new_points, tvec, avec, normal
