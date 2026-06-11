"""Closed-form CGPT matrices of disks and ellipses.

These exact formulas (M. Lim's PhD thesis, also reported in Ammari &
Kang 2007) serve as reference values to validate the boundary integral
computation of `theoretical_cgpt`.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb, factorial

__all__ = ["disk_cgpt", "ellipse_cgpt"]


def disk_cgpt(order: int, radius: float, cnd: float) -> NDArray:
    """Exact CGPT matrix of a disk centered at the origin.

    Parameters
    ----------
    order : int
        Maximum CGPT order.
    radius : float
        Radius of the disk.
    cnd : float
        Conductivity of the disk (background is one).

    Returns
    -------
    ndarray, shape (2 * order, 2 * order)
        The diagonal CGPT matrix of the disk.
    """
    n = np.arange(1, order + 1)
    diag = np.pi * n * radius ** (2 * n) * 2 * (cnd - 1) / (cnd + 1)
    return np.diag(np.repeat(diag, 2))


def ellipse_cgpt(order: int, axis_a: float, axis_b: float, cnd: float) -> NDArray:
    """Exact CGPT matrix of an ellipse centered at the origin.

    Parameters
    ----------
    order : int
        Maximum CGPT order.
    axis_a : float
        Semi-major axis length.
    axis_b : float
        Semi-minor axis length.
    cnd : float
        Conductivity of the ellipse (background is one). Complex values
        are not supported by the closed form.

    Returns
    -------
    ndarray, shape (2 * order, 2 * order)
        The CGPT matrix of the ellipse.
    """
    exact = np.zeros((2 * order, 2 * order))
    for m in range(1, order + 1):
        for n in range(m, order + 1):
            block = _ellipse_block(m, n, axis_a, axis_b, cnd)
            exact[2 * m - 2 : 2 * m, 2 * n - 2 : 2 * n] = block.real
    # Fill the lower triangle by symmetry of the CGPT.
    upper = np.triu(np.ones_like(exact), 1)
    exact = exact + (exact * upper).T
    return exact * (cnd - 1)


def _ellipse_block(m: int, n: int, a: float, b: float, k: float) -> NDArray:
    """Compute the 2x2 block ``(m, n)`` of the ellipse CGPT.

    Parameters
    ----------
    m, n : int
        Block indices with ``n >= m``.
    a, b : float
        Semi-axis lengths.
    k : float
        Conductivity.

    Returns
    -------
    ndarray, shape (2, 2)
        The diagonal block ``diag(M_cc, M_ss)``.
    """
    p = np.arctanh(b / a)
    r = np.sqrt(a**2 - b**2)
    block = np.zeros((2, 2))
    cm = m * np.pi * r ** (m + n) * 2.0 ** (1 - m - n)

    ca_r = ca_i = 0.0
    if (n - m) % 2 == 0:
        ca_r = comb(n, (n - m) // 2) * _bf(m, p, k, False) * np.sinh(2 * m * p)
        ca_i = comb(n, (n - m) // 2) * _bf(m, p, k, True) * np.sinh(2 * m * p)

    if m % 2 == 0 and n % 2 == 0:
        block[0, 0] = cm * (_even_sum(m, n, p, k, False) + ca_r)
        block[1, 1] = cm * (_even_sum(m, n, p, k, True) + ca_i)
    elif m % 2 == 1 and n % 2 == 1:
        block[0, 0] = cm * (_odd_sum(m, n, p, k, False) + ca_r)
        block[1, 1] = cm * (_odd_sum(m, n, p, k, True) + ca_i)
    return block


def _bf(v, p: float, k: float, tilde: bool):
    """Evaluate the contrast-dependent factor of the ellipse formulas.

    Parameters
    ----------
    v : int or ndarray
        Summation index.
    p : float
        Elliptic parameter ``arctanh(b / a)``.
    k : float
        Conductivity.
    tilde : bool
        Select the variant appearing in the sine terms.
    """
    num = np.sinh(v * p) + np.cosh(v * p)
    if tilde:
        return num / (np.sinh(v * p) + k * np.cosh(v * p))
    return num / (k * np.sinh(v * p) + np.cosh(v * p))


def _even_sum(m: int, n: int, p: float, k: float, tilde: bool) -> float:
    """Evaluate the even-order correction sum of the ellipse formulas."""
    upper = min(n // 2, (m - 2) // 2)
    if upper < 1:
        return 0.0
    t = np.arange(1, upper + 1)
    parta = 4 * factorial(m - 1) * factorial(n) * _bf(2 * t, p, k, tilde) * t * np.sinh(4 * t * p)
    partb = (
        (m + 2 * t)
        * factorial(m // 2 - t)
        * factorial(m // 2 - 1 + t)
        * factorial(n // 2 + t)
        * factorial(n // 2 - t)
    )
    return float(np.sum(parta / partb))


def _odd_sum(m: int, n: int, p: float, k: float, tilde: bool) -> float:
    """Evaluate the odd-order correction sum of the ellipse formulas."""
    upper = min((n - 1) // 2, (m - 3) // 2)
    if upper < 0:
        return 0.0
    t = np.arange(0, upper + 1)
    parta = (
        factorial(m - 1)
        * factorial(n)
        * _bf(2 * t + 1, p, k, tilde)
        * (4 * t + 2)
        * np.sinh((4 * t + 2) * p)
    )
    partb = (
        (m + 2 * t + 1)
        * factorial((m - 1) // 2 - t)
        * factorial((m - 1) // 2 + t)
        * factorial((n + 1) // 2 + t)
        * factorial((n - 1) // 2 - t)
    )
    return float(np.sum(parta / partb))
