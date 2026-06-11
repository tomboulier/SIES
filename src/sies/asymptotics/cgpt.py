r"""Numerical computation of contracted generalized polarization tensors.

The CGPT matrix of order ``k`` of inclusions $D_1, \dots, D_L$
with contrasts $\lambda_l$ is built from the boundary densities

$$
\phi_m = (\lambda I - K_D^*)^{-1}
\left[\frac{\partial \mathrm{Re}/\mathrm{Im}(z^m)}{\partial\nu}\right],
$$
see Ammari & Kang, *Polarization and Moment Tensors* (2007), chapter 4.
"""

import numpy as np
from numpy.typing import NDArray

from sies.asymptotics.transforms import join_cgpt
from sies.operators import KStar, SingleLayerNormalDerivative
from sies.shapes import C2Boundary

__all__ = ["contrast", "make_block_matrix", "make_system_matrix", "theoretical_cgpt"]

#: Threshold below which ``|lambda - 1/2|`` triggers the zero-mean constraint.
_HALF_CONTRAST_TOL = 1e-8


def contrast(
    cnd: NDArray,
    pmtt: NDArray | None = None,
    freq: float = 0.0,
) -> NDArray:
    r"""Compute the contrast constants $\lambda$ of inclusions.

    For conductivity ``k``, permittivity ``eps`` and frequency ``f``, the
    contrast is

    $$\lambda = \frac{k + 2\pi i f \epsilon + 1} {2(k + 2\pi i f \epsilon - 1)}.$$

    Parameters
    ----------
    cnd : array_like
        Conductivity of each inclusion; must be positive and different
        from one (the background conductivity).
    pmtt : array_like, optional
        Permittivity of each inclusion. Defaults to zero.
    freq : float, default 0.0
        Working frequency; must be nonnegative.

    Returns
    -------
    ndarray
        Complex contrast constant of each inclusion.

    Notes
    -----
    The $2\pi$ factor comes from the Fourier transform convention
    $\hat{f}(\omega) = \int f(x) e^{-2\pi i x \omega} dx$ used
    throughout the library (compatible with the FFT); published papers
    use the convention without the factor.
    """
    cnd = np.atleast_1d(np.asarray(cnd, dtype=float))
    if pmtt is None:
        pmtt = np.zeros_like(cnd)
    pmtt = np.atleast_1d(np.asarray(pmtt, dtype=float))

    if freq < 0:
        raise ValueError("Frequency must be a nonnegative scalar.")
    if np.any(cnd == 1) or np.any(cnd < 0):
        raise ValueError("Conductivity must be positive and different from 1.")

    kappa = cnd + 2j * np.pi * pmtt * freq
    lambdas = (kappa + 1) / (2 * (kappa - 1))
    return np.real_if_close(lambdas)


def _check_disjoint(inclusions: list[C2Boundary]) -> None:
    """Raise if any two inclusions are not mutually disjoint.

    Parameters
    ----------
    inclusions : list of C2Boundary
        The inclusions to check.
    """
    for m, first in enumerate(inclusions):
        for second in inclusions[m + 1 :]:
            if not first.is_disjoint(second):
                raise ValueError("Inclusions must be mutually disjoint.")


def make_block_matrix(inclusions: list[C2Boundary]) -> NDArray:
    r"""Assemble the frequency-independent blocks of the system matrix.

    The diagonal blocks are $-K_{D_n}^*$ and the off-diagonal
    ``(m, n)`` blocks are
    $-\partial_{\nu_m} S_{D_n}$. The contrast-dependent term
    $\lambda_n I$ is added later by `make_system_matrix`,
    which makes multi-frequency computations cheap.

    Parameters
    ----------
    inclusions : list of C2Boundary
        Mutually disjoint inclusions, all discretized with the same
        number of boundary points.

    Returns
    -------
    ndarray, shape (L * n, L * n)
        The block matrix, where ``L`` is the number of inclusions and
        ``n`` the number of boundary points.
    """
    _check_disjoint(inclusions)
    nb_incl = len(inclusions)
    nb_points = inclusions[0].nb_points
    blocks = np.empty((nb_incl * nb_points, nb_incl * nb_points))

    for m, image in enumerate(inclusions):
        rows = slice(m * nb_points, (m + 1) * nb_points)
        for n, domain in enumerate(inclusions):
            cols = slice(n * nb_points, (n + 1) * nb_points)
            if m == n:
                blocks[rows, cols] = -KStar(domain).matrix
            else:
                blocks[rows, cols] = -SingleLayerNormalDerivative.kernel_matrix(
                    domain.points, domain.sigma, image.points, image.normal
                )
    return blocks


def make_system_matrix(block_matrix: NDArray, lambdas: NDArray) -> NDArray:
    r"""Assemble the full system matrix $\lambda I - K_D^*$.

    Parameters
    ----------
    block_matrix : ndarray
        Frequency-independent blocks from `make_block_matrix`.
    lambdas : array_like
        Contrast constant of each inclusion.

    Returns
    -------
    ndarray
        The system matrix ``A`` of the linear system ``A phi = b``.
    """
    lambdas = np.atleast_1d(np.asarray(lambdas))
    nb_incl = len(lambdas)
    nb_points = block_matrix.shape[0] // nb_incl

    system = block_matrix.astype(np.result_type(block_matrix, lambdas), copy=True)
    for n, lam in enumerate(lambdas):
        idx = np.arange(n * nb_points, (n + 1) * nb_points)
        system[idx, idx] += lam
    return system


def _solve_densities(system: NDArray, rhs: NDArray, augmented: bool) -> NDArray:
    """Solve the boundary integral system for the densities.

    Parameters
    ----------
    system : ndarray
        System matrix, possibly augmented with zero-mean constraints.
    rhs : ndarray
        Right-hand sides, one column per source term.
    augmented : bool
        Whether the system is rectangular (least-squares solve).

    Returns
    -------
    ndarray
        The boundary densities.
    """
    if augmented:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]
    return np.linalg.solve(system, rhs)


def theoretical_cgpt(
    inclusions: list[C2Boundary] | C2Boundary,
    lambdas: NDArray,
    order: int,
    block_matrix: NDArray | None = None,
) -> NDArray:
    r"""Compute the CGPT matrix of inclusions up to a given order.

    Entries follow the convention

    $$
    M^{cc}_{mn} = \int_{\partial D} \mathrm{Re}(z^n) \,
    (\lambda I - K_D^*)^{-1} \left[\partial_\nu \mathrm{Re}(z^m)\right] ds,
    $$
    and similarly for the ``cs``, ``sc`` and ``ss`` blocks, interleaved
    in a ``(2 * order, 2 * order)`` matrix.

    Parameters
    ----------
    inclusions : C2Boundary or list of C2Boundary
        The inclusion(s).
    lambdas : array_like
        Contrast constant of each inclusion.
    order : int
        Maximum CGPT order.
    block_matrix : ndarray, optional
        Precomputed blocks from `make_block_matrix`, to be reused
        across frequencies.

    Returns
    -------
    ndarray, shape (2 * order, 2 * order)
        The CGPT matrix (complex if any contrast is complex).
    """
    if isinstance(inclusions, C2Boundary):
        inclusions = [inclusions]
    lambdas = np.atleast_1d(np.asarray(lambdas))
    if len(lambdas) < len(inclusions):
        raise ValueError("A contrast value must be specified for each inclusion.")

    if block_matrix is None:
        block_matrix = make_block_matrix(inclusions)

    nb_incl = len(inclusions)
    nb_points = inclusions[0].nb_points
    system = make_system_matrix(block_matrix, lambdas)

    # When lambda is close to 1/2 (infinite contrast), enforce that the
    # densities have zero mean by augmenting the system.
    augmented = bool(np.min(np.abs(lambdas - 0.5)) < _HALF_CONTRAST_TOL)
    if augmented:
        constraints = np.kron(np.eye(nb_incl), np.ones((1, nb_points)))
        system = np.vstack([system, constraints])

    # Right-hand sides nu . grad(z^m) for all orders m, stacked by inclusion.
    rhs = np.empty((nb_incl * nb_points, order), dtype=complex)
    for i, incl in enumerate(inclusions):
        rows = slice(i * nb_points, (i + 1) * nb_points)
        for m in range(1, order + 1):
            grad = m * incl.cpoints ** (m - 1)
            rhs[rows, m - 1] = (incl.normal[0] + 1j * incl.normal[1]) * grad
    if augmented:
        rhs = np.vstack([rhs, np.zeros((nb_incl, order))])

    phi_re = _solve_densities(system, rhs.real, augmented)
    phi_im = _solve_densities(system, rhs.imag, augmented)

    is_complex = np.iscomplexobj(phi_re) or np.iscomplexobj(phi_im)
    dtype = complex if is_complex else float
    cc = np.zeros((order, order), dtype=dtype)
    cs = np.zeros_like(cc)
    sc = np.zeros_like(cc)
    ss = np.zeros_like(cc)

    for i, incl in enumerate(inclusions):
        rows = slice(i * nb_points, (i + 1) * nb_points)
        # moments[n - 1] = z^n * sigma for n = 1..order, shape (order, nb_points)
        powers = incl.cpoints[np.newaxis, :] ** np.arange(1, order + 1)[:, np.newaxis]
        moments = powers * incl.sigma
        cc += moments.real @ phi_re[rows]
        cs += moments.imag @ phi_re[rows]
        sc += moments.real @ phi_im[rows]
        ss += moments.imag @ phi_im[rows]

    # The (m, n) entry pairs the m-th density with the n-th moment.
    return join_cgpt(cc.T, cs.T, sc.T, ss.T)
