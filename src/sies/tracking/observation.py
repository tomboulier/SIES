"""Observation model of the tracking problem.

The observation is the MSR matrix produced by a target of known CGPT
located at the (unknown) position and orientation encoded in the state
vector ``[vx, vy, x, y, phi]``.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb

from sies.asymptotics import transform_cgpt

__all__ = ["CGPTObservation"]


class CGPTObservation:
    """MSR observation of a moving target with known CGPT.

    The forward model is ``MSR = As M(x, phi) Ar^T`` where ``M(x, phi)``
    is the CGPT of the target translated to ``x`` and rotated by
    ``phi``.

    Parameters
    ----------
    src_matrix : ndarray, shape (ns, 2k)
        Source-side acquisition matrix ``As``.
    rcv_matrix : ndarray, shape (nr, 2k)
        Receiver-side acquisition matrix ``Ar``.
    cgpt : ndarray, shape (2k, 2k)
        CGPT matrix of the target at the origin with orientation zero.
    """

    def __init__(self, src_matrix: NDArray, rcv_matrix: NDArray, cgpt: NDArray):
        if cgpt.shape[0] != cgpt.shape[1]:
            raise ValueError("CGPT matrix must be square.")
        self.src_matrix = src_matrix
        self.rcv_matrix = rcv_matrix
        self.cgpt = cgpt
        self._order = cgpt.shape[0] // 2

    def __call__(self, state: NDArray) -> NDArray:
        """Evaluate the observation function ``h``.

        Parameters
        ----------
        state : ndarray, shape (5,)
            State vector ``[vx, vy, x, y, phi]``.

        Returns
        -------
        ndarray, shape (ns * nr,)
            The vectorized MSR matrix.
        """
        translation = state[2] + 1j * state[3]
        moved = transform_cgpt(self.cgpt, translation, 1.0, state[4])
        return (self.src_matrix @ moved @ self.rcv_matrix.T).ravel()

    def jacobian(self, state: NDArray) -> NDArray:
        """Evaluate the Jacobian of the observation function.

        The derivatives are taken with respect to the five state
        variables; only the position and orientation derivatives are
        nonzero.

        Parameters
        ----------
        state : ndarray, shape (5,)
            State vector ``[vx, vy, x, y, phi]``.

        Returns
        -------
        ndarray, shape (ns * nr, 5)
            The Jacobian matrix.
        """
        order = self._order
        t0 = state[2] + 1j * state[3]
        phi = state[4]

        # Transformation in the complex representation: M -> J^T M J with
        # J = U F, U the embedding of C^k in R^(2k).
        embed = np.kron(np.eye(order), np.array([[1.0], [1.0j]]))
        re_u, im_u = embed.real, embed.imag

        m_idx = np.arange(1, order + 1)[:, np.newaxis]
        n_idx = np.arange(1, order + 1)[np.newaxis, :]
        binom = np.triu(comb(n_idx, m_idx))
        rotation = np.exp(1j * m_idx * phi)

        diff = n_idx - m_idx
        powers = (t0 + 0j) ** np.maximum(diff, 0)
        transform = binom * powers * rotation

        d_powers = np.where(diff >= 1, diff * (t0 + 0j) ** np.maximum(diff - 1, 0), 0)
        d_x = binom * d_powers * rotation
        d_y = 1j * d_x
        d_phi = transform * (1j * m_idx)

        jac = np.zeros((self.src_matrix.shape[0] * self.rcv_matrix.shape[0], 5))
        for column, d_transform in ((2, d_x), (3, d_y), (4, d_phi)):
            jac[:, column] = self._assemble_derivative(embed, re_u, im_u, transform, d_transform)
        return jac

    def _assemble_derivative(
        self,
        embed: NDArray,
        re_u: NDArray,
        im_u: NDArray,
        transform: NDArray,
        d_transform: NDArray,
    ) -> NDArray:
        """Assemble one column of the Jacobian.

        Parameters
        ----------
        embed : ndarray
            Complex embedding matrix ``U``.
        re_u, im_u : ndarray
            Real and imaginary parts of ``U``.
        transform : ndarray
            Complex transformation matrix ``F``.
        d_transform : ndarray
            Derivative of ``F`` with respect to one state variable.

        Returns
        -------
        ndarray
            The vectorized derivative of the MSR matrix.
        """
        j_mat = embed @ transform
        dj_mat = embed @ d_transform
        re_j, im_j = j_mat.real, j_mat.imag
        d_re_j, d_im_j = dj_mat.real, dj_mat.imag

        m0 = self.cgpt

        def sym(a: NDArray, b: NDArray) -> NDArray:
            return a.T @ m0 @ b

        d_rmr = sym(d_re_j, re_j) + sym(re_j, d_re_j)
        d_rmi = sym(d_re_j, im_j) + sym(re_j, d_im_j)
        d_imr = sym(d_im_j, re_j) + sym(im_j, d_re_j)
        d_imi = sym(d_im_j, im_j) + sym(im_j, d_im_j)

        d_cgpt = (
            re_u @ d_rmr @ re_u.T
            + re_u @ d_rmi @ im_u.T
            + im_u @ d_imr @ re_u.T
            + im_u @ d_imi @ im_u.T
        )
        return (self.src_matrix @ d_cgpt @ self.rcv_matrix.T).ravel()
