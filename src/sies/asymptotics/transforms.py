"""Transformation rules of CGPT matrices under rigid motions and scaling.

A CGPT matrix is stored as a ``(2k, 2k)`` array interleaving the four
``cc``, ``cs``, ``sc`` and ``ss`` blocks. The *complex* CGPT pair
``(N1, N2)`` diagonalizes the action of translations, rotations and
dilations, which makes these transformations explicit; see Ammari et
al., *Target identification using dictionary matching of generalized
polarization tensors* (2014).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb

__all__ = [
    "cgpt_to_complex",
    "complex_to_cgpt",
    "join_cgpt",
    "split_cgpt",
    "transform_ccgpt",
    "transform_ccgpt_inverse",
    "transform_cgpt",
]


def split_cgpt(cgpt: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split an interleaved CGPT matrix into its four blocks.

    Parameters
    ----------
    cgpt : ndarray, shape (2k, 2k)
        Interleaved CGPT matrix.

    Returns
    -------
    cc, cs, sc, ss : ndarray, shape (k, k)
        The four blocks of the CGPT.
    """
    return cgpt[0::2, 0::2], cgpt[0::2, 1::2], cgpt[1::2, 0::2], cgpt[1::2, 1::2]


def join_cgpt(cc: NDArray, cs: NDArray, sc: NDArray, ss: NDArray) -> NDArray:
    """Interleave the four CGPT blocks into a single matrix.

    Parameters
    ----------
    cc, cs, sc, ss : ndarray, shape (k, k)
        The four blocks of the CGPT.

    Returns
    -------
    ndarray, shape (2k, 2k)
        The interleaved CGPT matrix.
    """
    order = cc.shape[0]
    cgpt = np.zeros((2 * order, 2 * order), dtype=np.result_type(cc, cs, sc, ss))
    cgpt[0::2, 0::2] = cc
    cgpt[0::2, 1::2] = cs
    cgpt[1::2, 0::2] = sc
    cgpt[1::2, 1::2] = ss
    return cgpt


def cgpt_to_complex(cgpt: NDArray) -> tuple[NDArray, NDArray]:
    """Build the complex CGPT pair from an interleaved CGPT matrix.

    Parameters
    ----------
    cgpt : ndarray, shape (2k, 2k)
        Interleaved CGPT matrix.

    Returns
    -------
    n1 : ndarray, shape (k, k)
        First complex CGPT; symmetric.
    n2 : ndarray, shape (k, k)
        Second complex CGPT; hermitian.
    """
    cc, cs, sc, ss = split_cgpt(cgpt)
    n1 = cc - ss + 1j * (cs + sc)
    n2 = cc + ss + 1j * (cs - sc)
    return n1, n2


def complex_to_cgpt(n1: NDArray, n2: NDArray) -> NDArray:
    """Convert the complex CGPT pair back to an interleaved CGPT matrix.

    Only valid for real contrasts (real-valued CGPT).

    Parameters
    ----------
    n1 : ndarray, shape (k, k)
        First complex CGPT.
    n2 : ndarray, shape (k, k)
        Second complex CGPT.

    Returns
    -------
    ndarray, shape (2k, 2k)
        The interleaved CGPT matrix.
    """
    cc = (n1 + n2).real / 2
    cs = (n1 + n2).imag / 2
    sc = (n1 - n2).imag / 2
    ss = (n2 - n1).real / 2
    return join_cgpt(cc, cs, sc, ss)


def _translation_matrix(t0: complex, order: int, inverse: bool) -> NDArray:
    """Build the lower-triangular translation matrix ``C_t``.

    Parameters
    ----------
    t0 : complex
        Translation encoded as a complex number.
    order : int
        Size of the matrix.
    inverse : bool
        If True, build the matrix of the inverse translation.
    """
    n = np.arange(1, order + 1)
    binom = np.tril(comb(n[:, np.newaxis], n[np.newaxis, :]))
    if inverse:
        t0 = -t0
    diff = n[:, np.newaxis] - n[np.newaxis, :]
    # Negative powers only occur where the binomial factor vanishes.
    powers = (t0 + 0j) ** np.maximum(diff, 0)
    return binom * powers


def transform_ccgpt(
    n1: NDArray,
    n2: NDArray,
    translation: complex,
    scaling: float = 1.0,
    rotation: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Apply rotation, then scaling, then translation to a complex CGPT pair.

    Parameters
    ----------
    n1, n2 : ndarray, shape (k, k)
        Complex CGPT pair of the original shape.
    translation : complex or array_like
        Translation, as a complex number ``tx + i ty`` or a 2-vector.
    scaling : float, default 1.0
        Scaling factor.
    rotation : float, default 0.0
        Rotation angle in radians.

    Returns
    -------
    z1, z2 : ndarray, shape (k, k)
        Complex CGPT pair of the transformed shape.
    """
    order = n1.shape[0]
    t0 = _as_complex(translation)
    ct = _translation_matrix(t0, order, inverse=False)
    gy = np.diag((scaling * np.exp(1j * rotation)) ** np.arange(1, order + 1))

    z1 = ct @ gy @ n1 @ gy @ ct.T
    z2 = ct.conj() @ gy.conj() @ n2 @ gy @ ct.T
    return z1, z2


def transform_ccgpt_inverse(
    n1: NDArray,
    n2: NDArray,
    translation: complex,
    scaling: float = 1.0,
    rotation: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Apply the inverse transformation to a complex CGPT pair.

    Undo a translation, then a scaling, then a rotation (the inverse of
    `transform_ccgpt` with the same arguments).

    Parameters
    ----------
    n1, n2 : ndarray, shape (k, k)
        Complex CGPT pair of the transformed shape.
    translation : complex or array_like
        Translation to undo.
    scaling : float, default 1.0
        Scaling factor to undo.
    rotation : float, default 0.0
        Rotation angle to undo, in radians.

    Returns
    -------
    z1, z2 : ndarray, shape (k, k)
        Complex CGPT pair with the transformation removed.
    """
    order = n1.shape[0]
    t0 = _as_complex(translation)
    ct = _translation_matrix(t0, order, inverse=True)
    gy = np.diag((np.exp(-1j * rotation) / scaling) ** np.arange(1, order + 1))

    z1 = gy @ ct @ n1 @ ct.T @ gy
    z2 = gy.conj() @ ct.conj() @ n2 @ ct.T @ gy
    return z1, z2


def transform_cgpt(
    cgpt: NDArray,
    translation: complex,
    scaling: float = 1.0,
    rotation: float = 0.0,
) -> NDArray:
    """Apply rotation, then scaling, then translation to a CGPT matrix.

    Parameters
    ----------
    cgpt : ndarray, shape (2k, 2k)
        Interleaved CGPT matrix of the original shape.
    translation : complex or array_like
        Translation, as a complex number ``tx + i ty`` or a 2-vector.
    scaling : float, default 1.0
        Scaling factor.
    rotation : float, default 0.0
        Rotation angle in radians.

    Returns
    -------
    ndarray, shape (2k, 2k)
        CGPT matrix of the transformed shape.
    """
    n1, n2 = cgpt_to_complex(cgpt)
    z1, z2 = transform_ccgpt(n1, n2, translation, scaling, rotation)
    return complex_to_cgpt(z1, z2)


def _as_complex(translation) -> complex:
    """Convert a translation given as scalar or 2-vector to a complex number.

    Parameters
    ----------
    translation : complex or array_like
        Translation specification.

    Returns
    -------
    complex
        The translation as ``tx + i ty``.

    Raises
    ------
    ValueError
        If an array-like translation does not have exactly two entries.
    """
    arr = np.asarray(translation)
    if arr.ndim == 0:
        return complex(arr)
    arr = arr.reshape(-1)
    if arr.size != 2:
        raise ValueError("A translation vector must have exactly two entries.")
    return complex(arr[0] + 1j * arr[1])
