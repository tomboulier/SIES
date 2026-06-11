"""Transformation-invariant shape descriptors built from CGPT matrices.

Reference: Ammari et al., *Target identification using dictionary
matching of generalized polarization tensors*, FoCM (2014).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sies.asymptotics import cgpt_to_complex, transform_ccgpt_inverse

__all__ = ["ShapeDescriptor"]


@dataclass(frozen=True)
class ShapeDescriptor:
    """Pair of invariant descriptors derived from a CGPT matrix.

    Attributes
    ----------
    i1 : ndarray, shape (k, k)
        First invariant descriptor (modulus of the normalized first
        complex CGPT).
    i2 : ndarray, shape (k, k)
        Second invariant descriptor.
    """

    i1: NDArray
    i2: NDArray

    @property
    def order(self) -> int:
        """int: Maximum CGPT order of the descriptor."""
        return self.i1.shape[0]

    @classmethod
    def from_cgpt(cls, cgpt: NDArray) -> "ShapeDescriptor":
        """Compute invariant descriptors from a CGPT matrix.

        The CGPT is first recentered at the equivalent center of the
        shape (estimated from its own low-order entries), then
        normalized by its diagonal, which removes the dependence on
        translation, rotation and scaling.

        Parameters
        ----------
        cgpt : ndarray, shape (2k, 2k)
            CGPT matrix of the shape.

        Returns
        -------
        ShapeDescriptor
            The invariant descriptors.

        Raises
        ------
        ValueError
            If the CGPT is degenerate (vanishing leading entry or
            diagonal), which prevents the normalization.
        """
        n1, n2 = cgpt_to_complex(cgpt)

        # Estimated (complex) offset of the equivalent center of the shape.
        if np.isclose(np.abs(n2[0, 0]), 0.0):
            raise ValueError("Cannot infer the descriptor center: N2[0, 0] is zero.")
        center = n2[0, 1] / n2[0, 0] / 2
        t1, t2 = transform_ccgpt_inverse(n1, n2, center, 1.0, 0.0)

        # Scaling invariance: normalize by the diagonal of T2 (stable at
        # high orders).
        diag_t2 = np.abs(np.diag(t2))
        if np.any(np.isclose(diag_t2, 0.0)):
            raise ValueError("Cannot normalize the descriptor: zero diagonal in the CGPT.")
        norm = np.diag(1 / np.sqrt(diag_t2))
        s1 = norm @ t1 @ norm
        s2 = norm @ t2 @ norm
        return cls(i1=np.abs(s1), i2=np.abs(s2))

    def distance(self, other: "ShapeDescriptor", order: int | None = None) -> float:
        """Frobenius distance between two descriptors up to a given order.

        Parameters
        ----------
        other : ShapeDescriptor
            Descriptor to compare with.
        order : int, optional
            Truncation order; the full descriptor is used by default.

        Returns
        -------
        float
            The combined Frobenius distance
            ``sqrt(|I1 - I1'|^2 + |I2 - I2'|^2)``.
        """
        order = order or min(self.order, other.order)
        d1 = self.i1[:order, :order] - other.i1[:order, :order]
        d2 = self.i2[:order, :order] - other.i2[:order, :order]
        return float(np.sqrt(np.linalg.norm(d1) ** 2 + np.linalg.norm(d2) ** 2))
