"""Discretized layer potential operators on smooth closed boundaries.

Each operator exposes a static `kernel_matrix` constructor (the
P0-element stiffness matrix) and, when mathematically well-defined, an
`evaluate` method computing the potential away from the boundary.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from sies.greens import green2d, green2d_grad
from sies.shapes import C2Boundary

__all__ = ["KStar", "SingleLayer", "SingleLayerNormalDerivative"]


class BoundaryOperator(ABC):
    """Abstract base class for boundary integral operators.

    An operator maps densities defined on the boundary `domain` to
    functions sampled on the boundary `image` (which may be the same
    boundary). The discretization uses P0 boundary elements, for which
    the stiffness matrix coincides with the kernel matrix.

    Parameters
    ----------
    domain : C2Boundary
        Boundary supporting the input density.
    image : C2Boundary, optional
        Boundary on which the output is sampled. Defaults to `domain`.

    Attributes
    ----------
    domain : C2Boundary
        Boundary of the operator's domain.
    image : C2Boundary
        Boundary of the operator's image.
    matrix : ndarray
        Kernel (stiffness) matrix of the discretized operator.
    """

    def __init__(self, domain: C2Boundary, image: C2Boundary | None = None):
        self.domain = domain
        self.image = domain if image is None else image
        self.matrix = self._build_matrix()

    @abstractmethod
    def _build_matrix(self) -> NDArray:
        """Assemble the kernel matrix of the operator."""

    def __call__(self, density: NDArray) -> NDArray:
        """Apply the operator to a density sampled on the domain boundary.

        Parameters
        ----------
        density : ndarray, shape (n,)
            Density sampled at the domain boundary points.

        Returns
        -------
        ndarray, shape (m,)
            The image function sampled at the image boundary points.
        """
        return self.matrix @ density


class SingleLayer(BoundaryOperator):
    r"""Single layer potential $S_D$.

    $$S_D[f](x) = \int_{\partial D} G(x - y) f(y) \, ds(y).$$

    The potential is continuous across the boundary; the singular
    diagonal of the kernel matrix is integrated analytically.
    """

    def _build_matrix(self) -> NDArray:
        """Assemble the single layer kernel matrix."""
        if self.image is self.domain:
            return self.kernel_matrix(self.domain.points, self.domain.sigma)
        return self.kernel_matrix(self.domain.points, self.domain.sigma, self.image.points)

    @staticmethod
    def kernel_matrix(
        points: NDArray, sigma: NDArray, image_points: NDArray | None = None
    ) -> NDArray:
        """Kernel matrix of the single layer potential.

        Parameters
        ----------
        points : ndarray, shape (2, n)
            Boundary points supporting the density.
        sigma : ndarray, shape (n,)
            Integration elements of the boundary.
        image_points : ndarray, shape (2, m), optional
            Evaluation boundary, disjoint from `points`. If omitted, the
            operator acts on its own boundary and the diagonal entries
            are computed analytically.

        Returns
        -------
        ndarray, shape (m, n)
            The kernel matrix.
        """
        if image_points is not None:
            return green2d(image_points, points) * sigma

        dx = points[0][:, np.newaxis] - points[0][np.newaxis, :]
        dy = points[1][:, np.newaxis] - points[1][np.newaxis, :]
        dist2 = dx**2 + dy**2
        np.fill_diagonal(dist2, 1.0)  # avoid log(0); diagonal overwritten below
        kernel = np.log(dist2) / (4 * np.pi) * sigma
        np.fill_diagonal(kernel, sigma * (np.log(sigma / 2) - 1) / (2 * np.pi))
        return kernel

    @staticmethod
    def evaluate(boundary: C2Boundary, density: NDArray, points: NDArray) -> NDArray:
        r"""Evaluate $S_D[f]$ at points away from the boundary.

        Parameters
        ----------
        boundary : C2Boundary
            Boundary $\partial D$ supporting the density.
        density : ndarray, shape (n,)
            Density ``f`` sampled on the boundary.
        points : ndarray, shape (2, m)
            Evaluation points, disjoint from the boundary.

        Returns
        -------
        ndarray, shape (m,)
            Values of the potential at the evaluation points.
        """
        kernel = green2d(boundary.points, points)
        return (np.ravel(density) * boundary.sigma) @ kernel

    @staticmethod
    def evaluate_gradient(boundary: C2Boundary, density: NDArray, points: NDArray) -> NDArray:
        r"""Evaluate $\nabla S_D[f]$ at points away from the boundary.

        Parameters
        ----------
        boundary : C2Boundary
            Boundary $\partial D$ supporting the density.
        density : ndarray, shape (n,)
            Density ``f`` sampled on the boundary.
        points : ndarray, shape (2, m)
            Evaluation points, disjoint from the boundary.

        Returns
        -------
        ndarray, shape (2, m)
            Gradient of the potential at the evaluation points.
        """
        gx, gy = green2d_grad(points, boundary.points)
        weights = np.ravel(density) * boundary.sigma
        return np.vstack([gx @ weights, gy @ weights])


class KStar(BoundaryOperator):
    r"""Adjoint Neumann-Poincaré operator $K_D^*$.

    $$
    K_D^*[f](x) = \frac{1}{2\pi} \,\mathrm{p.v.}\!\!\int_{\partial D}
    \frac{\langle x - y, \nu_x \rangle}{|x - y|^2} f(y) \, ds(y).
    $$
    Defined on a single boundary only; the diagonal of the kernel matrix
    involves the curvature of the boundary.
    """

    def __init__(self, domain: C2Boundary):
        super().__init__(domain)

    def _build_matrix(self) -> NDArray:
        """Assemble the Neumann-Poincaré kernel matrix."""
        return self.kernel_matrix(
            self.domain.points,
            self.domain.tvec,
            self.domain.normal,
            self.domain.avec,
            self.domain.sigma,
        )

    @staticmethod
    def kernel_matrix(
        points: NDArray,
        tvec: NDArray,
        normal: NDArray,
        avec: NDArray,
        sigma: NDArray,
    ) -> NDArray:
        """Kernel matrix of the adjoint Neumann-Poincaré operator.

        Parameters
        ----------
        points : ndarray, shape (2, n)
            Boundary points.
        tvec : ndarray, shape (2, n)
            Tangent vectors.
        normal : ndarray, shape (2, n)
            Outward unit normal vectors.
        avec : ndarray, shape (2, n)
            Acceleration vectors (used for the diagonal terms).
        sigma : ndarray, shape (n,)
            Integration elements.

        Returns
        -------
        ndarray, shape (n, n)
            The kernel matrix.
        """
        dx = points[0][:, np.newaxis] - points[0][np.newaxis, :]
        dy = points[1][:, np.newaxis] - points[1][np.newaxis, :]
        dist2 = dx**2 + dy**2
        np.fill_diagonal(dist2, 1.0)  # diagonal overwritten below

        inner = dx * normal[0][:, np.newaxis] + dy * normal[1][:, np.newaxis]
        kernel = inner * sigma / (2 * np.pi * dist2)

        curvature_term = (
            -np.sum(avec * normal, axis=0) * sigma / (4 * np.pi * np.sum(tvec**2, axis=0))
        )
        np.fill_diagonal(kernel, curvature_term)
        return kernel


class SingleLayerNormalDerivative(BoundaryOperator):
    r"""Normal derivative of the single layer potential.

    $$
    \frac{\partial}{\partial\nu} S_D[f](x) =
    \int_{\partial D} \langle \nabla G(x - y), \nu_x \rangle f(y) \, ds(y),
    $$
    defined for two *disjoint* boundaries only (the trace on
    $\partial D$ itself has a jump).
    """

    def __init__(self, domain: C2Boundary, image: C2Boundary):
        if image is domain:
            raise ValueError("This operator is not defined on a single boundary (jump).")
        super().__init__(domain, image)

    def _build_matrix(self) -> NDArray:
        """Assemble the kernel matrix of the normal derivative."""
        return self.kernel_matrix(
            self.domain.points, self.domain.sigma, self.image.points, self.image.normal
        )

    @staticmethod
    def kernel_matrix(
        points: NDArray,
        sigma: NDArray,
        image_points: NDArray,
        image_normal: NDArray,
    ) -> NDArray:
        """Kernel matrix of the normal derivative of the single layer.

        Parameters
        ----------
        points : ndarray, shape (2, n)
            Boundary supporting the density.
        sigma : ndarray, shape (n,)
            Integration elements of the density boundary.
        image_points : ndarray, shape (2, m)
            Evaluation boundary, disjoint from `points`.
        image_normal : ndarray, shape (2, m)
            Outward unit normals of the evaluation boundary.

        Returns
        -------
        ndarray, shape (m, n)
            The kernel matrix.

        Raises
        ------
        ValueError
            If the two boundaries share coincident points (the kernel is
            singular there).
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            gx, gy = green2d_grad(image_points, points)
            kernel = (
                image_normal[0][:, np.newaxis] * gx + image_normal[1][:, np.newaxis] * gy
            ) * sigma
        if not np.all(np.isfinite(kernel)):
            raise ValueError("Domain and image boundaries must be geometrically disjoint.")
        return kernel
