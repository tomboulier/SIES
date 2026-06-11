"""Discretized $C^2$-smooth closed boundaries.

A `C2Boundary` stores the sampled boundary points of a simply
connected planar domain together with the tangent, acceleration and
outward normal vectors. It supports rigid motions and scaling, which
return new objects (instances are immutable by convention).
"""

import copy

import numpy as np
from numpy.typing import NDArray

__all__ = ["C2Boundary"]


class C2Boundary:
    """A discretized $C^2$-smooth closed curve in the plane.

    Parameters
    ----------
    points : ndarray, shape (2, n)
        Coordinates of the boundary points. The curve must not be tied
        off: the first and last points are distinct.
    tvec : ndarray, shape (2, n)
        Tangent vectors at the boundary points.
    avec : ndarray, shape (2, n)
        Acceleration (second derivative) vectors.
    normal : ndarray, shape (2, n)
        Outward unit normal vectors.
    center_of_mass : ndarray, shape (2,), optional
        Center of mass of the enclosed domain. Computed from the Stokes
        formula if not provided.
    name : str, default ""
        Human-readable name of the shape.

    Attributes
    ----------
    points : ndarray, shape (2, n)
        Boundary points.
    tvec : ndarray, shape (2, n)
        Tangent vectors.
    avec : ndarray, shape (2, n)
        Acceleration vectors.
    normal : ndarray, shape (2, n)
        Outward unit normal vectors.
    name : str
        Name of the shape.
    """

    def __init__(
        self,
        points: NDArray,
        tvec: NDArray,
        avec: NDArray,
        normal: NDArray,
        center_of_mass: NDArray | None = None,
        name: str = "",
    ):
        for arr in (points, tvec, avec, normal):
            if arr.ndim != 2 or arr.shape[0] != 2:
                raise ValueError("Boundary arrays must have shape (2, n).")
        if not points.shape == tvec.shape == avec.shape == normal.shape:
            raise ValueError("All boundary arrays must have the same shape.")

        self.points = points
        self.tvec = tvec
        self.avec = avec
        self.normal = normal
        self.name = name

        if center_of_mass is None:
            center_of_mass = self.stokes_center_of_mass(points, tvec, normal)
        self.center_of_mass = np.asarray(center_of_mass, dtype=float).reshape(2)

    # ------------------------------------------------------------------
    # Derived geometric quantities
    # ------------------------------------------------------------------
    @property
    def nb_points(self) -> int:
        """int: Number of discretization points."""
        return self.points.shape[1]

    @property
    def theta(self) -> NDArray:
        """ndarray: Uniform parameterization in ``[0, 2 pi)``, not tied off."""
        return 2 * np.pi * np.arange(self.nb_points) / self.nb_points

    @property
    def cpoints(self) -> NDArray:
        """ndarray: Boundary points as complex numbers ``x + i y``."""
        return self.points[0] + 1j * self.points[1]

    @property
    def tvec_norm(self) -> NDArray:
        """ndarray: Euclidean norm of the tangent vectors."""
        return np.linalg.norm(self.tvec, axis=0)

    @property
    def sigma(self) -> NDArray:
        """ndarray: Curve integration elements.

        The boundary integral of a function ``f`` is approximated by
        ``sum(f(points) * sigma)``.
        """
        return 2 * np.pi / self.nb_points * self.tvec_norm

    @property
    def perimeter(self) -> float:
        """float: Perimeter of the boundary."""
        return float(self.sigma.sum())

    @property
    def area(self) -> float:
        """float: Area of the enclosed domain, from the Stokes formula."""
        return float(np.sum(self.points * self.normal * self.sigma) / 2)

    @property
    def diameter(self) -> float:
        """float: Upper bound of the shape diameter, from the center of mass."""
        dd = self.points - self.center_of_mass[:, np.newaxis]
        return 2 * float(np.max(np.linalg.norm(dd, axis=0)))

    @property
    def box(self) -> tuple[float, float]:
        """Tuple of float: Width and height of the minimal bounding box."""
        dd = self.points - self.center_of_mass[:, np.newaxis]
        return float(dd[0].max() - dd[0].min()), float(dd[1].max() - dd[1].min())

    @property
    def principal_direction(self) -> NDArray:
        """ndarray: Principal direction of the shape (unit vector).

        The sign is normalized so that the first nonzero component is
        positive (the direction is defined up to a sign).
        """
        dd = self.points - self.center_of_mass[:, np.newaxis]
        eigvec = np.linalg.svd(dd @ dd.T)[0][:, 0]
        if eigvec[0] < 0 or (eigvec[0] == 0 and eigvec[1] < 0):
            eigvec = -eigvec
        return eigvec

    # ------------------------------------------------------------------
    # Geometric transformations (return new objects)
    # ------------------------------------------------------------------
    def __add__(self, z0: NDArray) -> "C2Boundary":
        """Translate the boundary by the vector `z0`."""
        z0 = np.asarray(z0, dtype=float).reshape(2)
        out = copy.copy(self)
        out.points = self.points + z0[:, np.newaxis]
        out.center_of_mass = self.center_of_mass + z0
        return out

    def __sub__(self, z0: NDArray) -> "C2Boundary":
        """Translate the boundary by the vector ``-z0``."""
        return self + (-np.asarray(z0, dtype=float))

    #: Names of scalar attributes (lengths) that subclasses want rescaled
    #: along with the geometry when the shape is scaled.
    _scale_attrs: tuple[str, ...] = ()

    def __mul__(self, s: float) -> "C2Boundary":
        """Scale the boundary by the positive factor `s`."""
        if s <= 0:
            raise ValueError("Scaling factor must be positive.")
        out = copy.copy(self)
        out.points = self.points * s
        out.tvec = self.tvec * s
        out.avec = self.avec * s
        out.center_of_mass = self.center_of_mass * s
        for attr in self._scale_attrs:
            setattr(out, attr, getattr(self, attr) * s)
        return out

    __rmul__ = __mul__

    def rotate(self, phi: float) -> "C2Boundary":
        """Rotate the boundary around the origin.

        Parameters
        ----------
        phi : float
            Rotation angle in radians (counterclockwise).

        Returns
        -------
        C2Boundary
            The rotated boundary.
        """
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        out = copy.copy(self)
        out.points = rot @ self.points
        out.tvec = rot @ self.tvec
        out.avec = rot @ self.avec
        out.normal = rot @ self.normal
        out.center_of_mass = rot @ self.center_of_mass
        return out

    # ------------------------------------------------------------------
    # Predicates and utilities
    # ------------------------------------------------------------------
    def is_inside(self, x: NDArray) -> bool:
        """Check whether a point lies in the ball circumscribing the shape.

        Parameters
        ----------
        x : ndarray, shape (2,)
            Query point.

        Returns
        -------
        bool
            True if `x` is inside the ball centered at the center of
            mass with radius ``diameter / 2``.
        """
        return bool(
            np.linalg.norm(np.asarray(x).reshape(2) - self.center_of_mass) < self.diameter / 2
        )

    def is_disjoint(self, other: "C2Boundary") -> bool:
        """Check whether two shapes have disjoint circumscribed balls.

        Parameters
        ----------
        other : C2Boundary
            The other shape.

        Returns
        -------
        bool
            True if the balls ``B(com, diameter / 2)`` of both shapes do
            not intersect.
        """
        dist = np.linalg.norm(self.center_of_mass - other.center_of_mass)
        return bool(dist > (self.diameter + other.diameter) / 2)

    def plot(self, ax=None, **kwargs):
        """Plot the boundary curve.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if omitted.
        **kwargs
            Forwarded to `plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        closed = np.column_stack([self.points, self.points[:, :1]])
        ax.plot(closed[0], closed[1], **kwargs)
        ax.set_aspect("equal")
        return ax

    def __repr__(self) -> str:
        name = self.name or type(self).__name__
        return f"<{name}: {self.nb_points} points, center={np.round(self.center_of_mass, 3)}>"

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def stokes_center_of_mass(points: NDArray, tvec: NDArray, normal: NDArray) -> NDArray:
        """Compute the center of mass of a domain by the Stokes formula.

        Parameters
        ----------
        points : ndarray, shape (2, n)
            Boundary points.
        tvec : ndarray, shape (2, n)
            Tangent vectors.
        normal : ndarray, shape (2, n)
            Outward unit normal vectors.

        Returns
        -------
        ndarray, shape (2,)
            Center of mass of the enclosed domain.
        """
        nb_points = points.shape[1]
        sigma = 2 * np.pi / nb_points * np.linalg.norm(tvec, axis=0)
        mass = np.sum(points * normal * sigma) / 2
        first_moment = np.sum(points**2 * normal * sigma, axis=1) / 2
        return first_moment / mass

    @staticmethod
    def check_sampling(points: NDArray) -> bool:
        r"""Check that the sampled curve is locally consistent.

        By a Taylor expansion, a $C^1$ simple curve sampled finely
        enough satisfies
        $\langle f(t_{n+1}) - f(t_n), f(t_n) - f(t_{n-1}) \rangle > 0$.

        Parameters
        ----------
        points : ndarray, shape (2, n)
            Boundary points.

        Returns
        -------
        bool
            True if the condition holds at every point.
        """
        prev = np.roll(points, 1, axis=1)
        nxt = np.roll(points, -1, axis=1)
        forward = nxt - points
        backward = points - prev
        inner = np.sum(forward * backward, axis=0)
        return bool(np.all(inner > 0))
