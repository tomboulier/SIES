"""Catalog of standard parametric shapes.

Every class derives from `C2Boundary` and provides
the analytic tangent, acceleration and normal vectors of its boundary
whenever possible. Shapes with corners (triangle, rectangle) are
smoothed by spline resampling.
"""

import numpy as np
from numpy.typing import NDArray

from sies.shapes.boundary import C2Boundary
from sies.shapes.resampling import resample_curve

__all__ = ["Banana", "Ellipse", "Flower", "Rectangle", "Triangle"]


def _outward_normal(tvec: NDArray) -> NDArray:
    """Rotate tangent vectors to outward unit normals (CCW curves).

    Parameters
    ----------
    tvec : ndarray, shape (2, n)
        Tangent vectors of a counterclockwise-parameterized curve.

    Returns
    -------
    ndarray, shape (2, n)
        Outward unit normal vectors.
    """
    normal = np.vstack([tvec[1], -tvec[0]])
    return normal / np.linalg.norm(normal, axis=0)


class Ellipse(C2Boundary):
    """An ellipse centered at the origin.

    Parameters
    ----------
    axis_a : float
        Length of the semi-major axis.
    axis_b : float
        Length of the semi-minor axis; must satisfy ``axis_b <= axis_a``.
    nb_points : int
        Number of boundary discretization points.

    Attributes
    ----------
    axis_a, axis_b : float
        Semi-axis lengths.
    phi : float
        Cumulated rotation angle of the shape.
    """

    _scale_attrs = ("axis_a", "axis_b")

    def __init__(self, axis_a: float, axis_b: float, nb_points: int):
        if axis_a < axis_b:
            raise ValueError("The semi-major axis must be longer than the semi-minor one.")

        theta = 2 * np.pi * np.arange(nb_points) / nb_points
        points = np.vstack([axis_a * np.cos(theta), axis_b * np.sin(theta)])
        tvec = np.vstack([-axis_a * np.sin(theta), axis_b * np.cos(theta)])
        avec = -points
        name = "Circle" if axis_a == axis_b else "Ellipse"
        super().__init__(points, tvec, avec, _outward_normal(tvec), np.zeros(2), name)

        self.axis_a = axis_a
        self.axis_b = axis_b
        self.phi = 0.0

    def rotate(self, phi: float) -> "Ellipse":
        """Rotate the ellipse around the origin by the angle `phi`."""
        out = super().rotate(phi)
        out.phi = self.phi + phi
        return out


class Flower(C2Boundary):
    r"""A flower-like perturbation of an ellipse.

    The boundary is the curve
    $t \mapsto R_\phi A \, (1 + \epsilon \cos^k(nt)) (\cos t, \sin t)$
    where ``A = diag(a, b)`` and ``n`` is the number of petals.

    Parameters
    ----------
    axis_a : float
        Length of the semi-major axis of the underlying ellipse.
    axis_b : float
        Length of the semi-minor axis.
    nb_points : int
        Number of boundary discretization points.
    nb_petals : int, default 5
        Number of petals.
    epsilon : float, default 0.3
        Strength of the petal perturbation.
    pertb : int, default 1
        Exponent ``k`` of the perturbation; must be a positive integer.

    Attributes
    ----------
    nb_petals : int
        Number of petals.
    epsilon : float
        Perturbation strength.
    """

    _scale_attrs = ("axis_a", "axis_b")

    def __init__(
        self,
        axis_a: float,
        axis_b: float,
        nb_points: int,
        nb_petals: int = 5,
        epsilon: float = 0.3,
        pertb: int = 1,
    ):
        if pertb < 1:
            raise ValueError("The perturbation exponent must be a positive integer.")

        n, k, eps = nb_petals, pertb, epsilon
        theta = 2 * np.pi * np.arange(nb_points) / nb_points
        scale = np.diag([axis_a, axis_b])

        radius = 1 + eps * np.cos(n * theta) ** k
        points = scale @ np.vstack([np.cos(theta) * radius, np.sin(theta) * radius])

        dradius = -k * eps * n * np.sin(n * theta) * np.cos(n * theta) ** (k - 1)
        tvec = scale @ np.vstack(
            [
                -np.sin(theta) * radius + dradius * np.cos(theta),
                np.cos(theta) * radius + dradius * np.sin(theta),
            ]
        )

        ddradius = (
            -k
            * eps
            * n**2
            * (
                np.cos(n * theta) ** k
                - (k - 1) * np.sin(n * theta) ** 2 * np.cos(n * theta) ** (k - 2)
            )
        )
        avec = scale @ np.vstack(
            [
                -np.cos(theta) * radius + 2 * dradius * -np.sin(theta) + ddradius * np.cos(theta),
                -np.sin(theta) * radius + 2 * dradius * np.cos(theta) + ddradius * np.sin(theta),
            ]
        )

        super().__init__(points, tvec, avec, _outward_normal(tvec), np.zeros(2), "Flower")

        self.nb_petals = nb_petals
        self.epsilon = epsilon
        self.axis_a = axis_a
        self.axis_b = axis_b


class Triangle(C2Boundary):
    """An isosceles triangle with smoothed corners.

    Parameters
    ----------
    side : float
        Length of the two equal sides.
    angle : float
        Angle between the two equal sides, in radians.
    nb_points : int
        Number of boundary discretization points.
    downsample : int, default 10
        Down-sampling factor used to smooth out the corners; values
        larger than one produce a $C^2$ boundary.

    Attributes
    ----------
    side : float
        Length of the equal sides.
    angle : float
        Apex angle.
    """

    _scale_attrs = ("side",)

    def __init__(self, side: float, angle: float, nb_points: int, downsample: int = 10):
        height = side * np.cos(angle / 2)
        half_base = side * np.sin(angle / 2)

        vertices = np.array(
            [
                [0, 2 / 3 * height],
                [-half_base, -height / 3],
                [half_base, -height / 3],
            ]
        ).T
        fractions = np.array([side, 2 * half_base, side]) / (2 * (side + half_base))
        points0 = _polygon_points(vertices, fractions, nb_points)

        theta = 2 * np.pi * np.arange(nb_points) / nb_points
        points, tvec, avec, normal = resample_curve(
            points0, theta, nb_points, downsample=downsample
        )
        super().__init__(points, tvec, avec, normal, np.zeros(2), "Triangle")

        self.side = side
        self.angle = angle


class Rectangle(C2Boundary):
    """An axis-aligned rectangle with smoothed corners.

    Parameters
    ----------
    width : float
        Size along the horizontal axis.
    height : float
        Size along the vertical axis.
    nb_points : int
        Number of boundary discretization points.
    downsample : int, default 10
        Down-sampling factor used to smooth out the corners.

    Attributes
    ----------
    width, height : float
        Rectangle dimensions.
    """

    _scale_attrs = ("width", "height")

    def __init__(self, width: float, height: float, nb_points: int, downsample: int = 10):
        a, b = height, width
        vertices = np.array([[-b, a], [-b, -a], [b, -a], [b, a]]).T / 2
        fractions = np.array([a, b, a, b]) / (2 * (a + b))
        points0 = _polygon_points(vertices, fractions, nb_points)

        theta = 2 * np.pi * np.arange(nb_points) / nb_points
        points, tvec, avec, normal = resample_curve(
            points0, theta, nb_points, downsample=downsample
        )
        name = "Square" if width == height else "Rectangle"
        super().__init__(points, tvec, avec, normal, np.zeros(2), name)

        self.width = width
        self.height = height


class Banana(C2Boundary):
    """A banana-shaped object (an ellipse bent along a circular arc).

    Parameters
    ----------
    axis_a : float
        Length of the semi-major axis of the underlying ellipse.
    axis_b : float
        Length of the semi-minor axis.
    center : ndarray, shape (2,)
        Center of the underlying ellipse.
    curvature_center : ndarray, shape (2,)
        Center of the bending circle.
    nb_points : int
        Number of boundary discretization points.
    """

    _scale_attrs = ("axis_a", "axis_b")

    def __init__(
        self,
        axis_a: float,
        axis_b: float,
        center: NDArray,
        curvature_center: NDArray,
        nb_points: int,
    ):
        if axis_a < axis_b:
            raise ValueError("The semi-major axis must be longer than the semi-minor one.")

        x0, y0 = np.asarray(center, dtype=float)
        xc, yc = np.asarray(curvature_center, dtype=float)

        radius = np.hypot(xc - x0, yc - y0)
        theta0 = np.arctan2(y0 - yc, x0 - xc)
        alpha = axis_a / radius
        b = axis_b

        theta = 2 * np.pi * np.arange(nb_points) / nb_points
        t = theta0 + alpha * np.cos(theta)
        rho = radius + b * np.sin(theta)
        points = np.vstack([xc + rho * np.cos(t), yc + rho * np.sin(t)])

        tvec = np.vstack(
            [
                b * np.cos(theta) * np.cos(t) + alpha * np.sin(theta) * rho * np.sin(t),
                b * np.cos(theta) * np.sin(t) - alpha * np.sin(theta) * rho * np.cos(t),
            ]
        )

        avec = np.vstack(
            [
                -b * np.sin(theta) * np.cos(t)
                + alpha * b * np.cos(theta) * np.sin(t)
                + alpha * np.cos(theta) * rho * np.sin(t)
                + alpha * b * np.cos(theta) ** 2 * np.sin(t)
                - alpha**2 * np.sin(theta) ** 2 * rho * np.cos(t),
                -b * np.sin(theta) * np.cos(t)
                - alpha * b * np.sin(theta) * np.cos(t)
                - alpha * np.cos(theta) * rho * np.cos(t)
                - alpha * b * np.cos(theta) ** 2 * np.cos(t)
                - alpha**2 * np.sin(theta) ** 2 * rho * np.sin(t),
            ]
        )

        normal = np.vstack([-tvec[1], tvec[0]])
        normal = normal / np.linalg.norm(normal, axis=0)

        super().__init__(points, tvec, avec, normal, None, "Banana")

        self.axis_a = axis_a
        self.axis_b = axis_b


def _polygon_points(vertices: NDArray, fractions: NDArray, nb_points: int) -> NDArray:
    """Sample the boundary of a polygon edge by edge.

    Parameters
    ----------
    vertices : ndarray, shape (2, k)
        Polygon vertices in counterclockwise order.
    fractions : ndarray, shape (k,)
        Fraction of the points allotted to each edge (sums to one).
    nb_points : int
        Total number of sample points.

    Returns
    -------
    ndarray, shape (2, nb_points)
        Sampled polygon boundary, starting at the middle of the last
        edge (away from any corner).
    """
    nb_edges = vertices.shape[1]
    counts = np.floor(fractions * nb_points).astype(int)
    counts[-1] = nb_points - counts[:-1].sum()

    segments = []
    for i in range(nb_edges):
        start = vertices[:, i, np.newaxis]
        end = vertices[:, (i + 1) % nb_edges, np.newaxis]
        t = np.arange(counts[i]) / counts[i]
        segments.append(start + (end - start) * t)
    points = np.column_stack(segments)

    # Start the curve away from corners (middle of the last edge).
    return np.roll(points, counts[-1] // 2, axis=1)
