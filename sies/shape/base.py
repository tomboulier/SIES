import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class Shape:
    """
    Base class for C2-smooth closed boundary.

    Parameters
    ----------
    points : ndarray
        Coordinates of boundary points, an array of dimension 2 x nb_points.
    tvec : ndarray
        Tangent vectors of boundary points.
    avec : ndarray
        Acceleration vectors of boundary points.
    normal : ndarray
        Outward normal vectors of boundary points.
    center_of_mass : array_like, optional
        Center of mass of the boundary. If None, it is calculated from points.
    name : str, optional
        Name of the shape.
    """

    def __init__(self, points, tvec, avec, normal, center_of_mass=None, name=""):
        self.points = np.asarray(points, dtype=float)
        self.tvec = np.asarray(tvec, dtype=float)
        self.avec = np.asarray(avec, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.name = name

        if center_of_mass is not None:
            self.center_of_mass = np.asarray(center_of_mass, dtype=float).flatten()
        else:
            self.center_of_mass = self.get_com(self.points, self.tvec, self.normal)

    @property
    def nb_points(self):
        """int : Number of discrete boundary points."""
        return self.points.shape[1]

    @property
    def theta(self):
        """ndarray : Non tied-off parameterization between [0, 2pi)."""
        return 2 * np.pi * np.arange(self.nb_points) / self.nb_points

    @property
    def tvec_norm(self):
        """ndarray : Norm of the tangent vectors."""
        return np.sqrt(np.sum(self.tvec**2, axis=0))

    @property
    def sigma(self):
        """ndarray : Element of curve integration (integration weights)."""
        return (2 * np.pi / self.nb_points) * self.tvec_norm

    @property
    def box(self):
        """ndarray : A minimal rectangular box [width, height] containing the shape."""
        d = self.points - self.center_of_mass[:, None]
        w = np.max(d[0, :]) - np.min(d[0, :])
        h = np.max(d[1, :]) - np.min(d[1, :])
        return np.array([w, h])

    @property
    def diameter(self):
        """float : Upper bound of the diameter of the shape."""
        d = self.points - self.center_of_mass[:, None]
        return 2 * np.max(np.sqrt(np.sum(d**2, axis=0)))

    def __add__(self, z0):
        """
        Translate the shape.

        Parameters
        ----------
        z0 : array_like
            Translation vector [dx, dy].

        Returns
        -------
        Shape
            The translated shape.
        """
        z0 = np.asarray(z0).flatten()
        if z0.size != 2:
            raise ValueError("Translation vector must have size 2")
        new_points = self.points + z0[:, None]
        new_com = self.center_of_mass + z0
        return Shape(new_points, self.tvec, self.avec, self.normal, new_com, self.name)

    def __sub__(self, z0):
        """
        Translate the shape (subtraction).

        Parameters
        ----------
        z0 : array_like
            Translation vector [dx, dy].

        Returns
        -------
        Shape
            The translated shape.
        """
        return self.__add__(-np.asarray(z0))

    def __mul__(self, s):
        """
        Scale the shape.

        Parameters
        ----------
        s : float
            Positive scaling factor.

        Returns
        -------
        Shape
            The scaled shape.
        """
        if not isinstance(s, (int, float)) or s <= 0:
            raise ValueError("Scaling factor must be a positive scalar")
        new_points = self.points * s
        new_com = self.center_of_mass * s
        new_tvec = self.tvec * s
        new_avec = self.avec * s
        return Shape(new_points, new_tvec, new_avec, self.normal, new_com, self.name)

    def __matmul__(self, phi):
        """
        Rotate the shape.

        Parameters
        ----------
        phi : float
            Rotation angle in radians.

        Returns
        -------
        Shape
            The rotated shape.
        """
        if not isinstance(phi, (int, float)):
            raise ValueError("Rotation angle must be a scalar")

        rot = np.array([[np.cos(phi), -np.sin(phi)],
                        [np.sin(phi), np.cos(phi)]])

        new_points = rot @ self.points
        new_com = rot @ self.center_of_mass
        new_tvec = rot @ self.tvec
        new_normal = rot @ self.normal
        new_avec = rot @ self.avec

        return Shape(new_points, new_tvec, new_avec, new_normal, new_com, self.name)

    def plot(self, *args, **kwargs):
        """
        Plot the boundary points.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to matplotlib.plt.plot.
        """
        plt.plot(self.points[0, :], self.points[1, :], *args, **kwargs)

    def isinside(self, x):
        """
        Check if a point is inside the bounding ball of the shape.

        Parameters
        ----------
        x : array_like
            Point coordinates [x, y].

        Returns
        -------
        bool
            True if inside, False otherwise.
        """
        x = np.asarray(x).flatten()
        return np.linalg.norm(x - self.center_of_mass) < self.diameter / 2

    def isdisjoint(self, other):
        """
        Check if two shapes are disjoint based on their bounding balls.

        Parameters
        ----------
        other : Shape
            The other shape to check.

        Returns
        -------
        bool
            True if disjoint, False otherwise.
        """
        dist = np.linalg.norm(self.center_of_mass - other.center_of_mass)
        return dist > (self.diameter + other.diameter) / 2

    @staticmethod
    def get_com(points, tvec, normal):
        """
        Calculate the center of mass of a shape using the Stokes formula.

        Parameters
        ----------
        points : ndarray
            Boundary points.
        tvec : ndarray
            Tangent vectors.
        normal : ndarray
            Normal vectors.

        Returns
        -------
        ndarray
            Center of mass coordinates [cx, cy].
        """
        nb_points = points.shape[1]
        tvec_norm = np.sqrt(np.sum(tvec**2, axis=0))
        sigma = (2 * np.pi / nb_points) * tvec_norm

        mass = (np.sum(points[0, :] * normal[0, :] * sigma) +
                np.sum(points[1, :] * normal[1, :] * sigma)) / 2

        cx = np.sum(0.5 * (points[0, :]**2) * normal[0, :] * sigma)
        cy = np.sum(0.5 * (points[1, :]**2) * normal[1, :] * sigma)

        return np.array([cx, cy]) / mass

    @staticmethod
    def boundary_vec_interpl(points0, theta0, theta):
        """
        Interpolate boundary vectors using cubic splines.

        Parameters
        ----------
        points0 : ndarray
            Original boundary points.
        theta0 : ndarray
            Original parameterization.
        theta : ndarray
            New parameterization for resampling.

        Returns
        -------
        tuple
            (points, tvec, avec, normal) interpolated at theta.
        """
        p0_periodic = np.hstack([points0, points0[:, :1]])
        t0_periodic = np.append(theta0, theta0[-1] + (theta0[1]-theta0[0] if len(theta0)>1 else 2*np.pi))

        cs_x = CubicSpline(t0_periodic, p0_periodic[0, :], bc_type='periodic')
        cs_y = CubicSpline(t0_periodic, p0_periodic[1, :], bc_type='periodic')

        px = cs_x(theta)
        py = cs_y(theta)
        points = np.vstack([px, py])

        tx = cs_x(theta, 1)
        ty = cs_y(theta, 1)
        tvec = np.vstack([tx, ty])

        normal = np.vstack([tvec[1, :], -tvec[0, :]])
        normal /= np.sqrt(np.sum(normal**2, axis=0))

        ax = cs_x(theta, 2)
        ay = cs_y(theta, 2)
        avec = np.vstack([ax, ay])

        return points, tvec, avec, normal

    @staticmethod
    def rescale(D0, theta0, nb_points, nsize=None, dspl=1):
        """
        Rescale and resample the boundary.

        Parameters
        ----------
        D0 : ndarray
            Initial boundary points.
        theta0 : ndarray
            Initial parameterization.
        nb_points : int
            Number of points for resampling.
        nsize : tuple, optional
            Target [width, height] for rescaling.
        dspl : int, optional
            Down-sampling factor for smoothing.

        Returns
        -------
        tuple
            (points, tvec, avec, normal) after rescaling and resampling.
        """
        if nsize is not None:
            minx, maxx = np.min(D0[0, :]), np.max(D0[0, :])
            miny, maxy = np.min(D0[1, :]), np.max(D0[1, :])
            z0 = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
            D0 = np.vstack([
                (D0[0, :] - z0[0]) * (nsize[0] / (maxx - minx)),
                (D0[1, :] - z0[1]) * (nsize[1] / (maxy - miny))
            ])

        dspl = int(np.ceil(dspl))
        idx = np.arange(0, D0.shape[1], dspl)

        theta = np.linspace(theta0[0], theta0[-1], nb_points, endpoint=False)

        return Shape.boundary_vec_interpl(D0[:, idx], theta0[idx], theta)
