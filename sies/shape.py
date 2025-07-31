import numpy as np

class C2Boundary:
    """Simple C2 boundary representation.

    Parameters
    ----------
    points : array_like
        Array of shape ``(2, N)`` representing boundary points.
    name : str, optional
        Name of the boundary.
    """

    def __init__(self, points, name=""):
        self.points = np.asarray(points, dtype=float)
        if self.points.shape[0] != 2:
            raise ValueError("Points must have shape (2, N)")
        self.name = name
        self.center_of_mass = self.points.mean(axis=1)
        self.nb_points = self.points.shape[1]
        self._compute_geometry()

    def _compute_geometry(self):
        diff = np.roll(self.points, -1, axis=1) - self.points
        norm = np.linalg.norm(diff, axis=0)
        norm[norm == 0] = 1.0
        self.tvec = diff
        self.normal = np.vstack((-diff[1], diff[0])) / norm
        self.tvec_norm = norm
        self.sigma = 2 * np.pi / self.nb_points * norm

    @property
    def box(self):
        d = self.points - self.center_of_mass[:, None]
        w = d[0].max() - d[0].min()
        h = d[1].max() - d[1].min()
        return np.array([w, h])

    @property
    def diameter(self):
        d = self.points - self.center_of_mass[:, None]
        return 2 * np.sqrt((d ** 2).sum(axis=0)).max()

    def isinside(self, x):
        x = np.asarray(x).reshape(2)
        r = np.linalg.norm(x - self.center_of_mass)
        return r < self.diameter / 2
