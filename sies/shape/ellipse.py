import numpy as np
from .base import Shape

class Ellipse(Shape):
    def __init__(self, a, b, nb_points):
        """
        Ellipse shape.
        a, b: length of semi-major and semi-minor axis
        nb_points: number of discretization points
        """
        if a < b:
            raise ValueError("Semi-major axis a must be >= semi-minor axis b")

        self.axis_a = a
        self.axis_b = b
        self.phi_rot = 0

        com = np.array([0, 0])
        theta = 2 * np.pi * np.arange(nb_points) / nb_points

        points = np.vstack([a * np.cos(theta), b * np.sin(theta)])
        tvec = np.vstack([-a * np.sin(theta), b * np.cos(theta)])
        avec = np.vstack([-a * np.cos(theta), -b * np.sin(theta)])

        # rotation = [[0 1];[-1 0]] in Matlab
        normal = np.vstack([tvec[1, :], -tvec[0, :]])
        normal /= np.sqrt(np.sum(normal**2, axis=0))

        name = "Circle" if a == b else "Ellipse"
        super().__init__(points, tvec, avec, normal, com, name)

    def __mul__(self, s):
        new_obj = super().__mul__(s)
        new_obj.axis_a = self.axis_a * s
        new_obj.axis_b = self.axis_b * s
        new_obj.phi_rot = self.phi_rot
        return new_obj

    def __matmul__(self, phi):
        new_obj = super().__matmul__(phi)
        new_obj.axis_a = self.axis_a
        new_obj.axis_b = self.axis_b
        new_obj.phi_rot = self.phi_rot + phi
        return new_obj
