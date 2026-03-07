import numpy as np
from .base import Shape

class Triangle(Shape):
    def __init__(self, a, angl, nb_points, dspl=10):
        """
        Isosceles triangle.
        a: length of equal sides
        angl: angle between equal sides (radians)
        nb_points: number of discretization points
        dspl: down-sampling factor for smoothing
        """
        self.lside = a
        self.angl = angl

        h = a * np.cos(angl / 2)
        b = a * np.sin(angl / 2)

        t1 = a / (a + b) / 2
        t2 = b / (a + b)
        t3 = t1

        n1 = int(np.floor(t1 * nb_points))
        n2 = int(np.floor(t2 * nb_points))
        n3 = nb_points - n1 - n2

        A = np.array([[0], [2/3*h]])
        B = np.array([[-b], [-h/3]])
        C = np.array([[b], [-h/3]])

        t = np.linspace(0, 1, n1, endpoint=False)
        AB = A + (B - A) @ t[None, :]

        t = np.linspace(0, 1, n2, endpoint=False)
        BC = B + (C - B) @ t[None, :]

        t = np.linspace(0, 1, n3, endpoint=False)
        CA = C + (A - C) @ t[None, :]

        points0 = np.hstack([AB, BC, CA])
        theta0 = np.linspace(0, 2 * np.pi, nb_points, endpoint=False)

        if dspl >= 1:
            t0 = n3 // 2
            points = np.roll(points0, t0, axis=1)
            points, tvec, avec, normal = Shape.rescale(points, theta0, nb_points, dspl=dspl)
        else:
            # Fallback for dspl < 1 (not recommended)
            tvec = np.hstack([
                np.tile((B - A) / t1, (1, n1)),
                np.tile((C - B) / t2, (1, n2)),
                np.tile((A - C) / t3, (1, n3))
            ]) / (2 * np.pi)

            normal = np.vstack([tvec[1, :], -tvec[0, :]])
            normal /= np.sqrt(np.sum(normal**2, axis=0))
            avec = np.zeros((2, nb_points))

            t0 = n3 // 2
            points = np.roll(points0, t0, axis=1)
            tvec = np.roll(tvec, t0, axis=1)
            normal = np.roll(normal, t0, axis=1)

        super().__init__(points, tvec, avec, normal, center_of_mass=[0,0], name='Triangle')

    def __mul__(self, s):
        new_obj = super().__mul__(s)
        new_obj.lside = self.lside * s
        new_obj.angl = self.angl
        return new_obj
