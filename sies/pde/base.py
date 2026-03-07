import numpy as np
from ..tools.laplacian import Green2D_Dn, Green2D_Grad

class SmallInclusions:
    """Abstract class for several PDEs involving small inclusions in a homogeneous medium."""

    def __init__(self, inclusions, cfg):
        if not isinstance(inclusions, list):
            inclusions = [inclusions]

        self.D = []
        self.nb_incls = 0
        self.cfg = cfg

        for d in inclusions:
            self.add_inclusion(d)

    def add_inclusion(self, inclusion):
        if self.nb_incls >= 1:
            if self.D[self.nb_incls - 1].nb_points != inclusion.nb_points:
                raise ValueError("Number of boundary discretization points must be the same for all inclusions.")
            if not self.check_inclusions(inclusion):
                raise ValueError("Inclusions must be separated from each other.")

        self.D.append(inclusion)
        self.nb_incls += 1

    def check_inclusions(self, inclusion):
        for d in self.D:
            if not inclusion.isdisjoint(d):
                return False
        return True

    def plot(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        for d in self.D:
            d.plot(*args, **kwargs)
        self.cfg.plot(*args, **kwargs)

    def data_simulation(self, *args, **kwargs):
        raise NotImplementedError("data_simulation must be implemented by subclasses")

def Green2D(X, Y):
    """
    2D Green function G(x) = 1/2/pi * log|x| evaluated at the points X-Y.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    XY1 = X[0, :, None] - Y[0, None, :]
    XY2 = X[1, :, None] - Y[1, None, :]
    DN = XY1**2 + XY2**2
    # G = 1/2/pi * log|x| = 1/4/pi * log|x|^2
    G = (1 / (4 * np.pi)) * np.log(DN)
    return G

def SingleLayer_eval(D, F, X):
    """
    Evaluate the single layer potential S_D[F](X)
    """
    # G is (N_D, N_X) if we use Green2D(D.points, X)
    G = Green2D(D.points, X)
    # val = (F * sigma) * G
    val = (F.flatten() * D.sigma) @ G
    return val
