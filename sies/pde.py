import numpy as np
from .shape import C2Boundary
from .acq import MConfig

class SmallInclusions:
    """Base class for PDEs with small inclusions."""

    def __init__(self, inclusions, cfg):
        if not isinstance(cfg, MConfig):
            raise TypeError("cfg must be an instance of MConfig")
        self.cfg = cfg
        self.D = []
        self.nb_incls = 0
        if isinstance(inclusions, C2Boundary):
            self.add_inclusion(inclusions)
        else:
            for inc in inclusions:
                self.add_inclusion(inc)

    def add_inclusion(self, inc):
        if not isinstance(inc, C2Boundary):
            raise TypeError("inclusion must be C2Boundary")
        self.D.append(inc)
        self.nb_incls += 1


class ElectricFish(SmallInclusions):
    """Simplified electric fish model."""

    def __init__(self, D, cnd, pmtt, cfg, step_bem=1):
        super().__init__(D, cfg)
        self.Omega = cfg.Omega0 if hasattr(cfg, "Omega0") else cfg.rcv_prv[0]
        self.impd = getattr(cfg, "impd", 0.0)
        self.cnd = np.asarray(cnd, dtype=float)
        self.pmtt = np.asarray(pmtt, dtype=float)
        self.step_bem1 = step_bem
        self.step_bem2 = 1
        self.Psi = np.eye(self.Omega.nb_points)[::step_bem]

    @property
    def grammatrix(self):
        sigma = np.ones(self.Psi.shape[0])
        return self.Psi.T @ np.diag(sigma) @ self.Psi

