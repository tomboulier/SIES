import numpy as np

class MConfig:
    """Acquisition configuration with sources and receivers.

    Parameters
    ----------
    sources : array_like
        Source coordinates of shape ``(2, Ns)``.
    receivers : array_like
        Receiver coordinates of shape ``(2, Nr)``.
    center : array_like, optional
        Reference center of the system.
    """

    def __init__(self, sources, receivers, center=None):
        self.src_prv = [np.asarray(sources, dtype=float)]
        self.rcv_prv = [np.asarray(receivers, dtype=float)]
        self.Ng = 1
        if center is None:
            self.center = np.mean(np.hstack([self.src_prv[0], self.rcv_prv[0]]), axis=1)
        else:
            self.center = np.asarray(center, dtype=float)

    @property
    def Ns(self):
        return self.src_prv[0].shape[1]

    @property
    def Nr(self):
        return self.rcv_prv[0].shape[1]

    @property
    def Ns_total(self):
        return self.Ns * self.Ng

    @property
    def Nr_total(self):
        return self.Nr * self.Ng

    @property
    def data_dim(self):
        return self.Ns * self.Ng * self.Nr

    def group(self, g):
        return self.src_prv[g], self.rcv_prv[g]

    def src(self, sidx):
        return self.src_prv[0][:, sidx - 1]

    def rcv(self, s):
        return self.rcv_prv[0]
