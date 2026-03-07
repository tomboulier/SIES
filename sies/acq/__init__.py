import numpy as np
import matplotlib.pyplot as plt

class AcquisitionConfig:
    """
    Abstract class for the configuration of acquisition system.

    An acquisition system consists of sources and receivers. This class
    contains only the geometrical properties, such as the position of
    sources and receivers, but no physical properties like frequency.

    Parameters
    ----------
    src_prv : list of ndarray
        Coordinates of sources by group.
    rcv_prv : list of ndarray
        Coordinates of receivers by group.
    center : array_like, optional
        Reference center of the measurement system.
    """

    def __init__(self, src_prv, rcv_prv, center=None):
        self.src_prv = [np.asarray(s) for s in src_prv]
        self.rcv_prv = [np.asarray(r) for r in rcv_prv]
        self.Ng = len(self.src_prv)
        self.center = np.asarray(center).flatten() if center is not None else np.array([0, 0])

        self.Ns = self.src_prv[0].shape[1] if self.Ng > 0 else 0
        self.Nr = self.rcv_prv[0].shape[1] if self.Ng > 0 else 0

        self.Ns_total = self.Ns * self.Ng
        self.Nr_total = self.Nr * self.Ng
        self.data_dim = self.Ns * self.Ng * self.Nr

    def group(self, g):
        """
        Get coordinates of sources and receivers for a specific group.

        Parameters
        ----------
        g : int
            Group index (0-indexed).

        Returns
        -------
        tuple
            (src, rcv) coordinates for the group.
        """
        return self.src_prv[g], self.rcv_prv[g]

    def src_query(self, s):
        """
        For a global source index, get group index and local index.

        Parameters
        ----------
        s : int
            Global source index (0-indexed).

        Returns
        -------
        tuple
            (gid, sid) group index and local source index.
        """
        if s >= self.Ns_total or s < 0:
            raise IndexError("Source index out of range")
        gid = s // self.Ns
        sid = s % self.Ns
        return gid, sid

    def src(self, sidx=None):
        """
        Get coordinates of specific sources.

        Parameters
        ----------
        sidx : int or list of int, optional
            Index or indices of sources. If None, all sources are returned.

        Returns
        -------
        ndarray
            Coordinates of requested sources.
        """
        if sidx is None:
            sidx = np.arange(self.Ns_total)

        indices = np.atleast_1d(sidx)
        val = np.zeros((2, len(indices)))
        for i, s in enumerate(indices):
            gid, sid = self.src_query(s)
            val[:, i] = self.src_prv[gid][:, sid]

        return val if not isinstance(sidx, (int, np.integer)) else val[:, 0]

    def rcv(self, s):
        """
        Get coordinates of receivers responding to a specific source.

        Parameters
        ----------
        s : int
            Global source index (0-indexed).

        Returns
        -------
        ndarray
            Coordinates of receivers in the same group as source s.
        """
        gid, _ = self.src_query(s)
        return self.rcv_prv[gid]

    @property
    def all_src(self):
        """ndarray : Coordinates of all sources concatenated."""
        return np.hstack(self.src_prv)

    @property
    def all_rcv(self):
        """ndarray : Coordinates of all receivers concatenated."""
        return np.hstack(self.rcv_prv)

    def plot(self, *args, **kwargs):
        """
        Plot sources and receivers.
        """
        for g in range(self.Ng):
            src, rcv = self.group(g)
            plt.plot(src[0, :], src[1, :], 'x', label=f'Group {g} sources')
            plt.plot(rcv[0, :], rcv[1, :], 'o', label=f'Group {g} receivers')
        plt.plot(self.center[0], self.center[1], 'r*')
        plt.legend()


def src_rcv_circle(Na, N0, R0, Z, theta, aov=2*np.pi):
    """
    Generate sources/receivers placed on concentric arcs.

    Parameters
    ----------
    Na : int
        Number of arcs.
    N0 : int
        Number of sources/receivers per arc.
    R0 : float
        Radius of measurement circle.
    Z : array_like
        Center of measurement circle.
    theta : float
        Angular aperture of each arc.
    aov : float, optional
        Total angle of view coverage.

    Returns
    -------
    tuple
        (Xs, Thetas, Xscell) coordinates, angles, and grouped list of points.
    """
    import numpy as np
    Xs = np.zeros((2, N0 * Na))
    Thetas = np.zeros(N0 * Na)
    Xscell = []

    Z = np.asarray(Z).flatten()

    for n in range(Na):
        tt0 = n / Na * aov
        tt = tt0 + np.arange(N0) / N0 * theta

        rr = R0 * np.vstack([np.cos(tt), np.sin(tt)])
        start_idx = n * N0
        end_idx = (n + 1) * N0
        Xs[:, start_idx:end_idx] = rr
        Thetas[start_idx:end_idx] = tt
        Xscell.append(rr + Z[:, None])

    Xs = Xs + Z[:, None]
    return Xs, Thetas, Xscell

class Concentric(AcquisitionConfig):
    """
    Concentric configuration for sources and receivers.

    Parameters
    ----------
    Z : array_like
        Center of the measurement circle.
    Rs : float
        Radius of source circle.
    Ns : int
        Number of sources per arc.
    Rr : float
        Radius of receiver circle.
    Nr : int
        Number of receivers per arc.
    viewmode : tuple, optional
        (Na, theta, aov) specifying number of arcs, aperture, and total coverage.
    grouped : bool, optional
        If True, each arc is a separate group.
    neutCoeff : list or ndarray, optional
        Coefficients for neutral source condition.
    neutRad : float, optional
        Relative distance between Diracs for neutral source.
    """

    def __init__(self, Z, Rs, Ns, Rr, Nr, viewmode=(1, 2*np.pi, 2*np.pi), grouped=False, neutCoeff=None, neutRad=0.01):
        Na, theta, aov = viewmode

        Xs, _, Xscell = src_rcv_circle(Na, Ns, Rs, Z, theta, aov)
        Xr, _, Xrcell = src_rcv_circle(Na, Nr, Rr, Z, theta, aov)

        if grouped:
            src_prv = Xscell
            rcv_prv = Xrcell
        else:
            src_prv = [Xs]
            rcv_prv = [Xr]

        super().__init__(src_prv, rcv_prv, center=Z)

        self.radius_src = Rs
        self.radius_rcv = Rr
        self.neutRad = neutRad

        if neutCoeff is None or len(np.atleast_1d(neutCoeff)) <= 1:
            self.neutCoeff = np.array([1.0])
        else:
            self.neutCoeff = np.asarray(neutCoeff)
            if not np.isclose(np.sum(self.neutCoeff), 0) or np.all(self.neutCoeff == 0):
                 raise ValueError("Coefficients of Diracs must satisfy neutrality condition (sum=0) and be non-zero!")

    @property
    def nbDirac(self):
        """int : Number of Diracs for the neutral source."""
        return len(self.neutCoeff)

    def neutSrc(self, s):
        """
        Get the positions of Diracs for the s-th source.

        Parameters
        ----------
        s : int
            Global source index (0-indexed).

        Returns
        -------
        ndarray
            Positions of Diracs (2 x nbDirac).
        """
        psrc = self.src(s)
        if self.nbDirac == 1:
            return psrc[:, None]
        else:
            val = np.zeros((2, self.nbDirac))
            L = self.radius_src * self.neutRad
            toto = psrc - self.center
            q = np.array([toto[1], -toto[0]]) # tangent direction
            q = q / np.linalg.norm(q) * L

            for n in range(self.nbDirac):
                val[:, n] = psrc + (n / self.nbDirac) * q
            return val

class Coincided(Concentric):
    """
    Coincided sources and receivers on a circle.
    """
    def __init__(self, Z, Rs, Ns, viewmode=(1, 2*np.pi, 2*np.pi), grouped=False, neutCoeff=None, neutRad=0.01):
        super().__init__(Z, Rs, Ns, Rs, Ns, viewmode, grouped, neutCoeff, neutRad)
