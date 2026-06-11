r"""The conductivity problem in free space.

The potential $u$ solves

$$
\nabla \cdot (\rho_D \nabla u) = 0 \quad \text{in }
\mathbb{R}^2 \setminus \{x_s\}, \qquad
u - G(\cdot - x_s) = O(|x|^{-1}),
$$
where $\rho_D = 1 + \sum_l (k_l - 1) \chi_{D_l}$ and
$k_l = \sigma_l + i \omega \epsilon_l$. The Multi-Static
Response (MSR) matrix collects the perturbations $u - G$ measured
at the receivers; it is simulated here through a boundary integral
representation, and inverted for the CGPT of the inclusions.

Reference: Ammari et al., *Target identification using dictionary
matching of generalized polarization tensors*, FoCM (2014).
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sies.acquisition import AcquisitionConfig, Concentric
from sies.asymptotics import contrast, make_block_matrix, make_system_matrix
from sies.greens import green2d_dn
from sies.operators import SingleLayer
from sies.shapes import C2Boundary
from sies.utils import add_white_noise

__all__ = ["ConductivityR2", "MSRData", "ReconstructionResult"]


@dataclass
class MSRData:
    """Multi-static response data, possibly at several frequencies.

    Attributes
    ----------
    msr : list of ndarray
        One MSR matrix of shape ``(nb_sources, nb_receivers_per_group)``
        per frequency. Entry ``(s, r)`` is the field perturbation
        produced by source ``s`` and measured at the ``r``-th receiver
        of the source's group (for single-group configurations, simply
        the ``r``-th receiver).
    freqs : list of float
        Working frequencies.
    msr_noisy : list of ndarray
        Noisy version of `msr` (empty until noise is added).
    noise_sigma : list of float
        Standard deviation of the noise added at each frequency.
    """

    msr: list[NDArray]
    freqs: list[float]
    msr_noisy: list[NDArray] = field(default_factory=list)
    noise_sigma: list[float] = field(default_factory=list)


@dataclass
class ReconstructionResult:
    """Result of a CGPT reconstruction.

    Attributes
    ----------
    cgpt : list of ndarray
        Reconstructed CGPT matrix at each frequency.
    residual : list of float
        Norm of the data misfit at each frequency.
    relative_residual : list of float
        Data misfit relative to the data norm.
    source_matrix : ndarray
        Linear operator ``As`` acting on the source side.
    receiver_matrix : ndarray
        Linear operator ``Ar`` acting on the receiver side; the forward
        model is ``MSR = As @ CGPT @ Ar.T``.
    """

    cgpt: list[NDArray]
    residual: list[float]
    relative_residual: list[float]
    source_matrix: NDArray
    receiver_matrix: NDArray


class ConductivityR2:
    """Conductivity problem with small inclusions in free space.

    Parameters
    ----------
    inclusions : C2Boundary or list of C2Boundary
        Mutually disjoint inclusions, all discretized with the same
        number of boundary points.
    cnd : array_like
        Conductivity of each inclusion (positive, different from one).
    pmtt : array_like
        Permittivity of each inclusion (nonnegative).
    cfg : AcquisitionConfig
        Geometry of sources and receivers.

    Attributes
    ----------
    inclusions : list of C2Boundary
        The inclusions.
    cnd, pmtt : ndarray
        Material constants.
    cfg : AcquisitionConfig
        The acquisition configuration.
    """

    def __init__(
        self,
        inclusions: list[C2Boundary] | C2Boundary,
        cnd: NDArray,
        pmtt: NDArray,
        cfg: AcquisitionConfig,
    ):
        if isinstance(inclusions, C2Boundary):
            inclusions = [inclusions]
        nb_points = {incl.nb_points for incl in inclusions}
        if len(nb_points) != 1:
            raise ValueError("All inclusions must have the same number of boundary points.")

        cnd = np.atleast_1d(np.asarray(cnd, dtype=float))
        pmtt = np.atleast_1d(np.asarray(pmtt, dtype=float))
        if len(cnd) != len(inclusions) or len(pmtt) != len(inclusions):
            raise ValueError("Conductivity and permittivity must be given for each inclusion.")
        if np.any(cnd == 1) or np.any(cnd < 0):
            raise ValueError("Conductivity must be nonnegative and different from 1.")
        if np.any(pmtt < 0):
            raise ValueError("Permittivity must be nonnegative.")

        self.inclusions = inclusions
        self.cnd = cnd
        self.pmtt = pmtt
        self.cfg = cfg

        # Frequency-independent quantities, precomputed once.
        self._block_matrix = make_block_matrix(inclusions)
        self._dgdn = self._compute_dgdn()

    # ------------------------------------------------------------------
    # Forward problem
    # ------------------------------------------------------------------
    def _compute_dgdn(self) -> NDArray:
        """Compute the normal derivative of the source Green's functions.

        Returns
        -------
        ndarray, shape (L * n, nb_sources)
            Right-hand sides of the boundary integral system, stacked by
            inclusion.
        """
        sources = self.cfg.all_sources
        blocks = [green2d_dn(sources, incl.points, incl.normal).T for incl in self.inclusions]
        return np.concatenate(blocks, axis=0)

    def _compute_densities(self, freq: float) -> list[NDArray]:
        """Solve the boundary integral system at a given frequency.

        Parameters
        ----------
        freq : float
            Working frequency.

        Returns
        -------
        list of ndarray
            For each inclusion, the densities ``phi`` of all sources as
            an array of shape ``(n, nb_sources)``.
        """
        lambdas = contrast(self.cnd, self.pmtt, freq)
        system = make_system_matrix(self._block_matrix, lambdas)
        phi = np.linalg.solve(system, self._dgdn)

        nb_points = self.inclusions[0].nb_points
        return [phi[i * nb_points : (i + 1) * nb_points] for i in range(len(self.inclusions))]

    def simulate_data(self, freqs: float | list[float] = 0.0) -> MSRData:
        r"""Simulate the MSR matrices at the given frequencies.

        The perturbation measured by a receiver is represented as a sum
        of single layer potentials,
        $u - G = \sum_l S_{D_l}[\phi_l]$.

        Parameters
        ----------
        freqs : float or list of float, default 0.0
            Working frequencies.

        Returns
        -------
        MSRData
            The simulated data.
        """
        freqs = np.atleast_1d(np.asarray(freqs, dtype=float))

        msr_list = []
        for freq in freqs:
            densities = self._compute_densities(freq)
            dtype = complex if any(np.iscomplexobj(p) for p in densities) else float
            msr = np.zeros((self.cfg.nb_sources, self.cfg.nb_receivers_per_group), dtype=dtype)
            for incl, phi in zip(self.inclusions, densities, strict=True):
                for s in range(self.cfg.nb_sources):
                    rcv = self.cfg.receivers_of_source(s)
                    msr[s] += SingleLayer.evaluate(incl, phi[:, s], rcv)
            msr_list.append(msr)

        return MSRData(msr=msr_list, freqs=list(freqs))

    @staticmethod
    def add_white_noise(
        data: MSRData, level: float, rng: np.random.Generator | None = None
    ) -> MSRData:
        """Add white noise to simulated MSR data.

        The real and imaginary parts are corrupted independently, source
        by source.

        Parameters
        ----------
        data : MSRData
            Simulated data.
        level : float
            Noise level (e.g. ``0.1`` for 10% noise).
        rng : numpy.random.Generator, optional
            Random generator, for reproducibility.

        Returns
        -------
        MSRData
            A new data object with the `msr_noisy` and `noise_sigma`
            fields filled in.
        """
        rng = rng or np.random.default_rng()
        noisy, sigmas = [], []
        for msr in data.msr:
            real, sigma_r = add_white_noise(msr.real, level, rng)
            imag, sigma_i = add_white_noise(msr.imag, level, rng)
            noisy.append(real + 1j * imag if np.iscomplexobj(msr) else real)
            sigmas.append(abs(sigma_r + 1j * sigma_i))
        return MSRData(msr=data.msr, freqs=data.freqs, msr_noisy=noisy, noise_sigma=sigmas)

    # ------------------------------------------------------------------
    # Inverse problem: CGPT reconstruction
    # ------------------------------------------------------------------
    @staticmethod
    def make_matrix_a(points: NDArray, center: NDArray, order: int) -> NDArray:
        r"""Acquisition matrix of the linearized CGPT forward model.

        Row ``i`` contains the coefficients
        $[\cos(m\theta_i), \sin(m\theta_i)] / (2\pi m R_i^m)$
        for ``m = 1..order``, where $(R_i, \theta_i)$ are the
        polar coordinates of the ``i``-th point relative to `center`.

        Parameters
        ----------
        points : ndarray, shape (2, n)
            Coordinates of the sources or receivers.
        center : ndarray, shape (2,)
            Reference center.
        order : int
            Maximum CGPT order.

        Returns
        -------
        ndarray, shape (n, 2 * order)
            The acquisition matrix.
        """
        delta = points - np.asarray(center, dtype=float).reshape(2, 1)
        radius = np.linalg.norm(delta, axis=0)
        angle = np.arctan2(delta[1], delta[0])

        m = np.arange(1, order + 1)
        weights = 1 / (2 * np.pi * m * radius[:, np.newaxis] ** m)
        matrix = np.empty((points.shape[1], 2 * order))
        matrix[:, 0::2] = np.cos(angle[:, np.newaxis] * m) * weights
        matrix[:, 1::2] = np.sin(angle[:, np.newaxis] * m) * weights
        return matrix

    def make_linear_operator(self, order: int) -> tuple[NDArray, NDArray]:
        """Build the matrices of the forward model ``MSR = As CGPT Ar^T``.

        Only single-group (full or sparse view) configurations are
        supported.

        Parameters
        ----------
        order : int
            Maximum CGPT order.

        Returns
        -------
        source_matrix : ndarray, shape (nb_sources, 2 * order)
            Matrix ``As`` acting on the source side.
        receiver_matrix : ndarray, shape (nb_receivers, 2 * order)
            Matrix ``Ar`` acting on the receiver side.
        """
        if self.cfg.nb_groups != 1:
            raise NotImplementedError("Grouped (limited-view) operators are not supported.")
        src_matrix = self.make_matrix_a(self.cfg.all_sources, self.cfg.center, order)
        rcv_matrix = self.make_matrix_a(self.cfg.all_receivers, self.cfg.center, order)
        return src_matrix, rcv_matrix

    def reconstruct_cgpt(self, msr: NDArray | list[NDArray], order: int) -> ReconstructionResult:
        """Reconstruct the CGPT from MSR data by least squares.

        Parameters
        ----------
        msr : ndarray or list of ndarray
            MSR matrix (or one matrix per frequency).
        order : int
            Maximum order of the reconstruction.

        Returns
        -------
        ReconstructionResult
            The reconstructed CGPT and residuals.
        """
        src_matrix, rcv_matrix = self.make_linear_operator(order)
        return self._invert(msr, src_matrix, rcv_matrix)

    def reconstruct_cgpt_analytic(
        self, msr: NDArray | list[NDArray], order: int
    ) -> ReconstructionResult:
        r"""Reconstruct the CGPT with the closed-form least-squares inverse.

        For equispaced full-view circular configurations the acquisition
        matrices satisfy ``Cs' Cs = Ns / 2 I``, which yields the explicit
        solution

        $$M = \frac{4}{N_s N_r} D_s^{-1} C_s^T \, \mathrm{MSR} \, C_r D_r^{-1}.$$

        Parameters
        ----------
        msr : ndarray or list of ndarray
            MSR matrix (or one matrix per frequency).
        order : int
            Maximum order of the reconstruction. Internally capped at
            ``(min(Ns, Nr) - 1) / 2``, the maximum resolvable order.

        Returns
        -------
        ReconstructionResult
            The reconstructed CGPT and residuals.
        """
        cfg = self.cfg
        if not (isinstance(cfg, Concentric) and cfg.equispaced):
            raise ValueError(
                "Analytic reconstruction requires an equispaced concentric configuration."
            )

        nb_src, nb_rcv = cfg.nb_sources, cfg.nb_receivers
        order = min(order, (nb_src - 1) // 2, (nb_rcv - 1) // 2)

        cs, ds = _make_matrix_cd(nb_src, cfg.radius_src, order)
        cr, dr = _make_matrix_cd(nb_rcv, cfg.radius_rcv, order)

        msr_list = [msr] if not isinstance(msr, list) else msr
        ids = np.diag(1 / np.diag(ds))
        idr = np.diag(1 / np.diag(dr))

        cgpts, residuals, relative = [], [], []
        for data in msr_list:
            cgpt = 4 * ids @ cs.T @ data @ cr @ idr / (nb_src * nb_rcv)
            res = float(np.linalg.norm(data - cs @ ds @ cgpt @ (cr @ dr).T))
            cgpts.append(cgpt)
            residuals.append(res)
            relative.append(res / float(np.linalg.norm(data)))

        return ReconstructionResult(
            cgpt=cgpts,
            residual=residuals,
            relative_residual=relative,
            source_matrix=cs @ ds,
            receiver_matrix=cr @ dr,
        )

    @staticmethod
    def _invert(
        msr: NDArray | list[NDArray], src_matrix: NDArray, rcv_matrix: NDArray
    ) -> ReconstructionResult:
        """Solve ``MSR = As CGPT Ar^T`` by pseudo-inversion.

        Parameters
        ----------
        msr : ndarray or list of ndarray
            MSR matrices.
        src_matrix, rcv_matrix : ndarray
            Acquisition matrices ``As`` and ``Ar``.

        Returns
        -------
        ReconstructionResult
            The reconstructed CGPT and residuals.
        """
        msr_list = [msr] if not isinstance(msr, list) else msr
        src_pinv = np.linalg.pinv(src_matrix)
        rcv_pinv = np.linalg.pinv(rcv_matrix)

        cgpts, residuals, relative = [], [], []
        for data in msr_list:
            cgpt = src_pinv @ data @ rcv_pinv.T
            res = float(np.linalg.norm(data - src_matrix @ cgpt @ rcv_matrix.T))
            cgpts.append(cgpt)
            residuals.append(res)
            relative.append(res / float(np.linalg.norm(data)))

        return ReconstructionResult(
            cgpt=cgpts,
            residual=residuals,
            relative_residual=relative,
            source_matrix=src_matrix,
            receiver_matrix=rcv_matrix,
        )

    def plot(self, ax=None, **kwargs):
        """Plot the inclusions and the acquisition system.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if omitted.
        **kwargs
            Forwarded to the inclusion plot calls.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        for incl in self.inclusions:
            incl.plot(ax=ax, **kwargs)
        self.cfg.plot(ax=ax)
        return ax


def _make_matrix_cd(nb: int, radius: float, order: int) -> tuple[NDArray, NDArray]:
    """Factor the acquisition matrix of an equispaced circle as ``C D``.

    Parameters
    ----------
    nb : int
        Number of equispaced sources or receivers.
    radius : float
        Radius of the measurement circle.
    order : int
        Maximum CGPT order.

    Returns
    -------
    c : ndarray, shape (nb, 2 * order)
        Angular factor with orthogonality ``C^T C = nb / 2 * I``.
    d : ndarray, shape (2 * order, 2 * order)
        Diagonal radial factor.
    """
    theta = 2 * np.pi * np.arange(nb) / nb
    orders = np.arange(1, order + 1)
    phase = theta[:, np.newaxis] * orders

    c = np.empty((nb, 2 * order))
    c[:, 0::2] = np.cos(phase)
    c[:, 1::2] = np.sin(phase)

    d = np.diag(np.repeat(1 / (2 * np.pi * orders * radius**orders), 2))
    return c, d
