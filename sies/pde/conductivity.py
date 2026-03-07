import numpy as np
from scipy.sparse.linalg import lsqr as sparse_lsqr
from .base import SmallInclusions, SingleLayer_eval
from ..tools.laplacian import Green2D_Dn
from ..asymp import lambda_contrast, make_block_matrix, make_system_matrix_fast
from ..tools import add_white_noise

class Conductivity_R2(SmallInclusions):
    r"""
    Class for the conductivity problem in free space.

    Solves: div(rho_D * grad(u)) = 0 in R^2 \ {x_s}.

    Parameters
    ----------
    inclusions : list of Shape
        List of inclusions.
    cnd : list or ndarray
        Conductivity constants for each inclusion.
    pmtt : list or ndarray
        Permittivity constants for each inclusion.
    cfg : AcquisitionConfig
        Acquisition configuration.
    """

    def __init__(self, inclusions, cnd, pmtt, cfg):
        super().__init__(inclusions, cfg)

        self.cnd = np.atleast_1d(cnd)
        self.pmtt = np.atleast_1d(pmtt)

        if len(self.cnd) < self.nb_incls or len(self.pmtt) < self.nb_incls:
            raise ValueError("Conductivity and permittivity must be specified for each inclusion!")

        for c in self.cnd:
            if np.isclose(c, 1.0) or c < 0:
                raise ValueError("Conductivity must be positive and different from 1!")
        for p in self.pmtt:
            if p < 0:
                raise ValueError("Permittivity must be positive!")

        self.KsdS = make_block_matrix(self.D)
        self.dGdn = self._compute_dGdn()

    def _compute_dGdn(self, sidx=None):
        """Construct the right hand vector of the given source(s)."""
        if sidx is None:
            sidx = np.arange(self.cfg.Ns_total)
        else:
            sidx = np.atleast_1d(sidx)

        nb_points = self.D[0].nb_points
        val = np.zeros((nb_points, self.nb_incls, len(sidx)))

        src = self.cfg.src(sidx)
        for i in range(self.nb_incls):
            toto = Green2D_Dn(src, self.D[i].points, self.D[i].normal)
            val[:, i, :] = toto.T

        return val.reshape(nb_points * self.nb_incls, len(sidx), order='F')

    def _compute_phi(self, freq, sidx=None):
        """Compute the boundary function phi given frequency and source(s)."""
        nb_points = self.D[0].nb_points
        lamb = lambda_contrast(self.cnd, self.pmtt, freq)

        Amat = make_system_matrix_fast(self.KsdS, lamb)

        if sidx is None:
            dGdn = self.dGdn
        else:
            dGdn = self._compute_dGdn(sidx)

        phi = np.linalg.solve(Amat, dGdn) if Amat.shape[0] == Amat.shape[1] else np.linalg.lstsq(Amat, dGdn, rcond=None)[0]

        Phi = []
        idx = 0
        for i in range(self.nb_incls):
            Phi.append(phi[idx : idx + nb_points, :])
            idx += nb_points
        return Phi

    def data_simulation(self, freq_list):
        """
        Simulate MSR data for different frequencies.

        Parameters
        ----------
        freq_list : list or ndarray
            List of working frequencies.

        Returns
        -------
        dict
            Contains "MSR" (list of matrices) and "freq".
        """
        freq_list = np.atleast_1d(freq_list)
        msr_list = []

        for f in freq_list:
            Phi = self._compute_phi(f)
            msr = np.zeros((self.cfg.Ns_total, self.cfg.Nr), dtype=complex)

            for i in range(self.nb_incls):
                toto = np.zeros((self.cfg.Ns_total, self.cfg.Nr), dtype=complex)
                for s in range(self.cfg.Ns_total):
                    rcv = self.cfg.rcv(s)
                    toto[s, :] = SingleLayer_eval(self.D[i], Phi[i][:, s], rcv)
                msr += toto
            msr_list.append(msr)

        return {"MSR": msr_list, "freq": freq_list}

    @staticmethod
    def add_white_noise(data, nlvl):
        """
        Add white noise to simulated data.

        Parameters
        ----------
        data : dict
            Simulated data dictionary.
        nlvl : float
            Noise level.

        Returns
        -------
        dict
            Updated dictionary with "MSR_noisy" and "sigma".
        """
        out = data.copy()
        out["MSR_noisy"] = []
        out["sigma"] = np.zeros(len(data["MSR"]))

        for i, msr in enumerate(data["MSR"]):
            msr_real_noisy, sigma_real = add_white_noise(msr.real, nlvl, mode=1, rowmajor=True)
            msr_imag_noisy, sigma_imag = add_white_noise(msr.imag, nlvl, mode=1, rowmajor=True)

            out["MSR_noisy"].append(msr_real_noisy + 1j * msr_imag_noisy)
            out["sigma"][i] = np.abs(sigma_real + 1j * sigma_imag)

        return out

    @staticmethod
    def make_matrix_A(Xs, Z, order):
        """Construct the linear operator matrix A for sources/receivers."""
        N = Xs.shape[1]
        A = np.zeros((N, 2 * order))
        for n in range(N):
            toto = Xs[:, n] - Z
            R = np.linalg.norm(toto)
            T = np.arctan2(toto[1], toto[0])
            for m in range(1, order + 1):
                A[n, 2*(m-1):2*m] = [np.cos(m*T), np.sin(m*T)] / (2 * np.pi * m * R**m)
        return A

    def make_linop_CGPT(self, ord_val, symmode=False):
        """Make the linear operator for reconstruction."""
        if self.cfg.Ng != 1:
             raise NotImplementedError("make_linop_CGPT for Ng > 1 is not implemented yet.")

        As = self.make_matrix_A(self.cfg.all_src, self.cfg.center, ord_val)
        Ar = self.make_matrix_A(self.cfg.all_rcv, self.cfg.center, ord_val)

        Ms, Ns = As.shape
        Mr, Nr = Ar.shape

        def L(X, transp_flag='notransp'):
            if transp_flag == 'notransp':
                Xm = X.reshape((Ns, Nr), order='F')
                if symmode:
                    Y = As @ (Xm + Xm.T) @ Ar.T
                else:
                    Y = As @ Xm @ Ar.T
                return Y.flatten(order='F')
            elif transp_flag == 'transp':
                Xm = X.reshape((Ms, Mr), order='F')
                if symmode:
                    Y = As.T @ Xm @ Ar + Ar.T @ Xm.T @ As
                else:
                    Y = As.T @ Xm @ Ar
                return Y.flatten(order='F')

        class Op:
            def __init__(self, L_func, As, Ar):
                self.L = L_func
                self.As = As
                self.Ar = Ar

        return Op(L, As, Ar)

    def reconstruct_CGPT(self, MSR_list, ord_val, maxiter=100000, tol=1e-10, symmode=True, method='lsqr'):
        """
        Reconstruct contracted GPT from MSR data.

        Parameters
        ----------
        MSR_list : list of ndarray
            MSR data matrices.
        ord_val : int
            Maximum order of reconstruction.
        maxiter : int, optional
            Maximum iterations for LSQR.
        tol : float, optional
            Tolerance for solver.
        symmode : bool, optional
            If True, assume symmetric CGPT matrix.
        method : {'lsqr', 'pinv'}, optional
            Solver method.

        Returns
        -------
        dict
            Contains "CGPT", "res", and "rres".
        """
        op = self.make_linop_CGPT(ord_val, symmode)
        out = {"CGPT": [], "res": [], "rres": []}

        Ns, Nr = self.cfg.Ns_total, self.cfg.Nr

        for msr in MSR_list:
            if method == 'lsqr':
                from scipy.sparse.linalg import LinearOperator
                lin_op = LinearOperator((Ns * Nr, 4 * ord_val**2), matvec=lambda x: op.L(x, 'notransp'), rmatvec=lambda x: op.L(x, 'transp'))
                b = msr.flatten(order='F')

                res_real = sparse_lsqr(lin_op, b.real, atol=tol, btol=tol, iter_lim=maxiter)
                res_imag = sparse_lsqr(lin_op, b.imag, atol=tol, btol=tol, iter_lim=maxiter)

                X = res_real[0] + 1j * res_imag[0]
                CGPT = X.reshape((2 * ord_val, 2 * ord_val), order='F')

                if symmode:
                    CGPT = (CGPT + CGPT.T)

                out["CGPT"].append(CGPT)
                residual = msr - op.L(CGPT, 'notransp').reshape((Ns, Nr), order='F')
                out["res"].append(np.linalg.norm(residual, 'fro'))
                out["rres"].append(out["res"][-1] / np.linalg.norm(msr, 'fro'))
            else:
                CGPT = np.linalg.pinv(op.As) @ msr @ np.linalg.pinv(op.Ar.T)
                out["CGPT"].append(CGPT)
                residual = msr - op.As @ CGPT @ op.Ar.T
                out["res"].append(np.linalg.norm(residual, 'fro'))
                out["rres"].append(out["res"][-1] / np.linalg.norm(msr, 'fro'))
        return out
