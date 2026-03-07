import numpy as np
import scipy.linalg
from ..tools.laplacian import Green2D_Grad

def lambda_contrast(cnd, pmtt=None, freq=0):
    """
    Compute the contrast lambda from the conductivity, the permittivity and the frequency.
    """
    cnd = np.atleast_1d(cnd)
    if pmtt is None:
        pmtt = np.zeros_like(cnd)
    else:
        pmtt = np.atleast_1d(pmtt)

    if freq < 0:
        raise ValueError("Frequency must be a positive scalar.")

    for c in cnd:
        if np.isclose(c, 1.0) or c < 0:
            raise ValueError("Invalid value of conductivity.")

    # Using convention (1) from Matlab code: f^(w) = \int f(x) exp(-2*pi*1i*x*w) dx
    toto = cnd + 2 * np.pi * 1j * pmtt * freq
    return (toto + 1) / (toto - 1) / 2

def make_kernel_matrix_Kstar(points, tvec, normal, avec, sigma):
    M = points.shape[1]
    Ks = np.zeros((M, M))
    tvec_norm_square = np.sum(tvec**2, axis=0)

    for j in range(M):
        # xdoty = <D(:,j) - D, normal(:,j)>
        diff = points[:, j, None] - points
        xdoty = diff[0, :] * normal[0, j] + diff[1, :] * normal[1, j]
        norm_xy_square = np.sum(diff**2, axis=0)

        # Avoid division by zero at diagonal
        mask = np.ones(M, dtype=bool)
        mask[j] = False

        Ks[j, mask] = (1 / (2 * np.pi)) * xdoty[mask] * (sigma[mask] / norm_xy_square[mask])

        # Diagonal term
        # avec[:,j] is (2,), normal[:,j] is (2,)
        diag_val = (1 / (2 * np.pi)) * (-0.5) * np.dot(avec[:, j], normal[:, j]) / tvec_norm_square[j] * sigma[j]
        Ks[j, j] = diag_val

    return Ks

def make_kernel_matrix_dSLdn(D_points, D_sigma, E_points, E_normal):
    Gx, Gy = Green2D_Grad(E_points, D_points) # (N_E, N_D)
    # K = (diag(normal_E(1,:))*Gx + diag(normal_E(2,:))*Gy) * diag(sigma_D)
    K = (E_normal[0, :, None] * Gx + E_normal[1, :, None] * Gy) * D_sigma[None, :]
    return K

def make_block_matrix(inclusions):
    nb_incls = len(inclusions)
    KsdS = [[None for _ in range(nb_incls)] for _ in range(nb_incls)]

    for m in range(nb_incls):
        for n in range(nb_incls):
            if m == n:
                KsdS[m][n] = -1.0 * make_kernel_matrix_Kstar(
                    inclusions[n].points, inclusions[n].tvec, inclusions[n].normal,
                    inclusions[n].avec, inclusions[n].sigma
                )
            else:
                KsdS[m][n] = -1.0 * make_kernel_matrix_dSLdn(
                    inclusions[n].points, inclusions[n].sigma,
                    inclusions[m].points, inclusions[m].normal
                )
    return KsdS

def make_system_matrix_fast(KsdS, lambda_val):
    nb_incls = len(KsdS)
    lambda_val = np.atleast_1d(lambda_val)

    Acell = [[None for _ in range(nb_incls)] for _ in range(nb_incls)]
    for m in range(nb_incls):
        for n in range(nb_incls):
            if m == n:
                Acell[m][n] = lambda_val[m] * np.eye(KsdS[m][n].shape[0]) + KsdS[m][n]
            else:
                Acell[m][n] = KsdS[m][n]

    return np.block(Acell)

def cell2mat(CC, CS, SC, SS):
    ord_val = CC.shape[0]
    M = np.zeros((2 * ord_val, 2 * ord_val))
    M[0::2, 0::2] = CC
    M[0::2, 1::2] = CS
    M[1::2, 0::2] = SC
    M[1::2, 1::2] = SS
    return M

def theoretical_CGPT(inclusions, lambda_val, ord_val):
    KsdS = make_block_matrix(inclusions)
    return theoretical_CGPT_fast(inclusions, KsdS, lambda_val, ord_val)

def theoretical_CGPT_fast(inclusions, KsdS, lambda_val, ord_val):
    nb_points = inclusions[0].nb_points
    nb_incls = len(inclusions)
    lambda_val = np.atleast_1d(lambda_val)

    Amat = make_system_matrix_fast(KsdS, lambda_val)

    epsilon = 1e-8
    if np.min(np.abs(lambda_val - 0.5)) < epsilon:
        # Extra condition of L^2_0 function
        # [Amat; kron(eye(nbIncls), ones(1, size(Amat0,2)/nbIncls))]
        extra = np.zeros((nb_incls, Amat.shape[1]))
        for i in range(nb_incls):
            extra[i, i*nb_points:(i+1)*nb_points] = 1.0
        Amat = np.vstack([Amat, extra])

    CC = np.zeros((ord_val, ord_val))
    CS = np.zeros((ord_val, ord_val))
    SC = np.zeros((ord_val, ord_val))
    SS = np.zeros((ord_val, ord_val))

    for m in range(1, ord_val + 1):
        B = np.zeros((nb_points, nb_incls), dtype=complex)
        for i in range(nb_incls):
            # cpoints = points[0] + 1j * points[1]
            cpoints = inclusions[i].points[0, :] + 1j * inclusions[i].points[1, :]
            dm = m * (cpoints**(m-1))
            toto = inclusions[i].normal[0, :] * dm + inclusions[i].normal[1, :] * dm * 1j
            B[:, i] = toto

        b = B.flatten(order='F') # Column-major flattening
        if np.min(np.abs(lambda_val - 0.5)) < epsilon:
            b_real = np.append(b.real, np.zeros(nb_incls))
            b_imag = np.append(b.imag, np.zeros(nb_incls))
        else:
            b_real = b.real
            b_imag = b.imag

        # Solve Amat \ b
        phi_m_real_vec = np.linalg.solve(Amat, b_real) if Amat.shape[0] == Amat.shape[1] else np.linalg.lstsq(Amat, b_real, rcond=None)[0]
        phi_m_imag_vec = np.linalg.solve(Amat, b_imag) if Amat.shape[0] == Amat.shape[1] else np.linalg.lstsq(Amat, b_imag, rcond=None)[0]

        # reshape back to (nb_points, nb_incls)
        # We only take the first nb_points * nb_incls elements if we added extra rows
        realphim = phi_m_real_vec[:nb_points*nb_incls].reshape((nb_points, nb_incls), order='F')
        imagphim = phi_m_imag_vec[:nb_points*nb_incls].reshape((nb_points, nb_incls), order='F')

        for n in range(1, ord_val + 1):
            for i in range(nb_incls):
                cpoints = inclusions[i].points[0, :] + 1j * inclusions[i].points[1, :]
                zn = (cpoints**n) * inclusions[i].sigma

                CC[m-1, n-1] += np.sum(zn.real * realphim[:, i].real)
                CS[m-1, n-1] += np.sum(zn.imag * realphim[:, i].real)
                SC[m-1, n-1] += np.sum(zn.real * imagphim[:, i].real)
                SS[m-1, n-1] += np.sum(zn.imag * imagphim[:, i].real)

    return cell2mat(CC, CS, SC, SS)
