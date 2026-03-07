import numpy as np

def add_white_noise(X, nlvl, mode=0, rowmajor=False):
    """
    Add white noise to data.
    X: input data matrix
    nlvl: noise level
    mode: if not 0, each column/row (depending on rowmajor) is treated independently
    rowmajor: if True, noise is added row by row
    """
    X = np.asarray(X)
    if rowmajor:
        Y_T, sigma = add_white_noise_mat(X.T, nlvl, mode)
        return Y_T.T, sigma
    else:
        return add_white_noise_mat(X, nlvl, mode)

def add_white_noise_mat(X, nlvl, mode):
    M, N = X.shape
    if mode:
        Y = np.zeros_like(X)
        for c in range(N):
            t0 = np.linalg.norm(X[:, c]) / np.sqrt(M)
            Y[:, c] = X[:, c] + np.random.randn(M) * t0 * nlvl
    else:
        t0 = np.linalg.norm(X, 'fro') / np.sqrt(M * N)
        Y = X + np.random.randn(M, N) * t0 * nlvl

    sigma = np.linalg.norm(X, 'fro') / np.sqrt(M * N) * nlvl
    return Y, sigma
