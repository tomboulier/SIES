import numpy as np

def Green2D_Grad(X, Y):
    """
    Gradient of the 2D Green function G(x) = 1/2/pi * log|x| evaluated at the points X-Y.
    X: (2, M), Y: (2, N)
    Outputs: Gx, Gy (M, N) matrices
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # X[0, :] - Y[0, :] using broadcasting
    # X[0, :, None] is (M, 1), Y[0, None, :] is (1, N)
    # Result is (M, N)
    XY1 = X[0, :, None] - Y[0, None, :]
    XY2 = X[1, :, None] - Y[1, None, :]

    DN = XY1**2 + XY2**2
    # Handle zero division if any (e.g. X == Y)
    DN[DN == 0] = np.inf

    Gx = (1 / (2 * np.pi)) * XY1 / DN
    Gy = (1 / (2 * np.pi)) * XY2 / DN
    return Gx, Gy

def Green2D_Dn(X, Y, normal):
    """
    Normal derivative of the 2D Green function G(y,x) = 1/2/pi * log|y-x|
    with respect to y on a boundary.
    X: (2, M), Y: (2, N), normal: (2, N)
    Output: Gn (M, N) matrix
    """
    # Gx, Gy evaluated at (Y - X)
    # So Y is "X" in Green2D_Grad and X is "Y"
    Gx, Gy = Green2D_Grad(Y, X) # (N, M)

    # Gn = normal[0,:] * Gx + normal[1,:] * Gy
    # normal[0,:] is (N,), Gx is (N, M) -> broadcasting works if we do normal[0,:,None] * Gx
    Gn = normal[0, :, None] * Gx + normal[1, :, None] * Gy
    return Gn.T # (M, N)
