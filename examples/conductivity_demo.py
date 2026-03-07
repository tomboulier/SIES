import marimo

__generated_with = "0.1.0"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import sies.shape as shape
    import sies.acq as acq
    import sies.pde as pde
    import sies.asymp as asymp
    return acq, asymp, mo, np, pde, plt, shape


@app.cell
def __(np, shape):
    # Definition of small inclusions
    # B = shape.Triangle(0.5, np.pi/3, 2**10, 10)
    B = shape.Ellipse(0.5, 0.25, 2**10) # Using ellipse for easier visualization first

    # Multiple inclusions
    D = [B + [0.1, 0.1]]
    cnd = [10.0]
    pmtt = [1.0]
    return B, D, cnd, pmtt


@app.cell
def __(D, acq, np, pde, plt):
    # Set up an environment for experience
    # Neutrality: surprisingly, this has a better conditioning
    cfg = acq.Coincided([0, 0], 10, 50, viewmode=(1, np.pi/16, 2*np.pi), grouped=False, neutCoeff=[1, -1], neutRad=0.01)

    P = pde.Conductivity_R2(D, [10.0], [1.0], cfg)

    plt.figure(figsize=(6, 6))
    P.plot()
    plt.axis('equal')
    plt.title("Inclusion and Acquisition System")
    plt.gca()
    return P, cfg


@app.cell
def __(P, np):
    # Simulation of the MSR data
    freqlist = np.linspace(0, 100 * np.pi, 5)
    data = P.data_simulation(freqlist)
    return data, freqlist


@app.cell
def __(D, asymp, cnd, freqlist, pmtt):
    # Compute first the theoretical value of CGPT
    ord_val = 1
    M_theo = []
    for _f in freqlist:
        _lamb = asymp.lambda_contrast(cnd, pmtt, _f)
        M_theo.append(asymp.theoretical_CGPT(D, _lamb, ord_val))
    return M_theo, ord_val


@app.cell
def __(M_theo, P, data, freqlist, np, ord_val):
    # Reconstruct CGPT and show error
    nlvl = 0.01 # Add some noise
    data_noisy = P.add_white_noise(data, nlvl)

    K = max(1, ord_val)
    out = P.reconstruct_CGPT(data_noisy["MSR_noisy"], K, maxiter=100000, tol=1e-10, symmode=True, method='lsqr')

    print("Relative error between theoretical and reconstructed CGPT matrix at different frequencies:")
    for _f_idx, _f in enumerate(freqlist):
        _toto = out["CGPT"][_f_idx][:2*ord_val, :2*ord_val]
        _theo = M_theo[_f_idx]

        _err = np.linalg.norm(_theo - _toto, 'fro') / np.linalg.norm(_theo, 'fro')

        # SVD error (as in Matlab demo)
        _u0, _s0, _vh0 = np.linalg.svd(_theo)
        _u1, _s1, _vh1 = np.linalg.svd(_toto)
        _sv0 = _s0[0] / _s0[1]
        _sv1 = _s1[0] / _s1[1]
        _errsvd = np.abs(_sv0 - _sv1) / np.abs(_sv1)

        print(f"Frequency: {_f:.2f}, error: {_err:.4f}, error of sv: {_errsvd:.4f}")
    return K, data_noisy, out


if __name__ == "__main__":
    app.run()
