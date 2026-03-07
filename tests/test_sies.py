import numpy as np
import pytest
from sies import Shape, Triangle, Ellipse, Coincided, Conductivity_R2, theoretical_CGPT, lambda_contrast

def test_shape_dsl():
    # Test translation
    e = Ellipse(1, 0.5, 100)
    e2 = e + [1, 2]
    assert np.allclose(e2.center_of_mass, [1, 2])

    # Test scaling
    e3 = e * 2
    assert np.isclose(e3.diameter, 2 * e.diameter)

    # Test rotation
    e4 = e @ (np.pi/2)
    # After 90 deg rotation, semi-major axis (x) becomes semi-minor (y) and vice versa
    assert np.allclose(e4.box, [e.box[1], e.box[0]], atol=1e-7)

def test_conductivity_simulation():
    D = [Ellipse(0.5, 0.25, 256)]
    cfg = Coincided([0, 0], 10, 32)
    P = Conductivity_R2(D, [10.0], [0.0], cfg)

    freqs = [0, 10]
    data = P.data_simulation(freqs)

    assert len(data["MSR"]) == 2
    assert data["MSR"][0].shape == (32, 32)

def test_cgpt_reconstruction():
    # Use a simple circle for very stable reconstruction
    D = [Ellipse(0.5, 0.5, 256)]
    cfg = Coincided([0, 0], 10, 64)
    P = Conductivity_R2(D, [5.0], [0.0], cfg)

    freq = 0
    data = P.data_simulation([freq])

    ord_val = 1
    # Theoretical CGPT
    lamb = lambda_contrast([5.0], [0.0], freq)
    M_theo = theoretical_CGPT(D, lamb, ord_val)

    # Reconstruct
    out = P.reconstruct_CGPT(data["MSR"], ord_val, method='pinv')
    M_rec = out["CGPT"][0]

    # For a circle, M should be very close to theoretical
    assert np.linalg.norm(M_theo - M_rec) / np.linalg.norm(M_theo) < 1e-2
