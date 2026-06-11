"""Unit tests of the tracking module."""

import numpy as np
import pytest

from sies.acquisition import Coincided
from sies.asymptotics import contrast, theoretical_cgpt
from sies.pde import ConductivityR2
from sies.shapes import Ellipse
from sies.tracking import (
    CGPTObservation,
    ExtendedKalmanFilter,
    simulate_target_path,
    target_dynamics,
)


@pytest.fixture
def observation():
    cfg = Coincided(np.zeros(2), 3.0, 16)
    ellipse = Ellipse(1.0, 0.5, 128) * 0.3
    cgpt = theoretical_cgpt(ellipse, contrast(3.0), 2).real
    pde = ConductivityR2(ellipse, 3.0, 0.0, cfg)
    src_matrix, rcv_matrix = pde.make_linear_operator(2)
    return CGPTObservation(src_matrix, rcv_matrix, cgpt)


def test_target_dynamics_structure():
    state_matrix, noise_cov, noise_gain = target_dynamics(0.1, 1.0, 0.5)
    # Velocity integrates into position
    state = np.array([1.0, 2.0, 0.0, 0.0, 0.3])
    new = state_matrix @ state
    np.testing.assert_allclose(new[2:4], [0.1, 0.2])
    np.testing.assert_allclose(new[4], 0.3)
    # Covariance is symmetric positive semidefinite
    np.testing.assert_allclose(noise_cov, noise_cov.T)
    assert np.all(np.linalg.eigvalsh(noise_cov) > -1e-12)
    assert noise_gain[2, 2] == pytest.approx(0.005)


def test_simulate_target_path_reproducible():
    x0 = np.array([0.1, 0.0, -1.0, -1.0, 0.0])
    path1 = simulate_target_path(0.1, 20, x0, 0.5, 0.1, np.random.default_rng(7))
    path2 = simulate_target_path(0.1, 20, x0, 0.5, 0.1, np.random.default_rng(7))
    np.testing.assert_allclose(path1, path2)
    np.testing.assert_allclose(path1[:, 0], x0)
    assert path1.shape == (5, 20)


def test_jacobian_matches_finite_differences(observation):
    state = np.array([0.1, -0.2, 0.4, 0.3, 0.7])
    jac = observation.jacobian(state)
    eps = 1e-7
    for k in range(5):
        direction = np.zeros(5)
        direction[k] = eps
        fd = (observation(state + direction) - observation(state - direction)) / (2 * eps)
        np.testing.assert_allclose(jac[:, k], fd, atol=1e-7)


def test_observation_rejects_nonsquare_cgpt():
    with pytest.raises(ValueError, match="square"):
        CGPTObservation(np.eye(3), np.eye(3), np.zeros((4, 2)))


def test_ekf_on_linear_problem(rng):
    # Fully linear observable system: the EKF reduces to the KF and must
    # converge to the true constant state.
    state_matrix = np.eye(2)
    process_cov = 1e-6 * np.eye(2)
    h_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    obs_cov = 0.01 * np.eye(3)

    true_state = np.array([2.0, -1.0])
    nb_steps = 100
    measurements = h_matrix @ true_state[:, None] + 0.1 * rng.standard_normal((3, nb_steps))

    ekf = ExtendedKalmanFilter(
        state_matrix,
        process_cov,
        observation=lambda x: h_matrix @ x,
        observation_jacobian=lambda x: h_matrix,
        observation_cov=obs_cov,
    )
    estimates = ekf.run(measurements, np.zeros(2), initial_cov=np.eye(2))
    np.testing.assert_allclose(estimates[:, -1], true_state, atol=0.1)
