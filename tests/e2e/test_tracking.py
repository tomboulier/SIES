"""End-to-end test: tracking a mobile target with the EKF.

Reproduces the experiment of ``demo_tracking.m``: a small target moves
randomly inside the measurement circle; its position and orientation
are estimated from the stream of noisy MSR matrices.
"""

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


@pytest.mark.e2e
def test_ekf_tracks_moving_target():
    rng = np.random.default_rng(7)
    order = 2

    # Small elliptic target and full-view acquisition
    shape = Ellipse(1.0, 0.5, 128) * 0.2
    cgpt = theoretical_cgpt(shape, contrast(3.0), order).real
    cfg = Coincided(np.zeros(2), 3.0, 30)
    pde = ConductivityR2(shape, 3.0, 0.0, cfg)
    src_matrix, rcv_matrix = pde.make_linear_operator(order)
    observation = CGPTObservation(src_matrix, rcv_matrix, cgpt)

    # Simulate the trajectory and the measurement stream
    dt, nb_steps = 0.1, 60
    std_acc, std_acc_angle = 0.5, 0.2
    x0 = np.array([0.2, 0.1, -0.4, -0.3, 0.0])
    path = simulate_target_path(dt, nb_steps, x0, std_acc, std_acc_angle, rng)

    measurements = np.column_stack([observation(path[:, n]) for n in range(nb_steps)])
    noise_std = 0.01 * np.abs(measurements).max()
    measurements += noise_std * rng.standard_normal(measurements.shape)

    # Track
    state_matrix, process_cov, _ = target_dynamics(dt, std_acc, std_acc_angle)
    ekf = ExtendedKalmanFilter(
        state_matrix,
        process_cov,
        observation,
        observation.jacobian,
        observation_cov=noise_std**2 * np.eye(measurements.shape[0]),
    )
    estimates = ekf.run(measurements, x0, initial_cov=0.1 * np.eye(5))

    # Empirical thresholds: skip the transient (~1 s of simulated time)
    # and require an accuracy well below the target size (diameter 0.4).
    convergence_window = 10
    position_median_tol = 0.05
    position_final_tol = 0.1
    orientation_median_tol = 0.05

    position_error = np.linalg.norm(
        estimates[2:4, convergence_window:] - path[2:4, convergence_window:], axis=0
    )
    assert np.median(position_error) < position_median_tol
    assert position_error[-1] < position_final_tol

    orientation_error = np.abs(estimates[4, convergence_window:] - path[4, convergence_window:])
    assert np.median(orientation_error) < orientation_median_tol
