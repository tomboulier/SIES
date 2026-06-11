"""Extended Kalman Filter and target motion model.

The state vector is ``[vx, vy, x, y, phi]``: velocity, position and
angular position of the target.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["ExtendedKalmanFilter", "simulate_target_path", "target_dynamics"]


def target_dynamics(
    dt: float, std_acc: float, std_acc_angle: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Build the matrices of the linear target motion model.

    The target follows a constant-velocity model driven by white
    acceleration noise.

    Parameters
    ----------
    dt : float
        Time step.
    std_acc : float
        Standard deviation of the linear acceleration noise.
    std_acc_angle : float
        Standard deviation of the angular acceleration noise.

    Returns
    -------
    state_matrix : ndarray, shape (5, 5)
        State transition matrix ``F``.
    noise_cov : ndarray, shape (5, 5)
        Process noise covariance ``Q``.
    noise_gain : ndarray, shape (5, 5)
        Gain ``Q0`` mapping accelerations to state increments (used by
        the simulator).
    """
    state_matrix = np.eye(5)
    state_matrix[2, 0] = dt
    state_matrix[3, 1] = dt

    noise_gain = np.diag([dt, dt, dt**2 / 2, dt**2 / 2, dt])

    acc_block = std_acc**2 * np.eye(2)
    sigma = np.zeros((5, 5))
    sigma[:2, :2] = acc_block
    sigma[:2, 2:4] = acc_block
    sigma[2:4, :2] = acc_block
    sigma[2:4, 2:4] = acc_block
    sigma[4, 4] = std_acc_angle**2
    noise_cov = noise_gain @ sigma @ noise_gain.T

    return state_matrix, noise_cov, noise_gain


def simulate_target_path(
    dt: float,
    nb_steps: int,
    initial_state: NDArray,
    std_acc: float,
    std_acc_angle: float,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Simulate a random target trajectory.

    Parameters
    ----------
    dt : float
        Time step.
    nb_steps : int
        Number of time steps.
    initial_state : ndarray, shape (5,)
        Initial state ``[vx, vy, x, y, phi]``.
    std_acc : float
        Standard deviation of the linear acceleration noise.
    std_acc_angle : float
        Standard deviation of the angular acceleration noise.
    rng : numpy.random.Generator, optional
        Random generator, for reproducibility.

    Returns
    -------
    ndarray, shape (5, nb_steps)
        The simulated state at each time step.
    """
    rng = rng or np.random.default_rng()
    state_matrix, _, noise_gain = target_dynamics(dt, std_acc, std_acc_angle)

    path = np.zeros((5, nb_steps))
    path[:, 0] = np.asarray(initial_state, dtype=float).reshape(5)
    for n in range(1, nb_steps):
        acc = std_acc * rng.standard_normal(2)
        acc_angle = std_acc_angle * rng.standard_normal()
        noise = noise_gain @ np.concatenate([acc, acc, [acc_angle]])
        path[:, n] = state_matrix @ path[:, n - 1] + noise
    return path


class ExtendedKalmanFilter:
    """Extended Kalman Filter with a linear state equation.

    Parameters
    ----------
    state_matrix : ndarray, shape (d, d)
        State transition matrix ``F``.
    process_cov : ndarray, shape (d, d)
        Process noise covariance ``Q``.
    observation : callable
        Observation function ``h(state) -> ndarray``.
    observation_jacobian : callable
        Jacobian ``dh(state) -> ndarray`` of the observation function.
    observation_cov : ndarray
        Observation noise covariance ``R``.
    """

    def __init__(
        self,
        state_matrix: NDArray,
        process_cov: NDArray,
        observation: Callable[[NDArray], NDArray],
        observation_jacobian: Callable[[NDArray], NDArray],
        observation_cov: NDArray,
    ):
        self.state_matrix = state_matrix
        self.process_cov = process_cov
        self.observation = observation
        self.observation_jacobian = observation_jacobian
        self.observation_cov = observation_cov

    def run(
        self,
        measurements: NDArray,
        initial_state: NDArray,
        initial_cov: NDArray | None = None,
    ) -> NDArray:
        """Filter a stream of measurements.

        Parameters
        ----------
        measurements : ndarray, shape (m, nb_steps)
            One measurement vector per time step.
        initial_state : ndarray, shape (d,)
            Initial state guess.
        initial_cov : ndarray, shape (d, d), optional
            Initial state covariance guess; zero by default.

        Returns
        -------
        ndarray, shape (d, nb_steps)
            The estimated state at each time step.
        """
        nb_steps = measurements.shape[1]
        state = np.asarray(initial_state, dtype=float).reshape(-1)
        cov = np.zeros_like(self.process_cov) if initial_cov is None else initial_cov

        estimates = np.zeros((state.size, nb_steps))
        for n in range(nb_steps):
            # Prediction
            predicted = self.state_matrix @ state
            predicted_cov = self.state_matrix @ cov @ self.state_matrix.T + self.process_cov

            # Update
            innovation = measurements[:, n] - self.observation(predicted)
            jacobian = self.observation_jacobian(predicted)
            innovation_cov = jacobian @ predicted_cov @ jacobian.T + self.observation_cov
            gain = predicted_cov @ jacobian.T @ np.linalg.inv(innovation_cov)

            state = predicted + gain @ innovation
            cov = predicted_cov - gain @ jacobian @ predicted_cov
            estimates[:, n] = state

        return estimates
