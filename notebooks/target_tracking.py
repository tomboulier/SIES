"""Marimo notebook: tracking a mobile target with the EKF."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Tracking of a mobile target

        Reproduction of the experiment of *Tracking of a mobile target using
        generalized polarization tensors* (SIAM J. Imaging Sci., 2013): a
        small target of **known CGPT** moves randomly inside a circular
        sensor array; an **Extended Kalman Filter** estimates its position
        and orientation from the stream of noisy MSR matrices.
        """
    )
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

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

    return (
        CGPTObservation,
        Coincided,
        ConductivityR2,
        Ellipse,
        ExtendedKalmanFilter,
        contrast,
        np,
        plt,
        simulate_target_path,
        target_dynamics,
        theoretical_cgpt,
    )


@app.cell
def _(mo):
    seed = mo.ui.slider(0, 20, value=7, step=1, label="random seed of the trajectory")
    noise_pct = mo.ui.slider(0.0, 5.0, value=1.0, step=0.25, label="measurement noise (%)")
    mo.vstack([seed, noise_pct])
    return noise_pct, seed


@app.cell
def _(CGPTObservation, Coincided, ConductivityR2, Ellipse, contrast, np, theoretical_cgpt):
    ORDER = 2
    CND = 3.0

    # Small elliptic target and full-view circular acquisition.
    shape = Ellipse(1.0, 0.5, 128) * 0.2
    cgpt = theoretical_cgpt(shape, contrast(CND), ORDER).real
    cfg = Coincided(np.zeros(2), radius=3.0, nb_src=30)
    pde = ConductivityR2(shape, CND, 0.0, cfg)
    src_matrix, rcv_matrix = pde.make_linear_operator(ORDER)
    observation = CGPTObservation(src_matrix, rcv_matrix, cgpt)
    return (observation,)


@app.cell
def _(np, noise_pct, observation, seed, simulate_target_path):
    _rng = np.random.default_rng(seed.value)
    dt, nb_steps = 0.1, 100
    std_acc, std_acc_angle = 0.15, 0.2

    x0 = np.array([0.25, 0.1, -1.2, -0.8, 0.0])
    path = simulate_target_path(dt, nb_steps, x0, std_acc, std_acc_angle, _rng)

    measurements = np.column_stack([observation(path[:, _n]) for _n in range(nb_steps)])
    noise_std = noise_pct.value / 100 * np.abs(measurements).max() + 1e-12
    measurements = measurements + noise_std * _rng.standard_normal(measurements.shape)
    return dt, measurements, nb_steps, noise_std, path, std_acc, std_acc_angle, x0


@app.cell
def _(
    ExtendedKalmanFilter,
    dt,
    measurements,
    noise_std,
    np,
    observation,
    path,
    std_acc,
    std_acc_angle,
    target_dynamics,
    x0,
):
    state_matrix, process_cov, _ = target_dynamics(dt, std_acc, std_acc_angle)
    ekf = ExtendedKalmanFilter(
        state_matrix,
        process_cov,
        observation,
        observation.jacobian,
        observation_cov=noise_std**2 * np.eye(measurements.shape[0]),
    )
    estimates = ekf.run(measurements, x0, initial_cov=0.1 * np.eye(5))

    position_error = np.linalg.norm(estimates[2:4] - path[2:4], axis=0)
    print(f"median position error: {np.median(position_error):.4f}")
    return estimates, position_error


@app.cell
def _(dt, estimates, nb_steps, np, path, plt, position_error):
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4))

    _theta = 2 * np.pi * np.arange(200) / 199
    _axes[0].plot(3 * np.cos(_theta), 3 * np.sin(_theta), ":", color="gray", label="sensors")
    _axes[0].plot(path[2], path[3], "-", color="tab:blue", linewidth=2, label="true path")
    _axes[0].plot(estimates[2], estimates[3], "--", color="tab:red", label="EKF estimate")
    _axes[0].set_aspect("equal")
    _axes[0].legend(fontsize=8)
    _axes[0].set_title("position")

    _time = dt * np.arange(nb_steps)
    _axes[1].plot(_time, path[4], "-", color="tab:blue", linewidth=2, label="true")
    _axes[1].plot(_time, estimates[4], "--", color="tab:red", label="EKF estimate")
    _axes[1].set_xlabel("time")
    _axes[1].legend(fontsize=8)
    _axes[1].set_title("orientation (rad)")

    _axes[2].semilogy(_time, position_error, color="tab:purple")
    _axes[2].set_xlabel("time")
    _axes[2].set_title("position error")

    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
