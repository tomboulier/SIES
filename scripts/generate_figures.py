"""Generate the figures used in the README and the documentation.

Run from the repository root:

    python scripts/generate_figures.py

The figures are written to ``docs/assets/``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sies.acquisition import Coincided
from sies.asymptotics import contrast, theoretical_cgpt
from sies.dictionary import ShapeDescriptor, ShapeDictionary
from sies.pde import ConductivityR2
from sies.shapes import Banana, Ellipse, Flower, Rectangle, Triangle
from sies.tracking import (
    CGPTObservation,
    ExtendedKalmanFilter,
    simulate_target_path,
    target_dynamics,
)

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
NB_POINTS = 256
CND = 3.0

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def dictionary_shapes():
    return [
        Ellipse(1.0, 0.5, NB_POINTS),
        Flower(1.0, 1.0, NB_POINTS),
        Triangle(1.0, np.pi / 3, NB_POINTS),
        Rectangle(1.0, 1.0, NB_POINTS),
        Rectangle(2.0, 1.0, NB_POINTS) * 0.5,
        Banana(0.7, 0.15, [0.0, 0.0], [0.0, 1.0], NB_POINTS),
    ]


def figure_shapes_gallery():
    shapes = dictionary_shapes()
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for shape, ax in zip(shapes, axes.ravel(), strict=True):
        centered = shape - shape.center_of_mass
        centered.plot(ax=ax, color="tab:purple", linewidth=2)
        ax.fill(centered.points[0], centered.points[1], color="tab:purple", alpha=0.15)
        ax.set_title(shape.name)
        half = 0.62 * max(centered.box)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.axis("off")
    fig.suptitle("Dictionary of shapes", fontsize=14)
    fig.tight_layout()
    fig.savefig(ASSETS / "shapes_gallery.png", bbox_inches="tight")
    plt.close(fig)


def figure_acquisition():
    target = (Flower(1.0, 1.0, NB_POINTS).rotate(0.2 * np.pi) * 0.5) + np.array([0.25, 0.25])
    cfg = Coincided(np.zeros(2), 1.5, 32)
    pde = ConductivityR2(target, CND, 0.0, cfg)

    fig, ax = plt.subplots(figsize=(6, 6))
    pde.plot(ax=ax, color="tab:purple", linewidth=2)
    ax.fill(target.points[0], target.points[1], color="tab:purple", alpha=0.15)
    ax.legend(["target", "sources", "receivers"], loc="upper right")
    ax.set_title("Acquisition setup: full-view coincided array")
    fig.tight_layout()
    fig.savefig(ASSETS / "acquisition_setup.png", bbox_inches="tight")
    plt.close(fig)


def figure_matching():
    rng = np.random.default_rng(11)
    shapes = dictionary_shapes()
    dico = ShapeDictionary.build(shapes, cnd=CND, order=5)

    target = (Flower(1.0, 1.0, NB_POINTS).rotate(0.2 * np.pi) * 0.75) + np.array([0.25, 0.25])
    cfg = Coincided(np.zeros(2), 1.5, 100)
    pde = ConductivityR2(target, CND, 0.0, cfg)
    data = pde.simulate_data(0.0)
    noisy = pde.add_white_noise(data, 0.10, rng)
    result = pde.reconstruct_cgpt_analytic(noisy.msr_noisy[0], 5)
    descriptor = ShapeDescriptor.from_cgpt(result.cgpt[0])
    errors, ranking = dico.match(descriptor, order=3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[1, 1.4])
    centered = target - target.center_of_mass
    axes[0].fill(centered.points[0], centered.points[1], color="tab:purple", alpha=0.2)
    centered.plot(ax=axes[0], color="tab:purple", linewidth=2)
    axes[0].set_title("Unknown target\n(rotated, scaled, translated)")
    axes[0].axis("off")

    colors = ["tab:green" if k == ranking[0] else "tab:gray" for k in range(len(errors))]
    axes[1].bar(dico.names, errors, color=colors)
    axes[1].set_ylabel("descriptor distance")
    axes[1].set_title("Dictionary matching with 10% measurement noise")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(ASSETS / "matching_results.png", bbox_inches="tight")
    plt.close(fig)


def figure_tracking():
    rng = np.random.default_rng(7)
    order = 2

    shape = Ellipse(1.0, 0.5, 128) * 0.2
    cgpt = theoretical_cgpt(shape, contrast(CND), order).real
    cfg = Coincided(np.zeros(2), 3.0, 30)
    pde = ConductivityR2(shape, CND, 0.0, cfg)
    src_matrix, rcv_matrix = pde.make_linear_operator(order)
    observation = CGPTObservation(src_matrix, rcv_matrix, cgpt)

    dt, nb_steps = 0.1, 100
    std_acc, std_acc_angle = 0.15, 0.2
    x0 = np.array([0.25, 0.1, -1.2, -0.8, 0.0])
    path = simulate_target_path(dt, nb_steps, x0, std_acc, std_acc_angle, rng)

    measurements = np.column_stack([observation(path[:, n]) for n in range(nb_steps)])
    noise_std = 0.01 * np.abs(measurements).max()
    measurements += noise_std * rng.standard_normal(measurements.shape)

    state_matrix, process_cov, _ = target_dynamics(dt, std_acc, std_acc_angle)
    ekf = ExtendedKalmanFilter(
        state_matrix,
        process_cov,
        observation,
        observation.jacobian,
        observation_cov=noise_std**2 * np.eye(measurements.shape[0]),
    )
    estimates = ekf.run(measurements, x0, initial_cov=0.1 * np.eye(5))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[1, 1])
    theta = 2 * np.pi * np.arange(200) / 199
    axes[0].plot(3 * np.cos(theta), 3 * np.sin(theta), ":", color="gray", label="sensor array")
    axes[0].plot(path[2], path[3], "-", color="tab:blue", linewidth=2, label="true path")
    axes[0].plot(
        estimates[2], estimates[3], "--", color="tab:red", linewidth=1.5, label="EKF estimate"
    )
    axes[0].plot(path[2, 0], path[3, 0], "o", color="tab:blue")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_title("Target tracking with the EKF")

    time = dt * np.arange(nb_steps)
    axes[1].plot(time, path[4], "-", color="tab:blue", linewidth=2, label="true orientation")
    axes[1].plot(time, estimates[4], "--", color="tab:red", linewidth=1.5, label="EKF estimate")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("orientation (rad)")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Orientation estimate")
    fig.tight_layout()
    fig.savefig(ASSETS / "tracking.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    figure_shapes_gallery()
    figure_acquisition()
    figure_matching()
    figure_tracking()
    print(f"Figures written to {ASSETS}")
