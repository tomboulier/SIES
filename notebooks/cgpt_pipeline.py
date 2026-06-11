"""Marimo notebook: the CGPT forward and inverse pipeline."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Generalized polarization tensors: forward and inverse problem

        This notebook walks through the core pipeline of **SIES**:

        1. build a shape with a smooth boundary,
        2. compute its **contracted generalized polarization tensors** (CGPT)
           and validate them against closed-form formulas,
        3. simulate the **multistatic response** (MSR) of an acquisition
           system,
        4. reconstruct the CGPT from the (noisy) measurements.
        """
    )
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from sies.acquisition import Coincided
    from sies.asymptotics import contrast, ellipse_cgpt, theoretical_cgpt
    from sies.pde import ConductivityR2
    from sies.shapes import Ellipse

    return Coincided, ConductivityR2, Ellipse, contrast, ellipse_cgpt, np, plt, theoretical_cgpt


@app.cell
def _(mo):
    mo.md(
        """
        ## 1. A shape and its boundary geometry

        Every shape in SIES is a `C2Boundary`: a discretized smooth closed
        curve carrying its tangent, acceleration and outward normal vectors.
        Use the sliders to change the ellipse.
        """
    )
    return


@app.cell
def _(mo):
    axis_a = mo.ui.slider(0.5, 2.0, value=1.0, step=0.1, label="semi-major axis $a$")
    axis_b = mo.ui.slider(0.1, 1.0, value=0.5, step=0.1, label="semi-minor axis $b$")
    mo.vstack([axis_a, axis_b])
    return axis_a, axis_b


@app.cell
def _(Ellipse, axis_a, axis_b, np, plt):
    shape = Ellipse(max(axis_a.value, axis_b.value), min(axis_a.value, axis_b.value), 256)

    _fig, _ax = plt.subplots(figsize=(5, 5))
    shape.plot(ax=_ax, color="tab:purple", linewidth=2)
    _step = 16
    _ax.quiver(
        shape.points[0, ::_step],
        shape.points[1, ::_step],
        shape.normal[0, ::_step],
        shape.normal[1, ::_step],
        color="tab:gray",
        scale=12,
        width=0.004,
    )
    _ax.set_title(f"{shape.name}: area = {shape.area:.3f}, perimeter = {shape.perimeter:.3f}")
    _expected = np.pi * shape.axis_a * shape.axis_b
    print(f"Stokes area {shape.area:.6f} vs exact pi*a*b = {_expected:.6f}")
    _ax
    return (shape,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2. CGPT: boundary integral computation vs closed form

        The CGPT matrix is computed by solving the integral equation

        $$\phi_m = (\lambda I - K_D^*)^{-1}
        \left[\partial_\nu \mathrm{Re}/\mathrm{Im}(z^m)\right]$$

        on the boundary. For ellipses, an exact formula is known (M. Lim) —
        the two must agree to machine precision.
        """
    )
    return


@app.cell
def _(contrast, ellipse_cgpt, np, shape, theoretical_cgpt):
    cnd = 3.0  # conductivity of the inclusion (background is 1)
    order = 4

    cgpt_numeric = theoretical_cgpt(shape, contrast(cnd), order).real
    cgpt_exact = ellipse_cgpt(order, shape.axis_a, shape.axis_b, cnd)

    rel_err = np.linalg.norm(cgpt_numeric - cgpt_exact) / np.linalg.norm(cgpt_exact)
    print(f"relative error between integral equation and closed form: {rel_err:.2e}")
    return cnd, order


@app.cell
def _(mo):
    mo.md(
        """
        ## 3. Simulate the multistatic response

        A circle of coincided sources/receivers surrounds the target. Each
        source emits, every receiver of its group measures the perturbation
        $u - G$.
        """
    )
    return


@app.cell
def _(Coincided, ConductivityR2, cnd, mo, np, shape):
    noise_level = mo.ui.slider(0.0, 0.5, value=0.05, step=0.01, label="noise level")

    target = (shape * 0.5) + np.array([0.2, 0.1])
    cfg = Coincided(np.zeros(2), radius=1.5, nb_src=100)
    pde = ConductivityR2(target, cnd, 0.0, cfg)
    data = pde.simulate_data(freqs=0.0)
    noise_level
    return cfg, data, noise_level, pde, target


@app.cell
def _(cfg, data, np, noise_level, pde, plt):
    _rng = np.random.default_rng(0)
    noisy = pde.add_white_noise(data, noise_level.value, _rng) if noise_level.value > 0 else None
    msr = noisy.msr_noisy[0] if noisy else data.msr[0]

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    pde.plot(ax=_axes[0], color="tab:purple", linewidth=2)
    _axes[0].set_title("acquisition setup")
    _im = _axes[1].imshow(msr, cmap="RdBu")
    _axes[1].set_title(f"MSR matrix ({cfg.nb_sources} sources)")
    _fig.colorbar(_im, ax=_axes[1])
    _fig.tight_layout()
    _fig
    return (msr,)


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Reconstruct the CGPT from the data

        For equispaced circular acquisitions, the least-squares inverse of
        $\\mathrm{MSR} = A_s M A_r^T$ has a closed form. Compare the
        reconstruction with the theoretical CGPT of the target.
        """
    )
    return


@app.cell
def _(cnd, contrast, msr, np, order, pde, plt, target, theoretical_cgpt):
    result = pde.reconstruct_cgpt_analytic(msr, order)
    expected = theoretical_cgpt(target, contrast(cnd), order).real

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _vmax = np.abs(expected).max()
    for _ax, _mat, _title in zip(
        _axes,
        [expected, result.cgpt[0], expected - result.cgpt[0]],
        ["theoretical CGPT", "reconstructed CGPT", "difference"],
        strict=True,
    ):
        _h = _ax.imshow(_mat, cmap="RdBu", vmin=-_vmax, vmax=_vmax)
        _ax.set_title(_title)
        _fig.colorbar(_h, ax=_ax)
    _fig.tight_layout()
    print(f"relative residual of the reconstruction: {result.relative_residual[0]:.2e}")
    _fig
    return


if __name__ == "__main__":
    app.run()
