"""Marimo notebook: shape identification by dictionary matching."""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Shape identification by dictionary matching

        Reproduction of the experiment of *Target identification using
        dictionary matching of generalized polarization tensors* (FoCM,
        2014):

        1. build a **dictionary** of reference shapes and their invariant
           descriptors,
        2. simulate noisy measurements of an **unknown target**: a rotated,
           scaled and translated dictionary element,
        3. reconstruct its CGPT, compute its descriptors, and **identify**
           it in the dictionary.
        """
    )
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    from sies.acquisition import Coincided
    from sies.dictionary import ShapeDescriptor, ShapeDictionary
    from sies.pde import ConductivityR2
    from sies.shapes import Banana, Ellipse, Flower, Rectangle, Triangle

    return (
        Banana,
        Coincided,
        ConductivityR2,
        Ellipse,
        Flower,
        Rectangle,
        ShapeDescriptor,
        ShapeDictionary,
        Triangle,
        np,
        plt,
    )


@app.cell
def _(Banana, Ellipse, Flower, Rectangle, ShapeDictionary, Triangle, np):
    NB_POINTS = 256
    CND = 3.0

    shapes = [
        Ellipse(1.0, 0.5, NB_POINTS),
        Flower(1.0, 1.0, NB_POINTS),
        Triangle(1.0, np.pi / 3, NB_POINTS),
        Rectangle(1.0, 1.0, NB_POINTS),
        Rectangle(2.0, 1.0, NB_POINTS) * 0.5,
        Banana(0.7, 0.15, [0.0, 0.0], [0.0, 1.0], NB_POINTS),
    ]
    dico = ShapeDictionary.build(shapes, cnd=CND, order=5)
    print("dictionary:", ", ".join(dico.names))
    return CND, dico, shapes


@app.cell
def _(dico, plt, shapes):
    _fig, _axes = plt.subplots(1, len(shapes), figsize=(2.2 * len(shapes), 2.4))
    for _shape, _name, _ax in zip(shapes, dico.names, _axes, strict=True):
        _centered = _shape - _shape.center_of_mass
        _centered.plot(ax=_ax, color="tab:purple", linewidth=2)
        _ax.set_title(_name, fontsize=10)
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(dico, mo):
    target_choice = mo.ui.dropdown(
        options=dico.names, value="Flower", label="shape of the unknown target"
    )
    noise = mo.ui.slider(0.0, 0.5, value=0.02, step=0.01, label="noise level")
    comparison_order = mo.ui.slider(1, 5, value=3, step=1, label="descriptor comparison order")
    mo.vstack([target_choice, noise, comparison_order])
    return comparison_order, noise, target_choice


@app.cell
def _(CND, Coincided, ConductivityR2, dico, np, noise, target_choice):
    _index = dico.names.index(target_choice.value)
    # The unknown target: a transformed dictionary element.
    target = (dico.shapes[_index].rotate(0.2 * np.pi) * 0.75) + np.array([0.25, 0.25])

    cfg = Coincided(np.zeros(2), radius=1.5, nb_src=100)
    pde = ConductivityR2(target, CND, 0.0, cfg)
    data = pde.simulate_data(freqs=0.0)
    noisy = pde.add_white_noise(data, noise.value, np.random.default_rng(1))
    msr = noisy.msr_noisy[0] if noise.value > 0 else data.msr[0]
    return msr, pde, target


@app.cell
def _(ShapeDescriptor, comparison_order, dico, msr, pde):
    result = pde.reconstruct_cgpt_analytic(msr, 5)
    descriptor = ShapeDescriptor.from_cgpt(result.cgpt[0])
    errors, ranking = dico.match(descriptor, order=comparison_order.value)
    identified = dico.names[ranking[0]]
    return errors, identified, ranking


@app.cell
def _(dico, errors, identified, plt, ranking, target, target_choice):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=[1, 1.5])
    _centered = target - target.center_of_mass
    _centered.plot(ax=_axes[0], color="tab:purple", linewidth=2)
    _axes[0].fill(_centered.points[0], _centered.points[1], color="tab:purple", alpha=0.15)
    _axes[0].set_title(f"unknown target (true: {target_choice.value})")
    _axes[0].axis("off")

    _colors = ["tab:green" if _k == ranking[0] else "tab:gray" for _k in range(len(errors))]
    _axes[1].bar(dico.names, errors, color=_colors)
    _axes[1].set_ylabel("descriptor distance")
    _axes[1].set_title(f"identified: {identified}")
    _axes[1].tick_params(axis="x", rotation=20)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Robustness: confusion across the whole dictionary

        Repeat the experiment with each dictionary element as target, and
        check that each one is best matched by itself (the diagonal of the
        confusion matrix).

        As in the original paper, identification is perfect at low noise
        and degrades around 5–10% noise, where shapes with similar
        low-order descriptors (the flat ellipse and the 2:1 rectangle)
        start to be confused; lowering the comparison order helps.
        """
    )
    return


@app.cell
def _(CND, Coincided, ConductivityR2, ShapeDescriptor, comparison_order, dico, noise, np, plt):
    _rng = np.random.default_rng(2)
    _cfg = Coincided(np.zeros(2), radius=1.5, nb_src=100)
    confusion = np.zeros((len(dico.shapes), len(dico.shapes)))

    for _i, _shape in enumerate(dico.shapes):
        _target = (_shape.rotate(0.2 * np.pi) * 0.75) + np.array([0.25, 0.25])
        _pde = ConductivityR2(_target, CND, 0.0, _cfg)
        _data = _pde.simulate_data(freqs=0.0)
        _noisy = _pde.add_white_noise(_data, max(noise.value, 1e-6), _rng)
        _result = _pde.reconstruct_cgpt_analytic(_noisy.msr_noisy[0], 5)
        _descriptor = ShapeDescriptor.from_cgpt(_result.cgpt[0])
        _errors, _ = dico.match(_descriptor, order=comparison_order.value)
        confusion[_i] = _errors

    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    _ax.imshow(confusion, cmap="viridis_r")
    _ax.set_xticks(range(len(dico.names)), dico.names, rotation=30)
    _ax.set_yticks(range(len(dico.names)), dico.names)
    _ax.set_xlabel("dictionary element")
    _ax.set_ylabel("true target")
    _ax.set_title("descriptor distances (dark = similar)")
    _correct = int((confusion.argmin(axis=1) == np.arange(len(dico.names))).sum())
    print(f"correct identifications: {_correct}/{len(dico.names)}")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
