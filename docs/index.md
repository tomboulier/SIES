# SIES — Shape Identification in Electro-Sensing

**SIES** is a Python library for inverse problems in electro-sensing: it
simulates electric measurements of small inclusions, computes their
**Generalized Polarization Tensors** (GPT), and identifies or tracks targets
from noisy data.

![Shape gallery](assets/shapes_gallery.png)

## What can it do?

- **Shapes** — discretized $C^2$-smooth boundaries (ellipses, flowers,
  triangles, rectangles, bananas) with exact differential geometry and rigid
  motions.
- **Layer potentials** — boundary integral operators of potential theory:
  single layer $S_D$, Neumann–Poincaré $K_D^*$ and their derivatives.
- **CGPT** — contracted generalized polarization tensors, computed numerically
  for any shape and analytically for disks and ellipses, with their exact
  transformation rules under translation, rotation and scaling.
- **Forward problem** — multistatic response (MSR) matrices of the
  conductivity equation, at one or several frequencies, with calibrated noise.
- **Inverse problem** — least-squares and closed-form reconstruction of the
  CGPT from MSR data.
- **Dictionary matching** — invariant shape descriptors and identification of
  an unknown target in a dictionary of shapes.
- **Tracking** — Extended Kalman Filter estimating the position and
  orientation of a moving target.

## Quick example

```python
import numpy as np
from sies.shapes import Flower
from sies.acquisition import Coincided
from sies.pde import ConductivityR2

# A flower-shaped inclusion and a circle of 100 coincided sources/receivers
target = (Flower(1.0, 1.0, 256) * 0.5) + np.array([0.2, 0.1])
cfg = Coincided(np.zeros(2), radius=1.5, nb_src=100)

# Simulate the multistatic response and reconstruct the CGPT
pde = ConductivityR2(target, cnd=3.0, pmtt=0.0, cfg=cfg)
data = pde.simulate_data(freqs=0.0)
result = pde.reconstruct_cgpt_analytic(data.msr[0], order=5)
```

Head over to [Getting started](getting-started.md) for a complete walkthrough,
or browse the [API reference](api/shapes.md).

## References

The library reproduces numerical results from:

1. Ammari, Boulier, Garnier, Kang, Wang, *Target identification using
   dictionary matching of generalized polarization tensors*, Foundations of
   Computational Mathematics, 2014.
2. Ammari, Boulier, Garnier, Jing, Kang, Wang, *Tracking of a mobile target
   using generalized polarization tensors*, SIAM Journal on Imaging Sciences,
   2013.
3. Ammari, Boulier, Garnier, Wang, *Shape recognition and classification in
   electro-sensing*, PNAS, 2014.

## Credits

Python port of the original [MATLAB library](matlab.md) by **Han Wang**.
