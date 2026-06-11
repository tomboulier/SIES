# MATLAB legacy library

SIES started as a MATLAB library written by **Han Wang** (ENS Paris). The
original code is preserved in this repository (directories `+shape/`, `+ops/`,
`+asymp/`, `+PDE/`, `+dico/`, `+acq/`, `+tools/`, `+tracking/`, `+wavelet/`
and `examples/`).

## Correspondence between MATLAB and Python

| MATLAB | Python | Status |
| --- | --- | --- |
| `+shape/@C2boundary` and shape classes | `sies.shapes` | ✅ ported |
| `+ops` (layer potentials) | `sies.operators` | ✅ ported (P0 elements) |
| `+asymp/+CGPT` | `sies.asymptotics` | ✅ ported |
| `+acq` (acquisition) | `sies.acquisition` | ✅ ported (single-Dirac sources) |
| `+PDE/@Conductivity_R2` | `sies.pde.ConductivityR2` | ✅ ported |
| `+dico/+CGPT` | `sies.dictionary` | ✅ ported |
| `+tracking` | `sies.tracking` | ✅ ported |
| `+tools` | `sies.greens`, `sies.utils` | ✅ ported (the needed subset) |
| `+PDE/@Helmholtz_R2`, `+asymp/+SCT` (echolocation) | (none) | not yet ported |
| `+PDE/@ElectricFish`, `+PDE/@PulseImaging_R2` | (none) | not yet ported |
| `+wavelet` (wavelet polarization tensors) | (none) | not yet ported |
| `examples/` (demo scripts) | `notebooks/` (Marimo) | ✅ ported |

## Notable differences

- **Indices**: Python indices are 0-based; source `s` in MATLAB corresponds
  to source `s - 1` in Python.
- **Spline resampling**: shapes with corners are smoothed with *periodic*
  cubic splines (`scipy.interpolate.CubicSpline`), where MATLAB used
  not-a-knot splines (`csapi`); the periodic variant is the natural choice for
  closed curves.
- **Flower perturbation exponent**: the second derivative of the flower
  boundary for perturbation exponents `k > 1` follows the correct analytic
  formula (the MATLAB code had a typo in a branch never exercised, since it
  always used `k = 1`).
- **Operator overloading**: MATLAB overloaded `<` for rotations; Python uses
  the explicit `shape.rotate(phi)`. Translation (`+`, `-`) and scaling (`*`)
  are kept.
- **Randomness**: all stochastic functions accept a `numpy.random.Generator`
  for reproducibility.
