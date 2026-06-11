"""Asymptotic expansions: contracted generalized polarization tensors.

The Contracted Generalized Polarization Tensors (CGPT) are the building
blocks of the small-volume asymptotic expansion of the perturbation of
an electric potential due to inclusions. This subpackage computes them
numerically from boundary integral equations, analytically for disks
and ellipses, and provides the transformation rules under rigid motions
and scaling.
"""

from sies.asymptotics.cgpt import (
    contrast,
    make_block_matrix,
    make_system_matrix,
    theoretical_cgpt,
)
from sies.asymptotics.exact import disk_cgpt, ellipse_cgpt
from sies.asymptotics.transforms import (
    cgpt_to_complex,
    complex_to_cgpt,
    split_cgpt,
    transform_ccgpt,
    transform_ccgpt_inverse,
    transform_cgpt,
)

__all__ = [
    "cgpt_to_complex",
    "complex_to_cgpt",
    "contrast",
    "disk_cgpt",
    "ellipse_cgpt",
    "make_block_matrix",
    "make_system_matrix",
    "split_cgpt",
    "theoretical_cgpt",
    "transform_ccgpt",
    "transform_ccgpt_inverse",
    "transform_cgpt",
]
