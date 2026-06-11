"""SIES: Shape Identification in Electro-Sensing.

A Python library for the simulation and solution of inverse problems
arising in electro-sensing, based on Generalized Polarization Tensors
(GPT), dictionary matching and target tracking.

This is a Python port of the original MATLAB library by Han Wang.
"""

from sies import acquisition, asymptotics, dictionary, greens, operators, pde, shapes, tracking

__version__ = "1.0.0"

__all__ = [
    "acquisition",
    "asymptotics",
    "dictionary",
    "greens",
    "operators",
    "pde",
    "shapes",
    "tracking",
]
