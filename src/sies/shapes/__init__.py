"""Shapes with $C^2$-smooth closed boundaries.

This subpackage provides the `C2Boundary` base class
describing a discretized smooth closed curve, together with a catalog of
standard shapes (ellipse, flower, triangle, rectangle, banana) used to
build dictionaries of targets.
"""

from sies.shapes.boundary import C2Boundary
from sies.shapes.resampling import resample_curve
from sies.shapes.standard import Banana, Ellipse, Flower, Rectangle, Triangle

__all__ = [
    "Banana",
    "C2Boundary",
    "Ellipse",
    "Flower",
    "Rectangle",
    "Triangle",
    "resample_curve",
]
