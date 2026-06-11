"""Dictionary matching of shape descriptors."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sies.asymptotics import contrast, theoretical_cgpt
from sies.dictionary.descriptors import ShapeDescriptor
from sies.shapes import C2Boundary

__all__ = ["ShapeDictionary", "match_descriptor"]


def match_descriptor(
    descriptor: ShapeDescriptor,
    dictionary: list[ShapeDescriptor],
    order: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Rank dictionary elements by similarity with a query descriptor.

    Parameters
    ----------
    descriptor : ShapeDescriptor
        Descriptor of the unknown shape.
    dictionary : list of ShapeDescriptor
        Descriptors of the dictionary shapes.
    order : int, optional
        Truncation order used for the comparison.

    Returns
    -------
    errors : ndarray, shape (len(dictionary),)
        Distance between the query and each dictionary element.
    ranking : ndarray, shape (len(dictionary),)
        Indices of the dictionary elements sorted by increasing
        distance; ``ranking[0]`` is the identified shape.
    """
    errors = np.array([descriptor.distance(item, order) for item in dictionary])
    return errors, np.argsort(errors)


@dataclass
class ShapeDictionary:
    """A dictionary of shapes with precomputed invariant descriptors.

    Attributes
    ----------
    shapes : list of C2Boundary
        The reference shapes.
    descriptors : list of ShapeDescriptor
        Invariant descriptor of each shape.
    cnd : float
        Conductivity used to compute the CGPTs.
    pmtt : float
        Permittivity used to compute the CGPTs.
    """

    shapes: list[C2Boundary]
    descriptors: list[ShapeDescriptor]
    cnd: float
    pmtt: float

    @property
    def names(self) -> list[str]:
        """List of str: Names of the dictionary shapes."""
        return [shape.name or type(shape).__name__ for shape in self.shapes]

    @classmethod
    def build(
        cls,
        shapes: list[C2Boundary],
        cnd: float,
        pmtt: float = 0.0,
        order: int = 5,
        freq: float = 0.0,
    ) -> "ShapeDictionary":
        """Build a dictionary from a list of shapes.

        The theoretical CGPT of each shape is computed by boundary
        integral equations and converted to invariant descriptors.

        Parameters
        ----------
        shapes : list of C2Boundary
            The reference shapes (typically normalized to similar
            sizes).
        cnd : float
            Common conductivity of the shapes.
        pmtt : float, default 0.0
            Common permittivity.
        order : int, default 5
            Maximum CGPT order of the descriptors.
        freq : float, default 0.0
            Working frequency.

        Returns
        -------
        ShapeDictionary
            The assembled dictionary.

        Raises
        ------
        ValueError
            If `shapes` is empty.
        """
        if not shapes:
            raise ValueError("`shapes` must contain at least one reference shape.")
        lam = contrast(cnd, pmtt, freq)
        descriptors = [
            ShapeDescriptor.from_cgpt(theoretical_cgpt(shape, lam, order)) for shape in shapes
        ]
        return cls(shapes=shapes, descriptors=descriptors, cnd=cnd, pmtt=pmtt)

    def match(
        self, descriptor: ShapeDescriptor, order: int | None = None
    ) -> tuple[NDArray, NDArray]:
        """Rank the dictionary shapes by similarity with a descriptor.

        Parameters
        ----------
        descriptor : ShapeDescriptor
            Descriptor of the unknown shape.
        order : int, optional
            Truncation order used for the comparison.

        Returns
        -------
        errors : ndarray
            Distance to each dictionary element.
        ranking : ndarray
            Dictionary indices sorted by increasing distance.
        """
        return match_descriptor(descriptor, self.descriptors, order)

    def identify(self, descriptor: ShapeDescriptor, order: int | None = None) -> str:
        """Return the name of the best-matching dictionary shape.

        Parameters
        ----------
        descriptor : ShapeDescriptor
            Descriptor of the unknown shape.
        order : int, optional
            Truncation order used for the comparison.

        Returns
        -------
        str
            Name of the identified shape.
        """
        _, ranking = self.match(descriptor, order)
        return self.names[ranking[0]]
