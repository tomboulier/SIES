"""Shape identification in a dictionary.

Translation-, rotation- and scaling-invariant shape descriptors are
computed from CGPT matrices and compared against a precomputed
dictionary of shapes.
"""

from sies.dictionary.descriptors import ShapeDescriptor
from sies.dictionary.matching import ShapeDictionary, match_descriptor

__all__ = ["ShapeDescriptor", "ShapeDictionary", "match_descriptor"]
