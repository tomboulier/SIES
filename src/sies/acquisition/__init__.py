"""Acquisition systems: geometry of sources and receivers.

An acquisition configuration describes where point sources emit and
where receivers measure. Sources and receivers may be divided into
groups (arcs of limited view); within a group all sources and receivers
are mutually visible.
"""

from sies.acquisition.configurations import (
    AcquisitionConfig,
    Coincided,
    Concentric,
    ViewMode,
    sources_on_circle,
)

__all__ = ["AcquisitionConfig", "Coincided", "Concentric", "ViewMode", "sources_on_circle"]
