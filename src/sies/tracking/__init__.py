"""Tracking of a mobile target from multistatic measurements.

An Extended Kalman Filter estimates the position and orientation of a
moving target whose CGPT is known, following Ammari et al., *Tracking
of a mobile target using generalized polarization tensors*, SIAM J.
Imaging Sci. (2013).
"""

from sies.tracking.kalman import ExtendedKalmanFilter, simulate_target_path, target_dynamics
from sies.tracking.observation import CGPTObservation

__all__ = [
    "CGPTObservation",
    "ExtendedKalmanFilter",
    "simulate_target_path",
    "target_dynamics",
]
