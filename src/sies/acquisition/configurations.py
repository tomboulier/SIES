"""Concrete acquisition configurations.

Only the geometry of the acquisition system lives here (positions of
sources and receivers); the physics (frequency, contrast) belongs to
the PDE models of `pde`.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["AcquisitionConfig", "Coincided", "Concentric", "ViewMode", "sources_on_circle"]


@dataclass(frozen=True)
class ViewMode:
    """Angular coverage of an acquisition system.

    Parameters
    ----------
    nb_arcs : int, default 1
        Number of separated arcs carrying sources/receivers.
    aperture : float, default ``2 * pi``
        Angular aperture of each arc, in radians.
    angle_of_view : float, default ``2 * pi``
        Angular range covered by the starting points of the arcs.
    """

    nb_arcs: int = 1
    aperture: float = 2 * np.pi
    angle_of_view: float = 2 * np.pi

    @property
    def full_view(self) -> bool:
        """bool: True if the configuration is a single full circle."""
        return self.nb_arcs == 1 and self.aperture == 2 * np.pi


def sources_on_circle(
    nb_arcs: int,
    nb_per_arc: int,
    radius: float,
    center: NDArray,
    aperture: float,
    angle_of_view: float = 2 * np.pi,
) -> list[NDArray]:
    """Place sources/receivers on concentric circular arcs.

    The arcs are equally distributed over ``[0, angle_of_view)`` and each
    covers the angle `aperture`.

    Parameters
    ----------
    nb_arcs : int
        Number of arcs.
    nb_per_arc : int
        Number of points per arc.
    radius : float
        Radius of the measurement circle.
    center : ndarray, shape (2,)
        Center of the measurement circle.
    aperture : float
        Angular aperture of each arc.
    angle_of_view : float, default ``2 * pi``
        Total angle of view.

    Returns
    -------
    list of ndarray
        Coordinates of the points of each arc, as ``(2, nb_per_arc)``
        arrays.
    """
    center = np.asarray(center, dtype=float).reshape(2, 1)
    arcs = []
    for n in range(nb_arcs):
        start = n / nb_arcs * angle_of_view
        angles = start + np.arange(nb_per_arc) / nb_per_arc * aperture
        arcs.append(radius * np.vstack([np.cos(angles), np.sin(angles)]) + center)
    return arcs


class AcquisitionConfig:
    """Base class for acquisition configurations.

    Sources and receivers are organized in groups. Receivers of a group
    only listen to the sources of the same group; all groups contain the
    same number of sources and receivers.

    Parameters
    ----------
    src_groups : list of ndarray
        Source coordinates of each group, as ``(2, ns)`` arrays.
    rcv_groups : list of ndarray
        Receiver coordinates of each group, as ``(2, nr)`` arrays.
    center : ndarray, shape (2,)
        Reference center of the measurement system.
    """

    def __init__(self, src_groups: list[NDArray], rcv_groups: list[NDArray], center: NDArray):
        if len(src_groups) != len(rcv_groups):
            raise ValueError("Sources and receivers must have the same number of groups.")
        self._src_groups = src_groups
        self._rcv_groups = rcv_groups
        self.center = np.asarray(center, dtype=float).reshape(2)

    @property
    def nb_groups(self) -> int:
        """int: Number of source/receiver groups."""
        return len(self._src_groups)

    @property
    def nb_sources_per_group(self) -> int:
        """int: Number of sources in each group."""
        return self._src_groups[0].shape[1]

    @property
    def nb_receivers_per_group(self) -> int:
        """int: Number of receivers in each group."""
        return self._rcv_groups[0].shape[1]

    @property
    def nb_sources(self) -> int:
        """int: Total number of sources."""
        return self.nb_groups * self.nb_sources_per_group

    @property
    def nb_receivers(self) -> int:
        """int: Total number of receivers."""
        return self.nb_groups * self.nb_receivers_per_group

    @property
    def data_dim(self) -> int:
        """int: Total number of scalar measurements (sources x receivers)."""
        return self.nb_sources * self.nb_receivers_per_group

    def group(self, g: int) -> tuple[NDArray, NDArray]:
        """Return the sources and receivers of a group.

        Parameters
        ----------
        g : int
            Group index.

        Returns
        -------
        src : ndarray, shape (2, ns)
            Source coordinates of the group.
        rcv : ndarray, shape (2, nr)
            Receiver coordinates of the group.
        """
        return self._src_groups[g], self._rcv_groups[g]

    def source(self, s: int) -> NDArray:
        """Return the coordinates of the `s`-th source (global index).

        Parameters
        ----------
        s : int
            Global source index, in ``range(nb_sources)``.

        Returns
        -------
        ndarray, shape (2,)
            Coordinates of the source.
        """
        if not 0 <= s < self.nb_sources:
            raise IndexError("Source index out of range.")
        g, i = divmod(s, self.nb_sources_per_group)
        return self._src_groups[g][:, i]

    def receivers_of_source(self, s: int) -> NDArray:
        """Return the receivers listening to the `s`-th source.

        Parameters
        ----------
        s : int
            Global source index.

        Returns
        -------
        ndarray, shape (2, nr)
            Coordinates of the receivers of the source's group.
        """
        if not 0 <= s < self.nb_sources:
            raise IndexError("Source index out of range.")
        g = s // self.nb_sources_per_group
        return self._rcv_groups[g]

    @property
    def all_sources(self) -> NDArray:
        """ndarray: All source coordinates, concatenated group by group."""
        return np.concatenate(self._src_groups, axis=1)

    @property
    def all_receivers(self) -> NDArray:
        """ndarray: All receiver coordinates, concatenated group by group."""
        return np.concatenate(self._rcv_groups, axis=1)

    def plot(self, ax=None, **kwargs):
        """Plot sources (crosses) and receivers (circles).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if omitted.
        **kwargs
            Forwarded to `plot`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        src, rcv = self.all_sources, self.all_receivers
        ax.plot(src[0], src[1], "x", label="sources", **kwargs)
        ax.plot(rcv[0], rcv[1], "o", fillstyle="none", label="receivers", **kwargs)
        ax.plot(*self.center, "r*")
        ax.set_aspect("equal")
        return ax


class Concentric(AcquisitionConfig):
    """Sources and receivers on two concentric circles.

    Parameters
    ----------
    center : ndarray, shape (2,)
        Center of the measurement circles.
    radius_src : float
        Radius of the source circle.
    nb_src : int
        Number of sources per arc.
    radius_rcv : float
        Radius of the receiver circle.
    nb_rcv : int
        Number of receivers per arc.
    view : ViewMode, optional
        Angular coverage; full view by default.
    grouped : bool, default False
        If True, each arc forms an independent group (limited view); if
        False, every receiver listens to every source.

    Attributes
    ----------
    radius_src, radius_rcv : float
        Radii of the measurement circles.
    equispaced : bool
        True for a single full-view circle of equispaced positions, in
        which case the analytic CGPT reconstruction applies.
    """

    def __init__(
        self,
        center: NDArray,
        radius_src: float,
        nb_src: int,
        radius_rcv: float,
        nb_rcv: int,
        view: ViewMode | None = None,
        grouped: bool = False,
    ):
        view = view or ViewMode()
        src_arcs = sources_on_circle(
            view.nb_arcs, nb_src, radius_src, center, view.aperture, view.angle_of_view
        )
        rcv_arcs = sources_on_circle(
            view.nb_arcs, nb_rcv, radius_rcv, center, view.aperture, view.angle_of_view
        )

        if grouped:
            src_groups, rcv_groups = src_arcs, rcv_arcs
        else:
            src_groups = [np.concatenate(src_arcs, axis=1)]
            rcv_groups = [np.concatenate(rcv_arcs, axis=1)]

        super().__init__(src_groups, rcv_groups, center)
        self.radius_src = radius_src
        self.radius_rcv = radius_rcv
        self.view = view
        self.equispaced = self.nb_groups == 1 and view.full_view


class Coincided(Concentric):
    """Coincided sources and receivers on a single circle.

    Each source position also acts as a receiver: the simplest
    configuration for the conductivity problem.

    Parameters
    ----------
    center : ndarray, shape (2,)
        Center of the measurement circle.
    radius : float
        Radius of the measurement circle.
    nb_src : int
        Number of sources (and receivers) per arc.
    view : ViewMode, optional
        Angular coverage; full view by default.
    grouped : bool, default False
        See `Concentric`.
    """

    def __init__(
        self,
        center: NDArray,
        radius: float,
        nb_src: int,
        view: ViewMode | None = None,
        grouped: bool = False,
    ):
        super().__init__(center, radius, nb_src, radius, nb_src, view, grouped)
