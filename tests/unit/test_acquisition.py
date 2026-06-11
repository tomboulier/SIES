"""Unit tests of the acquisition configurations."""

import numpy as np
import pytest

from sies.acquisition import (
    AcquisitionConfig,
    Concentric,
    ViewMode,
    sources_on_circle,
)


def test_sources_on_circle_full_view():
    arcs = sources_on_circle(1, 8, 2.0, [1.0, -1.0], 2 * np.pi)
    assert len(arcs) == 1
    points = arcs[0]
    radii = np.linalg.norm(points - np.array([[1.0], [-1.0]]), axis=0)
    np.testing.assert_allclose(radii, 2.0)
    np.testing.assert_allclose(points[:, 0], [3.0, -1.0])


def test_sources_on_circle_limited_arcs():
    arcs = sources_on_circle(3, 4, 1.0, [0, 0], 0.2 * np.pi)
    assert len(arcs) == 3
    starts = [np.arctan2(arc[1, 0], arc[0, 0]) % (2 * np.pi) for arc in arcs]
    np.testing.assert_allclose(starts, [0, 2 * np.pi / 3, 4 * np.pi / 3], atol=1e-12)


def test_coincided_counts_and_geometry(full_view_cfg):
    cfg = full_view_cfg
    assert cfg.nb_sources == cfg.nb_receivers == 50
    assert cfg.nb_groups == 1
    assert cfg.data_dim == 2500
    assert cfg.equispaced
    np.testing.assert_allclose(cfg.all_sources, cfg.all_receivers)
    np.testing.assert_allclose(cfg.source(0), [1.5, 0.0])


def test_source_indexing_and_groups():
    view = ViewMode(nb_arcs=2, aperture=np.pi / 2)
    cfg = Concentric(np.zeros(2), 1.0, 4, 2.0, 6, view, grouped=True)
    assert cfg.nb_groups == 2
    assert cfg.nb_sources_per_group == 4
    assert cfg.nb_receivers_per_group == 6
    assert cfg.nb_sources == 8
    assert cfg.nb_receivers == 12
    assert not cfg.equispaced

    src, rcv = cfg.group(1)
    np.testing.assert_allclose(cfg.source(5), src[:, 1])
    np.testing.assert_allclose(cfg.receivers_of_source(5), rcv)

    with pytest.raises(IndexError):
        cfg.source(8)
    with pytest.raises(IndexError):
        cfg.receivers_of_source(8)


def test_ungrouped_concatenates_arcs():
    view = ViewMode(nb_arcs=2, aperture=np.pi / 2)
    cfg = Concentric(np.zeros(2), 1.0, 4, 2.0, 6, view, grouped=False)
    assert cfg.nb_groups == 1
    assert cfg.nb_sources == 8
    assert cfg.nb_receivers == 12


def test_viewmode_full_view_flag():
    assert ViewMode().full_view
    assert not ViewMode(nb_arcs=2).full_view
    assert not ViewMode(aperture=np.pi).full_view


def test_mismatched_groups_rejected():
    src = [np.zeros((2, 3))]
    rcv = [np.zeros((2, 3)), np.zeros((2, 3))]
    with pytest.raises(ValueError, match="same number of groups"):
        AcquisitionConfig(src, rcv, np.zeros(2))


def test_plot(full_view_cfg):
    ax = full_view_cfg.plot()
    assert len(ax.lines) == 3  # sources, receivers, center
