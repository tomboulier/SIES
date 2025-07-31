import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sies import C2Boundary, MConfig, ElectricFish


def test_boundary_properties():
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    points = np.vstack([np.cos(theta), np.sin(theta)])
    boundary = C2Boundary(points)
    assert boundary.nb_points == 20
    assert np.isclose(boundary.diameter, 2, atol=1e-2)


def test_electric_fish_grammatrix():
    theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    points = np.vstack([np.cos(theta), np.sin(theta)])
    boundary = C2Boundary(points)
    cfg = MConfig(sources=np.array([[0.0], [0.0]]), receivers=points)
    cfg.Omega0 = boundary
    fish = ElectricFish(boundary, [1.0], [1.0], cfg)
    G = fish.grammatrix
    assert G.shape[0] == fish.Psi.shape[1]
    assert np.allclose(G, np.eye(G.shape[0]))
