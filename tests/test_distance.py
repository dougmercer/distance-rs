from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from distance_rs import VerticalFactor, distance_accumulation, optimal_path_as_line
from distance_rs.baselines import raster_dijkstra, raster_dijkstra_baseline, trace_raster_path


def test_package_import_does_not_load_geo_dependencies() -> None:
    script = """
import sys
import distance_rs
print('distance_rs._geo' in sys.modules)
from distance_rs import *
print('distance_rs._geo' in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == ["False", "False"]


def test_flat_accumulation_matches_euclidean_distance_near_source() -> None:
    sources = np.zeros((21, 21), dtype=bool)
    sources[10, 10] = True

    result = distance_accumulation(sources, cost_surface=np.ones_like(sources, dtype=float))

    assert result.distance[10, 10] == 0.0
    assert result.distance[10, 15] == np.float64(5.0)
    assert math.isclose(result.distance[13, 14], 5.0, rel_tol=0.08)


def test_barrier_blocks_cells() -> None:
    sources = np.zeros((11, 11), dtype=bool)
    sources[5, 2] = True
    barriers = np.zeros((11, 11), dtype=bool)
    barriers[:, 5] = True

    result = distance_accumulation(sources, barriers=barriers, search_radius=3.0)

    assert math.isinf(result.distance[5, 5])
    assert math.isinf(result.distance[5, 8])


def test_binary_vertical_factor_blocks_upslope() -> None:
    sources = np.zeros((9, 9), dtype=bool)
    sources[4, 4] = True
    elevation = np.tile(np.arange(9, dtype=float), (9, 1))

    result = distance_accumulation(
        sources,
        elevation=elevation,
        vertical_factor=VerticalFactor("binary", low_cut_angle=-90.0, high_cut_angle=0.1),
        search_radius=3.0,
    )

    assert np.isfinite(result.distance[4, 3])
    assert math.isinf(result.distance[4, 8])


def test_distance_accumulation_rejects_non_finite_numeric_options() -> None:
    sources = np.zeros((3, 3), dtype=bool)
    sources[1, 1] = True

    with pytest.raises(ValueError, match="search_radius"):
        distance_accumulation(sources, search_radius=math.nan)

    with pytest.raises(ValueError, match="cell_size"):
        distance_accumulation(sources, cell_size=math.nan)

    with pytest.raises(ValueError, match="origin"):
        distance_accumulation(sources, origin=(0.0, math.inf))

    with pytest.raises(ValueError, match="vertical factor option"):
        distance_accumulation(sources, vertical_factor={"type": "linear", "slope": math.nan})


def test_optimal_path_as_line_reaches_source() -> None:
    sources = np.zeros((15, 15), dtype=bool)
    sources[7, 7] = True

    result = distance_accumulation(sources)
    line = optimal_path_as_line(result, (12, 12))

    assert line.shape[1] == 2
    assert np.allclose(line[0], [12.0, 12.0])
    assert np.linalg.norm(line[-1] - np.array([7.0, 7.0])) <= math.sqrt(2)


def test_raster_dijkstra_baseline_returns_distance_and_traceable_parent() -> None:
    sources = np.zeros((9, 9), dtype=bool)
    sources[4, 1] = True
    barriers = np.zeros((9, 9), dtype=bool)
    barriers[:, 4] = True
    barriers[4, 4] = False

    result = raster_dijkstra(sources, barriers=barriers, use_surface_distance=False)
    distance_only = raster_dijkstra_baseline(sources, barriers=barriers, use_surface_distance=False)
    line = trace_raster_path(result.parent, (4, 7))

    assert np.array_equal(result.distance, distance_only)
    assert math.isinf(result.distance[0, 4])
    assert np.isfinite(result.distance[4, 7])
    assert np.allclose(line[0], [7.0, 4.0])
    assert np.allclose(line[-1], [1.0, 4.0])
