from __future__ import annotations

import math

import numpy as np

from distance_rs import VerticalFactor, distance_accumulation, optimal_path_as_line


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


def test_optimal_path_as_line_reaches_source() -> None:
    sources = np.zeros((15, 15), dtype=bool)
    sources[7, 7] = True

    result = distance_accumulation(sources)
    line = optimal_path_as_line(result, (12, 12))

    assert line.shape[1] == 2
    assert np.allclose(line[0], [12.0, 12.0])
    assert np.linalg.norm(line[-1] - np.array([7.0, 7.0])) <= math.sqrt(2)
