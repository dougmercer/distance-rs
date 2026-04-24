from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from distance_rs import (
    RasterGrid,
    RasterSurface,
    SolverOptions,
    VerticalFactor,
    distance_accumulation,
    optimal_path_as_line,
)
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
    cost = np.ones((21, 21), dtype=float)

    result = distance_accumulation(cost, source=(10, 10))

    assert result.distance[10, 10] == 0.0
    assert result.distance[10, 15] == np.float64(5.0)
    assert math.isclose(result.distance[13, 14], 5.0, rel_tol=0.08)


def test_barrier_blocks_cells() -> None:
    cost = np.ones((11, 11), dtype=float)
    barriers = np.zeros((11, 11), dtype=bool)
    barriers[:, 5] = True
    surface = RasterSurface(cost, barriers=barriers)

    result = distance_accumulation(
        surface,
        source=(5, 2),
    )

    assert math.isinf(result.distance[5, 5])
    assert math.isinf(result.distance[5, 8])


def test_binary_vertical_factor_blocks_upslope() -> None:
    cost = np.ones((9, 9), dtype=float)
    elevation = np.tile(np.arange(9, dtype=float), (9, 1))
    surface = RasterSurface(cost, elevation=elevation)

    result = distance_accumulation(
        surface,
        source=(4, 4),
        options=SolverOptions(
            vertical_factor=VerticalFactor("binary", low_cut_angle=-90.0, high_cut_angle=0.1),
        ),
    )

    assert np.isfinite(result.distance[4, 3])
    assert math.isinf(result.distance[4, 8])


def test_distance_accumulation_rejects_non_finite_numeric_options() -> None:
    cost = np.ones((3, 3), dtype=float)

    with pytest.raises(ValueError, match="cell_size"):
        surface = RasterSurface(cost, grid=RasterGrid(cell_size=math.nan))
        distance_accumulation(surface, source=(1, 1))

    with pytest.raises(ValueError, match="origin"):
        surface = RasterSurface(cost, grid=RasterGrid(origin=(0.0, math.inf)))
        distance_accumulation(surface, source=(1, 1))

    with pytest.raises(ValueError, match="vertical factor option"):
        distance_accumulation(
            cost,
            source=(1, 1),
            options=SolverOptions(vertical_factor={"type": "linear", "slope": math.nan}),
        )


def test_distance_accumulation_rejects_non_integer_source_cell() -> None:
    with pytest.raises(ValueError, match="integer"):
        distance_accumulation(np.ones((3, 3), dtype=float), source=(1.2, 1))


def test_optimal_path_as_line_reaches_source() -> None:
    result = distance_accumulation(np.ones((15, 15), dtype=float), source=(7, 7))
    line = optimal_path_as_line(result, (12, 12))

    assert line.shape[1] == 2
    assert np.allclose(line[0], [12.0, 12.0])
    assert np.linalg.norm(line[-1] - np.array([7.0, 7.0])) <= math.sqrt(2)


def test_optimal_path_as_line_traces_back_direction_without_zig_zag() -> None:
    source = (16, 55)
    destination = (94, 100)
    result = distance_accumulation(
        np.ones((121, 151), dtype=float),
        source=source,
        options=SolverOptions(use_surface_distance=False),
    )

    line = optimal_path_as_line(result, destination)

    assert _path_area_from_straight_line(line, source, destination) < 50.0


def test_optimal_path_as_line_repairs_invalid_direction_step_locally() -> None:
    source, destination, cost, barriers = _make_maze_case(
        maze_rows=11,
        maze_cols=17,
        scale=7,
        seed=11,
    )
    result = distance_accumulation(
        RasterSurface(cost, barriers=barriers),
        source=source,
        options=SolverOptions(use_surface_distance=False),
    )

    line = optimal_path_as_line(result, destination)

    assert line.shape[1] == 2
    assert math.isfinite(result.distance[destination])
    assert _polyline_length(line) < 520.0


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


def _path_area_from_straight_line(
    line: np.ndarray,
    source: tuple[int, int],
    destination: tuple[int, int],
) -> float:
    source_xy = np.asarray([source[1], source[0]], dtype=np.float64)
    destination_xy = np.asarray([destination[1], destination[0]], dtype=np.float64)
    axis = destination_xy - source_xy
    length = np.linalg.norm(axis)
    if length <= 1.0e-12 or len(line) < 2:
        return 0.0

    unit = axis / length
    normal = np.asarray([-unit[1], unit[0]], dtype=np.float64)
    centered = line - source_xy
    progress = centered @ unit
    cross_track = np.abs(centered @ normal)
    order = np.argsort(progress)
    progress = progress[order]
    cross_track = cross_track[order]
    return float(np.sum(0.5 * np.diff(progress) * (cross_track[:-1] + cross_track[1:])))


def _polyline_length(line: np.ndarray) -> float:
    if len(line) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def _make_maze_case(
    *,
    maze_rows: int,
    maze_cols: int,
    scale: int,
    seed: int,
) -> tuple[tuple[int, int], tuple[int, int], np.ndarray, np.ndarray]:
    maze = _carve_perfect_maze(maze_rows, maze_cols, seed)
    barriers = np.kron(maze, np.ones((scale, scale), dtype=np.bool_))

    rows, cols = barriers.shape
    y, x = np.mgrid[:rows, :cols]
    corridor = ~barriers
    center_pull = 0.08 * np.cos(2.0 * math.pi * y / max(rows, 1))
    east_penalty = 0.10 * x / max(cols - 1, 1)
    cost = 1.0 + center_pull + east_penalty
    cost = np.where(corridor, cost, np.inf).astype(np.float64)

    source = _logical_maze_cell_center(0, 0, scale)
    destination = _logical_maze_cell_center(maze_rows - 1, maze_cols - 1, scale)
    _clear_endpoint(barriers, cost, source)
    _clear_endpoint(barriers, cost, destination)
    return source, destination, cost, barriers


def _carve_perfect_maze(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    maze = np.ones((2 * rows + 1, 2 * cols + 1), dtype=np.bool_)
    visited = np.zeros((rows, cols), dtype=np.bool_)
    stack = [(0, 0)]
    visited[0, 0] = True
    maze[1, 1] = False

    while stack:
        row, col = stack[-1]
        neighbors = []
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            next_row = row + dr
            next_col = col + dc
            if 0 <= next_row < rows and 0 <= next_col < cols and not visited[next_row, next_col]:
                neighbors.append((dr, dc, next_row, next_col))

        if not neighbors:
            stack.pop()
            continue

        dr, dc, next_row, next_col = neighbors[int(rng.integers(len(neighbors)))]
        maze[1 + 2 * row + dr, 1 + 2 * col + dc] = False
        maze[1 + 2 * next_row, 1 + 2 * next_col] = False
        visited[next_row, next_col] = True
        stack.append((next_row, next_col))

    return maze


def _logical_maze_cell_center(row: int, col: int, scale: int) -> tuple[int, int]:
    return (1 + 2 * row) * scale + scale // 2, (1 + 2 * col) * scale + scale // 2


def _clear_endpoint(barriers: np.ndarray, cost: np.ndarray, point: tuple[int, int]) -> None:
    row, col = point
    row_slice = slice(max(0, row - 1), min(barriers.shape[0], row + 2))
    col_slice = slice(max(0, col - 1), min(barriers.shape[1], col + 2))
    barriers[row_slice, col_slice] = False
    cost[row_slice, col_slice] = np.minimum(cost[row_slice, col_slice], 1.0)
