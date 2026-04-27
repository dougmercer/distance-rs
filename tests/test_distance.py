from __future__ import annotations

import math

import numpy as np
import pytest

from distance_rs import (
    RasterGrid,
    RasterSurface,
    VerticalFactor,
    distance_accumulation,
    evaluate_path_cost,
    optimal_path_as_line,
    optimal_path_trace,
    route_legs,
)
from distance_rs.baselines import (
    raster_dijkstra,
    trace_raster_path,
    whitebox_cost_distance,
)


def test_flat_accumulation_matches_euclidean_distance_near_source() -> None:
    cost = np.ones((21, 21), dtype=float)

    result = distance_accumulation(cost, source=(10, 10))

    assert result.distance[10, 10] == 0.0
    assert result.distance[10, 15] == np.float64(5.0)
    assert math.isclose(result.distance[13, 14], 5.0, rel_tol=0.08)


def test_route_legs_solves_multiple_legs_on_one_surface() -> None:
    cost = np.ones((21, 21), dtype=float)
    leg_windows = np.array(
        [
            [10, 10, 10, 15, 0, 21, 0, 21],
            [10, 10, 13, 14, 0, 21, 0, 21],
        ],
        dtype=np.int64,
    )

    solved = route_legs(cost, leg_windows)

    assert len(solved) == 2
    assert solved[0].cost == pytest.approx(5.0)
    assert solved[1].cost == pytest.approx(5.0, rel=0.08)
    assert np.allclose(solved[0].line[0], [15.0, 10.0])
    assert np.allclose(solved[0].line[-1], [10.0, 10.0])

    shifted = route_legs(
        RasterSurface(cost, grid=RasterGrid(origin=(100.0, 200.0))),
        leg_windows[:1],
    )
    assert np.allclose(shifted[0].line[0], [115.0, 210.0])
    assert np.allclose(shifted[0].line[-1], [110.0, 210.0])


def test_cost_accumulation_uses_target_cell_cost_for_local_slope() -> None:
    cost = np.array([[1.0, 9.0, 1.0]], dtype=float)

    result = distance_accumulation(cost, source=(0, 0))

    assert result.distance[0, 1] == np.float64(9.0)


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
        vertical_factor=VerticalFactor("binary", low_cut_angle=-90.0, high_cut_angle=0.1),
    )

    assert np.isfinite(result.distance[4, 3])
    assert math.isinf(result.distance[4, 8])


def test_elevation_uses_surface_distance() -> None:
    cost = np.ones((3, 3), dtype=float)
    elevation = np.zeros((3, 3), dtype=float)
    elevation[1, 2] = 3.0
    surface = RasterSurface(cost, elevation=elevation)

    result = distance_accumulation(surface, source=(1, 1))

    assert result.distance[1, 2] == np.float64(math.hypot(1.0, 3.0))


def test_evaluate_path_cost_uses_distance_rs_target_cell_cost() -> None:
    cost = np.array([[1.0, 9.0, 1.0]], dtype=float)
    line = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    assert evaluate_path_cost(cost, line) == pytest.approx(9.0)


def test_evaluate_path_cost_matches_adjacent_elevation_step() -> None:
    cost = np.ones((3, 3), dtype=float)
    elevation = np.zeros((3, 3), dtype=float)
    elevation[1, 2] = 3.0
    surface = RasterSurface(cost, elevation=elevation)
    source = (1, 1)
    target = (1, 2)
    line = np.array([[1.0, 1.0], [2.0, 1.0]], dtype=float)

    result = distance_accumulation(surface, source=source)

    assert evaluate_path_cost(surface, line) == pytest.approx(result.distance[target])


def test_evaluate_path_cost_returns_infinite_for_blocked_step() -> None:
    cost = np.ones((3, 3), dtype=float)
    barriers = np.zeros((3, 3), dtype=bool)
    barriers[1, 2] = True
    surface = RasterSurface(cost, barriers=barriers)
    line = np.array([[1.0, 1.0], [2.0, 1.0]], dtype=float)

    assert math.isinf(evaluate_path_cost(surface, line))


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
            vertical_factor={"type": "linear", "slope": math.nan},
        )


def test_distance_accumulation_rejects_non_integer_source_cell() -> None:
    with pytest.raises(ValueError, match="integer"):
        distance_accumulation(np.ones((3, 3), dtype=float), source=(1.2, 1))


def test_distance_accumulation_reports_progress() -> None:
    updates: list[tuple[int, int]] = []

    result = distance_accumulation(
        np.ones((5, 5), dtype=float),
        source=(2, 2),
        progress=lambda accepted, total: updates.append((accepted, total)),
        progress_interval=4,
    )

    assert np.isfinite(result.distance).all()
    assert updates[-1] == (25, 25)
    assert [accepted for accepted, _total in updates] == sorted(
        accepted for accepted, _total in updates
    )


def test_optimal_path_as_line_reaches_source() -> None:
    result = distance_accumulation(np.ones((15, 15), dtype=float), source=(7, 7))
    line = optimal_path_as_line(result, (12, 12))

    assert line.shape[1] == 2
    assert np.allclose(line[0], [12.0, 12.0])
    assert np.linalg.norm(line[-1] - np.array([7.0, 7.0])) <= math.sqrt(2)


def test_target_limited_accumulation_traces_destination_path() -> None:
    source = (7, 7)
    destination = (12, 12)
    cost = np.ones((31, 31), dtype=float)
    full = distance_accumulation(cost, source=source)
    limited = distance_accumulation(cost, source=source, target=destination)

    assert limited.distance[destination] == pytest.approx(full.distance[destination])
    assert np.count_nonzero(np.isfinite(limited.distance)) < np.count_nonzero(
        np.isfinite(full.distance)
    )

    line = optimal_path_as_line(limited, destination)

    assert line.shape[1] == 2
    assert np.allclose(line[0], [destination[1], destination[0]])
    assert np.linalg.norm(line[-1] - np.array([source[1], source[0]])) <= math.sqrt(2)


def test_target_limited_accumulation_accepts_multiple_targets() -> None:
    source = (10, 10)
    targets = [(12, 12), (14, 10)]
    cost = np.ones((31, 31), dtype=float)
    full = distance_accumulation(cost, source=source)
    limited = distance_accumulation(cost, source=source, target=targets)

    for target in targets:
        assert limited.distance[target] == pytest.approx(full.distance[target])


def test_target_limited_accumulation_rejects_invalid_target_cell() -> None:
    with pytest.raises(ValueError, match="target cell is outside"):
        distance_accumulation(np.ones((3, 3), dtype=float), source=(1, 1), target=(3, 1))


def test_optimal_path_as_line_traces_back_direction_without_zig_zag() -> None:
    source = (16, 55)
    destination = (94, 100)
    result = distance_accumulation(
        np.ones((121, 151), dtype=float),
        source=source,
    )

    line = optimal_path_as_line(result, destination)

    assert _path_area_from_straight_line(line, source, destination) < 50.0


def test_flat_back_direction_tracks_euclidean_angle() -> None:
    source = (10, 10)
    destination = (13, 14)
    result = distance_accumulation(np.ones((21, 21), dtype=float), source=source)

    expected = _back_direction_degrees(source, destination)

    assert _angle_delta(result._back_direction[destination], expected) < 5.0


def test_segment_parent_geometry_is_preserved() -> None:
    result = distance_accumulation(np.ones((21, 21), dtype=float), source=(10, 10))

    segment_parent_cells = result._parent_b >= 0

    assert np.any(segment_parent_cells)
    assert np.all((result._parent_weight[segment_parent_cells] > 0.0))
    assert np.all((result._parent_weight[segment_parent_cells] < 1.0))


def test_unrelated_barrier_does_not_replace_surface_back_direction() -> None:
    cost = np.ones((61, 61), dtype=float)
    source = (30, 30)
    destination = (31, 2)
    expected = distance_accumulation(cost, source=source)
    barriers = np.zeros_like(cost, dtype=bool)
    barriers[0, 0] = True

    result = distance_accumulation(RasterSurface(cost, barriers=barriers), source=source)

    assert (
        _angle_delta(
            result._back_direction[destination],
            expected._back_direction[destination],
        )
        < 0.01
    )


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
    )

    line = optimal_path_as_line(result, destination)

    assert line.shape[1] == 2
    assert math.isfinite(result.distance[destination])
    assert _polyline_length(line) < 520.0


def test_optimal_path_trace_reports_fallback_metadata() -> None:
    source, destination, cost, barriers = _make_maze_case(
        maze_rows=11,
        maze_cols=17,
        scale=7,
        seed=11,
    )
    result = distance_accumulation(
        RasterSurface(cost, barriers=barriers),
        source=source,
    )

    trace = optimal_path_trace(result, destination)

    assert trace.line.shape[1] == 2
    assert trace.metadata["direction_steps"] > 0
    assert trace.metadata["parent_lattice_fallbacks"] > 0
    assert trace.metadata["total_fallbacks"] == (
        trace.metadata["parent_lattice_fallbacks"]
        + trace.metadata["proposed_cell_center_fallbacks"]
        + trace.metadata["current_cell_center_fallbacks"]
        + trace.metadata["direct_parent_point_fallbacks"]
    )


def test_raster_dijkstra_returns_distance_and_traceable_parent() -> None:
    sources = np.zeros((9, 9), dtype=bool)
    sources[4, 1] = True
    barriers = np.zeros((9, 9), dtype=bool)
    barriers[:, 4] = True
    barriers[4, 4] = False

    result = raster_dijkstra(sources, barriers=barriers)
    line = trace_raster_path(result.parent, (4, 7))

    assert math.isinf(result.distance[0, 4])
    assert np.isfinite(result.distance[4, 7])
    assert np.allclose(line[0], [7.0, 4.0])
    assert np.allclose(line[-1], [1.0, 4.0])


def test_raster_dijkstra_reports_progress() -> None:
    sources = np.zeros((5, 5), dtype=bool)
    sources[2, 2] = True
    updates: list[tuple[int, int]] = []

    result = raster_dijkstra(
        sources,
        progress=lambda accepted, total: updates.append((accepted, total)),
        progress_interval=4,
    )

    assert np.isfinite(result.distance).all()
    assert updates[-1] == (25, 25)
    assert [accepted for accepted, _total in updates] == sorted(
        accepted for accepted, _total in updates
    )


def test_whitebox_cost_distance_burns_barriers_into_cost_raster() -> None:
    pytest.importorskip("whitebox")

    sources = np.zeros((9, 9), dtype=float)
    sources[4, 1] = 1.0
    barriers = np.zeros((9, 9), dtype=bool)
    barriers[:, 4] = True
    barriers[4, 4] = False

    result = whitebox_cost_distance(
        sources,
        barriers=barriers,
        destinations=np.eye(9, dtype=float),
    )

    assert math.isinf(result.distance[0, 4])
    assert np.isfinite(result.distance[4, 7])
    assert result.pathway is not None


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


def _back_direction_degrees(source: tuple[int, int], destination: tuple[int, int]) -> float:
    delta_x = source[1] - destination[1]
    delta_y = source[0] - destination[0]
    return (math.degrees(math.atan2(delta_x, -delta_y)) + 360.0) % 360.0


def _angle_delta(actual: float, expected: float) -> float:
    return abs((actual - expected + 180.0) % 360.0 - 180.0)


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
