#!/usr/bin/env python
"""Compare surface back directions with parent-direction replay in a barrier run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from distance_rs import DistanceAccumulationResult, RasterSurface, distance_accumulation
from distance_rs import optimal_path_as_line


Point = tuple[float, float]


@dataclass(frozen=True)
class ReplayResult:
    result: DistanceAccumulationResult
    segment_parent_cells: npt.NDArray[np.bool_]
    angle_delta: npt.NDArray[np.float64]


@dataclass(frozen=True)
class RouteComparison:
    current_length: float
    parent_direction_length: float
    max_offset: float
    mean_offset: float
    zoom_center: Point


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source, destination, cost, barriers = make_case()
    result = distance_accumulation(RasterSurface(cost, barriers=barriers), source=source)
    current_line = one_line(optimal_path_as_line(result, destination))

    parent_replay = parent_direction_replay(result)
    parent_direction_line = one_line(optimal_path_as_line(parent_replay.result, destination))

    comparison = compare_routes(current_line, parent_direction_line)
    output_path = args.output_dir / "barrier_back_direction_replay.png"
    plot_case(
        source,
        destination,
        cost,
        barriers,
        current_line,
        parent_direction_line,
        parent_replay,
        comparison,
        output_path,
    )

    changed_angles = parent_replay.angle_delta[np.isfinite(parent_replay.angle_delta)]
    print(f"wrote {output_path}")
    print(f"current route length: {comparison.current_length:.2f} cells")
    print(f"parent-direction replay length: {comparison.parent_direction_length:.2f} cells")
    print(f"max route offset after arclength resampling: {comparison.max_offset:.2f} cells")
    print(f"mean route offset after arclength resampling: {comparison.mean_offset:.2f} cells")
    print(f"segment-parent cells replayed as parent directions: {changed_angles.size:,}")
    print(f"largest changed back-direction angle: {float(np.max(changed_angles)):.2f} degrees")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/barrier-back-direction-replay"),
        help="Directory for the output plot.",
    )
    return parser.parse_args()


def make_case() -> tuple[
    tuple[int, int],
    tuple[int, int],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    maze_rows, maze_cols, scale, seed = 9, 13, 8, 34
    maze = carve_perfect_maze(maze_rows, maze_cols, seed)
    barriers = np.kron(maze, np.ones((scale, scale), dtype=np.bool_))

    rows, cols = barriers.shape
    y, x = np.mgrid[:rows, :cols]
    rng = np.random.default_rng(seed + 1000)
    cost = np.ones((rows, cols), dtype=np.float64)
    for _ in range(5):
        center_col = rng.uniform(0.0, cols)
        center_row = rng.uniform(0.0, rows)
        sigma_col = rng.uniform(0.05, 0.18) * cols
        sigma_row = rng.uniform(0.05, 0.18) * rows
        sign = float(rng.choice([-1, 1]))
        cost += (
            sign
            * 5.0
            * np.exp(
                -(
                    (x - center_col) ** 2 / (2.0 * sigma_col * sigma_col)
                    + (y - center_row) ** 2 / (2.0 * sigma_row * sigma_row)
                )
            )
        )

    cost += 0.4 * np.sin(x / (2.0 + seed % 5)) + 0.35 * np.cos(y / (3.0 + seed % 7))
    cost = np.maximum(cost, 0.05)
    cost = np.where(~barriers, cost, np.inf).astype(np.float64)

    source = logical_maze_cell_center(0, 0, scale)
    destination = (80, 175)
    clear_endpoint(barriers, cost, source)
    return source, destination, cost, barriers


def carve_perfect_maze(rows: int, cols: int, seed: int) -> npt.NDArray[np.bool_]:
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


def logical_maze_cell_center(row: int, col: int, scale: int) -> tuple[int, int]:
    return (1 + 2 * row) * scale + scale // 2, (1 + 2 * col) * scale + scale // 2


def clear_endpoint(
    barriers: npt.NDArray[np.bool_],
    cost: npt.NDArray[np.float64],
    point: tuple[int, int],
) -> None:
    row, col = point
    row_slice = slice(max(0, row - 1), min(barriers.shape[0], row + 2))
    col_slice = slice(max(0, col - 1), min(barriers.shape[1], col + 2))
    barriers[row_slice, col_slice] = False
    cost[row_slice, col_slice] = np.minimum(cost[row_slice, col_slice], 1.0)


def parent_direction_replay(result: DistanceAccumulationResult) -> ReplayResult:
    rows, cols = result.distance.shape
    back_direction = result._back_direction.copy()
    segment_parent_cells = (result._parent_b >= 0) & np.isfinite(result.distance)
    angle_delta = np.full(result.distance.shape, np.nan, dtype=np.float64)

    for row, col in zip(*np.nonzero(segment_parent_cells)):
        parent_point = continuous_parent_point(result, int(row), int(col))
        if parent_point is None:
            continue
        degrees = direction_to_point(result, int(row), int(col), parent_point)
        if degrees is None:
            continue

        existing = back_direction[row, col]
        if math.isfinite(float(existing)):
            angle_delta[row, col] = angular_difference(float(existing), degrees)
        back_direction[row, col] = degrees

    return ReplayResult(
        result=replace(result, _back_direction=back_direction),
        segment_parent_cells=segment_parent_cells,
        angle_delta=angle_delta,
    )


def continuous_parent_point(
    result: DistanceAccumulationResult,
    row: int,
    col: int,
) -> Point | None:
    _, cols = result.distance.shape
    parent_a = int(result._parent_a[row, col])
    parent_b = int(result._parent_b[row, col])
    if parent_a < 0 or parent_b < 0:
        return None

    weight_a = float(result._parent_weight[row, col])
    a_row, a_col = divmod(parent_a, cols)
    b_row, b_col = divmod(parent_b, cols)
    return (
        weight_a * a_row + (1.0 - weight_a) * b_row,
        weight_a * a_col + (1.0 - weight_a) * b_col,
    )


def direction_to_point(
    result: DistanceAccumulationResult,
    row: int,
    col: int,
    point: Point,
) -> float | None:
    dx = (point[1] - col) * result.cell_size[0]
    dy = (point[0] - row) * result.cell_size[1]
    if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) <= 1.0e-12:
        return None

    degrees = math.degrees(math.atan2(dx, -dy))
    if degrees < 0.0:
        degrees += 360.0
    return degrees


def angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def one_line(
    line: npt.NDArray[np.float64] | list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    if isinstance(line, list):
        raise RuntimeError("single destination unexpectedly produced multiple paths")
    return line


def compare_routes(
    current_line: npt.NDArray[np.float64],
    parent_direction_line: npt.NDArray[np.float64],
) -> RouteComparison:
    current_sample = resample_line(current_line, 1200)
    parent_sample = resample_line(parent_direction_line, 1200)
    offsets = np.linalg.norm(current_sample - parent_sample, axis=1)
    max_idx = int(np.argmax(offsets))
    center = 0.5 * (current_sample[max_idx] + parent_sample[max_idx])
    return RouteComparison(
        current_length=polyline_length(current_line),
        parent_direction_length=polyline_length(parent_direction_line),
        max_offset=float(offsets[max_idx]),
        mean_offset=float(np.mean(offsets)),
        zoom_center=(float(center[0]), float(center[1])),
    )


def resample_line(line_xy: npt.NDArray[np.float64], count: int) -> npt.NDArray[np.float64]:
    if len(line_xy) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(line_xy) == 1:
        return np.repeat(line_xy, count, axis=0)

    segment_lengths = np.linalg.norm(np.diff(line_xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    if cumulative[-1] <= 1.0e-12:
        return np.repeat(line_xy[:1], count, axis=0)

    samples = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [
            np.interp(samples, cumulative, line_xy[:, 0]),
            np.interp(samples, cumulative, line_xy[:, 1]),
        ]
    )


def plot_case(
    source: tuple[int, int],
    destination: tuple[int, int],
    cost: npt.NDArray[np.float64],
    barriers: npt.NDArray[np.bool_],
    current_line: npt.NDArray[np.float64],
    parent_direction_line: npt.NDArray[np.float64],
    parent_replay: ReplayResult,
    comparison: RouteComparison,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    barrier_cmap = ListedColormap(["white"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    full_ax, zoom_ax, segment_ax = axes

    draw_surface(full_ax, cost, barriers, barrier_cmap)
    draw_routes(full_ax, current_line, parent_direction_line)
    draw_endpoints(full_ax, source, destination)
    full_ax.set_title("Full route overlay")

    draw_surface(zoom_ax, cost, barriers, barrier_cmap)
    draw_routes(zoom_ax, current_line, parent_direction_line)
    zoom_ax.scatter(
        [comparison.zoom_center[0]],
        [comparison.zoom_center[1]],
        marker="x",
        c="#7b3294",
        s=90,
        zorder=6,
    )
    zoom_ax.annotate(
        f"{comparison.max_offset:.1f} cell offset",
        xy=comparison.zoom_center,
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#444444"},
    )
    zoom_to_center(zoom_ax, comparison.zoom_center, radius=18.0)
    zoom_ax.set_title("Largest route difference")

    draw_surface(segment_ax, cost, barriers, barrier_cmap, alpha=0.20)
    segment_ax.imshow(
        np.ma.masked_where(~parent_replay.segment_parent_cells, parent_replay.segment_parent_cells),
        cmap=ListedColormap(["#2166ac"]),
        origin="lower",
        interpolation="nearest",
        alpha=0.52,
        zorder=2,
    )
    draw_routes(segment_ax, current_line, parent_direction_line, lw=1.7)
    draw_endpoints(segment_ax, source, destination)
    segment_ax.set_title("Cells replayed with parent directions")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("column")
        ax.set_ylabel("row")

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="barrier cells"),
        Patch(facecolor="#2166ac", edgecolor="none", alpha=0.52, label="segment-parent cells"),
        Line2D([0], [0], color="#d7191c", lw=2.4, label="surface-direction path"),
        Line2D(
            [0],
            [0],
            color="#2c7bb6",
            lw=2.2,
            ls="--",
            label="parent-direction replay",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="#2ca25f",
            markeredgecolor="black",
            color="none",
            label="source",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            markerfacecolor="#ffd92f",
            markeredgecolor="black",
            color="none",
            markersize=12,
            label="destination",
        ),
    ]
    full_ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.suptitle("Keeping surface back directions changes the traced route")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_surface(
    ax: Any,
    cost: npt.NDArray[np.float64],
    barriers: npt.NDArray[np.bool_],
    barrier_cmap: Any,
    *,
    alpha: float = 1.0,
) -> None:
    ax.imshow(
        np.ma.masked_where(barriers, cost),
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
        alpha=alpha,
    )
    draw_barriers(ax, barriers, barrier_cmap)


def draw_barriers(ax: Any, barriers: npt.NDArray[np.bool_], barrier_cmap: Any) -> None:
    ax.imshow(
        np.ma.masked_where(~barriers, barriers),
        cmap=barrier_cmap,
        origin="lower",
        interpolation="nearest",
        zorder=2,
    )
    ax.contour(
        barriers.astype(np.float64),
        levels=[0.5],
        colors="#222222",
        linewidths=0.55,
        origin="lower",
        zorder=3,
    )


def draw_routes(
    ax: Any,
    current_line: npt.NDArray[np.float64],
    parent_direction_line: npt.NDArray[np.float64],
    *,
    lw: float = 2.4,
) -> None:
    ax.plot(current_line[:, 0], current_line[:, 1], color="#d7191c", lw=lw, zorder=5)
    ax.plot(
        parent_direction_line[:, 0],
        parent_direction_line[:, 1],
        color="#2c7bb6",
        lw=lw * 0.92,
        ls="--",
        zorder=4,
    )


def draw_endpoints(ax: Any, source: tuple[int, int], destination: tuple[int, int]) -> None:
    ax.scatter(
        [source[1]], [source[0]], marker="o", c="#2ca25f", edgecolors="black", s=78, zorder=6
    )
    ax.scatter(
        [destination[1]],
        [destination[0]],
        marker="*",
        c="#ffd92f",
        edgecolors="black",
        s=150,
        zorder=6,
    )


def zoom_to_center(ax: Any, center: Point, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)


def polyline_length(line_xy: npt.NDArray[np.float64]) -> float:
    if len(line_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(line_xy, axis=0), axis=1).sum())


if __name__ == "__main__":
    raise SystemExit(main())
