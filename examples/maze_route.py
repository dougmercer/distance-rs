#!/usr/bin/env python
"""Plot an ordered-upwind vs 8-neighbor route through a generated maze."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from distance_rs import RasterSurface, SolverOptions, distance_accumulation, optimal_path_as_line
from distance_rs.baselines import compare_distances, raster_dijkstra, trace_raster_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source, destination, cost, barriers = make_case(
        maze_rows=args.maze_rows,
        maze_cols=args.maze_cols,
        scale=args.scale,
        seed=args.seed,
    )
    sources = np.zeros_like(cost)
    sources[source] = 1.0

    ordered = distance_accumulation(
        RasterSurface(cost, barriers=barriers),
        source=source,
        options=SolverOptions(
            stencil_radius=args.search_radius,
            use_surface_distance=False,
        ),
    )
    ordered_line = optimal_path_as_line(ordered, destination)
    dijkstra = raster_dijkstra(
        sources,
        cost_surface=cost,
        barriers=barriers,
        use_surface_distance=False,
    )
    dijkstra_line = trace_raster_path(dijkstra.parent, destination)
    comparison = compare_distances(ordered.distance, dijkstra.distance)

    output_path = args.output_dir / "maze_route.png"
    plot_case(
        source,
        destination,
        cost,
        barriers,
        ordered.distance,
        ordered_line,
        dijkstra.distance,
        dijkstra_line,
        output_path,
    )

    print(f"wrote {output_path}")
    print(f"ordered-upwind destination cost: {ordered.distance[destination]:.2f}")
    print(f"8-neighbor destination cost: {dijkstra.distance[destination]:.2f}")
    print(f"ordered-upwind route length: {polyline_length(ordered_line):.2f} cells")
    print(f"8-neighbor route length: {polyline_length(dijkstra_line):.2f} cells")
    print(f"distance rmse vs 8-neighbor: {comparison['rmse']:.2f}")
    print(f"barrier cells: {int(np.count_nonzero(barriers))}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/maze-route"),
        help="Directory for the output plot.",
    )
    parser.add_argument("--maze-rows", type=int, default=11, help="Logical maze rows.")
    parser.add_argument("--maze-cols", type=int, default=17, help="Logical maze columns.")
    parser.add_argument(
        "--scale",
        type=int,
        default=7,
        help="Raster cells per logical maze cell or wall.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Maze generation seed.")
    parser.add_argument(
        "--search-radius",
        type=float,
        default=7.0,
        help="Ordered-upwind search radius in cells.",
    )
    args = parser.parse_args()

    if args.maze_rows < 2:
        parser.error("--maze-rows must be at least 2")
    if args.maze_cols < 2:
        parser.error("--maze-cols must be at least 2")
    if args.scale < 3:
        parser.error("--scale must be at least 3")
    if args.search_radius <= 0.0:
        parser.error("--search-radius must be positive")
    return args


def make_case(
    *,
    maze_rows: int,
    maze_cols: int,
    scale: int,
    seed: int,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    maze = carve_perfect_maze(maze_rows, maze_cols, seed)
    barriers = np.kron(maze, np.ones((scale, scale), dtype=np.bool_))

    rows, cols = barriers.shape
    y, x = np.mgrid[:rows, :cols]
    corridor = ~barriers
    center_pull = 0.08 * np.cos(2.0 * math.pi * y / max(rows, 1))
    east_penalty = 0.10 * x / max(cols - 1, 1)
    cost = 1.0 + center_pull + east_penalty
    cost = np.where(corridor, cost, np.inf).astype(np.float64)

    source = logical_cell_center(0, 0, scale)
    destination = logical_cell_center(maze_rows - 1, maze_cols - 1, scale)
    clear_endpoint(barriers, cost, source)
    clear_endpoint(barriers, cost, destination)
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


def logical_cell_center(row: int, col: int, scale: int) -> tuple[int, int]:
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


def plot_case(
    source: tuple[int, int],
    destination: tuple[int, int],
    cost: npt.NDArray[np.float64],
    barriers: npt.NDArray[np.bool_],
    ordered_distance: npt.NDArray[np.float64],
    ordered_line: npt.NDArray[np.float64],
    dijkstra_distance: npt.NDArray[np.float64],
    dijkstra_line: npt.NDArray[np.float64],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    barrier_cmap = ListedColormap(["white"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    route_ax, difference_ax = axes
    draw_surface(route_ax, cost, barriers, barrier_cmap)
    route_ax.plot(dijkstra_line[:, 0], dijkstra_line[:, 1], color="#2c7bb6", lw=1.9)
    route_ax.plot(ordered_line[:, 0], ordered_line[:, 1], color="#d7191c", lw=2.3)
    draw_endpoints(route_ax, source, destination)
    route_ax.set_title("Maze route overlay")
    route_ax.set_xlabel("column")
    route_ax.set_ylabel("row")

    finite = np.isfinite(ordered_distance) & np.isfinite(dijkstra_distance) & ~barriers
    delta = np.full_like(ordered_distance, np.nan)
    delta[finite] = ordered_distance[finite] - dijkstra_distance[finite]
    max_abs = float(np.max(np.abs(delta[finite]))) if np.any(finite) else 1.0
    max_abs = max(max_abs, 1.0e-9)
    image = difference_ax.imshow(
        np.ma.masked_where(~finite, delta),
        cmap="coolwarm",
        origin="lower",
        interpolation="nearest",
        vmin=-max_abs,
        vmax=max_abs,
    )
    draw_barriers(difference_ax, barriers, barrier_cmap)
    difference_ax.plot(dijkstra_line[:, 0], dijkstra_line[:, 1], color="#2c7bb6", lw=1.6)
    difference_ax.plot(ordered_line[:, 0], ordered_line[:, 1], color="#d7191c", lw=1.9)
    draw_endpoints(difference_ax, source, destination)
    difference_ax.set_title("Accumulated cost difference")
    difference_ax.set_xlabel("column")
    difference_ax.set_ylabel("row")
    fig.colorbar(image, ax=difference_ax, shrink=0.78, label="OUM minus 8-neighbor cost")

    route_ax.legend(handles=legend_handles(), loc="upper right")
    fig.suptitle("Generated maze: ordered upwind vs 8-neighbor grid")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_surface(
    ax: Any,
    cost: npt.NDArray[np.float64],
    barriers: npt.NDArray[np.bool_],
    barrier_cmap: Any,
) -> None:
    ax.imshow(np.ma.masked_where(barriers, cost), cmap="viridis", origin="lower")
    draw_barriers(ax, barriers, barrier_cmap)


def draw_barriers(ax: Any, barriers: npt.NDArray[np.bool_], barrier_cmap: Any) -> None:
    ax.imshow(np.ma.masked_where(~barriers, barriers), cmap=barrier_cmap, origin="lower", zorder=2)
    ax.contour(
        barriers.astype(np.float64),
        levels=[0.5],
        colors="#222222",
        linewidths=0.55,
        origin="lower",
        zorder=3,
    )


def draw_endpoints(ax: Any, source: tuple[int, int], destination: tuple[int, int]) -> None:
    ax.scatter(
        [source[1]], [source[0]], marker="o", c="#2ca25f", edgecolors="black", s=82, zorder=5
    )
    ax.scatter(
        [destination[1]],
        [destination[0]],
        marker="*",
        c="#ffd92f",
        edgecolors="black",
        s=155,
        zorder=5,
    )


def legend_handles() -> list[Any]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    return [
        Patch(facecolor="white", edgecolor="black", label="maze walls"),
        Line2D([0], [0], color="#d7191c", lw=2.3, label="ordered upwind"),
        Line2D([0], [0], color="#2c7bb6", lw=1.9, label="8-neighbor Dijkstra"),
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


def polyline_length(line_xy: npt.NDArray[np.float64]) -> float:
    if len(line_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(line_xy, axis=0), axis=1).sum())


if __name__ == "__main__":
    raise SystemExit(main())
