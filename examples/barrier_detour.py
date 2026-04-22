#!/usr/bin/env python
"""Plot a small barrier-detour routing example with an explicit legend."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from distance_rs import distance_accumulation, optimal_path_as_line
from distance_rs.baselines import compare_distances, raster_dijkstra, trace_raster_path


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source, destination, cost, barriers = make_case()
    sources = np.zeros_like(cost)
    sources[source] = 1.0

    ordered = distance_accumulation(
        sources,
        cost_surface=cost,
        barriers=barriers,
        search_radius=args.search_radius,
        use_surface_distance=False,
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

    output_path = args.output_dir / "barrier_detour.png"
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
        default=Path("results/barrier-detour"),
        help="Directory for the output plot.",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=8.0,
        help="Ordered-upwind search radius in cells.",
    )
    return parser.parse_args()


def make_case() -> tuple[
    tuple[int, int],
    tuple[int, int],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    rows, cols = 96, 180
    y, x = np.mgrid[:rows, :cols]

    centerline = rows / 2 + 14.0 * np.sin(2.2 * math.pi * x / cols)
    easy_corridor = np.exp(-((y - centerline) ** 2) / (2.0 * 9.0**2))
    rough_patch = np.exp(-(((x - 118.0) / 34.0) ** 2 + ((y - 68.0) / 18.0) ** 2))
    cost = 1.65 - 0.82 * easy_corridor + 0.72 * rough_patch
    cost = np.clip(cost, 0.35, 2.7).astype(np.float64)

    barriers = np.zeros((rows, cols), dtype=np.bool_)

    # A vertical wall with two gates: route choice is possible, but crossing the
    # white wall itself is impossible.
    wall_cols = slice(61, 64)
    barriers[:, wall_cols] = True
    barriers[13:27, wall_cols] = False
    barriers[62:82, wall_cols] = False

    # A horizontal fence with one opening.
    fence_rows = slice(58, 61)
    barriers[fence_rows, 92:151] = True
    barriers[fence_rows, 117:129] = False

    # Two small impassable ponds.
    for center_col, center_row, radius_col, radius_row in [
        (101.0, 36.0, 13.0, 11.0),
        (141.0, 72.0, 15.0, 10.0),
    ]:
        ellipse = ((x - center_col) / radius_col) ** 2 + ((y - center_row) / radius_row) ** 2
        barriers |= ellipse <= 1.0

    source = (48, 9)
    destination = (48, 170)
    clear_endpoint(barriers, source)
    clear_endpoint(barriers, destination)
    return source, destination, cost, barriers


def clear_endpoint(barriers: npt.NDArray[np.bool_], point: tuple[int, int]) -> None:
    row, col = point
    barriers[max(0, row - 2) : row + 3, max(0, col - 2) : col + 3] = False


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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    barrier_cmap = ListedColormap(["white"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    route_ax, distance_ax = axes
    draw_surface(route_ax, cost, barriers, barrier_cmap)
    route_ax.plot(dijkstra_line[:, 0], dijkstra_line[:, 1], color="#2c7bb6", lw=2.0)
    route_ax.plot(ordered_line[:, 0], ordered_line[:, 1], color="#d7191c", lw=2.4)
    draw_endpoints(route_ax, source, destination)
    route_ax.set_title("Route overlay through barrier openings")
    route_ax.set_xlabel("column")
    route_ax.set_ylabel("row")

    finite = np.isfinite(ordered_distance) & np.isfinite(dijkstra_distance) & ~barriers
    delta = np.full_like(ordered_distance, np.nan)
    delta[finite] = ordered_distance[finite] - dijkstra_distance[finite]
    max_abs = float(np.max(np.abs(delta[finite]))) if np.any(finite) else 1.0
    max_abs = max(max_abs, 1.0e-9)
    distance_plot = np.ma.masked_where(~finite, delta)
    image = distance_ax.imshow(
        distance_plot,
        cmap="coolwarm",
        origin="lower",
        interpolation="nearest",
        vmin=-max_abs,
        vmax=max_abs,
    )
    draw_barriers(distance_ax, barriers, barrier_cmap)
    distance_ax.plot(dijkstra_line[:, 0], dijkstra_line[:, 1], color="#2c7bb6", lw=1.8)
    distance_ax.plot(ordered_line[:, 0], ordered_line[:, 1], color="#d7191c", lw=2.0)
    draw_endpoints(distance_ax, source, destination)
    distance_ax.set_title("Accumulated cost difference")
    distance_ax.set_xlabel("column")
    distance_ax.set_ylabel("row")
    fig.colorbar(image, ax=distance_ax, shrink=0.82, label="OUM minus 8-neighbor cost")

    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="barrier cells"),
        Line2D([0], [0], color="#d7191c", lw=2.4, label="ordered upwind"),
        Line2D([0], [0], color="#2c7bb6", lw=2.0, label="8-neighbor Dijkstra"),
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
    route_ax.legend(handles=legend_handles, loc="upper right")

    fig.suptitle("Barrier detour example: ordered upwind vs 8-neighbor grid")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_surface(
    ax: Any,
    cost: npt.NDArray[np.float64],
    barriers: npt.NDArray[np.bool_],
    barrier_cmap: Any,
) -> None:
    image = np.ma.masked_where(barriers, cost)
    ax.imshow(image, cmap="viridis", origin="lower", interpolation="nearest")
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
        linewidths=0.8,
        origin="lower",
        zorder=3,
    )


def draw_endpoints(ax: Any, source: tuple[int, int], destination: tuple[int, int]) -> None:
    ax.scatter(
        [source[1]], [source[0]], marker="o", c="#2ca25f", edgecolors="black", s=78, zorder=5
    )
    ax.scatter(
        [destination[1]],
        [destination[0]],
        marker="*",
        c="#ffd92f",
        edgecolors="black",
        s=150,
        zorder=5,
    )


def polyline_length(line_xy: npt.NDArray[np.float64]) -> float:
    if len(line_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(line_xy, axis=0), axis=1).sum())


if __name__ == "__main__":
    raise SystemExit(main())
