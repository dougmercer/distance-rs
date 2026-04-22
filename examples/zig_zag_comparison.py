#!/usr/bin/env python
"""Plot flat-cost routes where raster Dijkstra stair-steps but ordered upwind does not."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from distance_rs import distance_accumulation, optimal_path_as_line
from distance_rs.baselines import raster_dijkstra, trace_raster_path


@dataclass(frozen=True)
class RouteMetrics:
    direct_length: float
    ordered_length: float
    dijkstra_length: float
    ordered_excess_pct: float
    dijkstra_excess_pct: float
    ordered_turn_degrees: float
    dijkstra_turn_degrees: float


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = 121, 151
    source = (16, 15)
    destinations = [(40, 55), (71, 48), (85, 130)]

    sources = np.zeros((rows, cols), dtype=np.float64)
    sources[source] = 1.0
    cost = np.ones((rows, cols), dtype=np.float64)

    ordered = distance_accumulation(
        sources,
        cost_surface=cost,
        search_radius=args.search_radius,
        use_surface_distance=False,
    )
    dijkstra = raster_dijkstra(
        sources,
        cost_surface=cost,
        use_surface_distance=False,
    )

    metrics = []
    routes = []
    for destination in destinations:
        ordered_line = optimal_path_as_line(ordered, destination)
        dijkstra_line = trace_raster_path(dijkstra.parent, destination)
        route_metrics = compute_metrics(source, destination, ordered_line, dijkstra_line)
        metrics.append(route_metrics)
        routes.append((destination, ordered_line, dijkstra_line, route_metrics))

    output_path = args.output_dir / "zig_zag_comparison.png"
    plot_routes(cost, source, routes, output_path)
    print_summary(routes, dijkstra.distance, ordered.distance, output_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for the comparison plot.",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=6.0,
        help="Ordered-upwind search radius in cells.",
    )
    return parser.parse_args()


def compute_metrics(
    source: tuple[int, int],
    destination: tuple[int, int],
    ordered_line: npt.NDArray[np.float64],
    dijkstra_line: npt.NDArray[np.float64],
) -> RouteMetrics:
    source_xy = np.asarray([source[1], source[0]], dtype=np.float64)
    destination_xy = np.asarray([destination[1], destination[0]], dtype=np.float64)
    direct_length = float(np.linalg.norm(destination_xy - source_xy))
    ordered_length = polyline_length(ordered_line)
    dijkstra_length = polyline_length(dijkstra_line)
    return RouteMetrics(
        direct_length=direct_length,
        ordered_length=ordered_length,
        dijkstra_length=dijkstra_length,
        ordered_excess_pct=100.0 * (ordered_length / direct_length - 1.0),
        dijkstra_excess_pct=100.0 * (dijkstra_length / direct_length - 1.0),
        ordered_turn_degrees=total_turn_degrees(ordered_line),
        dijkstra_turn_degrees=total_turn_degrees(dijkstra_line),
    )


def polyline_length(line: npt.NDArray[np.float64]) -> float:
    if len(line) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def total_turn_degrees(line: npt.NDArray[np.float64]) -> float:
    if len(line) < 3:
        return 0.0
    vectors = np.diff(line, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1.0e-12
    vectors = vectors[valid]
    lengths = lengths[valid]
    if len(vectors) < 2:
        return 0.0

    unit = vectors / lengths[:, np.newaxis]
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return float(angles.sum())


def plot_routes(
    cost: npt.NDArray[np.float64],
    source: tuple[int, int],
    routes: list[
        tuple[tuple[int, int], npt.NDArray[np.float64], npt.NDArray[np.float64], RouteMetrics]
    ],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        len(routes),
        figsize=(16, 8.0),
        constrained_layout=True,
        height_ratios=[3.0, 1.2],
    )
    for column, (destination, ordered_line, dijkstra_line, metrics) in enumerate(routes):
        ax = axes[0, column]
        error_ax = axes[1, column]
        ax.imshow(
            cost,
            cmap="Greys",
            origin="lower",
            vmin=0.0,
            vmax=1.2,
            interpolation="nearest",
            alpha=0.18,
        )
        ax.set_xticks(np.arange(-0.5, cost.shape[1], 10), minor=True)
        ax.set_yticks(np.arange(-0.5, cost.shape[0], 10), minor=True)
        ax.grid(which="minor", color="#d0d0d0", linewidth=0.45, alpha=0.7)

        source_xy = (source[1], source[0])
        destination_xy = (destination[1], destination[0])
        ax.plot(
            [source_xy[0], destination_xy[0]],
            [source_xy[1], destination_xy[1]],
            "--",
            color="black",
            linewidth=1.2,
            alpha=0.65,
            label="straight line",
        )
        ax.plot(
            ordered_line[:, 0],
            ordered_line[:, 1],
            color="#e41a1c",
            linewidth=2.1,
            zorder=3,
            label=f"ordered upwind (+{metrics.ordered_excess_pct:.1f}%)",
        )
        ax.plot(
            dijkstra_line[:, 0],
            dijkstra_line[:, 1],
            color="#377eb8",
            linewidth=1.35,
            alpha=0.9,
            zorder=2,
            label=f"8-neighbor Dijkstra (+{metrics.dijkstra_excess_pct:.1f}%)",
        )
        ax.scatter(
            dijkstra_line[:, 0],
            dijkstra_line[:, 1],
            color="#377eb8",
            s=13,
            alpha=0.9,
            linewidths=0.0,
            zorder=4,
        )
        ax.scatter(
            [source_xy[0], destination_xy[0]],
            [source_xy[1], destination_xy[1]],
            c=["white", "gold"],
            edgecolors="black",
            s=[70, 90],
            zorder=5,
        )
        ax.set_title(
            f"destination {destination}\n"
            f"turning: OUM {metrics.ordered_turn_degrees:.0f} deg, "
            f"Dijkstra {metrics.dijkstra_turn_degrees:.0f} deg"
        )
        ax.set_xlim(
            min(source_xy[0], destination_xy[0]) - 8, max(source_xy[0], destination_xy[0]) + 8
        )
        ax.set_ylim(
            min(source_xy[1], destination_xy[1]) - 8, max(source_xy[1], destination_xy[1]) + 8
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.legend(loc="upper left", fontsize=8)

        ordered_progress, ordered_error = cross_track_profile(ordered_line, source, destination)
        dijkstra_progress, dijkstra_error = cross_track_profile(dijkstra_line, source, destination)
        error_ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
        error_ax.plot(
            ordered_progress,
            ordered_error,
            color="#e41a1c",
            linewidth=2.0,
            label="ordered upwind",
        )
        error_ax.plot(
            dijkstra_progress,
            dijkstra_error,
            color="#377eb8",
            linewidth=1.3,
            marker=".",
            markersize=2.5,
            label="8-neighbor Dijkstra",
        )
        max_error = max(
            0.5,
            float(np.max(np.abs(ordered_error))) if len(ordered_error) else 0.0,
            float(np.max(np.abs(dijkstra_error))) if len(dijkstra_error) else 0.0,
        )
        error_ax.set_ylim(-1.15 * max_error, 1.15 * max_error)
        error_ax.set_title("cross-track error from straight line")
        error_ax.set_xlabel("distance along straight line")
        error_ax.set_ylabel("cells")
        error_ax.grid(True, color="#d0d0d0", linewidth=0.5)

    fig.suptitle("Flat unit-cost routes: ordered upwind avoids raster Dijkstra stair-steps")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def cross_track_profile(
    line: npt.NDArray[np.float64],
    source: tuple[int, int],
    destination: tuple[int, int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if len(line) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    source_xy = np.asarray([source[1], source[0]], dtype=np.float64)
    destination_xy = np.asarray([destination[1], destination[0]], dtype=np.float64)
    axis = destination_xy - source_xy
    length = np.linalg.norm(axis)
    if length <= 1.0e-12:
        return np.zeros(len(line), dtype=np.float64), np.zeros(len(line), dtype=np.float64)
    unit = axis / length
    normal = np.asarray([-unit[1], unit[0]], dtype=np.float64)
    delta = line - source_xy
    progress = delta @ unit
    cross_track = delta @ normal
    order = np.argsort(progress)
    return progress[order], cross_track[order]


def print_summary(
    routes: list[
        tuple[tuple[int, int], npt.NDArray[np.float64], npt.NDArray[np.float64], RouteMetrics]
    ],
    dijkstra_distance: npt.NDArray[np.float64],
    ordered_distance: npt.NDArray[np.float64],
    output_path: Path,
) -> None:
    print("destination  direct  ordered_len  dijkstra_len  ordered_cost  dijkstra_cost")
    print("-----------  ------  -----------  ------------  ------------  -------------")
    for destination, _ordered_line, _dijkstra_line, metrics in routes:
        print(
            f"{destination!s:>11}  "
            f"{metrics.direct_length:6.2f}  "
            f"{metrics.ordered_length:11.2f}  "
            f"{metrics.dijkstra_length:12.2f}  "
            f"{ordered_distance[destination]:12.2f}  "
            f"{dijkstra_distance[destination]:13.2f}"
        )
    print(f"\nplot: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
