#!/usr/bin/env python
"""Plot flat-cost routes where raster Dijkstra stair-steps but ordered upwind does not."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from distance_rs import SolverOptions, distance_accumulation, optimal_path_as_line
from distance_rs.baselines import raster_dijkstra, trace_raster_path


@dataclass(frozen=True)
class RouteMetrics:
    direct_length: float
    ordered_area: float
    dijkstra_area: float
    equal_dijkstra_path_count: int


@dataclass(frozen=True)
class RouteCase:
    angle_degrees: int
    destination: tuple[int, int]


@dataclass(frozen=True)
class DijkstraEnvelope:
    path_count: int
    min_line: npt.NDArray[np.float64]
    max_line: npt.NDArray[np.float64]
    polygon: npt.NDArray[np.float64]
    area_progress: npt.NDArray[np.float64]
    area_min: npt.NDArray[np.float64]
    area_max: npt.NDArray[np.float64]


@dataclass(frozen=True)
class DijkstraStepPlan:
    total_row_steps: int
    total_col_steps: int
    total_steps: int
    diagonal_steps: int
    straight_steps: int
    row_sign: int
    col_sign: int
    row_major: bool


@dataclass(frozen=True)
class RoutePlot:
    case: RouteCase
    ordered_line: npt.NDArray[np.float64]
    dijkstra_line: npt.NDArray[np.float64]
    dijkstra_envelope: DijkstraEnvelope
    metrics: RouteMetrics


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = 121, 151
    source = (16, 55)
    route_cases = [
        RouteCase(45, (86, 125)),  # Diagonal grid moves are enough.
        RouteCase(60, (94, 100)),
        RouteCase(75, (94, 76)),
        RouteCase(85, (96, 62)),
    ]

    sources = np.zeros((rows, cols), dtype=np.float64)
    sources[source] = 1.0
    cost = np.ones((rows, cols), dtype=np.float64)

    ordered = distance_accumulation(
        cost,
        source=source,
        options=SolverOptions(
            stencil_radius=args.search_radius,
            use_surface_distance=False,
        ),
    )
    dijkstra = raster_dijkstra(
        sources,
        cost_surface=cost,
        use_surface_distance=False,
    )

    routes = []
    for route_case in route_cases:
        destination = route_case.destination
        ordered_line = optimal_path_as_line(ordered, destination)
        dijkstra_line = trace_raster_path(dijkstra.parent, destination)
        dijkstra_envelope = equal_shortest_dijkstra_envelope(source, destination)
        route_metrics = compute_metrics(
            source,
            destination,
            ordered_line,
            dijkstra_line,
            dijkstra_envelope.path_count,
        )
        routes.append(
            RoutePlot(
                case=route_case,
                ordered_line=ordered_line,
                dijkstra_line=dijkstra_line,
                dijkstra_envelope=dijkstra_envelope,
                metrics=route_metrics,
            )
        )

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
        default=23.0,
        help="Compatibility search radius in cells; the native solver uses a local 3x3 stencil.",
    )
    return parser.parse_args()


def compute_metrics(
    source: tuple[int, int],
    destination: tuple[int, int],
    ordered_line: npt.NDArray[np.float64],
    dijkstra_line: npt.NDArray[np.float64],
    equal_dijkstra_path_count: int,
) -> RouteMetrics:
    source_xy = np.asarray([source[1], source[0]], dtype=np.float64)
    destination_xy = np.asarray([destination[1], destination[0]], dtype=np.float64)
    direct_length = float(np.linalg.norm(destination_xy - source_xy))
    return RouteMetrics(
        direct_length=direct_length,
        ordered_area=path_area_from_straight_line(ordered_line, source, destination),
        dijkstra_area=path_area_from_straight_line(dijkstra_line, source, destination),
        equal_dijkstra_path_count=equal_dijkstra_path_count,
    )


def plot_routes(
    cost: npt.NDArray[np.float64],
    source: tuple[int, int],
    routes: list[RoutePlot],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        len(routes),
        figsize=(20, 8.5),
        constrained_layout=True,
        height_ratios=[3.0, 1.2],
    )
    for column, route in enumerate(routes):
        route_case = route.case
        destination = route_case.destination
        ordered_line = route.ordered_line
        dijkstra_line = route.dijkstra_line
        envelope = route.dijkstra_envelope
        metrics = route.metrics
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
        straight_line = np.asarray([source_xy, destination_xy], dtype=np.float64)
        ax.plot(
            [source_xy[0], destination_xy[0]],
            [source_xy[1], destination_xy[1]],
            "--",
            color="black",
            linewidth=1.2,
            alpha=0.65,
            label="straight line",
        )
        if envelope.path_count > 1:
            ax.fill(
                envelope.polygon[:, 0],
                envelope.polygon[:, 1],
                color="#377eb8",
                alpha=0.3,
                linewidth=0.0,
                zorder=0,
                label="analytic equal-cost envelope",
            )
            for envelope_line in (envelope.min_line, envelope.max_line):
                ax.plot(
                    envelope_line[:, 0],
                    envelope_line[:, 1],
                    color="#377eb8",
                    linewidth=0.9,
                    linestyle=":",
                    alpha=0.6,
                    zorder=1,
                )
        ax.plot(
            ordered_line[:, 0],
            ordered_line[:, 1],
            color="#e41a1c",
            linewidth=2.1,
            zorder=3,
            label="ordered upwind",
        )
        ax.plot(
            dijkstra_line[:, 0],
            dijkstra_line[:, 1],
            color="#377eb8",
            linewidth=1.35,
            alpha=0.9,
            zorder=4,
            label="stored-parent Dijkstra",
        )
        ax.scatter(
            dijkstra_line[:, 0],
            dijkstra_line[:, 1],
            color="#377eb8",
            s=13,
            alpha=0.9,
            linewidths=0.0,
            zorder=5,
        )
        ax.scatter(
            [source_xy[0], destination_xy[0]],
            [source_xy[1], destination_xy[1]],
            c=["white", "gold"],
            edgecolors="black",
            s=[70, 90],
            zorder=6,
        )
        ax.set_title(f"{route_case.angle_degrees} deg", fontsize=10)
        set_centered_route_limits(
            ax,
            [
                straight_line,
                ordered_line,
                dijkstra_line,
                envelope.min_line,
                envelope.max_line,
            ],
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        ax.legend(loc="upper left", fontsize=7)

        ordered_progress, ordered_area = cumulative_cross_track_area_profile(
            ordered_line, source, destination
        )
        dijkstra_progress, dijkstra_area = cumulative_cross_track_area_profile(
            dijkstra_line, source, destination
        )
        if envelope.path_count > 1:
            error_ax.fill_between(
                envelope.area_progress,
                envelope.area_min,
                envelope.area_max,
                color="#377eb8",
                alpha=0.22,
                linewidth=0.0,
                zorder=0,
                label="analytic equal-cost envelope",
            )
        error_ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
        error_ax.plot(
            ordered_progress,
            ordered_area,
            color="#e41a1c",
            linewidth=2.0,
            label="ordered upwind",
        )
        error_ax.plot(
            dijkstra_progress,
            dijkstra_area,
            color="#377eb8",
            linewidth=1.3,
            marker=".",
            markersize=2.5,
            label="stored-parent Dijkstra",
        )
        max_area = max(
            1.0,
            float(ordered_area[-1]) if len(ordered_area) else 0.0,
            float(dijkstra_area[-1]) if len(dijkstra_area) else 0.0,
            float(np.max(envelope.area_max)) if len(envelope.area_max) else 0.0,
        )
        error_ax.set_xlim(0.0, metrics.direct_length)
        error_ax.set_ylim(-0.04 * max_area, 1.12 * max_area)
        error_ax.set_title("cumulative cross-track area")
        error_ax.set_xlabel("distance along straight line")
        error_ax.set_ylabel("cells^2")
        error_ax.grid(True, color="#d0d0d0", linewidth=0.5)

    fig.suptitle("Flat unit-cost routes: Dijkstra is exact at 45 deg, then drifts off-axis")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def set_centered_route_limits(
    ax: object,
    lines: list[npt.NDArray[np.float64]],
) -> None:
    points = [line for line in lines if len(line)]
    if not points:
        return
    combined = np.vstack(points)
    min_xy = np.min(combined, axis=0)
    max_xy = np.max(combined, axis=0)
    center = 0.5 * (min_xy + max_xy)
    span = max(24.0, float(np.max(max_xy - min_xy)) + 12.0)
    half_span = 0.5 * span
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)


def equal_shortest_dijkstra_envelope(
    source: tuple[int, int],
    destination: tuple[int, int],
) -> DijkstraEnvelope:
    step_plan = shortest_dijkstra_step_plan(source, destination)
    path_count = math.comb(step_plan.total_steps, step_plan.diagonal_steps)
    min_points: list[tuple[float, float]] = []
    max_points: list[tuple[float, float]] = []

    for step_index in range(step_plan.total_steps + 1):
        min_diagonals, max_diagonals = feasible_diagonal_range(step_plan, step_index)
        min_point, max_point = dijkstra_envelope_points_at_step(
            source,
            step_plan,
            step_index,
            min_diagonals,
            max_diagonals,
        )
        min_points.append(min_point)
        max_points.append(max_point)

    min_line = np.asarray(min_points, dtype=np.float64)
    max_line = np.asarray(max_points, dtype=np.float64)
    polygon = np.vstack([min_line, np.flipud(max_line)])
    area_progress, area_min, area_max = cumulative_cross_track_area_envelope(
        source,
        destination,
        step_plan,
    )
    return DijkstraEnvelope(
        path_count=path_count,
        min_line=min_line,
        max_line=max_line,
        polygon=polygon,
        area_progress=area_progress,
        area_min=area_min,
        area_max=area_max,
    )


def dijkstra_envelope_points_at_step(
    source: tuple[int, int],
    step_plan: DijkstraStepPlan,
    step_index: int,
    min_diagonals: int,
    max_diagonals: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    min_point = dijkstra_state_xy(source, step_plan, step_index, min_diagonals)
    max_point = dijkstra_state_xy(source, step_plan, step_index, max_diagonals)
    sort_axis = 0 if step_plan.row_major else 1
    if min_point[sort_axis] <= max_point[sort_axis]:
        return tuple(min_point), tuple(max_point)
    return tuple(max_point), tuple(min_point)


def cumulative_cross_track_area_envelope(
    source: tuple[int, int],
    destination: tuple[int, int],
    step_plan: DijkstraStepPlan,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    source_xy = np.asarray([source[1], source[0]], dtype=np.float64)
    destination_xy = np.asarray([destination[1], destination[0]], dtype=np.float64)
    axis = destination_xy - source_xy
    length = np.linalg.norm(axis)
    if length <= 1.0e-12:
        zeros = np.zeros(1, dtype=np.float64)
        return zeros, zeros, zeros
    unit = axis / length
    normal = np.asarray([-unit[1], unit[0]], dtype=np.float64)

    previous_min = {0: 0.0}
    previous_max = {0: 0.0}
    states_by_progress: dict[int, tuple[float, float, float]] = {0: (0.0, 0.0, 0.0)}

    for step_index in range(1, step_plan.total_steps + 1):
        current_min: dict[int, float] = {}
        current_max: dict[int, float] = {}
        min_diagonals, max_diagonals = feasible_diagonal_range(step_plan, step_index)

        for diagonals_used in range(min_diagonals, max_diagonals + 1):
            min_candidates = []
            max_candidates = []
            for previous_diagonals in (diagonals_used, diagonals_used - 1):
                if previous_diagonals not in previous_min:
                    continue
                segment_area = dijkstra_state_segment_area(
                    source,
                    step_plan,
                    source_xy,
                    unit,
                    normal,
                    step_index,
                    previous_diagonals,
                    diagonals_used,
                )
                min_candidates.append(previous_min[previous_diagonals] + segment_area)
                max_candidates.append(previous_max[previous_diagonals] + segment_area)

            current_min[diagonals_used] = min(min_candidates)
            current_max[diagonals_used] = max(max_candidates)
            progress, _cross_track = dijkstra_state_progress_and_abs_cross(
                source,
                step_plan,
                source_xy,
                unit,
                normal,
                step_index,
                diagonals_used,
            )
            key = dijkstra_state_progress_key(step_plan, step_index, diagonals_used)
            previous_state = states_by_progress.get(key)
            if previous_state is None:
                states_by_progress[key] = (
                    progress,
                    current_min[diagonals_used],
                    current_max[diagonals_used],
                )
            else:
                states_by_progress[key] = (
                    previous_state[0],
                    min(previous_state[1], current_min[diagonals_used]),
                    max(previous_state[2], current_max[diagonals_used]),
                )

        previous_min = current_min
        previous_max = current_max

    rows = [states_by_progress[key] for key in sorted(states_by_progress)]
    return (
        np.asarray([row[0] for row in rows], dtype=np.float64),
        np.asarray([row[1] for row in rows], dtype=np.float64),
        np.asarray([row[2] for row in rows], dtype=np.float64),
    )


def dijkstra_state_segment_area(
    source: tuple[int, int],
    step_plan: DijkstraStepPlan,
    source_xy: npt.NDArray[np.float64],
    unit: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    step_index: int,
    previous_diagonals: int,
    diagonals_used: int,
) -> float:
    previous_progress, previous_cross_track = dijkstra_state_progress_and_abs_cross(
        source,
        step_plan,
        source_xy,
        unit,
        normal,
        step_index - 1,
        previous_diagonals,
    )
    progress, cross_track = dijkstra_state_progress_and_abs_cross(
        source,
        step_plan,
        source_xy,
        unit,
        normal,
        step_index,
        diagonals_used,
    )
    return 0.5 * (previous_cross_track + cross_track) * (progress - previous_progress)


def dijkstra_state_progress_and_abs_cross(
    source: tuple[int, int],
    step_plan: DijkstraStepPlan,
    source_xy: npt.NDArray[np.float64],
    unit: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    step_index: int,
    diagonals_used: int,
) -> tuple[float, float]:
    point = np.asarray(
        dijkstra_state_xy(source, step_plan, step_index, diagonals_used),
        dtype=np.float64,
    )
    delta = point - source_xy
    return float(delta @ unit), float(abs(delta @ normal))


def dijkstra_state_xy(
    source: tuple[int, int],
    step_plan: DijkstraStepPlan,
    step_index: int,
    diagonals_used: int,
) -> tuple[float, float]:
    if step_plan.row_major:
        row = source[0] + step_plan.row_sign * step_index
        col = source[1] + step_plan.col_sign * diagonals_used
    else:
        row = source[0] + step_plan.row_sign * diagonals_used
        col = source[1] + step_plan.col_sign * step_index
    return float(col), float(row)


def dijkstra_state_progress_key(
    step_plan: DijkstraStepPlan,
    step_index: int,
    diagonals_used: int,
) -> int:
    if step_plan.row_major:
        return step_plan.total_col_steps * diagonals_used + step_plan.total_row_steps * step_index
    return step_plan.total_col_steps * step_index + step_plan.total_row_steps * diagonals_used


def feasible_diagonal_range(
    step_plan: DijkstraStepPlan,
    step_index: int,
) -> tuple[int, int]:
    return (
        max(0, step_index - step_plan.straight_steps),
        min(step_plan.diagonal_steps, step_index),
    )


def shortest_dijkstra_step_plan(
    source: tuple[int, int],
    destination: tuple[int, int],
) -> DijkstraStepPlan:
    dr = destination[0] - source[0]
    dc = destination[1] - source[1]
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    row_sign = sign(dr)
    col_sign = sign(dc)
    diagonal_steps = min(abs_dr, abs_dc)
    total_steps = max(abs_dr, abs_dc)
    return DijkstraStepPlan(
        total_row_steps=abs_dr,
        total_col_steps=abs_dc,
        total_steps=total_steps,
        diagonal_steps=diagonal_steps,
        straight_steps=total_steps - diagonal_steps,
        row_sign=row_sign,
        col_sign=col_sign,
        row_major=abs_dr >= abs_dc,
    )


def sign(value: int) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


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


def cumulative_cross_track_area_profile(
    line: npt.NDArray[np.float64],
    source: tuple[int, int],
    destination: tuple[int, int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    progress, cross_track = cross_track_profile(line, source, destination)
    if len(progress) < 2:
        return progress, np.zeros(len(progress), dtype=np.float64)
    segment_area = np.diff(progress) * (np.abs(cross_track[:-1]) + np.abs(cross_track[1:]))
    cumulative_area = np.concatenate(([0.0], np.cumsum(0.5 * segment_area)))
    return progress, cumulative_area


def path_area_from_straight_line(
    line: npt.NDArray[np.float64],
    source: tuple[int, int],
    destination: tuple[int, int],
) -> float:
    _progress, cumulative_area = cumulative_cross_track_area_profile(line, source, destination)
    return float(cumulative_area[-1]) if len(cumulative_area) else 0.0


def print_summary(
    routes: list[RoutePlot],
    dijkstra_distance: npt.NDArray[np.float64],
    ordered_distance: npt.NDArray[np.float64],
    output_path: Path,
) -> None:
    print(
        "angle  destination   direct  ordered_cost  dijkstra_cost  ordered_area  dijkstra_area  equal_paths"
    )
    print(
        "-----  -----------  ------  ------------  -------------  ------------  -------------  -----------"
    )
    for route in routes:
        route_case = route.case
        destination = route_case.destination
        metrics = route.metrics
        print(
            f"{route_case.angle_degrees:5.0f}  "
            f"{destination!s:>11}  "
            f"{metrics.direct_length:6.2f}  "
            f"{ordered_distance[destination]:12.2f}  "
            f"{dijkstra_distance[destination]:13.2f}  "
            f"{metrics.ordered_area:12.2f}  "
            f"{metrics.dijkstra_area:13.2f}  "
            f"{format_path_count(metrics.equal_dijkstra_path_count):>11}"
        )
    print(f"\nplot: {output_path}")


def format_path_count(count: int) -> str:
    if count < 1_000_000:
        return f"{count:,}"
    return f"{float(count):.3e}"


if __name__ == "__main__":
    raise SystemExit(main())
