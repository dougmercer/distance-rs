#!/usr/bin/env python
"""Create and plot a 1 m five-mile routing example.

The default domain is a long corridor: 5 miles by 101 meters at 1 meter
resolution. It is intentionally isotropic so WhiteboxTools CostDistance is a
like-for-like comparison. Use smaller --length-m/--width-m values for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from distance_rs import (
    RasterGrid,
    RasterSurface,
    VerticalFactor,
    distance_accumulation,
)
from distance_rs.baselines import (
    PathCostMetrics,
    compare_distances,
    json_safe,
    normalize_cell_size,
    path_cost_metrics,
    raster_dijkstra,
    trace_path_mask,
    trace_raster_path,
    whitebox_cost_distance,
)

METER_PER_MILE = 1609.344


@dataclass(frozen=True)
class CaseData:
    name: str
    description: str
    sources: npt.NDArray[np.float64]
    cost: npt.NDArray[np.float64]
    elevation: npt.NDArray[np.float64] | None
    barriers: npt.NDArray[np.bool_]
    vertical_factor: VerticalFactor
    cell_size: float | tuple[float, float] = 1.0
    exact_reference: npt.NDArray[np.float64] | None = None


@dataclass
class RouteResult:
    solver: str
    distance: npt.NDArray[np.float64]
    elapsed_sec: float
    destination_cost: float
    line_xy: npt.NDArray[np.float64] | None = None
    path_mask: npt.NDArray[np.bool_] | None = None
    reference_metrics: PathCostMetrics | None = None
    status: str = "ok"
    error: str | None = None


def source_cells(sources: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    return np.argwhere((sources != 0.0) & np.isfinite(sources)).astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    solvers = expand_solvers(args.solvers)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    case, destination = make_five_mile_case(
        length_m=args.length_m,
        width_m=args.width_m,
        cell_size=args.cell_size,
    )
    total_cells = case.cost.size
    print(
        f"domain: {case.cost.shape[1]} cols x {case.cost.shape[0]} rows "
        f"({total_cells:,} cells, cell_size={args.cell_size:g} m)",
        file=sys.stderr,
    )

    results: list[RouteResult] = []
    for solver in solvers:
        print(f"running {solver}", file=sys.stderr)
        try:
            if solver == "ordered_upwind":
                results.append(run_ordered_upwind(case, destination))
            elif solver == "raster_dijkstra":
                results.append(run_raster_dijkstra(case, destination))
            elif solver == "whitebox_cost_distance":
                results.append(run_whitebox(case, destination))
            else:
                raise ValueError(f"unhandled solver: {solver}")
        except Exception as exc:  # noqa: BLE001 - this is an example runner; keep going.
            print(f"{solver} failed: {exc}", file=sys.stderr)
            results.append(
                RouteResult(
                    solver=solver,
                    distance=np.full_like(case.cost, np.inf, dtype=np.float64),
                    elapsed_sec=math.nan,
                    destination_cost=math.inf,
                    status="failed",
                    error=str(exc),
                )
            )

    route_plot = output_dir / "five_mile_route_overlay.png"
    accumulation_plot = output_dir / "five_mile_accumulation_panels.png"
    plot_route_overlay(case, destination, results, route_plot, args.max_plot_cells)
    plot_accumulation_panels(case, destination, results, accumulation_plot, args.max_plot_cells)

    metadata_path = output_dir / "five_mile_route_results.json"
    write_metadata(metadata_path, case, destination, results, route_plot, accumulation_plot, args)

    print_summary(results, metadata_path, route_plot, accumulation_plot)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length-m",
        type=float,
        default=5.0 * METER_PER_MILE,
        help="Route corridor length in meters. Default is 5 miles.",
    )
    parser.add_argument(
        "--width-m",
        type=float,
        default=101.0,
        help="Route corridor width in meters.",
    )
    parser.add_argument("--cell-size", type=float, default=1.0, help="Cell size in meters.")
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["all"],
        help="Solvers to run: ordered_upwind, raster, whitebox, or all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/five_mile_route"),
        help="Directory for PNG and JSON outputs.",
    )
    parser.add_argument(
        "--max-plot-cells",
        type=int,
        default=3_000_000,
        help="Downsample plotting arrays above this many cells.",
    )
    args = parser.parse_args(argv)

    if args.length_m <= 0.0:
        parser.error("--length-m must be positive")
    if args.width_m <= 0.0:
        parser.error("--width-m must be positive")
    if args.cell_size <= 0.0:
        parser.error("--cell-size must be positive")
    if args.max_plot_cells < 10_000:
        parser.error("--max-plot-cells must be at least 10000")
    return args


def expand_solvers(names: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw_name in names:
        name = raw_name.lower()
        if name == "all":
            return ["ordered_upwind", "raster_dijkstra", "whitebox_cost_distance"]
        if name in {"ordered_upwind", "ours", "distance_rs", "oum"}:
            expanded.append("ordered_upwind")
        elif name in {"raster", "raster_dijkstra", "dijkstra", "local"}:
            expanded.append("raster_dijkstra")
        elif name in {"whitebox", "wbt", "whitebox_cost_distance"}:
            expanded.append("whitebox_cost_distance")
        else:
            raise SystemExit(f"unknown solver: {raw_name}")
    return list(dict.fromkeys(expanded))


def make_five_mile_case(
    *,
    length_m: float,
    width_m: float,
    cell_size: float,
) -> tuple[CaseData, tuple[int, int]]:
    cols = int(round(length_m / cell_size)) + 1
    rows = int(round(width_m / cell_size)) + 1
    if rows % 2 == 0:
        rows += 1

    yy, xx = np.mgrid[0:rows, 0:cols]
    x_m = xx * cell_size
    y_m = (yy - rows // 2) * cell_size
    normalized_x = x_m / max(length_m, cell_size)

    main_center = 0.25 * width_m * np.sin(
        2.0 * math.pi * normalized_x * 1.7
    ) + 0.11 * width_m * np.sin(2.0 * math.pi * normalized_x * 4.3 + 0.9)
    secondary_center = -0.28 * width_m * np.sin(2.0 * math.pi * normalized_x * 1.15 + 1.3)

    main_band = np.exp(-((y_m - main_center) ** 2) / (2.0 * (0.09 * width_m) ** 2))
    secondary_band = np.exp(-((y_m - secondary_center) ** 2) / (2.0 * (0.11 * width_m) ** 2))
    ridge = np.exp(-((y_m - 0.18 * width_m) ** 2) / (2.0 * (0.08 * width_m) ** 2))
    roughness = 0.12 * np.sin(2.0 * math.pi * normalized_x * 9.0 + y_m / max(width_m, 1.0))

    cost = 1.55 - 0.62 * main_band - 0.28 * secondary_band + 0.42 * ridge + roughness
    for center_x, center_y, radius_x, radius_y, penalty in [
        (0.18, -0.18, 0.055, 0.22, 1.10),
        (0.39, 0.20, 0.070, 0.18, 0.85),
        (0.64, -0.16, 0.080, 0.24, 1.10),
        (0.83, 0.12, 0.060, 0.20, 0.95),
    ]:
        ellipse = ((normalized_x - center_x) / radius_x) ** 2 + (
            (y_m / width_m - center_y) / radius_y
        ) ** 2
        cost += penalty * np.exp(-ellipse)
    cost = np.maximum(cost, 0.08).astype(np.float64)

    barriers = np.zeros((rows, cols), dtype=bool)
    for fraction, gap_y, thickness_m, gap_half_m in [
        (0.22, -0.24 * width_m, 9.0, 8.0),
        (0.47, 0.19 * width_m, 11.0, 10.0),
        (0.71, -0.10 * width_m, 10.0, 9.0),
    ]:
        col_start = int(round(fraction * (cols - 1)))
        thickness = max(1, int(round(thickness_m / cell_size)))
        gap_center = int(round(rows // 2 + gap_y / cell_size))
        gap_half = max(1, int(round(gap_half_m / cell_size)))
        col_slice = slice(max(0, col_start), min(cols, col_start + thickness))
        barriers[:, col_slice] = True
        barriers[
            max(0, gap_center - gap_half) : min(rows, gap_center + gap_half + 1), col_slice
        ] = False

    # Small impassable ponds near the otherwise attractive main band.
    for center_x, center_y, radius_x, radius_y in [
        (0.31, 0.06 * width_m, 0.020, 0.13 * width_m),
        (0.58, -0.22 * width_m, 0.025, 0.11 * width_m),
    ]:
        ellipse = ((normalized_x - center_x) / radius_x) ** 2 + ((y_m - center_y) / radius_y) ** 2
        barriers |= ellipse <= 1.0

    source_row = rows // 2
    source_col = min(cols - 1, max(0, int(round(5.0 / cell_size))))
    destination_row = rows // 2
    destination_col = max(0, cols - source_col - 1)
    barriers[source_row - 2 : source_row + 3, source_col - 2 : source_col + 3] = False
    barriers[
        destination_row - 2 : destination_row + 3,
        destination_col - 2 : destination_col + 3,
    ] = False

    sources = np.zeros((rows, cols), dtype=np.float64)
    sources[source_row, source_col] = 1.0

    case = CaseData(
        name="five_mile_corridor",
        description="5 mile by narrow 1 m isotropic cost-distance corridor",
        sources=sources,
        cost=cost,
        elevation=None,
        barriers=barriers,
        vertical_factor=VerticalFactor.from_any(None),
        cell_size=cell_size,
        exact_reference=None,
    )
    return case, (destination_row, destination_col)


def run_ordered_upwind(
    case: CaseData,
    destination: tuple[int, int],
) -> RouteResult:
    origin = origin_for_case(case)
    start = time.perf_counter()
    result = distance_accumulation(
        RasterSurface(
            case.cost,
            grid=RasterGrid(cell_size=case.cell_size, origin=origin),
            elevation=case.elevation,
            barriers=case.barriers,
        ),
        source=source_cells(case.sources),
        vertical_factor=case.vertical_factor,
    )
    elapsed = time.perf_counter() - start
    destination_cost = float(result.distance[destination])
    line_xy = (
        result.optimal_path_as_line(destination)
        if math.isfinite(destination_cost)
        else np.empty((0, 2), dtype=np.float64)
    )
    return RouteResult(
        solver="ordered_upwind",
        distance=result.distance,
        elapsed_sec=elapsed,
        destination_cost=destination_cost,
        line_xy=line_xy,
        reference_metrics=reference_metrics(case, destination, line_xy),
    )


def run_raster_dijkstra(case: CaseData, destination: tuple[int, int]) -> RouteResult:
    origin = origin_for_case(case)
    start = time.perf_counter()
    result = raster_dijkstra(
        case.sources,
        cost_surface=case.cost,
        elevation=case.elevation,
        barriers=case.barriers,
        vertical_factor=case.vertical_factor,
        cell_size=case.cell_size,
    )
    elapsed = time.perf_counter() - start
    distance = result.distance
    destination_cost = float(distance[destination])
    line_xy = (
        trace_raster_path(result.parent, destination, cell_size=case.cell_size, origin=origin)
        if math.isfinite(destination_cost)
        else np.empty((0, 2), dtype=np.float64)
    )
    return RouteResult(
        solver="raster_dijkstra",
        distance=distance,
        elapsed_sec=elapsed,
        destination_cost=destination_cost,
        line_xy=line_xy,
        reference_metrics=reference_metrics(case, destination, line_xy),
    )


def run_whitebox(case: CaseData, destination: tuple[int, int]) -> RouteResult:
    destination_raster = np.zeros(case.cost.shape, dtype=np.float64)
    destination_raster[destination] = 1.0
    start = time.perf_counter()
    result = whitebox_cost_distance(
        case.sources,
        cost_surface=case.cost,
        barriers=case.barriers,
        cell_size=case.cell_size,
        destinations=destination_raster,
    )
    elapsed = time.perf_counter() - start
    path_mask = (
        np.isfinite(result.pathway) & (result.pathway > 0.0)
        if result.pathway is not None
        else np.zeros_like(case.barriers)
    )
    line_xy = trace_path_mask(
        path_mask,
        case_source_cell(case),
        destination,
        cell_size=case.cell_size,
        origin=origin_for_case(case),
    )

    return RouteResult(
        solver="whitebox_cost_distance",
        distance=result.distance,
        elapsed_sec=elapsed,
        destination_cost=float(result.distance[destination]),
        path_mask=path_mask,
        reference_metrics=reference_metrics(case, destination, line_xy),
    )


def plot_route_overlay(
    case: CaseData,
    destination: tuple[int, int],
    results: list[RouteResult],
    output_path: Path,
    max_plot_cells: int,
) -> None:
    import matplotlib.pyplot as plt

    cost, barriers, stride = downsample_for_plot(case.cost, case.barriers, max_plot_cells)
    extent = plot_extent(case)
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    image = ax.imshow(
        np.ma.masked_where(barriers, cost),
        cmap="viridis",
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
    )
    if np.any(barriers):
        ax.imshow(
            np.ma.masked_where(~barriers, barriers),
            cmap="gray_r",
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            alpha=0.9,
        )

    colors = {
        "ordered_upwind": "#e41a1c",
        "raster_dijkstra": "#377eb8",
        "whitebox_cost_distance": "#ffcc00",
    }
    for result in results:
        if result.status != "ok":
            continue
        color = colors.get(result.solver, "white")
        label = f"{result.solver} ({result.elapsed_sec:.2f}s)"
        if result.line_xy is not None and len(result.line_xy) > 0:
            ax.plot(
                result.line_xy[:, 0] / METER_PER_MILE,
                result.line_xy[:, 1],
                color=color,
                linewidth=1.8,
                label=label,
            )
        elif result.path_mask is not None:
            mask = result.path_mask[::stride, ::stride]
            ax.imshow(
                np.ma.masked_where(~mask, mask),
                cmap=single_color_cmap(color),
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
                alpha=0.95,
            )
            ax.plot([], [], color=color, linewidth=2.0, label=label)

    source_row, source_col = np.argwhere(case.sources != 0.0)[0]
    source_x, source_y = cell_to_xy(case, int(source_row), int(source_col))
    dest_x, dest_y = cell_to_xy(case, destination[0], destination[1])
    ax.scatter([source_x / METER_PER_MILE], [source_y], marker="o", c="white", edgecolors="black")
    ax.scatter(
        [dest_x / METER_PER_MILE], [dest_y], marker="*", c="white", edgecolors="black", s=120
    )
    ax.set_title("Five-mile 1 m route comparison over synthetic cost surface")
    ax.set_xlabel("distance along corridor (miles)")
    ax.set_ylabel("cross-corridor offset (meters)")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(image, ax=ax, shrink=0.75)
    cbar.set_label("cost per meter")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_accumulation_panels(
    case: CaseData,
    destination: tuple[int, int],
    results: list[RouteResult],
    output_path: Path,
    max_plot_cells: int,
) -> None:
    import matplotlib.pyplot as plt

    ok_results = [result for result in results if result.status == "ok"]
    if not ok_results:
        return
    cost, barriers, stride = downsample_for_plot(case.cost, case.barriers, max_plot_cells)
    extent = plot_extent(case)
    fig, axes = plt.subplots(
        len(ok_results),
        1,
        figsize=(15, 3.3 * len(ok_results)),
        constrained_layout=True,
        squeeze=False,
    )
    source_row, source_col = np.argwhere(case.sources != 0.0)[0]
    source_x, source_y = cell_to_xy(case, int(source_row), int(source_col))
    dest_x, dest_y = cell_to_xy(case, destination[0], destination[1])

    for ax, result in zip(axes[:, 0], ok_results):
        ax.imshow(
            np.ma.masked_where(barriers, cost),
            cmap="gray",
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            alpha=0.45,
        )
        distance = result.distance[::stride, ::stride]
        finite = np.isfinite(distance)
        distance_plot = np.ma.masked_where(~finite, np.log1p(distance))
        image = ax.imshow(
            distance_plot,
            cmap="magma",
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            alpha=0.72,
        )
        if result.line_xy is not None and len(result.line_xy) > 0:
            ax.plot(result.line_xy[:, 0] / METER_PER_MILE, result.line_xy[:, 1], "cyan", lw=1.5)
        elif result.path_mask is not None:
            mask = result.path_mask[::stride, ::stride]
            ax.imshow(
                np.ma.masked_where(~mask, mask),
                cmap=single_color_cmap("#00ffff"),
                origin="lower",
                extent=extent,
                aspect="auto",
                interpolation="nearest",
                alpha=0.95,
            )
        ax.scatter(
            [source_x / METER_PER_MILE], [source_y], marker="o", c="white", edgecolors="black"
        )
        ax.scatter(
            [dest_x / METER_PER_MILE],
            [dest_y],
            marker="*",
            c="white",
            edgecolors="black",
            s=100,
        )
        ax.set_title(
            f"{result.solver}: dest cost={format_metric(result.destination_cost)}, "
            f"time={result.elapsed_sec:.2f}s"
        )
        ax.set_xlabel("distance along corridor (miles)")
        ax.set_ylabel("offset (m)")
        fig.colorbar(image, ax=ax, shrink=0.70, label="log1p accumulated cost")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def downsample_for_plot(
    cost: npt.NDArray[np.float64],
    barriers: npt.NDArray[np.bool_],
    max_plot_cells: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], int]:
    cells = cost.size
    stride = max(1, int(math.ceil(math.sqrt(cells / max_plot_cells))))
    return cost[::stride, ::stride], barriers[::stride, ::stride], stride


def single_color_cmap(color: str) -> Any:
    from matplotlib.colors import ListedColormap

    return ListedColormap([color])


def plot_extent(case: CaseData) -> list[float]:
    rows, cols = case.cost.shape
    cell_size_x, cell_size_y = normalize_cell_size(case.cell_size)
    y_min = -(rows // 2) * cell_size_y
    y_max = y_min + (rows - 1) * cell_size_y
    return [0.0, ((cols - 1) * cell_size_x) / METER_PER_MILE, y_min, y_max]


def origin_for_case(case: CaseData) -> tuple[float, float]:
    rows, _cols = case.cost.shape
    _cell_size_x, cell_size_y = normalize_cell_size(case.cell_size)
    return 0.0, -(rows // 2) * cell_size_y


def case_source_cell(case: CaseData) -> tuple[int, int]:
    source_cells_arr = np.argwhere((case.sources != 0.0) & np.isfinite(case.sources))
    if len(source_cells_arr) == 0:
        raise ValueError("case has no source cell")
    row, col = source_cells_arr[0]
    return int(row), int(col)


def cell_to_xy(case: CaseData, row: int, col: int) -> tuple[float, float]:
    cell_size_x, cell_size_y = normalize_cell_size(case.cell_size)
    origin_x, origin_y = origin_for_case(case)
    return origin_x + col * cell_size_x, origin_y + row * cell_size_y


def reference_metrics(
    case: CaseData,
    destination: tuple[int, int],
    line_xy: npt.NDArray[np.float64],
) -> PathCostMetrics | None:
    if len(line_xy) == 0:
        return None
    source = case_source_cell(case)
    return path_cost_metrics(
        RasterSurface(
            case.cost,
            grid=RasterGrid(cell_size=case.cell_size, origin=origin_for_case(case)),
            elevation=case.elevation,
            barriers=case.barriers,
        ),
        line_xy,
        vertical_factor=case.vertical_factor,
        source_xy=cell_to_xy(case, *source),
        destination_xy=cell_to_xy(case, *destination),
    )


def write_metadata(
    path: Path,
    case: CaseData,
    destination: tuple[int, int],
    results: list[RouteResult],
    route_plot: Path,
    accumulation_plot: Path,
    args: argparse.Namespace,
) -> None:
    comparisons: dict[str, Any] = {}
    ordered = next((result for result in results if result.solver == "ordered_upwind"), None)
    if ordered is not None and ordered.status == "ok":
        for result in results:
            if result is ordered or result.status != "ok":
                continue
            comparisons[result.solver] = compare_distances(ordered.distance, result.distance)

    payload = {
        "case": {
            "shape": list(case.cost.shape),
            "cell_size": case.cell_size,
            "length_m": args.length_m,
            "width_m": args.width_m,
            "destination": list(destination),
            "barrier_cells": int(np.count_nonzero(case.barriers)),
        },
        "outputs": {
            "route_overlay": str(route_plot),
            "accumulation_panels": str(accumulation_plot),
        },
        "solvers": [
            {
                "solver": result.solver,
                "status": result.status,
                "elapsed_sec": result.elapsed_sec,
                "destination_cost": result.destination_cost,
                "distance_rs_surface": None
                if result.reference_metrics is None
                else {
                    "cost": result.reference_metrics.cost,
                    "time_hours": result.reference_metrics.time_hours,
                    "distance": result.reference_metrics.distance,
                },
                "path_vertices": None if result.line_xy is None else int(len(result.line_xy)),
                "path_cells": None
                if result.path_mask is None
                else int(np.count_nonzero(result.path_mask)),
                "error": result.error,
            }
            for result in results
        ],
        "comparisons_to_ordered_upwind": comparisons,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def print_summary(
    results: list[RouteResult],
    metadata_path: Path,
    route_plot: Path,
    accumulation_plot: Path,
) -> None:
    print(
        "\nsolver                 status      seconds     destination_cost  "
        "rs_cost     rs_time_min  rs_distance"
    )
    print(
        "---------------------  ----------  ----------  ----------------  "
        "----------  -----------  -----------"
    )
    for result in results:
        ref = result.reference_metrics
        print(
            f"{result.solver:21}  {result.status:10}  "
            f"{format_metric(result.elapsed_sec):>10}  {format_metric(result.destination_cost):>16}  "
            f"{format_optional_metric(None if ref is None else ref.cost):>10}  "
            f"{format_optional_metric(None if ref is None else ref.time_hours * 60.0):>11}  "
            f"{format_optional_metric(None if ref is None else ref.distance):>11}"
        )
    print(f"\nroute overlay: {route_plot}")
    print(f"accumulation panels: {accumulation_plot}")
    print(f"metadata: {metadata_path}")


def format_metric(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    if abs(value) >= 1000.0 or (abs(value) < 0.001 and value != 0.0):
        return f"{value:.3e}"
    return f"{value:.3f}"


def format_optional_metric(value: float | None) -> str:
    return "n/a" if value is None else format_metric(value)


if __name__ == "__main__":
    raise SystemExit(main())
