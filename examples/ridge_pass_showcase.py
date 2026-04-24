#!/usr/bin/env python
"""Show where distance-rs differs from Dijkstra and friction-only Whitebox workflows.

The synthetic map has a cheap straight road across the center, but that road
crosses a steep ridge face. A lower saddle sits off-center on a curved ramp.
distance-rs and the local raster Dijkstra baseline both receive the elevation
surface and a vertical-factor cutoff, so they must use the saddle. Whitebox
CostDistance receives the usual friction raster only, so its route shows what a
standard isotropic cost-distance workflow would prefer when slope rules are not
part of the model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from distance_rs import (
    RasterGrid,
    RasterSurface,
    SolverOptions,
    VerticalFactor,
    distance_accumulation,
)
from distance_rs.baselines import (
    MIN_COST,
    WHITEBOX_NODATA,
    compare_distances,
    json_safe,
    raster_dijkstra,
    read_whitebox_raster,
    trace_raster_path,
    write_whitebox_raster,
)


@dataclass(frozen=True)
class CaseData:
    sources: npt.NDArray[np.float64]
    cost: npt.NDArray[np.float64]
    elevation: npt.NDArray[np.float64]
    barriers: npt.NDArray[np.bool_]
    vertical_factor: VerticalFactor
    source: tuple[int, int]
    destination: tuple[int, int]
    cell_size: float
    cutoff_degrees: float


@dataclass
class RouteResult:
    solver: str
    distance: npt.NDArray[np.float64]
    elapsed_sec: float
    destination_cost: float
    line_xy: npt.NDArray[np.float64] | None = None
    path_mask: npt.NDArray[np.bool_] | None = None
    status: str = "ok"
    error: str | None = None
    note: str | None = None


def source_cells(sources: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    return np.argwhere((sources != 0.0) & np.isfinite(sources)).astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    case = make_case(
        rows=args.rows,
        cols=args.cols,
        cell_size=args.cell_size,
        cutoff_degrees=args.cutoff_degrees,
    )
    solvers = expand_solvers(args.solvers)

    results: list[RouteResult] = []
    for solver in solvers:
        print(f"running {solver}", file=sys.stderr)
        try:
            if solver == "ordered_upwind":
                results.append(run_ordered_upwind(case, search_radius=args.search_radius))
            elif solver == "raster_dijkstra":
                results.append(run_raster_dijkstra(case))
            elif solver == "whitebox_cost_distance":
                results.append(run_whitebox(case))
            else:
                raise ValueError(f"unhandled solver: {solver}")
        except Exception as exc:  # noqa: BLE001 - example runner should keep going.
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

    plot_path = output_dir / "ridge_pass_showcase.png"
    metadata_path = output_dir / "ridge_pass_showcase.json"
    plot_showcase(case, results, plot_path)
    write_metadata(metadata_path, case, results, plot_path, args)
    print_summary(case, results, metadata_path, plot_path)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=121, help="Raster height in cells.")
    parser.add_argument("--cols", type=int, default=181, help="Raster width in cells.")
    parser.add_argument("--cell-size", type=float, default=10.0, help="Cell size in meters.")
    parser.add_argument(
        "--search-radius",
        type=float,
        default=60.0,
        help="Ordered-upwind search radius in meters.",
    )
    parser.add_argument(
        "--cutoff-degrees",
        type=float,
        default=14.0,
        help="Maximum uphill/downhill vertical angle allowed for distance-rs and Dijkstra.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["all"],
        help="Solvers to run: ordered_upwind, raster, whitebox, or all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ridge-pass-showcase"),
        help="Directory for generated PNG and JSON outputs.",
    )
    args = parser.parse_args(argv)

    if args.rows < 61:
        parser.error("--rows must be at least 61")
    if args.cols < 81:
        parser.error("--cols must be at least 81")
    if args.cell_size <= 0.0 or not math.isfinite(args.cell_size):
        parser.error("--cell-size must be a positive finite value")
    if args.search_radius <= 0.0 or not math.isfinite(args.search_radius):
        parser.error("--search-radius must be a positive finite value")
    if not 0.0 < args.cutoff_degrees < 89.0:
        parser.error("--cutoff-degrees must be between 0 and 89")
    return args


def expand_solvers(names: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw_name in names:
        name = raw_name.lower()
        if name == "all":
            return ["ordered_upwind", "raster_dijkstra", "whitebox_cost_distance"]
        if name in {"ordered_upwind", "ours", "distance_rs", "oum"}:
            expanded.append("ordered_upwind")
        elif name in {"raster", "dijkstra", "raster_dijkstra"}:
            expanded.append("raster_dijkstra")
        elif name in {"whitebox", "wbt", "whitebox_cost_distance"}:
            expanded.append("whitebox_cost_distance")
        else:
            raise SystemExit(f"unknown solver: {raw_name}")
    return list(dict.fromkeys(expanded))


def make_case(*, rows: int, cols: int, cell_size: float, cutoff_degrees: float) -> CaseData:
    if rows % 2 == 0:
        rows += 1
    yy, xx = np.mgrid[0:rows, 0:cols]
    x = xx / max(cols - 1, 1)
    y = (yy - rows // 2) / max(rows // 2, 1)

    source = (rows // 2, 8)
    destination = (rows // 2, cols - 9)
    sources = np.zeros((rows, cols), dtype=np.float64)
    sources[source] = 1.0
    barriers = np.zeros((rows, cols), dtype=np.bool_)

    # Tall ridge at center, but a low saddle opens around y ~= -0.55.
    ridge_height = 120.0 - 112.0 * np.exp(-(((y + 0.55) / 0.20) ** 2))
    ridge = ridge_height * np.exp(-(((x - 0.52) / 0.035) ** 2))
    rolling_ground = 10.0 * np.sin(2.0 * math.pi * x) * np.exp(-((y / 0.8) ** 2))
    elevation = (ridge + rolling_ground).astype(np.float64)

    straight_road = np.exp(-((y / 0.045) ** 2))
    saddle_ramp_center = -0.55 * np.sin(math.pi * x)
    saddle_ramp = np.exp(-(((y - saddle_ramp_center) / 0.050) ** 2))
    saddle_floor = np.exp(-(((y + 0.55) / 0.22) ** 2)) * np.exp(-(((x - 0.52) / 0.18) ** 2))
    texture = 0.08 * np.sin(13.0 * math.pi * x + 3.0 * y) + 0.05 * np.cos(9.0 * math.pi * y)
    cost = 3.0 - 2.55 * straight_road - 2.05 * saddle_ramp - 0.45 * saddle_floor + texture
    cost = np.maximum(cost, 0.28).astype(np.float64)

    vertical_factor = VerticalFactor.from_any(
        {
            "type": "binary",
            "low_cut_angle": -cutoff_degrees,
            "high_cut_angle": cutoff_degrees,
        }
    )
    return CaseData(
        sources=sources,
        cost=cost,
        elevation=elevation,
        barriers=barriers,
        vertical_factor=vertical_factor,
        source=source,
        destination=destination,
        cell_size=cell_size,
        cutoff_degrees=cutoff_degrees,
    )


def run_ordered_upwind(case: CaseData, *, search_radius: float) -> RouteResult:
    start = time.perf_counter()
    result = distance_accumulation(
        RasterSurface(
            case.cost,
            grid=RasterGrid(cell_size=case.cell_size, origin=origin_for_case(case)),
            elevation=case.elevation,
            barriers=case.barriers,
        ),
        source=source_cells(case.sources),
        options=SolverOptions(
            vertical_factor=case.vertical_factor,
            stencil_radius=search_radius,
            use_surface_distance=True,
        ),
    )
    elapsed = time.perf_counter() - start
    destination_cost = float(result.distance[case.destination])
    line_xy = (
        result.optimal_path_as_line(case.destination)[::-1].copy()
        if math.isfinite(destination_cost)
        else np.empty((0, 2), dtype=np.float64)
    )
    return RouteResult(
        solver="ordered_upwind",
        distance=result.distance,
        elapsed_sec=elapsed,
        destination_cost=destination_cost,
        line_xy=line_xy,
        note="Continuous ordered-upwind update with elevation and vertical-factor cutoff.",
    )


def run_raster_dijkstra(case: CaseData) -> RouteResult:
    start = time.perf_counter()
    result = raster_dijkstra(
        case.sources,
        cost_surface=case.cost,
        elevation=case.elevation,
        barriers=case.barriers,
        vertical_factor=case.vertical_factor,
        cell_size=case.cell_size,
        use_surface_distance=True,
    )
    elapsed = time.perf_counter() - start
    destination_cost = float(result.distance[case.destination])
    line_xy = (
        trace_raster_path(
            result.parent,
            case.destination,
            cell_size=case.cell_size,
            origin=origin_for_case(case),
        )[::-1].copy()
        if math.isfinite(destination_cost)
        else np.empty((0, 2), dtype=np.float64)
    )
    return RouteResult(
        solver="raster_dijkstra",
        distance=result.distance,
        elapsed_sec=elapsed,
        destination_cost=destination_cost,
        line_xy=line_xy,
        note="Same cost, elevation, and vertical-factor rules, but constrained to 8 neighbors.",
    )


def run_whitebox(case: CaseData) -> RouteResult:
    try:
        import whitebox
    except ImportError as exc:
        raise RuntimeError(
            "Whitebox is optional. Run `uv sync --group whitebox` to include "
            "the friction-only Whitebox route in this example."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="distance-rs-ridge-whitebox-") as tmp_name:
        tmp = Path(tmp_name)
        source_path = tmp / "source.dep"
        cost_path = tmp / "cost.dep"
        dest_path = tmp / "destination.dep"
        accum_path = tmp / "accum.dep"
        backlink_path = tmp / "backlink.dep"
        pathway_path = tmp / "pathway.dep"

        valid_cost = np.isfinite(case.cost) & ~case.barriers
        source = np.full(case.sources.shape, WHITEBOX_NODATA, dtype=np.float64)
        source[(case.sources != 0.0) & np.isfinite(case.sources) & valid_cost] = 1.0
        cost = np.where(valid_cost, np.maximum(case.cost, MIN_COST), WHITEBOX_NODATA)
        destination = np.zeros(case.cost.shape, dtype=np.float64)
        destination[case.destination] = 1.0

        write_whitebox_raster(source_path, source, cell_size=case.cell_size, nodata=WHITEBOX_NODATA)
        write_whitebox_raster(cost_path, cost, cell_size=case.cell_size, nodata=WHITEBOX_NODATA)
        write_whitebox_raster(
            dest_path,
            destination,
            cell_size=case.cell_size,
            nodata=WHITEBOX_NODATA,
        )

        wbt = whitebox.WhiteboxTools()
        wbt.set_working_dir(str(tmp))
        if hasattr(wbt, "set_verbose_mode"):
            wbt.set_verbose_mode(False)

        start = time.perf_counter()
        exit_code = wbt.cost_distance(
            source_path.name,
            cost_path.name,
            accum_path.name,
            backlink_path.name,
            callback=lambda _message: None,
        )
        if exit_code not in (0, None):
            raise RuntimeError(f"Whitebox CostDistance failed with exit code {exit_code}")
        exit_code = wbt.cost_pathway(
            dest_path.name,
            backlink_path.name,
            pathway_path.name,
            zero_background=True,
            callback=lambda _message: None,
        )
        if exit_code not in (0, None):
            raise RuntimeError(f"Whitebox CostPathway failed with exit code {exit_code}")
        elapsed = time.perf_counter() - start

        distance = read_whitebox_raster(accum_path, nodata_as_inf=True)
        pathway = read_whitebox_raster(pathway_path, nodata_as_inf=False)
        path_mask = np.isfinite(pathway) & (pathway > 0.0)

    return RouteResult(
        solver="whitebox_cost_distance",
        distance=distance,
        elapsed_sec=elapsed,
        destination_cost=float(distance[case.destination]),
        path_mask=path_mask,
        note="Friction-only CostDistance route; elevation and vertical-factor cutoff are not inputs.",
    )


def plot_showcase(case: CaseData, results: list[RouteResult], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    extent = plot_extent(case)
    colors = {
        "ordered_upwind": "#e41a1c",
        "raster_dijkstra": "#377eb8",
        "whitebox_cost_distance": "#ffd92f",
    }

    cost_image = axes[0].imshow(
        case.cost,
        origin="lower",
        extent=extent,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )
    axes[0].set_title("Friction-only surface: the straight road is cheapest")
    fig.colorbar(cost_image, ax=axes[0], shrink=0.76, label="cost per meter")

    elevation_image = axes[1].imshow(
        case.elevation,
        origin="lower",
        extent=extent,
        cmap="terrain",
        aspect="auto",
        interpolation="nearest",
    )
    steep = too_steep_mask(case)
    axes[1].imshow(
        np.ma.masked_where(~steep, steep),
        origin="lower",
        extent=extent,
        cmap=single_color_cmap("#d7191c"),
        aspect="auto",
        interpolation="nearest",
        alpha=0.35,
    )
    axes[1].set_title("Terrain-aware surface: red cells exceed the slope cutoff")
    fig.colorbar(elevation_image, ax=axes[1], shrink=0.76, label="elevation (m)")

    for ax in axes:
        draw_routes(ax, case, results, colors)
        draw_endpoints(ax, case)
        ax.set_xlabel("east-west distance (m)")
        ax.set_ylabel("north-south offset (m)")
        ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_routes(
    ax: Any,
    case: CaseData,
    results: list[RouteResult],
    colors: dict[str, str],
) -> None:
    for result in results:
        if result.status != "ok":
            continue
        color = colors.get(result.solver, "white")
        label = label_for_result(result)
        if result.line_xy is not None and len(result.line_xy) > 0:
            ax.plot(
                result.line_xy[:, 0], result.line_xy[:, 1], color=color, linewidth=2.2, label=label
            )
        elif result.path_mask is not None:
            ax.imshow(
                np.ma.masked_where(~result.path_mask, result.path_mask),
                origin="lower",
                extent=plot_extent(case),
                cmap=single_color_cmap(color),
                aspect="auto",
                interpolation="nearest",
                alpha=0.95,
            )
            ax.plot([], [], color=color, linewidth=2.2, label=label)


def draw_endpoints(ax: Any, case: CaseData) -> None:
    source_x, source_y = cell_to_xy(case, *case.source)
    dest_x, dest_y = cell_to_xy(case, *case.destination)
    ax.scatter([source_x], [source_y], marker="o", c="white", edgecolors="black", s=54, zorder=5)
    ax.scatter([dest_x], [dest_y], marker="*", c="white", edgecolors="black", s=135, zorder=5)


def label_for_result(result: RouteResult) -> str:
    name = {
        "ordered_upwind": "distance-rs ordered upwind",
        "raster_dijkstra": "8-neighbor Dijkstra",
        "whitebox_cost_distance": "Whitebox friction-only",
    }.get(result.solver, result.solver)
    return f"{name}: {format_metric(result.destination_cost)}"


def too_steep_mask(case: CaseData) -> npt.NDArray[np.bool_]:
    slope_x = np.zeros_like(case.elevation, dtype=np.float64)
    slope_y = np.zeros_like(case.elevation, dtype=np.float64)
    slope_x[:, 1:] = np.abs(np.diff(case.elevation, axis=1)) / case.cell_size
    slope_y[1:, :] = np.abs(np.diff(case.elevation, axis=0)) / case.cell_size
    cutoff = math.tan(math.radians(case.cutoff_degrees))
    return (slope_x > cutoff) | (slope_y > cutoff)


def write_metadata(
    path: Path,
    case: CaseData,
    results: list[RouteResult],
    plot_path: Path,
    args: argparse.Namespace,
) -> None:
    ordered = next((result for result in results if result.solver == "ordered_upwind"), None)
    comparisons: dict[str, Any] = {}
    if ordered is not None and ordered.status == "ok":
        for result in results:
            if result is ordered or result.status != "ok":
                continue
            comparisons[result.solver] = compare_distances(ordered.distance, result.distance)

    payload = {
        "case": {
            "shape": list(case.cost.shape),
            "cell_size_m": case.cell_size,
            "source_cell": list(case.source),
            "destination_cell": list(case.destination),
            "vertical_factor": {
                "type": case.vertical_factor.type,
                "low_cut_angle": case.vertical_factor.low_cut_angle,
                "high_cut_angle": case.vertical_factor.high_cut_angle,
            },
            "direct_road_max_slope_degrees": max_row_slope(case, case.source[0]),
            "saddle_max_slope_degrees": max_row_slope(case, saddle_row(case)),
            "steep_cells": int(np.count_nonzero(too_steep_mask(case))),
        },
        "parameters": {
            "search_radius_m": args.search_radius,
            "cutoff_degrees": args.cutoff_degrees,
        },
        "outputs": {"plot": str(plot_path)},
        "solvers": [
            {
                "solver": result.solver,
                "status": result.status,
                "elapsed_sec": result.elapsed_sec,
                "destination_cost": result.destination_cost,
                "path_length_m": None if result.line_xy is None else path_length(result.line_xy),
                "path_vertices": None if result.line_xy is None else int(len(result.line_xy)),
                "path_cells": None
                if result.path_mask is None
                else int(np.count_nonzero(result.path_mask)),
                "note": result.note,
                "error": result.error,
            }
            for result in results
        ],
        "comparisons_to_ordered_upwind": comparisons,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def print_summary(
    case: CaseData,
    results: list[RouteResult],
    metadata_path: Path,
    plot_path: Path,
) -> None:
    print("\nterrain facts")
    print(f"  direct road max slope angle: {max_row_slope(case, case.source[0]):.1f} deg")
    print(f"  saddle row max slope angle:  {max_row_slope(case, saddle_row(case)):.1f} deg")
    print(f"  vertical cutoff:             +/-{case.cutoff_degrees:.1f} deg")
    print("\nsolver                   status      seconds     destination_cost  path_length_m")
    print("-----------------------  ----------  ----------  ----------------  -------------")
    for result in results:
        length = None if result.line_xy is None else path_length(result.line_xy)
        print(
            f"{result.solver:23}  {result.status:10}  "
            f"{format_metric(result.elapsed_sec):>10}  "
            f"{format_metric(result.destination_cost):>16}  "
            f"{format_optional_metric(length):>13}"
        )
        if result.status != "ok" and result.error:
            print(f"  {result.solver}: {result.error}")
    print(f"\nplot: {plot_path}")
    print(f"metadata: {metadata_path}")


def path_length(line_xy: npt.NDArray[np.float64]) -> float:
    if len(line_xy) < 2:
        return 0.0
    delta = np.diff(line_xy, axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def max_row_slope(case: CaseData, row: int) -> float:
    dz = np.abs(np.diff(case.elevation[row]))
    if dz.size == 0:
        return 0.0
    return float(np.degrees(np.arctan(np.max(dz) / case.cell_size)))


def saddle_row(case: CaseData) -> int:
    return int(round(case.cost.shape[0] // 2 - 0.55 * (case.cost.shape[0] // 2)))


def cell_to_xy(case: CaseData, row: int, col: int) -> tuple[float, float]:
    origin_x, origin_y = origin_for_case(case)
    return origin_x + col * case.cell_size, origin_y + row * case.cell_size


def origin_for_case(case: CaseData) -> tuple[float, float]:
    return 0.0, -(case.cost.shape[0] // 2) * case.cell_size


def plot_extent(case: CaseData) -> list[float]:
    rows, cols = case.cost.shape
    _origin_x, origin_y = origin_for_case(case)
    return [
        0.0,
        (cols - 1) * case.cell_size,
        origin_y,
        origin_y + (rows - 1) * case.cell_size,
    ]


def single_color_cmap(color: str) -> Any:
    from matplotlib.colors import ListedColormap

    return ListedColormap([color])


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
