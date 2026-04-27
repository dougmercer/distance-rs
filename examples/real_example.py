#!/usr/bin/env python
"""Route real GeoTIFF/GeoJSON inputs with ordered upwind and raster baselines.

Example:

    uv run --group plot --group whitebox python examples/real_example.py \
        --land-use data/land_use.tif \
        --elevation data/elevation.tif \
        --waypoints data/waypoints.geojson \
        --barriers data/barriers.geojson \
        --costs '{"1": 0.7, "2": 1.0, "3": 2.2, "4": 4.0}'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt

from distance_rs import (
    CostRaster,
    GeoBarriers,
    GeoPoints,
    GeoSurface,
    GridSpec,
    PathMetrics,
    evaluate_path_cost,
    load_points,
    load_surface,
    route_path,
)
from distance_rs.baselines import (
    raster_dijkstra,
    trace_path_mask,
    trace_raster_path,
    whitebox_cost_distance,
)
from distance_rs.metrics import combine_path_metrics, path_metrics
from distance_rs.plotting import plot_route_map

XY = tuple[float, float]


@dataclass(frozen=True)
class SolverRoute:
    solver: str
    elapsed_sec: float
    destination_cost: float
    path_xy: npt.NDArray[np.float64] | None
    legs: tuple[Any, ...]
    metrics: PathMetrics | None
    reference_metrics: PathMetrics | None = None
    status: str = "ok"
    error: str | None = None


@dataclass(frozen=True)
class BaselineLeg:
    index: int
    solver: str
    cost: float
    path_xy: npt.NDArray[np.float64] | None = None
    mask_xy: npt.NDArray[np.float64] | None = None
    metrics: PathMetrics | None = None
    reference_metrics: PathMetrics | None = None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cost = CostRaster(
        args.land_use,
        values=load_number_map(args.costs) if args.costs else None,
        blocked_values=load_number_set(args.blocked_values),
    )
    grid = GridSpec(crs=args.grid_crs, resolution=args.resolution)
    waypoints = GeoPoints(args.waypoints, crs=args.waypoint_crs)
    barriers = GeoBarriers(args.barriers, crs=args.barrier_crs) if args.barriers else None
    vertical_factor = load_json_or_string(args.vertical_factor)
    baseline_speed = float(args.baseline_speed)

    with_crs = GridSpec(crs=target_crs(cost, grid), resolution=args.resolution)
    waypoint_xy = load_points(waypoints, target_crs=with_crs.crs)
    if len(waypoint_xy) < 2:
        raise SystemExit("waypoints must contain at least two coordinates")

    results = [
        run_solver(
            "ordered_upwind",
            run_ordered_upwind,
            cost,
            args.elevation,
            waypoints,
            barriers,
            with_crs,
            args.margin,
            vertical_factor,
            baseline_speed,
        ),
        run_solver(
            "raster_dijkstra",
            run_raster_dijkstra_route,
            cost,
            args.elevation,
            waypoints,
            barriers,
            with_crs,
            args.margin,
            vertical_factor,
            baseline_speed,
        ),
        run_solver(
            "whitebox_cost_distance",
            run_whitebox_route,
            cost,
            args.elevation,
            waypoints,
            barriers,
            with_crs,
            args.margin,
            vertical_factor,
            baseline_speed,
        ),
    ]

    plot_path = args.output_dir / "real_example_routes.png"
    geojson_path = args.output_dir / "real_example_routes.geojson"
    summary_path = args.output_dir / "real_example_summary.json"
    plot_results(
        plot_path,
        cost,
        elevation=args.elevation,
        waypoints=waypoints,
        barriers=barriers,
        grid=with_crs,
        routes=results,
        margin=args.plot_margin,
        max_pixels=args.max_plot_pixels,
        labels=load_label_map(args.labels),
    )
    write_geojson(geojson_path, waypoint_xy=waypoint_xy, routes=results)
    write_summary(
        summary_path,
        args=args,
        waypoint_xy=waypoint_xy,
        routes=results,
        plot_path=plot_path,
        geojson_path=geojson_path,
    )
    print_summary(results, plot_path=plot_path, geojson_path=geojson_path, summary_path=summary_path)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--land-use", type=Path, required=True, help="Land-use or cost GeoTIFF.")
    parser.add_argument("--elevation", type=Path, required=True, help="Elevation GeoTIFF.")
    parser.add_argument("--waypoints", type=Path, required=True, help="Waypoint GeoJSON.")
    parser.add_argument("--barriers", type=Path, help="Barrier GeoJSON.")
    parser.add_argument(
        "--costs",
        help="JSON object or path mapping land-use values to traversal costs. Omit to use raw raster values.",
    )
    parser.add_argument("--labels", help="Optional JSON object or path mapping land-use values to legend labels.")
    parser.add_argument("--blocked-values", help="Optional JSON array or comma list of impassable land-use values.")
    parser.add_argument("--vertical-factor", help="Optional JSON object/string or path for vertical-factor settings.")
    parser.add_argument("--grid-crs", help="Target projected CRS. Defaults to the land-use raster CRS.")
    parser.add_argument("--waypoint-crs", help="Override waypoint GeoJSON CRS.")
    parser.add_argument("--barrier-crs", help="Override barrier GeoJSON CRS.")
    parser.add_argument("--resolution", type=float, help="Optional target grid resolution in map units.")
    parser.add_argument("--margin", type=float, default=250.0, help="Route-leg crop margin in map units.")
    parser.add_argument("--plot-margin", type=float, default=250.0, help="Plot bounds margin in map units.")
    parser.add_argument("--baseline-speed", type=float, default=5.0, help="Baseline speed in km/h.")
    parser.add_argument("--max-plot-pixels", type=int, default=2_500_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/real-example"))
    args = parser.parse_args(argv)
    if args.margin <= 0.0 or not math.isfinite(args.margin):
        parser.error("--margin must be positive and finite")
    if args.plot_margin < 0.0 or not math.isfinite(args.plot_margin):
        parser.error("--plot-margin must be non-negative and finite")
    if args.baseline_speed <= 0.0 or not math.isfinite(args.baseline_speed):
        parser.error("--baseline-speed must be positive and finite")
    return args


def target_crs(cost: CostRaster, grid: GridSpec) -> Any:
    import rasterio

    if grid.crs is not None:
        return grid.crs
    with rasterio.open(cost.path) as dataset:
        if dataset.crs is None:
            raise ValueError("land-use raster has no CRS; pass --grid-crs")
        return dataset.crs


def run_solver(
    name: str,
    function: Any,
    cost: CostRaster,
    elevation: Path,
    waypoints: GeoPoints,
    barriers: GeoBarriers | None,
    grid: GridSpec,
    margin: float,
    vertical_factor: Any,
    baseline_speed: float,
) -> SolverRoute:
    print(f"running {name}", file=sys.stderr)
    try:
        return function(
            cost,
            elevation=elevation,
            waypoints=waypoints,
            barriers=barriers,
            grid=grid,
            margin=margin,
            vertical_factor=vertical_factor,
            baseline_speed=baseline_speed,
        )
    except Exception as exc:  # noqa: BLE001 - real data examples should still write partial results.
        print(f"{name} failed: {exc}", file=sys.stderr)
        return SolverRoute(
            solver=name,
            elapsed_sec=math.nan,
            destination_cost=math.inf,
            path_xy=None,
            legs=(),
            metrics=None,
            status="failed",
            error=str(exc),
        )


def run_ordered_upwind(
    cost: CostRaster,
    *,
    elevation: Path,
    waypoints: GeoPoints,
    barriers: GeoBarriers | None,
    grid: GridSpec,
    margin: float,
    vertical_factor: Any,
    baseline_speed: float,
) -> SolverRoute:
    start = time.perf_counter()
    route = route_path(
        cost,
        waypoints,
        elevation=elevation,
        barriers=barriers,
        grid=grid,
        margin=margin,
        vertical_factor=vertical_factor,
        baseline_speed=baseline_speed,
        parallel=True,
    )
    return SolverRoute(
        solver="ordered_upwind",
        elapsed_sec=time.perf_counter() - start,
        destination_cost=route.metrics.cost if route.metrics is not None else math.inf,
        path_xy=route.path_xy,
        legs=route.legs,
        metrics=route.metrics,
        reference_metrics=route.metrics,
    )


def run_raster_dijkstra_route(
    cost: CostRaster,
    *,
    elevation: Path,
    waypoints: GeoPoints,
    barriers: GeoBarriers | None,
    grid: GridSpec,
    margin: float,
    vertical_factor: Any,
    baseline_speed: float,
) -> SolverRoute:
    start = time.perf_counter()
    legs: list[BaselineLeg] = []
    path_parts: list[npt.NDArray[np.float64]] = []
    total_metrics: PathMetrics | None = None
    total_reference_metrics: PathMetrics | None = None
    total_cost = 0.0
    waypoint_xy = load_points(waypoints, target_crs=grid.crs)

    for index, (start_xy, end_xy) in enumerate(zip(waypoint_xy, waypoint_xy[1:])):
        geo = load_leg_surface(cost, elevation, barriers, grid, start_xy, end_xy, margin)
        source_cell = geo.grid.xy_to_cell(*start_xy)
        destination_cell = geo.grid.xy_to_cell(*end_xy)
        sources = np.zeros(geo.grid.shape, dtype=np.float64)
        sources[source_cell] = 1.0
        result = raster_dijkstra(
            sources,
            cost_surface=geo.surface.cost,
            elevation=geo.surface.elevation,
            vertical_factor=vertical_factor,
            barriers=geo.surface.barriers,
            cell_size=geo.grid.cell_size,
        )
        destination_cost = float(result.distance[destination_cell])
        if not math.isfinite(destination_cost):
            raise RuntimeError(f"raster Dijkstra could not reach leg {index}")

        cell_line = trace_raster_path(result.parent, destination_cell, cell_size=geo.grid.cell_size)
        path_xy = geo.grid.raster_line_to_xy(cell_line)[::-1].copy()
        metrics = path_metrics(
            path_xy,
            cost=destination_cost,
            surface=geo,
            vertical_factor=vertical_factor,
            baseline_speed=baseline_speed,
        )
        reference_cost = evaluate_path_cost(geo, path_xy, vertical_factor=vertical_factor)
        reference_metrics = path_metrics(
            path_xy,
            cost=reference_cost,
            surface=geo,
            vertical_factor=vertical_factor,
            baseline_speed=baseline_speed,
        )
        total_cost += destination_cost
        total_metrics = metrics if total_metrics is None else combine_path_metrics(total_metrics, metrics)
        total_reference_metrics = (
            reference_metrics
            if total_reference_metrics is None
            else combine_path_metrics(total_reference_metrics, reference_metrics)
        )
        legs.append(
            BaselineLeg(
                index=index,
                solver="raster_dijkstra",
                cost=destination_cost,
                path_xy=path_xy,
                metrics=metrics,
                reference_metrics=reference_metrics,
            )
        )
        path_parts.append(path_xy if not path_parts else path_xy[1:])

    full_path = np.vstack(path_parts) if path_parts else np.empty((0, 2), dtype=np.float64)
    return SolverRoute(
        solver="raster_dijkstra",
        elapsed_sec=time.perf_counter() - start,
        destination_cost=total_cost,
        path_xy=full_path,
        legs=tuple(legs),
        metrics=total_metrics,
        reference_metrics=total_reference_metrics,
    )


def run_whitebox_route(
    cost: CostRaster,
    *,
    elevation: Path,
    waypoints: GeoPoints,
    barriers: GeoBarriers | None,
    grid: GridSpec,
    margin: float,
    vertical_factor: Any,
    baseline_speed: float,
) -> SolverRoute:
    start = time.perf_counter()
    legs: list[BaselineLeg] = []
    mask_parts: list[npt.NDArray[np.float64]] = []
    total_reference_metrics: PathMetrics | None = None
    total_cost = 0.0
    waypoint_xy = load_points(waypoints, target_crs=grid.crs)

    for index, (start_xy, end_xy) in enumerate(zip(waypoint_xy, waypoint_xy[1:])):
        geo = load_leg_surface(cost, elevation, barriers, grid, start_xy, end_xy, margin)
        source_cell = geo.grid.xy_to_cell(*start_xy)
        destination_cell = geo.grid.xy_to_cell(*end_xy)
        sources = np.zeros(geo.grid.shape, dtype=np.float64)
        destinations = np.zeros(geo.grid.shape, dtype=np.float64)
        sources[source_cell] = 1.0
        destinations[destination_cell] = 1.0
        result = whitebox_cost_distance(
            sources,
            cost_surface=geo.surface.cost,
            barriers=geo.surface.barriers,
            cell_size=geo.grid.cell_size,
            destinations=destinations,
        )
        destination_cost = float(result.distance[destination_cell])
        if not math.isfinite(destination_cost):
            raise RuntimeError(f"Whitebox CostDistance could not reach leg {index}")

        path_mask = (
            np.isfinite(result.pathway) & (result.pathway > 0.0)
            if result.pathway is not None
            else np.zeros(geo.grid.shape, dtype=bool)
        )
        mask_xy = path_mask_to_xy(geo, path_mask)
        cell_line = trace_path_mask(
            path_mask,
            source_cell,
            destination_cell,
            cell_size=geo.grid.cell_size,
        )
        path_xy = geo.grid.raster_line_to_xy(cell_line) if len(cell_line) else mask_xy
        reference_metrics = None
        if len(path_xy) > 0:
            reference_cost = evaluate_path_cost(geo, path_xy, vertical_factor=vertical_factor)
            reference_metrics = path_metrics(
                path_xy,
                cost=reference_cost,
                surface=geo,
                vertical_factor=vertical_factor,
                baseline_speed=baseline_speed,
            )
            total_reference_metrics = (
                reference_metrics
                if total_reference_metrics is None
                else combine_path_metrics(total_reference_metrics, reference_metrics)
            )
        metrics = PathMetrics(
            cost=destination_cost,
            distance_m=math.nan,
            surface_distance_m=math.nan,
            time_hours=destination_cost / (baseline_speed * 1000.0),
            average_speed_kmh=math.nan,
        )
        total_cost += destination_cost
        legs.append(
            BaselineLeg(
                index=index,
                solver="whitebox_cost_distance",
                cost=destination_cost,
                path_xy=path_xy,
                mask_xy=mask_xy,
                metrics=metrics,
                reference_metrics=reference_metrics,
            )
        )
        if len(mask_xy) > 0:
            mask_parts.append(mask_xy)

    masks = np.vstack(mask_parts) if mask_parts else np.empty((0, 2), dtype=np.float64)
    return SolverRoute(
        solver="whitebox_cost_distance",
        elapsed_sec=time.perf_counter() - start,
        destination_cost=total_cost,
        path_xy=masks,
        legs=tuple(legs),
        metrics=PathMetrics(
            cost=total_cost,
            distance_m=math.nan,
            surface_distance_m=math.nan,
            time_hours=total_cost / (baseline_speed * 1000.0),
            average_speed_kmh=math.nan,
        ),
        reference_metrics=total_reference_metrics,
    )


def load_leg_surface(
    cost: CostRaster,
    elevation: Path,
    barriers: GeoBarriers | None,
    grid: GridSpec,
    start_xy: XY,
    end_xy: XY,
    margin: float,
) -> GeoSurface:
    return load_surface(
        cost,
        elevation=elevation,
        barriers=barriers,
        grid=GridSpec(
            crs=grid.crs,
            resolution=grid.resolution,
            bounds=leg_bounds(start_xy, end_xy, margin),
        ),
    )


def leg_bounds(start_xy: XY, end_xy: XY, margin: float) -> tuple[float, float, float, float]:
    return (
        min(start_xy[0], end_xy[0]) - margin,
        min(start_xy[1], end_xy[1]) - margin,
        max(start_xy[0], end_xy[0]) + margin,
        max(start_xy[1], end_xy[1]) + margin,
    )


def path_mask_to_xy(geo: GeoSurface, path_mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
    rows, cols = np.nonzero(path_mask)
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    local_line = np.column_stack(
        [
            cols.astype(np.float64) * geo.grid.cell_size[0],
            rows.astype(np.float64) * geo.grid.cell_size[1],
        ]
    )
    return geo.grid.raster_line_to_xy(local_line)


def plot_results(
    path: Path,
    cost: CostRaster,
    *,
    elevation: Path,
    waypoints: GeoPoints,
    barriers: GeoBarriers | None,
    grid: GridSpec,
    routes: list[SolverRoute],
    margin: float,
    max_pixels: int,
    labels: dict[float, str] | None,
) -> None:
    successful: dict[str, Any] = {
        route_label(route.solver): route for route in routes if route.status == "ok"
    }
    whitebox = next((route for route in routes if route.solver == "whitebox_cost_distance"), None)
    if whitebox is not None and whitebox.status == "ok":
        successful["Whitebox CostDistance"] = SimpleNamespace(path_xy=None, legs=whitebox.legs)
    title = "Real Route Comparison"
    ordered = next((route for route in routes if route.solver == "ordered_upwind"), None)
    if ordered is not None and ordered.metrics is not None:
        title = f"Real Route Comparison: ordered upwind {ordered.metrics.time_hours * 60.0:.1f} min"
    plot_route_map(
        path,
        cost,
        elevation=elevation,
        waypoints=waypoints,
        barriers=barriers,
        routes=successful,
        grid=grid,
        margin=margin,
        max_pixels=max_pixels,
        land_use_labels=labels,
        title=title,
    )


def write_geojson(
    path: Path,
    *,
    waypoint_xy: tuple[XY, ...],
    routes: list[SolverRoute],
) -> None:
    features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "properties": {"kind": "waypoints"},
            "geometry": {"type": "LineString", "coordinates": waypoint_xy},
        }
    ]
    for route in routes:
        if route.status != "ok":
            continue
        if route.path_xy is not None and len(route.path_xy) > 0:
            geometry_type = "MultiPoint" if route.solver == "whitebox_cost_distance" else "LineString"
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "kind": "route",
                        "solver": route.solver,
                        "cost": finite_or_none(route.destination_cost),
                    },
                    "geometry": {
                        "type": geometry_type,
                        "coordinates": route.path_xy.tolist(),
                    },
                }
            )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    waypoint_xy: tuple[XY, ...],
    routes: list[SolverRoute],
    plot_path: Path,
    geojson_path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "inputs": {
                    "land_use": str(args.land_use),
                    "elevation": str(args.elevation),
                    "waypoints": str(args.waypoints),
                    "barriers": str(args.barriers) if args.barriers else None,
                },
                "waypoints": waypoint_xy,
                "margin": args.margin,
                "baseline_speed_kmh": args.baseline_speed,
                "plot_path": str(plot_path),
                "geojson_path": str(geojson_path),
                "results": [route_to_json(route) for route in routes],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def route_to_json(route: SolverRoute) -> dict[str, Any]:
    return {
        "solver": route.solver,
        "status": route.status,
        "error": route.error,
        "elapsed_sec": finite_or_none(route.elapsed_sec),
        "destination_cost": finite_or_none(route.destination_cost),
        "metrics": metrics_to_json(route.metrics),
        "reference_metrics": metrics_to_json(route.reference_metrics),
        "vertices": len(route.path_xy) if route.path_xy is not None else None,
    }


def metrics_to_json(metrics: PathMetrics | None) -> dict[str, float | None] | None:
    if metrics is None:
        return None
    return {
        "cost": finite_or_none(metrics.cost),
        "distance_m": finite_or_none(metrics.distance_m),
        "surface_distance_m": finite_or_none(metrics.surface_distance_m),
        "time_hours": finite_or_none(metrics.time_hours),
        "average_speed_kmh": finite_or_none(metrics.average_speed_kmh),
    }


def print_summary(
    routes: list[SolverRoute],
    *,
    plot_path: Path,
    geojson_path: Path,
    summary_path: Path,
) -> None:
    for route in routes:
        if route.status != "ok":
            print(f"{route.solver}: failed ({route.error})")
            continue
        metrics = route.reference_metrics or route.metrics
        metric_text = "metrics=n/a"
        if metrics is not None:
            metric_text = (
                f"{metrics.distance_m:.1f} m, "
                f"{metrics.time_hours * 60.0:.2f} min, "
                f"cost={metrics.cost:.3f}"
            )
        print(f"{route.solver}: elapsed={route.elapsed_sec:.2f}s, {metric_text}")
    print(f"plot: {plot_path}")
    print(f"geojson: {geojson_path}")
    print(f"summary: {summary_path}")


def load_json_value(value: str | None) -> Any:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def load_json_or_string(value: str | None) -> Any:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if value[:1] in {"{", "[", '"'}:
        return json.loads(value)
    return value


def load_number_map(value: str) -> dict[float, float]:
    raw = load_json_value(value)
    if not isinstance(raw, dict):
        raise ValueError("--costs must be a JSON object")
    return {float(key): float(cost) for key, cost in raw.items()}


def load_label_map(value: str | None) -> dict[float, str] | None:
    if value is None:
        return None
    raw = load_json_value(value)
    if not isinstance(raw, dict):
        raise ValueError("--labels must be a JSON object")
    return {float(key): str(label) for key, label in raw.items()}


def load_number_set(value: str | None) -> set[float] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists() or value.strip().startswith("["):
        raw = load_json_value(value)
        if not isinstance(raw, list):
            raise ValueError("--blocked-values JSON must be an array")
        return {float(item) for item in raw}
    return {float(item) for item in value.split(",") if item.strip()}


def route_label(solver: str) -> str:
    return {
        "ordered_upwind": "Ordered Upwind",
        "raster_dijkstra": "Raster Dijkstra",
        "whitebox_cost_distance": "Whitebox CostDistance",
    }.get(solver, solver.replace("_", " ").title())


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


if __name__ == "__main__":
    raise SystemExit(main())
