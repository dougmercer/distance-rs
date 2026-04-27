#!/usr/bin/env python
"""Large synthetic GeoTIFF comparison for distance-rs, Dijkstra, and Whitebox.

The generated rasters are intentionally much larger than the routed corridor.
Each solver is run leg-by-leg through cropped GeoTIFF windows so the example
exercises geospatial loading, barrier rasterization, elevation handling, and
route stitching without requiring the full raster to fit in memory.
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
import rasterio
from distance_rs import (
    CostRaster,
    GeoBarriers,
    GeoPoints,
    GeoSurface,
    GridSpec,
    PathMetrics,
    evaluate_path_cost,
    load_surface,
    route_path,
)
from distance_rs.baselines import (
    raster_dijkstra,
    trace_path_mask,
    trace_raster_path,
    whitebox_cost_distance,
)
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import MultiPoint, Polygon

CRS = "EPSG:32618"
WEST = 500_000.0
NORTH = 4_400_000.0
DEFAULT_SIZE = 12_000
DEFAULT_CELL_SIZE = 2.0
BLOCK_SIZE = 512
DATASET_VERSION = "mega_v2"
LAND_USE_COSTS = {
    1: 0.62,  # graded trail
    2: 0.95,  # meadow
    3: 1.30,  # brush
    4: 1.75,  # rough grass
    5: 2.30,  # forest
    6: 3.10,  # talus
    7: 3.85,  # wetland
}
VERTICAL_FACTOR = {
    "type": "symmetric_linear",
    "low_cut_angle": -62.0,
    "high_cut_angle": 62.0,
    "slope": 0.018,
}
LAND_USE_LABELS = {
    1: "Trail",
    2: "Meadow",
    3: "Brush",
    4: "Rough Grass",
    5: "Forest",
    6: "Talus",
    7: "Wetland",
}
Bounds = tuple[float, float, float, float]


@dataclass
class ComparisonLeg:
    index: int
    solver: str
    cost: float
    elapsed_sec: float
    path_xy: npt.NDArray[np.float64] | None = None
    mask_xy: npt.NDArray[np.float64] | None = None
    metrics: PathMetrics | None = None
    reference_metrics: PathMetrics | None = None


@dataclass
class ComparisonRoute:
    solver: str
    elapsed_sec: float
    destination_cost: float
    legs: tuple[ComparisonLeg, ...]
    path_xy: npt.NDArray[np.float64] | None = None
    reference_metrics: PathMetrics | None = None
    status: str = "ok"
    error: str | None = None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    solvers = expand_solvers(args.solvers)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    land_use_path = args.data_dir / f"synthetic_{DATASET_VERSION}_land_use_{args.size}.tif"
    elevation_path = args.data_dir / f"synthetic_{DATASET_VERSION}_elevation_{args.size}.tif"
    generate_geotiffs(
        land_use_path,
        elevation_path,
        size=args.size,
        cell_size=args.cell_size,
        overwrite=args.overwrite,
    )
    print(args)

    waypoints = route_waypoints(args.size, args.cell_size)
    barriers = route_barriers(args.size, args.cell_size)
    results: list[ComparisonRoute] = []
    for solver in solvers:
        print(f"running {solver}", file=sys.stderr)
        try:
            if solver == "ordered_upwind":
                results.append(
                    run_ordered_upwind(
                        land_use_path,
                        elevation_path=elevation_path,
                        waypoints=waypoints,
                        barriers=barriers,
                        crop_buffer=args.crop_buffer,
                        baseline_speed=args.baseline_speed,
                    )
                )
            elif solver == "raster_dijkstra":
                results.append(
                    run_raster_dijkstra_route(
                        land_use_path,
                        elevation_path=elevation_path,
                        waypoints=waypoints,
                        barriers=barriers,
                        crop_buffer=args.crop_buffer,
                        baseline_speed=args.baseline_speed,
                    )
                )
            elif solver == "whitebox_cost_distance":
                results.append(
                    run_whitebox_route(
                        land_use_path,
                        elevation_path=elevation_path,
                        waypoints=waypoints,
                        barriers=barriers,
                        crop_buffer=args.crop_buffer,
                        baseline_speed=args.baseline_speed,
                    )
                )
            else:
                raise ValueError(f"unhandled solver: {solver}")
        except Exception as exc:  # noqa: BLE001 - comparison examples should keep going.
            print(f"{solver} failed: {exc}", file=sys.stderr)
            results.append(
                ComparisonRoute(
                    solver=solver,
                    elapsed_sec=math.nan,
                    destination_cost=math.inf,
                    legs=(),
                    status="failed",
                    error=str(exc),
                )
            )

    plot_path = args.output_dir / "mega_geotiff_comparison.png"
    geojson_path = args.output_dir / "mega_geotiff_comparison.geojson"
    metadata_path = args.output_dir / "mega_geotiff_comparison.json"
    plot_comparison(
        plot_path,
        land_use_path=land_use_path,
        elevation_path=elevation_path,
        waypoints=waypoints,
        barriers=barriers,
        results=results,
        max_plot_pixels=args.max_plot_pixels,
    )
    write_geojson(geojson_path, waypoints=waypoints, barriers=barriers, results=results)
    write_metadata(
        metadata_path,
        land_use_path=land_use_path,
        elevation_path=elevation_path,
        waypoints=waypoints,
        barriers=barriers,
        results=results,
        args=args,
        plot_path=plot_path,
        geojson_path=geojson_path,
    )
    print_summary(
        results, plot_path=plot_path, geojson_path=geojson_path, metadata_path=metadata_path
    )
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Raster width/height.")
    parser.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE)
    parser.add_argument("--crop-buffer", type=float, default=420.0)
    parser.add_argument("--baseline-speed", type=float, default=5.0)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["all"],
        help="Solvers to run: ordered_upwind, raster_dijkstra, whitebox, or all.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mega-geotiff-comparison"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mega-geotiff-comparison"),
    )
    parser.add_argument("--max-plot-pixels", type=int, default=2_500_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.size < 1200:
        parser.error("--size must be at least 1200")
    if args.cell_size <= 0.0 or not math.isfinite(args.cell_size):
        parser.error("--cell-size must be positive and finite")
    if args.crop_buffer <= 0.0:
        parser.error("--crop-buffer must be positive")
    if args.baseline_speed <= 0.0:
        parser.error("--baseline-speed must be positive")
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


def generate_geotiffs(
    land_use_path: Path,
    elevation_path: Path,
    *,
    size: int,
    cell_size: float,
    overwrite: bool,
) -> None:
    if not overwrite and land_use_path.exists() and elevation_path.exists():
        print(f"using existing rasters in {land_use_path.parent}", file=sys.stderr)
        return

    print(
        f"generating {size}x{size} GeoTIFFs at {cell_size:g} m in {land_use_path.parent}",
        file=sys.stderr,
    )
    transform = from_origin(WEST, NORTH, cell_size, cell_size)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "crs": CRS,
        "transform": transform,
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "compress": "deflate",
        "predictor": 2,
        "bigtiff": "IF_SAFER",
    }
    with (
        rasterio.open(
            land_use_path, "w", **{**profile, "dtype": "uint8", "nodata": 0}
        ) as land_dataset,
        rasterio.open(
            elevation_path, "w", **{**profile, "dtype": "float32", "nodata": -9999.0}
        ) as elevation_dataset,
    ):
        x = np.arange(size, dtype=np.float32)[None, :]
        for row_start in range(0, size, BLOCK_SIZE):
            block_height = min(BLOCK_SIZE, size - row_start)
            y = np.arange(row_start, row_start + block_height, dtype=np.float32)[:, None]
            window = Window(0, row_start, size, block_height)
            land_dataset.write(synthetic_land_use(x, y, size), 1, window=window)
            elevation_dataset.write(synthetic_elevation(x, y, size), 1, window=window)


def synthetic_land_use(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    size: int,
) -> npt.NDArray[np.uint8]:
    nx = x / float(size)
    ny = y / float(size)
    ridge_a = ridge_curve_a(nx)
    ridge_b = ridge_curve_b(nx)
    river = river_curve(nx)
    trail_a = np.exp(-((ny - trail_curve_a(nx)) ** 2) / (2.0 * 0.0028**2))
    trail_b = np.exp(-((ny - trail_curve_b(nx)) ** 2) / (2.0 * 0.0036**2))
    wetland = (
        np.exp(-((ny - river) ** 2) / (2.0 * 0.012**2))
        + 0.75 * gaussian(nx, ny, 0.34, 0.62, 0.055, 0.030)
        + 0.65 * gaussian(nx, ny, 0.72, 0.36, 0.070, 0.035)
    )
    talus = np.exp(-((ny - ridge_a) ** 2) / (2.0 * 0.014**2)) + 0.75 * np.exp(
        -((ny - ridge_b) ** 2) / (2.0 * 0.016**2)
    )
    forest = (
        0.85 * gaussian(nx, ny, 0.22, 0.32, 0.090, 0.070)
        + 1.10 * gaussian(nx, ny, 0.43, 0.50, 0.120, 0.090)
        + 0.90 * gaussian(nx, ny, 0.64, 0.70, 0.140, 0.080)
        + 0.85 * gaussian(nx, ny, 0.82, 0.42, 0.100, 0.100)
    )
    texture = (
        0.55 * np.sin(nx * 79.0 + ny * 23.0)
        + 0.35 * np.cos(nx * 31.0 - ny * 67.0)
        + 0.18 * np.sin((nx + ny) * 163.0)
    )
    brush = 0.65 * forest + 0.55 * wetland + 0.18 * texture
    rough = 0.45 * texture + 0.35 * np.sin(nx * 19.0 - ny * 17.0)

    land_use = np.full((y.shape[0], x.shape[1]), 2, dtype=np.uint8)
    land_use[rough > 0.38] = 4
    land_use[brush > 0.78] = 3
    land_use[forest + 0.18 * texture > 0.88] = 5
    land_use[talus > 0.82] = 6
    land_use[wetland > 0.82] = 7
    land_use[(trail_a > 0.55) | (trail_b > 0.62)] = 1
    return land_use


def synthetic_elevation(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    size: int,
) -> npt.NDArray[np.float32]:
    nx = x / float(size)
    ny = y / float(size)
    ridge_a = 260.0 * np.exp(-((ny - ridge_curve_a(nx)) ** 2) / (2.0 * 0.020**2))
    ridge_b = 190.0 * np.exp(-((ny - ridge_curve_b(nx)) ** 2) / (2.0 * 0.026**2))
    saddles = (
        180.0 * gaussian(nx, ny, 0.31, 0.39, 0.055, 0.045)
        + 150.0 * gaussian(nx, ny, 0.57, 0.58, 0.070, 0.060)
        + 130.0 * gaussian(nx, ny, 0.79, 0.46, 0.060, 0.050)
    )
    river_valley = 90.0 * np.exp(-((ny - river_curve(nx)) ** 2) / (2.0 * 0.025**2))
    rolling = (
        24.0 * np.sin(nx * 23.0) + 18.0 * np.cos(ny * 19.0) + 7.0 * np.sin((nx + 1.7 * ny) * 61.0)
    )
    trend = 400.0 + 130.0 * (1.0 - ny) + 55.0 * nx
    return (trend + ridge_a + ridge_b - saddles - river_valley + rolling).astype(np.float32)


def ridge_curve_a(nx: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 0.38 + 0.070 * np.sin(nx * 13.0 + 0.7) + 0.025 * np.sin(nx * 41.0)


def ridge_curve_b(nx: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 0.64 + 0.060 * np.sin(nx * 11.0 - 1.1) + 0.020 * np.cos(nx * 37.0)


def river_curve(nx: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 0.53 + 0.105 * np.sin(nx * 8.5 + 0.2) + 0.030 * np.sin(nx * 29.0)


def trail_curve_a(nx: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 0.77 - 0.42 * nx + 0.065 * np.sin(nx * 18.0)


def trail_curve_b(nx: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 0.23 + 0.48 * nx + 0.070 * np.sin(nx * 15.0 + 1.8)


def gaussian(
    nx: npt.NDArray[np.float32],
    ny: npt.NDArray[np.float32],
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> npt.NDArray[np.float32]:
    return np.exp(
        -((nx - center_x) ** 2 / (2.0 * radius_x**2) + (ny - center_y) ** 2 / (2.0 * radius_y**2))
    )


def route_waypoints(size: int, cell_size: float) -> list[tuple[float, float]]:
    span = size * cell_size
    fractions = [
        (0.08, 0.78),
        (0.27, 0.42),
        (0.48, 0.68),
        (0.70, 0.33),
        (0.82, 0.55),
        (0.92, 0.25),
    ]
    return [(WEST + span * x, NORTH - span * y) for x, y in fractions]


def route_barriers(size: int, cell_size: float) -> list[Polygon]:
    span = size * cell_size
    specs: list[tuple[float, float, list[tuple[float, float]]]] = [
        (
            0.18,
            0.60,
            [(-170.0, -300.0), (120.0, -280.0), (170.0, 260.0), (-120.0, 320.0)],
        ),
        (
            0.37,
            0.54,
            [(-230.0, -160.0), (40.0, -240.0), (230.0, 40.0), (-70.0, 250.0)],
        ),
        (
            0.60,
            0.47,
            [(-160.0, -260.0), (150.0, -310.0), (210.0, 180.0), (-110.0, 280.0)],
        ),
    ]
    return [meter_polygon(span, center_x, center_y, points) for center_x, center_y, points in specs]


def meter_polygon(
    span: float,
    center_x: float,
    center_y: float,
    points: list[tuple[float, float]],
) -> Polygon:
    origin_x = WEST + span * center_x
    origin_y = NORTH - span * center_y
    return Polygon([(origin_x + east, origin_y + north) for east, north in points])


def run_ordered_upwind(
    land_use_path: Path,
    *,
    elevation_path: Path,
    waypoints: list[tuple[float, float]],
    barriers: list[Polygon],
    crop_buffer: float,
    baseline_speed: float,
) -> ComparisonRoute:
    start = time.perf_counter()
    route = route_path(
        CostRaster(land_use_path, values=LAND_USE_COSTS),
        GeoPoints(waypoints, crs=CRS),
        barriers=[GeoBarriers(barrier, crs=CRS) for barrier in barriers],
        elevation=elevation_path,
        grid=GridSpec(crs=CRS),
        margin=crop_buffer,
        vertical_factor=VERTICAL_FACTOR,
        baseline_speed=baseline_speed,
        parallel=True,
    )
    elapsed = time.perf_counter() - start
    legs = tuple(
        ComparisonLeg(
            index=leg.index,
            solver="ordered_upwind",
            cost=leg.cost,
            elapsed_sec=math.nan,
            path_xy=leg.path_xy,
            metrics=leg.metrics,
            reference_metrics=leg.metrics,
        )
        for leg in route.legs
    )
    return ComparisonRoute(
        solver="ordered_upwind",
        elapsed_sec=elapsed,
        destination_cost=route.metrics.cost if route.metrics is not None else math.inf,
        legs=legs,
        path_xy=route.path_xy,
        reference_metrics=route.metrics,
    )


def run_raster_dijkstra_route(
    land_use_path: Path,
    *,
    elevation_path: Path,
    waypoints: list[tuple[float, float]],
    barriers: list[Polygon],
    crop_buffer: float,
    baseline_speed: float,
) -> ComparisonRoute:
    start_total = time.perf_counter()
    legs: list[ComparisonLeg] = []
    path_parts: list[npt.NDArray[np.float64]] = []
    total_cost = 0.0
    total_reference_metrics: PathMetrics | None = None
    for index, (start_xy, end_xy) in enumerate(zip(waypoints, waypoints[1:])):
        leg_start = time.perf_counter()
        geo = load_leg_surface(
            land_use_path, elevation_path, barriers, start_xy, end_xy, crop_buffer
        )
        source_cell = geo.grid.xy_to_cell(*start_xy)
        destination_cell = geo.grid.xy_to_cell(*end_xy)
        sources = np.zeros(geo.grid.shape, dtype=np.float64)
        sources[source_cell] = 1.0
        result = raster_dijkstra(
            sources,
            cost_surface=geo.surface.cost,
            elevation=geo.surface.elevation,
            vertical_factor=VERTICAL_FACTOR,
            barriers=geo.surface.barriers,
            cell_size=geo.grid.cell_size,
        )
        destination_cost = float(result.distance[destination_cell])
        if not math.isfinite(destination_cost):
            raise RuntimeError(f"raster Dijkstra could not reach leg {index}")
        cell_line = trace_raster_path(result.parent, destination_cell, cell_size=geo.grid.cell_size)
        path_xy = geo.grid.raster_line_to_xy(cell_line)[::-1].copy()
        metrics = path_metrics(
            path_xy, cost=destination_cost, geo=geo, baseline_speed=baseline_speed
        )
        reference_cost = evaluate_path_cost(geo, path_xy, vertical_factor=VERTICAL_FACTOR)
        reference_metrics = path_metrics(
            path_xy, cost=reference_cost, geo=geo, baseline_speed=baseline_speed
        )
        total_reference_metrics = (
            reference_metrics
            if total_reference_metrics is None
            else combine_metrics(total_reference_metrics, reference_metrics)
        )
        legs.append(
            ComparisonLeg(
                index=index,
                solver="raster_dijkstra",
                cost=destination_cost,
                elapsed_sec=time.perf_counter() - leg_start,
                path_xy=path_xy,
                metrics=metrics,
                reference_metrics=reference_metrics,
            )
        )
        total_cost += destination_cost
        path_parts.append(path_xy if not path_parts else path_xy[1:])
    return ComparisonRoute(
        solver="raster_dijkstra",
        elapsed_sec=time.perf_counter() - start_total,
        destination_cost=total_cost,
        legs=tuple(legs),
        path_xy=np.vstack(path_parts),
        reference_metrics=total_reference_metrics,
    )


def run_whitebox_route(
    land_use_path: Path,
    *,
    elevation_path: Path,
    waypoints: list[tuple[float, float]],
    barriers: list[Polygon],
    crop_buffer: float,
    baseline_speed: float,
) -> ComparisonRoute:
    start_total = time.perf_counter()
    legs: list[ComparisonLeg] = []
    total_cost = 0.0
    total_reference_metrics: PathMetrics | None = None
    for index, (start_xy, end_xy) in enumerate(zip(waypoints, waypoints[1:])):
        leg_start = time.perf_counter()
        geo = load_leg_surface(
            land_use_path, elevation_path, barriers, start_xy, end_xy, crop_buffer
        )
        source_cell = geo.grid.xy_to_cell(*start_xy)
        destination_cell = geo.grid.xy_to_cell(*end_xy)
        sources = np.zeros(geo.grid.shape, dtype=np.float64)
        sources[source_cell] = 1.0
        destinations = np.zeros(geo.grid.shape, dtype=np.float64)
        destinations[destination_cell] = 1.0
        result = whitebox_cost_distance(
            sources,
            cost_surface=geo.surface.cost,
            barriers=geo.surface.barriers,
            cell_size=geo.grid.cell_size,
            destinations=destinations,
        )
        distance = result.distance
        pathway = result.pathway
        path_mask = np.zeros(geo.grid.shape, dtype=bool)
        if pathway is not None:
            path_mask = np.isfinite(pathway) & (pathway > 0.0)
        mask_xy = path_mask_to_xy(geo, path_mask)
        cell_line = trace_path_mask(
            path_mask,
            source_cell,
            destination_cell,
            cell_size=geo.grid.cell_size,
        )
        path_xy = geo.grid.raster_line_to_xy(cell_line) if len(cell_line) else None

        destination_cost = float(distance[destination_cell])
        total_cost += destination_cost
        reference_metrics = None
        if path_xy is not None and len(path_xy) > 0:
            reference_cost = evaluate_path_cost(geo, path_xy, vertical_factor=VERTICAL_FACTOR)
            reference_metrics = path_metrics(
                path_xy, cost=reference_cost, geo=geo, baseline_speed=baseline_speed
            )
            total_reference_metrics = (
                reference_metrics
                if total_reference_metrics is None
                else combine_metrics(total_reference_metrics, reference_metrics)
            )
        metrics = PathMetrics(
            cost=destination_cost,
            distance_m=math.nan,
            surface_distance_m=math.nan,
            time_hours=destination_cost / (baseline_speed * 1000.0),
            average_speed_kmh=math.nan,
        )
        legs.append(
            ComparisonLeg(
                index=index,
                solver="whitebox_cost_distance",
                cost=destination_cost,
                elapsed_sec=time.perf_counter() - leg_start,
                mask_xy=mask_xy,
                metrics=metrics,
                reference_metrics=reference_metrics,
            )
        )
    return ComparisonRoute(
        solver="whitebox_cost_distance",
        elapsed_sec=time.perf_counter() - start_total,
        destination_cost=total_cost,
        legs=tuple(legs),
        reference_metrics=total_reference_metrics,
    )


def load_leg_surface(
    land_use_path: Path,
    elevation_path: Path,
    barriers: list[Polygon],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    crop_buffer: float,
) -> GeoSurface:
    return load_surface(
        CostRaster(land_use_path, values=LAND_USE_COSTS),
        elevation=elevation_path,
        barriers=[GeoBarriers(barrier, crs=CRS) for barrier in barriers],
        grid=GridSpec(crs=CRS, bounds=leg_bounds(start_xy, end_xy, crop_buffer)),
    )


def leg_bounds(start_xy: tuple[float, float], end_xy: tuple[float, float], margin: float) -> Bounds:
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


def path_metrics(
    path_xy: npt.NDArray[np.float64],
    *,
    cost: float,
    geo: GeoSurface,
    baseline_speed: float,
) -> PathMetrics:
    distance_m = path_distance(path_xy)
    surface_distance_m = path_surface_distance(path_xy, geo)
    time_hours = cost / (baseline_speed * 1000.0)
    average_speed_kmh = surface_distance_m / 1000.0 / time_hours if time_hours > 0.0 else math.inf
    return PathMetrics(
        cost=cost,
        distance_m=distance_m,
        surface_distance_m=surface_distance_m,
        time_hours=time_hours,
        average_speed_kmh=average_speed_kmh,
    )


def combine_metrics(first: PathMetrics, second: PathMetrics) -> PathMetrics:
    cost = first.cost + second.cost
    distance_m = first.distance_m + second.distance_m
    surface_distance_m = first.surface_distance_m + second.surface_distance_m
    time_hours = first.time_hours + second.time_hours
    average_speed_kmh = surface_distance_m / 1000.0 / time_hours if time_hours > 0.0 else math.inf
    return PathMetrics(
        cost=cost,
        distance_m=distance_m,
        surface_distance_m=surface_distance_m,
        time_hours=time_hours,
        average_speed_kmh=average_speed_kmh,
    )


def path_distance(path_xy: npt.NDArray[np.float64]) -> float:
    if len(path_xy) < 2:
        return 0.0
    delta = np.diff(path_xy, axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def path_surface_distance(path_xy: npt.NDArray[np.float64], geo: GeoSurface) -> float:
    plan_distances = segment_plan_distances(path_xy)
    if geo.surface.elevation is None or plan_distances.size == 0:
        return float(plan_distances.sum())
    elevations = sample_elevation(path_xy, geo)
    dz = np.diff(elevations)
    finite = np.isfinite(dz)
    surface = plan_distances.copy()
    surface[finite] = np.hypot(plan_distances[finite], dz[finite])
    return float(surface.sum())


def segment_plan_distances(path_xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if len(path_xy) < 2:
        return np.empty(0, dtype=np.float64)
    delta = np.diff(path_xy, axis=0)
    return np.hypot(delta[:, 0], delta[:, 1])


def sample_elevation(path_xy: npt.NDArray[np.float64], geo: GeoSurface) -> npt.NDArray[np.float64]:
    if geo.surface.elevation is None:
        return np.full(len(path_xy), np.nan, dtype=np.float64)
    rows, cols = rasterio.transform.rowcol(
        geo.grid.transform,
        path_xy[:, 0].tolist(),
        path_xy[:, 1].tolist(),
        op=math.floor,
    )
    height, width = geo.surface.elevation.shape
    out = np.full(len(path_xy), np.nan, dtype=np.float64)
    for point_index, (row, col) in enumerate(zip(rows, cols)):
        if 0 <= row < height and 0 <= col < width:
            out[point_index] = float(geo.surface.elevation[int(row), int(col)])
    return out


def plot_comparison(
    path: Path,
    *,
    land_use_path: Path,
    elevation_path: Path,
    waypoints: list[tuple[float, float]],
    barriers: list[Polygon],
    results: list[ComparisonRoute],
    max_plot_pixels: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; run with `uv run --group plot "
            "python examples/mega_geotiff_comparison.py`"
        ) from exc

    plot_bounds = route_plot_bounds(waypoints, barriers, results, margin=900.0)
    land_use, elevation, extent = read_plot_rasters(
        land_use_path,
        elevation_path,
        bounds=plot_bounds,
        max_plot_pixels=max_plot_pixels,
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7), constrained_layout=True, sharex=True, sharey=True
    )
    terrain_ax, land_ax = axes
    ls = LightSource(azdeg=315, altdeg=45)
    terrain_ax.imshow(
        ls.shade(elevation, cmap=plt.cm.terrain, blend_mode="overlay", vert_exag=0.55),
        extent=extent,
        origin="upper",
        interpolation="bilinear",
    )
    terrain_ax.set_title("Terrain and Solver Routes")

    land_cmap = ListedColormap(
        ["#d9ead3", "#93c47d", "#c9b458", "#b6a05b", "#3f6f3a", "#8e8e86", "#5d9ca6"]
    )
    land_ax.imshow(
        land_use,
        cmap=land_cmap,
        norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], land_cmap.N),
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    land_ax.set_title("Land Use and Burned Barriers")

    colors = {
        "ordered_upwind": "#d95f02",
        "raster_dijkstra": "#1f78b4",
        "whitebox_cost_distance": "#ffcc00",
    }
    labels = {
        "ordered_upwind": "Ordered Upwind",
        "raster_dijkstra": "Raster Dijkstra",
        "whitebox_cost_distance": "Whitebox CostDistance",
    }
    for ax in axes:
        draw_barriers(ax, barriers)
        draw_waypoints(ax, waypoints)
        for result in results:
            draw_result(ax, result, colors, labels)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    route_handles = [
        Line2D([0], [0], color=colors["ordered_upwind"], lw=2.6, label=labels["ordered_upwind"]),
        Line2D(
            [0],
            [0],
            color=colors["raster_dijkstra"],
            lw=2.1,
            ls="--",
            label=labels["raster_dijkstra"],
        ),
        Line2D(
            [0],
            [0],
            color=colors["whitebox_cost_distance"],
            lw=0,
            marker=".",
            markersize=10,
            label=labels["whitebox_cost_distance"],
        ),
        Line2D([0], [0], marker="o", color="black", lw=0, label="Waypoint"),
        Patch(facecolor="none", edgecolor="#111111", hatch="////", label="Barrier"),
    ]
    land_handles = [
        Patch(facecolor=land_cmap(index - 1), edgecolor="none", label=label)
        for index, label in LAND_USE_LABELS.items()
    ]
    terrain_ax.legend(handles=route_handles, loc="upper left")
    land_ax.legend(handles=land_handles + route_handles[:3], loc="upper left", ncols=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_result(
    ax: Any,
    result: ComparisonRoute,
    colors: dict[str, str],
    labels: dict[str, str],
) -> None:
    if result.status != "ok":
        return
    color = colors[result.solver]
    label = labels[result.solver]
    if result.path_xy is not None:
        linestyle = "--" if result.solver == "raster_dijkstra" else "-"
        ax.plot(result.path_xy[:, 0], result.path_xy[:, 1], color=color, lw=2.3, ls=linestyle)
    for leg in result.legs:
        if leg.mask_xy is not None and len(leg.mask_xy) > 0:
            ax.scatter(leg.mask_xy[:, 0], leg.mask_xy[:, 1], s=2.5, color=color, alpha=0.85)
    if result.path_xy is not None and len(result.path_xy) > 0:
        ax.plot([], [], color=color, label=label)


def draw_waypoints(ax: Any, waypoints: list[tuple[float, float]]) -> None:
    ax.scatter(
        [point[0] for point in waypoints],
        [point[1] for point in waypoints],
        s=30,
        c="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
    )


def draw_barriers(ax: Any, barriers: list[Polygon]) -> None:
    for barrier in barriers:
        xs, ys = barrier.exterior.xy
        ax.fill(xs, ys, facecolor="none", edgecolor="#111111", hatch="////", linewidth=1.0)


def route_plot_bounds(
    waypoints: list[tuple[float, float]],
    barriers: list[Polygon],
    results: list[ComparisonRoute],
    *,
    margin: float,
) -> Bounds:
    xs = [point[0] for point in waypoints]
    ys = [point[1] for point in waypoints]
    for barrier in barriers:
        bx, by = barrier.exterior.xy
        xs.extend(float(value) for value in bx)
        ys.extend(float(value) for value in by)
    for result in results:
        if result.path_xy is not None and len(result.path_xy) > 0:
            xs.extend(result.path_xy[:, 0].tolist())
            ys.extend(result.path_xy[:, 1].tolist())
        for leg in result.legs:
            if leg.mask_xy is not None and len(leg.mask_xy) > 0:
                xs.extend(leg.mask_xy[:, 0].tolist())
                ys.extend(leg.mask_xy[:, 1].tolist())
    return min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin


def read_plot_rasters(
    land_use_path: Path,
    elevation_path: Path,
    *,
    bounds: Bounds,
    max_plot_pixels: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], Bounds]:
    with rasterio.open(land_use_path) as land_dataset:
        window = clamped_window(land_dataset, bounds)
        out_shape = plot_out_shape(window, max_plot_pixels)
        land_use = land_dataset.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.nearest,
        )
        extent_bounds = window_bounds(window, land_dataset.transform)
    with rasterio.open(elevation_path) as elevation_dataset:
        elevation = elevation_dataset.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.bilinear,
        )
    left, bottom, right, top = extent_bounds
    return land_use, elevation, (left, right, bottom, top)


def clamped_window(dataset: Any, bounds: Bounds) -> Window:
    left, bottom, right, top = bounds
    raster_left, raster_bottom, raster_right, raster_top = dataset.bounds
    left = max(left, raster_left)
    right = min(right, raster_right)
    bottom = max(bottom, raster_bottom)
    top = min(top, raster_top)
    window = from_bounds(left, bottom, right, top, transform=dataset.transform)
    col_off = max(0, int(math.floor(window.col_off)))
    row_off = max(0, int(math.floor(window.row_off)))
    col_stop = min(dataset.width, int(math.ceil(window.col_off + window.width)))
    row_stop = min(dataset.height, int(math.ceil(window.row_off + window.height)))
    return Window(col_off, row_off, col_stop - col_off, row_stop - row_off)


def plot_out_shape(window: Window, max_plot_pixels: int) -> tuple[int, int]:
    cells = max(1.0, float(window.width * window.height))
    stride = max(1, int(math.ceil(math.sqrt(cells / max_plot_pixels))))
    return (
        max(1, int(math.ceil(window.height / stride))),
        max(1, int(math.ceil(window.width / stride))),
    )


def write_geojson(
    path: Path,
    *,
    waypoints: list[tuple[float, float]],
    barriers: list[Polygon],
    results: list[ComparisonRoute],
) -> None:
    features: list[dict[str, Any]] = []
    features.append(
        {
            "type": "Feature",
            "properties": {"kind": "waypoints", "crs": CRS},
            "geometry": {"type": "LineString", "coordinates": waypoints},
        }
    )
    for index, barrier in enumerate(barriers):
        features.append(
            {
                "type": "Feature",
                "properties": {"kind": "barrier", "index": index},
                "geometry": barrier.__geo_interface__,
            }
        )
    for result in results:
        if result.path_xy is not None:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "kind": "route",
                        "solver": result.solver,
                        "cost": finite_or_none(result.destination_cost),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": result.path_xy.tolist(),
                    },
                }
            )
        for leg in result.legs:
            if leg.mask_xy is not None and len(leg.mask_xy) > 0:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "kind": "route_mask",
                            "solver": result.solver,
                            "index": leg.index,
                            "cost": finite_or_none(leg.cost),
                        },
                        "geometry": MultiPoint(leg.mask_xy).__geo_interface__,
                    }
                )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def write_metadata(
    path: Path,
    *,
    land_use_path: Path,
    elevation_path: Path,
    waypoints: list[tuple[float, float]],
    barriers: list[Polygon],
    results: list[ComparisonRoute],
    args: argparse.Namespace,
    plot_path: Path,
    geojson_path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "land_use_path": str(land_use_path),
                "elevation_path": str(elevation_path),
                "shape": [args.size, args.size],
                "cell_size_m": args.cell_size,
                "crop_buffer_m": args.crop_buffer,
                "vertical_factor": VERTICAL_FACTOR,
                "crs": CRS,
                "waypoints": waypoints,
                "barrier_count": len(barriers),
                "plot_path": str(plot_path),
                "geojson_path": str(geojson_path),
                "results": [result_to_json(result) for result in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def result_to_json(result: ComparisonRoute) -> dict[str, Any]:
    return {
        "solver": result.solver,
        "status": result.status,
        "error": result.error,
        "elapsed_sec": finite_or_none(result.elapsed_sec),
        "destination_cost": finite_or_none(result.destination_cost),
        "vertices": len(result.path_xy) if result.path_xy is not None else None,
        "legs": [
            {
                "index": leg.index,
                "cost": finite_or_none(leg.cost),
                "elapsed_sec": finite_or_none(leg.elapsed_sec),
                "metrics": metrics_to_json(leg.metrics),
                "vertices": len(leg.path_xy) if leg.path_xy is not None else None,
                "mask_points": len(leg.mask_xy) if leg.mask_xy is not None else None,
            }
            for leg in result.legs
        ],
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


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def print_summary(
    results: list[ComparisonRoute],
    *,
    plot_path: Path,
    geojson_path: Path,
    metadata_path: Path,
) -> None:
    for result in results:
        if result.status != "ok":
            print(f"{result.solver}: failed ({result.error})")
            continue
        print(
            f"{result.solver}: cost={result.destination_cost:.3f}, "
            f"elapsed={result.elapsed_sec:.2f}s, legs={len(result.legs)}"
        )
    print("\ndistance-rs surface evaluation")
    for result in results:
        if result.status != "ok":
            continue
        ref = result.reference_metrics
        if ref is None:
            print(f"{result.solver}: n/a")
            continue
        print(
            f"{result.solver}: "
            f"{ref.distance_m:.1f} m, "
            f"{ref.time_hours * 60.0:.2f} min, "
            f"cost={ref.cost:.3f}"
        )
    print(f"plot: {plot_path}")
    print(f"geojson: {geojson_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    raise SystemExit(main())
