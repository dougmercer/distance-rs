"""Generate large synthetic GeoTIFFs and route across a cropped corridor.

The default rasters are 8000 x 8000 pixels at 1.5 meter resolution. The route
only spans a few hundred meters, so `route_path` should read/reproject
only a small corridor around each leg when `crop_buffer` is set.
"""

from __future__ import annotations

import argparse
import json
import math
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
    OptimalPathResult,
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
from rasterio.transform import from_origin, rowcol
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import Polygon

CRS = "EPSG:32618"
WEST = 500_000.0
NORTH = 4_400_000.0
DEFAULT_SIZE = 8000
DEFAULT_CELL_SIZE = 1.5
BLOCK_SIZE = 512
DATASET_VERSION = "complex_v3"
LAND_USE_COSTS = {
    1: 0.72,  # maintained track
    2: 1.0,  # open meadow
    3: 1.35,  # brush
    4: 1.75,  # rough grass
    5: 2.35,  # forest
    6: 2.9,  # talus / rock
    7: 3.6,  # wetland
}
VERTICAL_FACTOR = {
    "type": "symmetric_linear",
    "low_cut_angle": -35.0,
    "high_cut_angle": 35.0,
    "slope": 0.025,
}
LAND_USE_LABELS = {
    1: "Track",
    2: "Meadow",
    3: "Brush",
    4: "Rough Grass",
    5: "Forest",
    6: "Talus",
    7: "Wetland",
}
Bounds = tuple[float, float, float, float]


class BaselineLeg:
    def __init__(
        self,
        *,
        index: int,
        solver: str,
        path_xy: npt.NDArray[np.float64],
        cost: float,
        metrics: PathMetrics,
        reference_metrics: PathMetrics | None = None,
        mask_xy: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.index = index
        self.solver = solver
        self.path_xy = path_xy
        self.cost = cost
        self.metrics = metrics
        self.reference_metrics = reference_metrics
        self.mask_xy = mask_xy


class BaselineRoute:
    def __init__(
        self,
        *,
        solver: str,
        path_xy: npt.NDArray[np.float64],
        legs: tuple[BaselineLeg, ...],
        metrics: PathMetrics,
        reference_metrics: PathMetrics | None = None,
    ) -> None:
        self.solver = solver
        self.path_xy = path_xy
        self.legs = legs
        self.metrics = metrics
        self.reference_metrics = reference_metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = args.data_dir
    output_dir = args.output_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    land_use_path = data_dir / f"synthetic_{DATASET_VERSION}_land_use_{args.size}.tif"
    elevation_path = data_dir / f"synthetic_{DATASET_VERSION}_elevation_{args.size}.tif"
    route_geojson_path = output_dir / "large_geotiff_route.geojson"
    summary_path = output_dir / "large_geotiff_route_summary.json"
    plot_path = output_dir / "large_geotiff_route_map.png"

    generate_large_geotiffs(
        land_use_path,
        elevation_path,
        size=args.size,
        cell_size=args.cell_size,
        overwrite=args.overwrite,
    )

    waypoints = route_waypoints(args.size, args.cell_size)
    barriers = route_barriers(args.size, args.cell_size)
    route = route_path(
        CostRaster(land_use_path, values=LAND_USE_COSTS),
        GeoPoints(waypoints, crs=CRS),
        barriers=[GeoBarriers(barrier, crs=CRS) for barrier in barriers],
        elevation=elevation_path,
        grid=GridSpec(crs=CRS),
        margin=args.crop_buffer,
        vertical_factor=VERTICAL_FACTOR,
        baseline_speed=args.baseline_speed,
        parallel=True,
    )
    dijkstra_route = run_raster_dijkstra_route(
        land_use_path,
        elevation_path=elevation_path,
        waypoints=waypoints,
        barriers=barriers,
        crop_buffer=args.crop_buffer,
        baseline_speed=args.baseline_speed,
    )
    whitebox_route = run_whitebox_route(
        land_use_path,
        elevation_path=elevation_path,
        waypoints=waypoints,
        barriers=barriers,
        crop_buffer=args.crop_buffer,
        baseline_speed=args.baseline_speed,
    )

    write_route_geojson(
        route_geojson_path,
        route,
        dijkstra_route=dijkstra_route,
        whitebox_route=whitebox_route,
        barriers=barriers,
    )
    plot_route_map(
        plot_path,
        land_use_path=land_use_path,
        elevation_path=elevation_path,
        ordered_route=route,
        dijkstra_route=dijkstra_route,
        whitebox_route=whitebox_route,
        barriers=barriers,
        max_plot_pixels=args.max_plot_pixels,
    )
    write_summary(
        summary_path,
        route,
        dijkstra_route=dijkstra_route,
        whitebox_route=whitebox_route,
        land_use_path=land_use_path,
        elevation_path=elevation_path,
        size=args.size,
        cell_size=args.cell_size,
        crop_buffer=args.crop_buffer,
        baseline_speed=args.baseline_speed,
        plot_path=plot_path,
    )
    print_summary(
        route,
        dijkstra_route=dijkstra_route,
        whitebox_route=whitebox_route,
        route_geojson_path=route_geojson_path,
        summary_path=summary_path,
        plot_path=plot_path,
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/large-geotiff-route"),
        help="Directory for generated GeoTIFF inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/large-geotiff-route"),
        help="Directory for route GeoJSON and summary outputs.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help="Raster width and height in pixels.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=DEFAULT_CELL_SIZE,
        help="Square cell size in meters. Defaults to sub-2-meter resolution.",
    )
    parser.add_argument(
        "--crop-buffer",
        type=float,
        default=180.0,
        help="GeoTIFF corridor buffer around each route leg in meters.",
    )
    parser.add_argument(
        "--baseline-speed",
        type=float,
        default=5.0,
        help="Baseline travel speed in km/hr for cost-to-time metrics.",
    )
    parser.add_argument(
        "--max-plot-pixels",
        type=int,
        default=1_500_000,
        help="Maximum pixels to read for each plotted raster panel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate rasters even if they already exist.",
    )
    args = parser.parse_args(argv)
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.cell_size <= 0.0 or args.cell_size >= 2.0:
        parser.error("--cell-size must be positive and less than 2 meters")
    if args.crop_buffer <= 0.0:
        parser.error("--crop-buffer must be positive")
    if args.baseline_speed <= 0.0:
        parser.error("--baseline-speed must be positive")
    if args.max_plot_pixels < 10_000:
        parser.error("--max-plot-pixels must be at least 10000")
    return args


def generate_large_geotiffs(
    land_use_path: Path,
    elevation_path: Path,
    *,
    size: int,
    cell_size: float,
    overwrite: bool,
) -> None:
    if not overwrite and land_use_path.exists() and elevation_path.exists():
        print(f"using existing rasters in {land_use_path.parent}")
        return

    print(
        f"generating {size}x{size} GeoTIFFs at {cell_size:g} m resolution in {land_use_path.parent}"
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

    land_profile = {**profile, "dtype": "uint8", "nodata": 0}
    elevation_profile = {**profile, "dtype": "float32", "nodata": -9999.0}

    with (
        rasterio.open(land_use_path, "w", **land_profile) as land_dataset,
        rasterio.open(elevation_path, "w", **elevation_profile) as elevation_dataset,
    ):
        x = np.arange(size, dtype=np.float32)[None, :]
        for row_start in range(0, size, BLOCK_SIZE):
            block_height = min(BLOCK_SIZE, size - row_start)
            y = np.arange(row_start, row_start + block_height, dtype=np.float32)[:, None]
            window = Window(0, row_start, size, block_height)
            land_dataset.write(synthetic_land_use(x, y, size, cell_size), 1, window=window)
            elevation_dataset.write(synthetic_elevation(x, y, size, cell_size), 1, window=window)


def synthetic_land_use(
    x: np.ndarray[Any, np.dtype[np.float32]],
    y: np.ndarray[Any, np.dtype[np.float32]],
    size: int,
    cell_size: float,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    nx = x / float(size)
    ny = y / float(size)
    local_east, local_north = local_route_xy(x, y, size=size, cell_size=cell_size)
    creek_y = sinuous_creek_y(nx)
    upper_creek_y = 0.62 + 0.045 * np.sin(nx * 16.0 + 1.5)
    ridge_y = 0.23 + 0.05 * np.sin(nx * 19.0 + 0.4)
    track_y = maintained_track_y(nx)
    local_creek_north = local_creek_northing(local_east)
    local_track_north = local_track_northing(local_east)
    local_ridge_north = local_ridge_northing(local_east)

    fine_texture = (
        0.65 * np.sin(nx * 83.0 + ny * 11.0)
        + 0.52 * np.cos(nx * 29.0 - ny * 73.0)
        + 0.31 * np.sin((nx + 1.7 * ny) * 137.0)
        + 0.20 * np.cos((2.3 * nx - ny) * 211.0)
    )
    coarse_texture = (
        np.sin(nx * 17.0 + 0.7) + 0.75 * np.cos(ny * 21.0 - 1.3) + 0.45 * np.sin((nx - ny) * 33.0)
    )
    forest_blob = (
        1.05 * gaussian(nx, ny, 0.36, 0.21, 0.055, 0.035)
        + 1.00 * gaussian(nx, ny, 0.44, 0.35, 0.070, 0.045)
        + 0.85 * gaussian(nx, ny, 0.57, 0.29, 0.060, 0.050)
        + 0.85 * gaussian(nx, ny, 0.69, 0.48, 0.090, 0.060)
        + 0.70 * gaussian(nx, ny, 0.28, 0.54, 0.080, 0.050)
    )
    talus_score = (
        gaussian(nx, ny, 0.40, 0.24, 0.030, 0.018)
        + 0.95 * gaussian(nx, ny, 0.53, 0.40, 0.040, 0.020)
        + 0.75 * gaussian(nx, ny, 0.63, 0.34, 0.045, 0.022)
        + 0.40 * np.exp(-((ny - ridge_y) ** 2) / (2.0 * 0.008**2))
    )
    riparian = np.exp(-((ny - creek_y) ** 2) / (2.0 * 0.020**2))
    upper_riparian = np.exp(-((ny - upper_creek_y) ** 2) / (2.0 * 0.018**2))
    wetland = (
        np.exp(-((ny - creek_y) ** 2) / (2.0 * 0.008**2))
        + 0.75 * gaussian(nx, ny, 0.31, 0.30, 0.035, 0.018)
        + 0.55 * gaussian(nx, ny, 0.49, 0.27, 0.045, 0.016)
        + 0.45 * np.exp(-((ny - upper_creek_y) ** 2) / (2.0 * 0.007**2))
    )
    local_forest = (
        gaussian_m(local_east, local_north, 230.0, -55.0, 70.0, 42.0)
        + 0.95 * gaussian_m(local_east, local_north, 470.0, 145.0, 90.0, 55.0)
        + 0.80 * gaussian_m(local_east, local_north, 690.0, -15.0, 95.0, 45.0)
    )
    local_talus = np.exp(-((local_north - local_ridge_north) ** 2) / (2.0 * 16.0**2)) * np.exp(
        -((local_east - 420.0) ** 2) / (2.0 * 260.0**2)
    ) + 0.70 * gaussian_m(local_east, local_north, 585.0, 125.0, 55.0, 24.0)
    local_wetland = (
        np.exp(-((local_north - local_creek_north) ** 2) / (2.0 * 10.0**2))
        + 0.85 * gaussian_m(local_east, local_north, 355.0, -70.0, 65.0, 24.0)
        + 0.55 * gaussian_m(local_east, local_north, 610.0, 30.0, 75.0, 28.0)
    )
    local_brush = (
        np.exp(-((local_north - local_creek_north) ** 2) / (2.0 * 35.0**2))
        + 0.65 * local_forest
        + 0.22 * fine_texture
    )
    local_track = np.exp(-((local_north - local_track_north) ** 2) / (2.0 * 5.0**2))
    rough_grass = coarse_texture + 0.55 * fine_texture
    brush = 0.55 * coarse_texture + 0.90 * riparian + 0.55 * upper_riparian + 0.25 * fine_texture
    track = np.exp(-((ny - track_y) ** 2) / (2.0 * 0.0032**2))

    land_use = np.full((y.shape[0], x.shape[1]), 2, dtype=np.uint8)
    land_use[rough_grass > 0.75] = 4
    land_use[(brush > 1.15) | (local_brush > 1.10)] = 3
    land_use[forest_blob + local_forest + 0.22 * fine_texture > 0.85] = 5
    land_use[(talus_score > 0.72) | (local_talus > 0.72)] = 6
    land_use[(wetland > 0.88) | (local_wetland > 0.88)] = 7
    land_use[(track > 0.54) | (local_track > 0.58)] = 1
    return land_use


def synthetic_elevation(
    x: np.ndarray[Any, np.dtype[np.float32]],
    y: np.ndarray[Any, np.dtype[np.float32]],
    size: int,
    cell_size: float,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    nx = x / float(size)
    ny = y / float(size)
    local_east, local_north = local_route_xy(x, y, size=size, cell_size=cell_size)
    creek_y = sinuous_creek_y(nx)
    ridge_y = 0.23 + 0.05 * np.sin(nx * 19.0 + 0.4)
    local_creek_north = local_creek_northing(local_east)
    local_ridge_north = local_ridge_northing(local_east)
    ridge = (
        38.0 * np.exp(-((ny - ridge_y) ** 2) / (2.0 * 0.018**2))
        + 54.0 * gaussian(nx, ny, 0.39, 0.22, 0.052, 0.026)
        + 47.0 * gaussian(nx, ny, 0.54, 0.39, 0.065, 0.034)
        + 39.0 * gaussian(nx, ny, 0.68, 0.47, 0.090, 0.052)
        + 32.0
        * np.exp(-((local_north - local_ridge_north) ** 2) / (2.0 * 24.0**2))
        * np.exp(-((local_east - 430.0) ** 2) / (2.0 * 320.0**2))
        + 22.0 * gaussian_m(local_east, local_north, 585.0, 130.0, 60.0, 32.0)
    )
    valley = (
        34.0 * np.exp(-((ny - creek_y) ** 2) / (2.0 * 0.018**2))
        + 22.0 * gaussian(nx, ny, 0.30, 0.31, 0.050, 0.020)
        + 18.0 * gaussian(nx, ny, 0.50, 0.27, 0.060, 0.018)
        + 16.0 * np.exp(-((ny - (0.62 + 0.045 * np.sin(nx * 16.0 + 1.5))) ** 2) / (2.0 * 0.015**2))
        + 28.0
        * np.exp(-((local_north - local_creek_north) ** 2) / (2.0 * 18.0**2))
        * np.exp(-((local_east - 430.0) ** 2) / (2.0 * 410.0**2))
    )
    rolling = (
        10.0 * np.sin(nx * 18.0)
        + 8.0 * np.cos(ny * 16.0)
        + 5.5 * np.sin((nx + ny) * 41.0)
        + 2.2 * np.cos((2.0 * nx - 1.4 * ny) * 37.0)
        + 0.7 * np.sin((3.0 * nx + 2.0 * ny) * 59.0)
    )
    trend = 128.0 + 44.0 * (1.0 - ny) + 16.0 * nx
    return (trend + ridge - valley + rolling).astype(np.float32)


def sinuous_creek_y(nx: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 0.305 + 0.030 * np.sin(nx * 22.0 + 0.6) + 0.011 * np.sin(nx * 67.0 - 0.4)


def maintained_track_y(nx: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 0.275 + 0.045 * np.sin(nx * 13.0 - 1.1) + 0.018 * np.sin(nx * 37.0 + 0.5)


def local_route_xy(
    x: np.ndarray[Any, np.dtype[np.float32]],
    y: np.ndarray[Any, np.dtype[np.float32]],
    *,
    size: int,
    cell_size: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    origin = 0.3 * float(size)
    east = (x - origin) * cell_size
    north = (origin - y) * cell_size
    return east, north


def local_creek_northing(local_east: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return -60.0 + 34.0 * np.sin((local_east - 80.0) / 105.0) + 13.0 * np.sin(local_east / 39.0)


def local_track_northing(local_east: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 42.0 + 82.0 * np.sin((local_east - 150.0) / 235.0) + 20.0 * np.sin(local_east / 74.0)


def local_ridge_northing(local_east: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return 150.0 + 38.0 * np.sin((local_east + 60.0) / 120.0)


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


def gaussian_m(
    east: npt.NDArray[np.float32],
    north: npt.NDArray[np.float32],
    center_east: float,
    center_north: float,
    radius_east: float,
    radius_north: float,
) -> npt.NDArray[np.float32]:
    return np.exp(
        -(
            (east - center_east) ** 2 / (2.0 * radius_east**2)
            + (north - center_north) ** 2 / (2.0 * radius_north**2)
        )
    )


def route_waypoints(size: int, cell_size: float) -> list[tuple[float, float]]:
    offset = 0.3 * size * cell_size
    return [
        (WEST + offset + 0.0, NORTH - offset - 0.0),
        (WEST + offset + 300.0, NORTH - offset + 300.0),
        (WEST + offset + 700.0, NORTH - offset + 100.0),
    ]


def route_barriers(size: int, cell_size: float) -> list[Polygon]:
    offset = 0.3 * size * cell_size
    return [
        local_polygon(
            offset,
            [
                (145.0, -62.0),
                (188.0, -54.0),
                (190.0, 66.0),
                (226.0, 90.0),
                (219.0, 133.0),
                (184.0, 121.0),
                (179.0, 262.0),
                (145.0, 258.0),
                (151.0, 123.0),
                (120.0, 96.0),
                (124.0, 49.0),
                (150.0, 66.0),
            ],
        ),
        local_polygon(
            offset,
            [
                (302.0, -128.0),
                (421.0, -114.0),
                (506.0, -66.0),
                (498.0, -12.0),
                (388.0, -6.0),
                (298.0, -44.0),
            ],
        ),
        local_polygon(
            offset,
            [
                (518.0, 104.0),
                (608.0, 92.0),
                (708.0, 126.0),
                (694.0, 170.0),
                (574.0, 190.0),
                (508.0, 150.0),
            ],
        ),
    ]


def local_polygon(offset: float, points: list[tuple[float, float]]) -> Polygon:
    return Polygon([(WEST + offset + east, NORTH - offset + north) for east, north in points])


def run_raster_dijkstra_route(
    land_use_path: Path,
    *,
    elevation_path: Path,
    waypoints: list[tuple[float, float]],
    barriers: list[Polygon],
    crop_buffer: float,
    baseline_speed: float,
) -> BaselineRoute:
    legs: list[BaselineLeg] = []
    path_parts: list[npt.NDArray[np.float64]] = []
    total_metrics: PathMetrics | None = None
    total_reference_metrics: PathMetrics | None = None

    for index, (start_xy, end_xy) in enumerate(zip(waypoints, waypoints[1:])):
        geo = load_surface(
            CostRaster(land_use_path, values=LAND_USE_COSTS),
            elevation=elevation_path,
            barriers=[GeoBarriers(barrier, crs=CRS) for barrier in barriers],
            grid=GridSpec(crs=CRS, bounds=leg_bounds(start_xy, end_xy, crop_buffer)),
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
            raise ValueError(f"raster Dijkstra could not reach leg {index} destination")

        cell_line = trace_raster_path(result.parent, destination_cell, cell_size=geo.grid.cell_size)
        path_xy = geo.grid.raster_line_to_xy(cell_line)[::-1].copy()
        metrics = path_metrics(
            path_xy, cost=destination_cost, geo=geo, baseline_speed=baseline_speed
        )
        reference_cost = evaluate_path_cost(geo, path_xy, vertical_factor=VERTICAL_FACTOR)
        reference_metrics = path_metrics(
            path_xy, cost=reference_cost, geo=geo, baseline_speed=baseline_speed
        )
        total_metrics = (
            metrics if total_metrics is None else combine_metrics(total_metrics, metrics)
        )
        total_reference_metrics = (
            reference_metrics
            if total_reference_metrics is None
            else combine_metrics(total_reference_metrics, reference_metrics)
        )

        legs.append(
            BaselineLeg(
                index=index,
                solver="raster_dijkstra",
                path_xy=path_xy,
                cost=destination_cost,
                metrics=metrics,
                reference_metrics=reference_metrics,
            )
        )
        path_parts.append(path_xy if not path_parts else path_xy[1:])

    if total_metrics is None:
        raise ValueError("baseline route requires at least one leg")
    return BaselineRoute(
        solver="raster_dijkstra",
        path_xy=np.vstack(path_parts),
        legs=tuple(legs),
        metrics=total_metrics,
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
) -> BaselineRoute:
    legs: list[BaselineLeg] = []
    mask_parts: list[npt.NDArray[np.float64]] = []
    total_cost = 0.0
    total_reference_metrics: PathMetrics | None = None

    for index, (start_xy, end_xy) in enumerate(zip(waypoints, waypoints[1:])):
        geo = load_surface(
            CostRaster(land_use_path, values=LAND_USE_COSTS),
            elevation=elevation_path,
            barriers=[GeoBarriers(barrier, crs=CRS) for barrier in barriers],
            grid=GridSpec(crs=CRS, bounds=leg_bounds(start_xy, end_xy, crop_buffer)),
        )
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
            raise ValueError(f"Whitebox CostDistance could not reach leg {index} destination")

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
        total_cost += destination_cost
        legs.append(
            BaselineLeg(
                index=index,
                solver="whitebox_cost_distance",
                path_xy=mask_xy,
                cost=destination_cost,
                metrics=metrics,
                reference_metrics=reference_metrics,
                mask_xy=mask_xy,
            )
        )
        if len(mask_xy) > 0:
            mask_parts.append(mask_xy)

    if not legs:
        raise ValueError("baseline route requires at least one leg")
    total_metrics = PathMetrics(
        cost=total_cost,
        distance_m=math.nan,
        surface_distance_m=math.nan,
        time_hours=total_cost / (baseline_speed * 1000.0),
        average_speed_kmh=math.nan,
    )
    return BaselineRoute(
        solver="whitebox_cost_distance",
        path_xy=np.vstack(mask_parts) if mask_parts else np.empty((0, 2), dtype=np.float64),
        legs=tuple(legs),
        metrics=total_metrics,
        reference_metrics=total_reference_metrics,
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


def leg_bounds(start_xy: tuple[float, float], end_xy: tuple[float, float], margin: float) -> Bounds:
    return (
        min(start_xy[0], end_xy[0]) - margin,
        min(start_xy[1], end_xy[1]) - margin,
        max(start_xy[0], end_xy[0]) + margin,
        max(start_xy[1], end_xy[1]) + margin,
    )


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
    rows, cols = rowcol(
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


def write_route_geojson(
    path: Path,
    route: OptimalPathResult,
    *,
    dijkstra_route: BaselineRoute,
    whitebox_route: BaselineRoute,
    barriers: list[Polygon],
) -> None:
    features: list[dict[str, Any]] = [
        {
            "type": "Feature",
            "properties": {
                "kind": "route",
                "solver": "ordered_upwind",
                "crs": CRS,
                "cost": route.metrics.cost if route.metrics is not None else None,
                "time_hours": route.metrics.time_hours if route.metrics is not None else None,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": route.path_xy.tolist(),
            },
        }
    ]
    features.append(
        {
            "type": "Feature",
            "properties": {
                "kind": "route",
                "solver": "raster_dijkstra",
                "crs": CRS,
                "cost": dijkstra_route.metrics.cost,
                "time_hours": dijkstra_route.metrics.time_hours,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": dijkstra_route.path_xy.tolist(),
            },
        }
    )
    features.append(
        {
            "type": "Feature",
            "properties": {
                "kind": "route_mask",
                "solver": "whitebox_cost_distance",
                "crs": CRS,
                "cost": whitebox_route.metrics.cost,
                "time_hours": whitebox_route.metrics.time_hours,
                "note": "Whitebox CostPathway returns a raster pathway mask, not an ordered polyline.",
            },
            "geometry": {
                "type": "MultiPoint",
                "coordinates": whitebox_route.path_xy.tolist(),
            },
        }
    )
    for leg in route.legs:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "leg",
                    "solver": "ordered_upwind",
                    "index": leg.index,
                    "cost": leg.cost,
                    "time_hours": leg.metrics.time_hours if leg.metrics is not None else None,
                    "distance_m": leg.metrics.distance_m if leg.metrics is not None else None,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": leg.path_xy.tolist(),
                },
            }
        )
    for leg in dijkstra_route.legs:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "leg",
                    "solver": "raster_dijkstra",
                    "index": leg.index,
                    "cost": leg.cost,
                    "time_hours": leg.metrics.time_hours,
                    "distance_m": leg.metrics.distance_m,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": leg.path_xy.tolist(),
                },
            }
        )
    for leg in whitebox_route.legs:
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "kind": "leg_mask",
                    "solver": leg.solver,
                    "index": leg.index,
                    "cost": leg.cost,
                    "time_hours": leg.metrics.time_hours,
                    "note": "Whitebox CostPathway returns mask cells for this leg.",
                },
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": leg.path_xy.tolist(),
                },
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
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def plot_route_map(
    path: Path,
    *,
    land_use_path: Path,
    elevation_path: Path,
    ordered_route: OptimalPathResult,
    dijkstra_route: BaselineRoute,
    whitebox_route: BaselineRoute,
    barriers: list[Polygon],
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
            "python examples/large_geotiff_route.py`"
        ) from exc

    plot_bounds = route_plot_bounds(
        [ordered_route.path_xy, dijkstra_route.path_xy, whitebox_route.path_xy],
        barriers=barriers,
        margin=160.0,
    )
    land_use, elevation, extent = read_plot_rasters(
        land_use_path,
        elevation_path,
        bounds=plot_bounds,
        max_plot_pixels=max_plot_pixels,
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6.6), constrained_layout=True, sharex=True, sharey=True
    )
    terrain_ax, land_ax = axes

    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(elevation, cmap=plt.cm.terrain, blend_mode="overlay", vert_exag=0.8)
    terrain_ax.imshow(shaded, extent=extent, origin="upper", interpolation="bilinear")
    terrain_ax.set_title("Terrain with Routes")

    land_cmap = ListedColormap(
        [
            "#d9ead3",  # track
            "#93c47d",  # meadow
            "#c9b458",  # brush
            "#b6a05b",  # rough grass
            "#3f6f3a",  # forest
            "#8e8e86",  # talus
            "#5d9ca6",  # wetland
        ]
    )
    land_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], land_cmap.N)
    land_ax.imshow(
        land_use,
        cmap=land_cmap,
        norm=land_norm,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    land_ax.set_title("Land Use with Routes")

    for ax in axes:
        draw_barriers(ax, barriers)
        draw_route(
            ax, ordered_route.path_xy, color="#d95f02", label="Ordered Upwind", linewidth=2.8
        )
        draw_route(
            ax,
            dijkstra_route.path_xy,
            color="#1f78b4",
            label="Raster Dijkstra",
            linewidth=2.1,
            linestyle="--",
        )
        draw_mask_points(ax, whitebox_route.path_xy, color="#ffcc00", label="Whitebox CostDistance")
        draw_waypoints(ax, ordered_route.waypoint_xy)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    route_handles = [
        Line2D([0], [0], color="#d95f02", lw=2.8, label="Ordered Upwind"),
        Line2D([0], [0], color="#1f78b4", lw=2.1, ls="--", label="Raster Dijkstra"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#ffcc00",
            markeredgecolor="#9a7a00",
            markersize=5,
            label="Whitebox CostDistance",
        ),
        Line2D([0], [0], marker="o", color="black", lw=0, label="Waypoint"),
        Patch(facecolor="none", edgecolor="#111111", hatch="////", label="Barrier"),
    ]
    land_handles = [
        Patch(facecolor=land_cmap(index - 1), edgecolor="none", label=label)
        for index, label in LAND_USE_LABELS.items()
    ]
    terrain_ax.legend(handles=route_handles, loc="lower right")
    land_ax.legend(handles=land_handles + route_handles[:3], loc="lower right", ncols=2)

    ordered_metrics = ordered_route.metrics
    if ordered_metrics is not None:
        fig.suptitle(
            "Large GeoTIFF Route: "
            f"Ordered {ordered_metrics.time_hours * 60.0:.1f} min vs "
            f"Dijkstra {dijkstra_route.metrics.time_hours * 60.0:.1f} min vs "
            f"Whitebox {whitebox_route.metrics.time_hours * 60.0:.1f} min"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def route_plot_bounds(
    routes: list[npt.NDArray[np.float64]],
    *,
    barriers: list[Polygon],
    margin: float,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for route in routes:
        xs.extend(route[:, 0].tolist())
        ys.extend(route[:, 1].tolist())
    for barrier in barriers:
        bx, by = barrier.exterior.xy
        xs.extend(float(value) for value in bx)
        ys.extend(float(value) for value in by)
    return (
        min(xs) - margin,
        min(ys) - margin,
        max(xs) + margin,
        max(ys) + margin,
    )


def read_plot_rasters(
    land_use_path: Path,
    elevation_path: Path,
    *,
    bounds: tuple[float, float, float, float],
    max_plot_pixels: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], tuple[float, float, float, float]]:
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


def clamped_window(dataset: Any, bounds: tuple[float, float, float, float]) -> Window:
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


def draw_route(
    ax: Any,
    path_xy: npt.NDArray[np.float64],
    *,
    color: str,
    label: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    ax.plot(
        path_xy[:, 0],
        path_xy[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=4,
    )


def draw_mask_points(
    ax: Any,
    path_xy: npt.NDArray[np.float64],
    *,
    color: str,
    label: str,
) -> None:
    if len(path_xy) == 0:
        return
    ax.scatter(
        path_xy[:, 0],
        path_xy[:, 1],
        s=7,
        c=color,
        edgecolors="none",
        alpha=0.82,
        label=label,
        zorder=5,
    )


def draw_waypoints(ax: Any, waypoints: tuple[tuple[float, float], ...]) -> None:
    xs = [point[0] for point in waypoints]
    ys = [point[1] for point in waypoints]
    ax.scatter(xs, ys, s=28, c="black", edgecolors="white", linewidths=0.8, zorder=5)


def draw_barriers(ax: Any, barriers: list[Polygon]) -> None:
    for barrier in barriers:
        xs, ys = barrier.exterior.xy
        ax.fill(
            xs,
            ys,
            facecolor="none",
            edgecolor="#111111",
            hatch="////",
            linewidth=1.2,
            zorder=3,
        )


def write_summary(
    path: Path,
    route: OptimalPathResult,
    *,
    dijkstra_route: BaselineRoute,
    whitebox_route: BaselineRoute,
    land_use_path: Path,
    elevation_path: Path,
    size: int,
    cell_size: float,
    crop_buffer: float,
    baseline_speed: float,
    plot_path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "land_use_path": str(land_use_path),
                "elevation_path": str(elevation_path),
                "shape": [size, size],
                "cell_size_m": cell_size,
                "crop_buffer_m": crop_buffer,
                "baseline_speed_kmh": baseline_speed,
                "vertical_factor": VERTICAL_FACTOR,
                "barrier_count": len(route_barriers(size, cell_size)),
                "crs": CRS,
                "waypoints": route.waypoint_xy,
                "plot_path": str(plot_path),
                "ordered_upwind": {
                    "total": metrics_to_json(route.metrics),
                    "legs": [
                        {
                            "index": leg.index,
                            "start_xy": leg.start_xy,
                            "end_xy": leg.end_xy,
                            "source_cell": leg.source_cell,
                            "destination_cell": leg.destination_cell,
                            "cost": leg.cost,
                            "metrics": metrics_to_json(leg.metrics),
                        }
                        for leg in route.legs
                    ],
                },
                "raster_dijkstra": {
                    "total": metrics_to_json(dijkstra_route.metrics),
                    "legs": [
                        {
                            "index": leg.index,
                            "cost": leg.cost,
                            "metrics": metrics_to_json(leg.metrics),
                        }
                        for leg in dijkstra_route.legs
                    ],
                },
                "whitebox_cost_distance": {
                    "total": metrics_to_json(whitebox_route.metrics),
                    "legs": [
                        {
                            "index": leg.index,
                            "cost": leg.cost,
                            "path_mask_points": int(len(leg.path_xy)),
                            "metrics": metrics_to_json(leg.metrics),
                        }
                        for leg in whitebox_route.legs
                    ],
                    "note": (
                        "WhiteboxTools is run through distance_rs.baselines.whitebox_cost_distance. "
                        "It uses the cost surface and barriers, but does not support this example's "
                        "elevation or vertical-factor model."
                    ),
                },
                "comparisons": {
                    "raster_dijkstra": comparison_to_json(route.metrics, dijkstra_route.metrics),
                    "whitebox_cost_distance": comparison_to_json(
                        route.metrics, whitebox_route.metrics
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def metrics_to_json(metrics: Any | None) -> dict[str, float] | None:
    if metrics is None:
        return None
    return {
        "cost": finite_or_none(metrics.cost),
        "distance_m": finite_or_none(metrics.distance_m),
        "surface_distance_m": finite_or_none(metrics.surface_distance_m),
        "time_hours": finite_or_none(metrics.time_hours),
        "average_speed_kmh": finite_or_none(metrics.average_speed_kmh),
    }


def comparison_to_json(
    ordered: PathMetrics | None,
    dijkstra: PathMetrics,
) -> dict[str, float | None]:
    if ordered is None:
        return {}
    return {
        "cost_delta_ordered_minus_dijkstra": finite_or_none(ordered.cost - dijkstra.cost),
        "time_delta_hours_ordered_minus_dijkstra": finite_or_none(
            ordered.time_hours - dijkstra.time_hours
        ),
        "distance_delta_m_ordered_minus_dijkstra": finite_or_none(
            ordered.distance_m - dijkstra.distance_m
        ),
        "cost_ratio_ordered_to_dijkstra": finite_or_none(ordered.cost / dijkstra.cost)
        if dijkstra.cost != 0.0
        else None,
    }


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def print_summary(
    route: OptimalPathResult,
    *,
    dijkstra_route: BaselineRoute,
    whitebox_route: BaselineRoute,
    route_geojson_path: Path,
    summary_path: Path,
    plot_path: Path,
) -> None:
    metrics = route.metrics
    print(f"ordered-upwind vertices: {len(route.path_xy)}")
    print(f"raster-dijkstra vertices: {len(dijkstra_route.path_xy)}")
    print(f"whitebox pathway cells: {len(whitebox_route.path_xy)}")
    print(f"legs: {len(route.legs)}")
    if metrics is not None:
        print(
            "ordered upwind: "
            f"{metrics.distance_m:.1f} m, "
            f"{metrics.time_hours * 60.0:.2f} min, "
            f"cost {metrics.cost:.1f}"
        )
        print(
            "raster dijkstra: "
            f"{dijkstra_route.metrics.distance_m:.1f} m, "
            f"{dijkstra_route.metrics.time_hours * 60.0:.2f} min, "
            f"cost {dijkstra_route.metrics.cost:.1f}"
        )
        print(
            "whitebox costdistance: "
            f"{whitebox_route.metrics.time_hours * 60.0:.2f} min, "
            f"cost {whitebox_route.metrics.cost:.1f}"
        )
        print("\ndistance-rs surface evaluation")
        print(
            "ordered upwind: "
            f"{metrics.distance_m:.1f} m, "
            f"{metrics.time_hours * 60.0:.2f} min, "
            f"cost {metrics.cost:.1f}"
        )
        if dijkstra_route.reference_metrics is not None:
            ref = dijkstra_route.reference_metrics
            print(
                "raster dijkstra: "
                f"{ref.distance_m:.1f} m, "
                f"{ref.time_hours * 60.0:.2f} min, "
                f"cost {ref.cost:.1f}"
            )
        if whitebox_route.reference_metrics is not None:
            ref = whitebox_route.reference_metrics
            print(
                "whitebox costdistance: "
                f"{ref.distance_m:.1f} m, "
                f"{ref.time_hours * 60.0:.2f} min, "
                f"cost {ref.cost:.1f}"
            )
        print(
            "delta ordered-dijkstra: "
            f"{metrics.distance_m - dijkstra_route.metrics.distance_m:.1f} m, "
            f"{(metrics.time_hours - dijkstra_route.metrics.time_hours) * 60.0:.2f} min, "
            f"cost {metrics.cost - dijkstra_route.metrics.cost:.1f}"
        )
    print(f"map plot: {plot_path}")
    print(f"route geojson: {route_geojson_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
