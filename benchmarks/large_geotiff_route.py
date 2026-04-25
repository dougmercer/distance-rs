"""Generate large synthetic GeoTIFFs and run the ordered-upwind route solver.

The default rasters are 8000 x 8000 pixels at 1.5 meter resolution. The route
only spans a few hundred meters, so `route_path` should read/reproject
only a small corridor around each leg when `crop_buffer` is set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import Polygon

from distance_rs import (
    CostRaster,
    GeoBarriers,
    GeoPoints,
    GridSpec,
    route_path,
)


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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    land_use_path = data_dir / f"synthetic_{DATASET_VERSION}_land_use_{args.size}.tif"
    elevation_path = data_dir / f"synthetic_{DATASET_VERSION}_elevation_{args.size}.tif"

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
    )

    metrics = route.metrics
    if metrics is None:
        raise RuntimeError("route solver returned no metrics")
    print(
        "ordered upwind: "
        f"legs={len(route.legs)}, "
        f"vertices={len(route.path_xy)}, "
        f"distance_m={metrics.distance_m:.3f}, "
        f"surface_distance_m={metrics.surface_distance_m:.3f}, "
        f"cost={metrics.cost:.6f}"
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
        default=30.0,
        help="GeoTIFF corridor buffer around each route leg in meters.",
    )
    parser.add_argument(
        "--baseline-speed",
        type=float,
        default=5.0,
        help="Baseline travel speed in km/hr for cost-to-time metrics.",
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



if __name__ == "__main__":
    main()
