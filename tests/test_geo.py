from __future__ import annotations

import math
from pathlib import Path

import fiona
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_coords
from shapely.geometry import LineString, Polygon

from distance_rs import (
    CostRaster,
    GeoBarriers,
    GeoPoints,
    GridSpec,
    SolverOptions,
    load_points,
    load_surface,
    route_path,
)


def _write_geotiff(
    path: Path,
    data: np.ndarray,
    *,
    transform: rasterio.Affine,
    crs: str,
    nodata: float | None = None,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(data, 1)


def test_load_surface_aligns_geotiffs_and_cost_classes(tmp_path: Path) -> None:
    transform = from_origin(0.0, 40.0, 10.0, 10.0)
    land_use_path = tmp_path / "land_use.tif"
    elevation_path = tmp_path / "elevation.tif"

    land_use = np.array(
        [
            [1, 1, 2, 99],
            [1, 2, 2, 1],
            [1, 99, 3, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    elevation = np.array(
        [
            [10.0, 11.0],
            [12.0, 13.0],
        ],
        dtype=np.float32,
    )
    _write_geotiff(land_use_path, land_use, transform=transform, crs="EPSG:3857")
    _write_geotiff(
        elevation_path,
        elevation,
        transform=from_origin(0.0, 40.0, 20.0, 20.0),
        crs="EPSG:3857",
    )

    geo = load_surface(
        CostRaster(
            land_use_path,
            values={1: 1.0, 2: 2.0, 3: 3.0},
            blocked_values={99},
        ),
        elevation=elevation_path,
        grid=GridSpec(resolution=10.0),
    )

    assert geo.land_use.shape == (4, 4)
    assert geo.surface.elevation is not None
    assert geo.surface.elevation.shape == (4, 4)
    assert geo.surface.cost.shape == (4, 4)
    assert geo.grid.cell_size == (10.0, 10.0)
    assert geo.grid.xy_to_cell(5.0, 35.0) == (0, 0)
    assert geo.grid.xy_to_cell(35.0, 5.0) == (3, 3)
    assert geo.surface.cost[0, 2] == 2.0
    assert geo.surface.cost[2, 2] == 3.0
    assert geo.surface.barriers is not None
    assert geo.surface.barriers[0, 3]
    assert geo.surface.barriers[2, 1]

    line_map_xy = geo.grid.raster_line_to_xy(np.array([[0.0, 0.0], [30.0, 30.0]]))
    assert np.allclose(line_map_xy, np.array([[5.0, 35.0], [35.0, 5.0]]))


def test_load_points_reprojects_gpkg_waypoints(tmp_path: Path) -> None:
    route_path = tmp_path / "route.gpkg"
    xs, ys = transform_coords("EPSG:3857", "EPSG:4326", [5.0, 35.0], [35.0, 5.0])
    schema = {"geometry": "LineString", "properties": {}}
    with fiona.open(
        route_path,
        "w",
        driver="GPKG",
        crs="EPSG:4326",
        schema=schema,
        layer="route",
    ) as collection:
        collection.write(
            {
                "geometry": {"type": "LineString", "coordinates": list(zip(xs, ys))},
                "properties": {},
            }
        )

    points = load_points(route_path, target_crs="EPSG:3857")

    assert math.isclose(points[0][0], 5.0, abs_tol=1.0e-8)
    assert math.isclose(points[1][1], 5.0, abs_tol=1.0e-8)


def test_route_path_crops_to_margin_and_forwards_solver_options(tmp_path: Path) -> None:
    land_use_path = tmp_path / "large_land_use.tif"
    transform = from_origin(0.0, 1000.0, 10.0, 10.0)
    land_use = np.ones((100, 100), dtype=np.float32)
    _write_geotiff(land_use_path, land_use, transform=transform, crs="EPSG:3857")

    route = route_path(
        land_use_path,
        GeoPoints(LineString([(105.0, 895.0), (305.0, 795.0)]), crs="EPSG:3857"),
        grid=GridSpec(margin=25.0),
        solver=SolverOptions(stencil_radius=40.0),
    )

    leg = route.legs[0]
    assert leg.grid.shape == (15, 25)
    assert leg.grid.bounds == (80.0, 770.0, 330.0, 920.0)
    assert leg.grid.cell_to_xy(0, 0) == (85.0, 915.0)
    assert leg.source_cell == (2, 2)
    assert leg.destination_cell == (12, 22)
    assert route.path_xy.shape[1] == 2


def test_load_surface_rejects_invalid_grid_options(tmp_path: Path) -> None:
    land_use_path = tmp_path / "land_use.tif"
    _write_geotiff(
        land_use_path,
        np.ones((4, 4), dtype=np.float32),
        transform=from_origin(0.0, 40.0, 10.0, 10.0),
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="resolution"):
        load_surface(land_use_path, grid=GridSpec(resolution=0.0))

    with pytest.raises(ValueError, match="margin"):
        route_path(
            land_use_path,
            GeoPoints([(5.0, 35.0), (35.0, 5.0)], crs="EPSG:3857"),
            grid=GridSpec(margin=math.nan),
        )


def test_route_path_requires_crs_for_plain_coordinate_waypoints(tmp_path: Path) -> None:
    land_use_path = tmp_path / "land_use.tif"
    _write_geotiff(
        land_use_path,
        np.ones((4, 4), dtype=np.float32),
        transform=from_origin(0.0, 40.0, 10.0, 10.0),
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="GeoPoints"):
        route_path(land_use_path, [(5.0, 35.0), (35.0, 5.0)])


def test_load_surface_rasterizes_polygon_and_linestring_barriers(tmp_path: Path) -> None:
    land_use_path = tmp_path / "land_use.tif"
    _write_geotiff(
        land_use_path,
        np.ones((5, 5), dtype=np.float32),
        transform=from_origin(0.0, 50.0, 10.0, 10.0),
        crs="EPSG:3857",
    )

    geo = load_surface(
        land_use_path,
        barriers=[
            GeoBarriers(
                Polygon([(21.0, 1.0), (29.0, 1.0), (29.0, 49.0), (21.0, 49.0)]),
                crs="EPSG:3857",
                all_touched=False,
            ),
            GeoBarriers(LineString([(0.0, 25.0), (50.0, 25.0)]), crs="EPSG:3857"),
        ],
    )

    assert geo.surface.barriers is not None
    assert geo.surface.barriers[:, 2].all()
    assert geo.surface.barriers[2, :].all()


def test_route_path_stitches_waypoint_legs_and_metrics(tmp_path: Path) -> None:
    cost_path = tmp_path / "cost.tif"
    _write_geotiff(
        cost_path,
        np.ones((30, 30), dtype=np.float32),
        transform=from_origin(0.0, 300.0, 10.0, 10.0),
        crs="EPSG:3857",
    )

    route = route_path(
        cost_path,
        GeoPoints([(5.0, 295.0), (145.0, 155.0), (245.0, 55.0)], crs="EPSG:3857"),
        grid=GridSpec(margin=40.0),
        solver=SolverOptions(stencil_radius=40.0),
        baseline_speed=5.0,
    )

    assert len(route.legs) == 2
    assert route.path_xy.shape[1] == 2
    assert np.allclose(route.path_xy[0], [5.0, 295.0])
    assert np.allclose(route.path_xy[-1], [245.0, 55.0])
    assert route.metrics is not None
    assert route.legs[0].metrics is not None
    assert route.metrics.cost == pytest.approx(sum(leg.cost for leg in route.legs))
    assert route.metrics.time_hours == pytest.approx(route.metrics.cost / 5000.0)
    assert route.metrics.average_speed_kmh == pytest.approx(
        route.metrics.surface_distance_m / 1000.0 / route.metrics.time_hours
    )
