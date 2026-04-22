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

from distance_rs import compute_optimal_path, distance_accumulation
from distance_rs._geo import geo_distance_accumulation, geo_optimal_path_as_line, prepare_geo_inputs


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


def test_prepare_geo_inputs_aligns_geotiffs_and_linestring_waypoints(tmp_path: Path) -> None:
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

    geo = prepare_geo_inputs(
        land_use_path,
        elevation_path=elevation_path,
        waypoints=LineString([(5.0, 35.0), (35.0, 5.0)]),
        land_use_costs={1: 1.0, 2: 2.0, 3: 3.0},
        barrier_values={99},
        resolution=10.0,
    )

    assert geo.land_use.shape == (4, 4)
    assert geo.elevation is not None
    assert geo.elevation.shape == (4, 4)
    assert geo.cost_surface.shape == (4, 4)
    assert geo.cell_size == (10.0, 10.0)
    assert geo.source_cell == (0, 0)
    assert geo.destination_cell == (3, 3)
    assert geo.sources[0, 0] == 1.0
    assert geo.cost_surface[0, 2] == 2.0
    assert geo.cost_surface[2, 2] == 3.0
    assert geo.barriers[0, 3]
    assert geo.barriers[2, 1]

    line_map_xy = geo.raster_line_to_xy(np.array([[0.0, 0.0], [3.0, 3.0]]))
    assert np.allclose(line_map_xy, np.array([[5.0, 35.0], [35.0, 5.0]]))

    result = distance_accumulation(geo.sources, **geo.distance_kwargs())
    assert result.distance[geo.source_cell] == 0.0
    assert math.isfinite(float(result.distance[geo.destination_cell]))


def test_prepare_geo_inputs_reprojects_gpkg_waypoints(tmp_path: Path) -> None:
    land_use_path = tmp_path / "land_use.tif"
    route_path = tmp_path / "route.gpkg"
    transform = from_origin(0.0, 40.0, 10.0, 10.0)

    _write_geotiff(
        land_use_path,
        np.ones((4, 4), dtype=np.float32),
        transform=transform,
        crs="EPSG:3857",
    )

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

    geo = prepare_geo_inputs(land_use_path, waypoints=route_path)

    assert geo.source_cell == (0, 0)
    assert geo.destination_cell == (3, 3)
    assert math.isclose(geo.waypoint_xy[0][0], 5.0, abs_tol=1.0e-8)
    assert math.isclose(geo.waypoint_xy[1][1], 5.0, abs_tol=1.0e-8)


def test_prepare_geo_inputs_crops_to_waypoint_search_radius(tmp_path: Path) -> None:
    land_use_path = tmp_path / "large_land_use.tif"
    transform = from_origin(0.0, 1000.0, 10.0, 10.0)
    land_use = np.arange(100 * 100, dtype=np.float32).reshape((100, 100))
    _write_geotiff(land_use_path, land_use, transform=transform, crs="EPSG:3857")

    geo = prepare_geo_inputs(
        land_use_path,
        waypoints=LineString([(105.0, 895.0), (305.0, 795.0)]),
        search_radius=25.0,
    )

    assert geo.land_use.shape == (15, 25)
    assert geo.cost_surface.shape == (15, 25)
    assert geo.sources.shape == (15, 25)
    assert geo.bounds == (80.0, 770.0, 330.0, 920.0)
    assert geo.cell_to_xy(0, 0) == (85.0, 915.0)
    assert geo.source_cell == (2, 2)
    assert geo.destination_cell == (12, 22)
    assert geo.land_use[0, 0] == land_use[8, 8]
    assert geo.land_use[-1, -1] == land_use[22, 32]
    assert geo.distance_kwargs()["search_radius"] == 25.0


def test_prepare_geo_inputs_rejects_invalid_search_radius(tmp_path: Path) -> None:
    land_use_path = tmp_path / "land_use.tif"
    _write_geotiff(
        land_use_path,
        np.ones((4, 4), dtype=np.float32),
        transform=from_origin(0.0, 40.0, 10.0, 10.0),
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="search_radius"):
        prepare_geo_inputs(
            land_use_path,
            waypoints=LineString([(5.0, 35.0), (35.0, 5.0)]),
            search_radius=0.0,
        )


def test_prepare_geo_inputs_rasterizes_vector_barriers(tmp_path: Path) -> None:
    land_use_path = tmp_path / "land_use.tif"
    _write_geotiff(
        land_use_path,
        np.ones((5, 5), dtype=np.float32),
        transform=from_origin(0.0, 50.0, 10.0, 10.0),
        crs="EPSG:3857",
    )

    geo = prepare_geo_inputs(
        land_use_path,
        waypoints=LineString([(5.0, 45.0), (45.0, 5.0)]),
        barriers=Polygon([(21.0, 1.0), (29.0, 1.0), (29.0, 49.0), (21.0, 49.0)]),
        barrier_all_touched=False,
    )

    assert geo.barriers[:, 2].all()
    assert not geo.barriers[:, 0].any()


def test_geo_distance_accumulation_returns_map_path(tmp_path: Path) -> None:
    land_use_path = tmp_path / "land_use.tif"
    _write_geotiff(
        land_use_path,
        np.ones((5, 5), dtype=np.float32),
        transform=from_origin(0.0, 50.0, 10.0, 10.0),
        crs="EPSG:3857",
    )

    result = geo_distance_accumulation(
        land_use_path,
        waypoints=LineString([(5.0, 45.0), (45.0, 5.0)]),
        search_radius=30.0,
    )
    line_xy = geo_optimal_path_as_line(result, reverse=True)

    assert line_xy.shape[1] == 2
    assert np.allclose(line_xy[0], [5.0, 45.0])
    assert np.allclose(line_xy[-1], [45.0, 5.0])


def test_compute_optimal_path_stitches_waypoint_legs_and_metrics(tmp_path: Path) -> None:
    cost_path = tmp_path / "cost.tif"
    _write_geotiff(
        cost_path,
        np.ones((30, 30), dtype=np.float32),
        transform=from_origin(0.0, 300.0, 10.0, 10.0),
        crs="EPSG:3857",
    )

    route = compute_optimal_path(
        cost_path,
        [(5.0, 295.0), (145.0, 155.0), (245.0, 55.0)],
        search_radius=40.0,
        baseline_speed=5.0,
        waypoint_crs="EPSG:3857",
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
