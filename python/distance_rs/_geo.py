"""Geospatial adapters for preparing raster inputs for distance accumulation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import fiona
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol, xy
from rasterio.warp import (
    reproject,
    transform as transform_coords,
    transform_bounds,
    transform_geom,
)
from shapely.geometry.base import BaseGeometry

from ._distance import (
    DistanceAccumulationResult,
    VerticalFactor,
    distance_accumulation,
    optimal_path_as_line,
)


Bounds = tuple[float, float, float, float]
Cell = tuple[int, int]


@dataclass(frozen=True)
class GeoRasterData:
    """Aligned raster arrays and waypoint cells for the core raster solver.

    The arrays are all on the same target grid. `transform` and `crs` describe
    that grid in map space, while `distance_kwargs()` returns only plain raster
    arrays and cell sizes for the core solver.
    """

    land_use: npt.NDArray[np.float64]
    cost_surface: npt.NDArray[np.float64]
    elevation: npt.NDArray[np.float64] | None
    barriers: npt.NDArray[np.bool_]
    sources: npt.NDArray[np.float64]
    waypoint_cells: tuple[Cell, ...]
    waypoint_xy: tuple[tuple[float, float], ...]
    transform: Any
    crs: Any
    bounds: Bounds
    cell_size: tuple[float, float]
    search_radius: float | None = None

    @property
    def source_cell(self) -> Cell | None:
        return self.waypoint_cells[0] if self.waypoint_cells else None

    @property
    def destination_cell(self) -> Cell | None:
        return self.waypoint_cells[-1] if len(self.waypoint_cells) >= 2 else None

    def distance_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments suitable for `distance_accumulation`."""

        kwargs: dict[str, Any] = {
            "cost_surface": self.cost_surface,
            "barriers": self.barriers,
            "cell_size": self.cell_size,
            "origin": (0.0, 0.0),
        }
        if self.elevation is not None:
            kwargs["elevation"] = self.elevation
        if self.search_radius is not None:
            kwargs["search_radius"] = self.search_radius
        return kwargs

    def cell_to_xy(self, row: int, col: int, *, offset: str = "center") -> tuple[float, float]:
        """Convert one raster cell to map coordinates in the adapter CRS."""

        x_coord, y_coord = xy(self.transform, row, col, offset=offset)
        return float(x_coord), float(y_coord)

    def xy_to_cell(self, x: float, y: float) -> Cell:
        """Convert one map coordinate in the adapter CRS to a raster cell."""

        row, col = rowcol(self.transform, x, y, op=math.floor)
        return int(row), int(col)

    def raster_line_to_xy(self, line_xy: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Convert raster cell-coordinate path vertices to map-space coordinates."""

        line = np.asarray(line_xy, dtype=np.float64)
        if line.ndim != 2 or line.shape[1] != 2:
            raise ValueError("line_xy must have shape (n, 2)")

        out = np.empty_like(line, dtype=np.float64)
        for index, (col, row) in enumerate(line):
            out[index] = self._continuous_cell_to_xy(float(row), float(col))
        return out

    def _continuous_cell_to_xy(self, row: float, col: float) -> tuple[float, float]:
        # The solver's raster coordinates treat integer row/col values as cell
        # centers; affine geotransforms address pixel corners, so add half a cell.
        x, y = self.transform * (col + 0.5, row + 0.5)
        return float(x), float(y)


@dataclass(frozen=True)
class GeoDistanceAccumulationResult:
    """Distance accumulation result paired with its geospatial adapter."""

    geo: GeoRasterData
    accumulation: DistanceAccumulationResult

    @property
    def distance(self) -> npt.NDArray[np.float64]:
        return self.accumulation.distance

    @property
    def source_cell(self) -> Cell | None:
        return self.geo.source_cell

    @property
    def destination_cell(self) -> Cell | None:
        return self.geo.destination_cell

    def optimal_path_as_line(
        self,
        destination: Cell | Sequence[Cell] | None = None,
        *,
        max_steps: int | None = None,
        reverse: bool = False,
    ) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
        return geo_optimal_path_as_line(
            self,
            destination=destination,
            max_steps=max_steps,
            reverse=reverse,
        )


@dataclass(frozen=True)
class PathMetrics:
    """Metrics for one path or a stitched multi-leg route."""

    cost: float
    distance_m: float
    surface_distance_m: float
    time_hours: float
    average_speed_kmh: float


@dataclass(frozen=True)
class OptimalPathLeg:
    """One optimal route leg between consecutive waypoints."""

    index: int
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    source_cell: Cell
    destination_cell: Cell
    path_xy: npt.NDArray[np.float64]
    cost: float
    metrics: PathMetrics | None


@dataclass(frozen=True)
class OptimalPathResult:
    """A full optimal route stitched through all requested waypoints."""

    path_xy: npt.NDArray[np.float64]
    legs: tuple[OptimalPathLeg, ...]
    waypoint_xy: tuple[tuple[float, float], ...]
    crs: Any
    metrics: PathMetrics | None


def prepare_geo_inputs(
    land_use_path: str | Path,
    *,
    elevation_path: str | Path | None = None,
    waypoints: Any | None = None,
    land_use_costs: Mapping[float | int, float] | None = None,
    default_cost: float | None = None,
    barrier_values: set[float | int] | Sequence[float | int] | None = None,
    barriers: Any | None = None,
    target_crs: Any | None = None,
    resolution: float | tuple[float, float] | None = None,
    bounds: Bounds | None = None,
    bounds_crs: Any | None = None,
    waypoint_crs: Any | None = None,
    waypoint_layer: str | int | None = None,
    barrier_crs: Any | None = None,
    barrier_layer: str | int | None = None,
    barrier_all_touched: bool = True,
    land_use_resampling: str | Any = "nearest",
    elevation_resampling: str | Any = "bilinear",
    nodata_is_barrier: bool = True,
    clear_waypoint_cells: bool = False,
    search_radius: float | None = None,
) -> GeoRasterData:
    """Read GeoTIFF rasters and waypoint geometry onto one solver-ready grid.

    Parameters are intentionally GIS-facing; the returned arrays are plain
    NumPy rasters. If `land_use_costs` is provided, land-use class values are
    mapped to traversal costs. Otherwise the land-use raster values are treated
    as costs directly. GeoJSON waypoints default to EPSG:4326 unless
    `waypoint_crs` is supplied; plain coordinate sequences default to the target
    raster CRS. When `search_radius` is provided with at least two waypoints,
    only the target-grid cells inside the first/last waypoint bounding box
    buffered by that radius are read and reprojected. `barriers` may be a
    GeoJSON/GeoPackage path, GeoJSON-like mapping, Shapely geometry, or sequence
    of geometries that will be rasterized onto the same cropped grid.
    """

    with rasterio.open(land_use_path) as land_src:
        target_input = target_crs or land_src.crs
        if target_input is None:
            raise ValueError("land-use raster has no CRS; pass target_crs explicitly")
        target = rasterio.crs.CRS.from_user_input(target_input)

        cell_size = _target_resolution(land_src, resolution)
        waypoint_xy = _load_waypoint_xy(
            waypoints,
            target_crs=target,
            waypoint_crs=waypoint_crs,
            waypoint_layer=waypoint_layer,
        )
        target_bounds = _target_bounds(
            land_src,
            elevation_path=elevation_path,
            target_crs=target,
            bounds=bounds,
            bounds_crs=bounds_crs,
        )
        search_radius_value = _normalize_search_radius(search_radius)
        transform, width, height, adjusted_bounds = _target_grid(target_bounds, cell_size)
        if search_radius_value is not None:
            transform, width, height, adjusted_bounds = _restrict_grid_to_waypoint_radius(
                transform=transform,
                width=width,
                height=height,
                cell_size=cell_size,
                bounds=adjusted_bounds,
                waypoint_xy=waypoint_xy,
                search_radius=search_radius_value,
            )
        land_use = _read_to_grid(
            land_src,
            transform=transform,
            crs=target,
            width=width,
            height=height,
            resampling=land_use_resampling,
        )

    elevation = None
    if elevation_path is not None:
        with rasterio.open(elevation_path) as elevation_src:
            elevation = _read_to_grid(
                elevation_src,
                transform=transform,
                crs=target,
                width=width,
                height=height,
                resampling=elevation_resampling,
            )

    waypoint_cells = _waypoints_to_cells(waypoint_xy, transform=transform, shape=(height, width))

    cost_surface, barrier_mask = _cost_and_barriers(
        land_use,
        elevation=elevation,
        land_use_costs=land_use_costs,
        default_cost=default_cost,
        barrier_values=barrier_values,
        nodata_is_barrier=nodata_is_barrier,
    )
    if barriers is not None:
        barrier_mask |= _rasterize_barriers(
            barriers,
            transform=transform,
            crs=target,
            shape=(height, width),
            source_crs=barrier_crs,
            layer=barrier_layer,
            all_touched=barrier_all_touched,
        )
        cost_surface[barrier_mask] = np.nan
    sources = np.zeros((height, width), dtype=np.float64)
    if waypoint_cells:
        if clear_waypoint_cells:
            _clear_cells(cost_surface, barrier_mask, waypoint_cells)
        _validate_waypoint_cells(cost_surface, barrier_mask, waypoint_cells)
        source_row, source_col = waypoint_cells[0]
        sources[source_row, source_col] = 1.0

    return GeoRasterData(
        land_use=land_use,
        cost_surface=cost_surface,
        elevation=elevation,
        barriers=barrier_mask,
        sources=sources,
        waypoint_cells=tuple(waypoint_cells),
        waypoint_xy=tuple(waypoint_xy),
        transform=transform,
        crs=target,
        bounds=adjusted_bounds,
        cell_size=cell_size,
        search_radius=search_radius_value,
    )


def geo_distance_accumulation(
    cost_raster: str | Path,
    *,
    elevation_path: str | Path | None = None,
    waypoints: Any | None = None,
    land_use_costs: Mapping[float | int, float] | None = None,
    default_cost: float | None = None,
    barrier_values: set[float | int] | Sequence[float | int] | None = None,
    barriers: Any | None = None,
    target_crs: Any | None = None,
    resolution: float | tuple[float, float] | None = None,
    bounds: Bounds | None = None,
    bounds_crs: Any | None = None,
    waypoint_crs: Any | None = None,
    waypoint_layer: str | int | None = None,
    barrier_crs: Any | None = None,
    barrier_layer: str | int | None = None,
    barrier_all_touched: bool = True,
    land_use_resampling: str | Any = "nearest",
    elevation_resampling: str | Any = "bilinear",
    nodata_is_barrier: bool = True,
    clear_waypoint_cells: bool = False,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    search_radius: float | None = None,
    use_surface_distance: bool = True,
) -> GeoDistanceAccumulationResult:
    """Run distance accumulation from geospatial raster/vector inputs."""

    geo = prepare_geo_inputs(
        cost_raster,
        elevation_path=elevation_path,
        waypoints=waypoints,
        land_use_costs=land_use_costs,
        default_cost=default_cost,
        barrier_values=barrier_values,
        barriers=barriers,
        target_crs=target_crs,
        resolution=resolution,
        bounds=bounds,
        bounds_crs=bounds_crs,
        waypoint_crs=waypoint_crs,
        waypoint_layer=waypoint_layer,
        barrier_crs=barrier_crs,
        barrier_layer=barrier_layer,
        barrier_all_touched=barrier_all_touched,
        land_use_resampling=land_use_resampling,
        elevation_resampling=elevation_resampling,
        nodata_is_barrier=nodata_is_barrier,
        clear_waypoint_cells=clear_waypoint_cells,
        search_radius=search_radius,
    )
    accumulation = distance_accumulation(
        geo.sources,
        **geo.distance_kwargs(),
        vertical_factor=vertical_factor,
        use_surface_distance=use_surface_distance,
    )
    return GeoDistanceAccumulationResult(geo=geo, accumulation=accumulation)


def geo_optimal_path_as_line(
    result: GeoDistanceAccumulationResult,
    destination: Cell | Sequence[Cell] | None = None,
    *,
    max_steps: int | None = None,
    reverse: bool = False,
) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
    """Trace optimal paths and return coordinates in the raster's map CRS."""

    if destination is None:
        destination = result.geo.destination_cell
    if destination is None:
        raise ValueError("destination is required when the geo result has no destination waypoint")

    line = optimal_path_as_line(result.accumulation, destination, max_steps=max_steps)
    if isinstance(line, list):
        converted = [
            result.geo.raster_line_to_xy(_solver_line_to_cell_line(item, result.accumulation))
            for item in line
        ]
        return [item[::-1].copy() for item in converted] if reverse else converted

    converted = result.geo.raster_line_to_xy(_solver_line_to_cell_line(line, result.accumulation))
    return converted[::-1].copy() if reverse else converted


def compute_optimal_path(
    cost_raster: str | Path,
    waypoints: Any,
    barriers: Any | None = None,
    elevation: str | Path | None = None,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    search_radius: float | None = None,
    baseline_speed: float = 5.0,
    compute_metrics: bool = True,
    *,
    waypoint_crs: Any | None = None,
    waypoint_layer: str | int | None = None,
    barrier_crs: Any | None = None,
    barrier_layer: str | int | None = None,
    barrier_all_touched: bool = True,
    target_crs: Any | None = None,
    resolution: float | tuple[float, float] | None = None,
    land_use_costs: Mapping[float | int, float] | None = None,
    default_cost: float | None = None,
    barrier_values: set[float | int] | Sequence[float | int] | None = None,
    clear_waypoint_cells: bool = False,
    use_surface_distance: bool = True,
) -> OptimalPathResult:
    """Compute and stitch optimal route legs through consecutive waypoints.

    Plain coordinate lists default to lon/lat (`EPSG:4326`) unless
    `waypoint_crs` is supplied. Shapely geometries and vector files use their
    own coordinates/CRS unless a CRS override is provided.
    """

    baseline_speed_value = float(baseline_speed)
    if baseline_speed_value <= 0.0 or not math.isfinite(baseline_speed_value):
        raise ValueError("baseline_speed must be a positive finite km/hr value")

    target = _target_crs_from_raster(cost_raster, target_crs)
    if getattr(target, "is_geographic", False):
        raise ValueError(
            "compute_optimal_path expects a projected target CRS with meter-like units; "
            "pass target_crs for geographic rasters"
        )
    route_waypoint_crs = _compute_waypoint_crs(waypoints, waypoint_crs)
    waypoint_xy = _load_waypoint_xy(
        waypoints,
        target_crs=target,
        waypoint_crs=route_waypoint_crs,
        waypoint_layer=waypoint_layer,
    )
    if len(waypoint_xy) < 2:
        raise ValueError("at least two waypoints are required")

    route_barrier_crs = _compute_barrier_crs(barriers, barrier_crs, route_waypoint_crs)
    vf = VerticalFactor.from_any(vertical_factor)
    legs: list[OptimalPathLeg] = []
    path_parts: list[npt.NDArray[np.float64]] = []
    total_metrics: PathMetrics | None = None

    for index, (start_xy, end_xy) in enumerate(zip(waypoint_xy, waypoint_xy[1:])):
        leg_result = geo_distance_accumulation(
            cost_raster,
            elevation_path=elevation,
            waypoints=[start_xy, end_xy],
            waypoint_crs=target,
            land_use_costs=land_use_costs,
            default_cost=default_cost,
            barrier_values=barrier_values,
            barriers=barriers,
            target_crs=target,
            resolution=resolution,
            barrier_crs=route_barrier_crs,
            barrier_layer=barrier_layer,
            barrier_all_touched=barrier_all_touched,
            clear_waypoint_cells=clear_waypoint_cells,
            vertical_factor=vf,
            search_radius=search_radius,
            use_surface_distance=use_surface_distance,
        )
        if leg_result.source_cell is None or leg_result.destination_cell is None:
            raise ValueError("each route leg must have source and destination cells")

        path_xy = geo_optimal_path_as_line(leg_result, reverse=True)
        if isinstance(path_xy, list):
            raise RuntimeError("single destination unexpectedly produced multiple paths")

        cost = _destination_cost(leg_result)
        metrics = (
            _path_metrics(
                path_xy,
                cost=cost,
                geo=leg_result.geo,
                vertical_factor=vf,
                baseline_speed_kmh=baseline_speed_value,
            )
            if compute_metrics
            else None
        )
        if metrics is not None:
            total_metrics = metrics if total_metrics is None else _combine_metrics(total_metrics, metrics)

        leg = OptimalPathLeg(
            index=index,
            start_xy=start_xy,
            end_xy=end_xy,
            source_cell=leg_result.source_cell,
            destination_cell=leg_result.destination_cell,
            path_xy=path_xy,
            cost=cost,
            metrics=metrics,
        )
        legs.append(leg)
        path_parts.append(path_xy if not path_parts else path_xy[1:])

    full_path = np.vstack(path_parts) if path_parts else np.empty((0, 2), dtype=np.float64)
    return OptimalPathResult(
        path_xy=full_path,
        legs=tuple(legs),
        waypoint_xy=tuple(waypoint_xy),
        crs=target,
        metrics=total_metrics,
    )


def _target_crs_from_raster(path: str | Path, target_crs: Any | None) -> Any:
    with rasterio.open(path) as dataset:
        target_input = target_crs or dataset.crs
        if target_input is None:
            raise ValueError("cost raster has no CRS; pass target_crs explicitly")
    return rasterio.crs.CRS.from_user_input(target_input)


def _compute_waypoint_crs(waypoints: Any, waypoint_crs: Any | None) -> Any | None:
    if waypoint_crs is not None:
        return waypoint_crs
    if _is_plain_xy_sequence(waypoints):
        return "EPSG:4326"
    return None


def _compute_barrier_crs(
    barriers: Any | None,
    barrier_crs: Any | None,
    waypoint_crs: Any | None,
) -> Any | None:
    if barrier_crs is not None or barriers is None:
        return barrier_crs
    if isinstance(barriers, str | Path):
        return None
    if isinstance(barriers, Mapping) and _geojson_crs(barriers) is not None:
        return None
    return waypoint_crs


def _is_plain_xy_sequence(value: Any) -> bool:
    if isinstance(value, str | bytes | Path | Mapping):
        return False
    if isinstance(value, BaseGeometry) or hasattr(value, "__geo_interface__"):
        return False
    if not isinstance(value, Sequence) or len(value) == 0:
        return False
    first = value[0]
    if not isinstance(first, Sequence) or isinstance(first, str | bytes):
        return False
    if len(first) < 2:
        return False
    try:
        float(first[0])
        float(first[1])
    except (TypeError, ValueError):
        return False
    return True


def _destination_cost(result: GeoDistanceAccumulationResult) -> float:
    if result.destination_cell is None:
        raise ValueError("destination waypoint is required")
    row, col = result.destination_cell
    cost = float(result.distance[row, col])
    if not math.isfinite(cost):
        raise ValueError("destination has no finite accumulated cost")
    return cost


def _solver_line_to_cell_line(
    line_xy: npt.NDArray[np.float64],
    result: DistanceAccumulationResult,
) -> npt.NDArray[np.float64]:
    line = np.asarray(line_xy, dtype=np.float64)
    out = np.empty_like(line)
    out[:, 0] = (line[:, 0] - result.origin[0]) / result.cell_size[0]
    out[:, 1] = (line[:, 1] - result.origin[1]) / result.cell_size[1]
    return out


def _path_metrics(
    path_xy: npt.NDArray[np.float64],
    *,
    cost: float,
    geo: GeoRasterData,
    vertical_factor: VerticalFactor,
    baseline_speed_kmh: float,
) -> PathMetrics:
    distance_m = _path_distance(path_xy)
    surface_distance_m = _path_surface_distance(path_xy, geo)
    time_hours = _time_hours_from_cost(
        cost,
        vertical_factor=vertical_factor,
        baseline_speed_kmh=baseline_speed_kmh,
    )
    return PathMetrics(
        cost=cost,
        distance_m=distance_m,
        surface_distance_m=surface_distance_m,
        time_hours=time_hours,
        average_speed_kmh=_average_speed_kmh(surface_distance_m, time_hours),
    )


def _combine_metrics(first: PathMetrics, second: PathMetrics) -> PathMetrics:
    cost = first.cost + second.cost
    distance_m = first.distance_m + second.distance_m
    surface_distance_m = first.surface_distance_m + second.surface_distance_m
    time_hours = first.time_hours + second.time_hours
    return PathMetrics(
        cost=cost,
        distance_m=distance_m,
        surface_distance_m=surface_distance_m,
        time_hours=time_hours,
        average_speed_kmh=_average_speed_kmh(surface_distance_m, time_hours),
    )


def _path_distance(path_xy: npt.NDArray[np.float64]) -> float:
    if len(path_xy) < 2:
        return 0.0
    delta = np.diff(path_xy, axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def _path_surface_distance(path_xy: npt.NDArray[np.float64], geo: GeoRasterData) -> float:
    plan_distances = _segment_plan_distances(path_xy)
    if geo.elevation is None or plan_distances.size == 0:
        return float(plan_distances.sum())

    elevations = _sample_elevation(path_xy, geo)
    dz = np.diff(elevations)
    finite = np.isfinite(dz)
    surface = plan_distances.copy()
    surface[finite] = np.hypot(plan_distances[finite], dz[finite])
    return float(surface.sum())


def _segment_plan_distances(path_xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    if len(path_xy) < 2:
        return np.empty(0, dtype=np.float64)
    delta = np.diff(path_xy, axis=0)
    return np.hypot(delta[:, 0], delta[:, 1])


def _sample_elevation(path_xy: npt.NDArray[np.float64], geo: GeoRasterData) -> npt.NDArray[np.float64]:
    if geo.elevation is None:
        return np.full(len(path_xy), np.nan, dtype=np.float64)
    rows, cols = rowcol(
        geo.transform,
        path_xy[:, 0].tolist(),
        path_xy[:, 1].tolist(),
        op=math.floor,
    )
    height, width = geo.elevation.shape
    out = np.full(len(path_xy), np.nan, dtype=np.float64)
    for index, (row, col) in enumerate(zip(rows, cols)):
        if 0 <= row < height and 0 <= col < width:
            out[index] = float(geo.elevation[int(row), int(col)])
    return out


def _time_hours_from_cost(
    cost: float,
    *,
    vertical_factor: VerticalFactor,
    baseline_speed_kmh: float,
) -> float:
    if not math.isfinite(cost):
        return math.inf
    if vertical_factor.type in {"hiking_time", "bidir_hiking_time"}:
        return cost
    return cost / (baseline_speed_kmh * 1000.0)


def _average_speed_kmh(distance_m: float, time_hours: float) -> float:
    if time_hours > 0.0:
        return distance_m / 1000.0 / time_hours
    return 0.0 if distance_m == 0.0 else math.inf


def _target_resolution(
    dataset: Any, resolution: float | tuple[float, float] | None
) -> tuple[float, float]:
    if resolution is None:
        x_res, y_res = dataset.res
        return abs(float(x_res)), abs(float(y_res))
    if isinstance(resolution, tuple):
        if len(resolution) != 2:
            raise ValueError("resolution tuple must be (x_resolution, y_resolution)")
        x_res = float(resolution[0])
        y_res = float(resolution[1])
    else:
        x_res = y_res = float(resolution)
    if x_res <= 0.0 or y_res <= 0.0:
        raise ValueError("resolution values must be positive")
    return x_res, y_res


def _target_bounds(
    land_src: Any,
    *,
    elevation_path: str | Path | None,
    target_crs: Any,
    bounds: Bounds | None,
    bounds_crs: Any | None,
) -> Bounds:
    if bounds is not None:
        source_crs = rasterio.crs.CRS.from_user_input(bounds_crs or target_crs)
        if source_crs != target_crs:
            return tuple(transform_bounds(source_crs, target_crs, *bounds, densify_pts=21))  # type: ignore[return-value]
        return tuple(float(value) for value in bounds)  # type: ignore[return-value]

    land_bounds = _dataset_bounds_in_crs(land_src, target_crs)
    if elevation_path is None:
        return land_bounds

    with rasterio.open(elevation_path) as elevation_src:
        elevation_bounds = _dataset_bounds_in_crs(elevation_src, target_crs)
    return _intersect_bounds(land_bounds, elevation_bounds)


def _dataset_bounds_in_crs(dataset: Any, target_crs: Any) -> Bounds:
    if dataset.crs is None:
        raise ValueError(f"{dataset.name} has no CRS")
    if dataset.crs == target_crs:
        return (
            float(dataset.bounds.left),
            float(dataset.bounds.bottom),
            float(dataset.bounds.right),
            float(dataset.bounds.top),
        )
    return tuple(transform_bounds(dataset.crs, target_crs, *dataset.bounds, densify_pts=21))  # type: ignore[return-value]


def _intersect_bounds(first: Bounds, second: Bounds) -> Bounds:
    left = max(first[0], second[0])
    bottom = max(first[1], second[1])
    right = min(first[2], second[2])
    top = min(first[3], second[3])
    if left >= right or bottom >= top:
        raise ValueError("land-use and elevation rasters do not overlap in the target CRS")
    return left, bottom, right, top


def _target_grid(bounds: Bounds, cell_size: tuple[float, float]) -> tuple[Any, int, int, Bounds]:
    left, bottom, right, top = bounds
    x_res, y_res = cell_size
    width = int(math.floor((right - left) / x_res))
    height = int(math.floor((top - bottom) / y_res))
    if width <= 0 or height <= 0:
        raise ValueError("target bounds are smaller than one output cell")
    adjusted_bounds = (left, top - height * y_res, left + width * x_res, top)
    transform = from_origin(left, top, x_res, y_res)
    return transform, width, height, adjusted_bounds


def _normalize_search_radius(search_radius: float | None) -> float | None:
    if search_radius is None:
        return None
    value = float(search_radius)
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("search_radius must be a positive finite value")
    return value


def _restrict_grid_to_waypoint_radius(
    *,
    transform: Any,
    width: int,
    height: int,
    cell_size: tuple[float, float],
    bounds: Bounds,
    waypoint_xy: Sequence[tuple[float, float]],
    search_radius: float,
) -> tuple[Any, int, int, Bounds]:
    if len(waypoint_xy) < 2:
        return transform, width, height, bounds

    left, bottom, right, top = _waypoint_search_bounds(waypoint_xy, search_radius)
    base_left, _, _, base_top = bounds
    x_res, y_res = cell_size

    col_start = max(0, int(math.floor((left - base_left) / x_res)))
    col_stop = min(width, int(math.ceil((right - base_left) / x_res)))
    row_start = max(0, int(math.floor((base_top - top) / y_res)))
    row_stop = min(height, int(math.ceil((base_top - bottom) / y_res)))

    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError("waypoint search-radius bounds do not overlap the target raster grid")

    cropped_left = base_left + col_start * x_res
    cropped_top = base_top - row_start * y_res
    cropped_width = col_stop - col_start
    cropped_height = row_stop - row_start
    cropped_bounds = (
        cropped_left,
        cropped_top - cropped_height * y_res,
        cropped_left + cropped_width * x_res,
        cropped_top,
    )
    return (
        from_origin(cropped_left, cropped_top, x_res, y_res),
        cropped_width,
        cropped_height,
        cropped_bounds,
    )


def _waypoint_search_bounds(
    waypoint_xy: Sequence[tuple[float, float]], search_radius: float
) -> Bounds:
    start_x, start_y = waypoint_xy[0]
    end_x, end_y = waypoint_xy[-1]
    return (
        min(start_x, end_x) - search_radius,
        min(start_y, end_y) - search_radius,
        max(start_x, end_x) + search_radius,
        max(start_y, end_y) + search_radius,
    )


def _read_to_grid(
    dataset: Any,
    *,
    transform: Any,
    crs: Any,
    width: int,
    height: int,
    resampling: str | Any,
) -> npt.NDArray[np.float64]:
    destination = np.full((height, width), np.nan, dtype=np.float64)
    reproject(
        source=rasterio.band(dataset, 1),
        destination=destination,
        src_transform=dataset.transform,
        src_crs=dataset.crs,
        src_nodata=dataset.nodata,
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=np.nan,
        resampling=_resampling_from_any(resampling),
    )
    return destination


def _resampling_from_any(value: str | Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return getattr(Resampling, value)
    except AttributeError as exc:
        valid = ", ".join(item.name for item in Resampling)
        raise ValueError(f"unknown resampling method {value!r}; valid values: {valid}") from exc


def _cost_and_barriers(
    land_use: npt.NDArray[np.float64],
    *,
    elevation: npt.NDArray[np.float64] | None,
    land_use_costs: Mapping[float | int, float] | None,
    default_cost: float | None,
    barrier_values: set[float | int] | Sequence[float | int] | None,
    nodata_is_barrier: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    if land_use_costs is None:
        cost = land_use.astype(np.float64, copy=True)
    else:
        fill_value = np.nan if default_cost is None else float(default_cost)
        cost = np.full(land_use.shape, fill_value, dtype=np.float64)
        for raw_value, mapped_cost in land_use_costs.items():
            cost[land_use == float(raw_value)] = float(mapped_cost)

    barriers = np.zeros(land_use.shape, dtype=np.bool_)
    if barrier_values is not None:
        for raw_value in barrier_values:
            barriers |= land_use == float(raw_value)

    if nodata_is_barrier:
        barriers |= ~np.isfinite(land_use) | ~np.isfinite(cost)
        if elevation is not None:
            barriers |= ~np.isfinite(elevation)

    cost[barriers] = np.nan
    return cost, barriers


def _rasterize_barriers(
    barriers: Any,
    *,
    transform: Any,
    crs: Any,
    shape: tuple[int, int],
    source_crs: Any | None,
    layer: str | int | None,
    all_touched: bool,
) -> npt.NDArray[np.bool_]:
    geometries = _load_geometries(
        barriers,
        target_crs=crs,
        source_crs=source_crs,
        layer=layer,
    )
    if not geometries:
        return np.zeros(shape, dtype=np.bool_)

    burned = rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype=np.uint8,
    )
    return burned.astype(np.bool_)


def _load_geometries(
    value: Any,
    *,
    target_crs: Any,
    source_crs: Any | None,
    layer: str | int | None,
) -> list[Mapping[str, Any]]:
    geometry_crs = source_crs
    if isinstance(value, str | Path):
        path = Path(value)
        if path.suffix.lower() == ".gpkg":
            geometries, geometry_crs = _read_vector_file(path, layer=layer)
        else:
            geometry, file_crs = _read_geojson(path)
            geometry_crs = source_crs or file_crs
            geometries = _geometries_from_geometry(geometry)
    elif isinstance(value, Mapping):
        geometry_crs = source_crs or _geojson_crs(value)
        geometries = _geometries_from_geometry(value)
    elif isinstance(value, BaseGeometry):
        geometries = [value.__geo_interface__]
    elif hasattr(value, "__geo_interface__"):
        geometries = [value.__geo_interface__]
    elif isinstance(value, Sequence):
        geometries = []
        for item in value:
            geometries.extend(
                _load_geometries(
                    item,
                    target_crs=target_crs,
                    source_crs=source_crs,
                    layer=layer,
                )
            )
        return geometries
    else:
        raise TypeError(
            "barriers must be a path, GeoJSON-like mapping, geometry, or sequence of geometries"
        )

    if not geometries:
        return []

    resolved_source_crs = _resolve_geometry_crs(geometry_crs, target_crs)
    if resolved_source_crs == target_crs:
        return geometries

    return [
        transform_geom(resolved_source_crs, target_crs, geometry)  # type: ignore[arg-type]
        for geometry in geometries
    ]


def _resolve_geometry_crs(source_crs: Any | None, target_crs: Any) -> Any:
    if source_crs is None:
        return target_crs
    return rasterio.crs.CRS.from_user_input(source_crs)


def _geometries_from_geometry(geometry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    kind = str(geometry.get("type", ""))
    if kind == "FeatureCollection":
        geometries: list[Mapping[str, Any]] = []
        for feature in geometry.get("features", []):
            geometries.extend(_geometries_from_geometry(feature))
        return geometries
    if kind == "Feature":
        feature_geometry = geometry.get("geometry")
        if not isinstance(feature_geometry, Mapping):
            return []
        return _geometries_from_geometry(feature_geometry)
    if kind == "GeometryCollection":
        geometries = []
        for child in geometry.get("geometries", []):
            geometries.extend(_geometries_from_geometry(child))
        return geometries
    if kind in {
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
    }:
        return [geometry]
    raise ValueError(f"barrier geometry type {kind!r} is not supported")


def _load_waypoint_xy(
    waypoints: Any | None,
    *,
    target_crs: Any,
    waypoint_crs: Any | None,
    waypoint_layer: str | int | None,
) -> list[tuple[float, float]]:
    if waypoints is None:
        return []

    geometry_crs = waypoint_crs
    if isinstance(waypoints, str | Path):
        path = Path(waypoints)
        if path.suffix.lower() == ".gpkg":
            geometries, geometry_crs = _read_vector_file(path, layer=waypoint_layer)
            coords = [coord for geometry in geometries for coord in _coords_from_geometry(geometry)]
        else:
            geometry, file_crs = _read_geojson(path)
            geometry_crs = waypoint_crs or file_crs
            coords = _coords_from_geometry(geometry)
    elif isinstance(waypoints, Mapping):
        coords = _coords_from_geometry(waypoints)
    elif isinstance(waypoints, BaseGeometry):
        coords = _coords_from_geometry(waypoints.__geo_interface__)
    elif hasattr(waypoints, "__geo_interface__"):
        coords = _coords_from_geometry(waypoints.__geo_interface__)
    else:
        coords = _coords_from_sequence(waypoints)

    if not coords:
        return []

    source_crs = _resolve_waypoint_crs(geometry_crs, target_crs)
    if source_crs == target_crs:
        return [(float(x), float(y)) for x, y in coords]

    xs, ys = zip(*coords)
    out_xs, out_ys = transform_coords(source_crs, target_crs, xs, ys)
    return [(float(x), float(y)) for x, y in zip(out_xs, out_ys)]


def _resolve_waypoint_crs(source_crs: Any | None, target_crs: Any) -> Any:
    if source_crs is None:
        return target_crs
    return rasterio.crs.CRS.from_user_input(source_crs)


def _read_geojson(path: Path) -> tuple[Mapping[str, Any], Any | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    crs = _geojson_crs(data)
    return data, crs or "EPSG:4326"


def _geojson_crs(data: Mapping[str, Any]) -> Any | None:
    crs = data.get("crs")
    if not isinstance(crs, Mapping):
        return None
    properties = crs.get("properties")
    if isinstance(properties, Mapping) and "name" in properties:
        return properties["name"]
    return None


def _read_vector_file(
    path: Path, *, layer: str | int | None
) -> tuple[list[Mapping[str, Any]], Any]:
    open_kwargs = {"layer": layer} if layer is not None else {}
    with fiona.open(path, **open_kwargs) as collection:
        crs = collection.crs_wkt or collection.crs
        geometries = [feature["geometry"] for feature in collection if feature.get("geometry")]
    return geometries, crs


def _coords_from_geometry(geometry: Mapping[str, Any]) -> list[tuple[float, float]]:
    kind = str(geometry.get("type", ""))
    if kind == "FeatureCollection":
        coords: list[tuple[float, float]] = []
        for feature in geometry.get("features", []):
            coords.extend(_coords_from_geometry(feature))
        return coords
    if kind == "Feature":
        feature_geometry = geometry.get("geometry")
        if not isinstance(feature_geometry, Mapping):
            return []
        return _coords_from_geometry(feature_geometry)
    if kind == "GeometryCollection":
        coords = []
        for child in geometry.get("geometries", []):
            coords.extend(_coords_from_geometry(child))
        return coords
    if kind == "Point":
        return [_xy_pair(geometry.get("coordinates"))]
    if kind == "MultiPoint":
        return [_xy_pair(item) for item in geometry.get("coordinates", [])]
    if kind == "LineString":
        return [_xy_pair(item) for item in geometry.get("coordinates", [])]
    if kind == "MultiLineString":
        return [_xy_pair(item) for part in geometry.get("coordinates", []) for item in part]
    raise ValueError(f"waypoint geometry type {kind!r} is not supported")


def _coords_from_sequence(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, Sequence):
        raise TypeError(
            "waypoints must be a path, GeoJSON-like mapping, geometry, or coordinate sequence"
        )
    coords = [_xy_pair(item) for item in value]
    return coords


def _xy_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, Sequence) or len(value) < 2:
        raise ValueError(f"invalid waypoint coordinate: {value!r}")
    return float(value[0]), float(value[1])


def _waypoints_to_cells(
    waypoint_xy: Sequence[tuple[float, float]],
    *,
    transform: Any,
    shape: tuple[int, int],
) -> list[Cell]:
    if not waypoint_xy:
        return []

    rows, cols = rowcol(
        transform,
        [xy[0] for xy in waypoint_xy],
        [xy[1] for xy in waypoint_xy],
        op=math.floor,
    )
    cells = [(int(row), int(col)) for row, col in zip(rows, cols)]
    height, width = shape
    out: list[Cell] = []
    for row, col in cells:
        if row < 0 or col < 0 or row >= height or col >= width:
            raise ValueError(f"waypoint {(row, col)} falls outside the aligned raster grid")
        if not out or out[-1] != (row, col):
            out.append((row, col))
    return out


def _clear_cells(
    cost_surface: npt.NDArray[np.float64],
    barriers: npt.NDArray[np.bool_],
    cells: Sequence[Cell],
) -> None:
    finite_costs = cost_surface[np.isfinite(cost_surface)]
    fallback_cost = float(np.median(finite_costs)) if finite_costs.size else 1.0
    for row, col in cells:
        barriers[row, col] = False
        if not math.isfinite(float(cost_surface[row, col])):
            cost_surface[row, col] = fallback_cost


def _validate_waypoint_cells(
    cost_surface: npt.NDArray[np.float64],
    barriers: npt.NDArray[np.bool_],
    cells: Sequence[Cell],
) -> None:
    for row, col in cells:
        if barriers[row, col] or not math.isfinite(float(cost_surface[row, col])):
            raise ValueError(
                f"waypoint cell {(row, col)} is blocked or NoData; "
                "pass clear_waypoint_cells=True to force it open"
            )
