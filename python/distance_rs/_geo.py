"""Small geospatial adapters for raster distance accumulation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import fiona
import numpy as np
import numpy.typing as npt
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol, xy
from rasterio.windows import from_bounds
from rasterio.warp import (
    reproject,
    transform as transform_coords,
    transform_bounds,
    transform_geom,
)
from shapely.geometry.base import BaseGeometry

from ._distance import (
    RasterGrid,
    RasterSurface,
    VerticalFactor,
    distance_accumulation,
    evaluate_path_cost as _evaluate_raster_path_cost,
    optimal_path_trace,
    route_legs_windowed as _route_legs_windowed,
)


Bounds = tuple[float, float, float, float]
Cell = tuple[int, int]
XY = tuple[float, float]

__all__ = [
    "CostRaster",
    "evaluate_path_cost",
    "GeoBarriers",
    "GeoGrid",
    "GeoPoints",
    "GeoSurface",
    "GridSpec",
    "OptimalPathLeg",
    "OptimalPathResult",
    "PathMetrics",
    "load_points",
    "load_surface",
    "route_path",
]


@dataclass(frozen=True)
class GridSpec:
    """Target grid controls for GIS reads.

    `bounds`, when supplied, are in the target CRS.
    """

    crs: Any | None = None
    resolution: float | tuple[float, float] | None = None
    bounds: Bounds | None = None


@dataclass(frozen=True)
class CostRaster:
    """Raster cost input.

    If `values` is supplied, cost raster values are mapped through it.
    Otherwise raster values are used directly as traversal cost. `blocked_values`
    are converted to barriers before solving.
    """

    path: str | Path
    values: Mapping[float | int, float] | None = None
    default: float | None = None
    blocked_values: set[float | int] | Sequence[float | int] | None = None
    resampling: str | Any = "nearest"
    nodata_is_barrier: bool = True


@dataclass(frozen=True)
class GeoPoints:
    """Point, multipoint, or line waypoint input with optional CRS/layer."""

    data: Any
    crs: Any | None = None
    layer: str | int | None = None


@dataclass(frozen=True)
class GeoBarriers:
    """Barrier vector/raster input with optional CRS/layer."""

    data: Any
    crs: Any | None = None
    layer: str | int | None = None
    all_touched: bool = True


@dataclass(frozen=True)
class GeoGrid:
    shape: tuple[int, int]
    transform: Any
    crs: Any
    bounds: Bounds
    cell_size: tuple[float, float]

    @property
    def raster_grid(self) -> RasterGrid:
        return RasterGrid(cell_size=self.cell_size)

    def cell_to_xy(self, row: int, col: int, *, offset: str = "center") -> XY:
        x_coord, y_coord = xy(self.transform, row, col, offset=offset)
        return float(x_coord), float(y_coord)

    def xy_to_cell(self, x: float, y: float) -> Cell:
        row, col = rowcol(self.transform, x, y, op=math.floor)
        row = int(row)
        col = int(col)
        height, width = self.shape
        if row < 0 or col < 0 or row >= height or col >= width:
            raise ValueError(f"coordinate {(x, y)} falls outside the raster grid")
        return row, col

    def raster_line_to_xy(self, line_xy: npt.ArrayLike) -> npt.NDArray[np.float64]:
        line = np.asarray(line_xy, dtype=np.float64)
        if line.ndim != 2 or line.shape[1] != 2:
            raise ValueError("line_xy must have shape (n, 2)")

        out = np.empty_like(line, dtype=np.float64)
        for index, (x_coord, y_coord) in enumerate(line):
            col = float(x_coord) / self.cell_size[0]
            row = float(y_coord) / self.cell_size[1]
            out[index] = self._continuous_cell_to_xy(row, col)
        return out

    def xy_line_to_raster(self, line_xy: Any) -> npt.NDArray[np.float64]:
        line = _line_coordinates(line_xy)
        inverse = ~self.transform
        out = np.empty_like(line, dtype=np.float64)
        for index, (x_coord, y_coord) in enumerate(line):
            col, row = inverse * (float(x_coord), float(y_coord))
            out[index] = (
                (float(col) - 0.5) * self.cell_size[0],
                (float(row) - 0.5) * self.cell_size[1],
            )
        return out

    def _continuous_cell_to_xy(self, row: float, col: float) -> XY:
        # Solver coordinates use integer row/col values at cell centers.
        x_coord, y_coord = self.transform * (col + 0.5, row + 0.5)
        return float(x_coord), float(y_coord)


@dataclass(frozen=True)
class GeoSurface:
    """Solver-ready raster surface plus the geospatial grid that produced it."""

    surface: RasterSurface
    grid: GeoGrid
    land_use: npt.NDArray[np.float64]


@dataclass(frozen=True)
class PathMetrics:
    cost: float
    distance_m: float
    surface_distance_m: float
    time_hours: float
    average_speed_kmh: float


@dataclass(frozen=True)
class OptimalPathLeg:
    index: int
    start_xy: XY
    end_xy: XY
    source_cell: Cell
    destination_cell: Cell
    path_xy: npt.NDArray[np.float64]
    cost: float
    grid: GeoGrid
    metrics: PathMetrics | None
    trace_metadata: dict[str, int]


@dataclass(frozen=True)
class OptimalPathResult:
    path_xy: npt.NDArray[np.float64]
    legs: tuple[OptimalPathLeg, ...]
    waypoint_xy: tuple[XY, ...]
    crs: Any
    metrics: PathMetrics | None


def load_points(points: Any, *, target_crs: Any) -> tuple[XY, ...]:
    """Load waypoint coordinates and reproject them to `target_crs`."""

    target = rio.crs.CRS.from_user_input(target_crs)
    return tuple(_load_point_xy(points, target_crs=target))


def load_surface(
    cost: CostRaster | str | Path,
    *,
    elevation: str | Path | None = None,
    elevation_resampling: str | Any = "bilinear",
    barriers: Any | None = None,
    grid: GridSpec | None = None,
) -> GeoSurface:
    """Read cost/elevation/barriers onto one target grid."""

    cost_spec = _cost_spec(cost)
    grid_spec = grid or GridSpec()

    with rio.open(cost_spec.path) as cost_src:
        target = _target_crs(cost_src, grid_spec.crs, name="cost raster")
        cell_size = _target_resolution(cost_src, grid_spec.resolution)
        base_bounds = _target_bounds(
            cost_src,
            elevation_path=Path(elevation) if elevation is not None else None,
            target_crs=target,
        )
        target_bounds = (
            _restrict_bounds_to_grid(grid_spec.bounds, base_bounds, cell_size)
            if grid_spec.bounds is not None
            else base_bounds
        )
        transform, width, height, adjusted_bounds = _target_grid(target_bounds, cell_size)
        land_use = _read_to_grid(
            cost_src,
            transform=transform,
            crs=target,
            width=width,
            height=height,
            resampling=cost_spec.resampling,
        )

    elevation_arr = None
    if elevation is not None:
        with rio.open(elevation) as elevation_src:
            elevation_arr = _read_to_grid(
                elevation_src,
                transform=transform,
                crs=target,
                width=width,
                height=height,
                resampling=elevation_resampling,
            )

    cost_arr, barrier_mask = _cost_and_barriers(
        land_use,
        elevation=elevation_arr,
        cost_values=cost_spec.values,
        default_cost=cost_spec.default,
        blocked_values=cost_spec.blocked_values,
        nodata_is_barrier=cost_spec.nodata_is_barrier,
    )
    if barriers is not None:
        barrier_mask |= _load_barrier_mask(
            barriers,
            transform=transform,
            crs=target,
            shape=(height, width),
        )
    barrier_mask = _edge_connect_barriers(barrier_mask)
    cost_arr[barrier_mask] = np.nan

    geo_grid = GeoGrid(
        shape=(height, width),
        transform=transform,
        crs=target,
        bounds=adjusted_bounds,
        cell_size=cell_size,
    )
    surface = RasterSurface(
        cost=cost_arr,
        grid=geo_grid.raster_grid,
        elevation=elevation_arr,
        barriers=barrier_mask,
    )
    return GeoSurface(surface=surface, grid=geo_grid, land_use=land_use)


def route_path(
    cost: CostRaster | str | Path,
    waypoints: Any,
    *,
    elevation: str | Path | None = None,
    elevation_resampling: str | Any = "bilinear",
    barriers: Any | None = None,
    grid: GridSpec | None = None,
    margin: float | None = None,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    baseline_speed: float = 5.0,
    compute_metrics: bool = True,
    parallel: bool = False,
) -> OptimalPathResult:
    """Compute and stitch optimal route legs through consecutive waypoints."""

    baseline_speed_value = float(baseline_speed)
    if baseline_speed_value <= 0.0 or not math.isfinite(baseline_speed_value):
        raise ValueError("baseline_speed must be a positive finite km/hr value")

    cost_spec = _cost_spec(cost)
    grid_spec = grid or GridSpec()
    with rio.open(cost_spec.path) as cost_src:
        target = _target_crs(cost_src, grid_spec.crs, name="cost raster")

    if getattr(target, "is_geographic", False):
        raise ValueError("route_path expects a projected target CRS with meter-like units")

    waypoint_xy = load_points(waypoints, target_crs=target)
    if len(waypoint_xy) < 2:
        raise ValueError("at least two waypoints are required")

    vf = VerticalFactor.from_any(vertical_factor)
    if parallel:
        return _route_path_parallel(
            cost_spec,
            waypoint_xy,
            target=target,
            elevation=elevation,
            elevation_resampling=elevation_resampling,
            barriers=barriers,
            grid_spec=grid_spec,
            margin=margin,
            vertical_factor=vf,
            baseline_speed_value=baseline_speed_value,
            compute_metrics=compute_metrics,
        )

    legs: list[OptimalPathLeg] = []
    path_parts: list[npt.NDArray[np.float64]] = []
    total_metrics: PathMetrics | None = None

    for index, (start_xy, end_xy) in enumerate(zip(waypoint_xy, waypoint_xy[1:])):
        leg_grid = _leg_grid_spec(grid_spec, target, start_xy, end_xy, margin=margin)
        geo = load_surface(
            cost_spec,
            elevation=elevation,
            elevation_resampling=elevation_resampling,
            barriers=barriers,
            grid=leg_grid,
        )
        source_cell = geo.grid.xy_to_cell(*start_xy)
        destination_cell = geo.grid.xy_to_cell(*end_xy)
        _validate_endpoint(geo, source_cell, "source")
        _validate_endpoint(geo, destination_cell, "destination")

        accumulation = distance_accumulation(
            geo.surface,
            source_cell,
            target=destination_cell,
            vertical_factor=vf,
        )
        trace = optimal_path_trace(accumulation, destination_cell)
        if isinstance(trace, list):
            raise RuntimeError("single destination unexpectedly produced multiple paths")
        path_xy = geo.grid.raster_line_to_xy(trace.line)[::-1].copy()
        cost_value = _destination_cost(accumulation.distance, destination_cell)
        metrics = (
            _path_metrics(
                path_xy,
                cost=cost_value,
                geo=geo,
                vertical_factor=vf,
                baseline_speed_kmh=baseline_speed_value,
            )
            if compute_metrics
            else None
        )
        if metrics is not None:
            total_metrics = (
                metrics if total_metrics is None else _combine_metrics(total_metrics, metrics)
            )

        legs.append(
            OptimalPathLeg(
                index=index,
                start_xy=start_xy,
                end_xy=end_xy,
                source_cell=source_cell,
                destination_cell=destination_cell,
                path_xy=path_xy,
                cost=cost_value,
                grid=geo.grid,
                metrics=metrics,
                trace_metadata=trace.metadata,
            )
        )
        path_parts.append(path_xy if not path_parts else path_xy[1:])

    full_path = np.vstack(path_parts) if path_parts else np.empty((0, 2), dtype=np.float64)
    return OptimalPathResult(
        path_xy=full_path,
        legs=tuple(legs),
        waypoint_xy=waypoint_xy,
        crs=target,
        metrics=total_metrics,
    )


def _route_path_parallel(
    cost_spec: CostRaster,
    waypoint_xy: tuple[XY, ...],
    *,
    target: Any,
    elevation: str | Path | None,
    elevation_resampling: str | Any,
    barriers: Any | None,
    grid_spec: GridSpec,
    margin: float | None,
    vertical_factor: VerticalFactor,
    baseline_speed_value: float,
    compute_metrics: bool,
) -> OptimalPathResult:
    route_grid = _route_grid_spec(grid_spec, target, waypoint_xy, margin=margin)
    geo = load_surface(
        cost_spec,
        elevation=elevation,
        elevation_resampling=elevation_resampling,
        barriers=barriers,
        grid=route_grid,
    )

    leg_windows: list[tuple[int, int, int, int, int, int, int, int]] = []
    leg_endpoints: list[tuple[Cell, Cell]] = []
    for start_xy, end_xy in zip(waypoint_xy, waypoint_xy[1:]):
        source_cell = geo.grid.xy_to_cell(*start_xy)
        destination_cell = geo.grid.xy_to_cell(*end_xy)
        _validate_endpoint(geo, source_cell, "source")
        _validate_endpoint(geo, destination_cell, "destination")
        leg_windows.append(
            (
                *source_cell,
                *destination_cell,
                *_leg_window(geo.grid, start_xy, end_xy, margin=margin),
            )
        )
        leg_endpoints.append((source_cell, destination_cell))

    solved_legs = _route_legs_windowed(
        geo.surface,
        np.asarray(leg_windows, dtype=np.int64),
        vertical_factor=vertical_factor,
    )

    legs: list[OptimalPathLeg] = []
    path_parts: list[npt.NDArray[np.float64]] = []
    total_metrics: PathMetrics | None = None
    for index, (start_xy, end_xy) in enumerate(zip(waypoint_xy, waypoint_xy[1:])):
        source_cell, destination_cell = leg_endpoints[index]
        solved = solved_legs[index]
        path_xy = geo.grid.raster_line_to_xy(solved.line)[::-1].copy()
        metrics = (
            _path_metrics(
                path_xy,
                cost=solved.cost,
                geo=geo,
                vertical_factor=vertical_factor,
                baseline_speed_kmh=baseline_speed_value,
            )
            if compute_metrics
            else None
        )
        if metrics is not None:
            total_metrics = (
                metrics if total_metrics is None else _combine_metrics(total_metrics, metrics)
            )

        legs.append(
            OptimalPathLeg(
                index=index,
                start_xy=start_xy,
                end_xy=end_xy,
                source_cell=source_cell,
                destination_cell=destination_cell,
                path_xy=path_xy,
                cost=solved.cost,
                grid=geo.grid,
                metrics=metrics,
                trace_metadata=solved.metadata,
            )
        )
        path_parts.append(path_xy if not path_parts else path_xy[1:])

    full_path = np.vstack(path_parts) if path_parts else np.empty((0, 2), dtype=np.float64)
    return OptimalPathResult(
        path_xy=full_path,
        legs=tuple(legs),
        waypoint_xy=waypoint_xy,
        crs=target,
        metrics=total_metrics,
    )


def evaluate_path_cost(
    surface: GeoSurface | RasterSurface | npt.ArrayLike,
    line_xy: Any,
    *,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    max_step: float | None = None,
) -> float:
    """Evaluate a directional path against a raster or geospatial surface.

    For `GeoSurface`, `line_xy` is interpreted in the surface CRS and converted
    into the solver's raster x/y grid before delegating to the low-level
    evaluator. Plain `RasterSurface` inputs use solver-grid coordinates.
    """

    if isinstance(surface, GeoSurface):
        return _evaluate_raster_path_cost(
            surface.surface,
            surface.grid.xy_line_to_raster(line_xy),
            vertical_factor=vertical_factor,
            max_step=max_step,
        )
    return _evaluate_raster_path_cost(
        surface,
        line_xy,
        vertical_factor=vertical_factor,
        max_step=max_step,
    )


def _cost_spec(cost: CostRaster | str | Path) -> CostRaster:
    return cost if isinstance(cost, CostRaster) else CostRaster(cost)


def _target_crs(dataset: Any, crs: Any | None, *, name: str) -> Any:
    target_input = crs or dataset.crs
    if target_input is None:
        raise ValueError(f"{name} has no CRS; pass GridSpec(crs=...)")
    return rio.crs.CRS.from_user_input(target_input)


def _leg_grid_spec(
    grid: GridSpec,
    target_crs: Any,
    start_xy: XY,
    end_xy: XY,
    *,
    margin: float | None,
) -> GridSpec:
    margin = _normalize_optional_radius(margin, "margin")
    bounds = grid.bounds
    if margin is not None:
        bounds = (
            min(start_xy[0], end_xy[0]) - margin,
            min(start_xy[1], end_xy[1]) - margin,
            max(start_xy[0], end_xy[0]) + margin,
            max(start_xy[1], end_xy[1]) + margin,
        )
    return replace(grid, crs=target_crs, bounds=bounds)


def _route_grid_spec(
    grid: GridSpec,
    target_crs: Any,
    waypoint_xy: Sequence[XY],
    *,
    margin: float | None,
) -> GridSpec:
    margin = _normalize_optional_radius(margin, "margin")
    bounds = grid.bounds
    if margin is not None:
        xs = [point[0] for point in waypoint_xy]
        ys = [point[1] for point in waypoint_xy]
        bounds = (
            min(xs) - margin,
            min(ys) - margin,
            max(xs) + margin,
            max(ys) + margin,
        )
    return replace(grid, crs=target_crs, bounds=bounds)


def _leg_window(
    grid: GeoGrid,
    start_xy: XY,
    end_xy: XY,
    *,
    margin: float | None,
) -> tuple[int, int, int, int]:
    if margin is None:
        height, width = grid.shape
        return 0, height, 0, width

    margin = _normalize_optional_radius(margin, "margin")
    assert margin is not None
    bounds = (
        min(start_xy[0], end_xy[0]) - margin,
        min(start_xy[1], end_xy[1]) - margin,
        max(start_xy[0], end_xy[0]) + margin,
        max(start_xy[1], end_xy[1]) + margin,
    )
    window = from_bounds(*bounds, transform=grid.transform)
    height, width = grid.shape
    row_min = max(0, math.floor(window.row_off))
    col_min = max(0, math.floor(window.col_off))
    row_max = min(height, math.ceil(window.row_off + window.height))
    col_max = min(width, math.ceil(window.col_off + window.width))
    if row_min >= row_max or col_min >= col_max:
        raise ValueError("route leg crop does not overlap the loaded raster")
    return row_min, row_max, col_min, col_max


def _validate_endpoint(geo: GeoSurface, cell: Cell, name: str) -> None:
    row, col = cell
    if geo.surface.barriers is not None and bool(geo.surface.barriers[row, col]):
        raise ValueError(f"{name} cell {cell} is blocked")
    if not math.isfinite(float(np.asarray(geo.surface.cost)[row, col])):
        raise ValueError(f"{name} cell {cell} has no finite cost")


def _destination_cost(distance: npt.NDArray[np.float64], cell: Cell) -> float:
    value = float(distance[cell])
    if not math.isfinite(value):
        raise ValueError("destination has no finite accumulated cost")
    return value


def _path_metrics(
    path_xy: npt.NDArray[np.float64],
    *,
    cost: float,
    geo: GeoSurface,
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


def _path_surface_distance(path_xy: npt.NDArray[np.float64], geo: GeoSurface) -> float:
    plan_distances = _segment_plan_distances(path_xy)
    elevation = geo.surface.elevation
    if elevation is None or plan_distances.size == 0:
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


def _sample_elevation(path_xy: npt.NDArray[np.float64], geo: GeoSurface) -> npt.NDArray[np.float64]:
    elevation = geo.surface.elevation
    if elevation is None:
        return np.full(len(path_xy), np.nan, dtype=np.float64)
    rows, cols = rowcol(
        geo.grid.transform,
        path_xy[:, 0].tolist(),
        path_xy[:, 1].tolist(),
        op=math.floor,
    )
    height, width = elevation.shape
    out = np.full(len(path_xy), np.nan, dtype=np.float64)
    for index, (row, col) in enumerate(zip(rows, cols)):
        if 0 <= row < height and 0 <= col < width:
            out[index] = float(elevation[int(row), int(col)])
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
    if x_res <= 0.0 or y_res <= 0.0 or not math.isfinite(x_res) or not math.isfinite(y_res):
        raise ValueError("resolution values must be positive finite values")
    return x_res, y_res


def _target_bounds(
    cost_src: Any,
    *,
    elevation_path: str | Path | None,
    target_crs: Any,
) -> Bounds:
    cost_bounds = _dataset_bounds_in_crs(cost_src, target_crs)
    if elevation_path is None:
        return cost_bounds

    with rio.open(elevation_path) as elevation_src:
        elevation_bounds = _dataset_bounds_in_crs(elevation_src, target_crs)
    return _intersect_bounds(cost_bounds, elevation_bounds)


def _restrict_bounds_to_grid(
    requested: Bounds,
    base_bounds: Bounds,
    cell_size: tuple[float, float],
) -> Bounds:
    left, bottom, right, top = requested
    base_left, base_bottom, base_right, base_top = base_bounds
    x_res, y_res = cell_size
    base_width = int(math.floor((base_right - base_left) / x_res))
    base_height = int(math.floor((base_top - base_bottom) / y_res))

    col_start = max(0, int(math.floor((left - base_left) / x_res)))
    col_stop = min(base_width, int(math.ceil((right - base_left) / x_res)))
    row_start = max(0, int(math.floor((base_top - top) / y_res)))
    row_stop = min(base_height, int(math.ceil((base_top - bottom) / y_res)))
    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError("requested bounds do not overlap the target raster grid")

    snapped_left = base_left + col_start * x_res
    snapped_top = base_top - row_start * y_res
    snapped_width = col_stop - col_start
    snapped_height = row_stop - row_start
    return (
        snapped_left,
        snapped_top - snapped_height * y_res,
        snapped_left + snapped_width * x_res,
        snapped_top,
    )


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
        raise ValueError("cost and elevation rasters do not overlap in the target CRS")
    return left, bottom, right, top


def _target_grid(bounds: Bounds, cell_size: tuple[float, float]) -> tuple[Any, int, int, Bounds]:
    left, bottom, right, top = bounds
    x_res, y_res = cell_size
    width = int(math.floor((right - left) / x_res))
    height = int(math.floor((top - bottom) / y_res))
    if width <= 0 or height <= 0:
        raise ValueError("target bounds are smaller than one output cell")
    adjusted_bounds = (left, top - height * y_res, left + width * x_res, top)
    return from_origin(left, top, x_res, y_res), width, height, adjusted_bounds


def _normalize_optional_radius(radius: float | None, name: str) -> float | None:
    if radius is None:
        return None
    value = float(radius)
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a positive finite value")
    return value


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
        source=rio.band(dataset, 1),
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
    values: npt.NDArray[np.float64],
    *,
    elevation: npt.NDArray[np.float64] | None,
    cost_values: Mapping[float | int, float] | None,
    default_cost: float | None,
    blocked_values: set[float | int] | Sequence[float | int] | None,
    nodata_is_barrier: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    if cost_values is None:
        cost = values.astype(np.float64, copy=True)
    else:
        fill_value = np.nan if default_cost is None else float(default_cost)
        cost = np.full(values.shape, fill_value, dtype=np.float64)
        for raw_value, mapped_cost in cost_values.items():
            cost[values == float(raw_value)] = float(mapped_cost)

    barriers = np.zeros(values.shape, dtype=np.bool_)
    if blocked_values is not None:
        for raw_value in blocked_values:
            barriers |= values == float(raw_value)

    if nodata_is_barrier:
        barriers |= ~np.isfinite(values) | ~np.isfinite(cost)
        if elevation is not None:
            barriers |= ~np.isfinite(elevation)

    cost[barriers] = np.nan
    return cost, barriers


def _load_barrier_mask(
    value: Any,
    *,
    transform: Any,
    crs: Any,
    shape: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    if _is_sequence_of_inputs(value):
        mask = np.zeros(shape, dtype=np.bool_)
        for item in value:
            mask |= _load_barrier_mask(item, transform=transform, crs=crs, shape=shape)
        return mask

    wrapped = value if isinstance(value, GeoBarriers) else GeoBarriers(value)
    data = wrapped.data
    if isinstance(data, str | Path) and _is_raster_path(Path(data)):
        return _read_barrier_raster(
            Path(data),
            transform=transform,
            crs=crs,
            shape=shape,
        )

    geometries = _load_geometries(
        wrapped.data,
        target_crs=crs,
        source_crs=wrapped.crs,
        layer=wrapped.layer,
    )
    if not geometries:
        return np.zeros(shape, dtype=np.bool_)

    burned = rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=wrapped.all_touched,
        dtype=np.uint8,
    )
    return burned.astype(np.bool_)


def _read_barrier_raster(
    path: Path,
    *,
    transform: Any,
    crs: Any,
    shape: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    height, width = shape
    with rio.open(path) as dataset:
        values = _read_to_grid(
            dataset,
            transform=transform,
            crs=crs,
            width=width,
            height=height,
            resampling="nearest",
        )
    return np.isfinite(values) & (values != 0.0)


def _edge_connect_barriers(barriers: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    connected = barriers.copy()
    down_diag = barriers[:-1, :-1] & barriers[1:, 1:]
    connected[:-1, 1:] |= down_diag
    connected[1:, :-1] |= down_diag

    up_diag = barriers[1:, :-1] & barriers[:-1, 1:]
    connected[:-1, :-1] |= up_diag
    connected[1:, 1:] |= up_diag
    return connected


def _load_geometries(
    value: Any,
    *,
    target_crs: Any,
    source_crs: Any | None,
    layer: str | int | None,
) -> list[Mapping[str, Any]]:
    if isinstance(value, GeoBarriers):
        return _load_geometries(
            value.data,
            target_crs=target_crs,
            source_crs=value.crs,
            layer=value.layer,
        )
    if isinstance(value, str | Path):
        path = Path(value)
        if _is_geojson_path(path):
            geometry, file_crs = _read_geojson(path)
            geometry_crs = source_crs or file_crs
            geometries = _geometries_from_geometry(geometry)
        else:
            geometries, geometry_crs = _read_vector_file(path, layer=layer)
            geometry_crs = source_crs or geometry_crs
    elif isinstance(value, Mapping):
        geometry_crs = source_crs or _geojson_crs(value) or "EPSG:4326"
        geometries = _geometries_from_geometry(value)
    elif isinstance(value, BaseGeometry) or hasattr(value, "__geo_interface__"):
        if source_crs is None:
            raise ValueError(
                "geometry barriers without file metadata require GeoBarriers(..., crs=...)"
            )
        geometries = [value.__geo_interface__]
        geometry_crs = source_crs
    elif _is_sequence_of_inputs(value):
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
        raise TypeError("barriers must be a raster path, vector path, geometry, or sequence")

    if not geometries:
        return []

    resolved_source_crs = rio.crs.CRS.from_user_input(geometry_crs)
    if resolved_source_crs == target_crs:
        return geometries
    return [
        transform_geom(resolved_source_crs, target_crs, geometry)  # type: ignore[arg-type]
        for geometry in geometries
    ]


def _load_point_xy(points: Any, *, target_crs: Any) -> list[XY]:
    wrapped = points if isinstance(points, GeoPoints) else GeoPoints(points)
    value = wrapped.data
    geometry_crs = wrapped.crs

    if isinstance(value, str | Path):
        path = Path(value)
        if _is_geojson_path(path):
            geometry, file_crs = _read_geojson(path)
            geometry_crs = geometry_crs or file_crs
            coords = _coords_from_geometry(geometry)
        else:
            geometries, file_crs = _read_vector_file(path, layer=wrapped.layer)
            geometry_crs = geometry_crs or file_crs
            coords = [coord for geometry in geometries for coord in _coords_from_geometry(geometry)]
    elif isinstance(value, Mapping):
        geometry_crs = geometry_crs or _geojson_crs(value) or "EPSG:4326"
        coords = _coords_from_geometry(value)
    elif isinstance(value, BaseGeometry) or hasattr(value, "__geo_interface__"):
        if geometry_crs is None:
            raise ValueError("geometry waypoints require GeoPoints(..., crs=...)")
        coords = _coords_from_geometry(value.__geo_interface__)
    else:
        if geometry_crs is None:
            raise ValueError("plain coordinate waypoints require GeoPoints(..., crs=...)")
        coords = _coords_from_sequence(value)

    if not coords:
        return []
    if geometry_crs is None:
        raise ValueError("waypoint input has no CRS; pass GeoPoints(..., crs=...)")
    source_crs = rio.crs.CRS.from_user_input(geometry_crs)
    if source_crs == target_crs:
        return [(float(x_coord), float(y_coord)) for x_coord, y_coord in coords]

    xs, ys = zip(*coords)
    out_xs, out_ys = transform_coords(source_crs, target_crs, xs, ys)
    return [(float(x_coord), float(y_coord)) for x_coord, y_coord in zip(out_xs, out_ys)]


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
    if crs is None:
        raise ValueError(f"{path} has no CRS")
    return geometries, crs


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


def _coords_from_geometry(geometry: Mapping[str, Any]) -> list[XY]:
    kind = str(geometry.get("type", ""))
    if kind == "FeatureCollection":
        coords: list[XY] = []
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


def _line_coordinates(line_xy: Any) -> npt.NDArray[np.float64]:
    if isinstance(line_xy, Mapping):
        coords = _coords_from_geometry(line_xy)
    elif isinstance(line_xy, BaseGeometry) or hasattr(line_xy, "__geo_interface__"):
        coords = _coords_from_geometry(line_xy.__geo_interface__)
    elif hasattr(line_xy, "coords"):
        coords = [_xy_pair(item) for item in line_xy.coords]
    else:
        coords = line_xy

    line = np.asarray(coords, dtype=np.float64)
    if line.ndim != 2 or line.shape[1] != 2:
        raise ValueError("line_xy must have shape (n, 2)")
    if len(line) == 0:
        raise ValueError("line_xy must contain at least one point")
    if not np.all(np.isfinite(line)):
        raise ValueError("line_xy coordinates must be finite")
    return np.ascontiguousarray(line, dtype=np.float64)


def _coords_from_sequence(value: Any) -> list[XY]:
    if not isinstance(value, Sequence):
        raise TypeError("waypoints must be a path, GeoJSON-like mapping, geometry, or coordinates")
    return [_xy_pair(item) for item in value]


def _xy_pair(value: Any) -> XY:
    if not isinstance(value, Sequence) or len(value) < 2:
        raise ValueError(f"invalid coordinate: {value!r}")
    return float(value[0]), float(value[1])


def _is_sequence_of_inputs(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | Path)
        and not isinstance(value, Mapping)
        and not isinstance(value, BaseGeometry)
        and not hasattr(value, "__geo_interface__")
    )


def _is_geojson_path(path: Path) -> bool:
    return path.suffix.lower() in {".json", ".geojson"}


def _is_raster_path(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff", ".img", ".vrt"}
