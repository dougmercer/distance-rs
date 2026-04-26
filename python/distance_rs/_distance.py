from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeGuard, cast, overload

import numpy as np
import numpy.typing as npt

from . import _native

Cell = tuple[int, int]
ProgressCallback = Callable[[int, int], None]
ProgressOption = bool | ProgressCallback | None

__all__ = [
    "DistanceAccumulationResult",
    "evaluate_path_cost",
    "RasterGrid",
    "RasterSurface",
    "PathTraceResult",
    "RouteLegResult",
    "VerticalFactor",
    "distance_accumulation",
    "optimal_path_as_line",
    "optimal_path_trace",
    "route_legs",
    "route_legs_windowed",
]


_VERTICAL_FACTOR_OPTION_NAMES = (
    "zero_factor",
    "low_cut_angle",
    "high_cut_angle",
    "slope",
    "power",
    "cos_power",
    "sec_power",
)

_ALIASES = {
    "binary": "binary",
    "linear": "linear",
    "inverse_linear": "inverse_linear",
    "inverse linear": "inverse_linear",
    "sym_linear": "symmetric_linear",
    "symmetric_linear": "symmetric_linear",
    "symmetric linear": "symmetric_linear",
    "sym_inverse_linear": "symmetric_inverse_linear",
    "symmetric_inverse_linear": "symmetric_inverse_linear",
    "symmetric inverse linear": "symmetric_inverse_linear",
    "cos": "cos",
    "sec": "sec",
    "cos_sec": "cos_sec",
    "cos-sec": "cos_sec",
    "sec_cos": "sec_cos",
    "sec-cos": "sec_cos",
    "hiking": "hiking_time",
    "hiking_time": "hiking_time",
    "hiking time": "hiking_time",
    "bidir_hiking": "bidir_hiking_time",
    "bidir_hiking_time": "bidir_hiking_time",
    "bidirectional_hiking_time": "bidir_hiking_time",
    "bidirectional hiking time": "bidir_hiking_time",
}


_DEFAULTS: dict[str, dict[str, float]] = {
    "none": {
        "zero_factor": 1.0,
        "low_cut_angle": -90.0,
        "high_cut_angle": 90.0,
        "slope": 0.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "binary": {
        "zero_factor": 1.0,
        "low_cut_angle": -30.0,
        "high_cut_angle": 30.0,
        "slope": 0.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "linear": {
        "zero_factor": 1.0,
        "low_cut_angle": -90.0,
        "high_cut_angle": 90.0,
        "slope": 1.0 / 90.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "inverse_linear": {
        "zero_factor": 1.0,
        "low_cut_angle": -45.0,
        "high_cut_angle": 45.0,
        "slope": -1.0 / 45.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "symmetric_linear": {
        "zero_factor": 1.0,
        "low_cut_angle": -90.0,
        "high_cut_angle": 90.0,
        "slope": 1.0 / 90.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "symmetric_inverse_linear": {
        "zero_factor": 1.0,
        "low_cut_angle": -45.0,
        "high_cut_angle": 45.0,
        "slope": -1.0 / 45.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "cos": {
        "zero_factor": 1.0,
        "low_cut_angle": -90.0,
        "high_cut_angle": 90.0,
        "slope": 0.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "sec": {
        "zero_factor": 1.0,
        "low_cut_angle": -90.0,
        "high_cut_angle": 90.0,
        "slope": 0.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "cos_sec": {
        "zero_factor": 1.0,
        "low_cut_angle": -90.0,
        "high_cut_angle": 90.0,
        "slope": 0.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "sec_cos": {
        "zero_factor": 1.0,
        "low_cut_angle": -90.0,
        "high_cut_angle": 90.0,
        "slope": 0.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "hiking_time": {
        "zero_factor": 1.0,
        "low_cut_angle": -70.0,
        "high_cut_angle": 70.0,
        "slope": 0.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
    "bidir_hiking_time": {
        "zero_factor": 1.0,
        "low_cut_angle": -70.0,
        "high_cut_angle": 70.0,
        "slope": 0.0,
        "power": 1.0,
        "cos_power": 1.0,
        "sec_power": 1.0,
    },
}


class _ResolvedVerticalFactor(Protocol):
    type: str
    zero_factor: float
    low_cut_angle: float
    high_cut_angle: float
    slope: float
    power: float
    cos_power: float
    sec_power: float


@dataclass(frozen=True)
class VerticalFactor:
    """Vertical cost factor specification."""

    type: str = "none"
    zero_factor: float | None = None
    low_cut_angle: float | None = None
    high_cut_angle: float | None = None
    slope: float | None = None
    power: float | None = None
    cos_power: float | None = None
    sec_power: float | None = None

    @classmethod
    def from_any(cls, value: str | Mapping[str, Any] | VerticalFactor | None) -> VerticalFactor:
        if value is None:
            return cls("none").normalized()
        if isinstance(value, VerticalFactor):
            return value.normalized()
        if isinstance(value, str):
            return cls(value).normalized()
        if isinstance(value, Mapping):
            data = dict(value)
            kind = data.pop("type", data.pop("kind", data.pop("name", "none")))
            valid_keys = set(_VERTICAL_FACTOR_OPTION_NAMES)
            unknown = set(data) - valid_keys
            if unknown:
                joined = ", ".join(sorted(unknown))
                raise TypeError(f"unknown vertical factor option(s): {joined}")
            return cls(str(kind), **data).normalized()
        raise TypeError("vertical_factor must be a string, mapping, VerticalFactor, or None")

    def normalized(self) -> VerticalFactor:
        kind = _ALIASES.get(self.type.strip().lower(), self.type.strip().lower())
        if kind not in _DEFAULTS:
            valid = ", ".join(sorted(k for k in _DEFAULTS if k != "none"))
            raise ValueError(f"unknown vertical factor {self.type!r}; valid values: {valid}")
        return VerticalFactor(kind, **self._resolved_options(kind))

    def _resolved_options(self, kind: str) -> dict[str, float]:
        defaults = _DEFAULTS[kind]
        options = {
            name: defaults[name] if getattr(self, name) is None else float(getattr(self, name))
            for name in _VERTICAL_FACTOR_OPTION_NAMES
        }
        for option_name, option_value in options.items():
            if not math.isfinite(option_value):
                raise ValueError(f"vertical factor option {option_name} must be finite")
        if options["low_cut_angle"] >= options["high_cut_angle"]:
            raise ValueError("low_cut_angle must be less than high_cut_angle")
        return options

    def as_native(self) -> dict[str, float | str]:
        vf = self.normalized()
        return {
            "type": vf.type,
            **{name: getattr(vf, name) for name in _VERTICAL_FACTOR_OPTION_NAMES},
        }

    def factor(self, angle_degrees: float) -> float:
        vf = cast(_ResolvedVerticalFactor, self if self._is_resolved() else self.normalized())
        if vf.type == "none":
            return 1.0
        if (
            not math.isfinite(angle_degrees)
            or angle_degrees <= vf.low_cut_angle
            or angle_degrees >= vf.high_cut_angle
        ):
            return math.inf

        if vf.type == "binary":
            value = vf.zero_factor
        elif vf.type in {"linear", "inverse_linear"}:
            value = vf.zero_factor + vf.slope * angle_degrees
        elif vf.type in {"symmetric_linear", "symmetric_inverse_linear"}:
            value = vf.zero_factor + vf.slope * abs(angle_degrees)
        elif vf.type == "cos":
            value = math.cos(math.radians(angle_degrees)) ** vf.power
        elif vf.type == "sec":
            value = 1.0 / (math.cos(math.radians(angle_degrees)) ** vf.power)
        elif vf.type == "cos_sec":
            value = (
                math.cos(math.radians(angle_degrees)) ** vf.cos_power
                if angle_degrees < 0.0
                else 1.0 / (math.cos(math.radians(angle_degrees)) ** vf.sec_power)
            )
        elif vf.type == "sec_cos":
            value = (
                1.0 / (math.cos(math.radians(angle_degrees)) ** vf.sec_power)
                if angle_degrees < 0.0
                else math.cos(math.radians(angle_degrees)) ** vf.cos_power
            )
        elif vf.type == "hiking_time":
            value = _hiking_pace(angle_degrees)
        elif vf.type == "bidir_hiking_time":
            value = 0.5 * (_hiking_pace(angle_degrees) + _hiking_pace(-angle_degrees))
        else:
            raise ValueError(f"unsupported vertical factor: {vf.type}")

        return value if math.isfinite(value) and value > 0.0 else math.inf

    def _is_resolved(self) -> bool:
        return self.type in _DEFAULTS and all(
            getattr(self, name) is not None for name in _VERTICAL_FACTOR_OPTION_NAMES
        )


def _hiking_pace(angle_degrees: float) -> float:
    slope = math.tan(math.radians(angle_degrees))
    speed_km_per_hour = 6.0 * math.exp(-3.5 * abs(slope + 0.05))
    return 1.0 / (speed_km_per_hour * 1000.0)


@dataclass(frozen=True)
class RasterGrid:
    """Map raster cell coordinates into solver x/y coordinates."""

    cell_size: float | tuple[float, float] = 1.0
    origin: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class RasterSurface:
    """Solver-ready raster layers on one grid.

    `cost` is required. `elevation` and `barriers` must have the same shape when
    supplied. Non-finite cost/elevation cells are treated as blocked in addition
    to the explicit barrier mask.
    """

    cost: npt.ArrayLike
    grid: RasterGrid = field(default_factory=RasterGrid)
    elevation: npt.ArrayLike | None = None
    barriers: npt.ArrayLike | None = None


@dataclass
class PathTraceResult:
    """Path trace line plus fallback counters collected by the native tracer."""

    line: npt.NDArray[np.float64]
    metadata: dict[str, int]


@dataclass
class RouteLegResult:
    """Native route leg trace returned by a parallel batched solve."""

    line: npt.NDArray[np.float64]
    cost: float
    metadata: dict[str, int]


@dataclass
class DistanceAccumulationResult:
    distance: npt.NDArray[np.float64]
    _valid: npt.NDArray[np.bool_] = field(repr=False)
    _back_direction: npt.NDArray[np.float64] = field(repr=False)
    _parent_a: npt.NDArray[np.int64] = field(repr=False)
    _parent_b: npt.NDArray[np.int64] = field(repr=False)
    _parent_weight: npt.NDArray[np.float64] = field(repr=False)
    cell_size: tuple[float, float]
    origin: tuple[float, float]
    vertical_factor: VerticalFactor

    @overload
    def optimal_path_as_line(
        self,
        destination: Cell,
        *,
        max_steps: int | None = None,
    ) -> npt.NDArray[np.float64]: ...

    @overload
    def optimal_path_as_line(
        self,
        destination: Sequence[Cell],
        *,
        max_steps: int | None = None,
    ) -> list[npt.NDArray[np.float64]]: ...

    def optimal_path_as_line(
        self,
        destination: Cell | Sequence[Cell],
        *,
        max_steps: int | None = None,
    ) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
        return optimal_path_as_line(self, destination, max_steps=max_steps)

    @overload
    def optimal_path_trace(
        self,
        destination: Cell,
        *,
        max_steps: int | None = None,
    ) -> PathTraceResult: ...

    @overload
    def optimal_path_trace(
        self,
        destination: Sequence[Cell],
        *,
        max_steps: int | None = None,
    ) -> list[PathTraceResult]: ...

    def optimal_path_trace(
        self,
        destination: Cell | Sequence[Cell],
        *,
        max_steps: int | None = None,
    ) -> PathTraceResult | list[PathTraceResult]:
        return optimal_path_trace(self, destination, max_steps=max_steps)


def distance_accumulation(
    surface: RasterSurface | npt.ArrayLike,
    source: Cell | Sequence[Cell] | npt.ArrayLike,
    *,
    target: Cell | Sequence[Cell] | npt.ArrayLike | None = None,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    progress: ProgressOption = False,
    progress_interval: int | None = None,
) -> DistanceAccumulationResult:
    """Compute accumulated cost distance from one or more source cells.

    `surface` may be a `RasterSurface` or a plain 2D cost array. `source` is a
    `(row, col)` cell or an `(n, 2)` sequence of cells. The route-level API uses
    one source per leg; accepting multiple cells here keeps the native solver
    useful for accumulation/allocation workflows without exposing source rasters.
    When `target` is supplied, cells not needed to settle the requested target
    cell(s) may remain infinite in the returned distance raster.

    Set `progress=True` to show a `tqdm` progress bar, or pass a callback that
    accepts `(accepted_cells, total_valid_cells)`.
    """

    if not isinstance(surface, RasterSurface):
        surface = RasterSurface(surface)

    cost_arr = np.ascontiguousarray(np.asarray(surface.cost, dtype=np.float64))
    if cost_arr.ndim != 2:
        raise ValueError("surface cost must be a 2D array")

    rows, cols = cost_arr.shape
    elevation_arr = _optional_surface_array(surface.elevation, cost_arr.shape, "elevation")
    barrier_arr = _optional_barrier_array(surface.barriers, cost_arr.shape)
    source_cells = _normalize_source_cells(source, cost_arr.shape)
    target_cells = None if target is None else _normalize_target_cells(target, cost_arr.shape)
    valid_arr = _valid_surface_mask(cost_arr, elevation_arr, barrier_arr)

    cell_size_x, cell_size_y = _normalize_cell_size(surface.grid.cell_size)
    vf = VerticalFactor.from_any(vertical_factor)
    origin_x, origin_y = _normalize_origin(surface.grid.origin)
    callback, close_progress = _normalize_progress_callback(
        progress,
        total=None if target_cells is not None else int(np.count_nonzero(valid_arr)),
        label="distance_accumulation",
    )
    progress_interval_value = _normalize_progress_interval(
        progress_interval,
        int(np.count_nonzero(valid_arr)),
    )

    try:
        raw = _native.distance_accumulation(
            source_cells,
            cost_arr,
            elevation_arr,
            barrier_arr,
            vf.as_native(),
            cell_size_x,
            cell_size_y,
            target_cells,
            callback,
            progress_interval_value,
        )
    finally:
        if close_progress is not None:
            close_progress()

    return DistanceAccumulationResult(
        distance=raw["distance"],
        _valid=valid_arr,
        _back_direction=raw["back_direction"],
        _parent_a=raw["parent_a"],
        _parent_b=raw["parent_b"],
        _parent_weight=raw["parent_weight"],
        cell_size=(cell_size_x, cell_size_y),
        origin=(origin_x, origin_y),
        vertical_factor=vf,
    )


def route_legs(
    surface: RasterSurface | npt.ArrayLike,
    legs: npt.ArrayLike,
    *,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
) -> list[RouteLegResult]:
    """Solve independent source/destination legs on one shared raster surface.

    `legs` must have shape `(n, 4)` with columns
    `(source_row, source_col, destination_row, destination_col)`. The native
    implementation shares immutable raster layers across worker threads and
    uses Rayon to solve each leg in parallel.
    """

    if not isinstance(surface, RasterSurface):
        surface = RasterSurface(surface)

    cost_arr = np.ascontiguousarray(np.asarray(surface.cost, dtype=np.float64))
    if cost_arr.ndim != 2:
        raise ValueError("surface cost must be a 2D array")

    elevation_arr = _optional_surface_array(surface.elevation, cost_arr.shape, "elevation")
    barrier_arr = _optional_barrier_array(surface.barriers, cost_arr.shape)
    leg_arr = _normalize_leg_cells(legs, cost_arr.shape)
    cell_size_x, cell_size_y = _normalize_cell_size(surface.grid.cell_size)
    origin_x, origin_y = _normalize_origin(surface.grid.origin)
    vf = VerticalFactor.from_any(vertical_factor)

    raw_legs = _native.route_legs(
        leg_arr,
        cost_arr,
        elevation_arr,
        barrier_arr,
        vf.as_native(),
        cell_size_x,
        cell_size_y,
    )
    legs_out = []
    for raw in raw_legs:
        line = np.asarray(raw["line"], dtype=np.float64).copy()
        line[:, 0] += origin_x
        line[:, 1] += origin_y
        legs_out.append(
            RouteLegResult(
                line=line,
                cost=float(raw["cost"]),
                metadata={key: int(value) for key, value in raw["metadata"].items()},
            )
        )
    return legs_out


def route_legs_windowed(
    surface: RasterSurface | npt.ArrayLike,
    leg_windows: npt.ArrayLike,
    *,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
) -> list[RouteLegResult]:
    """Solve independent route legs using copied per-leg windows.

    `leg_windows` must have shape `(n, 8)` with columns
    `(source_row, source_col, destination_row, destination_col, row_min, row_max,
    col_min, col_max)`, where bounds are exclusive. The native implementation
    shares the full input raster, copies each window inside a Rayon worker, and
    solves only that local crop.
    """

    if not isinstance(surface, RasterSurface):
        surface = RasterSurface(surface)

    cost_arr = np.ascontiguousarray(np.asarray(surface.cost, dtype=np.float64))
    if cost_arr.ndim != 2:
        raise ValueError("surface cost must be a 2D array")

    elevation_arr = _optional_surface_array(surface.elevation, cost_arr.shape, "elevation")
    barrier_arr = _optional_barrier_array(surface.barriers, cost_arr.shape)
    window_arr = _normalize_leg_windows(leg_windows, cost_arr.shape)
    cell_size_x, cell_size_y = _normalize_cell_size(surface.grid.cell_size)
    origin_x, origin_y = _normalize_origin(surface.grid.origin)
    vf = VerticalFactor.from_any(vertical_factor)

    raw_legs = _native.route_legs_windowed(
        window_arr,
        cost_arr,
        elevation_arr,
        barrier_arr,
        vf.as_native(),
        cell_size_x,
        cell_size_y,
    )
    legs_out = []
    for raw in raw_legs:
        line = np.asarray(raw["line"], dtype=np.float64).copy()
        line[:, 0] += origin_x
        line[:, 1] += origin_y
        legs_out.append(
            RouteLegResult(
                line=line,
                cost=float(raw["cost"]),
                metadata={key: int(value) for key, value in raw["metadata"].items()},
            )
        )
    return legs_out


def evaluate_path_cost(
    surface: RasterSurface | npt.ArrayLike,
    line_xy: Any,
    *,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    max_step: float | None = None,
) -> float:
    """Evaluate a directional x/y path against a distance-rs raster surface.

    The line is evaluated in the order supplied. Each segment is charged with
    the same local rule used by the solver for point-to-point movement:
    surface distance times the destination cell's cost times the vertical
    factor. Dijkstra traces from `trace_raster_path` are destination-to-source,
    so reverse them before evaluating source-to-destination travel.

    `line_xy` may be an ``(n, 2)`` coordinate array, a GeoJSON-like LineString
    mapping, or an object with a Shapely-style ``.coords`` attribute. Coordinates
    must use the solver x/y grid for the supplied `RasterSurface`.
    """

    if not isinstance(surface, RasterSurface):
        surface = RasterSurface(surface)

    cost_arr = np.ascontiguousarray(np.asarray(surface.cost, dtype=np.float64))
    if cost_arr.ndim != 2:
        raise ValueError("surface cost must be a 2D array")

    elevation_arr = _optional_surface_array(surface.elevation, cost_arr.shape, "elevation")
    barrier_arr = _optional_barrier_array(surface.barriers, cost_arr.shape)
    valid_arr = _valid_surface_mask(cost_arr, elevation_arr, barrier_arr)
    cell_size_x, cell_size_y = _normalize_cell_size(surface.grid.cell_size)
    origin_x, origin_y = _normalize_origin(surface.grid.origin)
    vf = VerticalFactor.from_any(vertical_factor)
    line = _normalize_line_xy(line_xy)
    step_limit = _normalize_max_step(max_step)

    total = 0.0
    for start_xy, end_xy in zip(line[:-1], line[1:]):
        total += _evaluate_path_segment(
            start_xy,
            end_xy,
            cost=cost_arr,
            elevation=elevation_arr,
            valid=valid_arr,
            vertical_factor=vf,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            origin_x=origin_x,
            origin_y=origin_y,
            max_step=step_limit,
        )
        if not math.isfinite(total):
            return math.inf
    return float(total)


def _normalize_progress_callback(
    progress: ProgressOption,
    *,
    total: int | None,
    label: str,
) -> tuple[ProgressCallback | None, Callable[[], None] | None]:
    if progress is None or progress is False:
        return None, None
    if progress is True:
        from tqdm.auto import tqdm

        bar = tqdm(total=total, desc=label, unit="cell")
        last_accepted = 0

        def callback(accepted: int, _total: int) -> None:
            nonlocal last_accepted
            if accepted > last_accepted:
                bar.update(accepted - last_accepted)
                last_accepted = accepted

        return callback, bar.close
    if callable(progress):
        return progress, None
    raise TypeError("progress must be a bool, a callback, or None")


def _normalize_progress_interval(value: int | None, total: int) -> int:
    if value is None:
        return max(1, total // 200)
    interval = int(value)
    if interval < 1:
        raise ValueError("progress_interval must be at least 1")
    return interval


def _optional_surface_array(
    value: npt.ArrayLike | None,
    shape: tuple[int, int],
    name: str,
) -> npt.NDArray[np.float64] | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if array.shape != shape:
        raise ValueError(f"{name} must have the same shape as cost")
    return array


def _optional_barrier_array(
    value: npt.ArrayLike | None,
    shape: tuple[int, int],
) -> npt.NDArray[np.bool_] | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value, dtype=np.bool_))
    if array.shape != shape:
        raise ValueError("barriers must have the same shape as cost")
    return array


def _normalize_source_cells(
    source: Cell | Sequence[Cell] | npt.ArrayLike,
    shape: tuple[int, int],
) -> npt.NDArray[np.int64]:
    return _normalize_cells(source, shape, name="source")


def _normalize_target_cells(
    target: Cell | Sequence[Cell] | npt.ArrayLike,
    shape: tuple[int, int],
) -> npt.NDArray[np.int64]:
    return _normalize_cells(target, shape, name="target")


def _normalize_leg_cells(
    value: npt.ArrayLike,
    shape: tuple[int, int],
) -> npt.NDArray[np.int64]:
    cells_float = np.asarray(value, dtype=np.float64)
    if cells_float.ndim != 2 or cells_float.shape[1] != 4:
        raise ValueError(
            "legs must have shape (n, 4): source_row, source_col, destination_row, destination_col"
        )
    if cells_float.shape[0] == 0:
        raise ValueError("at least one route leg is required")
    if not np.all(np.isfinite(cells_float)):
        raise ValueError("route leg cells must be finite")

    rounded = np.rint(cells_float)
    if not np.array_equal(cells_float, rounded):
        raise ValueError("route leg cells must be integer row/col coordinates")
    cells = np.ascontiguousarray(rounded.astype(np.int64))

    rows, cols = shape
    row_cols = cells.reshape(-1, 2)
    if (
        np.any(row_cols[:, 0] < 0)
        or np.any(row_cols[:, 1] < 0)
        or np.any(row_cols[:, 0] >= rows)
        or np.any(row_cols[:, 1] >= cols)
    ):
        raise ValueError("route leg cell is outside the raster")
    return cells


def _normalize_leg_windows(
    value: npt.ArrayLike,
    shape: tuple[int, int],
) -> npt.NDArray[np.int64]:
    windows_float = np.asarray(value, dtype=np.float64)
    if windows_float.ndim != 2 or windows_float.shape[1] != 8:
        raise ValueError(
            "leg_windows must have shape (n, 8): source_row, source_col, "
            "destination_row, destination_col, row_min, row_max, col_min, col_max"
        )
    if windows_float.shape[0] == 0:
        raise ValueError("at least one route leg window is required")
    if not np.all(np.isfinite(windows_float)):
        raise ValueError("route leg windows must be finite")

    rounded = np.rint(windows_float)
    if not np.array_equal(windows_float, rounded):
        raise ValueError("route leg windows must use integer row/col coordinates")
    windows = np.ascontiguousarray(rounded.astype(np.int64))

    rows, cols = shape
    row_min = windows[:, 4]
    row_max = windows[:, 5]
    col_min = windows[:, 6]
    col_max = windows[:, 7]
    if (
        np.any(row_min < 0)
        or np.any(col_min < 0)
        or np.any(row_max <= row_min)
        or np.any(col_max <= col_min)
        or np.any(row_max > rows)
        or np.any(col_max > cols)
    ):
        raise ValueError("route leg window is outside the raster")
    for row_index, col_index in ((0, 1), (2, 3)):
        row = windows[:, row_index]
        col = windows[:, col_index]
        if (
            np.any(row < row_min)
            or np.any(row >= row_max)
            or np.any(col < col_min)
            or np.any(col >= col_max)
        ):
            raise ValueError("route leg endpoint is outside its window")
    return windows


def _normalize_cells(
    value: Cell | Sequence[Cell] | npt.ArrayLike,
    shape: tuple[int, int],
    *,
    name: str,
) -> npt.NDArray[np.int64]:
    cells_float = np.asarray(value, dtype=np.float64)
    if cells_float.ndim == 1 and cells_float.shape == (2,):
        cells_float = cells_float.reshape(1, 2)
    if cells_float.ndim != 2 or cells_float.shape[1] != 2:
        raise ValueError(f"{name} must be a (row, col) cell or an (n, 2) cell array")
    if cells_float.shape[0] == 0:
        raise ValueError(f"at least one {name} cell is required")
    if not np.all(np.isfinite(cells_float)):
        raise ValueError(f"{name} cells must be finite")

    rounded = np.rint(cells_float)
    if not np.array_equal(cells_float, rounded):
        raise ValueError(f"{name} cells must be integer row/col coordinates")
    cells = np.ascontiguousarray(rounded.astype(np.int64))

    rows, cols = shape
    if (
        np.any(cells[:, 0] < 0)
        or np.any(cells[:, 1] < 0)
        or np.any(cells[:, 0] >= rows)
        or np.any(cells[:, 1] >= cols)
    ):
        raise ValueError(f"{name} cell is outside the raster")
    return cells


def _valid_surface_mask(
    cost: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64] | None,
    barriers: npt.NDArray[np.bool_] | None,
) -> npt.NDArray[np.bool_]:
    valid = np.isfinite(cost)
    if elevation is not None:
        valid = valid & np.isfinite(elevation)
    if barriers is not None:
        valid = valid & ~barriers
    return np.ascontiguousarray(valid, dtype=np.bool_)


def _normalize_line_xy(line_xy: Any) -> npt.NDArray[np.float64]:
    if isinstance(line_xy, Mapping):
        if line_xy.get("type") != "LineString":
            raise ValueError("line_xy mapping must be a GeoJSON-like LineString")
        value = line_xy.get("coordinates")
    elif hasattr(line_xy, "coords"):
        value = line_xy.coords
    else:
        value = line_xy

    line = np.asarray(value, dtype=np.float64)
    if line.ndim != 2 or line.shape[1] != 2:
        raise ValueError("line_xy must have shape (n, 2)")
    if len(line) == 0:
        raise ValueError("line_xy must contain at least one point")
    if not np.all(np.isfinite(line)):
        raise ValueError("line_xy coordinates must be finite")
    return np.ascontiguousarray(line, dtype=np.float64)


def _normalize_max_step(max_step: float | None) -> float | None:
    if max_step is None:
        return None
    value = float(max_step)
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("max_step must be a positive finite value")
    return value


def _evaluate_path_segment(
    start_xy: npt.NDArray[np.float64],
    end_xy: npt.NDArray[np.float64],
    *,
    cost: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64] | None,
    valid: npt.NDArray[np.bool_],
    vertical_factor: VerticalFactor,
    cell_size_x: float,
    cell_size_y: float,
    origin_x: float,
    origin_y: float,
    max_step: float | None,
) -> float:
    delta = end_xy - start_xy
    plan_distance = float(math.hypot(float(delta[0]), float(delta[1])))
    if plan_distance <= 0.0:
        return 0.0

    steps = 1 if max_step is None else max(1, math.ceil(plan_distance / max_step))
    previous = start_xy
    subtotal = 0.0
    for step in range(1, steps + 1):
        current = start_xy + delta * (step / steps)
        subtotal += _evaluate_path_step(
            previous,
            current,
            cost=cost,
            elevation=elevation,
            valid=valid,
            vertical_factor=vertical_factor,
            cell_size_x=cell_size_x,
            cell_size_y=cell_size_y,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        if not math.isfinite(subtotal):
            return math.inf
        previous = current
    return subtotal


def _evaluate_path_step(
    start_xy: npt.NDArray[np.float64],
    end_xy: npt.NDArray[np.float64],
    *,
    cost: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64] | None,
    valid: npt.NDArray[np.bool_],
    vertical_factor: VerticalFactor,
    cell_size_x: float,
    cell_size_y: float,
    origin_x: float,
    origin_y: float,
) -> float:
    start_cell = _line_point_cell(
        start_xy,
        shape=cost.shape,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    end_cell = _line_point_cell(
        end_xy,
        shape=cost.shape,
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if start_cell is None or end_cell is None:
        return math.inf
    if not valid[start_cell] or not valid[end_cell]:
        return math.inf

    dx = float(end_xy[0] - start_xy[0])
    dy = float(end_xy[1] - start_xy[1])
    plan_distance = math.hypot(dx, dy)
    if plan_distance <= 0.0:
        return 0.0

    dz = 0.0
    if elevation is not None:
        dz = float(elevation[end_cell] - elevation[start_cell])
    factor = vertical_factor.factor(math.degrees(math.atan2(dz, plan_distance)))
    if not math.isfinite(factor):
        return math.inf

    surface_distance = math.hypot(plan_distance, dz) if elevation is not None else plan_distance
    return surface_distance * float(cost[end_cell]) * factor


def _line_point_cell(
    xy: npt.NDArray[np.float64],
    *,
    shape: tuple[int, int],
    cell_size_x: float,
    cell_size_y: float,
    origin_x: float,
    origin_y: float,
) -> tuple[int, int] | None:
    col = math.floor((float(xy[0]) - origin_x) / cell_size_x + 0.5)
    row = math.floor((float(xy[1]) - origin_y) / cell_size_y + 0.5)
    rows, cols = shape
    if row < 0 or col < 0 or row >= rows or col >= cols:
        return None
    return int(row), int(col)


@overload
def optimal_path_as_line(
    result: DistanceAccumulationResult,
    destination: Cell,
    *,
    max_steps: int | None = None,
) -> npt.NDArray[np.float64]: ...


@overload
def optimal_path_as_line(
    result: DistanceAccumulationResult,
    destination: Sequence[Cell],
    *,
    max_steps: int | None = None,
) -> list[npt.NDArray[np.float64]]: ...


def optimal_path_as_line(
    result: DistanceAccumulationResult,
    destination: Cell | Sequence[Cell],
    *,
    max_steps: int | None = None,
) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
    """Trace one or more optimal paths as x/y line coordinates."""

    if not isinstance(result, DistanceAccumulationResult):
        raise TypeError("result must be a DistanceAccumulationResult")

    if _is_one_destination(destination):
        row, col = destination
        max_steps_value = _normalize_max_steps(max_steps)
        return _native.optimal_path_as_line(
            result.distance,
            result._valid,
            result._back_direction,
            result._parent_a,
            result._parent_b,
            result._parent_weight,
            int(row),
            int(col),
            result.cell_size[0],
            result.cell_size[1],
            result.origin[0],
            result.origin[1],
            max_steps_value,
        )

    destinations = cast(Sequence[Cell], destination)
    return [
        optimal_path_as_line(result, one_destination, max_steps=max_steps)
        for one_destination in destinations
    ]


@overload
def optimal_path_trace(
    result: DistanceAccumulationResult,
    destination: Cell,
    *,
    max_steps: int | None = None,
) -> PathTraceResult: ...


@overload
def optimal_path_trace(
    result: DistanceAccumulationResult,
    destination: Sequence[Cell],
    *,
    max_steps: int | None = None,
) -> list[PathTraceResult]: ...


def optimal_path_trace(
    result: DistanceAccumulationResult,
    destination: Cell | Sequence[Cell],
    *,
    max_steps: int | None = None,
) -> PathTraceResult | list[PathTraceResult]:
    """Trace one or more paths and return fallback counters from the native tracer.

    Metadata keys:
    - `direction_steps`: ordinary back-direction lattice steps.
    - `parent_lattice_fallbacks`: steps that used the stored parent direction.
    - `proposed_cell_center_fallbacks`: accepted by snapping to the proposed cell center.
    - `current_cell_center_fallbacks`: accepted by recentering inside the current cell.
    - `direct_parent_point_fallbacks`: accepted by moving directly to the stored parent point.
    - `non_descending_rejections`: candidate steps rejected because they did not reduce distance.
    - `total_fallbacks`: sum of the fallback counters.
    """

    if not isinstance(result, DistanceAccumulationResult):
        raise TypeError("result must be a DistanceAccumulationResult")

    if _is_one_destination(destination):
        row, col = destination
        max_steps_value = _normalize_max_steps(max_steps)
        raw = _native.optimal_path_trace(
            result.distance,
            result._valid,
            result._back_direction,
            result._parent_a,
            result._parent_b,
            result._parent_weight,
            int(row),
            int(col),
            result.cell_size[0],
            result.cell_size[1],
            result.origin[0],
            result.origin[1],
            max_steps_value,
        )
        return PathTraceResult(
            line=raw["line"],
            metadata={key: int(value) for key, value in raw["metadata"].items()},
        )

    destinations = cast(Sequence[Cell], destination)
    return [
        optimal_path_trace(result, one_destination, max_steps=max_steps)
        for one_destination in destinations
    ]


def _normalize_cell_size(cell_size: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(cell_size, tuple):
        if len(cell_size) != 2:
            raise ValueError("cell_size tuple must be (x_size, y_size)")
        x_size = float(cell_size[0])
        y_size = float(cell_size[1])
    else:
        x_size = float(cell_size)
        y_size = float(cell_size)
    if x_size <= 0.0 or y_size <= 0.0 or not math.isfinite(x_size) or not math.isfinite(y_size):
        raise ValueError("cell_size values must be positive finite values")
    return x_size, y_size


def _normalize_origin(origin: Sequence[float]) -> tuple[float, float]:
    try:
        origin_len = len(origin)
    except TypeError as exc:
        raise ValueError("origin must be an (x, y) pair") from exc
    if origin_len != 2:
        raise ValueError("origin must be an (x, y) pair")
    origin_x = float(origin[0])
    origin_y = float(origin[1])
    if not math.isfinite(origin_x) or not math.isfinite(origin_y):
        raise ValueError("origin values must be finite")
    return origin_x, origin_y


def _normalize_max_steps(max_steps: int | None) -> int:
    if max_steps is None:
        return 0
    value = int(max_steps)
    if value <= 0:
        raise ValueError("max_steps must be positive")
    return value


def _is_one_destination(value: object) -> TypeGuard[Cell]:
    if not isinstance(value, Sequence) or len(value) != 2:  # type: ignore[arg-type]
        return False
    first = value[0]  # type: ignore[index]
    return not isinstance(first, Sequence)
