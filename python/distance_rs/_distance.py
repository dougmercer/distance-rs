from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from . import _native


Cell = tuple[int, int]


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
        vf = self if self._is_resolved() else self.normalized()
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


@dataclass(frozen=True)
class SolverOptions:
    """Numerical options for distance accumulation."""

    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None
    use_surface_distance: bool = True


@dataclass
class DistanceAccumulationResult:
    distance: npt.NDArray[np.float64]
    back_direction: npt.NDArray[np.float64]
    parent_a: npt.NDArray[np.int64]
    parent_b: npt.NDArray[np.int64]
    parent_weight: npt.NDArray[np.float64]
    cell_size: tuple[float, float]
    origin: tuple[float, float]
    vertical_factor: VerticalFactor

    def optimal_path_as_line(
        self,
        destination: tuple[int, int] | Sequence[tuple[int, int]],
        *,
        max_steps: int | None = None,
    ) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
        return optimal_path_as_line(self, destination, max_steps=max_steps)


def distance_accumulation(
    surface: RasterSurface | npt.ArrayLike,
    source: Cell | Sequence[Cell] | npt.ArrayLike,
    *,
    options: SolverOptions | None = None,
) -> DistanceAccumulationResult:
    """Compute accumulated cost distance from one or more source cells.

    `surface` may be a `RasterSurface` or a plain 2D cost array. `source` is a
    `(row, col)` cell or an `(n, 2)` sequence of cells. The route-level API uses
    one source per leg; accepting multiple cells here keeps the native solver
    useful for accumulation/allocation workflows without exposing source rasters.
    """

    if not isinstance(surface, RasterSurface):
        surface = RasterSurface(surface)
    if options is None:
        options = SolverOptions()
    if not isinstance(options, SolverOptions):
        raise TypeError("options must be a SolverOptions instance")

    cost_arr = np.ascontiguousarray(np.asarray(surface.cost, dtype=np.float64))
    if cost_arr.ndim != 2:
        raise ValueError("surface cost must be a 2D array")

    rows, cols = cost_arr.shape
    elevation_arr = _optional_surface_array(surface.elevation, cost_arr.shape, "elevation")
    barrier_arr = _optional_barrier_array(surface.barriers, cost_arr.shape)
    source_cells = _normalize_source_cells(source, cost_arr.shape)

    cell_size_x, cell_size_y = _normalize_cell_size(surface.grid.cell_size)
    vf = VerticalFactor.from_any(options.vertical_factor)
    origin_x, origin_y = _normalize_origin(surface.grid.origin)

    raw = _native.distance_accumulation(
        source_cells,
        cost_arr,
        elevation_arr,
        barrier_arr,
        options.use_surface_distance,
        vf.as_native(),
        cell_size_x,
        cell_size_y,
    )

    return DistanceAccumulationResult(
        distance=raw["distance"],
        back_direction=raw["back_direction"],
        parent_a=raw["parent_a"],
        parent_b=raw["parent_b"],
        parent_weight=raw["parent_weight"],
        cell_size=(cell_size_x, cell_size_y),
        origin=(origin_x, origin_y),
        vertical_factor=vf,
    )


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
    cells_float = np.asarray(source, dtype=np.float64)
    if cells_float.ndim == 1 and cells_float.shape == (2,):
        cells_float = cells_float.reshape(1, 2)
    if cells_float.ndim != 2 or cells_float.shape[1] != 2:
        raise ValueError("source must be a (row, col) cell or an (n, 2) cell array")
    if cells_float.shape[0] == 0:
        raise ValueError("at least one source cell is required")
    if not np.all(np.isfinite(cells_float)):
        raise ValueError("source cells must be finite")

    rounded = np.rint(cells_float)
    if not np.array_equal(cells_float, rounded):
        raise ValueError("source cells must be integer row/col coordinates")
    cells = np.ascontiguousarray(rounded.astype(np.int64))

    rows, cols = shape
    if (
        np.any(cells[:, 0] < 0)
        or np.any(cells[:, 1] < 0)
        or np.any(cells[:, 0] >= rows)
        or np.any(cells[:, 1] >= cols)
    ):
        raise ValueError("source cell is outside the raster")
    return cells


def optimal_path_as_line(
    result: DistanceAccumulationResult,
    destination: tuple[int, int] | Sequence[tuple[int, int]],
    *,
    max_steps: int | None = None,
) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
    """Trace one or more optimal paths as x/y line coordinates."""

    if not isinstance(result, DistanceAccumulationResult):
        raise TypeError("result must be a DistanceAccumulationResult")

    if _is_one_destination(destination):
        row, col = destination  # type: ignore[misc]
        max_steps_value = _normalize_max_steps(max_steps)
        return _native.optimal_path_as_line(
            result.distance,
            result.back_direction,
            result.parent_a,
            result.parent_b,
            result.parent_weight,
            int(row),
            int(col),
            result.cell_size[0],
            result.cell_size[1],
            result.origin[0],
            result.origin[1],
            max_steps_value,
        )

    return [
        optimal_path_as_line(result, one_destination, max_steps=max_steps)
        for one_destination in destination  # type: ignore[union-attr]
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


def _is_one_destination(value: object) -> bool:
    if not isinstance(value, Sequence) or len(value) != 2:  # type: ignore[arg-type]
        return False
    first = value[0]  # type: ignore[index]
    return not isinstance(first, Sequence)
