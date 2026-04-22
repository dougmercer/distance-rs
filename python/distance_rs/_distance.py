from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from . import _native


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
            valid_keys = {
                "zero_factor",
                "low_cut_angle",
                "high_cut_angle",
                "slope",
                "power",
                "cos_power",
                "sec_power",
            }
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
            "zero_factor": defaults["zero_factor"]
            if self.zero_factor is None
            else float(self.zero_factor),
            "low_cut_angle": defaults["low_cut_angle"]
            if self.low_cut_angle is None
            else float(self.low_cut_angle),
            "high_cut_angle": defaults["high_cut_angle"]
            if self.high_cut_angle is None
            else float(self.high_cut_angle),
            "slope": defaults["slope"] if self.slope is None else float(self.slope),
            "power": defaults["power"] if self.power is None else float(self.power),
            "cos_power": defaults["cos_power"] if self.cos_power is None else float(self.cos_power),
            "sec_power": defaults["sec_power"] if self.sec_power is None else float(self.sec_power),
        }
        for option_name, option_value in options.items():
            if not math.isfinite(option_value):
                raise ValueError(f"vertical factor option {option_name} must be finite")
        if options["low_cut_angle"] >= options["high_cut_angle"]:
            raise ValueError("low_cut_angle must be less than high_cut_angle")
        return options


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
    search_radius: float

    def optimal_path_as_line(
        self,
        destination: tuple[int, int] | Sequence[tuple[int, int]],
        *,
        max_steps: int | None = None,
    ) -> npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]:
        return optimal_path_as_line(self, destination, max_steps=max_steps)


def distance_accumulation(
    sources: npt.ArrayLike,
    *,
    cost_surface: npt.ArrayLike | None = None,
    elevation: npt.ArrayLike | None = None,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    barriers: npt.ArrayLike | None = None,
    cell_size: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    search_radius: float | None = None,
    use_surface_distance: bool = True,
) -> DistanceAccumulationResult:
    """Compute accumulated cost distance from source cells.

    `sources`, `cost_surface`, `elevation`, and `barriers` are two-dimensional
    rasters. Nonzero finite source values are treated as source cells. Non-finite
    cost/elevation values are treated as blocked cells.
    """

    source_arr = np.ascontiguousarray(np.asarray(sources, dtype=np.float64))
    if source_arr.ndim != 2:
        raise ValueError("sources must be a 2D array")

    rows, cols = source_arr.shape
    cost_arr = (
        np.ones((rows, cols), dtype=np.float64)
        if cost_surface is None
        else np.ascontiguousarray(np.asarray(cost_surface, dtype=np.float64))
    )
    if cost_arr.shape != source_arr.shape:
        raise ValueError("cost_surface must have the same shape as sources")

    has_elevation = elevation is not None
    elevation_arr = (
        None
        if elevation is None
        else np.ascontiguousarray(np.asarray(elevation, dtype=np.float64))
    )
    if elevation_arr is not None and elevation_arr.shape != source_arr.shape:
        raise ValueError("elevation must have the same shape as sources")

    barrier_arr = (
        None
        if barriers is None
        else np.ascontiguousarray(np.asarray(barriers, dtype=np.bool_))
    )
    if barrier_arr is not None and barrier_arr.shape != source_arr.shape:
        raise ValueError("barriers must have the same shape as sources")

    cell_size_x, cell_size_y = _normalize_cell_size(cell_size)
    if search_radius is None:
        search_radius_value = 4.0 * max(cell_size_x, cell_size_y)
    else:
        search_radius_value = float(search_radius)
    if search_radius_value <= 0.0 or not math.isfinite(search_radius_value):
        raise ValueError("search_radius must be a positive finite value")

    vf = VerticalFactor.from_any(vertical_factor)
    origin_x, origin_y = _normalize_origin(origin)

    raw = _native.distance_accumulation(
        source_arr,
        cost_arr,
        elevation_arr,
        barrier_arr,
        has_elevation,
        use_surface_distance,
        vf.type,
        vf.zero_factor,
        vf.low_cut_angle,
        vf.high_cut_angle,
        vf.slope,
        vf.power,
        vf.cos_power,
        vf.sec_power,
        cell_size_x,
        cell_size_y,
        search_radius_value,
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
        search_radius=search_radius_value,
    )


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
