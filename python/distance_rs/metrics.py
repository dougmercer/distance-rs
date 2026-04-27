"""Path measurement helpers for geospatial routes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy.typing as npt

from ._distance import VerticalFactor
from ._geo import (
    GeoSurface,
    PathMetrics,
    _combine_metrics,
    _line_coordinates,
    _path_metrics,
)

__all__ = ["combine_path_metrics", "path_metrics"]


def path_metrics(
    path_xy: npt.ArrayLike,
    *,
    cost: float,
    surface: GeoSurface,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    baseline_speed: float = 5.0,
) -> PathMetrics:
    """Measure a map-coordinate path against a loaded geospatial surface."""

    baseline_speed_value = float(baseline_speed)
    if baseline_speed_value <= 0.0 or not math.isfinite(baseline_speed_value):
        raise ValueError("baseline_speed must be a positive finite km/hr value")
    return _path_metrics(
        _line_coordinates(path_xy),
        cost=float(cost),
        geo=surface,
        vertical_factor=VerticalFactor.from_any(vertical_factor),
        baseline_speed_kmh=baseline_speed_value,
    )


def combine_path_metrics(first: PathMetrics, second: PathMetrics) -> PathMetrics:
    """Combine metrics for adjacent route legs."""

    return _combine_metrics(first, second)
