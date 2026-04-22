"""Ordered-upwind raster distance accumulation."""

from importlib import import_module
from typing import Any

from ._distance import (
    DistanceAccumulationResult,
    VerticalFactor,
    distance_accumulation,
    optimal_path_as_line,
)
from .baselines import (
    RasterDijkstraResult,
    raster_dijkstra,
    raster_dijkstra_baseline,
    trace_raster_path,
)

_GEO_EXPORTS = {
    "GeoDistanceAccumulationResult",
    "GeoRasterData",
    "OptimalPathLeg",
    "OptimalPathResult",
    "PathMetrics",
    "compute_optimal_path",
    "geo_distance_accumulation",
    "geo_optimal_path_as_line",
    "prepare_geo_inputs",
}

__all__ = [
    "DistanceAccumulationResult",
    "RasterDijkstraResult",
    "VerticalFactor",
    "distance_accumulation",
    "optimal_path_as_line",
    "raster_dijkstra",
    "raster_dijkstra_baseline",
    "trace_raster_path",
]


def __getattr__(name: str) -> Any:
    if name not in _GEO_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        module = import_module("._geo", __name__)
    except ModuleNotFoundError as exc:
        if exc.name in {"fiona", "rasterio", "shapely"}:
            raise ImportError(
                "Geo helpers require the optional geo dependencies; install "
                "`distance-rs[geo]` or run `uv sync --extra geo`."
            ) from exc
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*__all__, *_GEO_EXPORTS])
