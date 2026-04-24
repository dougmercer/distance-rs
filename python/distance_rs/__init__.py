"""Raster distance accumulation with local Eikonal updates."""

from importlib import import_module
from typing import Any

from ._distance import (
    DistanceAccumulationResult,
    RasterGrid,
    RasterSurface,
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
    "CostRaster",
    "ElevationRaster",
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
}

__all__ = [
    "DistanceAccumulationResult",
    "RasterGrid",
    "RasterSurface",
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
