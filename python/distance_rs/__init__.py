"""Ordered-upwind raster distance accumulation."""

from ._distance import (
    DistanceAccumulationResult,
    VerticalFactor,
    distance_accumulation,
    optimal_path_as_line,
)
from ._geo import (
    GeoDistanceAccumulationResult,
    GeoRasterData,
    OptimalPathLeg,
    OptimalPathResult,
    PathMetrics,
    compute_optimal_path,
    geo_distance_accumulation,
    geo_optimal_path_as_line,
    prepare_geo_inputs,
)
from .baselines import (
    RasterDijkstraResult,
    raster_dijkstra,
    raster_dijkstra_baseline,
    trace_raster_path,
)

__all__ = [
    "DistanceAccumulationResult",
    "GeoDistanceAccumulationResult",
    "GeoRasterData",
    "OptimalPathLeg",
    "OptimalPathResult",
    "PathMetrics",
    "RasterDijkstraResult",
    "VerticalFactor",
    "compute_optimal_path",
    "distance_accumulation",
    "geo_distance_accumulation",
    "geo_optimal_path_as_line",
    "optimal_path_as_line",
    "prepare_geo_inputs",
    "raster_dijkstra",
    "raster_dijkstra_baseline",
    "trace_raster_path",
]
