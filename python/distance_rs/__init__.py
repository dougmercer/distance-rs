"""Raster distance accumulation with local Eikonal updates."""

from ._distance import *  # noqa: F403

try:
    from ._geo import *  # noqa: F403
except ModuleNotFoundError as exc:
    if exc.name not in {"fiona", "rasterio", "shapely"}:
        raise
