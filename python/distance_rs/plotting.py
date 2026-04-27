"""Matplotlib plotting helpers for geospatial route examples."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import rasterio as rio

from ._geo import (
    XY,
    Bounds,
    CostRaster,
    GridSpec,
    _cost_spec,
    _line_coordinates,
    _load_geometries,
    _target_bounds,
    _target_crs,
    _target_resolution,
    load_points,
    load_surface,
)

__all__ = ["plot_route_map"]


def plot_route_map(
    path: str | Path,
    cost: CostRaster | str | Path,
    *,
    elevation: str | Path | None = None,
    barriers: Any | None = None,
    waypoints: Any | None = None,
    routes: Any | None = None,
    grid: GridSpec | None = None,
    bounds: Bounds | None = None,
    margin: float = 250.0,
    max_pixels: int = 2_500_000,
    land_use_labels: Mapping[float | int, str] | None = None,
    land_use_colors: Mapping[float | int, str] | Sequence[str] | None = None,
    title: str | None = None,
) -> None:
    """Render a compact terrain/land-use route map from GeoTIFF and vector inputs.

    `routes` may be a route result, an ``(n, 2)`` coordinate array, a GeoJSON-like
    line, a sequence of those, or a mapping of legend label to route.
    """

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for plotting; run with `uv run --group plot ...`") from exc

    cost_spec = _cost_spec(cost)
    grid_spec = grid or GridSpec()
    with rio.open(cost_spec.path) as cost_src:
        target_crs = _target_crs(cost_src, grid_spec.crs, name="cost raster")
        source_resolution = _target_resolution(cost_src, grid_spec.resolution)
        base_bounds = _target_bounds(
            cost_src,
            elevation_path=Path(elevation) if elevation is not None else None,
            target_crs=target_crs,
        )

    route_layers = _plot_route_layers(routes)
    waypoint_xy = (
        tuple(_waypoints_from_routes(route_layers))
        if waypoints is None
        else load_points(waypoints, target_crs=target_crs)
    )
    barrier_geometries = (
        [] if barriers is None else _load_geometries(barriers, target_crs=target_crs, source_crs=None, layer=None)
    )
    plot_bounds = (
        bounds
        or grid_spec.bounds
        or _plot_bounds(
            base_bounds,
            waypoints=waypoint_xy,
            barriers=barrier_geometries,
            routes=route_layers,
            margin=margin,
        )
    )
    plot_resolution = grid_spec.resolution or _plot_resolution(
        plot_bounds,
        source_resolution,
        max_pixels=max_pixels,
    )
    geo = load_surface(
        cost_spec,
        elevation=elevation,
        barriers=barriers,
        grid=replace(grid_spec, crs=target_crs, bounds=plot_bounds, resolution=plot_resolution),
    )

    has_elevation = geo.surface.elevation is not None and np.isfinite(geo.surface.elevation).any()
    fig, axes = plt.subplots(
        1,
        2 if has_elevation else 1,
        figsize=(15 if has_elevation else 8, 7),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axis_list = list(np.atleast_1d(axes))
    extent = _imshow_extent(geo.grid.bounds)

    if has_elevation:
        terrain_ax = axis_list[0]
        assert geo.surface.elevation is not None
        terrain_ax.imshow(
            LightSource(azdeg=315, altdeg=45).shade(
                np.asarray(geo.surface.elevation, dtype=np.float64),
                cmap=plt.get_cmap("terrain"),
                blend_mode="overlay",
                vert_exag=0.55,
            ),
            extent=extent,
            origin="upper",
            interpolation="bilinear",
        )
        terrain_ax.set_title(title or "Terrain and Routes")
        land_ax = axis_list[1]
    else:
        land_ax = axis_list[0]

    land_image, land_handles = _land_use_image(
        land_ax,
        geo.land_use,
        extent=extent,
        labels=land_use_labels,
        colors=land_use_colors,
    )
    land_ax.set_title("Land Use and Routes" if has_elevation else (title or "Routes"))

    route_handles = _draw_plot_overlays(
        axis_list,
        route_layers=route_layers,
        waypoints=waypoint_xy,
        barriers=barrier_geometries,
    )
    land_legend_handles = land_handles + route_handles
    if route_handles and has_elevation:
        axis_list[0].legend(handles=route_handles, loc="upper left")
    if land_legend_handles:
        land_ax.legend(handles=land_legend_handles, loc="upper left", ncols=2)
    elif land_image is not None:
        fig.colorbar(land_image, ax=land_ax, shrink=0.80)

    for ax in axis_list:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@dataclass(frozen=True)
class _PlotRouteLayer:
    label: str
    line_xy: npt.NDArray[np.float64] | None
    points_xy: npt.NDArray[np.float64] | None = None
    waypoints: tuple[XY, ...] = ()


def _plot_route_layers(routes: Any) -> list[_PlotRouteLayer]:
    if routes is None:
        return []
    if isinstance(routes, Mapping) and "type" not in routes:
        return [_plot_route_layer(route, label=str(label)) for label, route in routes.items()]

    layer = _plot_route_layer(routes, label=None)
    if layer.line_xy is not None or layer.points_xy is not None:
        return [layer]

    if _is_sequence_of_plot_routes(routes):
        return [_plot_route_layer(route, label=f"Route {index}") for index, route in enumerate(routes, start=1)]
    return []


def _plot_route_layer(route: Any, *, label: str | None) -> _PlotRouteLayer:
    return _PlotRouteLayer(
        label=label or _route_label(route),
        line_xy=_route_line(route),
        points_xy=_route_points(route),
        waypoints=tuple(getattr(route, "waypoint_xy", ()) or ()),
    )


def _route_label(route: Any) -> str:
    raw = getattr(route, "solver", None)
    if isinstance(raw, str) and raw:
        return raw.replace("_", " ").title()
    return "Route"


def _route_line(route: Any) -> npt.NDArray[np.float64] | None:
    value = getattr(route, "path_xy", route)
    if value is None:
        return None
    try:
        line = _line_coordinates(value)
    except (TypeError, ValueError):
        return None
    return line if len(line) > 0 else None


def _route_points(route: Any) -> npt.NDArray[np.float64] | None:
    parts = []
    for leg in getattr(route, "legs", ()) or ():
        mask_xy = getattr(leg, "mask_xy", None)
        if mask_xy is None:
            continue
        points = np.asarray(mask_xy, dtype=np.float64)
        if points.ndim == 2 and points.shape[1] == 2 and len(points) > 0:
            parts.append(points)
    if not parts:
        return None
    return np.vstack(parts)


def _is_sequence_of_plot_routes(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | Path):
        return False
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return True
    return arr.ndim != 2 or arr.shape[1] != 2


def _waypoints_from_routes(route_layers: Sequence[_PlotRouteLayer]) -> list[XY]:
    points: list[XY] = []
    for layer in route_layers:
        points.extend(layer.waypoints)
    return points


def _plot_bounds(
    base_bounds: Bounds,
    *,
    waypoints: Sequence[XY],
    barriers: Sequence[Mapping[str, Any]],
    routes: Sequence[_PlotRouteLayer],
    margin: float,
) -> Bounds:
    xs: list[float] = []
    ys: list[float] = []
    for x_coord, y_coord in waypoints:
        xs.append(float(x_coord))
        ys.append(float(y_coord))
    for geometry in barriers:
        left, bottom, right, top = _geometry_bounds(geometry)
        xs.extend([left, right])
        ys.extend([bottom, top])
    for layer in routes:
        for line in (layer.line_xy, layer.points_xy):
            if line is not None and len(line) > 0:
                xs.extend(line[:, 0].tolist())
                ys.extend(line[:, 1].tolist())

    if not xs or not ys:
        return base_bounds

    value = float(margin)
    if value < 0.0 or not math.isfinite(value):
        raise ValueError("margin must be a non-negative finite value")
    return min(xs) - value, min(ys) - value, max(xs) + value, max(ys) + value


def _geometry_bounds(geometry: Mapping[str, Any]) -> Bounds:
    from shapely.geometry import shape

    left, bottom, right, top = shape(cast(dict[str, Any], dict(geometry))).bounds
    return float(left), float(bottom), float(right), float(top)


def _plot_resolution(
    bounds: Bounds,
    source_resolution: tuple[float, float],
    *,
    max_pixels: int,
) -> tuple[float, float]:
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    left, bottom, right, top = bounds
    x_res, y_res = source_resolution
    width = max(1.0, (right - left) / x_res)
    height = max(1.0, (top - bottom) / y_res)
    stride = max(1, int(math.ceil(math.sqrt(width * height / float(max_pixels)))))
    return x_res * stride, y_res * stride


def _imshow_extent(bounds: Bounds) -> tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    return left, right, bottom, top


def _land_use_image(
    ax: Any,
    land_use: npt.NDArray[np.float64],
    *,
    extent: tuple[float, float, float, float],
    labels: Mapping[float | int, str] | None,
    colors: Mapping[float | int, str] | Sequence[str] | None,
) -> tuple[Any, list[Any]]:
    if labels:
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch

        label_map = {float(value): str(label) for value, label in labels.items()}
        values = sorted(label_map)
        color_list = _land_use_color_list(values, colors)
        cmap = ListedColormap(color_list)
        image = ax.imshow(
            land_use,
            cmap=cmap,
            norm=BoundaryNorm(_class_boundaries(values), cmap.N),
            extent=extent,
            origin="upper",
            interpolation="nearest",
        )
        handles = [
            Patch(facecolor=color, edgecolor="none", label=label_map[value]) for value, color in zip(values, color_list)
        ]
        return image, handles

    image = ax.imshow(
        land_use,
        cmap="viridis",
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    return image, []


def _land_use_color_list(
    values: Sequence[float],
    colors: Mapping[float | int, str] | Sequence[str] | None,
) -> list[str]:
    default = [
        "#d9ead3",
        "#93c47d",
        "#c9b458",
        "#b6a05b",
        "#3f6f3a",
        "#8e8e86",
        "#5d9ca6",
        "#bf7f4a",
        "#6d76b8",
        "#d975a8",
    ]
    if isinstance(colors, Mapping):
        color_map = {float(value): str(color) for value, color in colors.items()}
        return [color_map.get(value, default[index % len(default)]) for index, value in enumerate(values)]
    if colors is not None:
        supplied = [str(color) for color in colors]
        if supplied:
            return [supplied[index % len(supplied)] for index, _ in enumerate(values)]
    return [default[index % len(default)] for index, _ in enumerate(values)]


def _class_boundaries(values: Sequence[float]) -> list[float]:
    if len(values) == 1:
        return [values[0] - 0.5, values[0] + 0.5]
    boundaries = [values[0] - (values[1] - values[0]) / 2.0]
    boundaries.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    boundaries.append(values[-1] + (values[-1] - values[-2]) / 2.0)
    return boundaries


def _draw_plot_overlays(
    axes: Sequence[Any],
    *,
    route_layers: Sequence[_PlotRouteLayer],
    waypoints: Sequence[XY],
    barriers: Sequence[Mapping[str, Any]],
) -> list[Any]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    route_colors = [
        "#d95f02",
        "#1f78b4",
        "#ffcc00",
        "#7570b3",
        "#1b9e77",
        "#e7298a",
    ]
    for ax in axes:
        for geometry in barriers:
            _draw_geometry(ax, geometry)
        if waypoints:
            ax.scatter(
                [point[0] for point in waypoints],
                [point[1] for point in waypoints],
                s=30,
                c="black",
                edgecolors="white",
                linewidths=0.8,
                zorder=6,
            )
        for index, layer in enumerate(route_layers):
            color = route_colors[index % len(route_colors)]
            if layer.line_xy is not None and len(layer.line_xy) > 0:
                ax.plot(layer.line_xy[:, 0], layer.line_xy[:, 1], color=color, lw=2.3, zorder=5)
            if layer.points_xy is not None and len(layer.points_xy) > 0:
                ax.scatter(
                    layer.points_xy[:, 0],
                    layer.points_xy[:, 1],
                    s=2.5,
                    color=color,
                    alpha=0.85,
                    zorder=5,
                )

    handles: list[Any] = [
        Line2D([0], [0], color=route_colors[index % len(route_colors)], lw=2.3, label=layer.label)
        for index, layer in enumerate(route_layers)
        if layer.line_xy is not None or layer.points_xy is not None
    ]
    if waypoints:
        handles.append(Line2D([0], [0], marker="o", color="black", lw=0, label="Waypoint"))
    if barriers:
        handles.append(Patch(facecolor="none", edgecolor="#111111", hatch="////", label="Barrier"))
    return handles


def _draw_geometry(ax: Any, geometry: Mapping[str, Any]) -> None:
    from shapely.geometry import (
        LinearRing,
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
        shape,
    )

    geom = shape(cast(dict[str, Any], dict(geometry)))
    if isinstance(geom, Polygon):
        xs, ys = geom.exterior.xy
        ax.fill(xs, ys, facecolor="none", edgecolor="#111111", hatch="////", linewidth=1.0, zorder=4)
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            xs, ys = part.exterior.xy
            ax.fill(xs, ys, facecolor="none", edgecolor="#111111", hatch="////", linewidth=1.0, zorder=4)
    elif isinstance(geom, LineString | LinearRing):
        xs, ys = geom.xy
        ax.plot(xs, ys, color="#111111", lw=1.2, zorder=4)
    elif isinstance(geom, MultiLineString):
        for part in geom.geoms:
            xs, ys = part.xy
            ax.plot(xs, ys, color="#111111", lw=1.2, zorder=4)
    elif isinstance(geom, Point):
        ax.scatter([geom.x], [geom.y], s=12, c="#111111", zorder=4)
    elif isinstance(geom, MultiPoint):
        ax.scatter(
            [point.x for point in geom.geoms],
            [point.y for point in geom.geoms],
            s=12,
            c="#111111",
            zorder=4,
        )
