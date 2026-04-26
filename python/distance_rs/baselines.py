"""Reference baseline algorithms used by benchmarks and examples."""

from __future__ import annotations

import heapq
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from ._distance import (
    ProgressOption,
    RasterSurface,
    VerticalFactor,
    _normalize_progress_callback,
    _normalize_progress_interval,
    evaluate_path_cost,
)

MIN_COST = 1.0e-12
WHITEBOX_NODATA = -32768.0
NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

__all__ = [
    "MIN_COST",
    "PathCostMetrics",
    "RasterDijkstraResult",
    "WHITEBOX_NODATA",
    "WhiteboxCostDistanceResult",
    "compare_distances",
    "euclidean_distance_to_sources",
    "json_safe",
    "normalize_cell_size",
    "orient_line",
    "path_cost_metrics",
    "path_length",
    "raster_dijkstra",
    "read_whitebox_header",
    "read_whitebox_raster",
    "trace_raster_path",
    "trace_path_mask",
    "whitebox_dtype",
    "whitebox_cost_distance",
    "write_whitebox_raster",
]


@dataclass(frozen=True)
class RasterDijkstraResult:
    """Distance and parent field for the local 8-neighbor Dijkstra baseline."""

    distance: npt.NDArray[np.float64]
    parent: npt.NDArray[np.int64]


@dataclass(frozen=True)
class WhiteboxCostDistanceResult:
    """Outputs from a WhiteboxTools CostDistance baseline run.

    `distance` is always populated. `backlink` is Whitebox's backlink raster,
    and `pathway` is populated only when `destinations` is passed to
    `whitebox_cost_distance`.
    """

    distance: npt.NDArray[np.float64]
    backlink: npt.NDArray[np.float64]
    pathway: npt.NDArray[np.float64] | None = None


@dataclass(frozen=True)
class PathCostMetrics:
    """Distance-rs surface evaluation for an already traced path."""

    cost: float
    distance: float
    time_hours: float


def raster_dijkstra(
    sources: npt.ArrayLike,
    *,
    cost_surface: npt.ArrayLike | None = None,
    elevation: npt.ArrayLike | None = None,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    barriers: npt.ArrayLike | None = None,
    cell_size: float | tuple[float, float] = 1.0,
    progress: ProgressOption = True,
    progress_interval: int | None = None,
) -> RasterDijkstraResult:
    """Compute distance and parents with an 8-neighbor raster Dijkstra stencil.

    Supports, relative to the main ordered-upwind solver:
    - multiple sources;
    - finite cost/friction rasters;
    - blocked-cell barriers;
    - optional elevation as 3D step length;
    - the same Python `VerticalFactor` definitions; and
    - square or rectangular cell sizes.

    Does not support:
    - continuous ordered-upwind/eikonal updates across cell faces;
    - sub-cell parent points or back-direction rasters;
    - line-of-sight segment barrier checks between nonadjacent parent points;
    - source allocation/direction rasters; or
    - any geospatial IO/reprojection on its own.

    This is a conventional grid-constrained baseline, not a ground truth for
    the continuous solver.
    """

    source_arr = np.asarray(sources, dtype=np.float64)
    if source_arr.ndim != 2:
        raise ValueError("sources must be a 2D array")
    rows, cols = source_arr.shape

    cost = (
        np.ones((rows, cols), dtype=np.float64)
        if cost_surface is None
        else np.asarray(cost_surface, dtype=np.float64)
    )
    if cost.shape != source_arr.shape:
        raise ValueError("cost_surface must match sources shape")
    cost = np.maximum(cost, MIN_COST)

    has_elevation = elevation is not None
    elev = (
        np.zeros((rows, cols), dtype=np.float64)
        if elevation is None
        else np.asarray(elevation, dtype=np.float64)
    )
    if elev.shape != source_arr.shape:
        raise ValueError("elevation must match sources shape")

    blocked = (
        np.zeros((rows, cols), dtype=bool)
        if barriers is None
        else np.asarray(barriers, dtype=bool).copy()
    )
    if blocked.shape != source_arr.shape:
        raise ValueError("barriers must match sources shape")

    valid = ~blocked & np.isfinite(cost)
    if has_elevation:
        valid &= np.isfinite(elev)

    cell_size_x, cell_size_y = normalize_cell_size(cell_size)
    vf = VerticalFactor.from_any(vertical_factor)
    total_valid = int(np.count_nonzero(valid))

    distance = np.full((rows, cols), np.inf, dtype=np.float64)
    parent = np.full((rows, cols), -1, dtype=np.int64)
    heap: list[tuple[float, int, int]] = []
    for row, col in zip(*np.nonzero((source_arr != 0.0) & np.isfinite(source_arr) & valid)):
        row = int(row)
        col = int(col)
        distance[row, col] = 0.0
        parent[row, col] = row * cols + col
        heapq.heappush(heap, (0.0, row, col))

    if not heap:
        raise ValueError("at least one source cell must be valid")

    progress_callback, close_progress = _normalize_progress_callback(
        progress,
        total=total_valid,
        label="raster_dijkstra",
    )
    progress_interval_value = _normalize_progress_interval(progress_interval, total_valid)

    accepted = 0
    next_report = progress_interval_value
    try:
        while heap:
            current, row, col = heapq.heappop(heap)
            if current > distance[row, col]:
                continue
            accepted += 1
            if progress_callback is not None and accepted >= next_report:
                progress_callback(accepted, total_valid)
                while accepted >= next_report:
                    next_report += progress_interval_value
            for dr, dc in NEIGHBORS_8:
                next_row = row + dr
                next_col = col + dc
                if next_row < 0 or next_col < 0 or next_row >= rows or next_col >= cols:
                    continue
                if not valid[next_row, next_col]:
                    continue
                step = transition_cost(
                    row,
                    col,
                    next_row,
                    next_col,
                    cost=cost,
                    elevation=elev,
                    has_elevation=has_elevation,
                    vertical_factor=vf,
                    cell_size_x=cell_size_x,
                    cell_size_y=cell_size_y,
                )
                if not math.isfinite(step):
                    continue
                candidate = current + step
                if candidate < distance[next_row, next_col]:
                    distance[next_row, next_col] = candidate
                    parent[next_row, next_col] = row * cols + col
                    heapq.heappush(heap, (candidate, next_row, next_col))
        if progress_callback is not None:
            progress_callback(accepted, total_valid)
    finally:
        if close_progress is not None:
            close_progress()

    return RasterDijkstraResult(distance=distance, parent=parent)


def trace_raster_path(
    parent: npt.NDArray[np.int64],
    destination: tuple[int, int],
    *,
    cell_size: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> npt.NDArray[np.float64]:
    """Trace one raster Dijkstra path from destination back to its source."""

    rows, cols = parent.shape
    cell_size_x, cell_size_y = normalize_cell_size(cell_size)
    origin_x, origin_y = origin
    row, col = destination
    coords: list[tuple[float, float]] = []

    for _ in range(rows * cols):
        coords.append((origin_x + col * cell_size_x, origin_y + row * cell_size_y))
        parent_idx = int(parent[row, col])
        if parent_idx < 0:
            break
        parent_row, parent_col = divmod(parent_idx, cols)
        if parent_row == row and parent_col == col:
            break
        row, col = parent_row, parent_col

    return np.asarray(coords, dtype=np.float64)


def path_cost_metrics(
    surface: RasterSurface | npt.ArrayLike,
    line_xy: npt.ArrayLike,
    *,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    baseline_speed_kmh: float = 5.0,
    source_xy: tuple[float, float] | None = None,
    destination_xy: tuple[float, float] | None = None,
    max_step: float | None = None,
) -> PathCostMetrics:
    """Evaluate path distance, cost, and time on the distance-rs cost surface."""

    line = np.asarray(line_xy, dtype=np.float64)
    if source_xy is not None and destination_xy is not None:
        line = orient_line(line, source_xy, destination_xy)
    distance = path_length(line)
    cost = evaluate_path_cost(
        surface,
        line,
        vertical_factor=vertical_factor,
        max_step=max_step,
    )
    time_hours = time_hours_from_cost(
        cost,
        vertical_factor=VerticalFactor.from_any(vertical_factor),
        baseline_speed_kmh=baseline_speed_kmh,
    )
    return PathCostMetrics(cost=cost, distance=distance, time_hours=time_hours)


def orient_line(
    line_xy: npt.ArrayLike,
    source_xy: tuple[float, float],
    destination_xy: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Return a copy of `line_xy` oriented from source to destination."""

    line = np.asarray(line_xy, dtype=np.float64)
    if len(line) < 2:
        return np.ascontiguousarray(line, dtype=np.float64)
    source = np.asarray(source_xy, dtype=np.float64)
    destination = np.asarray(destination_xy, dtype=np.float64)
    forward_error = np.linalg.norm(line[0] - source) + np.linalg.norm(line[-1] - destination)
    reverse_error = np.linalg.norm(line[-1] - source) + np.linalg.norm(line[0] - destination)
    if reverse_error < forward_error:
        line = line[::-1]
    return np.ascontiguousarray(line, dtype=np.float64)


def path_length(line_xy: npt.ArrayLike) -> float:
    line = np.asarray(line_xy, dtype=np.float64)
    if len(line) < 2:
        return 0.0
    delta = np.diff(line, axis=0)
    return float(np.hypot(delta[:, 0], delta[:, 1]).sum())


def time_hours_from_cost(
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


def trace_path_mask(
    path_mask: npt.ArrayLike,
    source: tuple[int, int],
    destination: tuple[int, int],
    *,
    cell_size: float | tuple[float, float] = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
) -> npt.NDArray[np.float64]:
    """Trace an ordered cell-center line through a pathway mask when possible."""

    mask = np.asarray(path_mask, dtype=bool).copy()
    rows, cols = mask.shape
    source_row, source_col = source
    dest_row, dest_col = destination
    if not (0 <= source_row < rows and 0 <= source_col < cols):
        return np.empty((0, 2), dtype=np.float64)
    if not (0 <= dest_row < rows and 0 <= dest_col < cols):
        return np.empty((0, 2), dtype=np.float64)
    mask[source] = True
    mask[destination] = True

    distance = np.full(mask.shape, np.inf, dtype=np.float64)
    parent = np.full(mask.shape, -1, dtype=np.int64)
    heap: list[tuple[float, int, int]] = [(0.0, source_row, source_col)]
    distance[source] = 0.0
    parent[source] = source_row * cols + source_col
    cell_size_x, cell_size_y = normalize_cell_size(cell_size)

    while heap:
        current, row, col = heapq.heappop(heap)
        if current > distance[row, col]:
            continue
        if (row, col) == destination:
            break
        for dr, dc in NEIGHBORS_8:
            next_row = row + dr
            next_col = col + dc
            if next_row < 0 or next_col < 0 or next_row >= rows or next_col >= cols:
                continue
            if not mask[next_row, next_col]:
                continue
            step = math.hypot(dc * cell_size_x, dr * cell_size_y)
            candidate = current + step
            if candidate < distance[next_row, next_col]:
                distance[next_row, next_col] = candidate
                parent[next_row, next_col] = row * cols + col
                heapq.heappush(heap, (candidate, next_row, next_col))

    if not math.isfinite(distance[destination]):
        return np.empty((0, 2), dtype=np.float64)

    line = trace_raster_path(parent, destination, cell_size=cell_size, origin=origin)
    return line[::-1].copy()


def transition_cost(
    row: int,
    col: int,
    next_row: int,
    next_col: int,
    *,
    cost: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64],
    has_elevation: bool,
    vertical_factor: VerticalFactor,
    cell_size_x: float,
    cell_size_y: float,
) -> float:
    plan_distance = math.hypot((next_col - col) * cell_size_x, (next_row - row) * cell_size_y)
    dz = float(elevation[next_row, next_col] - elevation[row, col]) if has_elevation else 0.0
    surface_distance = math.hypot(plan_distance, dz) if has_elevation else plan_distance
    angle = math.degrees(math.atan2(dz, plan_distance)) if has_elevation else 0.0
    vf = vertical_factor.factor(angle)
    if not math.isfinite(vf):
        return math.inf
    return surface_distance * 0.5 * (float(cost[row, col]) + float(cost[next_row, next_col])) * vf


def compare_distances(
    actual: npt.NDArray[np.float64],
    expected: npt.NDArray[np.float64],
) -> dict[str, float | int]:
    actual_finite = np.isfinite(actual)
    expected_finite = np.isfinite(expected)
    overlap = actual_finite & expected_finite
    mismatch = actual_finite ^ expected_finite

    if np.any(overlap):
        diff = actual[overlap] - expected[overlap]
        abs_diff = np.abs(diff)
        denom = np.maximum(np.abs(expected[overlap]), MIN_COST)
        mae = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        max_abs = float(np.max(abs_diff))
        rel_mae = float(np.mean(abs_diff / denom))
    else:
        mae = math.inf
        rmse = math.inf
        max_abs = math.inf
        rel_mae = math.inf

    total = actual.size
    return {
        "finite_overlap": int(np.count_nonzero(overlap)),
        "actual_reachable": int(np.count_nonzero(actual_finite)),
        "expected_reachable": int(np.count_nonzero(expected_finite)),
        "reachability_mismatch": int(np.count_nonzero(mismatch)),
        "reachability_mismatch_rate": float(np.count_nonzero(mismatch) / total),
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "relative_mae": rel_mae,
    }


def euclidean_distance_to_sources(
    sources: npt.NDArray[np.float64],
    cell_size: float | tuple[float, float] = 1.0,
) -> npt.NDArray[np.float64]:
    cell_size_x, cell_size_y = normalize_cell_size(cell_size)
    source_rows, source_cols = np.nonzero((sources != 0.0) & np.isfinite(sources))
    if len(source_rows) == 0:
        raise ValueError("at least one source cell is required")
    rows, cols = sources.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    out = np.full((rows, cols), np.inf, dtype=np.float64)
    for source_row, source_col in zip(source_rows, source_cols):
        candidate = np.hypot((xx - source_col) * cell_size_x, (yy - source_row) * cell_size_y)
        out = np.minimum(out, candidate)
    return out


def normalize_cell_size(cell_size: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(cell_size, tuple):
        if len(cell_size) != 2:
            raise ValueError("cell_size tuple must be (x_size, y_size)")
        x_size = float(cell_size[0])
        y_size = float(cell_size[1])
    else:
        x_size = float(cell_size)
        y_size = float(cell_size)
    if x_size <= 0.0 or y_size <= 0.0:
        raise ValueError("cell_size values must be positive")
    return x_size, y_size


def write_whitebox_raster(
    path: Path,
    array: npt.NDArray[np.float64],
    *,
    cell_size: float | tuple[float, float],
    nodata: float,
) -> None:
    path = path.with_suffix(".dep")
    data = np.asarray(array, dtype=np.float64)
    rows, cols = data.shape
    cell_size_x, cell_size_y = normalize_cell_size(cell_size)
    writable = data.copy()
    writable[~np.isfinite(writable)] = nodata
    valid = np.isfinite(writable) & (writable != nodata)
    minimum = float(np.min(writable[valid])) if np.any(valid) else 0.0
    maximum = float(np.max(writable[valid])) if np.any(valid) else 0.0
    header = "\n".join(
        [
            f"Min:\t{minimum}",
            f"Max:\t{maximum}",
            f"North:\t{rows * cell_size_y}",
            "South:\t0",
            f"East:\t{cols * cell_size_x}",
            "West:\t0",
            f"Cols:\t{cols}",
            f"Rows:\t{rows}",
            "Data Type:\tfloat",
            "Z Units:\tnot specified",
            "XY Units:\tnot specified",
            "Projection:\tnot specified",
            "Data Scale:\tcontinuous",
            f"Display Min:\t{minimum}",
            f"Display Max:\t{maximum}",
            "Preferred Palette:\tspectrum.plt",
            f"NoData:\t{nodata}",
            "Byte Order:\tLITTLE_ENDIAN",
            "Metadata Entry:\tCreated by distance-rs benchmark harness",
            "",
        ]
    )
    path.write_text(header, encoding="utf-8")
    writable.astype("<f4").tofile(path.with_suffix(".tas"))


def whitebox_cost_distance(
    sources: npt.ArrayLike,
    *,
    cost_surface: npt.ArrayLike | None = None,
    barriers: npt.ArrayLike | None = None,
    cell_size: float | tuple[float, float] = 1.0,
    destinations: npt.ArrayLike | None = None,
) -> WhiteboxCostDistanceResult:
    """Run WhiteboxTools CostDistance with distance-rs-style barrier burn-in.

    Supports, relative to the main ordered-upwind solver:
    - multiple source cells;
    - finite cost/friction rasters;
    - blocked-cell barriers, burned into Whitebox's source and cost rasters as
      NoData; and
    - square or rectangular cell sizes in the temporary Whitebox rasters.

    When `destinations` is supplied, this also runs Whitebox `CostPathway` and
    returns the pathway raster. Destination cells are interpreted as
    nonzero finite cells in the supplied 2D array.

    Does not support:
    - elevation, 3D surface distance, or vertical-factor rules;
    - continuous ordered-upwind/eikonal updates;
    - parent/back-direction rasters compatible with `optimal_path_as_line`;
    - source allocation/direction rasters;
    - CRS, affine transforms, or reprojection metadata; or
    - arbitrary Whitebox options beyond `CostDistance` plus optional
      `CostPathway`.

    This is a friction-only external-product comparison. It is useful for
    matching common GIS cost-distance workflows, not for validating terrain-
    aware vertical-factor behavior.
    """

    if importlib.util.find_spec("whitebox") is None:
        raise RuntimeError(
            "Whitebox baseline requested but the 'whitebox' package is not installed. "
            "Run 'uv sync --group whitebox' or use 'uv run --with whitebox ...'."
        )

    source_arr = np.asarray(sources, dtype=np.float64)
    if source_arr.ndim != 2:
        raise ValueError("sources must be a 2D array")
    rows, cols = source_arr.shape

    cost = (
        np.ones((rows, cols), dtype=np.float64)
        if cost_surface is None
        else np.asarray(cost_surface, dtype=np.float64)
    )
    if cost.shape != source_arr.shape:
        raise ValueError("cost_surface must match sources shape")

    blocked = (
        np.zeros((rows, cols), dtype=bool)
        if barriers is None
        else np.asarray(barriers, dtype=bool).copy()
    )
    if blocked.shape != source_arr.shape:
        raise ValueError("barriers must match sources shape")

    destination_arr = None if destinations is None else np.asarray(destinations, dtype=np.float64)
    if destination_arr is not None and destination_arr.shape != source_arr.shape:
        raise ValueError("destinations must match sources shape")

    valid_cost = np.isfinite(cost) & ~blocked
    source = np.full(source_arr.shape, WHITEBOX_NODATA, dtype=np.float64)
    source[(source_arr != 0.0) & np.isfinite(source_arr) & valid_cost] = 1.0
    if not np.any(source != WHITEBOX_NODATA):
        raise ValueError("at least one source cell must be valid")

    burned_cost = np.where(valid_cost, np.maximum(cost, MIN_COST), WHITEBOX_NODATA)
    destination = None
    if destination_arr is not None:
        destination = np.zeros(destination_arr.shape, dtype=np.float64)
        destination[(destination_arr != 0.0) & np.isfinite(destination_arr) & valid_cost] = 1.0
        if not np.any(destination != 0.0):
            raise ValueError("at least one destination cell must be valid")

    return _run_whitebox_cost_distance(
        source=source,
        cost=burned_cost,
        destination=destination,
        cell_size=cell_size,
    )


def _run_whitebox_cost_distance(
    *,
    source: npt.NDArray[np.float64],
    cost: npt.NDArray[np.float64],
    destination: npt.NDArray[np.float64] | None,
    cell_size: float | tuple[float, float],
) -> WhiteboxCostDistanceResult:
    import tempfile

    import whitebox

    with tempfile.TemporaryDirectory(prefix="distance-rs-whitebox-") as tmp_name:
        tmp = Path(tmp_name)
        source_path = tmp / "source.dep"
        cost_path = tmp / "cost.dep"
        accum_path = tmp / "accum.dep"
        backlink_path = tmp / "backlink.dep"
        destination_path = tmp / "destination.dep"
        pathway_path = tmp / "pathway.dep"

        write_whitebox_raster(source_path, source, cell_size=cell_size, nodata=WHITEBOX_NODATA)
        write_whitebox_raster(cost_path, cost, cell_size=cell_size, nodata=WHITEBOX_NODATA)
        if destination is not None:
            write_whitebox_raster(
                destination_path,
                destination,
                cell_size=cell_size,
                nodata=WHITEBOX_NODATA,
            )

        wbt = whitebox.WhiteboxTools()
        wbt.set_working_dir(str(tmp))
        if hasattr(wbt, "set_verbose_mode"):
            wbt.set_verbose_mode(False)
        exit_code = wbt.cost_distance(
            source_path.name,
            cost_path.name,
            accum_path.name,
            backlink_path.name,
            callback=lambda _message: None,
        )
        if exit_code not in (0, None):
            raise RuntimeError(f"Whitebox CostDistance failed with exit code {exit_code}")

        pathway = None
        if destination is not None:
            exit_code = wbt.cost_pathway(
                destination_path.name,
                backlink_path.name,
                pathway_path.name,
                zero_background=True,
                callback=lambda _message: None,
            )
            if exit_code not in (0, None):
                raise RuntimeError(f"Whitebox CostPathway failed with exit code {exit_code}")
            pathway = read_whitebox_raster(pathway_path, nodata_as_inf=False)

        return WhiteboxCostDistanceResult(
            distance=read_whitebox_raster(accum_path, nodata_as_inf=True),
            backlink=read_whitebox_raster(backlink_path, nodata_as_inf=False),
            pathway=pathway,
        )


def read_whitebox_raster(path: Path, *, nodata_as_inf: bool = False) -> npt.NDArray[np.float64]:
    header = read_whitebox_header(path)
    rows = int(header["rows"])
    cols = int(header["cols"])
    nodata = float(header.get("nodata", WHITEBOX_NODATA))
    dtype = whitebox_dtype(header)
    data = np.fromfile(path.with_suffix(".tas"), dtype=dtype)
    expected_size = rows * cols
    if data.size != expected_size:
        raise RuntimeError(
            f"Whitebox raster {path} has {data.size} cells, expected {expected_size}"
        )
    out = data.reshape((rows, cols)).astype(np.float64)
    if nodata_as_inf:
        out[out == nodata] = np.inf
    return out


def read_whitebox_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        header[key.strip().lower()] = value.strip()
    return header


def whitebox_dtype(header: dict[str, str]) -> np.dtype[Any]:
    data_type = header.get("data type", "float").strip().lower()
    byte_order = header.get("byte order", "little_endian").strip().lower()
    endian = ">" if "big" in byte_order else "<"

    if data_type in {"float", "f32", "real", "single"}:
        return np.dtype(f"{endian}f4")
    if data_type in {"double", "f64"}:
        return np.dtype(f"{endian}f8")
    if data_type in {"integer", "int", "i16"}:
        return np.dtype(f"{endian}i2")
    if data_type in {"i32", "int32"}:
        return np.dtype(f"{endian}i4")
    raise RuntimeError(f"unsupported Whitebox raster data type: {data_type}")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
