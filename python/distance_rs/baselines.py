"""Reference baseline algorithms used by benchmarks and examples."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from ._distance import VerticalFactor


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


@dataclass(frozen=True)
class RasterDijkstraResult:
    """Distance and parent field for the 8-neighbor raster Dijkstra baseline."""

    distance: npt.NDArray[np.float64]
    parent: npt.NDArray[np.int64]


def raster_dijkstra_baseline(
    sources: npt.ArrayLike,
    *,
    cost_surface: npt.ArrayLike | None = None,
    elevation: npt.ArrayLike | None = None,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    barriers: npt.ArrayLike | None = None,
    cell_size: float | tuple[float, float] = 1.0,
    use_surface_distance: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute distance with a conventional 8-neighbor raster Dijkstra stencil."""

    return raster_dijkstra(
        sources,
        cost_surface=cost_surface,
        elevation=elevation,
        vertical_factor=vertical_factor,
        barriers=barriers,
        cell_size=cell_size,
        use_surface_distance=use_surface_distance,
    ).distance


def raster_dijkstra(
    sources: npt.ArrayLike,
    *,
    cost_surface: npt.ArrayLike | None = None,
    elevation: npt.ArrayLike | None = None,
    vertical_factor: str | Mapping[str, Any] | VerticalFactor | None = None,
    barriers: npt.ArrayLike | None = None,
    cell_size: float | tuple[float, float] = 1.0,
    use_surface_distance: bool = True,
) -> RasterDijkstraResult:
    """Compute distance and parent links with an 8-neighbor raster Dijkstra stencil."""

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

    while heap:
        current, row, col = heapq.heappop(heap)
        if current > distance[row, col]:
            continue
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
                use_surface_distance=use_surface_distance,
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


def transition_cost(
    row: int,
    col: int,
    next_row: int,
    next_col: int,
    *,
    cost: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64],
    has_elevation: bool,
    use_surface_distance: bool,
    vertical_factor: VerticalFactor,
    cell_size_x: float,
    cell_size_y: float,
) -> float:
    plan_distance = math.hypot((next_col - col) * cell_size_x, (next_row - row) * cell_size_y)
    dz = float(elevation[next_row, next_col] - elevation[row, col]) if has_elevation else 0.0
    surface_distance = (
        math.hypot(plan_distance, dz) if has_elevation and use_surface_distance else plan_distance
    )
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
