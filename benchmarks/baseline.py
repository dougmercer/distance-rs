#!/usr/bin/env python
"""Compare distance-rs with raster and optional Whitebox CostDistance baselines.

The local baseline intentionally uses an 8-neighbor raster stencil. WhiteboxTools
CostDistance is available as an external customer-facing comparison for isotropic
cost-distance cases. Neither baseline is a mathematical ground truth for the
ordered-upwind solver.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from distance_rs import VerticalFactor, distance_accumulation


MIN_COST = 1.0e-12
WHITEBOX_NODATA = -32768.0
RASTER_BASELINE = "raster_dijkstra"
WHITEBOX_BASELINE = "whitebox_cost_distance"
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
class CaseData:
    name: str
    description: str
    sources: npt.NDArray[np.float64]
    cost: npt.NDArray[np.float64]
    elevation: npt.NDArray[np.float64] | None
    barriers: npt.NDArray[np.bool_]
    vertical_factor: VerticalFactor
    cell_size: float | tuple[float, float] = 1.0
    use_surface_distance: bool = True
    exact_reference: npt.NDArray[np.float64] | None = None


@dataclass(frozen=True)
class TimedResult:
    result: Any
    times_sec: list[float]

    @property
    def median_sec(self) -> float:
        return statistics.median(self.times_sec)

    @property
    def min_sec(self) -> float:
        return min(self.times_sec)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sizes = args.sizes
    case_names = expand_cases(args.cases)
    baseline_names = expand_baselines(args.baselines)

    rows: list[dict[str, Any]] = []
    failed = False

    for size in sizes:
        for case_name in case_names:
            case = make_case(case_name, size)
            print(f"running case={case.name} size={size}", file=sys.stderr)

            ours = time_call(
                lambda: distance_accumulation(
                    case.sources,
                    cost_surface=case.cost,
                    elevation=case.elevation,
                    barriers=case.barriers,
                    vertical_factor=case.vertical_factor,
                    cell_size=case.cell_size,
                    search_radius=args.search_radius,
                    use_surface_distance=case.use_surface_distance,
                ).distance,
                repeats=args.repeats,
                warmups=args.warmups,
            )

            baselines: dict[str, dict[str, Any]] = {}
            if RASTER_BASELINE in baseline_names:
                baseline = time_call(
                    lambda: raster_dijkstra_baseline(
                        case.sources,
                        cost_surface=case.cost,
                        elevation=case.elevation,
                        barriers=case.barriers,
                        vertical_factor=case.vertical_factor,
                        cell_size=case.cell_size,
                        use_surface_distance=case.use_surface_distance,
                    ),
                    repeats=args.repeats,
                    warmups=args.warmups,
                )
                baselines[RASTER_BASELINE] = make_baseline_payload(
                    baseline, ours, case.exact_reference
                )

            if WHITEBOX_BASELINE in baseline_names:
                comparable, reason = whitebox_comparable(case)
                if comparable:
                    whitebox = time_call(
                        lambda: whitebox_cost_distance_baseline(case),
                        repeats=args.repeats,
                        warmups=args.warmups,
                    )
                    baselines[WHITEBOX_BASELINE] = make_baseline_payload(
                        whitebox, ours, case.exact_reference
                    )
                else:
                    baselines[WHITEBOX_BASELINE] = {
                        "status": "skipped",
                        "reason": reason,
                    }

            row = {
                "case": case.name,
                "description": case.description,
                "size": size,
                "search_radius": args.search_radius,
                "ordered_upwind_times_sec": ours.times_sec,
                "ordered_upwind_median_sec": ours.median_sec,
                "baselines": baselines,
                "reference": compare_distances(ours.result, case.exact_reference)
                if case.exact_reference is not None
                else None,
            }
            rows.append(row)

            for baseline_payload in baselines.values():
                if baseline_payload.get("status") != "ok":
                    continue
                comparison = baseline_payload["comparison"]
                if args.max_rmse is not None and comparison["rmse"] > args.max_rmse:
                    failed = True
                if (
                    args.max_reachability_mismatch_rate is not None
                    and comparison["reachability_mismatch_rate"]
                    > args.max_reachability_mismatch_rate
                ):
                    failed = True

    print_summary(rows)
    if args.json is not None:
        write_json(args.json, rows, args)

    return 2 if failed else 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["flat", "cost", "terrain"],
        help="Case names to run: flat, cost, terrain, downslope, or all.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[32, 64],
        help="Square raster sizes to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats per solver.")
    parser.add_argument("--warmups", type=int, default=1, help="Untimed warmup runs per solver.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=[RASTER_BASELINE],
        help="Baselines to run: raster, whitebox, or all.",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=4.0,
        help="Ordered-upwind search radius in map units.",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--max-rmse",
        type=float,
        help="Exit nonzero if ordered-upwind vs raster baseline RMSE exceeds this value.",
    )
    parser.add_argument(
        "--max-reachability-mismatch-rate",
        type=float,
        default=None,
        help="Exit nonzero if finite/infinite reachability mismatch rate exceeds this value.",
    )
    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if args.search_radius <= 0.0:
        parser.error("--search-radius must be positive")
    if any(size < 3 for size in args.sizes):
        parser.error("--sizes values must be at least 3")
    if (
        args.max_reachability_mismatch_rate is not None
        and not 0.0 <= args.max_reachability_mismatch_rate <= 1.0
    ):
        parser.error("--max-reachability-mismatch-rate must be between 0 and 1")
    return args


def expand_baselines(baselines: list[str]) -> list[str]:
    expanded: list[str] = []
    for baseline in baselines:
        name = baseline.lower()
        if name == "all":
            return [RASTER_BASELINE, WHITEBOX_BASELINE]
        if name in {"raster", "local", "dijkstra", RASTER_BASELINE}:
            expanded.append(RASTER_BASELINE)
        elif name in {"whitebox", "wbt", "cost_distance", WHITEBOX_BASELINE}:
            expanded.append(WHITEBOX_BASELINE)
        else:
            raise SystemExit(f"unknown baseline: {baseline}")
    return list(dict.fromkeys(expanded))


def expand_cases(case_names: list[str]) -> list[str]:
    available = ["flat", "cost", "terrain", "downslope"]
    lowered = [name.lower() for name in case_names]
    if "all" in lowered:
        return available
    unknown = sorted(set(lowered) - set(available))
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(unknown)}")
    return lowered


def make_case(name: str, size: int) -> CaseData:
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size // 2, size // 2)
    sources = np.zeros((size, size), dtype=np.float64)
    sources[center] = 1.0
    barriers = np.zeros((size, size), dtype=bool)

    if name == "flat":
        cost = np.ones((size, size), dtype=np.float64)
        exact = euclidean_distance_to_sources(sources)
        return CaseData(
            name=name,
            description="flat unit-cost surface with exact Euclidean reference",
            sources=sources,
            cost=cost,
            elevation=np.zeros((size, size), dtype=np.float64),
            barriers=barriers,
            vertical_factor=VerticalFactor.from_any(None),
            exact_reference=exact,
        )

    if name == "cost":
        x = xx / max(size - 1, 1)
        y = yy / max(size - 1, 1)
        cost = 1.0 + 2.0 * x + 0.35 * np.sin(6.0 * math.pi * y)
        cost = np.maximum(cost, 0.05).astype(np.float64)
        barriers[size // 4 : 3 * size // 4, size // 2] = True
        barriers[size // 2 - max(1, size // 16) : size // 2 + max(2, size // 16), size // 2] = False
        return CaseData(
            name=name,
            description="smooth cost gradient with a single barrier and gap",
            sources=sources,
            cost=cost,
            elevation=np.zeros((size, size), dtype=np.float64),
            barriers=barriers,
            vertical_factor=VerticalFactor.from_any(None),
        )

    if name == "terrain":
        x = (xx - center[1]) / max(size, 1)
        y = (yy - center[0]) / max(size, 1)
        elevation = 40.0 * np.exp(-28.0 * (x * x + y * y)) + 8.0 * x
        cost = 1.0 + 0.4 * np.cos(4.0 * math.pi * x) * np.sin(3.0 * math.pi * y)
        cost = np.maximum(cost, 0.2).astype(np.float64)
        barriers[size // 3 : 2 * size // 3, size // 3] = True
        barriers[size // 2, size // 3] = False
        return CaseData(
            name=name,
            description="hill terrain with bidirectional Tobler hiking time",
            sources=sources,
            cost=cost,
            elevation=elevation.astype(np.float64),
            barriers=barriers,
            vertical_factor=VerticalFactor.from_any("bidir_hiking_time"),
        )

    if name == "downslope":
        sources.fill(0.0)
        sources[size // 2, size - 2] = 1.0
        elevation = np.tile(np.linspace(0.0, 50.0, size), (size, 1))
        return CaseData(
            name=name,
            description="binary vertical factor permitting flat/downslope travel from high ground",
            sources=sources,
            cost=np.ones((size, size), dtype=np.float64),
            elevation=elevation.astype(np.float64),
            barriers=barriers,
            vertical_factor=VerticalFactor.from_any(
                {"type": "binary", "low_cut_angle": -90.0, "high_cut_angle": 0.25}
            ),
        )

    raise ValueError(f"unhandled case: {name}")


def make_baseline_payload(
    baseline: TimedResult,
    ours: TimedResult,
    exact_reference: npt.NDArray[np.float64] | None,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "times_sec": baseline.times_sec,
        "median_sec": baseline.median_sec,
        "speedup": baseline.median_sec / ours.median_sec if ours.median_sec > 0.0 else math.inf,
        "comparison": compare_distances(ours.result, baseline.result),
        "reference": compare_distances(baseline.result, exact_reference)
        if exact_reference is not None
        else None,
    }


def raster_dijkstra_baseline(
    sources: npt.ArrayLike,
    *,
    cost_surface: npt.ArrayLike | None = None,
    elevation: npt.ArrayLike | None = None,
    vertical_factor: str | dict[str, Any] | VerticalFactor | None = None,
    barriers: npt.ArrayLike | None = None,
    cell_size: float | tuple[float, float] = 1.0,
    use_surface_distance: bool = True,
) -> npt.NDArray[np.float64]:
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

    dist = np.full((rows, cols), np.inf, dtype=np.float64)
    heap: list[tuple[float, int, int]] = []
    for row, col in zip(*np.nonzero((source_arr != 0.0) & np.isfinite(source_arr) & valid)):
        dist[row, col] = 0.0
        heapq.heappush(heap, (0.0, int(row), int(col)))

    if not heap:
        raise ValueError("at least one source cell must be valid")

    while heap:
        current, row, col = heapq.heappop(heap)
        if current > dist[row, col]:
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
            if candidate < dist[next_row, next_col]:
                dist[next_row, next_col] = candidate
                heapq.heappush(heap, (candidate, next_row, next_col))

    return dist


def whitebox_comparable(case: CaseData) -> tuple[bool, str | None]:
    if case.vertical_factor.normalized().type != "none":
        return False, "Whitebox CostDistance has no vertical-factor input."
    if case.elevation is not None and case.use_surface_distance:
        finite_elevation = case.elevation[np.isfinite(case.elevation)]
        if finite_elevation.size and float(np.max(np.abs(finite_elevation))) > 1.0e-12:
            return False, "Whitebox CostDistance does not apply 3D surface-distance elevation."
    return True, None


def whitebox_cost_distance_baseline(case: CaseData) -> npt.NDArray[np.float64]:
    try:
        import whitebox
    except ImportError as exc:
        raise RuntimeError(
            "Whitebox baseline requested but the 'whitebox' package is not installed. "
            "Run 'uv sync --group whitebox' or use "
            "'uv run --with whitebox python benchmarks/baseline.py --baselines whitebox ...'."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="distance-rs-whitebox-") as tmp_name:
        tmp = Path(tmp_name)
        source_path = tmp / "source.dep"
        cost_path = tmp / "cost.dep"
        accum_path = tmp / "accum.dep"
        backlink_path = tmp / "backlink.dep"

        valid_cost = np.isfinite(case.cost) & ~case.barriers
        source = np.full(case.sources.shape, WHITEBOX_NODATA, dtype=np.float64)
        source[(case.sources != 0.0) & np.isfinite(case.sources) & valid_cost] = 1.0
        cost = np.where(valid_cost, np.maximum(case.cost, MIN_COST), WHITEBOX_NODATA)

        write_whitebox_raster(source_path, source, cell_size=case.cell_size, nodata=WHITEBOX_NODATA)
        write_whitebox_raster(cost_path, cost, cell_size=case.cell_size, nodata=WHITEBOX_NODATA)

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

        return read_whitebox_raster(accum_path, nodata_as_inf=True)


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
    surface_distance = math.hypot(plan_distance, dz) if has_elevation and use_surface_distance else plan_distance
    angle = math.degrees(math.atan2(dz, plan_distance)) if has_elevation else 0.0
    vf = vertical_factor_multiplier(vertical_factor, angle)
    if not math.isfinite(vf):
        return math.inf
    return surface_distance * 0.5 * (float(cost[row, col]) + float(cost[next_row, next_col])) * vf


def vertical_factor_multiplier(vertical_factor: VerticalFactor, angle_degrees: float) -> float:
    vf = vertical_factor.normalized()
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
        value = hiking_pace(angle_degrees)
    elif vf.type == "bidir_hiking_time":
        value = 0.5 * (hiking_pace(angle_degrees) + hiking_pace(-angle_degrees))
    else:
        raise ValueError(f"unsupported vertical factor: {vf.type}")

    return value if math.isfinite(value) and value > 0.0 else math.inf


def hiking_pace(angle_degrees: float) -> float:
    slope = math.tan(math.radians(angle_degrees))
    speed_km_per_hour = 6.0 * math.exp(-3.5 * abs(slope + 0.05))
    return 1.0 / (speed_km_per_hour * 1000.0)


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
        denom = np.maximum(np.abs(expected[overlap]), 1.0e-12)
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


def time_call(fn: Callable[[], Any], *, repeats: int, warmups: int) -> TimedResult:
    for _ in range(warmups):
        fn()

    times: list[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return TimedResult(result=result, times_sec=times)


def normalize_cell_size(cell_size: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(cell_size, tuple):
        return float(cell_size[0]), float(cell_size[1])
    value = float(cell_size)
    return value, value


def print_summary(rows: list[dict[str, Any]]) -> None:
    headers = [
        "case",
        "size",
        "baseline",
        "ours_ms",
        "base_ms",
        "speedup",
        "rmse",
        "max_abs",
        "reach_mis",
        "reach_mis_rate",
        "status",
    ]
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        for baseline_name, baseline in row["baselines"].items():
            comparison = baseline.get("comparison")
            table_rows.append(
                {
                    "case": row["case"],
                    "size": row["size"],
                    "baseline": baseline_name,
                    "ours_ms": 1000.0 * row["ordered_upwind_median_sec"],
                    "base_ms": 1000.0 * baseline["median_sec"]
                    if baseline.get("status") == "ok"
                    else None,
                    "speedup": baseline.get("speedup"),
                    "rmse": comparison["rmse"] if comparison else None,
                    "max_abs": comparison["max_abs"] if comparison else None,
                    "reach_mis": comparison["reachability_mismatch"] if comparison else None,
                    "reach_mis_rate": comparison["reachability_mismatch_rate"]
                    if comparison
                    else None,
                    "status": baseline.get("status", "unknown"),
                }
            )

    rendered_rows = [
        [
            str(row["case"]),
            str(row["size"]),
            str(row["baseline"]),
            format_float(row["ours_ms"]),
            format_optional_float(row["base_ms"]),
            format_optional_float(row["speedup"]),
            format_optional_float(row["rmse"]),
            format_optional_float(row["max_abs"]),
            "n/a" if row["reach_mis"] is None else str(row["reach_mis"]),
            format_optional_float(row["reach_mis_rate"]),
            str(row["status"]),
        ]
        for row in table_rows
    ]
    widths = [
        max(len(header), *(len(rendered[index]) for rendered in rendered_rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for rendered in rendered_rows:
        print("  ".join(value.rjust(widths[index]) for index, value in enumerate(rendered)))

    reference_rows = [row for row in rows if row["reference"] is not None]
    if reference_rows:
        print("\nflat exact-reference errors:")
        for row in reference_rows:
            ours = row["reference"]
            print(
                f"  size={row['size']}: "
                f"ordered_upwind rmse={format_float(ours['rmse'])}"
            )
            for baseline_name, baseline in row["baselines"].items():
                reference = baseline.get("reference")
                if reference is not None:
                    print(
                        f"    {baseline_name} rmse={format_float(reference['rmse'])}"
                    )


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    if abs(value) >= 1000.0 or (abs(value) < 0.001 and value != 0.0):
        return f"{value:.3e}"
    return f"{value:.3f}"


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return format_float(value)


def write_json(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "sizes": args.sizes,
            "cases": expand_cases(args.cases),
            "baselines": expand_baselines(args.baselines),
        },
        "benchmarks": rows,
    }
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")


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


if __name__ == "__main__":
    raise SystemExit(main())
