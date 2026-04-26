from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

def distance_accumulation(
    source_cells: npt.NDArray[np.int64],
    cost: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64] | None,
    barriers: npt.NDArray[np.bool_] | None,
    vertical_factor: Mapping[str, float | str],
    cell_size_x: float,
    cell_size_y: float,
    target_cells: npt.NDArray[np.int64] | None,
    progress: Callable[[int, int], None] | None,
    progress_interval: int,
) -> dict[str, Any]: ...
def route_legs(
    legs: npt.NDArray[np.int64],
    cost: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64] | None,
    barriers: npt.NDArray[np.bool_] | None,
    vertical_factor: Mapping[str, float | str],
    cell_size_x: float,
    cell_size_y: float,
) -> list[dict[str, Any]]: ...
def route_legs_windowed(
    leg_windows: npt.NDArray[np.int64],
    cost: npt.NDArray[np.float64],
    elevation: npt.NDArray[np.float64] | None,
    barriers: npt.NDArray[np.bool_] | None,
    vertical_factor: Mapping[str, float | str],
    cell_size_x: float,
    cell_size_y: float,
) -> list[dict[str, Any]]: ...
def optimal_path_as_line(
    distance: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    back_direction: npt.NDArray[np.float64],
    parent_a: npt.NDArray[np.int64],
    parent_b: npt.NDArray[np.int64],
    parent_weight: npt.NDArray[np.float64],
    row: int,
    col: int,
    cell_size_x: float,
    cell_size_y: float,
    origin_x: float,
    origin_y: float,
    max_steps: int,
) -> npt.NDArray[np.float64]: ...
def optimal_path_trace(
    distance: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    back_direction: npt.NDArray[np.float64],
    parent_a: npt.NDArray[np.int64],
    parent_b: npt.NDArray[np.int64],
    parent_weight: npt.NDArray[np.float64],
    row: int,
    col: int,
    cell_size_x: float,
    cell_size_y: float,
    origin_x: float,
    origin_y: float,
    max_steps: int,
) -> dict[str, Any]: ...
