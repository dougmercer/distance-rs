use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use rayon::prelude::*;

use crate::grid::MIN_COST;
use crate::path::{trace_optimal_path, PathTraceError, TraceMetadata, TraceRequest};
use crate::solver::{SharedSolverInput, Solver, SolverInput};
use crate::vertical::VerticalFactor;

struct NativeRouteLeg {
    line: Vec<f64>,
    cost: f64,
    metadata: TraceMetadata,
}

#[derive(Clone, Copy)]
struct NativeRouteLegWindow {
    source: usize,
    destination: usize,
    row_min: usize,
    row_max: usize,
    col_min: usize,
    col_max: usize,
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn distance_accumulation<'py>(
    py: Python<'py>,
    source_cells: PyReadonlyArray2<'py, i64>,
    cost_surface: PyReadonlyArray2<'py, f64>,
    elevation: Option<PyReadonlyArray2<'py, f64>>,
    barriers: Option<PyReadonlyArray2<'py, bool>>,
    vertical_factor: &Bound<'py, PyDict>,
    cell_size_x: f64,
    cell_size_y: f64,
    target_cells: Option<PyReadonlyArray2<'py, i64>>,
    progress_callback: Option<Bound<'py, PyAny>>,
    progress_interval: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let source_cells = source_cells.as_array();
    let cost_surface = cost_surface.as_array();
    let elevation = elevation.as_ref().map(|array| array.as_array());
    let barriers = barriers.as_ref().map(|array| array.as_array());
    let target_cells = target_cells.as_ref().map(|array| array.as_array());

    let source_shape = source_cells.shape();
    if source_shape.len() != 2 || source_shape[1] != 2 {
        return Err(PyValueError::new_err("source_cells must have shape (n, 2)"));
    }
    if source_shape[0] == 0 {
        return Err(PyValueError::new_err(
            "at least one source cell is required",
        ));
    }

    let shape = cost_surface.shape();
    let rows = shape[0];
    let cols = shape[1];
    if let Some(elevation) = &elevation {
        if elevation.shape() != shape {
            return Err(PyValueError::new_err(
                "elevation must have the same shape as cost_surface",
            ));
        }
    }
    if let Some(barriers) = &barriers {
        if barriers.shape() != shape {
            return Err(PyValueError::new_err(
                "barriers must have the same shape as cost_surface",
            ));
        }
    }
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("input rasters must not be empty"));
    }
    if cell_size_x <= 0.0
        || cell_size_y <= 0.0
        || !cell_size_x.is_finite()
        || !cell_size_y.is_finite()
    {
        return Err(PyValueError::new_err(
            "cell sizes must be positive finite values",
        ));
    }
    let vf = VerticalFactor::from_py_dict(vertical_factor)?;

    let shared_input = shared_solver_input_from_arrays(
        rows,
        cols,
        cost_surface,
        elevation,
        barriers,
        vf,
        cell_size_x,
        cell_size_y,
    );

    let mut sources_flat = Vec::with_capacity(source_shape[0]);
    for source_index in 0..source_shape[0] {
        let row = source_cells[[source_index, 0]];
        let col = source_cells[[source_index, 1]];
        if row < 0 || col < 0 || row >= rows as i64 || col >= cols as i64 {
            return Err(PyValueError::new_err("source cell is outside the raster"));
        }
        sources_flat.push(row as usize * cols + col as usize);
    }

    let mut targets_flat = Vec::new();
    if let Some(target_cells) = target_cells {
        let target_shape = target_cells.shape();
        if target_shape.len() != 2 || target_shape[1] != 2 {
            return Err(PyValueError::new_err("target_cells must have shape (n, 2)"));
        }
        targets_flat.reserve(target_shape[0]);
        for target_index in 0..target_shape[0] {
            let row = target_cells[[target_index, 0]];
            let col = target_cells[[target_index, 1]];
            if row < 0 || col < 0 || row >= rows as i64 || col >= cols as i64 {
                return Err(PyValueError::new_err("target cell is outside the raster"));
            }
            targets_flat.push(row as usize * cols + col as usize);
        }
    }

    let solver = Solver::from_shared(shared_input);
    let output = solver.solve(
        &sources_flat,
        if targets_flat.is_empty() {
            None
        } else {
            Some(&targets_flat)
        },
        progress_callback.as_ref(),
        progress_interval,
    )?;

    let parent_a: Vec<i64> = output.parent.iter().map(|parent| parent.a).collect();
    let parent_b: Vec<i64> = output.parent.iter().map(|parent| parent.b).collect();
    let parent_weight: Vec<f64> = output.parent.iter().map(|parent| parent.weight).collect();

    let dict = PyDict::new(py);
    dict.set_item(
        "distance",
        Array2::from_shape_vec((rows, cols), output.distance)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "back_direction",
        Array2::from_shape_vec((rows, cols), output.back_direction)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "parent_a",
        Array2::from_shape_vec((rows, cols), parent_a)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "parent_b",
        Array2::from_shape_vec((rows, cols), parent_b)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "parent_weight",
        Array2::from_shape_vec((rows, cols), parent_weight)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn route_legs<'py>(
    py: Python<'py>,
    leg_cells: PyReadonlyArray2<'py, i64>,
    cost_surface: PyReadonlyArray2<'py, f64>,
    elevation: Option<PyReadonlyArray2<'py, f64>>,
    barriers: Option<PyReadonlyArray2<'py, bool>>,
    vertical_factor: &Bound<'py, PyDict>,
    cell_size_x: f64,
    cell_size_y: f64,
) -> PyResult<Bound<'py, PyList>> {
    let leg_cells = leg_cells.as_array();
    let leg_shape = leg_cells.shape();
    if leg_shape.len() != 2 || leg_shape[1] != 4 {
        return Err(PyValueError::new_err(
            "leg_cells must have shape (n, 4): source_row, source_col, destination_row, destination_col",
        ));
    }

    let cost_surface = cost_surface.as_array();
    let elevation = elevation.as_ref().map(|array| array.as_array());
    let barriers = barriers.as_ref().map(|array| array.as_array());
    let shape = cost_surface.shape();
    let rows = shape[0];
    let cols = shape[1];
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("input rasters must not be empty"));
    }
    if let Some(elevation) = &elevation {
        if elevation.shape() != shape {
            return Err(PyValueError::new_err(
                "elevation must have the same shape as cost_surface",
            ));
        }
    }
    if let Some(barriers) = &barriers {
        if barriers.shape() != shape {
            return Err(PyValueError::new_err(
                "barriers must have the same shape as cost_surface",
            ));
        }
    }
    if cell_size_x <= 0.0
        || cell_size_y <= 0.0
        || !cell_size_x.is_finite()
        || !cell_size_y.is_finite()
    {
        return Err(PyValueError::new_err(
            "cell sizes must be positive finite values",
        ));
    }

    let mut legs = Vec::with_capacity(leg_shape[0]);
    for leg_index in 0..leg_shape[0] {
        let source_row = leg_cells[[leg_index, 0]];
        let source_col = leg_cells[[leg_index, 1]];
        let destination_row = leg_cells[[leg_index, 2]];
        let destination_col = leg_cells[[leg_index, 3]];
        for (row, col, name) in [
            (source_row, source_col, "source"),
            (destination_row, destination_col, "destination"),
        ] {
            if row < 0 || col < 0 || row >= rows as i64 || col >= cols as i64 {
                return Err(PyValueError::new_err(format!(
                    "{name} cell is outside the raster"
                )));
            }
        }
        legs.push((
            source_row as usize * cols + source_col as usize,
            destination_row as usize * cols + destination_col as usize,
        ));
    }

    let vf = VerticalFactor::from_py_dict(vertical_factor)?;
    let shared_input = shared_solver_input_from_arrays(
        rows,
        cols,
        cost_surface,
        elevation,
        barriers,
        vf,
        cell_size_x,
        cell_size_y,
    );

    let solved = py
        .detach(|| solve_route_legs_parallel(shared_input, &legs))
        .map_err(PyRuntimeError::new_err)?;

    let output = PyList::empty(py);
    for leg in solved {
        let vertices = leg.line.len() / 2;
        let dict = PyDict::new(py);
        dict.set_item(
            "line",
            Array2::from_shape_vec((vertices, 2), leg.line)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                .into_pyarray(py),
        )?;
        dict.set_item("cost", leg.cost)?;
        dict.set_item("metadata", trace_metadata_dict(py, leg.metadata)?)?;
        output.append(dict)?;
    }
    Ok(output)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn route_legs_windowed<'py>(
    py: Python<'py>,
    leg_windows: PyReadonlyArray2<'py, i64>,
    cost_surface: PyReadonlyArray2<'py, f64>,
    elevation: Option<PyReadonlyArray2<'py, f64>>,
    barriers: Option<PyReadonlyArray2<'py, bool>>,
    vertical_factor: &Bound<'py, PyDict>,
    cell_size_x: f64,
    cell_size_y: f64,
) -> PyResult<Bound<'py, PyList>> {
    let leg_windows = leg_windows.as_array();
    let window_shape = leg_windows.shape();
    if window_shape.len() != 2 || window_shape[1] != 8 {
        return Err(PyValueError::new_err(
            "leg_windows must have shape (n, 8): source_row, source_col, destination_row, \
             destination_col, row_min, row_max, col_min, col_max",
        ));
    }

    let cost_surface = cost_surface.as_array();
    let elevation = elevation.as_ref().map(|array| array.as_array());
    let barriers = barriers.as_ref().map(|array| array.as_array());
    let shape = cost_surface.shape();
    let rows = shape[0];
    let cols = shape[1];
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("input rasters must not be empty"));
    }
    if let Some(elevation) = &elevation {
        if elevation.shape() != shape {
            return Err(PyValueError::new_err(
                "elevation must have the same shape as cost_surface",
            ));
        }
    }
    if let Some(barriers) = &barriers {
        if barriers.shape() != shape {
            return Err(PyValueError::new_err(
                "barriers must have the same shape as cost_surface",
            ));
        }
    }
    if cell_size_x <= 0.0
        || cell_size_y <= 0.0
        || !cell_size_x.is_finite()
        || !cell_size_y.is_finite()
    {
        return Err(PyValueError::new_err(
            "cell sizes must be positive finite values",
        ));
    }

    let mut windows = Vec::with_capacity(window_shape[0]);
    for leg_index in 0..window_shape[0] {
        let source_row = leg_windows[[leg_index, 0]];
        let source_col = leg_windows[[leg_index, 1]];
        let destination_row = leg_windows[[leg_index, 2]];
        let destination_col = leg_windows[[leg_index, 3]];
        let row_min = leg_windows[[leg_index, 4]];
        let row_max = leg_windows[[leg_index, 5]];
        let col_min = leg_windows[[leg_index, 6]];
        let col_max = leg_windows[[leg_index, 7]];

        if row_min < 0
            || col_min < 0
            || row_max <= row_min
            || col_max <= col_min
            || row_max > rows as i64
            || col_max > cols as i64
        {
            return Err(PyValueError::new_err(
                "route leg window is outside the raster",
            ));
        }
        for (row, col, name) in [
            (source_row, source_col, "source"),
            (destination_row, destination_col, "destination"),
        ] {
            if row < row_min || col < col_min || row >= row_max || col >= col_max {
                return Err(PyValueError::new_err(format!(
                    "{name} cell is outside its route leg window"
                )));
            }
        }

        windows.push(NativeRouteLegWindow {
            source: source_row as usize * cols + source_col as usize,
            destination: destination_row as usize * cols + destination_col as usize,
            row_min: row_min as usize,
            row_max: row_max as usize,
            col_min: col_min as usize,
            col_max: col_max as usize,
        });
    }

    let vf = VerticalFactor::from_py_dict(vertical_factor)?;
    let shared_input = shared_solver_input_from_arrays(
        rows,
        cols,
        cost_surface,
        elevation,
        barriers,
        vf,
        cell_size_x,
        cell_size_y,
    );

    let solved = py
        .detach(|| solve_route_leg_windows_parallel(shared_input, &windows))
        .map_err(PyRuntimeError::new_err)?;

    let output = PyList::empty(py);
    for leg in solved {
        let vertices = leg.line.len() / 2;
        let dict = PyDict::new(py);
        dict.set_item(
            "line",
            Array2::from_shape_vec((vertices, 2), leg.line)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                .into_pyarray(py),
        )?;
        dict.set_item("cost", leg.cost)?;
        dict.set_item("metadata", trace_metadata_dict(py, leg.metadata)?)?;
        output.append(dict)?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn shared_solver_input_from_arrays(
    rows: usize,
    cols: usize,
    cost_surface: ArrayView2<'_, f64>,
    elevation: Option<ArrayView2<'_, f64>>,
    barriers: Option<ArrayView2<'_, bool>>,
    vf: VerticalFactor,
    cell_size_x: f64,
    cell_size_y: f64,
) -> SharedSolverInput {
    let has_elevation = elevation.is_some();
    let mut cost = Vec::with_capacity(rows * cols);
    let mut elev = if has_elevation {
        Vec::with_capacity(rows * cols)
    } else {
        Vec::new()
    };
    let mut valid = Vec::with_capacity(rows * cols);
    let mut has_blocked_cells = false;

    for row in 0..rows {
        for col in 0..cols {
            let raw_cost = cost_surface[[row, col]];
            let raw_elevation = elevation
                .as_ref()
                .map_or(0.0, |elevation| elevation[[row, col]]);
            let blocked = barriers
                .as_ref()
                .is_some_and(|barriers| barriers[[row, col]])
                || !raw_cost.is_finite()
                || (has_elevation && !raw_elevation.is_finite());
            has_blocked_cells |= blocked;
            valid.push(!blocked);
            cost.push(if raw_cost.is_finite() {
                raw_cost.max(MIN_COST)
            } else {
                f64::INFINITY
            });
            if has_elevation {
                elev.push(if raw_elevation.is_finite() {
                    raw_elevation
                } else {
                    0.0
                });
            }
        }
    }

    SharedSolverInput::new(SolverInput {
        rows,
        cols,
        cost,
        elevation: elev,
        valid,
        has_blocked_cells,
        has_elevation,
        vf,
        cell_size_x,
        cell_size_y,
    })
}

fn solve_route_legs_parallel(
    shared_input: SharedSolverInput,
    legs: &[(usize, usize)],
) -> Result<Vec<NativeRouteLeg>, String> {
    legs.par_iter()
        .map(|&(source, target)| solve_route_leg(shared_input.clone(), source, target))
        .collect()
}

fn solve_route_leg(
    shared_input: SharedSolverInput,
    source: usize,
    target: usize,
) -> Result<NativeRouteLeg, String> {
    let rows = shared_input.rows;
    let cols = shared_input.cols;
    let cell_size_x = shared_input.cell_size_x;
    let cell_size_y = shared_input.cell_size_y;
    let solver = Solver::from_shared(shared_input.clone());
    let output = solver
        .solve(&[source], Some(&[target]), None, 1)
        .map_err(|err| err.to_string())?;
    let target_cost = output.distance[target];
    let parent_a: Vec<i64> = output.parent.iter().map(|parent| parent.a).collect();
    let parent_b: Vec<i64> = output.parent.iter().map(|parent| parent.b).collect();
    let parent_weight: Vec<f64> = output.parent.iter().map(|parent| parent.weight).collect();
    let distance =
        ArrayView2::from_shape((rows, cols), &output.distance).map_err(|err| err.to_string())?;
    let back_direction = ArrayView2::from_shape((rows, cols), &output.back_direction)
        .map_err(|err| err.to_string())?;
    let parent_a =
        ArrayView2::from_shape((rows, cols), &parent_a).map_err(|err| err.to_string())?;
    let parent_b =
        ArrayView2::from_shape((rows, cols), &parent_b).map_err(|err| err.to_string())?;
    let parent_weight =
        ArrayView2::from_shape((rows, cols), &parent_weight).map_err(|err| err.to_string())?;
    let valid = ArrayView2::from_shape((rows, cols), shared_input.barriers.valid())
        .map_err(|err| err.to_string())?;
    let trace = trace_optimal_path(TraceRequest {
        distance,
        valid,
        back_direction,
        parent_a,
        parent_b,
        parent_weight,
        destination_row: (target / cols) as isize,
        destination_col: (target % cols) as isize,
        cell_size_x,
        cell_size_y,
        origin_x: 0.0,
        origin_y: 0.0,
        max_steps: 0,
    })
    .map_err(path_trace_message)?;

    Ok(NativeRouteLeg {
        line: trace.coords,
        cost: target_cost,
        metadata: trace.metadata,
    })
}

fn solve_route_leg_windows_parallel(
    shared_input: SharedSolverInput,
    windows: &[NativeRouteLegWindow],
) -> Result<Vec<NativeRouteLeg>, String> {
    windows
        .par_iter()
        .map(|&window| solve_route_leg_window(shared_input.clone(), window))
        .collect()
}

fn solve_route_leg_window(
    shared_input: SharedSolverInput,
    window: NativeRouteLegWindow,
) -> Result<NativeRouteLeg, String> {
    let window_rows = window.row_max - window.row_min;
    let window_cols = window.col_max - window.col_min;
    let has_elevation = shared_input.has_elevation;
    let mut cost = Vec::with_capacity(window_rows * window_cols);
    let mut elevation = if has_elevation {
        Vec::with_capacity(window_rows * window_cols)
    } else {
        Vec::new()
    };
    let mut valid = Vec::with_capacity(window_rows * window_cols);
    let full_valid = shared_input.barriers.valid();

    for row in window.row_min..window.row_max {
        let start = row * shared_input.cols + window.col_min;
        let end = row * shared_input.cols + window.col_max;
        cost.extend_from_slice(&shared_input.cost[start..end]);
        valid.extend_from_slice(&full_valid[start..end]);
        if has_elevation {
            elevation.extend_from_slice(&shared_input.elevation[start..end]);
        }
    }

    let source_row = window.source / shared_input.cols - window.row_min;
    let source_col = window.source % shared_input.cols - window.col_min;
    let destination_row = window.destination / shared_input.cols - window.row_min;
    let destination_col = window.destination % shared_input.cols - window.col_min;
    let source = source_row * window_cols + source_col;
    let target = destination_row * window_cols + destination_col;
    let trace_valid = valid.clone();
    let solver = Solver::new(SolverInput {
        rows: window_rows,
        cols: window_cols,
        cost,
        elevation,
        valid,
        has_blocked_cells: false,
        has_elevation,
        vf: shared_input.vf,
        cell_size_x: shared_input.cell_size_x,
        cell_size_y: shared_input.cell_size_y,
    });

    let mut leg = solve_route_leg_from_solver(
        solver,
        source,
        target,
        window_rows,
        window_cols,
        trace_valid,
        shared_input.cell_size_x,
        shared_input.cell_size_y,
    )?;
    let offset_x = window.col_min as f64 * shared_input.cell_size_x;
    let offset_y = window.row_min as f64 * shared_input.cell_size_y;
    for coord in leg.line.chunks_exact_mut(2) {
        coord[0] += offset_x;
        coord[1] += offset_y;
    }
    Ok(leg)
}

fn solve_route_leg_from_solver(
    solver: Solver,
    source: usize,
    target: usize,
    rows: usize,
    cols: usize,
    valid: Vec<bool>,
    cell_size_x: f64,
    cell_size_y: f64,
) -> Result<NativeRouteLeg, String> {
    let output = solver
        .solve(&[source], Some(&[target]), None, 1)
        .map_err(|err| err.to_string())?;
    let target_cost = output.distance[target];
    let parent_a: Vec<i64> = output.parent.iter().map(|parent| parent.a).collect();
    let parent_b: Vec<i64> = output.parent.iter().map(|parent| parent.b).collect();
    let parent_weight: Vec<f64> = output.parent.iter().map(|parent| parent.weight).collect();
    let distance =
        ArrayView2::from_shape((rows, cols), &output.distance).map_err(|err| err.to_string())?;
    let back_direction = ArrayView2::from_shape((rows, cols), &output.back_direction)
        .map_err(|err| err.to_string())?;
    let parent_a =
        ArrayView2::from_shape((rows, cols), &parent_a).map_err(|err| err.to_string())?;
    let parent_b =
        ArrayView2::from_shape((rows, cols), &parent_b).map_err(|err| err.to_string())?;
    let parent_weight =
        ArrayView2::from_shape((rows, cols), &parent_weight).map_err(|err| err.to_string())?;
    let valid = ArrayView2::from_shape((rows, cols), &valid).map_err(|err| err.to_string())?;
    let trace = trace_optimal_path(TraceRequest {
        distance,
        valid,
        back_direction,
        parent_a,
        parent_b,
        parent_weight,
        destination_row: (target / cols) as isize,
        destination_col: (target % cols) as isize,
        cell_size_x,
        cell_size_y,
        origin_x: 0.0,
        origin_y: 0.0,
        max_steps: 0,
    })
    .map_err(path_trace_message)?;

    Ok(NativeRouteLeg {
        line: trace.coords,
        cost: target_cost,
        metadata: trace.metadata,
    })
}

fn trace_metadata_dict(py: Python<'_>, metadata: TraceMetadata) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("direction_steps", metadata.direction_steps)?;
    dict.set_item(
        "parent_lattice_fallbacks",
        metadata.parent_lattice_fallbacks,
    )?;
    dict.set_item(
        "proposed_cell_center_fallbacks",
        metadata.proposed_cell_center_fallbacks,
    )?;
    dict.set_item(
        "current_cell_center_fallbacks",
        metadata.current_cell_center_fallbacks,
    )?;
    dict.set_item(
        "direct_parent_point_fallbacks",
        metadata.direct_parent_point_fallbacks,
    )?;
    dict.set_item(
        "non_descending_rejections",
        metadata.non_descending_rejections,
    )?;
    dict.set_item(
        "total_fallbacks",
        metadata.parent_lattice_fallbacks
            + metadata.proposed_cell_center_fallbacks
            + metadata.current_cell_center_fallbacks
            + metadata.direct_parent_point_fallbacks,
    )?;
    Ok(dict)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn optimal_path_as_line<'py>(
    py: Python<'py>,
    distance: PyReadonlyArray2<'py, f64>,
    valid: PyReadonlyArray2<'py, bool>,
    back_direction: PyReadonlyArray2<'py, f64>,
    parent_a: PyReadonlyArray2<'py, i64>,
    parent_b: PyReadonlyArray2<'py, i64>,
    parent_weight: PyReadonlyArray2<'py, f64>,
    destination_row: isize,
    destination_col: isize,
    cell_size_x: f64,
    cell_size_y: f64,
    origin_x: f64,
    origin_y: f64,
    max_steps: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let output = trace_optimal_path(TraceRequest {
        distance: distance.as_array(),
        valid: valid.as_array(),
        back_direction: back_direction.as_array(),
        parent_a: parent_a.as_array(),
        parent_b: parent_b.as_array(),
        parent_weight: parent_weight.as_array(),
        destination_row,
        destination_col,
        cell_size_x,
        cell_size_y,
        origin_x,
        origin_y,
        max_steps,
    })
    .map_err(path_trace_py_error)?;

    let coords = output.coords;
    let vertices = coords.len() / 2;
    Ok(Array2::from_shape_vec((vertices, 2), coords)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
        .into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn optimal_path_trace<'py>(
    py: Python<'py>,
    distance: PyReadonlyArray2<'py, f64>,
    valid: PyReadonlyArray2<'py, bool>,
    back_direction: PyReadonlyArray2<'py, f64>,
    parent_a: PyReadonlyArray2<'py, i64>,
    parent_b: PyReadonlyArray2<'py, i64>,
    parent_weight: PyReadonlyArray2<'py, f64>,
    destination_row: isize,
    destination_col: isize,
    cell_size_x: f64,
    cell_size_y: f64,
    origin_x: f64,
    origin_y: f64,
    max_steps: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let output = trace_optimal_path(TraceRequest {
        distance: distance.as_array(),
        valid: valid.as_array(),
        back_direction: back_direction.as_array(),
        parent_a: parent_a.as_array(),
        parent_b: parent_b.as_array(),
        parent_weight: parent_weight.as_array(),
        destination_row,
        destination_col,
        cell_size_x,
        cell_size_y,
        origin_x,
        origin_y,
        max_steps,
    })
    .map_err(path_trace_py_error)?;

    let coords = output.coords;
    let vertices = coords.len() / 2;
    let line = Array2::from_shape_vec((vertices, 2), coords)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
        .into_pyarray(py);
    let metadata = PyDict::new(py);
    metadata.set_item("direction_steps", output.metadata.direction_steps)?;
    metadata.set_item(
        "parent_lattice_fallbacks",
        output.metadata.parent_lattice_fallbacks,
    )?;
    metadata.set_item(
        "proposed_cell_center_fallbacks",
        output.metadata.proposed_cell_center_fallbacks,
    )?;
    metadata.set_item(
        "current_cell_center_fallbacks",
        output.metadata.current_cell_center_fallbacks,
    )?;
    metadata.set_item(
        "direct_parent_point_fallbacks",
        output.metadata.direct_parent_point_fallbacks,
    )?;
    metadata.set_item(
        "non_descending_rejections",
        output.metadata.non_descending_rejections,
    )?;
    let total_fallbacks = output.metadata.parent_lattice_fallbacks
        + output.metadata.proposed_cell_center_fallbacks
        + output.metadata.current_cell_center_fallbacks
        + output.metadata.direct_parent_point_fallbacks;
    metadata.set_item("total_fallbacks", total_fallbacks)?;

    let dict = PyDict::new(py);
    dict.set_item("line", line)?;
    dict.set_item("metadata", metadata)?;
    Ok(dict)
}

fn path_trace_py_error(err: PathTraceError) -> PyErr {
    match err {
        PathTraceError::Value(message) => PyValueError::new_err(message),
        PathTraceError::Runtime(message) => PyRuntimeError::new_err(message),
    }
}

fn path_trace_message(err: PathTraceError) -> String {
    match err {
        PathTraceError::Value(message) | PathTraceError::Runtime(message) => message,
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_accumulation, m)?)?;
    m.add_function(wrap_pyfunction!(route_legs, m)?)?;
    m.add_function(wrap_pyfunction!(route_legs_windowed, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_path_as_line, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_path_trace, m)?)?;
    Ok(())
}
