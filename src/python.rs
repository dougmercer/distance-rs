use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::grid::MIN_COST;
use crate::path::{trace_optimal_path, PathTraceError, TraceRequest};
use crate::solver::{Solver, SolverInput};
use crate::vertical::VerticalFactor;

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
    let has_elevation = elevation.is_some();
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

    let solver = Solver::new(SolverInput {
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
    });
    let output = solver.solve(
        &sources_flat,
        if targets_flat.is_empty() {
            None
        } else {
            Some(&targets_flat)
        },
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
fn optimal_path_as_line<'py>(
    py: Python<'py>,
    distance: PyReadonlyArray2<'py, f64>,
    valid: PyReadonlyArray2<'py, bool>,
    back_direction: PyReadonlyArray2<'py, f64>,
    parent_a: PyReadonlyArray2<'py, i64>,
    parent_b: PyReadonlyArray2<'py, i64>,
    parent_weight: PyReadonlyArray2<'py, f64>,
    start_row: isize,
    start_col: isize,
    cell_size_x: f64,
    cell_size_y: f64,
    origin_x: f64,
    origin_y: f64,
    max_steps: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let coords = trace_optimal_path(TraceRequest {
        distance: distance.as_array(),
        valid: valid.as_array(),
        back_direction: back_direction.as_array(),
        parent_a: parent_a.as_array(),
        parent_b: parent_b.as_array(),
        parent_weight: parent_weight.as_array(),
        start_row,
        start_col,
        cell_size_x,
        cell_size_y,
        origin_x,
        origin_y,
        max_steps,
    })
    .map_err(|err| match err {
        PathTraceError::Value(message) => PyValueError::new_err(message),
        PathTraceError::Runtime(message) => PyRuntimeError::new_err(message),
    })?;

    let vertices = coords.len() / 2;
    Ok(Array2::from_shape_vec((vertices, 2), coords)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
        .into_pyarray(py))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_accumulation, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_path_as_line, m)?)?;
    Ok(())
}
