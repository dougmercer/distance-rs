use std::f64::consts::PI;

use ndarray::ArrayView2;

use crate::grid_segment;
use crate::solver::SolveOutput;

impl SolveOutput {
    pub(crate) fn back_direction(&self) -> Vec<f64> {
        let mut back_direction = vec![f64::NAN; self.rows * self.cols];
        for (idx, direction) in back_direction.iter_mut().enumerate() {
            let parent = self.parent[idx];
            if parent.a < 0 {
                continue;
            }

            let row = idx / self.cols;
            let col = idx % self.cols;
            let (parent_row, parent_col) = if parent.b < 0 {
                let a = parent.a as usize;
                ((a / self.cols) as f64, (a % self.cols) as f64)
            } else {
                let a = parent.a as usize;
                let b = parent.b as usize;
                let weight_b = 1.0 - parent.weight;
                (
                    parent.weight * (a / self.cols) as f64 + weight_b * (b / self.cols) as f64,
                    parent.weight * (a % self.cols) as f64 + weight_b * (b % self.cols) as f64,
                )
            };

            let d_x = (parent_col - col as f64) * self.cell_size_x;
            let d_y = (parent_row - row as f64) * self.cell_size_y;
            let mut degrees = d_x.atan2(-d_y) * 180.0 / PI;
            if degrees < 0.0 {
                degrees += 360.0;
            }
            *direction = degrees;
        }
        back_direction
    }
}

#[derive(Debug)]
pub(crate) enum PathTraceError {
    Value(String),
    Runtime(String),
}

pub(crate) struct TraceRequest<'a> {
    pub(crate) distance: ArrayView2<'a, f64>,
    pub(crate) back_direction: ArrayView2<'a, f64>,
    pub(crate) parent_a: ArrayView2<'a, i64>,
    pub(crate) start_row: isize,
    pub(crate) start_col: isize,
    pub(crate) cell_size_x: f64,
    pub(crate) cell_size_y: f64,
    pub(crate) origin_x: f64,
    pub(crate) origin_y: f64,
    pub(crate) max_steps: usize,
}

pub(crate) fn trace_optimal_path(request: TraceRequest<'_>) -> Result<Vec<f64>, PathTraceError> {
    let shape = request.distance.shape();
    let rows = shape[0];
    let cols = shape[1];

    if request.back_direction.shape() != shape || request.parent_a.shape() != shape {
        return Err(PathTraceError::Value(
            "direction and parent arrays must match distance shape".to_string(),
        ));
    }
    if request.start_row < 0
        || request.start_col < 0
        || request.start_row >= rows as isize
        || request.start_col >= cols as isize
    {
        return Err(PathTraceError::Value(
            "destination is outside the raster".to_string(),
        ));
    }

    if !request.distance[[request.start_row as usize, request.start_col as usize]].is_finite() {
        return Err(PathTraceError::Value(
            "destination has no finite accumulated distance".to_string(),
        ));
    }

    let step_limit = if request.max_steps == 0 {
        rows.saturating_mul(cols).saturating_mul(4).max(1)
    } else {
        request.max_steps
    };
    let mut coords = Vec::with_capacity(step_limit.min(1024) * 2);
    let mut row = request.start_row as f64;
    let mut col = request.start_col as f64;
    let mut guard = 0usize;

    push_coord(
        &mut coords,
        row,
        col,
        request.cell_size_x,
        request.cell_size_y,
        request.origin_x,
        request.origin_y,
    );

    loop {
        let current = nearest_cell_in_direction(row, col, 0.0, 0.0, rows, cols)?;
        let current_row = current / cols;
        let current_col = current % cols;
        if !request.distance[[current_row, current_col]].is_finite() {
            return Err(path_trace_error("entered a non-finite cell"));
        }

        let a = request.parent_a[[current_row, current_col]];
        if a < 0 {
            if (row - current_row as f64).abs() > 1.0e-7
                || (col - current_col as f64).abs() > 1.0e-7
            {
                push_coord(
                    &mut coords,
                    current_row as f64,
                    current_col as f64,
                    request.cell_size_x,
                    request.cell_size_y,
                    request.origin_x,
                    request.origin_y,
                );
            }
            break;
        }

        let degrees = request.back_direction[[current_row, current_col]];
        if !degrees.is_finite() {
            return Err(path_trace_error("encountered a non-finite direction"));
        }
        let (dr, dc) = direction_vector(degrees, request.cell_size_x, request.cell_size_y);
        if dr.abs() <= 1.0e-12 && dc.abs() <= 1.0e-12 {
            return Err(path_trace_error("encountered a zero direction"));
        }

        let previous_row = row;
        let previous_col = col;
        let t = next_lattice_crossing(row, col, dr, dc)?;
        row += dr * t;
        col += dc * t;

        if row < -1.0e-7
            || col < -1.0e-7
            || row > (rows - 1) as f64 + 1.0e-7
            || col > (cols - 1) as f64 + 1.0e-7
        {
            return Err(path_trace_error("stepped outside the raster"));
        }
        row = row.clamp(0.0, (rows - 1) as f64);
        col = col.clamp(0.0, (cols - 1) as f64);
        if !finite_segment_clear(&request.distance, previous_row, previous_col, row, col) {
            let center_row = current_row as f64;
            let center_col = current_col as f64;
            if ((previous_row - center_row).abs() > 1.0e-7
                || (previous_col - center_col).abs() > 1.0e-7)
                && finite_segment_clear(
                    &request.distance,
                    previous_row,
                    previous_col,
                    center_row,
                    center_col,
                )
            {
                row = center_row;
                col = center_col;
            } else {
                return Err(path_trace_error("crossed a non-finite cell"));
            }
        }

        push_coord(
            &mut coords,
            row,
            col,
            request.cell_size_x,
            request.cell_size_y,
            request.origin_x,
            request.origin_y,
        );

        guard += 1;
        if guard >= step_limit {
            return Err(PathTraceError::Runtime(
                "path tracing exceeded max_steps before reaching a source".to_string(),
            ));
        }
    }

    Ok(coords)
}

fn path_trace_error(reason: &str) -> PathTraceError {
    PathTraceError::Runtime(format!("path tracing {reason}"))
}

fn direction_vector(degrees: f64, cell_size_x: f64, cell_size_y: f64) -> (f64, f64) {
    let radians = degrees.to_radians();
    (-radians.cos() / cell_size_y, radians.sin() / cell_size_x)
}

fn next_lattice_crossing(row: f64, col: f64, dr: f64, dc: f64) -> Result<f64, PathTraceError> {
    let row_t = next_axis_crossing(row, dr);
    let col_t = next_axis_crossing(col, dc);
    let t = row_t.min(col_t);
    if t.is_finite() && t > 1.0e-10 {
        Ok(t)
    } else {
        Err(PathTraceError::Runtime(
            "path tracing could not find the next lattice crossing".to_string(),
        ))
    }
}

fn next_axis_crossing(value: f64, delta: f64) -> f64 {
    if delta.abs() <= 1.0e-12 {
        return f64::INFINITY;
    }

    let nearest = value.round();
    let target = if (value - nearest).abs() <= 1.0e-9 {
        nearest + delta.signum()
    } else if delta > 0.0 {
        value.floor() + 1.0
    } else {
        value.ceil() - 1.0
    };
    (target - value) / delta
}

fn nearest_cell_in_direction(
    row: f64,
    col: f64,
    dr: f64,
    dc: f64,
    rows: usize,
    cols: usize,
) -> Result<usize, PathTraceError> {
    let sample_row = row + dr * 1.0e-7;
    let sample_col = col + dc * 1.0e-7;
    let row_idx = sample_row.round() as isize;
    let col_idx = sample_col.round() as isize;
    if row_idx < 0 || col_idx < 0 || row_idx >= rows as isize || col_idx >= cols as isize {
        return Err(PathTraceError::Runtime(
            "path tracing stepped outside the raster".to_string(),
        ));
    }
    Ok(row_idx as usize * cols + col_idx as usize)
}

fn finite_segment_clear(
    distance: &ArrayView2<'_, f64>,
    row0: f64,
    col0: f64,
    row1: f64,
    col1: f64,
) -> bool {
    let shape = distance.shape();
    grid_segment::segment_clear(shape[0], shape[1], row0, col0, row1, col1, |row, col| {
        distance[[row, col]].is_finite()
    })
}

fn push_coord(
    coords: &mut Vec<f64>,
    row: f64,
    col: f64,
    cell_size_x: f64,
    cell_size_y: f64,
    origin_x: f64,
    origin_y: f64,
) {
    coords.push(origin_x + col * cell_size_x);
    coords.push(origin_y + row * cell_size_y);
}
