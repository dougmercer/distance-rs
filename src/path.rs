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

#[derive(Clone, Copy, Debug)]
struct TracePoint {
    row: f64,
    col: f64,
}

impl TracePoint {
    fn at_cell_center(row: usize, col: usize) -> Self {
        Self {
            row: row as f64,
            col: col as f64,
        }
    }

    fn is_near(self, other: Self) -> bool {
        (self.row - other.row).abs() <= 1.0e-7 && (self.col - other.col).abs() <= 1.0e-7
    }
}

#[derive(Clone, Copy, Debug)]
struct TraceCursor {
    point: TracePoint,
    rows: usize,
    cols: usize,
}

impl TraceCursor {
    fn new(row: isize, col: isize, rows: usize, cols: usize) -> Self {
        Self {
            point: TracePoint {
                row: row as f64,
                col: col as f64,
            },
            rows,
            cols,
        }
    }

    fn current_cell(self) -> Result<(usize, usize), PathTraceError> {
        let row_idx = self.point.row.round() as isize;
        let col_idx = self.point.col.round() as isize;
        if row_idx < 0
            || col_idx < 0
            || row_idx >= self.rows as isize
            || col_idx >= self.cols as isize
        {
            return Err(PathTraceError::Runtime(
                "path tracing stepped outside the raster".to_string(),
            ));
        }
        Ok((row_idx as usize, col_idx as usize))
    }

    fn move_to(&mut self, point: TracePoint) {
        self.point = point;
    }
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
    let mut cursor = TraceCursor::new(request.start_row, request.start_col, rows, cols);
    let mut guard = 0usize;

    push_world_coord(
        &mut coords,
        cursor.point,
        request.cell_size_x,
        request.cell_size_y,
        request.origin_x,
        request.origin_y,
    );

    loop {
        let (current_row, current_col) = cursor.current_cell()?;
        if !request.distance[[current_row, current_col]].is_finite() {
            return Err(path_trace_error("entered a non-finite cell"));
        }

        let a = request.parent_a[[current_row, current_col]];
        if a < 0 {
            let source_center = TracePoint::at_cell_center(current_row, current_col);
            if !cursor.point.is_near(source_center) {
                push_world_coord(
                    &mut coords,
                    source_center,
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
        let previous = cursor.point;
        let proposed = next_trace_step(&cursor, degrees, request.cell_size_x, request.cell_size_y)?;
        let repaired = repair_blocked_step(
            &request.distance,
            previous,
            proposed,
            TracePoint::at_cell_center(current_row, current_col),
        )?;
        cursor.move_to(repaired);

        push_world_coord(
            &mut coords,
            cursor.point,
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

fn next_trace_step(
    cursor: &TraceCursor,
    degrees: f64,
    cell_size_x: f64,
    cell_size_y: f64,
) -> Result<TracePoint, PathTraceError> {
    let (dr, dc) = direction_vector(degrees, cell_size_x, cell_size_y);
    if dr.abs() <= 1.0e-12 && dc.abs() <= 1.0e-12 {
        return Err(path_trace_error("encountered a zero direction"));
    }

    let t = next_lattice_crossing(cursor.point.row, cursor.point.col, dr, dc)?;
    let mut next = TracePoint {
        row: cursor.point.row + dr * t,
        col: cursor.point.col + dc * t,
    };

    if next.row < -1.0e-7
        || next.col < -1.0e-7
        || next.row > (cursor.rows - 1) as f64 + 1.0e-7
        || next.col > (cursor.cols - 1) as f64 + 1.0e-7
    {
        return Err(path_trace_error("stepped outside the raster"));
    }

    next.row = next.row.clamp(0.0, (cursor.rows - 1) as f64);
    next.col = next.col.clamp(0.0, (cursor.cols - 1) as f64);
    Ok(next)
}

fn repair_blocked_step(
    distance: &ArrayView2<'_, f64>,
    previous: TracePoint,
    proposed: TracePoint,
    current_center: TracePoint,
) -> Result<TracePoint, PathTraceError> {
    if finite_segment_clear(distance, previous, proposed) {
        return Ok(proposed);
    }
    if !previous.is_near(current_center) && finite_segment_clear(distance, previous, current_center)
    {
        return Ok(current_center);
    }
    Err(path_trace_error("crossed a non-finite cell"))
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

fn finite_segment_clear(
    distance: &ArrayView2<'_, f64>,
    start: TracePoint,
    end: TracePoint,
) -> bool {
    let shape = distance.shape();
    grid_segment::segment_clear(
        shape[0],
        shape[1],
        start.row,
        start.col,
        end.row,
        end.col,
        |row, col| distance[[row, col]].is_finite(),
    )
}

fn push_world_coord(
    coords: &mut Vec<f64>,
    point: TracePoint,
    cell_size_x: f64,
    cell_size_y: f64,
    origin_x: f64,
    origin_y: f64,
) {
    coords.push(origin_x + point.col * cell_size_x);
    coords.push(origin_y + point.row * cell_size_y);
}
