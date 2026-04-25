use ndarray::ArrayView2;

use crate::grid_segment;

#[derive(Debug)]
pub(crate) enum PathTraceError {
    Value(String),
    Runtime(String),
}

pub(crate) struct TraceRequest<'a> {
    pub(crate) distance: ArrayView2<'a, f64>,
    pub(crate) valid: ArrayView2<'a, bool>,
    pub(crate) back_direction: ArrayView2<'a, f64>,
    pub(crate) parent_a: ArrayView2<'a, i64>,
    pub(crate) parent_b: ArrayView2<'a, i64>,
    pub(crate) parent_weight: ArrayView2<'a, f64>,
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

    if request.valid.shape() != shape
        || request.back_direction.shape() != shape
        || request.parent_a.shape() != shape
        || request.parent_b.shape() != shape
        || request.parent_weight.shape() != shape
    {
        return Err(PathTraceError::Value(
            "valid, direction, and parent arrays must match distance shape".to_string(),
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

    if !request.valid[[request.start_row as usize, request.start_col as usize]]
        || !request.distance[[request.start_row as usize, request.start_col as usize]].is_finite()
    {
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
    let mut segment_crossings = Vec::new();
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
        if !request.valid[[current_row, current_col]]
            || !request.distance[[current_row, current_col]].is_finite()
        {
            return Err(path_trace_error("entered a non-finite cell"));
        }

        let Some(parent_point) = parent_trace_point(&request, current_row, current_col)? else {
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
        };

        let degrees = request.back_direction[[current_row, current_col]];
        let next = next_clear_trace_step(
            &request,
            &cursor,
            degrees,
            parent_point,
            request.cell_size_x,
            request.cell_size_y,
            &mut segment_crossings,
        )?;
        cursor.move_to(next);

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

fn parent_trace_point(
    request: &TraceRequest<'_>,
    row: usize,
    col: usize,
) -> Result<Option<TracePoint>, PathTraceError> {
    let a = request.parent_a[[row, col]];
    if a < 0 {
        return Ok(None);
    }

    let shape = request.distance.shape();
    let point_a = trace_point_from_flat_index(a, shape[0], shape[1])?;
    let b = request.parent_b[[row, col]];
    if b < 0 {
        return Ok(Some(point_a));
    }

    let weight_a = request.parent_weight[[row, col]];
    if !weight_a.is_finite() {
        return Err(path_trace_error("encountered a non-finite parent weight"));
    }
    let point_b = trace_point_from_flat_index(b, shape[0], shape[1])?;
    let weight_b = 1.0 - weight_a;
    Ok(Some(TracePoint {
        row: weight_a * point_a.row + weight_b * point_b.row,
        col: weight_a * point_a.col + weight_b * point_b.col,
    }))
}

fn trace_point_from_flat_index(
    idx: i64,
    rows: usize,
    cols: usize,
) -> Result<TracePoint, PathTraceError> {
    if idx < 0 || idx as usize >= rows * cols {
        return Err(path_trace_error("encountered an out-of-bounds parent"));
    }
    let idx = idx as usize;
    Ok(TracePoint::at_cell_center(idx / cols, idx % cols))
}

fn next_clear_trace_step(
    request: &TraceRequest<'_>,
    cursor: &TraceCursor,
    degrees: f64,
    parent_point: TracePoint,
    cell_size_x: f64,
    cell_size_y: f64,
    segment_crossings: &mut Vec<f64>,
) -> Result<TracePoint, PathTraceError> {
    if degrees.is_finite() {
        if let Ok(proposed) = next_trace_step(cursor, degrees, cell_size_x, cell_size_y) {
            if let Some(step) = clear_continuable_step(
                request,
                cursor,
                proposed,
                cell_size_x,
                cell_size_y,
                segment_crossings,
            )? {
                return Ok(step);
            }
        }
    }

    let parent_step = next_trace_step_toward(cursor, parent_point)?;
    if let Some(step) = clear_continuable_step(
        request,
        cursor,
        parent_step,
        cell_size_x,
        cell_size_y,
        segment_crossings,
    )? {
        return Ok(step);
    }

    Err(path_trace_error("crossed a non-finite cell"))
}

fn clear_continuable_step(
    request: &TraceRequest<'_>,
    cursor: &TraceCursor,
    proposed: TracePoint,
    cell_size_x: f64,
    cell_size_y: f64,
    segment_crossings: &mut Vec<f64>,
) -> Result<Option<TracePoint>, PathTraceError> {
    if !trace_segment_clear(request, cursor.point, proposed, segment_crossings) {
        return Ok(None);
    }
    if trace_can_continue_from(
        request,
        proposed,
        cell_size_x,
        cell_size_y,
        segment_crossings,
    )? {
        return Ok(Some(proposed));
    }

    let (row, col) = cell_for_point(proposed, cursor.rows, cursor.cols)?;
    let center = TracePoint::at_cell_center(row, col);
    if !proposed.is_near(center)
        && trace_segment_clear(request, cursor.point, center, segment_crossings)
    {
        return Ok(Some(center));
    }
    Ok(None)
}

fn trace_can_continue_from(
    request: &TraceRequest<'_>,
    point: TracePoint,
    cell_size_x: f64,
    cell_size_y: f64,
    segment_crossings: &mut Vec<f64>,
) -> Result<bool, PathTraceError> {
    let shape = request.distance.shape();
    let (row, col) = cell_for_point(point, shape[0], shape[1])?;
    if !request.valid[[row, col]] || !request.distance[[row, col]].is_finite() {
        return Ok(false);
    }
    let Some(parent_point) = parent_trace_point(request, row, col)? else {
        return Ok(true);
    };

    let cursor = TraceCursor {
        point,
        rows: shape[0],
        cols: shape[1],
    };
    let degrees = request.back_direction[[row, col]];
    if degrees.is_finite() {
        if let Ok(proposed) = next_trace_step(&cursor, degrees, cell_size_x, cell_size_y) {
            if trace_segment_clear(request, point, proposed, segment_crossings) {
                return Ok(true);
            }
        }
    }

    let Ok(parent_step) = next_trace_step_toward(&cursor, parent_point) else {
        return Ok(false);
    };
    Ok(trace_segment_clear(
        request,
        point,
        parent_step,
        segment_crossings,
    ))
}

fn cell_for_point(
    point: TracePoint,
    rows: usize,
    cols: usize,
) -> Result<(usize, usize), PathTraceError> {
    let row_idx = point.row.round() as isize;
    let col_idx = point.col.round() as isize;
    if row_idx < 0 || col_idx < 0 || row_idx >= rows as isize || col_idx >= cols as isize {
        return Err(PathTraceError::Runtime(
            "path tracing stepped outside the raster".to_string(),
        ));
    }
    Ok((row_idx as usize, col_idx as usize))
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

fn next_trace_step_toward(
    cursor: &TraceCursor,
    target: TracePoint,
) -> Result<TracePoint, PathTraceError> {
    let dr = target.row - cursor.point.row;
    let dc = target.col - cursor.point.col;
    if dr.abs() <= 1.0e-12 && dc.abs() <= 1.0e-12 {
        return Err(path_trace_error("encountered a zero parent direction"));
    }

    let t = next_lattice_crossing(cursor.point.row, cursor.point.col, dr, dc)?.min(1.0);
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

fn trace_segment_clear(
    request: &TraceRequest<'_>,
    start: TracePoint,
    end: TracePoint,
    segment_crossings: &mut Vec<f64>,
) -> bool {
    let shape = request.distance.shape();
    grid_segment::segment_clear_with_crossings(
        shape[0],
        shape[1],
        start.row,
        start.col,
        end.row,
        end.col,
        segment_crossings,
        |row, col| request.valid[[row, col]],
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
