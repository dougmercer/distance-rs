use std::f64::consts::PI;

use ndarray::ArrayView2;

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

pub(crate) fn trace_optimal_path(request: TraceRequest<'_>) -> Result<Vec<f64>, PathTraceError> {
    let shape = request.distance.shape();
    let rows = shape[0];
    let cols = shape[1];

    if request.parent_a.shape() != shape
        || request.parent_b.shape() != shape
        || request.parent_weight.shape() != shape
    {
        return Err(PathTraceError::Value(
            "parent arrays must match distance shape".to_string(),
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

    let mut idx = request.start_row as usize * cols + request.start_col as usize;
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
    let mut guard = 0usize;

    loop {
        let row = idx / cols;
        let col = idx % cols;
        push_coord(
            &mut coords,
            row as f64,
            col as f64,
            request.cell_size_x,
            request.cell_size_y,
            request.origin_x,
            request.origin_y,
        );

        let a = request.parent_a[[row, col]];
        if a < 0 {
            break;
        }
        let b = request.parent_b[[row, col]];
        let weight = request.parent_weight[[row, col]];
        if b < 0 {
            idx = a as usize;
        } else {
            let a_idx = a as usize;
            let b_idx = b as usize;
            let a_row = a_idx / cols;
            let a_col = a_idx % cols;
            let b_row = b_idx / cols;
            let b_col = b_idx % cols;
            let weight_b = 1.0 - weight;
            let interp_row = weight * a_row as f64 + weight_b * b_row as f64;
            let interp_col = weight * a_col as f64 + weight_b * b_col as f64;
            push_coord(
                &mut coords,
                interp_row,
                interp_col,
                request.cell_size_x,
                request.cell_size_y,
                request.origin_x,
                request.origin_y,
            );

            let a_dist = request.distance[[a_row, a_col]];
            let b_dist = request.distance[[b_row, b_col]];
            idx = if a_dist <= b_dist { a_idx } else { b_idx };
        }

        guard += 1;
        if guard >= step_limit {
            return Err(PathTraceError::Runtime(
                "path tracing exceeded max_steps before reaching a source".to_string(),
            ));
        }
    }

    Ok(coords)
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
