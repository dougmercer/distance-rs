use std::cmp::Ordering;

use crate::grid::{Grid, EPS, GRID_EPS};

#[derive(Clone, Debug)]
pub(crate) struct BarrierMask {
    valid: Vec<bool>,
    blocked_prefix: Vec<usize>,
    has_blocked_cells: bool,
}

impl BarrierMask {
    pub(crate) fn new(
        rows: usize,
        cols: usize,
        mut valid: Vec<bool>,
        has_blocked_cells: bool,
    ) -> Self {
        if has_blocked_cells {
            connect_diagonal_barriers(rows, cols, &mut valid);
        }
        let has_blocked_cells = valid.iter().any(|cell| !*cell);
        let blocked_prefix = if has_blocked_cells {
            build_blocked_prefix(rows, cols, &valid)
        } else {
            Vec::new()
        };
        Self {
            valid,
            blocked_prefix,
            has_blocked_cells,
        }
    }

    pub(crate) fn is_valid(&self, idx: usize) -> bool {
        self.valid[idx]
    }

    pub(crate) fn has_blocked_cells(&self) -> bool {
        self.has_blocked_cells
    }

    pub(crate) fn segment_clear_index_to_index(&self, grid: &Grid, a: usize, b: usize) -> bool {
        if !self.has_blocked_cells {
            return true;
        }
        let (a_row, a_col) = grid.row_col(a);
        self.segment_clear_coord_to_index(grid, a_row as f64, a_col as f64, b)
    }

    pub(crate) fn segment_clear_coord_to_index(
        &self,
        grid: &Grid,
        row0: f64,
        col0: f64,
        idx: usize,
    ) -> bool {
        if !self.has_blocked_cells {
            return true;
        }
        let (row1, col1) = grid.row_col(idx);
        if self.segment_bounds_clear(grid, row0, col0, row1, col1) {
            return true;
        }
        self.segment_grid_clear(grid, row0, col0, row1 as f64, col1 as f64)
    }

    fn segment_grid_clear(&self, grid: &Grid, row0: f64, col0: f64, row1: f64, col1: f64) -> bool {
        if !row0.is_finite() || !col0.is_finite() || !row1.is_finite() || !col1.is_finite() {
            return false;
        }

        let x0 = col0 + 0.5;
        let y0 = row0 + 0.5;
        let x1 = col1 + 0.5;
        let y1 = row1 + 0.5;
        let dx = x1 - x0;
        let dy = y1 - y0;
        let mut crossings = Vec::new();

        push_axis_crossings(&mut crossings, x0, dx);
        push_axis_crossings(&mut crossings, y0, dy);
        crossings.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        crossings.dedup_by(|a, b| (*a - *b).abs() <= GRID_EPS);

        if !self.segment_point_clear(grid, x0, y0) || !self.segment_point_clear(grid, x1, y1) {
            return false;
        }
        for &t in &crossings {
            if !self.segment_point_clear(grid, x0 + dx * t, y0 + dy * t) {
                return false;
            }
        }

        let mut previous = 0.0;
        for next in crossings.into_iter().chain(std::iter::once(1.0)) {
            if next - previous > GRID_EPS {
                let midpoint = 0.5 * (previous + next);
                if !self.segment_cell_clear(grid, x0 + dx * midpoint, y0 + dy * midpoint) {
                    return false;
                }
            }
            previous = next;
        }
        true
    }

    fn segment_point_clear(&self, grid: &Grid, x: f64, y: f64) -> bool {
        let col_a = x.floor() as isize;
        let col_b = (x - GRID_EPS).floor() as isize;
        let row_a = y.floor() as isize;
        let row_b = (y - GRID_EPS).floor() as isize;
        let mut touched_any_cell = false;

        for row in [row_a, row_b] {
            for col in [col_a, col_b] {
                if row < 0 || col < 0 || row >= grid.rows as isize || col >= grid.cols as isize {
                    continue;
                }
                touched_any_cell = true;
                if !self.valid[grid.idx(row as usize, col as usize)] {
                    return false;
                }
            }
        }
        touched_any_cell
    }

    fn segment_cell_clear(&self, grid: &Grid, x: f64, y: f64) -> bool {
        let row = y.floor() as isize;
        let col = x.floor() as isize;
        if row < 0 || col < 0 || row >= grid.rows as isize || col >= grid.cols as isize {
            return false;
        }
        self.valid[grid.idx(row as usize, col as usize)]
    }

    fn segment_bounds_clear(
        &self,
        grid: &Grid,
        row0: f64,
        col0: f64,
        row1: usize,
        col1: usize,
    ) -> bool {
        if !row0.is_finite() || !col0.is_finite() {
            return false;
        }
        let max_row = (grid.rows - 1) as f64;
        let max_col = (grid.cols - 1) as f64;
        if row0 < 0.0 || col0 < 0.0 || row0 > max_row || col0 > max_col {
            return false;
        }

        let row_min = row0.min(row1 as f64).floor().max(0.0) as usize;
        let row_max = row0.max(row1 as f64).ceil().min(max_row) as usize;
        let col_min = col0.min(col1 as f64).floor().max(0.0) as usize;
        let col_max = col0.max(col1 as f64).ceil().min(max_col) as usize;
        self.blocked_count_in(grid, row_min, row_max, col_min, col_max) == 0
    }

    fn blocked_count_in(
        &self,
        grid: &Grid,
        row_min: usize,
        row_max: usize,
        col_min: usize,
        col_max: usize,
    ) -> usize {
        let stride = grid.cols + 1;
        let row0 = row_min;
        let row1 = row_max + 1;
        let col0 = col_min;
        let col1 = col_max + 1;
        let total = self.blocked_prefix[row1 * stride + col1] as isize
            - self.blocked_prefix[row0 * stride + col1] as isize
            - self.blocked_prefix[row1 * stride + col0] as isize
            + self.blocked_prefix[row0 * stride + col0] as isize;
        total.max(0) as usize
    }
}

fn push_axis_crossings(crossings: &mut Vec<f64>, start: f64, delta: f64) {
    if delta.abs() <= EPS {
        return;
    }

    let end = start + delta;
    let min_boundary = start.min(end).floor() as isize + 1;
    let max_boundary = start.max(end).floor() as isize;
    for boundary in min_boundary..=max_boundary {
        let t = (boundary as f64 - start) / delta;
        if t > GRID_EPS && t < 1.0 - GRID_EPS {
            crossings.push(t);
        }
    }
}

fn build_blocked_prefix(rows: usize, cols: usize, valid: &[bool]) -> Vec<usize> {
    let stride = cols + 1;
    let mut prefix = vec![0; (rows + 1) * stride];
    for row in 0..rows {
        let mut row_blocked = 0usize;
        for col in 0..cols {
            if !valid[row * cols + col] {
                row_blocked += 1;
            }
            prefix[(row + 1) * stride + col + 1] = prefix[row * stride + col + 1] + row_blocked;
        }
    }
    prefix
}

fn connect_diagonal_barriers(rows: usize, cols: usize, valid: &mut [bool]) {
    if rows < 2 || cols < 2 {
        return;
    }

    let original = valid.to_vec();
    let mut extra_blocked = vec![false; valid.len()];
    for row in 0..rows - 1 {
        for col in 0..cols - 1 {
            let nw = row * cols + col;
            let ne = nw + 1;
            let sw = (row + 1) * cols + col;
            let se = sw + 1;

            if !original[nw] && !original[se] && original[ne] && original[sw] {
                extra_blocked[ne] = true;
                extra_blocked[sw] = true;
            }
            if !original[ne] && !original[sw] && original[nw] && original[se] {
                extra_blocked[nw] = true;
                extra_blocked[se] = true;
            }
        }
    }

    for (is_valid, should_block) in valid.iter_mut().zip(extra_blocked) {
        if should_block {
            *is_valid = false;
        }
    }
}
