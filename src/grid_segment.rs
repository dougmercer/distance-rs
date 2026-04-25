use std::cmp::Ordering;

use crate::grid::{EPS, GRID_EPS};

#[allow(clippy::too_many_arguments)]
pub(crate) fn segment_clear_with_crossings<F>(
    rows: usize,
    cols: usize,
    row0: f64,
    col0: f64,
    row1: f64,
    col1: f64,
    crossings: &mut Vec<f64>,
    mut cell_clear: F,
) -> bool
where
    F: FnMut(usize, usize) -> bool,
{
    if !row0.is_finite() || !col0.is_finite() || !row1.is_finite() || !col1.is_finite() {
        return false;
    }

    let x0 = col0 + 0.5;
    let y0 = row0 + 0.5;
    let x1 = col1 + 0.5;
    let y1 = row1 + 0.5;
    let dx = x1 - x0;
    let dy = y1 - y0;
    crossings.clear();

    push_axis_crossings(crossings, x0, dx);
    push_axis_crossings(crossings, y0, dy);
    crossings.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    crossings.dedup_by(|a, b| (*a - *b).abs() <= GRID_EPS);

    if !segment_point_clear(rows, cols, x0, y0, &mut cell_clear)
        || !segment_point_clear(rows, cols, x1, y1, &mut cell_clear)
    {
        return false;
    }
    for &t in crossings.iter() {
        if !segment_point_clear(rows, cols, x0 + dx * t, y0 + dy * t, &mut cell_clear) {
            return false;
        }
    }

    let mut previous = 0.0;
    for &next in crossings.iter().chain(std::iter::once(&1.0)) {
        if next - previous > GRID_EPS {
            let midpoint = 0.5 * (previous + next);
            if !segment_cell_clear(
                rows,
                cols,
                x0 + dx * midpoint,
                y0 + dy * midpoint,
                &mut cell_clear,
            ) {
                return false;
            }
        }
        previous = next;
    }
    true
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

fn segment_point_clear<F>(rows: usize, cols: usize, x: f64, y: f64, cell_clear: &mut F) -> bool
where
    F: FnMut(usize, usize) -> bool,
{
    let rows = rows as isize;
    let cols = cols as isize;
    let col_a = x.floor() as isize;
    let col_b = (x - GRID_EPS).floor() as isize;
    let row_a = y.floor() as isize;
    let row_b = (y - GRID_EPS).floor() as isize;
    let mut touched_any_cell = false;

    for row in [row_a, row_b] {
        for col in [col_a, col_b] {
            if row < 0 || col < 0 || row >= rows || col >= cols {
                continue;
            }
            touched_any_cell = true;
            if !cell_clear(row as usize, col as usize) {
                return false;
            }
        }
    }
    touched_any_cell
}

fn segment_cell_clear<F>(rows: usize, cols: usize, x: f64, y: f64, cell_clear: &mut F) -> bool
where
    F: FnMut(usize, usize) -> bool,
{
    let row = y.floor() as isize;
    let col = x.floor() as isize;
    if row < 0 || col < 0 || row >= rows as isize || col >= cols as isize {
        return false;
    }
    cell_clear(row as usize, col as usize)
}
