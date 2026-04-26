pub(crate) const MIN_COST: f64 = 1.0e-12;
pub(crate) const EPS: f64 = 1.0e-12;
pub(crate) const GRID_EPS: f64 = 1.0e-9;

use crate::math::fast_hypot;

pub(crate) const NEIGHBORS_8: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

#[derive(Clone, Debug)]
pub(crate) struct Grid {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) cell_size_x: f64,
    pub(crate) cell_size_y: f64,
    pub(crate) stencil_offsets: Vec<StencilOffset>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StencilOffset {
    pub(crate) dr: isize,
    pub(crate) dc: isize,
    pub(crate) distance: f64,
    pub(crate) back_direction: Option<f64>,
}

impl Grid {
    pub(crate) fn new(rows: usize, cols: usize, cell_size_x: f64, cell_size_y: f64) -> Self {
        let mut stencil_offsets = Vec::new();
        for dr in -1..=1 {
            for dc in -1..=1 {
                let dx = dc as f64 * cell_size_x;
                let dy = dr as f64 * cell_size_y;
                stencil_offsets.push(StencilOffset {
                    dr,
                    dc,
                    distance: fast_hypot(dx, dy),
                    back_direction: direction_from_delta(dx, dy),
                });
            }
        }

        Self {
            rows,
            cols,
            cell_size_x,
            cell_size_y,
            stencil_offsets,
        }
    }

    pub(crate) fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    pub(crate) fn row_col(&self, idx: usize) -> (usize, usize) {
        (idx / self.cols, idx % self.cols)
    }

    pub(crate) fn physical_distance_coords(
        &self,
        row0: f64,
        col0: f64,
        row1: f64,
        col1: f64,
    ) -> f64 {
        fast_hypot(
            (col1 - col0) * self.cell_size_x,
            (row1 - row0) * self.cell_size_y,
        )
    }

    pub(crate) fn offset_idx(
        &self,
        row: usize,
        col: usize,
        offset: StencilOffset,
    ) -> Option<usize> {
        let offset_row = row as isize + offset.dr;
        let offset_col = col as isize + offset.dc;
        if offset_row < 0
            || offset_col < 0
            || offset_row >= self.rows as isize
            || offset_col >= self.cols as isize
        {
            return None;
        }
        Some(self.idx(offset_row as usize, offset_col as usize))
    }

    pub(crate) fn offset_within_radius(&self, dr: isize, dc: isize) -> bool {
        let dx = dc as f64 * self.cell_size_x;
        let dy = dr as f64 * self.cell_size_y;
        let radius_sq = self.cell_size_x * self.cell_size_x + self.cell_size_y * self.cell_size_y;
        dx * dx + dy * dy <= radius_sq + EPS
    }

    pub(crate) fn back_direction_for_offset(&self, dr: isize, dc: isize) -> Option<f64> {
        if !(-1..=1).contains(&dr) || !(-1..=1).contains(&dc) {
            return None;
        }
        let offset_index = ((dr + 1) * 3 + (dc + 1)) as usize;
        self.stencil_offsets[offset_index].back_direction
    }
}

pub(crate) fn direction_from_delta(dx: f64, dy: f64) -> Option<f64> {
    if !dx.is_finite() || !dy.is_finite() || dx * dx + dy * dy <= EPS * EPS {
        return None;
    }
    let mut degrees = dx.atan2(-dy).to_degrees();
    if degrees < 0.0 {
        degrees += 360.0;
    }
    Some(degrees)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::{direction_from_delta, Grid};

    #[test]
    fn square_grid_caches_eight_neighbor_back_directions() {
        let grid = Grid::new(3, 3, 1.0, 1.0);
        let expected = [
            ((-1, -1), 315.0),
            ((-1, 0), 0.0),
            ((-1, 1), 45.0),
            ((0, -1), 270.0),
            ((0, 1), 90.0),
            ((1, -1), 225.0),
            ((1, 0), 180.0),
            ((1, 1), 135.0),
        ];

        for ((dr, dc), degrees) in expected {
            assert_abs_diff_eq!(
                grid.back_direction_for_offset(dr, dc).unwrap(),
                degrees,
                epsilon = 1.0e-12
            );
        }
        assert!(grid.back_direction_for_offset(0, 0).is_none());
    }

    #[test]
    fn rectangular_grid_cached_directions_match_delta_formula() {
        let grid = Grid::new(3, 3, 2.0, 0.5);
        for dr in -1..=1 {
            for dc in -1..=1 {
                let dx = dc as f64 * grid.cell_size_x;
                let dy = dr as f64 * grid.cell_size_y;
                match (
                    grid.back_direction_for_offset(dr, dc),
                    direction_from_delta(dx, dy),
                ) {
                    (Some(actual), Some(expected)) => {
                        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-12);
                    }
                    (None, None) => {}
                    (actual, expected) => panic!("direction mismatch: {actual:?} != {expected:?}"),
                }
            }
        }
    }
}
