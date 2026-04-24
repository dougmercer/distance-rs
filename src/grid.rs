pub(crate) const MIN_COST: f64 = 1.0e-12;
pub(crate) const EPS: f64 = 1.0e-12;
pub(crate) const GRID_EPS: f64 = 1.0e-9;

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
    pub(crate) search_radius_sq: f64,
    pub(crate) stencil_offsets: Vec<StencilOffset>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StencilOffset {
    pub(crate) dr: isize,
    pub(crate) dc: isize,
    pub(crate) distance: f64,
}

impl Grid {
    pub(crate) fn new(
        rows: usize,
        cols: usize,
        cell_size_x: f64,
        cell_size_y: f64,
        search_radius: f64,
    ) -> Self {
        let radius_rows = (search_radius / cell_size_y).ceil().max(1.0) as isize;
        let radius_cols = (search_radius / cell_size_x).ceil().max(1.0) as isize;
        let search_radius_sq = search_radius * search_radius;
        let mut stencil_offsets = Vec::new();
        for dr in -radius_rows..=radius_rows {
            for dc in -radius_cols..=radius_cols {
                let dx = dc as f64 * cell_size_x;
                let dy = dr as f64 * cell_size_y;
                let distance_sq = dx * dx + dy * dy;
                if distance_sq <= search_radius_sq + EPS {
                    stencil_offsets.push(StencilOffset {
                        dr,
                        dc,
                        distance: distance_sq.sqrt(),
                    });
                }
            }
        }

        Self {
            rows,
            cols,
            cell_size_x,
            cell_size_y,
            search_radius_sq,
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
        ((col1 - col0) * self.cell_size_x).hypot((row1 - row0) * self.cell_size_y)
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
        dx * dx + dy * dy <= self.search_radius_sq + EPS
    }
}
