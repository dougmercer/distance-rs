mod barriers;
mod grid;
mod path;
mod python;
mod solver;
mod updates;
mod vertical;

use pyo3::prelude::*;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}

#[cfg(test)]
mod tests {
    use crate::solver::{Solver, SolverInput, SolverOptions};
    use crate::vertical::{VerticalFactor, VerticalFactorKind};

    fn flat_vf() -> VerticalFactor {
        VerticalFactor {
            kind: VerticalFactorKind::None,
            zero_factor: 1.0,
            low_cut_angle: -90.0,
            high_cut_angle: 90.0,
            slope: 0.0,
            power: 1.0,
            cos_power: 1.0,
            sec_power: 1.0,
        }
    }

    #[test]
    fn barrier_segment_check_rejects_corner_touching_blocked_cell() {
        let mut valid = vec![true; 9];
        valid[1] = false;
        let solver = Solver::new(
            SolverInput {
                rows: 3,
                cols: 3,
                cost: vec![1.0; 9],
                elevation: Vec::new(),
                valid,
                has_blocked_cells: true,
            },
            SolverOptions {
                has_elevation: false,
                use_surface_distance: true,
                vf: flat_vf(),
                cell_size_x: 1.0,
                cell_size_y: 1.0,
                search_radius: 4.0,
            },
        );

        assert!(!solver.segment_clear_coord_to_index(0.0, 0.0, solver.idx(2, 2)));
    }
}
