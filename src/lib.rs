mod barriers;
mod grid;
mod grid_segment;
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
    use crate::solver::{Solver, SolverInput};
    use crate::vertical::{VerticalFactor, VerticalFactorKind};

    fn flat_vf() -> VerticalFactor {
        VerticalFactor::none()
    }

    #[test]
    fn barrier_segment_check_rejects_corner_touching_blocked_cell() {
        let mut valid = vec![true; 9];
        valid[1] = false;
        let solver = Solver::new(SolverInput {
            rows: 3,
            cols: 3,
            cost: vec![1.0; 9],
            elevation: Vec::new(),
            valid,
            has_blocked_cells: true,
            has_elevation: false,
            vf: flat_vf(),
            cell_size_x: 1.0,
            cell_size_y: 1.0,
        });

        assert!(!solver.segment_clear_coord_to_index(0.0, 0.0, solver.idx(2, 2)));
    }

    #[test]
    fn diagonal_barriers_are_thickened_into_blocked_edges() {
        let valid = vec![false, true, true, false];
        let solver = Solver::new(SolverInput {
            rows: 2,
            cols: 2,
            cost: vec![1.0; 4],
            elevation: Vec::new(),
            valid,
            has_blocked_cells: true,
            has_elevation: false,
            vf: flat_vf(),
            cell_size_x: 1.0,
            cell_size_y: 1.0,
        });

        assert!(!solver.is_valid(solver.idx(0, 1)));
        assert!(!solver.is_valid(solver.idx(1, 0)));
    }

    #[test]
    fn local_stencil_does_not_jump_over_high_cost_cells() {
        let rows = 5;
        let cols = 7;
        let mut cost = vec![1.0; rows * cols];
        for row in 0..rows {
            cost[row * cols + 3] = 1000.0;
        }
        let solver = Solver::new(SolverInput {
            rows,
            cols,
            cost,
            elevation: Vec::new(),
            valid: vec![true; rows * cols],
            has_blocked_cells: false,
            has_elevation: false,
            vf: flat_vf(),
            cell_size_x: 1.0,
            cell_size_y: 1.0,
        });

        let output = solver.solve(&[2 * cols + 1], None, None, 1).unwrap();

        assert!(output.distance[2 * cols + 5] > 500.0);
    }

    #[test]
    fn wide_binary_vertical_factor_scales_elevated_surface_distance() {
        let rows = 9;
        let cols = 9;
        let cost = vec![1.0; rows * cols];
        let valid = vec![true; rows * cols];
        let elevation: Vec<f64> = (0..rows)
            .flat_map(|row| {
                (0..cols).map(move |col| {
                    ((row as f64) * 0.7).sin() * 12.0 + ((col as f64) * 0.5).cos() * 8.0
                })
            })
            .collect();

        let base = Solver::new(SolverInput {
            rows,
            cols,
            cost: cost.clone(),
            elevation: elevation.clone(),
            valid: valid.clone(),
            has_blocked_cells: false,
            has_elevation: true,
            vf: flat_vf(),
            cell_size_x: 1.0,
            cell_size_y: 1.0,
        })
        .solve(&[4 * cols + 4], None, None, 1)
        .unwrap();

        let binary = Solver::new(SolverInput {
            rows,
            cols,
            cost,
            elevation,
            valid,
            has_blocked_cells: false,
            has_elevation: true,
            vf: VerticalFactor::from_degrees(
                VerticalFactorKind::Binary,
                1.4,
                -90.0,
                90.0,
                0.0,
                1.0,
                1.0,
                1.0,
            )
            .unwrap(),
            cell_size_x: 1.0,
            cell_size_y: 1.0,
        })
        .solve(&[4 * cols + 4], None, None, 1)
        .unwrap();

        for (scaled, base) in binary.distance.iter().zip(base.distance.iter()) {
            approx::assert_abs_diff_eq!(*scaled, base * 1.4, epsilon = 1.0e-8);
        }
    }
}
