use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::barriers::BarrierMask;
use crate::grid::{Grid, StencilOffset};
use crate::vertical::{VerticalFactor, VerticalFactorKind};

pub(crate) const FAR: u8 = 0;
pub(crate) const TRIAL: u8 = 1;
pub(crate) const ACCEPTED: u8 = 2;

#[derive(Clone, Copy, Debug)]
pub(crate) struct HeapEntry {
    pub(crate) value: f64,
    pub(crate) idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.value.total_cmp(&other.value) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .value
            .total_cmp(&self.value)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Parent {
    pub(crate) a: i64,
    pub(crate) b: i64,
    pub(crate) weight: f64,
}

impl Parent {
    pub(crate) fn none() -> Self {
        Self {
            a: -1,
            b: -1,
            weight: f64::NAN,
        }
    }

    pub(crate) fn point(a: usize) -> Self {
        Self {
            a: a as i64,
            b: -1,
            weight: 1.0,
        }
    }

    pub(crate) fn segment(a: usize, b: usize, weight: f64) -> Self {
        Self {
            a: a as i64,
            b: b as i64,
            weight,
        }
    }
}

pub(crate) struct Solver {
    pub(crate) grid: Grid,
    pub(crate) cost: Vec<f64>,
    pub(crate) elevation: Vec<f64>,
    pub(crate) barriers: BarrierMask,
    pub(crate) has_elevation: bool,
    pub(crate) flat_cost_mode: bool,
    pub(crate) vf: VerticalFactor,
    pub(crate) distance: Vec<f64>,
    pub(crate) parent: Vec<Parent>,
    pub(crate) back_direction: Vec<f64>,
    pub(crate) state: Vec<u8>,
    pub(crate) heap: BinaryHeap<HeapEntry>,
    segment_clear_crossings: RefCell<Vec<f64>>,
}

pub(crate) struct SolverInput {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) cost: Vec<f64>,
    pub(crate) elevation: Vec<f64>,
    pub(crate) valid: Vec<bool>,
    pub(crate) has_blocked_cells: bool,
    pub(crate) has_elevation: bool,
    pub(crate) vf: VerticalFactor,
    pub(crate) cell_size_x: f64,
    pub(crate) cell_size_y: f64,
}

pub(crate) struct SolveOutput {
    pub(crate) distance: Vec<f64>,
    pub(crate) parent: Vec<Parent>,
    pub(crate) back_direction: Vec<f64>,
}

impl Solver {
    pub(crate) fn new(input: SolverInput) -> Self {
        let grid = Grid::new(input.rows, input.cols, input.cell_size_x, input.cell_size_y);
        let n = input.rows * input.cols;
        let barriers =
            BarrierMask::new(input.rows, input.cols, input.valid, input.has_blocked_cells);
        Self {
            grid,
            cost: input.cost,
            elevation: input.elevation,
            barriers,
            has_elevation: input.has_elevation,
            flat_cost_mode: !input.has_elevation && input.vf.kind == VerticalFactorKind::None,
            vf: input.vf,
            distance: vec![f64::INFINITY; n],
            parent: vec![Parent::none(); n],
            back_direction: vec![f64::NAN; n],
            state: vec![FAR; n],
            heap: BinaryHeap::new(),
            segment_clear_crossings: RefCell::new(Vec::new()),
        }
    }

    pub(crate) fn solve(
        mut self,
        source_indices: &[usize],
        target_indices: Option<&[usize]>,
    ) -> PyResult<SolveOutput> {
        if source_indices.is_empty() {
            return Err(PyValueError::new_err(
                "at least one source cell is required",
            ));
        }

        for &idx in source_indices {
            if self.is_valid(idx) {
                self.distance[idx] = 0.0;
                self.state[idx] = ACCEPTED;
            }
        }

        let accepted_sources: Vec<usize> = source_indices
            .iter()
            .copied()
            .filter(|&idx| self.is_valid(idx))
            .collect();
        if accepted_sources.is_empty() {
            return Err(PyValueError::new_err(
                "all source cells are blocked or outside valid data",
            ));
        }

        let (mut target_mask, mut remaining_targets) = self.target_mask(target_indices);
        for &idx in &accepted_sources {
            if !target_mask.is_empty() && target_mask[idx] {
                target_mask[idx] = false;
                remaining_targets -= 1;
            }
        }
        if target_indices.is_some() && remaining_targets == 0 {
            return Ok(self.into_output());
        }

        for idx in accepted_sources {
            self.update_around_full(idx);
        }

        while let Some(entry) = self.heap.pop() {
            if self.state[entry.idx] == ACCEPTED {
                continue;
            }
            if entry.value > self.distance[entry.idx] + 1.0e-10 {
                continue;
            }

            self.state[entry.idx] = ACCEPTED;
            if !target_mask.is_empty() && target_mask[entry.idx] {
                target_mask[entry.idx] = false;
                remaining_targets -= 1;
                if remaining_targets == 0 {
                    break;
                }
            }
            self.update_around_incremental(entry.idx);
        }

        Ok(self.into_output())
    }

    fn target_mask(&self, target_indices: Option<&[usize]>) -> (Vec<bool>, usize) {
        let Some(target_indices) = target_indices else {
            return (Vec::new(), 0);
        };

        let mut mask = vec![false; self.distance.len()];
        let mut count = 0usize;
        for &idx in target_indices {
            if idx >= mask.len() || !self.is_valid(idx) || mask[idx] {
                continue;
            }
            mask[idx] = true;
            count += 1;
        }
        (mask, count)
    }

    fn into_output(self) -> SolveOutput {
        SolveOutput {
            distance: self.distance,
            parent: self.parent,
            back_direction: self.back_direction,
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.grid.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.grid.cols
    }

    pub(crate) fn idx(&self, row: usize, col: usize) -> usize {
        self.grid.idx(row, col)
    }

    pub(crate) fn row_col(&self, idx: usize) -> (usize, usize) {
        self.grid.row_col(idx)
    }

    pub(crate) fn is_valid(&self, idx: usize) -> bool {
        self.barriers.is_valid(idx)
    }

    pub(crate) fn is_accepted(&self, idx: usize) -> bool {
        self.state[idx] == ACCEPTED
    }

    pub(crate) fn offset_idx(
        &self,
        row: usize,
        col: usize,
        offset: StencilOffset,
    ) -> Option<usize> {
        self.grid.offset_idx(row, col, offset)
    }

    pub(crate) fn offset_within_radius(&self, dr: isize, dc: isize) -> bool {
        self.grid.offset_within_radius(dr, dc)
    }

    pub(crate) fn physical_distance_coords(
        &self,
        row0: f64,
        col0: f64,
        row1: f64,
        col1: f64,
    ) -> f64 {
        self.grid.physical_distance_coords(row0, col0, row1, col1)
    }

    pub(crate) fn segment_clear_index_to_index(&self, a: usize, b: usize) -> bool {
        let mut crossings = self.segment_clear_crossings.borrow_mut();
        self.barriers
            .segment_clear_index_to_index_with_crossings(&self.grid, a, b, &mut crossings)
    }

    pub(crate) fn segment_clear_coord_to_index(&self, row0: f64, col0: f64, idx: usize) -> bool {
        let mut crossings = self.segment_clear_crossings.borrow_mut();
        self.barriers.segment_clear_coord_to_index_with_crossings(
            &self.grid,
            row0,
            col0,
            idx,
            &mut crossings,
        )
    }
}
