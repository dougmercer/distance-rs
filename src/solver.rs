use std::cell::RefCell;
use std::cmp::Ordering;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::barriers::BarrierMask;
use crate::grid::{Grid, StencilOffset};
use crate::vertical::{VerticalFactor, VerticalFactorKind};

pub(crate) const FAR: u8 = 0;
pub(crate) const TRIAL: u8 = 1;
pub(crate) const ACCEPTED: u8 = 2;
const DISTANCE_EPS_ABS: f64 = 1.0e-12;
const DISTANCE_EPS_REL: f64 = 1.0e-12;
const HEAP_NO_POS: usize = usize::MAX;

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

#[derive(Clone, Debug)]
pub(crate) struct IndexedMinHeap {
    heap: Vec<usize>,
    positions: Vec<usize>,
    values: Vec<f64>,
}

impl IndexedMinHeap {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            heap: Vec::new(),
            positions: vec![HEAP_NO_POS; len],
            values: vec![f64::INFINITY; len],
        }
    }

    pub(crate) fn push_or_decrease(&mut self, idx: usize, value: f64) {
        debug_assert!(value.is_finite());
        let position = self.positions[idx];
        if position == HEAP_NO_POS {
            self.values[idx] = value;
            self.positions[idx] = self.heap.len();
            self.heap.push(idx);
            self.sift_up(self.heap.len() - 1);
        } else if value < self.values[idx] {
            self.values[idx] = value;
            self.sift_up(position);
        }
    }

    pub(crate) fn pop(&mut self) -> Option<HeapEntry> {
        if self.heap.is_empty() {
            return None;
        }

        let min_idx = self.heap[0];
        let min_value = self.values[min_idx];
        let last = self.heap.pop().expect("heap is not empty");
        self.positions[min_idx] = HEAP_NO_POS;
        self.values[min_idx] = f64::INFINITY;

        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.positions[last] = 0;
            self.sift_down(0);
        }

        Some(HeapEntry {
            value: min_value,
            idx: min_idx,
        })
    }

    fn sift_up(&mut self, mut position: usize) {
        while position > 0 {
            let parent = (position - 1) / 2;
            if !self.less(position, parent) {
                break;
            }
            self.swap(position, parent);
            position = parent;
        }
    }

    fn sift_down(&mut self, mut position: usize) {
        loop {
            let left = 2 * position + 1;
            let right = left + 1;
            let mut smallest = position;

            if left < self.heap.len() && self.less(left, smallest) {
                smallest = left;
            }
            if right < self.heap.len() && self.less(right, smallest) {
                smallest = right;
            }
            if smallest == position {
                break;
            }
            self.swap(position, smallest);
            position = smallest;
        }
    }

    fn less(&self, a_position: usize, b_position: usize) -> bool {
        let a = self.heap[a_position];
        let b = self.heap[b_position];
        let a_value = self.values[a];
        let b_value = self.values[b];
        a_value < b_value || (a_value == b_value && a < b)
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        self.positions[self.heap[a]] = a;
        self.positions[self.heap[b]] = b;
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

struct ProgressReporter<'py, 'a> {
    callback: Option<&'a Bound<'py, PyAny>>,
    total: usize,
    every: usize,
    accepted: usize,
    next_report: usize,
}

impl<'py, 'a> ProgressReporter<'py, 'a> {
    fn new(callback: Option<&'a Bound<'py, PyAny>>, total: usize, every: usize) -> Self {
        Self {
            callback,
            total,
            every,
            accepted: 0,
            next_report: every,
        }
    }

    fn set_accepted(&mut self, accepted: usize) -> PyResult<()> {
        self.accepted = accepted;
        self.maybe_report(false)
    }

    fn increment(&mut self) -> PyResult<()> {
        self.accepted += 1;
        self.maybe_report(false)
    }

    fn finish(&mut self) -> PyResult<()> {
        self.maybe_report(true)
    }

    fn maybe_report(&mut self, force: bool) -> PyResult<()> {
        let Some(callback) = self.callback else {
            return Ok(());
        };
        if force || self.accepted >= self.next_report {
            callback.call1((self.accepted, self.total))?;
            while self.accepted >= self.next_report {
                self.next_report += self.every;
            }
        }
        Ok(())
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
    pub(crate) heap: IndexedMinHeap,
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
            heap: IndexedMinHeap::new(n),
            segment_clear_crossings: RefCell::new(Vec::new()),
        }
    }

    pub(crate) fn solve(
        mut self,
        source_indices: &[usize],
        target_indices: Option<&[usize]>,
        progress_callback: Option<&Bound<'_, PyAny>>,
        progress_interval: usize,
    ) -> PyResult<SolveOutput> {
        if source_indices.is_empty() {
            return Err(PyValueError::new_err(
                "at least one source cell is required",
            ));
        }

        let total_valid = if progress_callback.is_some() {
            self.valid_count()
        } else {
            0
        };
        let mut progress =
            ProgressReporter::new(progress_callback, total_valid, progress_interval.max(1));

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
        progress.set_accepted(self.accepted_count())?;
        if target_indices.is_some() && remaining_targets == 0 {
            progress.finish()?;
            return Ok(self.into_output());
        }

        for idx in accepted_sources {
            self.update_around_full(idx);
        }

        while let Some(entry) = self.heap.pop() {
            if self.state[entry.idx] == ACCEPTED {
                continue;
            }

            self.state[entry.idx] = ACCEPTED;
            progress.increment()?;
            if !target_mask.is_empty() && target_mask[entry.idx] {
                target_mask[entry.idx] = false;
                remaining_targets -= 1;
                if remaining_targets == 0 {
                    break;
                }
            }
            self.update_around_incremental(entry.idx);
        }

        progress.finish()?;
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

    fn valid_count(&self) -> usize {
        (0..self.distance.len())
            .filter(|&idx| self.is_valid(idx))
            .count()
    }

    fn accepted_count(&self) -> usize {
        self.state
            .iter()
            .filter(|&&state| state == ACCEPTED)
            .count()
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

pub(crate) fn distance_tolerance(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if scale.is_finite() {
        DISTANCE_EPS_ABS.max(scale * DISTANCE_EPS_REL)
    } else {
        DISTANCE_EPS_ABS
    }
}

#[cfg(test)]
pub(crate) fn value_is_stale(value: f64, best: f64) -> bool {
    value > best + distance_tolerance(value, best)
}

pub(crate) fn value_improves(value: f64, best: f64) -> bool {
    value < best - distance_tolerance(value, best)
}

#[cfg(test)]
mod tests {
    use super::{distance_tolerance, value_improves, value_is_stale, IndexedMinHeap};

    #[test]
    fn indexed_heap_orders_by_value_then_index() {
        let mut heap = IndexedMinHeap::new(3);
        heap.push_or_decrease(0, 2.0);
        heap.push_or_decrease(1, 1.0);
        heap.push_or_decrease(2, 1.0);

        assert_eq!(heap.pop().unwrap().idx, 1);
        assert_eq!(heap.pop().unwrap().idx, 2);
        assert_eq!(heap.pop().unwrap().idx, 0);
    }

    #[test]
    fn indexed_heap_decreases_existing_entry() {
        let mut heap = IndexedMinHeap::new(3);
        heap.push_or_decrease(0, 5.0);
        heap.push_or_decrease(1, 4.0);
        heap.push_or_decrease(0, 3.0);

        let first = heap.pop().unwrap();
        assert_eq!(first.idx, 0);
        assert_eq!(first.value, 3.0);
        assert_eq!(heap.pop().unwrap().idx, 1);
        assert!(heap.pop().is_none());
    }

    #[test]
    fn distance_tolerance_scales_with_accumulated_cost() {
        assert_eq!(distance_tolerance(0.0, 0.0), 1.0e-12);
        approx::assert_abs_diff_eq!(distance_tolerance(1.0e8, 1.0e8), 1.0e-4);
        assert!(value_is_stale(1.0e8 + 2.0e-4, 1.0e8));
        assert!(!value_is_stale(1.0e8 + 0.5e-4, 1.0e8));
        assert!(value_improves(1.0e8 - 2.0e-4, 1.0e8));
        assert!(!value_improves(1.0e8 - 0.5e-4, 1.0e8));
    }
}
