use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

const FAR: u8 = 0;
const TRIAL: u8 = 1;
const ACCEPTED: u8 = 2;
const MIN_COST: f64 = 1.0e-12;
const EPS: f64 = 1.0e-12;
const GRID_EPS: f64 = 1.0e-9;
const PARALLEL_UPDATE_MIN_STENCIL: usize = 16;
const PARALLEL_BLOCKED_FLAT_UPDATE_MIN_STENCIL: usize = 128;
const PARALLEL_FLAT_UPDATE_MIN_STENCIL: usize = 256;

const NEIGHBORS_8: [(isize, isize); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

#[derive(Clone, Copy, Debug)]
struct HeapEntry {
    value: f64,
    idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.value.to_bits() == other.value.to_bits()
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
            .partial_cmp(&self.value)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.idx.cmp(&self.idx))
    }
}

#[derive(Clone, Copy, Debug)]
struct Parent {
    a: i64,
    b: i64,
    weight: f64,
}

impl Parent {
    fn none() -> Self {
        Self {
            a: -1,
            b: -1,
            weight: f64::NAN,
        }
    }

    fn point(a: usize) -> Self {
        Self {
            a: a as i64,
            b: -1,
            weight: 1.0,
        }
    }

    fn segment(a: usize, b: usize, weight: f64) -> Self {
        Self {
            a: a as i64,
            b: b as i64,
            weight,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VerticalFactorKind {
    None,
    Binary,
    Linear,
    InverseLinear,
    SymmetricLinear,
    SymmetricInverseLinear,
    Cos,
    Sec,
    CosSec,
    SecCos,
    HikingTime,
    BidirHikingTime,
}

impl VerticalFactorKind {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "none" => Ok(Self::None),
            "binary" => Ok(Self::Binary),
            "linear" => Ok(Self::Linear),
            "inverse_linear" => Ok(Self::InverseLinear),
            "symmetric_linear" => Ok(Self::SymmetricLinear),
            "symmetric_inverse_linear" => Ok(Self::SymmetricInverseLinear),
            "cos" => Ok(Self::Cos),
            "sec" => Ok(Self::Sec),
            "cos_sec" => Ok(Self::CosSec),
            "sec_cos" => Ok(Self::SecCos),
            "hiking_time" => Ok(Self::HikingTime),
            "bidir_hiking_time" => Ok(Self::BidirHikingTime),
            _ => Err(PyValueError::new_err(format!(
                "unknown vertical factor kind: {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct VerticalFactor {
    kind: VerticalFactorKind,
    zero_factor: f64,
    low_cut_angle: f64,
    high_cut_angle: f64,
    slope: f64,
    power: f64,
    cos_power: f64,
    sec_power: f64,
}

impl VerticalFactor {
    fn factor(&self, angle_degrees: f64) -> f64 {
        if self.kind == VerticalFactorKind::None {
            return 1.0;
        }
        if !angle_degrees.is_finite()
            || angle_degrees <= self.low_cut_angle
            || angle_degrees >= self.high_cut_angle
        {
            return f64::INFINITY;
        }

        let factor = match self.kind {
            VerticalFactorKind::None => 1.0,
            VerticalFactorKind::Binary => self.zero_factor,
            VerticalFactorKind::Linear => self.zero_factor + self.slope * angle_degrees,
            VerticalFactorKind::InverseLinear => self.zero_factor + self.slope * angle_degrees,
            VerticalFactorKind::SymmetricLinear => {
                self.zero_factor + self.slope * angle_degrees.abs()
            }
            VerticalFactorKind::SymmetricInverseLinear => {
                self.zero_factor + self.slope * angle_degrees.abs()
            }
            VerticalFactorKind::Cos => cos_factor(angle_degrees, self.power),
            VerticalFactorKind::Sec => sec_factor(angle_degrees, self.power),
            VerticalFactorKind::CosSec => {
                if angle_degrees < 0.0 {
                    cos_factor(angle_degrees, self.cos_power)
                } else {
                    sec_factor(angle_degrees, self.sec_power)
                }
            }
            VerticalFactorKind::SecCos => {
                if angle_degrees < 0.0 {
                    sec_factor(angle_degrees, self.sec_power)
                } else {
                    cos_factor(angle_degrees, self.cos_power)
                }
            }
            VerticalFactorKind::HikingTime => hiking_pace(angle_degrees),
            VerticalFactorKind::BidirHikingTime => {
                0.5 * (hiking_pace(angle_degrees) + hiking_pace(-angle_degrees))
            }
        };

        if factor.is_finite() && factor > 0.0 {
            factor
        } else {
            f64::INFINITY
        }
    }
}

fn cos_factor(angle_degrees: f64, power: f64) -> f64 {
    angle_degrees.to_radians().cos().powf(power)
}

fn sec_factor(angle_degrees: f64, power: f64) -> f64 {
    1.0 / angle_degrees.to_radians().cos().powf(power)
}

fn hiking_pace(angle_degrees: f64) -> f64 {
    let slope = angle_degrees.to_radians().tan();
    let speed_km_per_hour = 6.0 * (-3.5 * (slope + 0.05).abs()).exp();
    1.0 / (speed_km_per_hour * 1000.0)
}

struct Solver {
    rows: usize,
    cols: usize,
    cost: Vec<f64>,
    elevation: Vec<f64>,
    valid: Vec<bool>,
    blocked_prefix: Vec<usize>,
    has_blocked_cells: bool,
    has_elevation: bool,
    flat_cost_mode: bool,
    use_surface_distance: bool,
    vf: VerticalFactor,
    cell_size_x: f64,
    cell_size_y: f64,
    search_radius_sq: f64,
    stencil_offsets: Vec<StencilOffset>,
    distance: Vec<f64>,
    parent: Vec<Parent>,
    state: Vec<u8>,
    heap: BinaryHeap<HeapEntry>,
}

#[derive(Clone, Copy, Debug)]
struct StencilOffset {
    dr: isize,
    dc: isize,
    distance: f64,
}

#[derive(Clone, Copy, Debug)]
struct CandidateUpdate {
    idx: usize,
    value: f64,
    parent: Parent,
}

#[derive(Clone, Copy, Debug)]
struct BestCandidate {
    value: f64,
    parent: Parent,
}

impl BestCandidate {
    fn new() -> Self {
        Self {
            value: f64::INFINITY,
            parent: Parent::none(),
        }
    }

    fn consider(&mut self, value: f64, parent: Parent) {
        if value < self.value {
            self.value = value;
            self.parent = parent;
        }
    }
}

struct SegmentContext<'a> {
    solver: &'a Solver,
    idx: usize,
    a_row: f64,
    a_col: f64,
    b_row: f64,
    b_col: f64,
    p_row: f64,
    p_col: f64,
    cost_idx: f64,
    cost_a: f64,
    cost_b: f64,
    distance_a: f64,
    distance_b: f64,
    elevation_idx: f64,
    elevation_a: f64,
    elevation_b: f64,
}

impl<'a> SegmentContext<'a> {
    fn new(solver: &'a Solver, idx: usize, a: usize, b: usize) -> Self {
        let (a_row, a_col) = solver.row_col(a);
        let (b_row, b_col) = solver.row_col(b);
        let (p_row, p_col) = solver.row_col(idx);
        let (elevation_idx, elevation_a, elevation_b) = if solver.has_elevation {
            (
                solver.elevation[idx],
                solver.elevation[a],
                solver.elevation[b],
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        Self {
            solver,
            idx,
            a_row: a_row as f64,
            a_col: a_col as f64,
            b_row: b_row as f64,
            b_col: b_col as f64,
            p_row: p_row as f64,
            p_col: p_col as f64,
            cost_idx: solver.cost[idx],
            cost_a: solver.cost[a],
            cost_b: solver.cost[b],
            distance_a: solver.distance[a],
            distance_b: solver.distance[b],
            elevation_idx,
            elevation_a,
            elevation_b,
        }
    }

    fn objective(&self, weight_a: f64) -> f64 {
        let weight_b = 1.0 - weight_a;
        let y_row = weight_a * self.a_row + weight_b * self.b_row;
        let y_col = weight_a * self.a_col + weight_b * self.b_col;
        if self.solver.has_blocked_cells
            && !self
                .solver
                .segment_clear_coord_to_index(y_row, y_col, self.idx)
        {
            return f64::INFINITY;
        }

        let plan_distance = self
            .solver
            .physical_distance_coords(y_row, y_col, self.p_row, self.p_col);
        if plan_distance <= EPS {
            return f64::INFINITY;
        }

        let y_cost = weight_a * self.cost_a + weight_b * self.cost_b;
        let local_cost = 0.5 * (self.cost_idx + y_cost);
        let front_value = weight_a * self.distance_a + weight_b * self.distance_b;
        if self.solver.flat_cost_mode {
            return front_value + plan_distance * local_cost;
        }

        let y_elevation = weight_a * self.elevation_a + weight_b * self.elevation_b;
        let dz = if self.solver.has_elevation {
            self.elevation_idx - y_elevation
        } else {
            0.0
        };
        let surface_distance = self.solver.surface_distance(plan_distance, dz);
        let angle = self.solver.vertical_angle(plan_distance, dz);
        let vf = self.solver.vf.factor(angle);
        if !vf.is_finite() {
            return f64::INFINITY;
        }

        front_value + surface_distance * local_cost * vf
    }

    fn golden_section(&self, mut lo: f64, mut hi: f64) -> Option<(f64, f64)> {
        let gr = (5.0_f64.sqrt() - 1.0) * 0.5;
        let mut c = hi - gr * (hi - lo);
        let mut d = lo + gr * (hi - lo);
        let mut fc = self.objective(c);
        let mut fd = self.objective(d);

        for _ in 0..24 {
            if fc < fd {
                hi = d;
                d = c;
                fd = fc;
                c = hi - gr * (hi - lo);
                fc = self.objective(c);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + gr * (hi - lo);
                fd = self.objective(d);
            }
        }

        let weight = 0.5 * (lo + hi);
        let value = self.objective(weight);
        if value.is_finite() {
            Some((value, weight))
        } else {
            None
        }
    }

    fn constant_cost_stationary_weight(&self) -> Option<f64> {
        let ux = (self.a_col - self.b_col) * self.solver.cell_size_x;
        let uy = (self.a_row - self.b_row) * self.solver.cell_size_y;
        let vx = (self.b_col - self.p_col) * self.solver.cell_size_x;
        let vy = (self.b_row - self.p_row) * self.solver.cell_size_y;

        let u_sq = ux * ux + uy * uy;
        if u_sq <= EPS {
            return None;
        }

        let local_cost = 0.5 * (self.cost_idx + self.cost_a);
        if local_cost <= EPS {
            return None;
        }

        let front_slope = self.distance_a - self.distance_b;
        let scaled_slope = -front_slope / local_cost;
        if scaled_slope.abs() >= u_sq.sqrt() {
            return None;
        }

        let uv = ux * vx + uy * vy;
        let v_sq = vx * vx + vy * vy;
        let perpendicular_sq = (v_sq - uv * uv / u_sq).max(0.0);
        if perpendicular_sq <= EPS {
            return None;
        }

        let denominator = 1.0 - scaled_slope * scaled_slope / u_sq;
        if denominator <= EPS {
            return None;
        }

        let stationary_projection = scaled_slope.signum()
            * (scaled_slope * scaled_slope * perpendicular_sq / denominator).sqrt();
        Some((stationary_projection - uv) / u_sq)
    }
}

struct SolverInput {
    rows: usize,
    cols: usize,
    cost: Vec<f64>,
    elevation: Vec<f64>,
    valid: Vec<bool>,
    has_blocked_cells: bool,
}

struct SolverOptions {
    has_elevation: bool,
    use_surface_distance: bool,
    vf: VerticalFactor,
    cell_size_x: f64,
    cell_size_y: f64,
    search_radius: f64,
}

impl Solver {
    fn new(input: SolverInput, options: SolverOptions) -> Self {
        let radius_rows = (options.search_radius / options.cell_size_y)
            .ceil()
            .max(1.0) as isize;
        let radius_cols = (options.search_radius / options.cell_size_x)
            .ceil()
            .max(1.0) as isize;
        let search_radius_sq = options.search_radius * options.search_radius;
        let mut stencil_offsets = Vec::new();
        for dr in -radius_rows..=radius_rows {
            for dc in -radius_cols..=radius_cols {
                let dx = dc as f64 * options.cell_size_x;
                let dy = dr as f64 * options.cell_size_y;
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
        let n = input.rows * input.cols;
        let blocked_prefix = if input.has_blocked_cells {
            Self::build_blocked_prefix(input.rows, input.cols, &input.valid)
        } else {
            Vec::new()
        };
        Self {
            rows: input.rows,
            cols: input.cols,
            cost: input.cost,
            elevation: input.elevation,
            valid: input.valid,
            blocked_prefix,
            has_blocked_cells: input.has_blocked_cells,
            has_elevation: options.has_elevation,
            flat_cost_mode: !options.has_elevation && options.vf.kind == VerticalFactorKind::None,
            use_surface_distance: options.use_surface_distance,
            vf: options.vf,
            cell_size_x: options.cell_size_x,
            cell_size_y: options.cell_size_y,
            search_radius_sq,
            stencil_offsets,
            distance: vec![f64::INFINITY; n],
            parent: vec![Parent::none(); n],
            state: vec![FAR; n],
            heap: BinaryHeap::new(),
        }
    }

    fn solve(mut self, source_indices: &[usize]) -> PyResult<SolveOutput> {
        if source_indices.is_empty() {
            return Err(PyValueError::new_err(
                "at least one source cell is required",
            ));
        }

        for &idx in source_indices {
            if self.valid[idx] {
                self.distance[idx] = 0.0;
                self.state[idx] = ACCEPTED;
            }
        }

        let accepted_sources: Vec<usize> = source_indices
            .iter()
            .copied()
            .filter(|&idx| self.valid[idx])
            .collect();
        if accepted_sources.is_empty() {
            return Err(PyValueError::new_err(
                "all source cells are blocked or outside valid data",
            ));
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
            self.update_around_incremental(entry.idx);
        }

        Ok(SolveOutput {
            rows: self.rows,
            cols: self.cols,
            distance: self.distance,
            parent: self.parent,
            cell_size_x: self.cell_size_x,
            cell_size_y: self.cell_size_y,
        })
    }

    fn update_around_full(&mut self, center: usize) {
        let (center_row, center_col) = self.row_col(center);
        if self.should_parallelize_update() {
            let updates: Vec<CandidateUpdate> = self
                .stencil_offsets
                .par_iter()
                .filter_map(|offset| {
                    let idx = self.offset_idx(center_row, center_col, *offset)?;
                    if !self.valid[idx] || self.state[idx] == ACCEPTED {
                        return None;
                    }
                    self.compute_update(idx)
                })
                .collect();

            for update in updates {
                self.apply_update(update);
            }
        } else {
            for offset_index in 0..self.stencil_offsets.len() {
                let offset = self.stencil_offsets[offset_index];
                let Some(idx) = self.offset_idx(center_row, center_col, offset) else {
                    continue;
                };
                if !self.valid[idx] || self.state[idx] == ACCEPTED {
                    continue;
                }
                if let Some(update) = self.compute_update(idx) {
                    self.apply_update(update);
                }
            }
        }
    }

    fn update_around_incremental(&mut self, center: usize) {
        let (center_row, center_col) = self.row_col(center);
        if self.should_parallelize_update() {
            let updates: Vec<CandidateUpdate> = self
                .stencil_offsets
                .par_iter()
                .filter_map(|offset| {
                    let idx = self.offset_idx(center_row, center_col, *offset)?;
                    if !self.valid[idx] || self.state[idx] == ACCEPTED {
                        return None;
                    }
                    self.compute_incremental_update(idx, center, offset.distance)
                })
                .collect();

            for update in updates {
                self.apply_update(update);
            }
        } else {
            for offset_index in 0..self.stencil_offsets.len() {
                let offset = self.stencil_offsets[offset_index];
                let Some(idx) = self.offset_idx(center_row, center_col, offset) else {
                    continue;
                };
                if !self.valid[idx] || self.state[idx] == ACCEPTED {
                    continue;
                }
                if let Some(update) = self.compute_incremental_update(idx, center, offset.distance)
                {
                    self.apply_update(update);
                }
            }
        }
    }

    fn compute_update(&self, idx: usize) -> Option<CandidateUpdate> {
        let (row, col) = self.row_col(idx);

        let mut best = BestCandidate::new();

        for offset in &self.stencil_offsets {
            if let Some(q) = self.offset_idx(row, col, *offset) {
                let q_row = (row as isize + offset.dr) as usize;
                let q_col = (col as isize + offset.dc) as usize;
                if self.state[q] != ACCEPTED {
                    continue;
                }
                if let Some(value) = self.point_candidate(idx, q, offset.distance) {
                    best.consider(value, Parent::point(q));
                }

                for (dr, dc) in NEIGHBORS_8 {
                    let r_row = q_row as isize + dr;
                    let r_col = q_col as isize + dc;
                    if r_row < 0
                        || r_col < 0
                        || r_row >= self.rows as isize
                        || r_col >= self.cols as isize
                    {
                        continue;
                    }
                    let r = self.idx(r_row as usize, r_col as usize);
                    if r <= q || self.state[r] != ACCEPTED {
                        continue;
                    }
                    if !self.offset_within_radius(offset.dr + dr, offset.dc + dc) {
                        continue;
                    }
                    if let Some((value, weight)) = self.segment_candidate(idx, q, r) {
                        best.consider(value, Parent::segment(q, r, weight));
                    }
                }
            }
        }

        if best.value.is_finite() {
            Some(CandidateUpdate {
                idx,
                value: best.value,
                parent: best.parent,
            })
        } else {
            None
        }
    }

    fn compute_incremental_update(
        &self,
        idx: usize,
        accepted: usize,
        plan_distance: f64,
    ) -> Option<CandidateUpdate> {
        let mut best = BestCandidate::new();

        let (idx_row, idx_col) = self.row_col(idx);
        let (accepted_row, accepted_col) = self.row_col(accepted);
        let center_dr = accepted_row as isize - idx_row as isize;
        let center_dc = accepted_col as isize - idx_col as isize;
        let mut lower_neighbors = [(0isize, 0isize, 0usize); NEIGHBORS_8.len()];
        let mut lower_neighbor_count = 0usize;

        for (dr, dc) in NEIGHBORS_8 {
            let neighbor_row = accepted_row as isize + dr;
            let neighbor_col = accepted_col as isize + dc;
            if neighbor_row < 0
                || neighbor_col < 0
                || neighbor_row >= self.rows as isize
                || neighbor_col >= self.cols as isize
            {
                continue;
            }

            let neighbor = self.idx(neighbor_row as usize, neighbor_col as usize);
            if self.state[neighbor] != ACCEPTED {
                continue;
            }

            let neighbor_dr = neighbor_row - idx_row as isize;
            let neighbor_dc = neighbor_col - idx_col as isize;
            if !self.offset_within_radius(neighbor_dr, neighbor_dc) {
                continue;
            }

            if neighbor < accepted {
                lower_neighbors[lower_neighbor_count] = (neighbor_dr, neighbor_dc, neighbor);
                lower_neighbor_count += 1;
            }
        }
        let lower_neighbors = &mut lower_neighbors[..lower_neighbor_count];
        lower_neighbors.sort_unstable_by_key(|&(dr, dc, _neighbor)| (dr, dc));

        let consider_center = |best: &mut BestCandidate| {
            if let Some(value) = self.point_candidate(idx, accepted, plan_distance) {
                best.consider(value, Parent::point(accepted));
            }

            for (dr, dc) in NEIGHBORS_8 {
                let neighbor_row = accepted_row as isize + dr;
                let neighbor_col = accepted_col as isize + dc;
                if neighbor_row < 0
                    || neighbor_col < 0
                    || neighbor_row >= self.rows as isize
                    || neighbor_col >= self.cols as isize
                {
                    continue;
                }

                let neighbor = self.idx(neighbor_row as usize, neighbor_col as usize);
                if neighbor <= accepted || self.state[neighbor] != ACCEPTED {
                    continue;
                }

                let neighbor_dr = neighbor_row - idx_row as isize;
                let neighbor_dc = neighbor_col - idx_col as isize;
                if !self.offset_within_radius(neighbor_dr, neighbor_dc) {
                    continue;
                }

                if let Some((value, weight)) = self.segment_candidate(idx, accepted, neighbor) {
                    best.consider(value, Parent::segment(accepted, neighbor, weight));
                }
            }
        };

        let mut center_considered = false;
        for &(neighbor_dr, neighbor_dc, neighbor) in lower_neighbors.iter() {
            if !center_considered && (center_dr, center_dc) < (neighbor_dr, neighbor_dc) {
                consider_center(&mut best);
                center_considered = true;
            }

            if let Some((value, weight)) = self.segment_candidate(idx, neighbor, accepted) {
                best.consider(value, Parent::segment(neighbor, accepted, weight));
            }
        }

        if !center_considered {
            consider_center(&mut best);
        }

        if best.value.is_finite() {
            Some(CandidateUpdate {
                idx,
                value: best.value,
                parent: best.parent,
            })
        } else {
            None
        }
    }

    fn point_candidate(&self, idx: usize, q: usize, plan_distance: f64) -> Option<f64> {
        if !self.segment_clear_index_to_index(q, idx) {
            return None;
        }
        if plan_distance <= EPS {
            return None;
        }
        let local_cost = 0.5 * (self.cost[idx] + self.cost[q]);
        if self.flat_cost_mode {
            return Some(self.distance[q] + plan_distance * local_cost);
        }
        let dz = if self.has_elevation {
            self.elevation[idx] - self.elevation[q]
        } else {
            0.0
        };
        let surface_distance = self.surface_distance(plan_distance, dz);
        let angle = self.vertical_angle(plan_distance, dz);
        let vf = self.vf.factor(angle);
        if !vf.is_finite() {
            return None;
        }
        Some(self.distance[q] + surface_distance * local_cost * vf)
    }

    fn segment_candidate(&self, idx: usize, a: usize, b: usize) -> Option<(f64, f64)> {
        let context = SegmentContext::new(self, idx, a, b);
        if self.flat_cost_mode
            && !self.has_blocked_cells
            && (context.cost_a - context.cost_b).abs() <= EPS
        {
            return self.constant_cost_segment_candidate(idx, a, b, &context);
        }

        let samples = 8usize;
        let mut best_value = f64::INFINITY;
        let mut best_weight = 0.0;
        let mut best_sample = 0usize;

        for i in 0..=samples {
            let weight = i as f64 / samples as f64;
            let value = if i == 0 {
                self.endpoint_segment_value(idx, b, context.b_row, context.b_col)
            } else if i == samples {
                self.endpoint_segment_value(idx, a, context.a_row, context.a_col)
            } else {
                context.objective(weight)
            };
            if value < best_value {
                best_value = value;
                best_weight = weight;
                best_sample = i;
            }
        }

        if !best_value.is_finite() {
            return None;
        }

        if best_sample > 0 && best_sample < samples {
            let lo = (best_sample - 1) as f64 / samples as f64;
            let hi = (best_sample + 1) as f64 / samples as f64;
            if let Some((value, weight)) = context.golden_section(lo, hi) {
                if value < best_value {
                    best_value = value;
                    best_weight = weight;
                }
            }
        }

        Some((best_value, best_weight))
    }

    fn constant_cost_segment_candidate(
        &self,
        idx: usize,
        a: usize,
        b: usize,
        context: &SegmentContext<'_>,
    ) -> Option<(f64, f64)> {
        let mut best_value = self.endpoint_segment_value(idx, b, context.b_row, context.b_col);
        let mut best_weight = 0.0;

        let endpoint_a = self.endpoint_segment_value(idx, a, context.a_row, context.a_col);
        if endpoint_a < best_value {
            best_value = endpoint_a;
            best_weight = 1.0;
        }

        if let Some(weight) = context.constant_cost_stationary_weight() {
            if weight > 0.0 && weight < 1.0 {
                let value = context.objective(weight);
                if value < best_value {
                    best_value = value;
                    best_weight = weight;
                }
            }
        }

        if best_value.is_finite() {
            Some((best_value, best_weight))
        } else {
            None
        }
    }

    fn endpoint_segment_value(&self, idx: usize, q: usize, q_row: f64, q_col: f64) -> f64 {
        let (idx_row, idx_col) = self.row_col(idx);
        let plan_distance =
            self.physical_distance_coords(q_row, q_col, idx_row as f64, idx_col as f64);
        self.point_candidate(idx, q, plan_distance)
            .unwrap_or(f64::INFINITY)
    }

    fn surface_distance(&self, plan_distance: f64, dz: f64) -> f64 {
        if self.has_elevation && self.use_surface_distance {
            plan_distance.hypot(dz)
        } else {
            plan_distance
        }
    }

    fn vertical_angle(&self, plan_distance: f64, dz: f64) -> f64 {
        if self.has_elevation {
            dz.atan2(plan_distance).to_degrees()
        } else {
            0.0
        }
    }

    fn segment_clear_index_to_index(&self, a: usize, b: usize) -> bool {
        if !self.has_blocked_cells {
            return true;
        }
        let (a_row, a_col) = self.row_col(a);
        self.segment_clear_coord_to_index(a_row as f64, a_col as f64, b)
    }

    fn segment_clear_coord_to_index(&self, row0: f64, col0: f64, idx: usize) -> bool {
        if !self.has_blocked_cells {
            return true;
        }
        let (row1, col1) = self.row_col(idx);
        if self.segment_bounds_clear(row0, col0, row1, col1) {
            return true;
        }
        self.segment_grid_clear(row0, col0, row1 as f64, col1 as f64)
    }

    fn segment_grid_clear(&self, row0: f64, col0: f64, row1: f64, col1: f64) -> bool {
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

        Self::push_axis_crossings(&mut crossings, x0, dx);
        Self::push_axis_crossings(&mut crossings, y0, dy);
        crossings.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        crossings.dedup_by(|a, b| (*a - *b).abs() <= GRID_EPS);

        if !self.segment_point_clear(x0, y0) || !self.segment_point_clear(x1, y1) {
            return false;
        }
        for &t in &crossings {
            if !self.segment_point_clear(x0 + dx * t, y0 + dy * t) {
                return false;
            }
        }

        let mut previous = 0.0;
        for next in crossings.into_iter().chain(std::iter::once(1.0)) {
            if next - previous > GRID_EPS {
                let midpoint = 0.5 * (previous + next);
                if !self.segment_cell_clear(x0 + dx * midpoint, y0 + dy * midpoint) {
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

    fn segment_point_clear(&self, x: f64, y: f64) -> bool {
        let col_a = x.floor() as isize;
        let col_b = (x - GRID_EPS).floor() as isize;
        let row_a = y.floor() as isize;
        let row_b = (y - GRID_EPS).floor() as isize;
        let mut touched_any_cell = false;

        for row in [row_a, row_b] {
            for col in [col_a, col_b] {
                if row < 0 || col < 0 || row >= self.rows as isize || col >= self.cols as isize {
                    continue;
                }
                touched_any_cell = true;
                if !self.valid[self.idx(row as usize, col as usize)] {
                    return false;
                }
            }
        }
        touched_any_cell
    }

    fn segment_cell_clear(&self, x: f64, y: f64) -> bool {
        let row = y.floor() as isize;
        let col = x.floor() as isize;
        if row < 0 || col < 0 || row >= self.rows as isize || col >= self.cols as isize {
            return false;
        }
        self.valid[self.idx(row as usize, col as usize)]
    }

    fn segment_bounds_clear(&self, row0: f64, col0: f64, row1: usize, col1: usize) -> bool {
        if !row0.is_finite() || !col0.is_finite() {
            return false;
        }
        let max_row = (self.rows - 1) as f64;
        let max_col = (self.cols - 1) as f64;
        if row0 < 0.0 || col0 < 0.0 || row0 > max_row || col0 > max_col {
            return false;
        }

        let row_min = row0.min(row1 as f64).floor().max(0.0) as usize;
        let row_max = row0.max(row1 as f64).ceil().min(max_row) as usize;
        let col_min = col0.min(col1 as f64).floor().max(0.0) as usize;
        let col_max = col0.max(col1 as f64).ceil().min(max_col) as usize;
        self.blocked_count_in(row_min, row_max, col_min, col_max) == 0
    }

    fn blocked_count_in(
        &self,
        row_min: usize,
        row_max: usize,
        col_min: usize,
        col_max: usize,
    ) -> usize {
        let stride = self.cols + 1;
        let row0 = row_min;
        let row1 = row_max + 1;
        let col0 = col_min;
        let col1 = col_max + 1;
        self.blocked_prefix[row1 * stride + col1]
            - self.blocked_prefix[row0 * stride + col1]
            - self.blocked_prefix[row1 * stride + col0]
            + self.blocked_prefix[row0 * stride + col0]
    }

    fn physical_distance_coords(&self, row0: f64, col0: f64, row1: f64, col1: f64) -> f64 {
        ((col1 - col0) * self.cell_size_x).hypot((row1 - row0) * self.cell_size_y)
    }

    fn offset_idx(&self, row: usize, col: usize, offset: StencilOffset) -> Option<usize> {
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

    fn offset_within_radius(&self, dr: isize, dc: isize) -> bool {
        let dx = dc as f64 * self.cell_size_x;
        let dy = dr as f64 * self.cell_size_y;
        dx * dx + dy * dy <= self.search_radius_sq + EPS
    }

    fn should_parallelize_update(&self) -> bool {
        let min_stencil = if self.vf.kind != VerticalFactorKind::None {
            PARALLEL_UPDATE_MIN_STENCIL
        } else if self.has_blocked_cells {
            PARALLEL_BLOCKED_FLAT_UPDATE_MIN_STENCIL
        } else {
            PARALLEL_FLAT_UPDATE_MIN_STENCIL
        };
        self.stencil_offsets.len() >= min_stencil && rayon::current_num_threads() > 1
    }

    fn apply_update(&mut self, update: CandidateUpdate) {
        if update.value + 1.0e-10 < self.distance[update.idx] {
            self.distance[update.idx] = update.value;
            self.parent[update.idx] = update.parent;
            self.state[update.idx] = TRIAL;
            self.heap.push(HeapEntry {
                value: update.value,
                idx: update.idx,
            });
        }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    fn row_col(&self, idx: usize) -> (usize, usize) {
        (idx / self.cols, idx % self.cols)
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
}

struct SolveOutput {
    rows: usize,
    cols: usize,
    distance: Vec<f64>,
    parent: Vec<Parent>,
    cell_size_x: f64,
    cell_size_y: f64,
}

impl SolveOutput {
    fn back_direction(&self) -> Vec<f64> {
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

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn distance_accumulation<'py>(
    py: Python<'py>,
    sources: PyReadonlyArray2<'py, f64>,
    cost_surface: PyReadonlyArray2<'py, f64>,
    elevation: Option<PyReadonlyArray2<'py, f64>>,
    barriers: Option<PyReadonlyArray2<'py, bool>>,
    has_elevation: bool,
    use_surface_distance: bool,
    vf_kind: &str,
    zero_factor: f64,
    low_cut_angle: f64,
    high_cut_angle: f64,
    slope: f64,
    power: f64,
    cos_power: f64,
    sec_power: f64,
    cell_size_x: f64,
    cell_size_y: f64,
    search_radius: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sources = sources.as_array();
    let cost_surface = cost_surface.as_array();
    let elevation = elevation.as_ref().map(|array| array.as_array());
    let barriers = barriers.as_ref().map(|array| array.as_array());

    let shape = sources.shape();
    let rows = shape[0];
    let cols = shape[1];
    if cost_surface.shape() != shape {
        return Err(PyValueError::new_err(
            "cost_surface must have the same shape as sources",
        ));
    }
    if let Some(elevation) = &elevation {
        if elevation.shape() != shape {
            return Err(PyValueError::new_err(
                "elevation must have the same shape as sources",
            ));
        }
    }
    if let Some(barriers) = &barriers {
        if barriers.shape() != shape {
            return Err(PyValueError::new_err(
                "barriers must have the same shape as sources",
            ));
        }
    }
    if has_elevation && elevation.is_none() {
        return Err(PyValueError::new_err(
            "elevation is required when has_elevation is true",
        ));
    }
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("input rasters must not be empty"));
    }
    if cell_size_x <= 0.0
        || cell_size_y <= 0.0
        || search_radius <= 0.0
        || !cell_size_x.is_finite()
        || !cell_size_y.is_finite()
        || !search_radius.is_finite()
    {
        return Err(PyValueError::new_err(
            "cell sizes and search radius must be positive finite values",
        ));
    }
    if !zero_factor.is_finite()
        || !low_cut_angle.is_finite()
        || !high_cut_angle.is_finite()
        || !slope.is_finite()
        || !power.is_finite()
        || !cos_power.is_finite()
        || !sec_power.is_finite()
    {
        return Err(PyValueError::new_err(
            "vertical factor parameters must be finite",
        ));
    }
    if low_cut_angle >= high_cut_angle {
        return Err(PyValueError::new_err(
            "low_cut_angle must be less than high_cut_angle",
        ));
    }

    let vf = VerticalFactor {
        kind: VerticalFactorKind::parse(vf_kind)?,
        zero_factor,
        low_cut_angle,
        high_cut_angle,
        slope,
        power,
        cos_power,
        sec_power,
    };

    let mut cost = Vec::with_capacity(rows * cols);
    let mut elev = if has_elevation {
        Vec::with_capacity(rows * cols)
    } else {
        Vec::new()
    };
    let mut valid = Vec::with_capacity(rows * cols);
    let mut sources_flat = Vec::new();
    let mut has_blocked_cells = false;

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let raw_cost = cost_surface[[row, col]];
            let raw_elevation = elevation
                .as_ref()
                .map_or(0.0, |elevation| elevation[[row, col]]);
            let blocked = barriers
                .as_ref()
                .is_some_and(|barriers| barriers[[row, col]])
                || !raw_cost.is_finite()
                || (has_elevation && !raw_elevation.is_finite());
            has_blocked_cells |= blocked;
            valid.push(!blocked);
            cost.push(if raw_cost.is_finite() {
                raw_cost.max(MIN_COST)
            } else {
                f64::INFINITY
            });
            if has_elevation {
                elev.push(if raw_elevation.is_finite() {
                    raw_elevation
                } else {
                    0.0
                });
            }
            let source = sources[[row, col]];
            if source.is_finite() && source != 0.0 && !blocked {
                sources_flat.push(idx);
            }
        }
    }

    let solver = Solver::new(
        SolverInput {
            rows,
            cols,
            cost,
            elevation: elev,
            valid,
            has_blocked_cells,
        },
        SolverOptions {
            has_elevation,
            use_surface_distance,
            vf,
            cell_size_x,
            cell_size_y,
            search_radius,
        },
    );
    let output = solver.solve(&sources_flat)?;

    let back_direction = output.back_direction();
    let parent_a: Vec<i64> = output.parent.iter().map(|parent| parent.a).collect();
    let parent_b: Vec<i64> = output.parent.iter().map(|parent| parent.b).collect();
    let parent_weight: Vec<f64> = output.parent.iter().map(|parent| parent.weight).collect();

    let dict = PyDict::new(py);
    dict.set_item(
        "distance",
        Array2::from_shape_vec((rows, cols), output.distance)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "back_direction",
        Array2::from_shape_vec((rows, cols), back_direction)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "parent_a",
        Array2::from_shape_vec((rows, cols), parent_a)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "parent_b",
        Array2::from_shape_vec((rows, cols), parent_b)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    dict.set_item(
        "parent_weight",
        Array2::from_shape_vec((rows, cols), parent_weight)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .into_pyarray(py),
    )?;
    Ok(dict)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn optimal_path_as_line<'py>(
    py: Python<'py>,
    distance: PyReadonlyArray2<'py, f64>,
    parent_a: PyReadonlyArray2<'py, i64>,
    parent_b: PyReadonlyArray2<'py, i64>,
    parent_weight: PyReadonlyArray2<'py, f64>,
    start_row: isize,
    start_col: isize,
    cell_size_x: f64,
    cell_size_y: f64,
    origin_x: f64,
    origin_y: f64,
    max_steps: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let distance = distance.as_array();
    let parent_a = parent_a.as_array();
    let parent_b = parent_b.as_array();
    let parent_weight = parent_weight.as_array();
    let shape = distance.shape();
    let rows = shape[0];
    let cols = shape[1];

    if parent_a.shape() != shape || parent_b.shape() != shape || parent_weight.shape() != shape {
        return Err(PyValueError::new_err(
            "parent arrays must match distance shape",
        ));
    }
    if start_row < 0 || start_col < 0 || start_row >= rows as isize || start_col >= cols as isize {
        return Err(PyValueError::new_err("destination is outside the raster"));
    }

    let mut idx = start_row as usize * cols + start_col as usize;
    if !distance[[start_row as usize, start_col as usize]].is_finite() {
        return Err(PyValueError::new_err(
            "destination has no finite accumulated distance",
        ));
    }

    let step_limit = if max_steps == 0 {
        rows.saturating_mul(cols).saturating_mul(4).max(1)
    } else {
        max_steps
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
            cell_size_x,
            cell_size_y,
            origin_x,
            origin_y,
        );

        let a = parent_a[[row, col]];
        if a < 0 {
            break;
        }
        let b = parent_b[[row, col]];
        let weight = parent_weight[[row, col]];
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
                cell_size_x,
                cell_size_y,
                origin_x,
                origin_y,
            );

            let a_dist = distance[[a_row, a_col]];
            let b_dist = distance[[b_row, b_col]];
            idx = if a_dist <= b_dist { a_idx } else { b_idx };
        }

        guard += 1;
        if guard >= step_limit {
            return Err(PyRuntimeError::new_err(
                "path tracing exceeded max_steps before reaching a source",
            ));
        }
    }

    let vertices = coords.len() / 2;
    Ok(Array2::from_shape_vec((vertices, 2), coords)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
        .into_pyarray(py))
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_accumulation, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_path_as_line, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

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
    fn hiking_pace_matches_esri_table_at_zero_degrees() {
        assert_abs_diff_eq!(hiking_pace(0.0), 0.000198541, epsilon = 1.0e-7);
    }

    #[test]
    fn binary_factor_obeys_cut_angles() {
        let vf = VerticalFactor {
            kind: VerticalFactorKind::Binary,
            zero_factor: 2.0,
            low_cut_angle: -30.0,
            high_cut_angle: 30.0,
            slope: 0.0,
            power: 1.0,
            cos_power: 1.0,
            sec_power: 1.0,
        };

        assert_eq!(vf.factor(0.0), 2.0);
        assert!(vf.factor(30.0).is_infinite());
        assert!(vf.factor(-30.0).is_infinite());
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
