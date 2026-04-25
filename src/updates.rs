use crate::grid::{EPS, NEIGHBORS_8};
use crate::solver::{value_improves, HeapEntry, Parent, Solver, TRIAL};
use crate::vertical::VerticalFactorKind;

const GOLDEN_SECTION_ITERATIONS: usize = 14;

#[derive(Clone, Copy, Debug)]
struct CandidateUpdate {
    idx: usize,
    value: f64,
    parent: Parent,
    back_direction: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct AcceptedNeighbor {
    dr: isize,
    dc: isize,
    idx: usize,
    distance: f64,
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

    fn into_update(self, solver: &Solver, idx: usize) -> Option<CandidateUpdate> {
        if self.value.is_finite() {
            let back_direction = solver.back_direction_for_parent(idx, self.parent, self.value)?;
            Some(CandidateUpdate {
                idx,
                value: self.value,
                parent: self.parent,
                back_direction,
            })
        } else {
            None
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
    distance_a: f64,
    distance_b: f64,
    elevation_idx: f64,
    elevation_a: f64,
    elevation_b: f64,
    barrier_bounds_clear: bool,
}

impl<'a> SegmentContext<'a> {
    fn new(solver: &'a Solver, idx: usize, a: usize, b: usize) -> Self {
        let (a_row, a_col) = solver.row_col(a);
        let (b_row, b_col) = solver.row_col(b);
        let (p_row, p_col) = solver.row_col(idx);
        let row_min = a_row.min(b_row).min(p_row);
        let row_max = a_row.max(b_row).max(p_row);
        let col_min = a_col.min(b_col).min(p_col);
        let col_max = a_col.max(b_col).max(p_col);
        let barrier_bounds_clear =
            solver
                .barriers
                .cell_bounds_clear(&solver.grid, row_min, row_max, col_min, col_max);
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
            distance_a: solver.distance[a],
            distance_b: solver.distance[b],
            elevation_idx,
            elevation_a,
            elevation_b,
            barrier_bounds_clear,
        }
    }

    fn objective(&self, weight_a: f64) -> f64 {
        let weight_b = 1.0 - weight_a;
        let y_row = weight_a * self.a_row + weight_b * self.b_row;
        let y_col = weight_a * self.a_col + weight_b * self.b_col;
        if !self.barrier_bounds_clear
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

        let front_value = weight_a * self.distance_a + weight_b * self.distance_b;
        if self.solver.flat_cost_mode {
            return front_value + plan_distance * self.cost_idx;
        }

        let y_elevation = weight_a * self.elevation_a + weight_b * self.elevation_b;
        let dz = if self.solver.has_elevation {
            self.elevation_idx - y_elevation
        } else {
            0.0
        };
        let vf = self.solver.vf.factor_from_rise_run(plan_distance, dz);
        if !vf.is_finite() {
            return f64::INFINITY;
        }
        let surface_distance = self.solver.surface_distance(plan_distance, dz);

        front_value + surface_distance * self.cost_idx * vf
    }

    fn golden_section(&self, mut lo: f64, mut hi: f64) -> Option<(f64, f64)> {
        let gr = (5.0_f64.sqrt() - 1.0) * 0.5;
        let mut c = hi - gr * (hi - lo);
        let mut d = lo + gr * (hi - lo);
        let mut fc = self.objective(c);
        let mut fd = self.objective(d);

        for _ in 0..GOLDEN_SECTION_ITERATIONS {
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

    fn plane_back_direction(&self, value_idx: f64) -> Option<f64> {
        let p_x = self.p_col * self.solver.grid.cell_size_x;
        let p_y = self.p_row * self.solver.grid.cell_size_y;
        let a_x = self.a_col * self.solver.grid.cell_size_x;
        let a_y = self.a_row * self.solver.grid.cell_size_y;
        let b_x = self.b_col * self.solver.grid.cell_size_x;
        let b_y = self.b_row * self.solver.grid.cell_size_y;

        let ap_x = a_x - p_x;
        let ap_y = a_y - p_y;
        let bp_x = b_x - p_x;
        let bp_y = b_y - p_y;
        let det = ap_x * bp_y - ap_y * bp_x;
        if det.abs() <= EPS {
            return None;
        }

        let da = self.distance_a - value_idx;
        let db = self.distance_b - value_idx;
        let grad_x = (da * bp_y - ap_y * db) / det;
        let grad_y = (ap_x * db - da * bp_x) / det;
        Solver::direction_from_delta(-grad_x, -grad_y)
    }

    fn parent_back_direction(&self, weight_a: f64) -> Option<f64> {
        let weight_b = 1.0 - weight_a;
        let parent_row = weight_a * self.a_row + weight_b * self.b_row;
        let parent_col = weight_a * self.a_col + weight_b * self.b_col;
        self.solver
            .back_direction_to_coord(self.idx, parent_row, parent_col)
    }

    fn constant_surface_factor(&self) -> Option<f64> {
        let factor = match self.solver.vf.kind {
            VerticalFactorKind::None => 1.0,
            VerticalFactorKind::Binary
                if self.solver.vf.zero_factor > 0.0
                    && (!self.solver.has_elevation
                        || (self.solver.vf.low_cut_slope == f64::NEG_INFINITY
                            && self.solver.vf.high_cut_slope == f64::INFINITY)) =>
            {
                self.solver.vf.zero_factor
            }
            _ => return None,
        };
        factor.is_finite().then_some(factor)
    }

    fn constant_factor_stationary_weight(&self, factor: f64) -> Option<f64> {
        let ux = (self.a_col - self.b_col) * self.solver.grid.cell_size_x;
        let uy = (self.a_row - self.b_row) * self.solver.grid.cell_size_y;
        let uz = if self.solver.has_elevation {
            self.elevation_a - self.elevation_b
        } else {
            0.0
        };
        let vx = (self.b_col - self.p_col) * self.solver.grid.cell_size_x;
        let vy = (self.b_row - self.p_row) * self.solver.grid.cell_size_y;
        let vz = if self.solver.has_elevation {
            self.elevation_b - self.elevation_idx
        } else {
            0.0
        };

        let u_sq = ux * ux + uy * uy + uz * uz;
        if u_sq <= EPS {
            return None;
        }

        let local_cost = self.cost_idx * factor;
        if local_cost <= EPS {
            return None;
        }

        let front_slope = self.distance_a - self.distance_b;
        let scaled_slope = -front_slope / local_cost;
        if scaled_slope.abs() >= u_sq.sqrt() {
            return None;
        }

        let uv = ux * vx + uy * vy + uz * vz;
        let v_sq = vx * vx + vy * vy + vz * vz;
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

impl Solver {
    pub(crate) fn update_around_full(&mut self, center: usize) {
        let (center_row, center_col) = self.row_col(center);
        for offset_index in 0..self.grid.stencil_offsets.len() {
            let offset = self.grid.stencil_offsets[offset_index];
            let Some(idx) = self.offset_idx(center_row, center_col, offset) else {
                continue;
            };
            if !self.is_valid(idx) || self.is_accepted(idx) {
                continue;
            }
            if let Some(update) = self.compute_update(idx) {
                self.apply_update(update);
            }
        }
    }

    pub(crate) fn update_around_incremental(&mut self, center: usize) {
        let (center_row, center_col) = self.row_col(center);
        for offset_index in 0..self.grid.stencil_offsets.len() {
            let offset = self.grid.stencil_offsets[offset_index];
            let Some(idx) = self.offset_idx(center_row, center_col, offset) else {
                continue;
            };
            if !self.is_valid(idx) || self.is_accepted(idx) {
                continue;
            }
            if let Some(update) = self.compute_incremental_update(idx, center, offset.distance) {
                self.apply_update(update);
            }
        }
    }

    fn compute_update(&self, idx: usize) -> Option<CandidateUpdate> {
        let mut best = BestCandidate::new();
        let (neighbors, neighbor_count) = self.accepted_neighbors(idx);
        let neighbors = &neighbors[..neighbor_count];

        for &q in neighbors {
            self.consider_point_candidate(&mut best, idx, q.idx, q.distance);

            let (adjacent_neighbors, adjacent_neighbor_count) =
                self.accepted_neighbors_adjacent_to(idx, q.idx);
            for &r in &adjacent_neighbors[..adjacent_neighbor_count] {
                if r.idx > q.idx {
                    self.consider_segment_candidate(&mut best, idx, q.idx, r.idx);
                }
            }
        }

        best.into_update(self, idx)
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
        let mut lower_neighbors = [AcceptedNeighbor::default(); NEIGHBORS_8.len()];
        let mut lower_neighbor_count = 0usize;
        let (adjacent_neighbors, adjacent_neighbor_count) =
            self.accepted_neighbors_adjacent_to(idx, accepted);

        for &neighbor in &adjacent_neighbors[..adjacent_neighbor_count] {
            if neighbor.idx < accepted {
                lower_neighbors[lower_neighbor_count] = neighbor;
                lower_neighbor_count += 1;
            }
        }
        let lower_neighbors = &mut lower_neighbors[..lower_neighbor_count];
        lower_neighbors.sort_unstable_by_key(|neighbor| (neighbor.dr, neighbor.dc));

        let consider_center = |best: &mut BestCandidate| {
            self.consider_point_candidate(best, idx, accepted, plan_distance);
            for &neighbor in &adjacent_neighbors[..adjacent_neighbor_count] {
                if neighbor.idx > accepted {
                    self.consider_segment_candidate(best, idx, accepted, neighbor.idx);
                }
            }
        };

        let mut center_considered = false;
        for &neighbor in lower_neighbors.iter() {
            if !center_considered && (center_dr, center_dc) < (neighbor.dr, neighbor.dc) {
                consider_center(&mut best);
                center_considered = true;
            }

            self.consider_segment_candidate(&mut best, idx, neighbor.idx, accepted);
        }

        if !center_considered {
            consider_center(&mut best);
        }

        best.into_update(self, idx)
    }

    fn accepted_neighbors(&self, idx: usize) -> ([AcceptedNeighbor; 9], usize) {
        let (row, col) = self.row_col(idx);
        let mut neighbors = [AcceptedNeighbor::default(); 9];
        let mut count = 0usize;

        for offset in &self.grid.stencil_offsets {
            let Some(neighbor) = self.offset_idx(row, col, *offset) else {
                continue;
            };
            if !self.is_accepted(neighbor) {
                continue;
            }
            neighbors[count] = AcceptedNeighbor {
                dr: offset.dr,
                dc: offset.dc,
                idx: neighbor,
                distance: offset.distance,
            };
            count += 1;
        }

        (neighbors, count)
    }

    fn accepted_neighbors_adjacent_to(
        &self,
        idx: usize,
        accepted: usize,
    ) -> ([AcceptedNeighbor; 8], usize) {
        let (idx_row, idx_col) = self.row_col(idx);
        let (accepted_row, accepted_col) = self.row_col(accepted);
        let mut neighbors = [AcceptedNeighbor::default(); 8];
        let mut count = 0usize;

        for (dr, dc) in NEIGHBORS_8 {
            let neighbor_row = accepted_row as isize + dr;
            let neighbor_col = accepted_col as isize + dc;
            if neighbor_row < 0
                || neighbor_col < 0
                || neighbor_row >= self.rows() as isize
                || neighbor_col >= self.cols() as isize
            {
                continue;
            }

            let neighbor = self.idx(neighbor_row as usize, neighbor_col as usize);
            if !self.is_accepted(neighbor) {
                continue;
            }

            let neighbor_dr = neighbor_row - idx_row as isize;
            let neighbor_dc = neighbor_col - idx_col as isize;
            if !self.offset_within_radius(neighbor_dr, neighbor_dc) {
                continue;
            }

            neighbors[count] = AcceptedNeighbor {
                dr: neighbor_dr,
                dc: neighbor_dc,
                idx: neighbor,
                distance: 0.0,
            };
            count += 1;
        }

        (neighbors, count)
    }

    fn consider_point_candidate(
        &self,
        best: &mut BestCandidate,
        idx: usize,
        q: usize,
        plan_distance: f64,
    ) {
        if let Some(value) = self.point_candidate(idx, q, plan_distance) {
            best.consider(value, Parent::point(q));
        }
    }

    fn consider_segment_candidate(&self, best: &mut BestCandidate, idx: usize, a: usize, b: usize) {
        if let Some((value, weight)) = self.segment_candidate(idx, a, b) {
            best.consider(value, Parent::segment(a, b, weight));
        }
    }

    fn point_candidate(&self, idx: usize, q: usize, plan_distance: f64) -> Option<f64> {
        if !self.segment_clear_index_to_index(q, idx) {
            return None;
        }
        if plan_distance <= EPS {
            return None;
        }
        if self.flat_cost_mode {
            return Some(self.distance[q] + plan_distance * self.cost[idx]);
        }
        let dz = if self.has_elevation {
            self.elevation[idx] - self.elevation[q]
        } else {
            0.0
        };
        let vf = self.vf.factor_from_rise_run(plan_distance, dz);
        if !vf.is_finite() {
            return None;
        }
        let surface_distance = self.surface_distance(plan_distance, dz);
        Some(self.distance[q] + surface_distance * self.cost[idx] * vf)
    }

    fn segment_candidate(&self, idx: usize, a: usize, b: usize) -> Option<(f64, f64)> {
        let context = SegmentContext::new(self, idx, a, b);
        if context.barrier_bounds_clear {
            if let Some(factor) = context.constant_surface_factor() {
                return self.constant_factor_segment_candidate(idx, a, b, &context, factor);
            }
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

    fn constant_factor_segment_candidate(
        &self,
        idx: usize,
        a: usize,
        b: usize,
        context: &SegmentContext<'_>,
        factor: f64,
    ) -> Option<(f64, f64)> {
        let mut best_value = self.endpoint_segment_value(idx, b, context.b_row, context.b_col);
        let mut best_weight = 0.0;

        let endpoint_a = self.endpoint_segment_value(idx, a, context.a_row, context.a_col);
        if endpoint_a < best_value {
            best_value = endpoint_a;
            best_weight = 1.0;
        }

        if let Some(weight) = context.constant_factor_stationary_weight(factor) {
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

    fn segment_back_direction(
        &self,
        context: &SegmentContext<'_>,
        value: f64,
        weight: f64,
    ) -> Option<f64> {
        context
            .plane_back_direction(value)
            .or_else(|| context.parent_back_direction(weight))
    }

    fn back_direction_for_parent(&self, idx: usize, parent: Parent, value: f64) -> Option<f64> {
        if parent.a < 0 {
            return None;
        }
        let a = parent.a as usize;
        if parent.b < 0 {
            return self.back_direction_to_index(idx, a);
        }

        let b = parent.b as usize;
        let context = SegmentContext::new(self, idx, a, b);
        self.segment_back_direction(&context, value, parent.weight)
    }

    fn surface_distance(&self, plan_distance: f64, dz: f64) -> f64 {
        if self.has_elevation {
            plan_distance.hypot(dz)
        } else {
            plan_distance
        }
    }

    fn apply_update(&mut self, update: CandidateUpdate) {
        if value_improves(update.value, self.distance[update.idx]) {
            self.distance[update.idx] = update.value;
            self.parent[update.idx] = update.parent;
            self.back_direction[update.idx] = update.back_direction;
            self.state[update.idx] = TRIAL;
            self.heap.push(HeapEntry {
                value: update.value,
                idx: update.idx,
            });
        }
    }

    fn back_direction_to_index(&self, idx: usize, target: usize) -> Option<f64> {
        let (target_row, target_col) = self.row_col(target);
        self.back_direction_to_coord(idx, target_row as f64, target_col as f64)
    }

    fn back_direction_to_coord(&self, idx: usize, target_row: f64, target_col: f64) -> Option<f64> {
        let (row, col) = self.row_col(idx);
        let dx = (target_col - col as f64) * self.grid.cell_size_x;
        let dy = (target_row - row as f64) * self.grid.cell_size_y;
        Self::direction_from_delta(dx, dy)
    }

    fn direction_from_delta(dx: f64, dy: f64) -> Option<f64> {
        if !dx.is_finite() || !dy.is_finite() || dx.hypot(dy) <= EPS {
            return None;
        }
        let mut degrees = dx.atan2(-dy).to_degrees();
        if degrees < 0.0 {
            degrees += 360.0;
        }
        Some(degrees)
    }
}
