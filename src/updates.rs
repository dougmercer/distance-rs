use crate::grid::{EPS, NEIGHBORS_8};
use crate::solver::{HeapEntry, Parent, Solver, TRIAL};

#[derive(Clone, Copy, Debug)]
struct CandidateUpdate {
    idx: usize,
    value: f64,
    parent: Parent,
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

    fn into_update(self, idx: usize) -> Option<CandidateUpdate> {
        if self.value.is_finite() {
            Some(CandidateUpdate {
                idx,
                value: self.value,
                parent: self.parent,
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
        if self.solver.barriers.has_blocked_cells()
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
        let ux = (self.a_col - self.b_col) * self.solver.grid.cell_size_x;
        let uy = (self.a_row - self.b_row) * self.solver.grid.cell_size_y;
        let vx = (self.b_col - self.p_col) * self.solver.grid.cell_size_x;
        let vy = (self.b_row - self.p_row) * self.solver.grid.cell_size_y;

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

        best.into_update(idx)
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

        best.into_update(idx)
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
            && !self.barriers.has_blocked_cells()
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
        if self.has_elevation {
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
}
