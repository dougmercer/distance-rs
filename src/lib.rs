use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::f64::consts::PI;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const FAR: u8 = 0;
const TRIAL: u8 = 1;
const ACCEPTED: u8 = 2;
const MIN_COST: f64 = 1.0e-12;
const EPS: f64 = 1.0e-12;

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
    has_elevation: bool,
    use_surface_distance: bool,
    vf: VerticalFactor,
    cell_size_x: f64,
    cell_size_y: f64,
    search_radius: f64,
    radius_rows: isize,
    radius_cols: isize,
    distance: Vec<f64>,
    parent: Vec<Parent>,
    state: Vec<u8>,
    heap: BinaryHeap<HeapEntry>,
}

struct SolverInput {
    rows: usize,
    cols: usize,
    cost: Vec<f64>,
    elevation: Vec<f64>,
    valid: Vec<bool>,
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
        let n = input.rows * input.cols;
        Self {
            rows: input.rows,
            cols: input.cols,
            cost: input.cost,
            elevation: input.elevation,
            valid: input.valid,
            has_elevation: options.has_elevation,
            use_surface_distance: options.use_surface_distance,
            vf: options.vf,
            cell_size_x: options.cell_size_x,
            cell_size_y: options.cell_size_y,
            search_radius: options.search_radius,
            radius_rows,
            radius_cols,
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
            self.update_around(idx);
        }

        while let Some(entry) = self.heap.pop() {
            if self.state[entry.idx] == ACCEPTED {
                continue;
            }
            if entry.value > self.distance[entry.idx] + 1.0e-10 {
                continue;
            }

            self.state[entry.idx] = ACCEPTED;
            self.update_around(entry.idx);
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

    fn update_around(&mut self, center: usize) {
        let (center_row, center_col) = self.row_col(center);
        let row_min = (center_row as isize - self.radius_rows).max(0) as usize;
        let row_max = (center_row as isize + self.radius_rows).min(self.rows as isize - 1) as usize;
        let col_min = (center_col as isize - self.radius_cols).max(0) as usize;
        let col_max = (center_col as isize + self.radius_cols).min(self.cols as isize - 1) as usize;

        for row in row_min..=row_max {
            for col in col_min..=col_max {
                let idx = self.idx(row, col);
                if !self.valid[idx] || self.state[idx] == ACCEPTED {
                    continue;
                }
                if self.physical_distance_indices(center, idx) > self.search_radius + EPS {
                    continue;
                }
                if let Some((value, parent)) = self.compute_update(idx) {
                    if value + 1.0e-10 < self.distance[idx] {
                        self.distance[idx] = value;
                        self.parent[idx] = parent;
                        self.state[idx] = TRIAL;
                        self.heap.push(HeapEntry { value, idx });
                    }
                }
            }
        }
    }

    fn compute_update(&self, idx: usize) -> Option<(f64, Parent)> {
        let (row, col) = self.row_col(idx);
        let row_min = (row as isize - self.radius_rows).max(0) as usize;
        let row_max = (row as isize + self.radius_rows).min(self.rows as isize - 1) as usize;
        let col_min = (col as isize - self.radius_cols).max(0) as usize;
        let col_max = (col as isize + self.radius_cols).min(self.cols as isize - 1) as usize;

        let mut best = f64::INFINITY;
        let mut best_parent = Parent::none();

        for q_row in row_min..=row_max {
            for q_col in col_min..=col_max {
                let q = self.idx(q_row, q_col);
                if self.state[q] != ACCEPTED {
                    continue;
                }
                if self.physical_distance_indices(q, idx) > self.search_radius + EPS {
                    continue;
                }
                if let Some(value) = self.point_candidate(idx, q) {
                    if value < best {
                        best = value;
                        best_parent = Parent::point(q);
                    }
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
                    if self.physical_distance_indices(r, idx) > self.search_radius + EPS {
                        continue;
                    }
                    if let Some((value, weight)) = self.segment_candidate(idx, q, r) {
                        if value < best {
                            best = value;
                            best_parent = Parent::segment(q, r, weight);
                        }
                    }
                }
            }
        }

        if best.is_finite() {
            Some((best, best_parent))
        } else {
            None
        }
    }

    fn point_candidate(&self, idx: usize, q: usize) -> Option<f64> {
        if !self.segment_clear_index_to_index(q, idx) {
            return None;
        }
        let (q_row, q_col) = self.row_col(q);
        let (p_row, p_col) = self.row_col(idx);
        let plan_distance =
            self.physical_distance_coords(q_row as f64, q_col as f64, p_row as f64, p_col as f64);
        if plan_distance <= EPS {
            return None;
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
        let local_cost = 0.5 * (self.cost[idx] + self.cost[q]);
        Some(self.distance[q] + surface_distance * local_cost * vf)
    }

    fn segment_candidate(&self, idx: usize, a: usize, b: usize) -> Option<(f64, f64)> {
        let samples = 8usize;
        let mut best_value = f64::INFINITY;
        let mut best_weight = 0.0;
        let mut best_sample = 0usize;

        for i in 0..=samples {
            let weight = i as f64 / samples as f64;
            let value = self.segment_objective(idx, a, b, weight);
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
            if let Some((value, weight)) = self.golden_section(idx, a, b, lo, hi) {
                if value < best_value {
                    best_value = value;
                    best_weight = weight;
                }
            }
        }

        Some((best_value, best_weight))
    }

    fn golden_section(
        &self,
        idx: usize,
        a: usize,
        b: usize,
        mut lo: f64,
        mut hi: f64,
    ) -> Option<(f64, f64)> {
        let gr = (5.0_f64.sqrt() - 1.0) * 0.5;
        let mut c = hi - gr * (hi - lo);
        let mut d = lo + gr * (hi - lo);
        let mut fc = self.segment_objective(idx, a, b, c);
        let mut fd = self.segment_objective(idx, a, b, d);

        for _ in 0..24 {
            if fc < fd {
                hi = d;
                d = c;
                fd = fc;
                c = hi - gr * (hi - lo);
                fc = self.segment_objective(idx, a, b, c);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + gr * (hi - lo);
                fd = self.segment_objective(idx, a, b, d);
            }
        }

        let weight = 0.5 * (lo + hi);
        let value = self.segment_objective(idx, a, b, weight);
        if value.is_finite() {
            Some((value, weight))
        } else {
            None
        }
    }

    fn segment_objective(&self, idx: usize, a: usize, b: usize, weight_a: f64) -> f64 {
        let weight_b = 1.0 - weight_a;
        let (a_row, a_col) = self.row_col(a);
        let (b_row, b_col) = self.row_col(b);
        let (p_row, p_col) = self.row_col(idx);

        let y_row = weight_a * a_row as f64 + weight_b * b_row as f64;
        let y_col = weight_a * a_col as f64 + weight_b * b_col as f64;
        if !self.segment_clear_coord_to_index(y_row, y_col, idx) {
            return f64::INFINITY;
        }

        let plan_distance = self.physical_distance_coords(y_row, y_col, p_row as f64, p_col as f64);
        if plan_distance <= EPS {
            return f64::INFINITY;
        }

        let y_elevation = weight_a * self.elevation[a] + weight_b * self.elevation[b];
        let dz = if self.has_elevation {
            self.elevation[idx] - y_elevation
        } else {
            0.0
        };
        let surface_distance = self.surface_distance(plan_distance, dz);
        let angle = self.vertical_angle(plan_distance, dz);
        let vf = self.vf.factor(angle);
        if !vf.is_finite() {
            return f64::INFINITY;
        }

        let y_cost = weight_a * self.cost[a] + weight_b * self.cost[b];
        let local_cost = 0.5 * (self.cost[idx] + y_cost);
        let front_value = weight_a * self.distance[a] + weight_b * self.distance[b];

        front_value + surface_distance * local_cost * vf
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
        let (a_row, a_col) = self.row_col(a);
        self.segment_clear_coord_to_index(a_row as f64, a_col as f64, b)
    }

    fn segment_clear_coord_to_index(&self, row0: f64, col0: f64, idx: usize) -> bool {
        let (row1, col1) = self.row_col(idx);
        let d_row = row1 as f64 - row0;
        let d_col = col1 as f64 - col0;
        let steps = ((d_row.abs().max(d_col.abs())) * 2.0).ceil().max(1.0) as usize;

        for step in 0..=steps {
            let t = step as f64 / steps as f64;
            let row = row0 + t * d_row;
            let col = col0 + t * d_col;
            let nearest_row = row.round() as isize;
            let nearest_col = col.round() as isize;
            if nearest_row < 0
                || nearest_col < 0
                || nearest_row >= self.rows as isize
                || nearest_col >= self.cols as isize
            {
                return false;
            }
            let sample = self.idx(nearest_row as usize, nearest_col as usize);
            if !self.valid[sample] {
                return false;
            }
        }
        true
    }

    fn physical_distance_indices(&self, a: usize, b: usize) -> f64 {
        let (a_row, a_col) = self.row_col(a);
        let (b_row, b_col) = self.row_col(b);
        self.physical_distance_coords(a_row as f64, a_col as f64, b_row as f64, b_col as f64)
    }

    fn physical_distance_coords(&self, row0: f64, col0: f64, row1: f64, col1: f64) -> f64 {
        ((col1 - col0) * self.cell_size_x).hypot((row1 - row0) * self.cell_size_y)
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    fn row_col(&self, idx: usize) -> (usize, usize) {
        (idx / self.cols, idx % self.cols)
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
    elevation: PyReadonlyArray2<'py, f64>,
    barriers: PyReadonlyArray2<'py, bool>,
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
    let elevation = elevation.as_array();
    let barriers = barriers.as_array();

    let shape = sources.shape();
    let rows = shape[0];
    let cols = shape[1];
    if cost_surface.shape() != shape || elevation.shape() != shape || barriers.shape() != shape {
        return Err(PyValueError::new_err(
            "all input rasters must have the same shape",
        ));
    }
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err("input rasters must not be empty"));
    }
    if cell_size_x <= 0.0 || cell_size_y <= 0.0 || search_radius <= 0.0 {
        return Err(PyValueError::new_err(
            "cell sizes and search radius must be positive",
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
    let mut elev = Vec::with_capacity(rows * cols);
    let mut valid = Vec::with_capacity(rows * cols);
    let mut sources_flat = Vec::new();

    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let raw_cost = cost_surface[[row, col]];
            let raw_elevation = elevation[[row, col]];
            let blocked = barriers[[row, col]]
                || !raw_cost.is_finite()
                || (has_elevation && !raw_elevation.is_finite());
            valid.push(!blocked);
            cost.push(if raw_cost.is_finite() {
                raw_cost.max(MIN_COST)
            } else {
                f64::INFINITY
            });
            elev.push(if raw_elevation.is_finite() {
                raw_elevation
            } else {
                0.0
            });
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
}
