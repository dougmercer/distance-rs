use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;

const DEGREES_PER_RADIAN: f64 = 180.0 / PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VerticalFactorKind {
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
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
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
pub(crate) struct VerticalFactor {
    pub(crate) kind: VerticalFactorKind,
    pub(crate) zero_factor: f64,
    pub(crate) low_cut_radians: f64,
    pub(crate) high_cut_radians: f64,
    pub(crate) low_cut_slope: f64,
    pub(crate) high_cut_slope: f64,
    pub(crate) slope_per_radian: f64,
    pub(crate) power: f64,
    pub(crate) cos_power: f64,
    pub(crate) sec_power: f64,
}

impl VerticalFactor {
    pub(crate) fn from_py_dict(value: &Bound<'_, PyDict>) -> PyResult<Self> {
        let kind = required_string(value, "type")?;
        Self::from_degrees(
            VerticalFactorKind::parse(&kind)?,
            required_f64(value, "zero_factor")?,
            required_f64(value, "low_cut_angle")?,
            required_f64(value, "high_cut_angle")?,
            required_f64(value, "slope")?,
            required_f64(value, "power")?,
            required_f64(value, "cos_power")?,
            required_f64(value, "sec_power")?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_degrees(
        kind: VerticalFactorKind,
        zero_factor: f64,
        low_cut_angle: f64,
        high_cut_angle: f64,
        slope: f64,
        power: f64,
        cos_power: f64,
        sec_power: f64,
    ) -> PyResult<Self> {
        if low_cut_angle >= high_cut_angle {
            return Err(PyValueError::new_err(
                "low_cut_angle must be less than high_cut_angle",
            ));
        }
        Ok(Self {
            kind,
            zero_factor,
            low_cut_radians: low_cut_angle.to_radians(),
            high_cut_radians: high_cut_angle.to_radians(),
            low_cut_slope: cut_slope(low_cut_angle),
            high_cut_slope: cut_slope(high_cut_angle),
            slope_per_radian: slope * DEGREES_PER_RADIAN,
            power,
            cos_power,
            sec_power,
        })
    }

    #[cfg(test)]
    pub(crate) fn none() -> Self {
        Self::from_degrees(
            VerticalFactorKind::None,
            1.0,
            -90.0,
            90.0,
            0.0,
            1.0,
            1.0,
            1.0,
        )
        .expect("default vertical factor is valid")
    }

    pub(crate) fn factor_from_rise_run(&self, plan_distance: f64, dz: f64) -> f64 {
        if self.kind == VerticalFactorKind::None {
            return 1.0;
        }

        let slope = dz / plan_distance;
        if slope.is_finite() {
            self.factor_from_slope(slope)
        } else {
            self.factor_radians(dz.atan2(plan_distance))
        }
    }

    #[cfg(test)]
    pub(crate) fn factor(&self, angle_degrees: f64) -> f64 {
        self.factor_radians(angle_degrees.to_radians())
    }

    fn factor_from_slope(&self, slope: f64) -> f64 {
        if self.kind == VerticalFactorKind::None {
            return 1.0;
        }
        if !slope.is_finite() || slope <= self.low_cut_slope || slope >= self.high_cut_slope {
            return f64::INFINITY;
        }

        let factor = match self.kind {
            VerticalFactorKind::None => 1.0,
            VerticalFactorKind::Binary => self.zero_factor,
            VerticalFactorKind::Linear => self.zero_factor + self.slope_per_radian * slope.atan(),
            VerticalFactorKind::InverseLinear => {
                self.zero_factor + self.slope_per_radian * slope.atan()
            }
            VerticalFactorKind::SymmetricLinear => {
                self.zero_factor + self.slope_per_radian * slope.atan().abs()
            }
            VerticalFactorKind::SymmetricInverseLinear => {
                self.zero_factor + self.slope_per_radian * slope.atan().abs()
            }
            VerticalFactorKind::Cos => cos_factor_from_slope(slope, self.power),
            VerticalFactorKind::Sec => sec_factor_from_slope(slope, self.power),
            VerticalFactorKind::CosSec => {
                if slope < 0.0 {
                    cos_factor_from_slope(slope, self.cos_power)
                } else {
                    sec_factor_from_slope(slope, self.sec_power)
                }
            }
            VerticalFactorKind::SecCos => {
                if slope < 0.0 {
                    sec_factor_from_slope(slope, self.sec_power)
                } else {
                    cos_factor_from_slope(slope, self.cos_power)
                }
            }
            VerticalFactorKind::HikingTime => hiking_pace_from_slope(slope),
            VerticalFactorKind::BidirHikingTime => {
                0.5 * (hiking_pace_from_slope(slope) + hiking_pace_from_slope(-slope))
            }
        };

        finite_positive_or_infinite(factor)
    }

    fn factor_radians(&self, angle_radians: f64) -> f64 {
        if self.kind == VerticalFactorKind::None {
            return 1.0;
        }
        if !angle_radians.is_finite()
            || angle_radians <= self.low_cut_radians
            || angle_radians >= self.high_cut_radians
        {
            return f64::INFINITY;
        }

        let factor = match self.kind {
            VerticalFactorKind::None => 1.0,
            VerticalFactorKind::Binary => self.zero_factor,
            VerticalFactorKind::Linear => self.zero_factor + self.slope_per_radian * angle_radians,
            VerticalFactorKind::InverseLinear => {
                self.zero_factor + self.slope_per_radian * angle_radians
            }
            VerticalFactorKind::SymmetricLinear => {
                self.zero_factor + self.slope_per_radian * angle_radians.abs()
            }
            VerticalFactorKind::SymmetricInverseLinear => {
                self.zero_factor + self.slope_per_radian * angle_radians.abs()
            }
            VerticalFactorKind::Cos => cos_factor(angle_radians, self.power),
            VerticalFactorKind::Sec => sec_factor(angle_radians, self.power),
            VerticalFactorKind::CosSec => {
                if angle_radians < 0.0 {
                    cos_factor(angle_radians, self.cos_power)
                } else {
                    sec_factor(angle_radians, self.sec_power)
                }
            }
            VerticalFactorKind::SecCos => {
                if angle_radians < 0.0 {
                    sec_factor(angle_radians, self.sec_power)
                } else {
                    cos_factor(angle_radians, self.cos_power)
                }
            }
            VerticalFactorKind::HikingTime => hiking_pace_from_slope(angle_radians.tan()),
            VerticalFactorKind::BidirHikingTime => {
                let slope = angle_radians.tan();
                0.5 * (hiking_pace_from_slope(slope) + hiking_pace_from_slope(-slope))
            }
        };

        finite_positive_or_infinite(factor)
    }
}

fn required_string(value: &Bound<'_, PyDict>, name: &str) -> PyResult<String> {
    let Some(item) = value.get_item(name)? else {
        return Err(PyValueError::new_err(format!(
            "vertical factor missing {name}"
        )));
    };
    item.extract()
}

fn required_f64(value: &Bound<'_, PyDict>, name: &str) -> PyResult<f64> {
    let Some(item) = value.get_item(name)? else {
        return Err(PyValueError::new_err(format!(
            "vertical factor missing {name}"
        )));
    };
    let number = item.extract::<f64>()?;
    if number.is_finite() {
        Ok(number)
    } else {
        Err(PyValueError::new_err(format!(
            "vertical factor option {name} must be finite"
        )))
    }
}

fn cut_slope(angle_degrees: f64) -> f64 {
    if angle_degrees <= -90.0 {
        f64::NEG_INFINITY
    } else if angle_degrees >= 90.0 {
        f64::INFINITY
    } else {
        angle_degrees.to_radians().tan()
    }
}

fn finite_positive_or_infinite(factor: f64) -> f64 {
    if factor.is_finite() && factor > 0.0 {
        factor
    } else {
        f64::INFINITY
    }
}

fn cos_factor(angle_radians: f64, power: f64) -> f64 {
    angle_radians.cos().powf(power)
}

fn sec_factor(angle_radians: f64, power: f64) -> f64 {
    1.0 / angle_radians.cos().powf(power)
}

fn cos_factor_from_slope(slope: f64, power: f64) -> f64 {
    slope.hypot(1.0).recip().powf(power)
}

fn sec_factor_from_slope(slope: f64, power: f64) -> f64 {
    slope.hypot(1.0).powf(power)
}

#[cfg(test)]
pub(crate) fn hiking_pace(angle_degrees: f64) -> f64 {
    hiking_pace_from_slope(angle_degrees.to_radians().tan())
}

fn hiking_pace_from_slope(slope: f64) -> f64 {
    let speed_km_per_hour = 6.0 * (-3.5 * (slope + 0.05).abs()).exp();
    1.0 / (speed_km_per_hour * 1000.0)
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
        let vf = VerticalFactor::from_degrees(
            VerticalFactorKind::Binary,
            2.0,
            -30.0,
            30.0,
            0.0,
            1.0,
            1.0,
            1.0,
        )
        .unwrap();

        assert_eq!(vf.factor(0.0), 2.0);
        assert!(vf.factor(30.0).is_infinite());
        assert!(vf.factor(-30.0).is_infinite());
        assert_eq!(vf.factor_from_slope(0.0), 2.0);
        assert!(vf
            .factor_from_slope(30.0_f64.to_radians().tan())
            .is_infinite());
        assert!(vf
            .factor_from_slope((-30.0_f64).to_radians().tan())
            .is_infinite());
    }

    #[test]
    fn slope_cutoffs_keep_ninety_degree_defaults_open() {
        let vf = VerticalFactor::from_degrees(
            VerticalFactorKind::Binary,
            2.0,
            -90.0,
            90.0,
            0.0,
            1.0,
            1.0,
            1.0,
        )
        .unwrap();

        assert_eq!(vf.factor_from_slope(-1.0e20), 2.0);
        assert_eq!(vf.factor_from_slope(1.0e20), 2.0);
    }

    #[test]
    fn rise_run_factor_matches_degree_factor() {
        let kinds = [
            VerticalFactorKind::Binary,
            VerticalFactorKind::Linear,
            VerticalFactorKind::InverseLinear,
            VerticalFactorKind::SymmetricLinear,
            VerticalFactorKind::SymmetricInverseLinear,
            VerticalFactorKind::Cos,
            VerticalFactorKind::Sec,
            VerticalFactorKind::CosSec,
            VerticalFactorKind::SecCos,
            VerticalFactorKind::HikingTime,
            VerticalFactorKind::BidirHikingTime,
        ];
        let angles: [f64; 5] = [-45.0, -12.5, 0.0, 18.0, 45.0];

        for kind in kinds {
            let vf =
                VerticalFactor::from_degrees(kind, 1.4, -70.0, 70.0, 0.01, 1.3, 0.7, 1.6).unwrap();
            for angle in angles {
                let slope = angle.to_radians().tan();
                assert_abs_diff_eq!(
                    vf.factor_from_slope(slope),
                    vf.factor(angle),
                    epsilon = 1.0e-12
                );
            }
        }
    }
}
