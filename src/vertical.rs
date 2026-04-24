use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    pub(crate) low_cut_angle: f64,
    pub(crate) high_cut_angle: f64,
    pub(crate) slope: f64,
    pub(crate) power: f64,
    pub(crate) cos_power: f64,
    pub(crate) sec_power: f64,
}

impl VerticalFactor {
    pub(crate) fn from_py_dict(value: &Bound<'_, PyDict>) -> PyResult<Self> {
        let kind = required_string(value, "type")?;
        let vf = Self {
            kind: VerticalFactorKind::parse(&kind)?,
            zero_factor: required_f64(value, "zero_factor")?,
            low_cut_angle: required_f64(value, "low_cut_angle")?,
            high_cut_angle: required_f64(value, "high_cut_angle")?,
            slope: required_f64(value, "slope")?,
            power: required_f64(value, "power")?,
            cos_power: required_f64(value, "cos_power")?,
            sec_power: required_f64(value, "sec_power")?,
        };
        if vf.low_cut_angle >= vf.high_cut_angle {
            return Err(PyValueError::new_err(
                "low_cut_angle must be less than high_cut_angle",
            ));
        }
        Ok(vf)
    }

    pub(crate) fn factor(&self, angle_degrees: f64) -> f64 {
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

fn cos_factor(angle_degrees: f64, power: f64) -> f64 {
    angle_degrees.to_radians().cos().powf(power)
}

fn sec_factor(angle_degrees: f64, power: f64) -> f64 {
    1.0 / angle_degrees.to_radians().cos().powf(power)
}

pub(crate) fn hiking_pace(angle_degrees: f64) -> f64 {
    let slope = angle_degrees.to_radians().tan();
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
