#[inline]
pub(crate) fn fast_hypot(a: f64, b: f64) -> f64 {
    let squared = a * a + b * b;
    if squared.is_finite() && (squared > 0.0 || (a == 0.0 && b == 0.0)) {
        squared.sqrt()
    } else {
        a.hypot(b)
    }
}

#[cfg(test)]
mod tests {
    use super::fast_hypot;

    #[test]
    fn fast_hypot_matches_regular_values() {
        assert_eq!(fast_hypot(3.0, 4.0), 5.0);
    }

    #[test]
    fn fast_hypot_falls_back_for_overflow() {
        assert_eq!(fast_hypot(1.0e308, 1.0e308), 1.0e308_f64.hypot(1.0e308));
    }

    #[test]
    fn fast_hypot_falls_back_for_underflow() {
        assert_eq!(fast_hypot(1.0e-300, 1.0e-300), 1.0e-300_f64.hypot(1.0e-300));
    }
}
