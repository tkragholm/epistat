/// How far a value may be from its reference. A value passes when it is
/// within any one of the bounds that are set; with none set it must be equal.
///
/// NaN matches NaN, and an infinity matches only the same infinity.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tolerance {
    abs: Option<f64>,
    rel: Option<f64>,
    ulps: Option<u64>,
}

impl Tolerance {
    /// Bit-for-bit equality, apart from the sign of zero.
    #[must_use]
    pub const fn exact() -> Self {
        Self {
            abs: None,
            rel: None,
            ulps: None,
        }
    }

    #[must_use]
    pub const fn abs(self, bound: f64) -> Self {
        Self {
            abs: Some(bound),
            ..self
        }
    }

    /// `|got - expected| <= bound * |expected|`.
    #[must_use]
    pub const fn rel(self, bound: f64) -> Self {
        Self {
            rel: Some(bound),
            ..self
        }
    }

    /// At most `bound` representable doubles apart.
    #[must_use]
    pub const fn ulps(self, bound: u64) -> Self {
        Self {
            ulps: Some(bound),
            ..self
        }
    }

    #[must_use]
    #[expect(
        clippy::float_cmp,
        reason = "an exact match passes before any bound is tried"
    )]
    pub fn accepts(&self, got: f64, expected: f64) -> bool {
        if got.is_nan() || expected.is_nan() {
            return got.is_nan() && expected.is_nan();
        }
        if got == expected {
            return true;
        }
        if got.is_infinite() || expected.is_infinite() {
            return false;
        }
        let gap = (got - expected).abs();
        self.abs.is_some_and(|bound| gap <= bound)
            || self.rel.is_some_and(|bound| gap <= bound * expected.abs())
            || self
                .ulps
                .is_some_and(|bound| ulps_apart(got, expected) <= bound)
    }
}

/// How many representable doubles lie between `a` and `b`, counting across
/// zero; `u64::MAX` when either is NaN.
#[must_use]
#[expect(
    clippy::cast_possible_wrap,
    reason = "the bits are reordered, not converted"
)]
pub fn ulps_apart(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    // Map the bits to integers that are ordered as the doubles are.
    let ordered = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    ordered(a).abs_diff(ordered(b))
}

/// Checks that `got` and `expected` have the same length and that
/// `tolerance` accepts every pair.
///
/// # Panics
///
/// At the first pair it rejects, naming the index and the distance.
#[track_caller]
pub fn assert_close(got: &[f64], expected: &[f64], tolerance: Tolerance) {
    assert_eq!(got.len(), expected.len(), "lengths differ");
    for (index, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            tolerance.accepts(g, e),
            "index {index}: got {g:e}, expected {e:e} ({} ulps apart, tolerance {tolerance:?})",
            ulps_apart(g, e)
        );
    }
}

/// Checks that two estimates of one quantity agree within `k` combined
/// standard errors: `|a - b| <= k * sqrt(se_a^2 + se_b^2)`.
///
/// Use it where both values carry sampling error that is independent between
/// them, such as a Monte Carlo estimate against a reference computed with
/// other draws, or a posterior mean against a reference chain. The standard
/// errors must come from the draws themselves, not be set by trying seeds. If
/// the reference is exact, pass `0.0` for its standard error. A tolerance in
/// [`Tolerance`] is the tool for values that should agree up to rounding.
///
/// If the difference is normal with the combined standard error, a `k` of 4
/// rejects a correct implementation with probability about 6e-5.
///
/// # Panics
///
/// If `k` or a standard error is negative or NaN, if both standard errors are
/// zero and the values differ, or if the difference exceeds the bound. The
/// message gives the difference in combined standard errors.
#[track_caller]
pub fn assert_within_standard_errors(a: f64, se_a: f64, b: f64, se_b: f64, k: f64) {
    assert!(
        k >= 0.0 && se_a >= 0.0 && se_b >= 0.0,
        "k and the standard errors must be non-negative: k {k}, se {se_a}, {se_b}"
    );
    let combined = se_a.hypot(se_b);
    let gap = (a - b).abs();
    assert!(
        gap <= k * combined,
        "{a:e} (se {se_a:e}) and {b:e} (se {se_b:e}) differ by {:.3} combined standard errors, \
         allowed {k}",
        gap / combined
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulps_count_across_zero() {
        assert_eq!(ulps_apart(1.0, 1.0), 0);
        assert_eq!(ulps_apart(1.0, f64::from_bits(1.0f64.to_bits() + 3)), 3);
        assert_eq!(ulps_apart(0.0, -0.0), 0);
        assert_eq!(ulps_apart(f64::from_bits(1), -f64::from_bits(1)), 2);
        assert_eq!(ulps_apart(f64::NAN, 1.0), u64::MAX);
    }

    #[test]
    fn any_bound_suffices() {
        let tolerance = Tolerance::exact().abs(1e-12).rel(1e-6);
        assert!(tolerance.accepts(1.0 + 1e-13, 1.0));
        assert!(tolerance.accepts(1e6 + 0.5, 1e6));
        assert!(!tolerance.accepts(1.1, 1.0));
        assert!(Tolerance::exact().accepts(f64::NAN, f64::NAN));
        assert!(!Tolerance::exact().abs(1.0).accepts(f64::INFINITY, 1e308));
        assert!(!Tolerance::exact().accepts(1.0, f64::from_bits(1.0f64.to_bits() + 1)));
    }

    #[test]
    #[should_panic(expected = "index 1")]
    fn assert_close_names_the_index() {
        assert_close(&[1.0, 2.0], &[1.0, 2.5], Tolerance::exact().abs(0.1));
    }

    #[test]
    fn within_standard_errors_passes_inside_the_bound() {
        assert_within_standard_errors(1.0, 0.3, 1.5, 0.4, 1.0);
        assert_within_standard_errors(1.0, 0.0, 1.0, 0.0, 3.0);
        assert_within_standard_errors(2.0, 0.5, 0.0, 0.0, 4.0);
    }

    #[test]
    #[should_panic(expected = "differ by 2.000 combined standard errors")]
    fn within_standard_errors_reports_the_gap_in_standard_errors() {
        assert_within_standard_errors(1.0, 0.3, 2.0, 0.4, 1.0);
    }

    #[test]
    #[should_panic(expected = "differ by inf")]
    fn within_standard_errors_needs_an_error_to_allow_a_difference() {
        assert_within_standard_errors(1.0, 0.0, 1.1, 0.0, 3.0);
    }
}
