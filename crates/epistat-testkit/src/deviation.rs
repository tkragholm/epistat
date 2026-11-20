/// What [`worst_deviation`] does with NaN in either argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanRule {
    /// Both slices must be finite throughout, otherwise it panics.
    Finite,
    /// A position is NaN in both or in neither, otherwise it panics. Positions
    /// that are NaN in both are skipped. For aliased coefficients, which
    /// the fit reports as `None` and R as `NA`.
    Coincide,
}

/// The largest ratio of `|got - expected|` to `bound * max(|expected|, floor)`
/// over the positions, so a result of at most 1 means every position is within
/// the tolerance `bound` relative to its expected value, or absolute where the
/// expected value is below `floor`.
///
/// The ratio is computed as `(g - e).abs() / (bound * e.abs().max(floor))`.
/// An empty input gives 0.
///
/// # Panics
///
/// If the slices differ in length, or where `rule` says so.
#[must_use]
#[track_caller]
pub fn worst_deviation(
    got: &[f64],
    expected: &[f64],
    bound: f64,
    floor: f64,
    rule: NanRule,
) -> f64 {
    assert_eq!(got.len(), expected.len(), "lengths");
    let mut worst = 0.0f64;
    for (g, e) in got.iter().zip(expected) {
        match rule {
            NanRule::Finite => {
                assert!(g.is_finite() && e.is_finite(), "not finite: {g} {e}");
            }
            NanRule::Coincide => {
                assert_eq!(
                    g.is_nan(),
                    e.is_nan(),
                    "NaN differs: {got:?} against {expected:?}"
                );
                if e.is_nan() {
                    continue;
                }
            }
        }
        worst = worst.max((g - e).abs() / (bound * e.abs().max(floor)));
    }
    worst
}

/// The largest `|got - expected|` over the positions, in units of `bound`,
/// for values that may be undefined. A position that is `None` in both counts
/// as 0.
///
/// # Panics
///
/// If the slices differ in length, or a position is `None` in one slice only.
#[must_use]
#[track_caller]
pub fn worst_deviation_defined(got: &[Option<f64>], expected: &[Option<f64>], bound: f64) -> f64 {
    assert_eq!(got.len(), expected.len(), "lengths");
    got.iter()
        .zip(expected)
        .map(|(g, e)| defined_ratio(*g, *e, bound))
        .fold(0.0, f64::max)
}

/// [`worst_deviation_defined`] with an allowance per position instead of one
/// `bound`.
///
/// # Panics
///
/// If the slices differ in length, or a position is `None` in one slice only.
#[must_use]
#[track_caller]
pub fn worst_deviation_defined_each(
    got: &[Option<f64>],
    expected: &[Option<f64>],
    allowed: &[f64],
) -> f64 {
    assert_eq!(got.len(), expected.len(), "lengths");
    got.iter()
        .zip(expected)
        .zip(allowed)
        .map(|((g, e), a)| defined_ratio(*g, *e, *a))
        .fold(0.0, f64::max)
}

#[track_caller]
fn defined_ratio(got: Option<f64>, expected: Option<f64>, allowed: f64) -> f64 {
    match (got, expected) {
        (Some(g), Some(e)) => (g - e).abs() / allowed,
        (None, None) => 0.0,
        _ => panic!("limits differ in being defined: {got:?} {expected:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ratio_is_relative_above_the_floor_and_absolute_below() {
        let got = [110.0, 0.5e-3];
        let expected = [100.0, 0.0];
        // 10 / (0.1 * 100) = 1 and 0.5e-3 / (1e-3 * 1) = 0.5.
        let worst = worst_deviation(&got, &expected, 0.1, 1.0, NanRule::Finite);
        assert!((worst - 1.0).abs() < 1e-15);
        let worst = worst_deviation(&got[1..], &expected[1..], 1e-3, 1.0, NanRule::Finite);
        assert!((worst - 0.5).abs() < 1e-15);
    }

    #[test]
    fn an_empty_input_gives_zero() {
        assert_eq!(
            worst_deviation(&[], &[], 1.0, 1.0, NanRule::Finite).to_bits(),
            0.0f64.to_bits()
        );
    }

    #[test]
    #[should_panic(expected = "not finite")]
    fn finite_refuses_a_nan() {
        let _ = worst_deviation(&[f64::NAN], &[1.0], 1.0, 1.0, NanRule::Finite);
    }

    #[test]
    fn coincide_skips_positions_that_are_nan_in_both() {
        let worst = worst_deviation(
            &[f64::NAN, 2.0],
            &[f64::NAN, 1.0],
            1.0,
            1.0,
            NanRule::Coincide,
        );
        assert!((worst - 1.0).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "NaN differs")]
    fn coincide_refuses_a_nan_in_one() {
        let _ = worst_deviation(&[f64::NAN], &[1.0], 1.0, 1.0, NanRule::Coincide);
    }

    #[test]
    #[should_panic(expected = "lengths")]
    fn lengths_must_agree() {
        let _ = worst_deviation(&[1.0], &[], 1.0, 1.0, NanRule::Finite);
    }

    #[test]
    fn defined_values_are_in_units_of_the_bound() {
        let got = [Some(1.0), None, Some(2.0)];
        let expected = [Some(1.5), None, Some(2.0)];
        assert!((worst_deviation_defined(&got, &expected, 0.25) - 2.0).abs() < 1e-15);
        let each = worst_deviation_defined_each(&got, &expected, &[1.0, 1.0, 1.0]);
        assert!((each - 0.5).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "differ in being defined")]
    fn defined_refuses_a_mismatch() {
        let _ = worst_deviation_defined(&[None], &[Some(1.0)], 1.0);
    }
}
