//! Knot placement for restricted cubic splines and natural cubic splines, as
//! R's `Hmisc::rcspline.eval(knots.only = TRUE)` and `splines::ns` place them.
//!
//! # Harrell's rules
//!
//! [`harrell_knots`] reproduces the default knots of `rcspline.eval` \[1\] for
//! `nk` knots. With `outer = 0.1` for `nk = 3`, `0.05` for `nk = 4, 5, 6` and
//! `0.025` for `nk >= 7`, the knots are the type 7 sample quantiles at
//!
//! ```text
//! p_i = outer + i (1 - 2 outer) / (nk - 1),    i = 0, ..., nk - 1
//! ```
//!
//! (the last one is `1 - outer` exactly, as in R's `seq`). Four rules change
//! this, applied in R's order:
//!
//! 1. With at most `nk + 2` distinct values the knots are the distinct values
//!    without the smallest and the largest.
//! 2. When no interior distinct value holds `fractied` (default 0.05) or more
//!    of the sample and the smallest value does, that value's successor
//!    becomes the first knot and the smallest value is removed from the
//!    sample the other knots are computed on. The same for the largest value
//!    and the last knot. Each such knot takes one of the `nk` places. With one
//!    place left the knot is the median, with two the quantiles start at 0.5.
//! 3. When fewer than 100 values remain in the sample, the first knot is the
//!    fifth smallest value and the last the fifth largest, unless rule 2 set
//!    them.
//! 4. When the quantile knots coincide, so that fewer than `min(nk, 3)` are
//!    distinct, the quantiles are taken again at `2 nk` points, and if
//!    that fails and a tied first knot exists, the first knot, the middle value
//!    and the last knot are used. Fewer than three distinct knots is an error.
//!
//! The values of rule 2 are compared as exact doubles. R compares the labels
//! of `table(x)`, which are the values formatted to 15 significant digits, so
//! values that differ only beyond the fifteenth digit are one value in R and
//! two here.
//!
//! # Natural spline knots
//!
//! [`natural_spline_knots`] gives the interior knots of `splines::ns(x, df)`
//! \[2\]: `df - 1 - intercept` knots at the type 7 quantiles of the values
//! inside the boundary knots, at the probabilities
//! `seq(0, 1, length.out = k + 2)` without its ends. An interior knot equal to
//! a boundary knot is moved inward by an eighth of the gap to the nearest
//! different knot, as R does.
//!
//! # References
//!
//! 1. Harrell FE Jr. Regression Modeling Strategies. 2nd ed. Cham: Springer;
//!    2015. Section 2.4.5. The `Hmisc` function `rcspline.eval`.
//! 2. R Core Team. The `splines` package, function `ns`.

#![expect(
    clippy::float_cmp,
    reason = "ties and boundary knots are equalities of the sample's own values"
)]

use epistat_stats::quantile::type7;

use epistat_core::check;

use crate::Error;

/// R's `seq(from, to, length.out = n)`: the first and last values are exact.
fn seq(from: f64, to: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![from],
        _ => {
            let by = (to - from) / (n - 1) as f64;
            let mut out = Vec::with_capacity(n);
            out.push(from);
            out.extend((1..n - 1).map(|i| from + i as f64 * by));
            out.push(to);
            out
        }
    }
}

fn sorted_unique(sorted: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    for &v in sorted {
        if out.last() != Some(&v) {
            out.push(v);
        }
    }
    out
}

fn quantiles(sorted: &[f64], probabilities: &[f64]) -> Result<Vec<f64>, Error> {
    probabilities
        .iter()
        .map(|&p| type7(sorted, p).ok_or(Error::TooFewObservations))
        .collect()
}

fn median(sorted: &[f64]) -> Result<f64, Error> {
    let n = sorted.len();
    if n == 0 {
        return Err(Error::TooFewObservations);
    }
    Ok(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        sorted[n / 2 - 1].midpoint(sorted[n / 2])
    })
}

/// The knots `rcspline.eval(x, nk = nk, fractied = fractied, knots.only =
/// TRUE)` returns.
///
/// # Errors
///
/// [`Error::Input`] for a value that is not finite,
/// [`Error::TooFewObservations`] for fewer than six values or too few left
/// after the tied end values are set aside, [`Error::TooFewKnots`] for
/// `nk < 3`, and [`Error::TooFewDistinct`] where R stops because fewer
/// than three distinct knots result.
pub fn harrell_knots(x: &[f64], nk: usize, fractied: f64) -> Result<Vec<f64>, Error> {
    check::finite("x", x)?;
    let n = x.len();
    if n < 6 {
        return Err(Error::TooFewObservations);
    }
    if nk < 3 {
        return Err(Error::TooFewKnots);
    }
    let mut xx = x.to_vec();
    xx.sort_unstable_by(f64::total_cmp);
    let unique = sorted_unique(&xx);
    let nxu = unique.len();
    let mut knots: Vec<f64>;
    if nxu <= nk + 2 {
        knots = if nxu >= 2 {
            unique[1..nxu - 1].to_vec()
        } else {
            Vec::new()
        };
    } else {
        let mut outer = if nk > 3 { 0.05 } else { 0.1 };
        if nk > 6 {
            outer = 0.025;
        }
        let (mut override_first, mut override_last) = (false, false);
        let mut nke = nk;
        let (mut first_knot, mut last_knot): (Option<f64>, Option<f64>) = (None, None);
        if fractied > 0.0 && fractied < 1.0 {
            let frequency = |sorted: &[f64], value: f64| {
                let lo = sorted.partition_point(|&v| v < value);
                let hi = sorted.partition_point(|&v| v <= value);
                (hi - lo) as f64 / n as f64
            };
            let interior_max = unique[1..nxu - 1]
                .iter()
                .map(|&v| frequency(&xx, v))
                .fold(0.0_f64, f64::max);
            let (first_frequency, last_frequency) =
                (frequency(&xx, unique[0]), frequency(&xx, unique[nxu - 1]));
            if interior_max < fractied {
                if first_frequency >= fractied {
                    let first = unique[1];
                    xx.retain(|&v| v > first);
                    first_knot = Some(first);
                    nke -= 1;
                    override_first = true;
                }
                if last_frequency >= fractied {
                    let top = xx.last().copied().ok_or(Error::TooFewObservations)?;
                    let last = xx
                        .iter()
                        .rev()
                        .find(|&&v| v < top)
                        .copied()
                        .ok_or(Error::TooFewObservations)?;
                    xx.retain(|&v| v < last);
                    last_knot = Some(last);
                    nke -= 1;
                    override_last = true;
                }
            }
        }
        if nke == 1 {
            knots = vec![median(&xx)?];
        } else {
            if nxu <= nke {
                knots = unique.clone();
            } else {
                let p = if nke == 2 {
                    seq(0.5, 1.0 - outer, nke)
                } else {
                    seq(outer, 1.0 - outer, nke)
                };
                knots = quantiles(&xx, &p)?;
                if sorted_unique_count(&knots) < nke.min(3) {
                    knots = quantiles(&xx, &seq(outer, 1.0 - outer, 2 * nke))?;
                    if let Some(first) = first_knot
                        && sorted_unique_count(&knots) < 3
                    {
                        let mid = match (first_knot, last_knot) {
                            (Some(a), Some(b)) => a.midpoint(b),
                            _ => median(&xx)?,
                        };
                        let last = match last_knot {
                            Some(b) => b,
                            None => type7(&xx, 1.0 - outer).ok_or(Error::TooFewObservations)?,
                        };
                        knots = vec![first, mid, last];
                        knots.sort_unstable_by(f64::total_cmp);
                    }
                    if sorted_unique_count(&knots) < 3 {
                        return Err(Error::TooFewDistinct);
                    }
                }
            }
            if xx.len() < 100 {
                if xx.len() < 5 || knots.len() < nke {
                    return Err(Error::TooFewObservations);
                }
                if !override_first {
                    knots[0] = xx[4];
                }
                if !override_last {
                    knots[nke - 1] = xx[xx.len() - 5];
                }
            }
        }
        let mut all = Vec::with_capacity(knots.len() + 2);
        all.extend(first_knot);
        all.extend(knots);
        all.extend(last_knot);
        knots = all;
    }
    knots.sort_unstable_by(f64::total_cmp);
    knots.dedup();
    if knots.len() < 3 {
        return Err(Error::TooFewDistinct);
    }
    Ok(knots)
}

fn sorted_unique_count(values: &[f64]) -> usize {
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    sorted.dedup();
    sorted.len()
}

/// The interior knots of `splines::ns(x, df = df, intercept = intercept,
/// Boundary.knots = boundary)`, with `boundary` the range of `x` when `None`.
///
/// # Errors
///
/// [`Error::Input`] for a value that is not finite,
/// [`Error::Boundary`] for a boundary that is not increasing,
/// [`Error::TooFewObservations`] when no value lies inside the boundary,
/// and [`Error::OutsideBoundary`] when every interior knot equals a
/// boundary knot, where R stops.
pub fn natural_spline_knots(
    x: &[f64],
    df: usize,
    intercept: bool,
    boundary: Option<(f64, f64)>,
) -> Result<Vec<f64>, Error> {
    check::finite("x", x)?;
    let (lower, upper) = match boundary {
        Some(b) => b,
        None => (
            x.iter().copied().fold(f64::INFINITY, f64::min),
            x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ),
    };
    if !(lower.is_finite() && upper.is_finite() && lower < upper) {
        return Err(Error::Boundary);
    }
    let count = df.saturating_sub(1 + usize::from(intercept));
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut inside: Vec<f64> = x
        .iter()
        .copied()
        .filter(|&v| v >= lower && v <= upper)
        .collect();
    inside.sort_unstable_by(f64::total_cmp);
    let probabilities = seq(0.0, 1.0, count + 2);
    let mut knots = quantiles(&inside, &probabilities[1..=count])?;
    let at_lower = knots[0] == lower;
    let at_upper = knots[count - 1] == upper;
    if at_lower {
        let next = knots
            .iter()
            .copied()
            .filter(|&k| k > lower)
            .fold(f64::INFINITY, f64::min);
        if !next.is_finite() {
            return Err(Error::OutsideBoundary);
        }
        for k in &mut knots {
            if *k == lower {
                *k += (next - lower) / 8.0;
            }
        }
    }
    if at_upper {
        let previous = knots
            .iter()
            .copied()
            .filter(|&k| k < upper)
            .fold(f64::NEG_INFINITY, f64::max);
        if !previous.is_finite() {
            return Err(Error::OutsideBoundary);
        }
        for k in &mut knots {
            if *k == upper {
                *k -= (upper - previous) / 8.0;
            }
        }
    }
    Ok(knots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::InputError;

    #[test]
    fn seq_ends_are_exact() {
        let p = seq(0.05, 0.95, 5);
        assert_eq!(p[0], 0.05);
        assert_eq!(p[4], 0.95);
        assert!((p[2] - 0.5).abs() < 1e-16);
    }

    #[test]
    fn four_knots_of_a_large_sample_are_quantiles() {
        let x: Vec<f64> = (0..1000).map(|i| f64::from(i) / 10.0).collect();
        let knots = harrell_knots(&x, 4, 0.05).unwrap();
        assert_eq!(knots.len(), 4);
        for (k, p) in knots.iter().zip([0.05, 0.35, 0.65, 0.95]) {
            assert!((k - p * 99.9).abs() < 1e-9, "{k} {p}");
        }
    }

    #[test]
    fn a_small_sample_uses_the_fifth_smallest_and_largest() {
        let x: Vec<f64> = (1..=50).map(f64::from).collect();
        let knots = harrell_knots(&x, 5, 0.05).unwrap();
        assert_eq!(knots[0], 5.0);
        assert_eq!(knots[4], 46.0);
    }

    #[test]
    fn invalid_input_is_an_error() {
        assert_eq!(
            harrell_knots(&[1.0; 5], 4, 0.05),
            Err(Error::TooFewObservations)
        );
        assert_eq!(
            harrell_knots(&[f64::NAN; 10], 4, 0.05),
            Err(Error::Input(InputError::NotFinite { what: "x", row: 0 }))
        );
        assert_eq!(harrell_knots(&[1.0; 10], 2, 0.05), Err(Error::TooFewKnots));
        assert_eq!(
            harrell_knots(&[1.0; 200], 4, 0.05),
            Err(Error::TooFewDistinct)
        );
    }
}
