//! The approximations that Studies I and II reproduce.
//!
//! These functions are less accurate than the ones in [`crate::dist`] and
//! [`crate::quantile`], and a new caller should use those. They are kept
//! because published results were computed with them: the cost-model
//! intervals of Study II (`societal_costs`) with [`cdf_as26`], [`erf_as26`],
//! [`quantile_acklam`] and [`rounded_rank`], and the aggregate quantiles of
//! Study I (`parental_income`) with [`quantile_acklam`]. Each is the formula
//! below, in a fixed order of operations, so it gives the same bits as those
//! analyses did.
//!
//! # Algorithm
//!
//! [`erf_as26`] is formula 7.1.26 of Abramowitz and Stegun \[1\]. With
//! `t = 1 / (1 + 0.3275911 x)` for `x >= 0`,
//!
//! ```text
//! erf(x) = 1 - (a_1 t + a_2 t² + a_3 t³ + a_4 t⁴ + a_5 t⁵) exp(-x²),
//! ```
//!
//! and `erf(-x) = -erf(x)`. The absolute error is below `1.5e-7`. [`cdf_as26`] is
//! `(1 + erf(x / sqrt(2))) / 2`, with the same bound. [`quantile_acklam`] is the
//! rational approximation of Acklam \[2\], a ratio of degree 5 and 5 polynomials in
//! `q²` times `q` for `0.02425 <= p <= 0.97575`, and a ratio in
//! `sqrt(-2 ln p)` for the tails, mirrored in the upper tail. Its relative error is
//! below `1.2e-9`. The tests check both bounds against the accurate normal
//! distribution function and quantile.
//!
//! [`rounded_rank`] returns the order statistic at the zero-based position
//! `round(p (n - 1))`, rounding halves away from zero. It is not one of the
//! sample quantiles of Hyndman and Fan; R's type 7 interpolates between the
//! two order statistics around `1 + (n - 1) p` instead.
//!
//! # References
//!
//! 1. Abramowitz M, Stegun IA, editors. Handbook of Mathematical Functions with
//!    Formulas, Graphs, and Mathematical Tables. National Bureau of Standards
//!    Applied Mathematics Series 55. Washington, DC: US Government Printing
//!    Office; 1964. Formula 7.1.26.
//! 2. Acklam PJ. An algorithm for computing the inverse normal cumulative
//!    distribution function. Web note; 2003.

#![expect(
    clippy::float_cmp,
    reason = "the endpoints 0 and 1 are compared exactly"
)]

use std::f64::consts::SQRT_2;

/// [`crate::dist::normal::cdf`] by the error-function approximation 7.1.26 of Abramowitz and
/// Stegun (1964), whose absolute error is below `1.5e-7`.
#[must_use]
pub fn cdf_as26(x: f64) -> f64 {
    f64::midpoint(1.0, erf_as26(x / SQRT_2))
}

/// The error function by Abramowitz and Stegun 7.1.26.
#[must_use]
pub fn erf_as26(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = (((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736) * t
        + 0.254_829_592;
    sign * (1.0 - poly * t * (-x * x).exp())
}

/// [`crate::dist::normal::quantile`] by Acklam's rational approximation, whose relative error is
/// below `1.2e-9`. Returns `-inf` at 0, `inf` at 1 and NaN outside `[0, 1]`.
#[must_use]
pub fn quantile_acklam(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;

    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    if p < P_LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    }
}

/// The order statistic nearest to position `p (n - 1)`, rounding halves away
/// from zero. Not one of the Hyndman and Fan types. `None` for an empty
/// sample, a sample with NaN or one that is not sorted ascending, or `p`
/// outside `[0, 1]`.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a non-negative position below the length"
)]
pub fn rounded_rank(sorted: &[f64], p: f64) -> Option<f64> {
    let n = sorted.len();
    if n == 0 || !(0.0..=1.0).contains(&p) || !crate::quantile::sorted_without_nan(sorted) {
        return None;
    }
    let index = (p * (n - 1) as f64).round() as usize;
    Some(sorted[index.min(n - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::normal::{cdf, quantile};

    #[test]
    fn the_approximations_stay_within_their_published_errors() {
        for i in 1..1000 {
            let p = f64::from(i) / 1000.0;
            let exact = quantile(p);
            assert!((quantile_acklam(p) - exact).abs() <= 1.2e-9 * exact.abs().max(1e-300));
            let x = f64::from(i - 500) / 80.0;
            assert!((cdf_as26(x) - cdf(x)).abs() < 1.5e-7);
        }
    }

    #[test]
    fn edges() {
        assert!(quantile_acklam(1.1).is_nan());
        assert_eq!(quantile_acklam(0.0), f64::NEG_INFINITY);
        assert_eq!(quantile_acklam(1.0), f64::INFINITY);
    }

    #[test]
    fn rounded_rank_rounds_half_away_from_zero() {
        let sorted = [1.0, 2.0, 3.0];
        assert_eq!(rounded_rank(&sorted, 0.25), Some(2.0));
        assert_eq!(rounded_rank(&sorted, 0.24), Some(1.0));
        assert_eq!(rounded_rank(&sorted, 1.0), Some(3.0));
        assert_eq!(rounded_rank(&[], 0.5), None);
        assert_eq!(rounded_rank(&[1.0, f64::NAN], 0.5), None);
        assert_eq!(rounded_rank(&sorted, 1.5), None);
    }
}
