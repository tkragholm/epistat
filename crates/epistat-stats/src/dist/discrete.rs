//! Probabilities of the Poisson and binomial distributions, behind
//! [`super::poisson`] and [`super::binomial`].
//!
//! # Algorithm
//!
//! The point probabilities follow Loader \[1\], as R's `dpois` and `dbinom`
//! do. With `stirlerr(n) = ln n! - ln(sqrt(2 pi n) (n / e)^n)` and the
//! deviance term `bd0(x, np) = x ln(x / np) + np - x`,
//!
//! ```text
//! dpois(x, m)       = exp(-stirlerr(x) - bd0(x, m)) / sqrt(2 pi x)
//! dbinom(x, n, p)   = exp(stirlerr(n) - stirlerr(x) - stirlerr(n - x)
//!                         - bd0(x, n p) - bd0(n - x, n q)) sqrt(n / (2 pi x (n - x)))
//! ```
//!
//! which avoids the cancellation of `x ln m - m - ln x!` for large `x`.
//! `stirlerr` is the Stirling correction `ln Γ*(x)` and `bd0(x, m)` is
//! `x (λ - 1 - ln λ)` at `λ = m / x`, both from `dist/stirling.rs`, which
//! the incomplete gamma function shares. The tails are regularised gamma and
//! beta functions: `P(X <= x) = Q(x + 1, m)` for the Poisson distribution and
//! `I_{1 - p}(n - x, x + 1)` for the binomial.
//!
//! # Accuracy
//!
//! `stirlerr` from 10 on is the Stirling series to the term in `x^-15`,
//! whose first omitted term is below `2e-18`, and below 10 the difference
//! `ln Γ(x) - (x - 1/2) ln x + x - ln(2π)/2`, with an absolute error of about
//! `1e-14` at most. `bd0` is formed without cancellation near `x = m`. The
//! point probabilities are then accurate to about `1e-14` relative, the tails
//! to the accuracy of the incomplete gamma and beta functions.
//!
//! # References
//!
//! 1. Loader C. Fast and accurate computation of binomial probabilities.
//!    2000. <https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf>

use super::beta::regularised as regularised_beta;
use super::gamma::{lower_regularised, upper_regularised};
use super::stirling::{bd0, ln_gamma_star as stirlerr};

const TWO_PI: f64 = std::f64::consts::TAU;

/// The count `x` stands for, R's rule for a point probability: `x` rounded
/// when it is within `1e-7 max(1, |x|)` of a non-negative integer, `None`
/// otherwise.
pub(crate) fn support_point(x: f64) -> Option<f64> {
    let k = x.round();
    ((x - k).abs() <= 1e-7 * x.abs().max(1.0) && k >= 0.0).then_some(k)
}

/// The count up to `x`, R's rule for a distribution function:
/// `floor(x + 1e-7)`.
pub(crate) fn floor_count(x: f64) -> f64 {
    (x + 1e-7).floor()
}

/// `P(X = x)` for `X` Poisson with mean `m >= 0`.
pub(crate) fn dpois(x: f64, m: f64) -> f64 {
    if m == 0.0 {
        return f64::from(x == 0.0);
    }
    if x == 0.0 {
        return (-m).exp();
    }
    (-stirlerr(x) - bd0(x, m)).exp() / (TWO_PI * x).sqrt()
}

/// `ln P(X = x)` for `X` Poisson with mean `m >= 0`, from the terms of
/// [`dpois`] without the exponential, so it is finite where `dpois`
/// underflows.
pub(crate) fn ln_dpois(x: f64, m: f64) -> f64 {
    if m == 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if x == 0.0 {
        return -m;
    }
    -stirlerr(x) - bd0(x, m) - 0.5 * (TWO_PI * x).ln()
}

/// `P(X <= x)` for `X` Poisson with mean `m > 0`, and `x` an integer.
pub(crate) fn ppois_lower(x: f64, m: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        upper_regularised(x + 1.0, m)
    }
}

/// `P(X > x)` for `X` Poisson with mean `m > 0`, and `x` an integer.
pub(crate) fn ppois_upper(x: f64, m: f64) -> f64 {
    if x < 0.0 {
        1.0
    } else {
        lower_regularised(x + 1.0, m)
    }
}

/// `P(X = x)` for `X` binomial with `n` trials and probability `p`.
#[expect(
    clippy::float_cmp,
    reason = "the end points of the support are integers"
)]
pub(crate) fn dbinom(x: f64, n: f64, p: f64) -> f64 {
    let q = 1.0 - p;
    if p == 0.0 {
        return f64::from(x == 0.0);
    }
    if q == 0.0 {
        return f64::from(x == n);
    }
    if x == 0.0 {
        return (n * q.ln()).exp();
    }
    if x == n {
        return (n * p.ln()).exp();
    }
    let lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
    (lc).exp() * (n / (TWO_PI * x * (n - x))).sqrt()
}

/// `ln P(X = x)` for `X` binomial with `n` trials and probability `p`, from
/// the terms of [`dbinom`] without the exponential. The end points use
/// `ln(1 - p)` from `p` directly.
#[expect(
    clippy::float_cmp,
    reason = "the end points of the support are integers"
)]
pub(crate) fn ln_dbinom(x: f64, n: f64, p: f64) -> f64 {
    let q = 1.0 - p;
    if p == 0.0 {
        return if x == 0.0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if q == 0.0 {
        return if x == n { 0.0 } else { f64::NEG_INFINITY };
    }
    if x == 0.0 {
        return n * (-p).ln_1p();
    }
    if x == n {
        return n * p.ln();
    }
    let lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
    lc + 0.5 * (n / (TWO_PI * x * (n - x))).ln()
}

/// The relative amount by which a lower-tail probability is lowered before
/// a discrete quantile compares the distribution function with it, so that
/// a `p` computed as `P(X <= k)` with a rounding error of a few units in the
/// last place still gives `k`. R's `qpois` and `qbinom` keep such a `p` at
/// `k` as well, by a margin between 8 and 16 eps measured from outside.
pub(crate) const LOWER_FUZZ: f64 = 12.0 * f64::EPSILON;

/// The same for an upper-tail probability, which is raised; R's margin
/// there measures between 4 and 8 eps.
pub(crate) const UPPER_FUZZ: f64 = 6.0 * f64::EPSILON;

/// The smallest integer `k` in `[0, top]` with `reached(k)`, for a predicate
/// that is false up to some count and true from it on, searched from
/// `start`: steps that double in length find a bracket on the side where the
/// predicate changes, and bisection on the integers closes it. `top` is
/// returned when the predicate does not hold below it.
pub(crate) fn smallest_count(start: f64, top: f64, reached: impl Fn(f64) -> bool) -> f64 {
    let first = if start.is_nan() {
        0.0
    } else {
        start.floor().clamp(0.0, top)
    };
    let (mut lo, mut hi);
    if reached(first) {
        hi = first;
        let mut step = 1.0;
        loop {
            if hi == 0.0 {
                return 0.0;
            }
            let below = (hi - step).max(0.0);
            if !reached(below) {
                lo = below;
                break;
            }
            hi = below;
            step *= 2.0;
        }
    } else {
        lo = first;
        let mut step = 1.0;
        loop {
            if lo >= top {
                return top;
            }
            let above = (lo + step).min(top);
            if reached(above) {
                hi = above;
                break;
            }
            lo = above;
            step *= 2.0;
        }
    }
    while hi - lo > 1.0 {
        let mid = (lo + (hi - lo) / 2.0).floor();
        if reached(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// The start of a discrete quantile search: the Cornish-Fisher expansion
/// `mean + sd (z + skewness (z^2 - 1) / 6)` at the normal quantile `z`.
pub(crate) fn cornish_fisher(mean: f64, sd: f64, skewness: f64, z: f64) -> f64 {
    mean + sd * (z + skewness * (z * z - 1.0) / 6.0)
}

/// `P(X <= x)` for `X` binomial with `n` trials and probability `p` in
/// `(0, 1)`, and `x` an integer.
pub(crate) fn pbinom_lower(x: f64, n: f64, p: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x >= n {
        1.0
    } else {
        regularised_beta(n - x, x + 1.0, 1.0 - p, p)
    }
}

/// `P(X > x)` for `X` binomial with `n` trials and probability `p` in
/// `(0, 1)`, and `x` an integer.
pub(crate) fn pbinom_upper(x: f64, n: f64, p: f64) -> f64 {
    if x < 0.0 {
        1.0
    } else if x >= n {
        0.0
    } else {
        regularised_beta(x + 1.0, n - x, p, 1.0 - p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_cases_are_closed_form() {
        assert!((dpois(3.0, 2.0) - 8.0 * (-2.0_f64).exp() / 6.0).abs() < 1e-15);
        assert!((dbinom(2.0, 5.0, 0.3) - 10.0 * 0.09 * 0.343).abs() < 1e-15);
        assert!((ppois_lower(2.0, 2.0) - 5.0 * (-2.0_f64).exp()).abs() < 1e-14);
        let total: f64 = (0..=5).map(|i| dbinom(f64::from(i), 5.0, 0.3)).sum();
        assert!((total - 1.0).abs() < 1e-14);
        assert!((pbinom_lower(2.0, 5.0, 0.3) + pbinom_upper(2.0, 5.0, 0.3) - 1.0).abs() < 1e-14);
    }
}
