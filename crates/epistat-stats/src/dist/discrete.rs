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
//! `bd0` is summed as a series when `x` is within a tenth of `np`. The tails
//! are regularised gamma and beta functions: `P(X <= x) = Q(x + 1, m)` for
//! the Poisson distribution and `I_{1 - p}(n - x, x + 1)` for the binomial.
//!
//! # Accuracy
//!
//! `stirlerr` below 15 is computed from `ln_gamma` and above 15 from its
//! series to the term in `n^-9`, whose first omitted term is below `2e-16`.
//! The point probabilities then agree with R's to about `1e-14` relative, the
//! tails to the accuracy of the incomplete gamma and beta functions.
//!
//! # References
//!
//! 1. Loader C. Fast and accurate computation of binomial probabilities.
//!    2000. <https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf>

use super::beta::regularised as regularised_beta;
use super::gamma::{ln_gamma, lower_regularised, upper_regularised};

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;
const TWO_PI: f64 = std::f64::consts::TAU;

fn stirlerr(n: f64) -> f64 {
    if n <= 15.0 {
        return ln_gamma(n + 1.0) - (n + 0.5) * n.ln() + n - LN_SQRT_2PI;
    }
    let n2 = n * n;
    let s0 = 1.0 / 12.0;
    let s1 = 1.0 / 360.0;
    let s2 = 1.0 / 1260.0;
    let s3 = 1.0 / 1680.0;
    let s4 = 1.0 / 1188.0;
    (s0 - (s1 - (s2 - (s3 - s4 / n2) / n2) / n2) / n2) / n
}

#[expect(
    clippy::float_cmp,
    reason = "the series stops when a term no longer changes the sum"
)]
fn bd0(x: f64, np: f64) -> f64 {
    if (x - np).abs() < 0.1 * (x + np) {
        let v = (x - np) / (x + np);
        let mut s = (x - np) * v;
        let mut ej = 2.0 * x * v;
        let v2 = v * v;
        for j in 1..1000 {
            ej *= v2;
            let s1 = s + ej / f64::from(2 * j + 1);
            if s1 == s {
                return s1;
            }
            s = s1;
        }
        s
    } else {
        x * (x / np).ln() + np - x
    }
}

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
