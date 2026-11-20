//! The Student t distribution and the regularised incomplete beta function.
//!
//! The tail is a regularised incomplete beta function. With `t` the
//! variable, `nu` the degrees of freedom and `x = nu / (nu + t^2)`,
//!
//! ```text
//! P(T > |t|) = (1/2) I_x(nu/2, 1/2),
//! ```
//!
//! and `I_x(a, b)` is evaluated by its continued fraction with the modified
//! Lentz method \[1\] on the side of `(a + 1) / (a + b + 2)` where it
//! converges, using `I_x(a, b) = 1 - I_{1-x}(b, a)` on the other. The tail is
//! therefore computed directly and keeps its relative precision far out. The
//! prefactor is formed on the log scale from [`crate::gamma::ln_gamma`], whose
//! relative error is about `1e-15`. The quantile is found by Newton's method
//! on the tail inside a bracket.
//!
//! The functions agree with R's `pt`, `qt` and `pbeta` to about `1e-13`
//! relative, not bit for bit.
//!
//! # References
//!
//! 1. Press WH, Teukolsky SA, Vetterling WT, Flannery BP. Numerical Recipes,
//!    3rd edition. Cambridge University Press 2007, section 6.4.

use std::f64::consts::PI;

use crate::gamma::ln_gamma;

const MAX_TERMS: usize = 10_000;
const FLOOR: f64 = 1e-300;

/// `ln f(z)` of the t density with `nu` degrees of freedom.
#[must_use]
pub fn t_ln_density(z: f64, nu: f64) -> f64 {
    let constant = ln_gamma(f64::midpoint(nu, 1.0)) - ln_gamma(0.5 * nu) - 0.5 * (nu * PI).ln();
    constant - f64::midpoint(nu, 1.0) * (z * z / nu).ln_1p()
}

/// The continued fraction of the incomplete beta function, or `None` when it
/// has not converged in `MAX_TERMS` terms.
fn beta_fraction(a: f64, b: f64, x: f64) -> Option<f64> {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FLOOR {
        d = FLOOR;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_TERMS {
        let m = m as f64;
        let m2 = 2.0 * m;
        let even = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + even * d;
        if d.abs() < FLOOR {
            d = FLOOR;
        }
        c = 1.0 + even / c;
        if c.abs() < FLOOR {
            c = FLOOR;
        }
        d = 1.0 / d;
        h *= d * c;
        let odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + odd * d;
        if d.abs() < FLOOR {
            d = FLOOR;
        }
        c = 1.0 + odd / c;
        if c.abs() < FLOOR {
            c = FLOOR;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= f64::EPSILON {
            return Some(h);
        }
    }
    None
}

/// The regularised incomplete beta function `I_x(a, b)`, R's `pbeta(x, a, b)`,
/// for `x` in `[0, 1]` given with `y = 1 - x` computed without cancellation.
/// NaN when the continued fraction does not converge in 10 000 terms.
#[must_use]
pub fn regularised_beta(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if y <= 0.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * y.ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        beta_fraction(a, b, x).map_or(f64::NAN, |cf| front * cf / a)
    } else {
        let front = (b * y.ln() + a * x.ln() - ln_beta).exp();
        beta_fraction(b, a, y).map_or(f64::NAN, |cf| 1.0 - front * cf / b)
    }
}

/// `P(T > |z|)`, half the two-sided tail.
fn upper_tail(z: f64, nu: f64) -> f64 {
    if z.is_infinite() {
        return 0.0;
    }
    let q = nu + z * z;
    0.5 * regularised_beta(0.5 * nu, 0.5, nu / q, z * z / q)
}

/// The distribution function and the survival function of the t distribution
/// with `nu` degrees of freedom at `z`, R's `pt(z, nu)` and
/// `pt(z, nu, lower.tail = FALSE)`. The smaller of the two is computed
/// directly, so it keeps its relative precision far out. NaN when the
/// incomplete beta function does not converge.
#[must_use]
pub fn t_cdf_sf(z: f64, nu: f64) -> (f64, f64) {
    let tail = upper_tail(z.abs(), nu);
    if z > 0.0 {
        (1.0 - tail, tail)
    } else {
        (tail, 1.0 - tail)
    }
}

/// The quantile function of the t distribution for `p` in `(0, 1)`, R's
/// `qt(p, nu)`: the root of the tail probability by Newton's method inside a
/// bracket, which is halved whenever a step leaves it. It stops when the step
/// is below `4 eps |t|`. `None` when the search for a bracket passes `1e300`.
#[must_use]
pub fn t_quantile(p: f64, nu: f64) -> Option<f64> {
    if p.to_bits() == 0.5_f64.to_bits() {
        return Some(0.0);
    }
    let lower = p < 0.5;
    let target = if lower { p } else { 1.0 - p };
    // P(T > t) = target, t > 0.
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while upper_tail(hi, nu) > target {
        lo = hi;
        hi *= 2.0;
        if hi > 1e300 {
            return None;
        }
    }
    let mut t = f64::midpoint(lo, hi);
    for _ in 0..200 {
        let tail = upper_tail(t, nu);
        if tail > target {
            lo = t;
        } else {
            hi = t;
        }
        let density = t_ln_density(t, nu).exp();
        let newton = t + (tail - target) / density;
        let next = if newton > lo && newton < hi {
            newton
        } else {
            f64::midpoint(lo, hi)
        };
        let done = (next - t).abs() <= 4.0 * f64::EPSILON * next.abs();
        t = next;
        if done {
            break;
        }
    }
    Some(if lower { -t } else { t })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One degree of freedom is the Cauchy distribution, `1/2 + atan(z) / pi`.
    /// The error of the evaluation is a few eps in the tail it computes, and
    /// the check allows 64 eps relative to the smaller of the two tails.
    #[test]
    fn one_degree_of_freedom_is_the_cauchy_distribution() {
        for z in [-30.0, -3.0, -0.4, 0.0, 0.7, 2.0, 25.0] {
            let (cdf, sf) = t_cdf_sf(z, 1.0);
            let (want_cdf, want_sf) = (0.5 + z.atan() / PI, 0.5 - z.atan() / PI);
            let (small, want) = if z > 0.0 {
                (sf, want_sf)
            } else {
                (cdf, want_cdf)
            };
            assert!((small - want).abs() <= 64.0 * f64::EPSILON * want, "{z}");
            assert!((cdf + sf - 1.0).abs() <= 4.0 * f64::EPSILON);
        }
    }

    #[test]
    fn the_quantile_inverts_the_distribution_function() {
        for nu in [0.5, 1.0, 2.5, 4.0, 30.0, 400.0] {
            for p in [1e-10, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-10] {
                let z = t_quantile(p, nu).unwrap();
                let (cdf, sf) = t_cdf_sf(z, nu);
                let (got, want) = if p < 0.5 { (cdf, p) } else { (sf, 1.0 - p) };
                assert!((got - want).abs() <= 1e-13 * want, "{nu} {p}: {z}");
            }
        }
    }
}
