//! The gamma distribution with scale 1: the log-gamma function, its first two
//! derivatives (digamma and trigamma), the regularised incomplete gamma
//! functions, which are its distribution and survival functions, and its
//! quantile.
//!
//! # Algorithm
//!
//! [`ln_gamma`] is the Lanczos approximation with `g = 7` and nine
//! coefficients \[1\]:
//!
//! ```text
//! Γ(x + 1) = sqrt(2π) (x + g + 1/2)^(x + 1/2) exp(-(x + g + 1/2)) A_g(x),
//! A_g(x)   = c_0 + c_1 / (x + 1) + ... + c_8 / (x + 8),
//! ```
//!
//! evaluated on the log scale, with the reflection
//! `ln Γ(x) = ln(π / sin(π x)) - ln Γ(1 - x)` below `x = 1/2`. Its relative
//! error is about `1e-15` for positive arguments.
//!
//! [`digamma`] `ψ(x) = d ln Γ(x) / dx` and [`trigamma`] `ψ'(x)` use the upward
//! recurrence
//!
//! ```text
//! ψ(x) = ψ(x + 1) - 1/x,        ψ'(x) = ψ'(x + 1) + 1/x²
//! ```
//!
//! until `x >= 10`, and then the asymptotic series \[3\]
//!
//! ```text
//! ψ(x)  = ln x - 1/(2x) - Σ_{k>=1} B_2k / (2k x^2k)
//! ψ'(x) = 1/x + 1/(2x²) + Σ_{k>=1} B_2k / x^(2k+1)
//! ```
//!
//! with the Bernoulli numbers `B_2 ... B_14`. The first omitted term is below
//! `1e-17` of the result at `x = 10`. For `x < 0` the reflection
//! `ψ(1 - x) - ψ(x) = π cot(π x)` and
//! `ψ'(1 - x) + ψ'(x) = π² / sin²(π x)` is used. Both are NaN at `x = NaN`,
//! digamma is NaN and trigamma infinite at zero and the negative integers.
//!
//! [`cdf`] `P(a, x)` and [`sf`] `Q(a, x)`, at `x` for the shape `a`, are the
//! regularised incomplete gamma functions,
//!
//! ```text
//! P(a, x) = γ(a, x) / Γ(a),    Q(a, x) = 1 - P(a, x) = Γ(a, x) / Γ(a).
//! ```
//!
//! For `x < a + 1` the series
//!
//! ```text
//! P(a, x) = x^a e^-x / Γ(a + 1) * sum_{n >= 0} x^n / ((a + 1) ... (a + n))
//! ```
//!
//! converges quickly, and `Q = 1 - P`. Otherwise `Q` comes from its continued
//! fraction, evaluated by the modified Lentz method \[2\], and `P = 1 - Q`.
//! Each function computes the side it converges on directly, so a small tail
//! keeps its relative precision.
//!
//! [`quantile`] inverts [`cdf`]. It solves `ln F(x) = ln p` for the tail that is smaller,
//! so a quantile in the far tail keeps its relative precision. The start is
//! the Wilson-Hilferty cube-root approximation \[4\]. Each step is a Newton
//! step, and a step that leaves the bracket found so far is replaced by a
//! bisection, so the iteration always ends. It stops when a step is within
//! four units in the last place of the value.
//!
//! The chi-square distribution of [`super::chi_square`] is the gamma
//! distribution with shape `k/2` at `x/2`.
//!
//! # Agreement with R
//!
//! The tests compare [`ln_gamma`] with R's `lgamma()`, [`digamma`] and
//! [`trigamma`] with R's `digamma()` and `trigamma()`, the distribution
//! functions with `pgamma()`, both tails, to `1e-13` relative, and the
//! quantile with `qgamma()`.
//!
//! # References
//!
//! 1. Lanczos C. A precision approximation of the gamma function. Journal of
//!    the Society for Industrial and Applied Mathematics, Series B: Numerical
//!    Analysis 1964;1:86-96.
//! 2. Press WH, Teukolsky SA, Vetterling WT, Flannery BP. Numerical Recipes:
//!    The Art of Scientific Computing. 3rd ed. Cambridge: Cambridge University
//!    Press; 2007. Section 6.2.
//! 3. Abramowitz M, Stegun IA. Handbook of Mathematical Functions. New York:
//!    Dover; 1972. Sections 6.3 and 6.4.
//! 4. Wilson EB, Hilferty MM. The distribution of chi-square. Proc Natl Acad
//!    Sci 1931;17:684-688.

use std::f64::consts::PI;

const LANCZOS_G: f64 = 7.0;
#[expect(
    clippy::excessive_precision,
    reason = "the coefficients are written as published"
)]
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_571_6e-6,
    1.505_632_735_149_311_6e-7,
];

/// `ln |Γ(x)|`, infinite at zero and the negative integers.
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.5 {
        return (PI / (PI * x).sin()).abs().ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut sum = LANCZOS[0];
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    let t = x + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

const ASYMPTOTIC_FROM: f64 = 10.0;

/// The digamma function `ψ(x) = d ln Γ(x) / dx`. NaN at zero and the negative
/// integers.
#[must_use]
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() && x < 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::NAN;
    }
    if x < 0.0 {
        return digamma(1.0 - x) - PI / (PI * x).tan();
    }
    let mut x = x;
    let mut shift = 0.0;
    while x < ASYMPTOTIC_FROM {
        shift += 1.0 / x;
        x += 1.0;
    }
    let r = 1.0 / (x * x);
    // B_2k / (2k) for k = 1 ... 7, by Horner's rule in 1/x².
    let series = r
        * (1.0 / 12.0
            + r * (-1.0 / 120.0
                + r * (1.0 / 252.0
                    + r * (-1.0 / 240.0 + r * (1.0 / 132.0 + r * (-691.0 / 32760.0 + r / 12.0))))));
    x.ln() - 0.5 / x - series - shift
}

/// The trigamma function `ψ'(x)`. Infinite at zero and the negative integers.
#[must_use]
pub fn trigamma(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() && x < 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 && x.fract() == 0.0 {
        return f64::INFINITY;
    }
    if x < 0.0 {
        let s = (PI * x).sin();
        return PI * PI / (s * s) - trigamma(1.0 - x);
    }
    let mut x = x;
    let mut shift = 0.0;
    while x < ASYMPTOTIC_FROM {
        shift += 1.0 / (x * x);
        x += 1.0;
    }
    let r = 1.0 / (x * x);
    // B_2k for k = 1 ... 7 over x^(2k+1), by Horner's rule in 1/x².
    let series = r
        * (1.0 / 6.0
            + r * (-1.0 / 30.0
                + r * (1.0 / 42.0
                    + r * (-1.0 / 30.0
                        + r * (5.0 / 66.0 + r * (-691.0 / 2730.0 + r * 7.0 / 6.0))))));
    1.0 / x + 0.5 * r + series / x + shift
}

/// `P(X <= x)` for `X` gamma with shape `shape` and scale 1, R's
/// `pgamma(x, shape)`: the regularised lower incomplete gamma function
/// `P(shape, x)`. NaN for `shape` not positive or `x` negative or NaN.
#[must_use]
pub fn cdf(x: f64, shape: f64) -> f64 {
    lower_regularised(shape, x)
}

/// `P(X > x)`, R's `pgamma(x, shape, lower.tail = FALSE)`: the regularised
/// upper incomplete gamma function `Q(shape, x)`, computed directly so that a
/// small tail keeps its relative precision. NaN as for [`cdf`].
#[must_use]
pub fn sf(x: f64, shape: f64) -> f64 {
    upper_regularised(shape, x)
}

/// `P(a, x)`, the regularised lower incomplete gamma function, for `a > 0`
/// and `x >= 0`; NaN otherwise.
pub(crate) fn lower_regularised(a: f64, x: f64) -> f64 {
    unimplemented!("P({a}, {x})")
}

/// `Q(a, x) = 1 - P(a, x)`, the regularised upper incomplete gamma function,
/// for `a > 0` and `x >= 0`; NaN otherwise.
pub(crate) fn upper_regularised(a: f64, x: f64) -> f64 {
    unimplemented!("Q({a}, {x})")
}

/// The `p` quantile of the gamma distribution with shape `shape` and scale 1,
/// R's `qgamma(p, shape)`. NaN when `shape` is not positive and finite or `p`
/// is outside `[0, 1]`. The quantile of `p = 0` is 0 and of `p = 1` is
/// infinity.
#[must_use]
pub fn quantile(p: f64, shape: f64) -> f64 {
    quantile_or_none(p, shape).unwrap_or(f64::NAN)
}

/// [`quantile`], `None` where it is NaN.
pub(crate) fn quantile_or_none(p: f64, shape: f64) -> Option<f64> {
    unimplemented!("qgamma({p}, {shape})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert!((ln_gamma(1.0)).abs() < 1e-15);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-14);
        assert!((ln_gamma(0.5) - PI.sqrt().ln()).abs() < 1e-15);
        assert_eq!(ln_gamma(0.0).to_bits(), f64::INFINITY.to_bits());
        assert_eq!(ln_gamma(-2.0).to_bits(), f64::INFINITY.to_bits());
        // P(1, x) = 1 - exp(-x).
        assert!((lower_regularised(1.0, 2.0) - (1.0 - (-2.0f64).exp())).abs() < 1e-15);
        assert!((upper_regularised(1.0, 50.0) / (-50.0f64).exp() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn digamma_and_trigamma_known_values() {
        let euler = 0.577_215_664_901_532_9;
        assert!((digamma(1.0) + euler).abs() < 1e-15);
        assert!((digamma(0.5) + euler + 2.0 * 2f64.ln()).abs() < 1e-14);
        assert!((trigamma(1.0) - PI * PI / 6.0).abs() < 1e-14);
        assert!((trigamma(0.5) - PI * PI / 2.0).abs() < 1e-14);
        assert!(digamma(0.0).is_nan() && digamma(-3.0).is_nan());
        assert!(trigamma(0.0).is_infinite() && trigamma(-3.0).is_infinite());
        // The recurrence psi(x + 1) = psi(x) + 1/x holds across the switch to
        // the series at 10, and for negative arguments.
        for x in [0.3, 1.7, 8.9, 9.999, -0.4, -2.5] {
            assert!(
                (digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-13,
                "{x}"
            );
            assert!(
                (trigamma(x) - trigamma(x + 1.0) - 1.0 / (x * x)).abs()
                    < 1e-12 * trigamma(x).abs().max(1.0),
                "{x}"
            );
        }
    }

    #[test]
    fn edges() {
        assert!(lower_regularised(0.0, 1.0).is_nan());
        assert!(upper_regularised(1.0, -1.0).is_nan());
        assert_eq!(
            cdf(1.0, 2.0).to_bits(),
            lower_regularised(2.0, 1.0).to_bits()
        );
        assert_eq!(
            sf(1.0, 2.0).to_bits(),
            upper_regularised(2.0, 1.0).to_bits()
        );
        assert!(quantile(0.5, 0.0).is_nan());
        assert!(quantile(-0.5, 1.0).is_nan());
        assert_eq!(quantile(0.0, 1.0), 0.0);
        assert_eq!(quantile(1.0, 1.0), f64::INFINITY);
    }

    #[test]
    fn exponential_quantile_is_closed_form() {
        // Shape 1 is the exponential distribution: q = -ln(1 - p).
        for p in [1e-10, 0.01, 0.3, 0.5, 0.9, 0.999] {
            let got = quantile(p, 1.0);
            let want = -(-p).ln_1p();
            assert!((got - want).abs() <= 1e-12 * want, "{p}: {got} {want}");
        }
    }

    #[test]
    fn quantile_inverts_the_cdf() {
        for shape in [0.3, 1.0, 2.5, 10.0, 80.0, 1000.0] {
            for p in [1e-8, 0.025, 0.5, 0.975, 1.0 - 1e-8] {
                let x = quantile(p, shape);
                let back = lower_regularised(shape, x);
                assert!((back - p).abs() <= 1e-10 * p.min(1.0 - p), "{shape} {p}");
            }
        }
    }
}
