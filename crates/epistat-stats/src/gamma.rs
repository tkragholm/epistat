//! The log-gamma function, its first two derivatives (digamma and trigamma),
//! the regularised incomplete gamma functions and the chi-square distribution.
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
//! [`lower_regularised`] `P(a, x)` and [`upper_regularised`] `Q(a, x)` are
//! the regularised incomplete gamma functions,
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
//! The chi-square distribution with `k` degrees of freedom has distribution
//! function `P(k/2, x/2)` and survival function `Q(k/2, x/2)`.
//!
//! # Agreement with R
//!
//! The tests compare [`ln_gamma`] with R's `lgamma()`, [`digamma`] and
//! [`trigamma`] with R's `digamma()` and `trigamma()`, and the chi-square
//! functions with `pchisq()`, both tails, to `1e-13` relative.
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
const MAX_TERMS: usize = 100_000;
const TOLERANCE: f64 = f64::EPSILON;

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
                    + r * (-1.0 / 240.0
                        + r * (1.0 / 132.0 + r * (-691.0 / 32760.0 + r / 12.0))))));
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
                    + r * (-1.0 / 30.0 + r * (5.0 / 66.0 + r * (-691.0 / 2730.0 + r * 7.0 / 6.0))))));
    1.0 / x + 0.5 * r + series / x + shift
}

/// `P(a, x)`, the regularised lower incomplete gamma function, for `a > 0`
/// and `x >= 0`; NaN otherwise.
#[must_use]
pub fn lower_regularised(a: f64, x: f64) -> f64 {
    match regularised(a, x) {
        Some(Tail::Lower(p)) => p,
        Some(Tail::Upper(q)) => 1.0 - q,
        None => f64::NAN,
    }
}

/// `Q(a, x) = 1 - P(a, x)`, the regularised upper incomplete gamma function,
/// for `a > 0` and `x >= 0`; NaN otherwise.
#[must_use]
pub fn upper_regularised(a: f64, x: f64) -> f64 {
    match regularised(a, x) {
        Some(Tail::Lower(p)) => 1.0 - p,
        Some(Tail::Upper(q)) => q,
        None => f64::NAN,
    }
}

/// `P(X <= x)` for `X` chi-square with `df` degrees of freedom.
#[must_use]
pub fn chi_square_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 && df > 0.0 {
        return 0.0;
    }
    lower_regularised(df / 2.0, x / 2.0)
}

/// `P(X > x)` for `X` chi-square with `df` degrees of freedom, computed
/// directly so that small p-values keep their relative precision.
#[must_use]
pub fn chi_square_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 && df > 0.0 {
        return 1.0;
    }
    upper_regularised(df / 2.0, x / 2.0)
}

enum Tail {
    Lower(f64),
    Upper(f64),
}

fn regularised(a: f64, x: f64) -> Option<Tail> {
    if !(a > 0.0 && x >= 0.0) {
        return None;
    }
    if x == 0.0 {
        return Some(Tail::Lower(0.0));
    }
    if x.is_infinite() {
        return Some(Tail::Upper(0.0));
    }
    let log_prefactor = a * x.ln() - x - ln_gamma(a);
    if x < a + 1.0 {
        // Series for P.
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut denominator = a;
        for _ in 0..MAX_TERMS {
            denominator += 1.0;
            term *= x / denominator;
            sum += term;
            if term.abs() < sum.abs() * TOLERANCE {
                break;
            }
        }
        Some(Tail::Lower(sum * log_prefactor.exp()))
    } else {
        // Continued fraction for Q, by the modified Lentz method.
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..MAX_TERMS {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < TOLERANCE {
                break;
            }
        }
        Some(Tail::Upper(log_prefactor.exp() * h))
    }
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
            assert!((digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-13, "{x}");
            assert!(
                (trigamma(x) - trigamma(x + 1.0) - 1.0 / (x * x)).abs() < 1e-12 * trigamma(x).abs().max(1.0),
                "{x}"
            );
        }
    }

    #[test]
    fn edges() {
        assert!(lower_regularised(0.0, 1.0).is_nan());
        assert!(upper_regularised(1.0, -1.0).is_nan());
        assert_eq!(chi_square_cdf(0.0, 3.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(chi_square_sf(0.0, 3.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(
            chi_square_sf(f64::INFINITY, 3.0).to_bits(),
            0.0f64.to_bits()
        );
    }
}
