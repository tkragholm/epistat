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
//! until `x >= 10`, and then the asymptotic series \[2\]
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
//! They follow Gil, Segura and Temme \[3\]. One of the two, the primary, is
//! computed directly and the other is one minus it: `P` where
//! `a >= α(x)`, `Q` otherwise, with `α(x) = x` for `x >= 1/2` and
//! `α(x) = ln(1/2) / ln(x/2)` below. The primary is at most about `0.7`, so
//! the subtraction costs at most a factor of about 2 in the other tail. With
//! `λ = x / a`, `η² / 2 = λ - 1 - ln λ` (the sign of `η` that of `x - a`) and
//! the prefactor `D(a, x) = x^a e^-x / Γ(a + 1)`, there are four regions:
//!
//! 1. For `a >= 12` and `|η| <= 1`, the uniform asymptotic expansion of
//!    Temme \[4\] (DLMF 8.12.3, 8.12.4):
//!
//!    ```text
//!    Q(a, x) = erfc(η sqrt(a/2)) / 2 + R,   P(a, x) = erfc(-η sqrt(a/2)) / 2 - R,
//!    R = exp(-a η²/2) / sqrt(2πa) S_a(η),
//!    S_a(η) = a / (a + β_1) Σ_{n=0}^{25} β_n η^n,
//!    β_n = (n + 2) β_{n+2} / a + d_{n+1},   β_26 = β_27 = 0,
//!    ```
//!
//!    the expansion of `S_a` in powers of `η` of \[3, eqs. 2.33-2.40\], with
//!    `d_n` the coefficients of `η / (λ - 1) = Σ d_n η^n`, generated in exact
//!    rational arithmetic by series reversion
//!    (`tests/data/gamma_coefficients.py`). `erfc(z) / 2` is the normal tail
//!    `Φ(-z sqrt 2)` of [`super::normal`]. `P` is primary for `η <= 0`.
//! 2. For `a >= α(x)` elsewhere, the series for `P` (DLMF 8.11.4),
//!
//!    ```text
//!    P(a, x) = D(a, x) Σ_{n>=0} x^n / ((a + 1) ... (a + n)),
//!    ```
//!
//!    whose terms are positive and fall by `x / (a + n) < 1`; it stops when
//!    a term is below `eps / 2` of the sum.
//! 3. For `a < α(x)` and `x <= 1.5`, the series for `Q` at small shapes of
//!    Gautschi \[6\] in the form of \[3, section 2.3\]:
//!
//!    ```text
//!    Q(a, x) = -(1/Γ(1 + a) - 1) - expm1(a ln x) / Γ(1 + a)
//!              - x^a / Γ(1 + a) a Σ_{n>=1} (-x)^n / ((a + n) n!),
//!    ```
//!
//!    with `1/Γ(1 + a) - 1` from its Taylor series (DLMF 5.7.1, the
//!    coefficients from mpmath). `Q` is never formed as `1 - P` here, which
//!    keeps it positive and accurate down to subnormal shapes, where it is
//!    `a E1(x)` to first order.
//! 4. For `a < α(x)` and `x > 1.5`, the continued fraction for `Q` in the
//!    even contraction of DLMF 8.9.2 \[3, eqs. 2.25, 2.26\],
//!
//!    ```text
//!    Q(a, x) = D(a, x) a / (b_0 + a_1 / (b_1 + a_2 / (b_2 + ...))),
//!    b_k = x + 2k + 1 - a,   a_k = -k (k - a),
//!    ```
//!
//!    evaluated by the modified Lentz method \[7, 8\], which stops when a
//!    step changes the value by at most eps. It terminates for integer `a`.
//!
//! The prefactor is not formed as `exp(a ln x - x - ln Γ(a + 1))` for large
//! shapes, whose terms of size `a ln a` cancel. From `a = 10` it is
//! `exp(-a (λ - 1 - ln λ) - ln Γ*(a)) / sqrt(2πa)` \[3, eqs. 2.4-2.7\], with
//! the gap and the Stirling correction `ln Γ*(a)` of `dist/stirling.rs`,
//! which are also Loader's `bd0` and `stirlerr` \[11\] behind the Poisson
//! and binomial point probabilities. Below
//! `a = 10` it is the product `x^a e^-x / Γ(a + 1)`, with
//! `Γ(a + 1) = (f + 1) ... (f + n) / Γ(1 + f)` for `a = n + f` and
//! `1 / Γ(1 + f)` from its Taylor series (DLMF 5.7.1), a few roundings in
//! all, or its exponent where the product leaves the normal range. A prefactor near underflow is multiplied on the log scale, so
//! that a subnormal result keeps its gradual underflow.
//!
//! The term limits come from \[3, Tables 2.1, 2.2\] and from a scan of the
//! regions: at most 39 terms of the series for `P` and 69 of the fraction
//! were needed, against limits of 200 and 2,000. The series for `Q` needs
//! at most 22 terms. A function that reaches its limit returns NaN.
//!
//! At the edges: `P(a, 0) = 0`, `P(a, inf) = 1`, both before any logarithm.
//! An infinite shape gives `P = 0` and `Q = 1` at finite `x` (the limit, as
//! in R) and NaN at `x = inf`.
//!
//! [`quantile`] inverts [`cdf`]. It solves `ln T(x) = ln t` for the smaller
//! tail `T` (the upper tail, as `1 - p`, when `p > 1/2`), so a quantile in a
//! far tail keeps its relative precision. The start is the first terms of
//! the inversion of the series at small `x`, `x = r + r² / (a + 1)` with
//! `r = (p Γ(1 + a))^(1/a)` \[3, section 3\], where that is small, the
//! leading term of the upper tail, `Q ≈ x^(a-1) e^-x / Γ(a)`, for small upper
//! tails at shapes below 1, and the Wilson-Hilferty approximation \[9\]
//! otherwise. Temme's asymptotic inversion in `η` \[5\] would give a closer
//! start at large shapes; it is not used, since Newton from these starts
//! needs only a few steps there. Each step is a Newton step, with the density
//! `x^(a-1) e^-x / Γ(a) = D(a, x) a / x` from the same prefactor. A step that
//! leaves the bracket of the values seen, or is longer than half the step
//! before the last, is replaced by a bisection (doubling or halving while
//! the bracket is open on one side, geometric while its ends differ by more
//! than a factor 2), so the iteration always ends. It stops when a step is
//! within four units in the last place of the value, or when no double lies
//! inside the bracket.
//!
//! The chi-square distribution of [`super::chi_square`] is the gamma
//! distribution with shape `k/2` at `x/2`.
//!
//! # Accuracy
//!
//! Against mpmath at 60 digits (`tests/gamma_mpmath.rs`), the relative error
//! of the smaller tail is below `4e-14` for `a < 12` and `2e-13` for shapes
//! to `10^7`. The largest errors are in far tails, and are a few eps times
//! `a (λ - 1 - ln λ) + a |λ - 1|`, the exponent of the prefactor and the
//! rounding of `λ` within it, which is the function's own conditioning; the
//! tests hold every case to `8 eps` times that sum.
//!
//! # Agreement with R
//!
//! The tests compare [`ln_gamma`] with R's `lgamma()`, [`digamma`] and
//! [`trigamma`] with R's `digamma()` and `trigamma()`, the distribution
//! functions with `pchisq()`, both tails, to `1e-13` relative, and the
//! quantile with `qgamma()`.
//!
//! # References
//!
//! 1. Lanczos C. A precision approximation of the gamma function. Journal of
//!    the Society for Industrial and Applied Mathematics, Series B: Numerical
//!    Analysis 1964;1:86-96.
//! 2. Abramowitz M, Stegun IA. Handbook of Mathematical Functions. New York:
//!    Dover; 1972. Sections 6.3 and 6.4.
//! 3. Gil A, Segura J, Temme NM. Efficient and accurate algorithms for the
//!    computation and inversion of the incomplete gamma function ratios.
//!    SIAM J Sci Comput 2012;34(6):A2965-81.
//! 4. Temme NM. The asymptotic expansion of the incomplete gamma functions.
//!    SIAM J Math Anal 1979;10(4):757-66.
//! 5. Temme NM. Asymptotic inversion of incomplete gamma functions. Math
//!    Comput 1992;58(198):755-64.
//! 6. Gautschi W. A computational procedure for incomplete gamma functions.
//!    ACM Trans Math Softw 1979;5(4):466-81.
//! 7. Lentz WJ. Generating Bessel functions in Mie scattering calculations
//!    using continued fractions. Appl Opt 1976;15(3):668-71.
//! 8. Thompson IJ, Barnett AR. Coulomb and Bessel functions of complex
//!    arguments and order. J Comput Phys 1986;64(2):490-509.
//! 9. Wilson EB, Hilferty MM. The distribution of chi-square. Proc Natl Acad
//!    Sci 1931;17:684-688.
//! 10. Paris RB. Incomplete gamma and related functions. In: Olver FWJ, Olde
//!     Daalhuis AB, Lozier DW, Schneider BI, Boisvert RF, Clark CW, et al.,
//!     editors. NIST Digital Library of Mathematical Functions, release
//!     1.2.4. Chapter 8. <https://dlmf.nist.gov/8>
//! 11. Loader C. Fast and accurate computation of binomial probabilities.
//!     2000. <https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf>
//!
//! DLMF \[10\] is the source of the equation numbers above. The method was
//! written from these papers; no other implementation was read. mpmath
//! serves as the reference in the tests only.

use std::f64::consts::{LN_2, PI};

use super::normal;
use super::stirling::{
    STIRLING_FROM, gamma_one_plus, ln_gamma_star, ratio_gap, reciprocal_gamma_minus_one,
};

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
///
/// An infinite shape gives 0 at every finite `x`, the limit as the shape
/// grows, and NaN at `x = inf`.
#[must_use]
pub fn cdf(x: f64, shape: f64) -> f64 {
    lower_regularised(shape, x)
}

/// `P(X > x)`, R's `pgamma(x, shape, lower.tail = FALSE)`: the regularised
/// upper incomplete gamma function `Q(shape, x)`, computed directly so that a
/// small tail keeps its relative precision. NaN as for [`cdf`].
///
/// An infinite shape gives 1 at every finite `x` and NaN at `x = inf`.
#[must_use]
pub fn sf(x: f64, shape: f64) -> f64 {
    upper_regularised(shape, x)
}

/// `P(a, x)`, the regularised lower incomplete gamma function, for `a > 0`
/// and `x >= 0`; NaN otherwise.
pub(crate) fn lower_regularised(a: f64, x: f64) -> f64 {
    regularised(a, x).map_or(f64::NAN, |t| t.lower)
}

/// `Q(a, x) = 1 - P(a, x)`, the regularised upper incomplete gamma function,
/// for `a > 0` and `x >= 0`; NaN otherwise.
pub(crate) fn upper_regularised(a: f64, x: f64) -> f64 {
    regularised(a, x).map_or(f64::NAN, |t| t.upper)
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
    if !(shape > 0.0 && shape.is_finite() && (0.0..=1.0).contains(&p)) {
        return None;
    }
    if p <= 0.0 {
        return Some(0.0);
    }
    if p >= 1.0 {
        return Some(f64::INFINITY);
    }
    // 1 - p is exact for p >= 1/2.
    if p > 0.5 {
        invert(1.0 - p, shape, Side::Upper)
    } else {
        invert(p, shape, Side::Lower)
    }
}

/// `P(a, x)` and `Q(a, x)`.
#[derive(Clone, Copy, Debug)]
struct Tails {
    lower: f64,
    upper: f64,
}

impl Tails {
    fn from_lower(p: f64) -> Self {
        Self {
            lower: p,
            upper: 1.0 - p,
        }
    }

    fn from_upper(q: f64) -> Self {
        Self {
            lower: 1.0 - q,
            upper: q,
        }
    }

    fn side(self, side: Side) -> f64 {
        match side {
            Side::Lower => self.lower,
            Side::Upper => self.upper,
        }
    }
}

/// The tail a quantile solves for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Lower,
    Upper,
}

/// The shape from which the uniform asymptotic expansion is used.
const UNIFORM_FROM: f64 = 12.0;

/// The largest `η² / 2 = λ - 1 - ln λ` of the uniform asymptotic expansion,
/// `|η| <= 1`.
const UNIFORM_GAP: f64 = 0.5;

/// The largest `x` of the series for `Q` at small shapes.
const SMALL_SHAPE_UP_TO: f64 = 1.5;

/// The number of terms `N` of the expansion of `S_a(η)`, enough for double
/// precision from `a = 12` (Gil, Segura and Temme 2012, section 2.5).
const UNIFORM_TERMS: usize = 25;

/// `d_n` for `n = 0 ... 26`, the coefficients of
/// `η / (λ - 1) = Σ d_n η^n`, from the reversion of `η² / 2 = λ - 1 - ln λ`
/// in exact rational arithmetic by `tests/data/gamma_coefficients.py`.
/// `d_0 ... d_11` are `1, -1/3, 1/12, -2/135, 1/864, 1/2835, -139/777600,
/// 1/25515, -571/261273600, -281/151559100, 163879/197522841600` and
/// `-5221/29554024500`.
#[expect(
    clippy::unreadable_literal,
    reason = "the coefficients are written as generated"
)]
const ETA_RATIO: [f64; UNIFORM_TERMS + 2] = [
    1.0,
    -0.3333333333333333,
    0.08333333333333333,
    -0.014814814814814815,
    0.0011574074074074073,
    0.0003527336860670194,
    -0.0001787551440329218,
    3.919263178522438e-05,
    -2.185448510679992e-06,
    -1.85406221071516e-06,
    8.296711340953087e-07,
    -1.7665952736826078e-07,
    6.707853543401498e-09,
    1.0261809784240309e-08,
    -4.382036018453353e-09,
    9.14769958223679e-10,
    -2.5514193994946248e-11,
    -5.830772132550426e-11,
    2.4361948020667415e-11,
    -5.0276692801141755e-12,
    1.1004392031956135e-13,
    3.371763262400985e-13,
    -1.392388722418162e-13,
    2.8534893807047445e-14,
    -5.139111834242572e-16,
    -1.9752288294349442e-15,
    8.099521156704561e-16,
];

/// The term limit of the series for `P`. Its terms are positive and fall
/// by the factor `x / (a + n) < 1` each; a scan of the region on a grid of
/// 40 points per decade in `a` and `x` needed at most 39 terms, at
/// `x = a` just below 12, and Gil, Segura and Temme (2012, Table 2.1) report
/// about 60 near the diagonal.
const LOWER_SERIES_TERMS: usize = 200;

/// The term limit of the series in `v` of the small-shape `Q`, whose terms
/// are of size `x^n / n!` with `x <= 1.5`: `1.5^21 / 22!` is below
/// `eps / 2` of the sum, so at most 22 terms are needed.
const SMALL_SHAPE_TERMS: usize = 40;

/// The term limit of the continued fraction for `Q`. The same scan needed
/// at most 69 terms, near `x = 1.5` with a small shape, where the prefactor
/// does not underflow; Gil, Segura and Temme (2012, Table 2.2) report up to
/// about 160 near the border of the uniform expansion at `x = 10^4`.
const FRACTION_TERMS: usize = 2_000;

/// The value that stands in for a zero denominator in the modified Lentz
/// method.
const LENTZ_TINY: f64 = 1e-300;

/// `ln 2^-1075`, below which a positive value rounds to zero.
const LN_UNDERFLOW: f64 = -745.2;

/// The exponent below which [`Prefactor::times`] forms its product on the
/// log scale, so that a result in the subnormal range keeps its gradual
/// underflow.
const LN_NEAR_UNDERFLOW: f64 = -700.0;

/// The `x` below which `e^-x` is a normal double.
const EXP_NORMAL_UP_TO: f64 = 700.0;

/// `sqrt(2π)`.
const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// `P(a, x)` and `Q(a, x)`, the smaller tail computed directly, or `None`
/// outside the domain or where a series or fraction did not converge within
/// its term limit.
fn regularised(a: f64, x: f64) -> Option<Tails> {
    if !(a > 0.0 && x >= 0.0) {
        return None;
    }
    if x == 0.0 {
        return Some(Tails::from_lower(0.0));
    }
    if x == f64::INFINITY {
        return (a < f64::INFINITY).then_some(Tails::from_lower(1.0));
    }
    if a == f64::INFINITY {
        return Some(Tails::from_lower(0.0));
    }
    if a >= UNIFORM_FROM {
        let gap = ratio_gap(x, a);
        if gap <= UNIFORM_GAP {
            return Some(uniform(a, x, gap));
        }
    }
    if a >= alpha(x) {
        lower_series(a, x).map(Tails::from_lower)
    } else if x <= SMALL_SHAPE_UP_TO {
        Some(Tails::from_upper(small_shape_upper(a, x)))
    } else {
        upper_fraction(a, x).map(Tails::from_upper)
    }
}

/// The border `α(x)` between the regions where `P` (shapes above it) and
/// `Q` are computed directly: `x` from `x = 1/2`, and
/// `ln(1/2) / ln(x / 2)` below.
fn alpha(x: f64) -> f64 {
    if x >= 0.5 {
        x
    } else {
        // ln(x / 2) as a difference, so that it does not underflow for
        // subnormal x.
        -LN_2 / (x.ln() - LN_2)
    }
}

/// `D(a, x) = x^a e^-x / Γ(a + 1)` as `exp(log) * scale`.
#[derive(Clone, Copy, Debug)]
struct Prefactor {
    log: f64,
    scale: f64,
}

impl Prefactor {
    /// Below [`STIRLING_FROM`] the product `x^a e^-x / Γ(a + 1)`, with
    /// `Γ(a + 1)` from [`gamma_one_plus`], where it is a normal double, and
    /// otherwise its exponent `a ln x - x - ln Γ(a + 1)`; from there `exp(-a (λ - 1 - ln λ) - ln Γ*(a)) / sqrt(2πa)`, whose
    /// exponent has no terms of size `a ln a` to cancel.
    fn new(a: f64, x: f64) -> Self {
        if a < STIRLING_FROM {
            let gamma = gamma_one_plus(a);
            if x < EXP_NORMAL_UP_TO {
                let d = x.powf(a) * (-x).exp() / gamma;
                if d.is_finite() && d >= f64::MIN_POSITIVE {
                    return Self { log: 0.0, scale: d };
                }
            }
            Self {
                log: a * x.ln() - x - gamma.ln(),
                scale: 1.0,
            }
        } else {
            Self {
                log: -(a * ratio_gap(x, a)) - ln_gamma_star(a),
                scale: 1.0 / (SQRT_2PI * a.sqrt()),
            }
        }
    }

    /// `D(a, x) * m` for `m > 0`.
    fn times(self, m: f64) -> f64 {
        if self.log > LN_NEAR_UNDERFLOW {
            self.log.exp() * self.scale * m
        } else {
            (self.log + (self.scale * m).ln()).exp()
        }
    }
}

/// The uniform asymptotic expansion (DLMF 8.12.3, 8.12.4), for `a >= 12`
/// and `|η| <= 1`, with `gap = η² / 2 = λ - 1 - ln λ`:
/// `Q = erfc(η sqrt(a/2)) / 2 + R` and `P = erfc(-η sqrt(a/2)) / 2 - R`,
/// `R = exp(-a η²/2) / sqrt(2πa) S_a(η)`, with `S_a` from the backward
/// recursion of Gil, Segura and Temme (2012, eqs. 2.33-2.40).
fn uniform(a: f64, x: f64, gap: f64) -> Tails {
    let eta = (2.0 * gap).sqrt().copysign(x - a);
    let (mut next, mut after) = (0.0, 0.0);
    let mut sum = 0.0;
    let mut beta_1 = 0.0;
    for n in (0..=UNIFORM_TERMS).rev() {
        let beta = (n as f64 + 2.0) * after / a + ETA_RATIO[n + 1];
        sum = sum * eta + beta;
        if n == 1 {
            beta_1 = beta;
        }
        after = next;
        next = beta;
    }
    let s = a / (a + beta_1) * sum;
    let r = (-(a * gap)).exp() / (SQRT_2PI * a.sqrt()) * s;
    // erfc(z) / 2 = Φ(-z sqrt 2), so erfc(η sqrt(a/2)) / 2 = Φ(-η sqrt a).
    let y = eta * a.sqrt();
    if eta <= 0.0 {
        Tails::from_lower(normal::cdf(y) - r)
    } else {
        Tails::from_upper(normal::cdf(-y) + r)
    }
}

/// The series for `P` (DLMF 8.11.4), for `a >= α(x)`:
/// `P(a, x) = D(a, x) Σ_{n>=0} x^n / ((a + 1) ... (a + n))`, stopped when a
/// term is below `eps / 2` of the sum.
fn lower_series(a: f64, x: f64) -> Option<f64> {
    let (mut term, mut sum) = (1.0, 1.0);
    for n in 1..=LOWER_SERIES_TERMS {
        term *= x / (a + n as f64);
        sum += term;
        if term <= 0.5 * f64::EPSILON * sum {
            return Some(Prefactor::new(a, x).times(sum));
        }
    }
    None
}

/// The series for `Q` at small shapes, for `a < α(x)` and `x <= 1.5` (Gil,
/// Segura and Temme 2012, section 2.3, after Gautschi 1979): `Q = u + v`
/// with
///
/// ```text
/// u = 1 - x^a / Γ(1 + a) = -(1/Γ(1 + a) - 1) - expm1(a ln x) / Γ(1 + a),
/// v = -x^a / Γ(1 + a) a Σ_{n>=1} (-x)^n / ((a + n) n!).
/// ```
fn small_shape_upper(a: f64, x: f64) -> f64 {
    let g = reciprocal_gamma_minus_one(a);
    let e = (a * x.ln()).exp_m1();
    let reciprocal = 1.0 + g;
    let u = -g - e * reciprocal;
    let (mut term, mut sum) = (1.0, 0.0);
    for n in 1..=SMALL_SHAPE_TERMS {
        let n = n as f64;
        term *= -x / n;
        let c = term / (a + n);
        sum += c;
        if c.abs() <= 0.5 * f64::EPSILON * sum.abs() {
            break;
        }
    }
    let v = -((1.0 + e) * reciprocal) * (a * sum);
    u + v
}

/// The continued fraction for `Q` (Gil, Segura and Temme 2012, eqs. 2.25
/// and 2.26, the even contraction of DLMF 8.9.2), for `x > 1.5` and
/// `a < α(x)`:
///
/// ```text
/// Q(a, x) = D(a, x) a / (b_0 + a_1 / (b_1 + a_2 / (b_2 + ...))),
/// b_k = x + 2k + 1 - a,   a_k = -k (k - a),
/// ```
///
/// by the modified Lentz method. The fraction is at least 1 here (`Q` is
/// below `a D(a, x)` for `x > a`, from `Γ(a, x) <= x^a e^-x / (x - a + 1)`
/// for `a > 1` and `<= x^(a - 1) e^-x` for `a <= 1`), so where `a D(a, x)`
/// is below the underflow threshold `Q` is 0 without the fraction.
fn upper_fraction(a: f64, x: f64) -> Option<f64> {
    let prefactor = Prefactor::new(a, x);
    if prefactor.log + (prefactor.scale * a).ln() < LN_UNDERFLOW {
        return Some(0.0);
    }
    let b_0 = x + 1.0 - a;
    let mut f = if b_0 == 0.0 { LENTZ_TINY } else { b_0 };
    let (mut c, mut d) = (f, 0.0);
    for k in 1..=FRACTION_TERMS {
        let k = k as f64;
        let a_k = -k * (k - a);
        let b_k = b_0 + 2.0 * k;
        d = b_k + a_k * d;
        if d == 0.0 {
            d = LENTZ_TINY;
        }
        c = b_k + a_k / c;
        if c == 0.0 {
            c = LENTZ_TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() <= f64::EPSILON {
            return Some(prefactor.times(a / f));
        }
    }
    None
}

/// The step limit of the quantile's iteration. A step is a Newton step no
/// longer than half the step before the last, or a bisection of the
/// bracket: a doubling or halving while it is open on one side, which
/// crosses the 2,098 binades of positive doubles at most once, then
/// geometric bisections while its ends differ by more than a factor 2,
/// at most 12, and arithmetic ones, at most 53. Newton steps halve every
/// second step, so they can add at most twice the 2,151 binades a step can
/// fall through before it is below an ulp.
const QUANTILE_STEPS: usize = 8_192;

/// The quantile with tail probability `t` on `side`, `0 < t <= 1/2`: Newton
/// on `ln T(x) = ln t`, `T` the tail, safeguarded by the bracket of the
/// values seen.
fn invert(t: f64, a: f64, side: Side) -> Option<f64> {
    let mut x = start(t, a, side);
    if x == 0.0 {
        x = f64::from_bits(1);
    }
    let (mut lo, mut hi) = (0.0_f64, f64::INFINITY);
    // The distance of the tail from t at each end of the bracket: t at
    // x = 0 for the lower tail, where it is 0, and 1 - t there for the upper.
    let (mut lo_miss, mut hi_miss) = match side {
        Side::Lower => (t, 1.0 - t),
        Side::Upper => (1.0 - t, t),
    };
    let (mut last, mut before) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..QUANTILE_STEPS {
        let value = regularised(a, x)?.side(side);
        if (value - t).abs() == 0.0 {
            return Some(x);
        }
        // P rises and Q falls with x.
        let below = (value < t) == (side == Side::Lower);
        if below {
            (lo, lo_miss) = (x, (value - t).abs());
        } else {
            (hi, hi_miss) = (x, (value - t).abs());
        }
        // d ln T / dx = ±f(x) / T(x), with the density
        // f(x) = x^(a-1) e^-x / Γ(a) = D(a, x) a / x.
        let density = Prefactor::new(a, x).times(a / x);
        let gap = if (value - t).abs() <= 0.5 * t {
            ((value - t) / t).ln_1p()
        } else {
            value.ln() - t.ln()
        };
        let slope = match side {
            Side::Lower => density / value,
            Side::Upper => -density / value,
        };
        let step = -gap / slope;
        let newton = x + step;
        let accepted =
            newton.is_finite() && newton > lo && newton < hi && step.abs() <= 0.5 * before;
        let next = if accepted {
            if step.abs() <= 4.0 * f64::EPSILON * x {
                return Some(newton);
            }
            (before, last) = (last, step.abs());
            newton
        } else {
            let next = bisect(lo, hi);
            (before, last) = (last, (next - x).abs());
            next
        };
        if next <= lo || next >= hi {
            // No double lies strictly inside the bracket: the end whose tail
            // is nearer t.
            return Some(if lo_miss <= hi_miss { lo } else { hi });
        }
        x = next;
    }
    None
}

/// The next point inside `(lo, hi)`: a doubling or halving while the bracket
/// is open on one side, its geometric mean while the ends differ by more
/// than a factor 2, and its midpoint after.
fn bisect(lo: f64, hi: f64) -> f64 {
    if hi == f64::INFINITY {
        2.0 * lo
    } else if lo == 0.0 {
        0.5 * hi
    } else if hi > 2.0 * lo {
        lo.sqrt() * hi.sqrt()
    } else {
        lo + 0.5 * (hi - lo)
    }
}

/// A starting value for [`invert`]. The lower tail starts from the first
/// terms of the inversion of the series for `P` at small `x` (Gil, Segura and
/// Temme 2012, section 3), `x = r + r² / (a + 1)` with
/// `r = (t Γ(1 + a))^(1/a)`, where `r` is small, and the upper tail from
/// `Q ≈ x^(a-1) e^-x / Γ(a)` at large `x`. Elsewhere the start is the
/// Wilson-Hilferty approximation (Wilson and Hilferty 1931).
fn start(t: f64, a: f64, side: Side) -> f64 {
    let z = normal::quantile(t);
    let z = match side {
        Side::Lower => z,
        Side::Upper => -z,
    };
    let cube = 1.0 - 1.0 / (9.0 * a) + z / (3.0 * a.sqrt());
    let wilson_hilferty = a * cube * cube * cube;
    match side {
        Side::Lower => {
            let r = ((t.ln() + ln_gamma(a + 1.0)) / a).exp();
            if r < 0.2 * (a + 1.0) || wilson_hilferty.is_nan() || wilson_hilferty <= 0.0 {
                r + r * r / (a + 1.0)
            } else {
                wilson_hilferty
            }
        }
        Side::Upper => {
            let base = -(t.ln() + ln_gamma(a));
            if a < 1.0 && base > 1.0 {
                let mut x = base;
                for _ in 0..3 {
                    x = base + (a - 1.0) * x.ln();
                }
                x
            } else if wilson_hilferty > 0.0 {
                wilson_hilferty
            } else {
                1.0
            }
        }
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
