//! Two quantities behind the gamma and Poisson probabilities, each formed
//! without the cancellation of its defining expression: the gap
//! `z - ln(1 + z)` and the Stirling correction `ln Γ*(a)`.
//!
//! # Algorithm
//!
//! [`log1pmx`] is `z - ln(1 + z)`, which is `η² / 2` of the uniform
//! asymptotics of the incomplete gamma function at `λ = 1 + z` (DLMF 8.12.1)
//! and the deviance term of Loader \[3\]. For `|z| < 0.3` it is the series
//!
//! ```text
//! z - ln(1 + z) = z²/2 - z³/3 + z⁴/4 - ...,
//! ```
//!
//! an alternating series whose terms decrease, so its truncation error is
//! below the first omitted term. The sum is at least `0.4 z²` there, and the
//! term in `z^31` is below `eps / 2` of it at `|z| = 0.3`, so the terms up to
//! `z^30` are kept. Elsewhere it is formed directly, where the cancellation
//! costs a factor of at most `|z| / (z - ln(1 + z))`, about 8 at the border.
//!
//! [`ratio_gap`] is `λ - 1 - ln λ` for `λ = x / a`. For `λ < 0.7` it is
//! `(λ - 1) - ln λ`, because `ln(1 + z)` of `z` near `-1` multiplies the
//! rounding error of `z` by `1 / λ`; otherwise it is [`log1pmx`] of
//! `z = (x - a) / a`, where `x - a` is exact for `x` within a factor 2 of
//! `a`. [`bd0`] is Loader's `bd0(x, m) = x ln(x / m) + m - x`, which is
//! `x (λ - 1 - ln λ)` at `λ = m / x`.
//!
//! [`ln_gamma_star`] is
//!
//! ```text
//! ln Γ*(a) = ln Γ(a) - [(a - 1/2) ln a - a + ln(2π) / 2],
//! ```
//!
//! which is Loader's `stirlerr(a)`, `ln a! - ln(sqrt(2πa) (a/e)^a)`. From
//! `a = 10` it is the Stirling series (DLMF 5.11.1)
//!
//! ```text
//! ln Γ*(a) = Σ_{k=1}^{8} B_2k / (2k (2k - 1) a^(2k - 1)),
//! ```
//!
//! whose first omitted term, `B_18 / (306 a^17)`, is below `2e-18` there.
//! Below `a = 10` it is the difference above with `ln Γ(1 + a)` from
//! [`gamma_one_plus`], `Γ(1 + a) = (f + 1) ... (f + n) / Γ(1 + f)` for
//! `a = n + f`, with `1 / Γ(1 + f)` from its Taylor series (DLMF 5.7.1). Its
//! absolute error is a few eps times the magnitudes of its terms, below
//! `5e-15`.
//!
//! # References
//!
//! 1. Paris RB. Incomplete gamma and related functions. In: Olver FWJ, et
//!    al., editors. NIST Digital Library of Mathematical Functions, release
//!    1.2.4. Chapter 8. <https://dlmf.nist.gov/8>
//! 2. Gil A, Segura J, Temme NM. Efficient and accurate algorithms for the
//!    computation and inversion of the incomplete gamma function ratios.
//!    SIAM J Sci Comput 2012;34(6):A2965-81.
//! 3. Loader C. Fast and accurate computation of binomial probabilities.
//!    2000. <https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf>

/// `ln(2π) / 2`.
pub(crate) const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_8;

/// The shape from which [`ln_gamma_star`] is the Stirling series.
pub(crate) const STIRLING_FROM: f64 = 10.0;

/// `|z|` below which [`log1pmx`] is summed as a series.
const SERIES_WITHIN: f64 = 0.3;

/// `1 / k` for `k = 2 ... 30`, the coefficients of the series of
/// [`log1pmx`] in `-z`.
const INVERSES: [f64; 29] = {
    let mut c = [0.0; 29];
    let mut i = 0;
    while i < c.len() {
        c[i] = 1.0 / (i as f64 + 2.0);
        i += 1;
    }
    c
};

/// `B_2k / (2k (2k - 1))` for `k = 1 ... 8`, the Stirling series of
/// [`ln_gamma_star`] in `1 / a²`.
const STIRLING: [f64; 8] = [
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360_360.0,
    1.0 / 156.0,
    -3617.0 / 122_400.0,
];

/// `z - ln(1 + z)` for `z > -1`, without cancellation for small `z`.
pub(crate) fn log1pmx(z: f64) -> f64 {
    if z.abs() < SERIES_WITHIN {
        let w = -z;
        let mut sum = 0.0;
        for &c in INVERSES.iter().rev() {
            sum = sum * w + c;
        }
        z * z * sum
    } else {
        z - z.ln_1p()
    }
}

/// `λ - 1 - ln λ` for `λ = x / a`, with `x >= 0` and `a > 0`.
pub(crate) fn ratio_gap(x: f64, a: f64) -> f64 {
    let lambda = x / a;
    if lambda < 1.0 - SERIES_WITHIN {
        (lambda - 1.0) - lambda.ln()
    } else {
        log1pmx((x - a) / a)
    }
}

/// Loader's deviance term `bd0(x, m) = x ln(x / m) + m - x`, for `x > 0`
/// and `m > 0`.
pub(crate) fn bd0(x: f64, m: f64) -> f64 {
    x * ratio_gap(m, x)
}

/// The Stirling correction `ln Γ*(a) = ln Γ(a) - (a - 1/2) ln a + a - ln(2π)/2`
/// for `a > 0`, Loader's `stirlerr(a)`. Below [`STIRLING_FROM`] it is formed
/// as `ln Γ(1 + a) - (a + 1/2) ln a + a - ln(2π)/2` with [`gamma_one_plus`].
pub(crate) fn ln_gamma_star(a: f64) -> f64 {
    if a >= STIRLING_FROM {
        let r = 1.0 / (a * a);
        let mut sum = 0.0;
        for &c in STIRLING.iter().rev() {
            sum = sum * r + c;
        }
        sum / a
    } else {
        gamma_one_plus(a).ln() - ((a + 0.5) * a.ln() - a + LN_SQRT_2PI)
    }
}

/// The coefficients `g_k`, `k = 1 ... 32`, of `1 / Γ(1 + a) - 1 = Σ g_k a^k`
/// (DLMF 5.7.1), from mpmath at 40 digits by
/// `tests/data/gamma_coefficients.py`. For `a <= 1.5` the first omitted
/// term is below `2e-17` of the sum.
#[expect(
    clippy::unreadable_literal,
    reason = "the coefficients are written as generated"
)]
const RECIPROCAL_GAMMA: [f64; 32] = [
    0.5772156649015329,
    -0.6558780715202539,
    -0.04200263503409524,
    0.16653861138229148,
    -0.04219773455554433,
    -0.009621971527876973,
    0.0072189432466631,
    -0.0011651675918590652,
    -0.00021524167411495098,
    0.0001280502823881162,
    -2.013485478078824e-05,
    -1.2504934821426706e-06,
    1.133027231981696e-06,
    -2.056338416977607e-07,
    6.116095104481416e-09,
    5.002007644469223e-09,
    -1.18127457048702e-09,
    1.0434267116911005e-10,
    7.782263439905071e-12,
    -3.696805618642206e-12,
    5.100370287454476e-13,
    -2.0583260535665066e-14,
    -5.348122539423018e-15,
    1.2267786282382608e-15,
    -1.1812593016974588e-16,
    1.1866922547516004e-18,
    1.4123806553180319e-18,
    -2.29874568443537e-19,
    1.7144063219273374e-20,
    1.337351730493693e-22,
    -2.0542335517666728e-22,
    2.736030048608e-23,
];

/// `1 / Γ(1 + a) - 1` for `0 <= a <= 1.5`, from its Taylor series.
pub(crate) fn reciprocal_gamma_minus_one(a: f64) -> f64 {
    let mut g = 0.0;
    for &c in RECIPROCAL_GAMMA.iter().rev() {
        g = g * a + c;
    }
    g * a
}

/// `Γ(1 + a)` for `0 <= a < STIRLING_FROM`, as
/// `(f + 1) (f + 2) ... (f + n) / (1 + (1/Γ(1 + f) - 1))` with `a = n + f`,
/// `0 <= f < 1`: at most ten roundings in the product and a few in the
/// series. `f + k` is exact.
pub(crate) fn gamma_one_plus(a: f64) -> f64 {
    let n = a.floor();
    let f = a - n;
    let mut product = 1.0;
    let mut k = 1.0;
    while k <= n {
        product *= f + k;
        k += 1.0;
    }
    product / (1.0 + reciprocal_gamma_minus_one(f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gap_series_meets_the_direct_form_at_its_border() {
        // Either side of |z| = 0.3 the two forms agree to the direct form's
        // cancellation, a factor |z| / gap of about 8, times a few eps.
        for z in [0.3_f64, -0.3] {
            for w in [z.next_down(), z, z.next_up()] {
                let series = {
                    let mut s = 0.0;
                    for &c in INVERSES.iter().rev() {
                        s = s * -w + c;
                    }
                    w * w * s
                };
                let direct = w - w.ln_1p();
                assert!(
                    (series - direct).abs() <= 32.0 * f64::EPSILON * direct,
                    "{w}"
                );
            }
        }
        assert_eq!(log1pmx(0.0).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn stirling_meets_the_direct_form_at_ten() {
        // The direct form's absolute error is a few eps times
        // |ln Γ(10)| + 9.5 ln 10 + 10, about 45 eps.
        let a = STIRLING_FROM;
        let direct = gamma_one_plus(a).ln() - ((a + 0.5) * a.ln() - a + LN_SQRT_2PI);
        assert!((ln_gamma_star(a) - direct).abs() <= 200.0 * f64::EPSILON);
        // ln Γ*(1) = 1 - ln(2π)/2.
        assert!((ln_gamma_star(1.0) - (1.0 - LN_SQRT_2PI)).abs() <= 8.0 * f64::EPSILON);
    }

    #[test]
    fn gamma_at_small_arguments() {
        // Integers are exact; 1/2 and 3/2 within a few eps of sqrt(pi)/2 and
        // 3 sqrt(pi)/4.
        assert_eq!(gamma_one_plus(0.0), 1.0);
        assert_eq!(gamma_one_plus(4.0), 24.0);
        assert_eq!(gamma_one_plus(9.0), 362_880.0);
        let half = std::f64::consts::PI.sqrt() / 2.0;
        assert!((gamma_one_plus(0.5) - half).abs() <= 4.0 * f64::EPSILON * half);
        assert!((gamma_one_plus(1.5) - 1.5 * half).abs() <= 4.0 * f64::EPSILON * half);
    }

    #[test]
    fn bd0_is_the_deviance() {
        for (x, m) in [(3.0_f64, 2.0_f64), (100.0, 101.0), (1e-3, 5.0), (50.0, 0.5)] {
            let direct: f64 = x * (x / m).ln() + m - x;
            assert!((bd0(x, m) - direct).abs() <= 1e-13 * direct.abs().max(1.0));
        }
    }
}
