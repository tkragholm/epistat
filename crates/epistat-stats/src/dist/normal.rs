//! The standard normal distribution.
//!
//! [`cdf`] and [`quantile`] are the algorithms behind R's `pnorm()` and `qnorm()`.
//! Both use plain operations in a fixed order and no fused multiply-add, so the
//! results are the same on every platform. [`sf`] is the upper tail and
//! [`log_cdf`] the log of the distribution function, R's `pnorm(x, lower.tail =
//! FALSE)` and `pnorm(x, log.p = TRUE)`. [`truncated_quantile`],
//! [`folded_quantile`] and [`two_sided_z`] build on [`cdf`] and [`quantile`].
//! Two older approximations, Abramowitz and Stegun's 7.1.26 and Acklam's
//! quantile, are in [`crate::compat`].
//!
//! # Algorithm
//!
//! [`cdf`] is the rational Chebyshev approximation of Cody \[1\]. With `y = |x|`
//! it has three regions. In the first, `y <= 0.67448975`, the function is odd
//! about 1/2:
//!
//! ```text
//! Φ(x) = 1/2 + x * (a_4 x² + ... ) / (x² + ... )
//! ```
//!
//! that is, a ratio of two polynomials in `x²`, with a numerator of degree 4 and a
//! monic denominator of degree 4. For `y` below half the machine epsilon the
//! polynomials are skipped and the result is exactly 1/2. In the second region,
//! `0.67448975 < y <= sqrt(32)`, the lower tail `Φ(-y)` is
//!
//! ```text
//! Φ(-y) = exp(-y²/2) * (c_8 y^8 + ... + c_0) / (y^8 + d_7 y^7 + ... + d_0)
//! ```
//!
//! In the third region, `y > sqrt(32)`, with `s = 1 / y²`:
//!
//! ```text
//! Φ(-y) = exp(-y²/2) / y * (1/sqrt(2π) - s R(s)),    R(s) = (p_5 s^5 + ... ) / (s^5 + ... )
//! ```
//!
//! The region is chosen on `|x|`, and the result is `Φ(-y)` for `x < 0` and
//! `1 - Φ(-y)` for `x > 0`. In the third region the cutoffs are `x > -38.4674` and
//! `x < 8.2924`. Outside them the lower tail is set to 0, so the result is 0 for
//! `x <= -38.4674` and 1 for `x >= 8.2924`. A NaN input is returned unchanged.
//!
//! `exp(-y²/2)` is formed in `gaussian_tail` as a product of two exponentials. With
//! `h = trunc(16 y) / 16`, which has few significant bits, the code uses
//!
//! ```text
//! y² = h² + (y - h)(y + h),    exp(-y²/2) = exp(-h²/2) * exp(-(y - h)(y + h)/2).
//! ```
//!
//! Here `h²` is exact in a `f64`, so the rounding error of `y²` does not get
//! multiplied by a large exponent. Without the split, a relative error of `2^-53`
//! in `y²` becomes a relative error of about `y² 2^-54` in the result, which
//! grows with `y²` across the tail.
//!
//! [`quantile`] is Wichura's AS 241, algorithm PPND16 \[2\]. With `q = p - 1/2`
//! and `r = 0.180625 - q²`, the central region `|q| <= 0.425` is
//!
//! ```text
//! z = q * (a_7 r^7 + ... + a_0) / (b_7 r^7 + ... + 1).
//! ```
//!
//! Otherwise `t = min(p, 1 - p)` and `r = sqrt(-ln t)`. For `r <= 5` the code
//! evaluates a ratio of degree 7 polynomials in `r - 1.6`, and for `r > 5` a
//! second ratio in `r - 5`. The result takes the sign of `q`. The value is
//! `-inf` at 0, `inf` at 1 and NaN outside `[0, 1]`, and `1 - p` is rounded
//! before the log in the upper tail. AS 241 states a relative accuracy of about
//! `10^-16`.
//!
//! # Agreement with R
//!
//! All polynomials are evaluated by Horner's rule with separate multiplication
//! and addition. [`cdf`] gives the same doubles as R's `pnorm()` on the values the
//! tests use. [`quantile`] is within two units in the last place of `qnorm()`.
//! R built with fused multiply-add rounds its polynomials differently, and a
//! single fused step changes the last bit, so bit equality is not expected for
//! the quantile.
//!
//! # Upper tail and log scale
//!
//! [`sf`] is `cdf(-x)`, which is the upper tail computed directly: Cody's
//! central region is odd about 1/2, and in the tails the same lower-tail value
//! is returned, so `sf` keeps the relative precision of the upper tail as R's
//! `lower.tail = FALSE` does.
//!
//! [`log_cdf`] is `ln Φ(x)`. For `x < -sqrt(32)`, Cody's third region is taken
//! on the log scale,
//!
//! ```text
//! ln Φ(-y) = -h²/2 - (y - h)(y + h)/2 + ln((1/sqrt(2π) - s R(s)) / y),
//! ```
//!
//! with `h = trunc(16 y) / 16` as above, so it does not underflow where `Φ(x)`
//! does, below `x = -38.4674`; this is R's `pnorm(x, log.p = TRUE)` in that
//! region. For `-sqrt(32) <= x <= 0` it is `ln Φ(x)`, which keeps the relative
//! precision of the lower tail, and for `x > 0` it is `ln(1 - Φ(-x))` by
//! `ln_1p`, which keeps that of a value near 0.
//!
//! # Truncated and folded quantiles
//!
//! [`truncated_quantile`] inverts the distribution function of `Z` given
//! `lower <= Z <= upper`:
//!
//! ```text
//! z = Φ^-1( Φ(lower) + p (Φ(upper) - Φ(lower)) ),
//! ```
//!
//! clamped to `[lower, upper]`. An interval with `lower > 0` is mirrored to
//! `(-upper, -lower)` with `1 - p`, and the answer is negated. The lower tail of
//! the normal is where [`cdf`] keeps its relative precision, so mirroring keeps
//! intervals far out in either tail resolvable.
//!
//! Where `Φ(upper) <= Φ(lower)` in floating point, the interval lies so far in
//! a tail that its probability underflows, dozens of standard deviations from
//! zero. There the quantile is computed without `Φ`. An interval below zero is
//! mirrored to `[a, b] = [-upper, -lower]` with the quantile `1 - p`, so that
//! `0 <= a < b`. With `Q(x) = 1 - Φ(x)` the upper tail and `r(x) = Q(x) / Q(a)`,
//! the quantile `x` solves
//!
//! ```text
//! r(x) = 1 - p (1 - r(b)),
//! ln r(x) = (a - x)(a + x) / 2 + ln M(x) - ln M(a),
//! ```
//!
//! where `M(x) = Q(x) / φ(x)` is Mills' ratio, evaluated by Laplace's continued
//! fraction `M(x) = 1 / (x + 1 / (x + 2 / (x + 3 / (x + ...))))` with 200 terms
//! from the inside out \[3\]. Every quantity stays a ratio or a logarithm, so
//! nothing underflows. Since `d ln r(x) / dx = -1 / M(x)`, Newton's step is
//! `x + (ln r(x) - target) M(x)`, started from `sqrt(a² - 2 target)`, the root
//! when `M` is taken as constant, and kept inside `[a, b]`. `M` decreases, so
//! the start lies above the root, and `ln r` is concave and decreasing, so from
//! there every step stays above the root and moves down to it. The iteration
//! stops when a step no longer moves `x`. This is the logarithmic Newton
//! iteration of `TruncatedNormal::norminvp` in R \[4\].
//!
//! The quantile is NaN for `p` outside `[0, 1]`, a NaN bound, `lower >= upper`,
//! and an interval that contains zero yet has no probability in floating point.
//!
//! [`folded_quantile`] is the quantile of `|Z + mu|`. Its distribution function is
//!
//! ```text
//! F(q) = Φ(q - |mu|) + Φ(q + |mu|) - 1,    q >= 0.
//! ```
//!
//! The code brackets the root in `[0, |mu| + 8]`, doubling the upper end until
//! `F` reaches `p`, and bisects until the midpoint equals an endpoint, which is the
//! precision of an `f64`. The sign of `mu` does not matter. It is NaN for `p`
//! outside `[0, 1)` and for a NaN `mu`.
//!
//! # References
//!
//! 1. Cody WJ. Rational Chebyshev approximations for the error function.
//!    Mathematics of Computation 1969;23:631-637.
//! 2. Wichura MJ. Algorithm AS 241: the percentage points of the normal
//!    distribution. Journal of the Royal Statistical Society, Series C (Applied
//!    Statistics) 1988;37:477-484.
//! 3. Laplace PS. Traité de mécanique céleste, volume 4. Paris; 1805. The
//!    continued fraction of the normal tail.
//! 4. Botev ZI. The normal law under linear restrictions: simulation and
//!    estimation via minimax tilting. Journal of the Royal Statistical Society,
//!    Series B 2017;79:125-148.

#![expect(
    clippy::excessive_precision,
    reason = "the coefficients are written as published"
)]
#![expect(
    clippy::float_cmp,
    reason = "the endpoints 0 and 1 are compared exactly"
)]

use epistat_core::Level;

/// `1 / sqrt(2 pi)`.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_677_939_946_059_934;

/// The standard normal density `exp(-x^2 / 2) / sqrt(2 pi)`.
///
/// Returns NaN for NaN and 0 for infinite `x`.
#[must_use]
pub fn pdf(x: f64) -> f64 {
    FRAC_1_SQRT_2PI * (-0.5 * x * x).exp()
}

/// `ln f(x)` of the standard normal density, `-x^2 / 2 - ln(2 pi) / 2`, R's
/// `dnorm(x, log = TRUE)`. Finite where [`pdf`] underflows; NaN for NaN.
#[must_use]
pub fn ln_pdf(x: f64) -> f64 {
    -0.5 * x * x - super::stirling::LN_SQRT_2PI
}

/// `P(Z <= x)` for a standard normal `Z`.
#[must_use]
pub fn cdf(x: f64) -> f64 {
    const A: [f64; 5] = [
        2.235_252_035_460_683_928_7,
        161.028_231_068_555_878_81,
        1_067.689_485_460_370_958_2,
        18_154.981_253_343_561_249,
        0.065_682_337_918_207_449_113,
    ];
    const B: [f64; 4] = [
        47.202_581_904_688_241_87,
        976.098_551_737_776_693_22,
        10_260.932_208_618_978_205,
        45_507.789_335_026_729_956,
    ];
    const C: [f64; 9] = [
        0.398_941_512_088_134_667_64,
        8.883_149_794_388_375_941_2,
        93.506_656_132_177_855_979,
        597.270_276_394_800_262_26,
        2_494.537_585_290_372_671_1,
        6_848.190_450_536_282_332_6,
        11_602.651_437_647_350_124,
        9_842.714_838_383_978_021_8,
        1.076_557_677_372_019_231_7e-8,
    ];
    const D: [f64; 8] = [
        22.266_688_044_328_115_691,
        235.387_901_782_624_998_61,
        1_519.377_599_407_554_805,
        6_485.558_298_266_760_755,
        18_615.571_640_885_098_091,
        34_900.952_721_145_977_266,
        38_912.003_286_093_271_411,
        19_685.429_676_859_990_727,
    ];
    if x.is_nan() {
        return x;
    }
    let y = x.abs();
    if y <= 0.674_489_75 {
        let (mut num, mut den) = (0.0, 0.0);
        if y > f64::EPSILON * 0.5 {
            let xsq = x * x;
            num = A[4] * xsq;
            den = xsq;
            for i in 0..3 {
                num = (num + A[i]) * xsq;
                den = (den + B[i]) * xsq;
            }
        }
        return 0.5 + x * (num + A[3]) / (den + B[3]);
    }
    let lower = if y <= 32.0_f64.sqrt() {
        let mut num = C[8] * y;
        let mut den = y;
        for i in 0..7 {
            num = (num + C[i]) * y;
            den = (den + D[i]) * y;
        }
        gaussian_tail(y, (num + C[7]) / (den + D[7]))
    } else if x > -38.4674 && x < 8.2924 {
        gaussian_tail(x, far_factor(x, y))
    } else {
        0.0
    };
    if x > 0.0 { 1.0 - lower } else { lower }
}

/// Cody's coefficients of the third region, `|x| > sqrt(32)`.
const P: [f64; 6] = [
    0.215_898_534_057_956_99,
    0.127_401_161_160_247_363_9,
    0.022_235_277_870_649_807,
    0.001_421_619_193_227_893_466,
    2.911_287_495_116_879_2e-5,
    0.023_073_441_764_940_173_03,
];
const Q: [f64; 5] = [
    1.284_260_096_144_911_21,
    0.468_238_212_480_865_118,
    0.065_988_137_868_928_551_5,
    0.003_782_396_332_027_582_44,
    7.297_515_550_839_662_05e-5,
];

/// `(1/sqrt(2 pi) - s R(s)) / y` with `s = 1 / x^2`, the factor of
/// `exp(-x^2 / 2)` in the lower tail of the third region.
fn far_factor(x: f64, y: f64) -> f64 {
    let xsq = 1.0 / (x * x);
    let mut num = P[5] * xsq;
    let mut den = xsq;
    for i in 0..4 {
        num = (num + P[i]) * xsq;
        den = (den + Q[i]) * xsq;
    }
    let ratio = xsq * (num + P[4]) / (den + Q[4]);
    (FRAC_1_SQRT_2PI - ratio) / y
}

/// `exp(-x^2 / 2) * factor`, with `x^2` split so that the exponential loses
/// no precision in the tail.
fn gaussian_tail(x: f64, factor: f64) -> f64 {
    let head = (x * 16.0).trunc() / 16.0;
    let rest = (x - head) * (x + head);
    (-head * head * 0.5).exp() * (-rest * 0.5).exp() * factor
}

/// `P(Z > x)` for a standard normal `Z`, computed directly so that the upper
/// tail keeps its relative precision; R's `pnorm(x, lower.tail = FALSE)`.
#[must_use]
pub fn sf(x: f64) -> f64 {
    cdf(-x)
}

/// `ln P(Z > x)`, R's `pnorm(x, lower.tail = FALSE, log.p = TRUE)`: [`log_cdf`]
/// at `-x`, finite far above the point where [`sf`] underflows.
#[must_use]
pub fn ln_sf(x: f64) -> f64 {
    log_cdf(-x)
}

/// `2 P(Z > |z|)`, the two-sided p-value of a normal statistic. NaN for NaN.
#[must_use]
pub fn two_sided_p(z: f64) -> f64 {
    2.0 * cdf(-z.abs())
}

/// `ln P(Z <= x)` for a standard normal `Z`, R's `pnorm(x, log.p = TRUE)`.
///
/// Finite far below the point where [`cdf`] underflows to 0; `-inf` at
/// `-inf`, 0 at `inf`, NaN for NaN.
#[must_use]
pub fn log_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 0.0 {
        return (-cdf(-x)).ln_1p();
    }
    if x == f64::NEG_INFINITY {
        return x;
    }
    let y = -x;
    if y <= 32.0_f64.sqrt() {
        return cdf(x).ln();
    }
    let head = (y * 16.0).trunc() / 16.0;
    let rest = (y - head) * (y + head);
    -head * head * 0.5 - rest * 0.5 + far_factor(x, y).ln()
}

/// The `p` quantile of the standard normal: the `z` with `cdf(z) = p`.
///
/// Returns `-inf` at 0, `inf` at 1 and NaN outside `[0, 1]`.
#[must_use]
pub fn quantile(p: f64) -> f64 {
    // Wichura's coefficients, highest power first.
    const A: [f64; 8] = [
        2_509.080_928_730_122_672_7,
        33_430.575_583_588_128_105,
        67_265.770_927_008_700_853,
        45_921.953_931_549_871_457,
        13_731.693_765_509_461_125,
        1_971.590_950_306_551_442_7,
        133.141_667_891_784_377_45,
        3.387_132_872_796_366_608,
    ];
    const B: [f64; 8] = [
        5_226.495_278_852_545_925,
        28_729.085_735_721_942_674,
        39_307.895_800_092_710_61,
        21_213.794_301_586_595_867,
        5_394.196_021_424_751_107_7,
        687.187_007_492_057_908_3,
        42.313_330_701_600_911_252,
        1.0,
    ];
    const C: [f64; 8] = [
        7.745_450_142_783_414_076_4e-4,
        0.022_723_844_989_269_184_583_3,
        0.241_780_725_177_450_611_77,
        1.270_458_252_452_368_382_58,
        3.647_848_324_763_204_605_04,
        5.769_497_221_460_691_405_5,
        4.630_337_846_156_545_295_9,
        1.423_437_110_749_683_577_34,
    ];
    const D: [f64; 8] = [
        1.050_750_071_644_416_843_24e-9,
        5.475_938_084_995_344_946e-4,
        0.015_198_666_563_616_457_196_6,
        0.148_103_976_427_480_074_59,
        0.689_767_334_985_100_004_55,
        1.676_384_830_183_803_849_4,
        2.053_191_626_637_758_821_87,
        1.0,
    ];
    const E: [f64; 8] = [
        2.010_334_399_292_288_132_65e-7,
        2.711_555_568_743_487_578_15e-5,
        0.001_242_660_947_388_078_438_6,
        0.026_532_189_526_576_123_093,
        0.296_560_571_828_504_891_23,
        1.784_826_539_917_291_335_8,
        5.463_784_911_164_114_369_9,
        6.657_904_643_501_103_777_2,
    ];
    const F: [f64; 8] = [
        2.044_263_103_389_939_785_64e-15,
        1.421_511_758_316_445_888_7e-7,
        1.846_318_317_510_054_681_8e-5,
        7.868_691_311_456_132_591e-4,
        0.014_875_361_290_850_614_852_5,
        0.136_929_880_922_735_805_31,
        0.599_832_206_555_887_937_69,
        1.0,
    ];

    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return q * horner(r, &A) / horner(r, &B);
    }
    let tail = if q > 0.0 { 1.0 - p } else { p };
    let r = (-tail.ln()).sqrt();
    let value = if r <= 5.0 {
        let r = r - 1.6;
        horner(r, &C) / horner(r, &D)
    } else {
        let r = r - 5.0;
        horner(r, &E) / horner(r, &F)
    };
    if q < 0.0 { -value } else { value }
}

/// The polynomial with `coefficients`, highest power first, at `x`.
fn horner(x: f64, coefficients: &[f64]) -> f64 {
    coefficients[1..]
        .iter()
        .fold(coefficients[0], |sum, &c| sum * x + c)
}

/// The upper-tail quantile, the `z` with `sf(z) = p`, R's
/// `qnorm(p, lower.tail = FALSE)`: `-quantile(p)`, by the symmetry of the
/// distribution. It keeps the precision of a small `p`, which
/// `quantile(1.0 - p)` loses when it rounds `1 - p`: `isf(1e-17)` is 8.49
/// where `quantile(1.0 - 1e-17)` is infinite. `inf` at 0, `-inf` at 1 and NaN
/// outside `[0, 1]`.
#[must_use]
pub fn isf(p: f64) -> f64 {
    -quantile(p)
}

/// The `p` quantile of a standard normal truncated to `[lower, upper]`.
///
/// Computed in the lower tail, mirroring an interval that lies above zero, so
/// that intervals far out in either tail keep their precision, and without the
/// distribution function where the interval's probability underflows (see the
/// module documentation). NaN when `p` is outside `[0, 1]`, a bound is NaN,
/// `lower >= upper`, or an interval that contains zero has no probability.
#[must_use]
pub fn truncated_quantile(p: f64, lower: f64, upper: f64) -> f64 {
    truncated_quantile_central(p, lower, upper)
        .or_else(|| far_tail_truncated_quantile(p, lower, upper))
        .unwrap_or(f64::NAN)
}

/// [`truncated_quantile`] by inverting [`cdf`] alone: `None` where the
/// interval's probability underflows.
pub(crate) fn truncated_quantile_central(p: f64, lower: f64, upper: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) || lower.is_nan() || upper.is_nan() || lower >= upper {
        return None;
    }
    if lower > 0.0 {
        return truncated_quantile_central(1.0 - p, -upper, -lower).map(|z| -z);
    }
    let (cdf_lower, cdf_upper) = (cdf(lower), cdf(upper));
    if cdf_upper <= cdf_lower {
        return None;
    }
    let z = quantile(cdf_lower + p * (cdf_upper - cdf_lower));
    Some(z.clamp(lower, upper))
}

/// Terms of the continued fraction in [`mills_ratio`]. The far-tail quantile
/// meets it above about 37 standard deviations, where a few terms reach full
/// precision; with 200 the relative error stays below `1e-15` down to `x = 2`.
const MILLS_TERMS: u32 = 200;

/// Newton steps allowed in [`far_tail_truncated_quantile`]; it converges in
/// under ten.
const MAX_NEWTON_STEPS: usize = 100;

/// The `p` quantile of a standard normal truncated to `[l, u]`, for an
/// interval on one side of zero, computed without the normal distribution
/// function so that it holds where that function underflows. `None` for an
/// interval that contains zero, whose probability cannot underflow, for `a`
/// infinite, or for an empty interval.
fn far_tail_truncated_quantile(p: f64, l: f64, u: f64) -> Option<f64> {
    let (a, b, q, sign) = if l >= 0.0 {
        (l, u, p, 1.0)
    } else if u <= 0.0 {
        (-u, -l, 1.0 - p, -1.0)
    } else {
        return None;
    };
    if !a.is_finite() || a >= b {
        return None;
    }
    if q == 0.0 {
        return Some(sign * a);
    }
    if q == 1.0 {
        return Some(sign * b);
    }
    let ln_mills_a = mills_ratio(a).ln();
    let ln_ratio = |x: f64| {
        if x == f64::INFINITY {
            f64::NEG_INFINITY
        } else {
            (a - x) * (a + x) / 2.0 + mills_ratio(x).ln() - ln_mills_a
        }
    };
    // ln(1 - q (1 - r(b))), with 1 - r(b) = -expm1(ln r(b)).
    let target = (-q * -ln_ratio(b).exp_m1()).ln_1p();
    let mut x = (a * a - 2.0 * target).sqrt().clamp(a, b);
    for _ in 0..MAX_NEWTON_STEPS {
        let next = (x + (ln_ratio(x) - target) * mills_ratio(x)).clamp(a, b);
        if next == x {
            break;
        }
        x = next;
    }
    Some(sign * x)
}

/// Mills' ratio `Q(x) / phi(x)` of the standard normal for `x > 0`, by
/// Laplace's continued fraction evaluated from the inside out.
fn mills_ratio(x: f64) -> f64 {
    let mut denominator = x;
    for k in (1..=MILLS_TERMS).rev() {
        denominator = x + f64::from(k) / denominator;
    }
    1.0 / denominator
}

/// The critical value of a two-sided interval at `level`, the
/// [`Level::upper_probability`] quantile of the standard normal: 1.96 for a
/// confidence of 0.95.
#[must_use]
pub fn two_sided_z(level: Level) -> f64 {
    quantile(level.upper_probability())
}

/// The `p` quantile of `|Z + mu|` for a standard normal `Z`, by bisection on
/// its distribution function `cdf(q - |mu|) + cdf(q + |mu|) - 1` to the
/// precision of an `f64`. NaN when `p` is outside `[0, 1)` or `mu` is NaN.
#[must_use]
pub fn folded_quantile(p: f64, mu: f64) -> f64 {
    if !(0.0..1.0).contains(&p) || mu.is_nan() {
        return f64::NAN;
    }
    let mu = mu.abs();
    let folded_cdf = |q: f64| cdf(q - mu) + cdf(q + mu) - 1.0;
    let (mut low, mut high) = (0.0_f64, mu + 8.0);
    while folded_cdf(high) < p {
        high *= 2.0;
    }
    loop {
        let mid = f64::midpoint(low, high);
        if mid <= low || mid >= high {
            return high;
        }
        if folded_cdf(mid) >= p {
            high = mid;
        } else {
            low = mid;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_density_is_the_derivative_of_the_distribution_function() {
        assert!((pdf(0.0) - FRAC_1_SQRT_2PI).abs() < 1e-17);
        assert_eq!(pdf(f64::INFINITY), 0.0);
        let h = 1e-5;
        for x in [-2.0, -0.3, 0.0, 1.1, 3.0] {
            let numeric = (cdf(x + h) - cdf(x - h)) / (2.0 * h);
            assert!((numeric - pdf(x)).abs() < 1e-9, "{x}");
        }
    }

    #[test]
    fn a_nan_bound_has_no_truncated_quantile() {
        assert!(truncated_quantile(0.5, f64::NAN, 1.0).is_nan());
        assert!(truncated_quantile(0.5, -1.0, f64::NAN).is_nan());
        assert!(truncated_quantile(0.5, f64::NAN, f64::NAN).is_nan());
    }

    #[test]
    fn the_two_sided_critical_value_is_the_upper_quantile_of_half_the_tail() {
        // qnorm(0.975) in R.
        let z = two_sided_z(Level::default());
        assert!((z - 1.959_963_984_540_054).abs() < 4.5e-16);
        let level = Level::confidence(0.9).unwrap();
        assert_eq!(two_sided_z(level).to_bits(), quantile(0.95).to_bits());
        // The confidence form keeps the expression 1 - (1 - c) / 2 and the
        // alpha form 1 - a / 2.
        let c = 0.95_f64;
        assert_eq!(
            two_sided_z(Level::confidence(c).unwrap()).to_bits(),
            quantile(1.0 - (1.0 - c) / 2.0).to_bits()
        );
        let a = 0.05_f64;
        assert_eq!(
            two_sided_z(Level::alpha(a).unwrap()).to_bits(),
            quantile(1.0 - a / 2.0).to_bits()
        );
    }

    #[test]
    fn quantile_and_cdf_invert_each_other() {
        for i in 1..200 {
            let p = f64::from(i) / 200.0;
            assert!((cdf(quantile(p)) - p).abs() < 1e-15);
        }
    }

    #[test]
    fn the_upper_tail_is_the_mirrored_lower_tail() {
        for i in -400..=400 {
            let x = f64::from(i) / 20.0;
            assert_eq!(sf(x).to_bits(), cdf(-x).to_bits(), "{x}");
        }
        // pnorm(10, lower.tail = FALSE) in R, which 1 - cdf(10) loses.
        let tail = 7.619_853_024_160_526_9e-24;
        assert!(((sf(10.0) - tail) / tail).abs() < 1e-14);
        assert!(sf(f64::NAN).is_nan());
    }

    /// `pnorm(x, log.p = TRUE)` in R 4.5. The tolerance is the error of the
    /// distribution function, about `1e-15` relative in the lower tail, carried
    /// into the log as an absolute error, plus four units in the last place of
    /// the log itself; above zero `ln_1p` keeps the relative error of the upper
    /// tail.
    #[test]
    fn the_log_distribution_function_agrees_with_r() {
        let reference = [
            (-1000.0, -500_007.826_694_812_16),
            (-200.0, -20_006.217_280_898_189),
            (-40.0, -804.608_442_013_753_8),
            (-38.0, -726.557_216_018_820_1),
            (-10.0, -53.231_285_150_512_47),
            (-5.7, -18.933_112_423_224_742),
            (-5.0, -15.064_998_393_988_725),
            (-1.0, -1.841_021_645_009_263_6),
            (-0.5, -1.175_911_761_593_618_5),
            (0.0, -std::f64::consts::LN_2),
            (0.3, -0.481_410_161_588_481_16),
            (1.0, -0.172_753_779_023_449_88),
            (5.0, -2.866_516_129_637_635_8e-7),
            (8.0, -6.220_960_574_271_786_8e-16),
            (10.0, -7.619_853_024_160_526_9e-24),
        ];
        for (x, want) in reference {
            let got = log_cdf(x);
            let tolerance = if x > 0.0 {
                1e-14 * want.abs()
            } else {
                1e-15 + 4.0 * f64::EPSILON * want.abs()
            };
            assert!(
                (got - want).abs() <= tolerance,
                "x = {x}: {got} against {want}"
            );
        }
        assert_eq!(log_cdf(40.0), 0.0);
        assert_eq!(log_cdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(log_cdf(f64::NAN).is_nan());
        // Where cdf does not underflow, the log of it.
        for i in -100..=0 {
            let x = f64::from(i) / 20.0;
            assert_eq!(log_cdf(x).to_bits(), cdf(x).ln().to_bits(), "{x}");
        }
    }

    #[test]
    fn truncated_and_folded_quantiles() {
        // Truncation to the whole line changes nothing.
        let z = truncated_quantile(0.3, f64::NEG_INFINITY, f64::INFINITY);
        assert!((z - quantile(0.3)).abs() < 1e-15);
        // Far in the upper tail the mirror keeps the interval resolvable.
        let z = truncated_quantile(0.5, 30.0, 31.0);
        assert!((30.0..30.1).contains(&z));
        assert!(truncated_quantile(0.5, 2.0, 1.0).is_nan());
        // The quantile has the requested share of the interval's probability.
        let (lower, upper) = (-1.5, 0.75);
        let z = truncated_quantile(0.2, lower, upper);
        let share = (cdf(z) - cdf(lower)) / (cdf(upper) - cdf(lower));
        assert!((share - 0.2).abs() < 1e-14);
        // With mu = 0 the folded normal is the half-normal.
        let q = folded_quantile(0.5, 0.0);
        assert!((q - quantile(0.75)).abs() < 1e-12);
        assert!((folded_quantile(0.95, -3.0) - folded_quantile(0.95, 3.0)).abs() < 1e-15);
        assert!(folded_quantile(1.0, 0.0).is_nan());
        assert!(folded_quantile(0.5, f64::NAN).is_nan());
    }

    /// Where the distribution function does not underflow, the far-tail
    /// route agrees with the direct one, which inverts `Phi` and so loses
    /// relative precision in proportion to the tail's depth: `1e-12`
    /// relative holds to 30 standard deviations.
    #[test]
    fn far_tail_route_agrees_with_the_direct_inversion() {
        for &(p, l, u) in &[
            (0.5, 5.0, 6.0),
            (0.9, 10.0, 12.0),
            (0.1, -30.0, -20.0),
            (0.7, 20.0, f64::INFINITY),
            (0.3, f64::NEG_INFINITY, -8.0),
        ] {
            let direct = truncated_quantile_central(p, l, u).expect("direct");
            let tail = far_tail_truncated_quantile(p, l, u).expect("tail");
            assert!(
                ((direct - tail) / direct).abs() < 1e-12,
                "p = {p}, [{l}, {u}]: direct {direct}, far tail {tail}"
            );
        }
    }

    /// Beyond 38.5 standard deviations `Phi` underflows and only the far tail
    /// gives a quantile. The interval's quantile then lies within it, and the
    /// share of the tail beyond it is `1 - p` of the interval's, which the
    /// Mills ratio states exactly: `Q(x) / Q(a) = exp((a² - x²)/2) M(x) / M(a)`.
    #[test]
    fn intervals_where_the_distribution_function_underflows_have_a_quantile() {
        for &(p, l, u) in &[
            (0.5, 40.0, 41.0),
            (0.25, -60.0, -50.0),
            (0.9, 45.0, f64::INFINITY),
        ] {
            assert_eq!(truncated_quantile_central(p, l, u), None);
            let x = truncated_quantile(p, l, u);
            assert!((l..=u).contains(&x), "p = {p}, [{l}, {u}]: {x}");
            let (a, b, q) = if l >= 0.0 {
                (l, u, p)
            } else {
                (-u, -l, 1.0 - p)
            };
            let y = x.abs();
            let ratio = |t: f64| {
                if t.is_infinite() {
                    0.0
                } else {
                    ((a - t) * (a + t) / 2.0).exp() * mills_ratio(t) / mills_ratio(a)
                }
            };
            let share = (1.0 - ratio(y)) / (1.0 - ratio(b));
            assert!(
                (share - q).abs() < 1e-12,
                "p = {p}, [{l}, {u}]: share {share}"
            );
        }
        // An interval with no probability that straddles zero cannot occur;
        // an empty one is NaN.
        assert!(truncated_quantile(0.5, 41.0, 40.0).is_nan());
    }

    #[test]
    fn mills_ratio_matches_the_normal_tail() {
        // Q(x) / phi(x): at 2, 5 and 8 R's pnorm(x, lower.tail = FALSE) /
        // dnorm(x); at 40, where pnorm underflows, the asymptotic series
        // (1 / x) sum_k (-1)^k (2k - 1)!! / x^(2k) to eight terms, whose
        // truncation error is below 1e-20.
        for &(x, expected) in &[
            (2.0, 0.421_369_229_288_054_5),
            (5.0, 0.192_808_104_715_315_76),
            (8.0, 0.123_131_963_257_932_3),
            (40.0, 0.024_984_404_205_720_574),
        ] {
            let ours = mills_ratio(x);
            assert!(
                ((ours - expected) / expected).abs() < 1e-15,
                "x = {x}: {ours}"
            );
        }
    }

    #[test]
    fn edges() {
        assert!(quantile(-0.1).is_nan());
        assert!(cdf(f64::NAN).is_nan());
        assert_eq!(quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(quantile(1.0), f64::INFINITY);
    }
}
