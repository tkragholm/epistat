//! The standard normal distribution function and its inverse.
//!
//! [`cdf`] is Cody's rational Chebyshev approximation (Cody 1969, Math.
//! Comp. 23:631-637) and [`quantile`] is Wichura's AS 241 (1988, Appl.
//! Statist. 37:477-484), the algorithms behind R's `pnorm()` and `qnorm()`.
//! [`cdf`] gives the same doubles as R; [`quantile`] is within two units in
//! the last place of it, since R built with fused multiply-add rounds its
//! polynomials differently.
//!
//! [`cdf_as26`] and [`quantile_acklam`] are shorter approximations that older
//! code uses, kept so that such code can move here without its results
//! changing.
//!
//! Everything here uses plain operations in a fixed order, without fused
//! multiply-add, so the results are the same on every platform.

// The coefficients are written digit for digit as published.
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]
// Exact comparisons are with the endpoints 0 and 1.
#![allow(clippy::float_cmp)]

use std::f64::consts::SQRT_2;

/// `1 / sqrt(2 pi)`.
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_677_939_946_059_934;

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
        let xsq = 1.0 / (x * x);
        let mut num = P[5] * xsq;
        let mut den = xsq;
        for i in 0..4 {
            num = (num + P[i]) * xsq;
            den = (den + Q[i]) * xsq;
        }
        let ratio = xsq * (num + P[4]) / (den + Q[4]);
        gaussian_tail(x, (FRAC_1_SQRT_2PI - ratio) / y)
    } else {
        0.0
    };
    if x > 0.0 { 1.0 - lower } else { lower }
}

/// `exp(-x^2 / 2) * factor`, with `x^2` split so that the exponential loses
/// no precision in the tail.
#[allow(clippy::float_cmp)]
fn gaussian_tail(x: f64, factor: f64) -> f64 {
    let head = (x * 16.0).trunc() / 16.0;
    let rest = (x - head) * (x + head);
    (-head * head * 0.5).exp() * (-rest * 0.5).exp() * factor
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

/// [`cdf`] by the error-function approximation 7.1.26 of Abramowitz and
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

/// [`quantile`] by Acklam's rational approximation, whose relative error is
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn quantile_and_cdf_invert_each_other() {
        for i in 1..200 {
            let p = f64::from(i) / 200.0;
            assert!((cdf(quantile(p)) - p).abs() < 1e-15);
        }
    }

    #[test]
    fn edges() {
        assert!(quantile(-0.1).is_nan());
        assert!(quantile_acklam(1.1).is_nan());
        assert!(cdf(f64::NAN).is_nan());
        assert_eq!(quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(quantile_acklam(1.0), f64::INFINITY);
    }
}
