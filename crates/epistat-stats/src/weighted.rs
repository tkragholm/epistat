//! The weighted mean and the variance with reliability weights.
//!
//! # Algorithm
//!
//! With values `x_i`, weights `w_i > 0` and `W = sum w_i`, [`mean`] is
//!
//! ```text
//! mean = sum(w_i x_i) / W
//! ```
//!
//! and [`variance`] is the reliability-weight estimator \[1\]
//!
//! ```text
//! var = sum(w_i (x_i - mean)^2) / (W - sum(w_i^2) / W).
//! ```
//!
//! The weights give the relative importance of rows, not the number of
//! observations a row stands for, so multiplying every weight by a constant
//! does not change `var`. With unit weights the divisor is `n - 1`. The
//! deviations are taken from the mean in a second pass, not through
//! `E[x^2] - mean^2`, which loses precision for a large mean and a small
//! spread. Without weights every weight is 1.
//!
//! # Summation order
//!
//! [`mean`] rounds each term `x_i w_i` and each `w_i` to 15 decimals, to the
//! nearest with ties to even, and adds them as integers, so the result does
//! not depend on the order of the rows. [`fixed_point_sum`] is that sum for any
//! number of decimals, and it gives the bits of a polars `Decimal(38, places)`
//! cast and sum converted to `Float64`. A term, or a total, whose magnitude times
//! `10^places` reaches `10^38` is refused. The unweighted mean divides by the
//! number of rows.
//!
//! [`variance`] adds `w_i`, `w_i^2`, `w_i x_i` and the weighted squared
//! deviations with [`crate::sum::blocked`], the order of the polars float sum
//! over one chunk. The result depends on the order of the rows, as that sum
//! does. The unweighted case uses the row count for `W` and `sum(w_i^2)`.
//!
//! # Edge cases
//!
//! Rows with a weight that is zero or negative are dropped. `None` is returned
//! when the values and weights differ in length, a value or weight is NaN or
//! infinite, no row remains, `W` is not positive and, for [`variance`], when
//! fewer than two rows remain or the divisor is not positive.
//!
//! # Agreement with other software
//!
//! The tests compare [`variance`] with R's `cov.wt()` on normalised weights, which
//! uses the same divisor, to `1e-14` relative, and [`mean`] and [`variance`] with
//! the polars computation that `weighted_mean` and `weighted_variance` of the
//! `dansk_register` Python package run, bit for bit, on fixed data.
//!
//! # References
//!
//! 1. Cochran WG. Sampling Techniques. 3rd ed. New York: Wiley; 1977.
//!    Section 2.
//! 2. R Core Team. `cov.wt` in the R package stats, method `"unbiased"`.

use crate::sum::blocked;

/// The number of decimals each term of the weighted mean is rounded to.
pub const WEIGHTED_DECIMAL_PLACES: u32 = 15;

/// The largest number of decimals for which `10^places` is exact in an `f64`.
pub const MAX_PLACES: u32 = 22;

const POWERS_OF_TEN: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

/// `10^38`, the first magnitude `Decimal(38, _)` cannot hold.
const LIMIT: i128 = 10i128.pow(38);

/// Terms below this scaled magnitude are converted through an `i64`; 2^63 is
/// 9.22e18.
const I64_BOUND: f64 = 9.0e18;

/// `value` scaled by `scale` and rounded to the nearest integer, ties to even.
/// `None` when it is NaN, infinite or `10^38` or more.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the rounded value is below 2^63 or checked against 10^38 before the cast"
)]
pub(crate) fn fixed_point(value: f64, scale: f64) -> Option<i128> {
    let scaled = (value * scale).round_ties_even();
    if scaled.abs() < I64_BOUND {
        return Some(i128::from(scaled as i64));
    }
    if !scaled.is_finite() {
        return None;
    }
    let integer = scaled as i128;
    (integer.unsigned_abs() < LIMIT.unsigned_abs()).then_some(integer)
}

/// `10^places` as an `f64`, `None` above [`MAX_PLACES`].
pub(crate) fn scale_for(places: u32) -> Option<f64> {
    POWERS_OF_TEN.get(places as usize).copied()
}

/// An integer total as the `f64` it is converted to: the integer rounded to
/// the nearest `f64`, divided by `scale`. `None` at `10^38` or more.
pub(crate) fn total_to_f64(total: i128, scale: f64) -> Option<f64> {
    (total.unsigned_abs() < LIMIT.unsigned_abs()).then(|| total as f64 / scale)
}

/// The sum of `values` with each term rounded to `places` decimals, to the
/// nearest with ties to even, and the terms added exactly as integers, so the
/// result does not depend on their order.
///
/// `None` when `places` is above 22, a term is NaN or infinite, or a term or
/// the total has a magnitude of `10^(38 - places)` or more. An empty sum is 0.
pub fn fixed_point_sum(values: impl IntoIterator<Item = f64>, places: u32) -> Option<f64> {
    let scale = scale_for(places)?;
    let mut total = 0i128;
    for value in values {
        total = total.checked_add(fixed_point(value, scale)?)?;
    }
    total_to_f64(total, scale)
}

/// The rows with a positive weight, as values and weights, or `None` for the
/// refusals above. Without weights every row stays and `weights` is empty.
fn usable(values: &[f64], weights: Option<&[f64]>) -> Option<(Vec<f64>, Vec<f64>)> {
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let Some(weights) = weights else {
        return Some((values.to_vec(), Vec::new()));
    };
    if weights.len() != values.len() || weights.iter().any(|w| !w.is_finite()) {
        return None;
    }
    let (x, w): (Vec<f64>, Vec<f64>) = values
        .iter()
        .zip(weights)
        .filter(|&(_, &w)| w > 0.0)
        .map(|(&x, &w)| (x, w))
        .unzip();
    Some((x, w))
}

/// The weighted mean `sum(w x) / sum(w)`, or the mean when `weights` is
/// `None`. Terms are summed by [`fixed_point_sum`] with
/// [`WEIGHTED_DECIMAL_PLACES`] decimals.
///
/// `None` for the cases in the module documentation.
#[must_use]
pub fn mean(values: &[f64], weights: Option<&[f64]>) -> Option<f64> {
    let (x, w) = usable(values, weights)?;
    if x.is_empty() {
        return None;
    }
    let (numerator, denominator) = if weights.is_some() {
        (
            fixed_point_sum(
                x.iter().zip(&w).map(|(&x, &w)| x * w),
                WEIGHTED_DECIMAL_PLACES,
            )?,
            fixed_point_sum(w.iter().copied(), WEIGHTED_DECIMAL_PLACES)?,
        )
    } else {
        (
            fixed_point_sum(x.iter().map(|&x| x * 1.0), WEIGHTED_DECIMAL_PLACES)?,
            x.len() as f64,
        )
    };
    (denominator > 0.0).then(|| numerator / denominator)
}

/// The variance with reliability weights, `sum(w (x - mean)^2) / (W - sum(w^2)
/// / W)`, or the sample variance when `weights` is `None`.
///
/// `None` for the cases in the module documentation.
#[must_use]
pub fn variance(values: &[f64], weights: Option<&[f64]>) -> Option<f64> {
    let (x, w) = usable(values, weights)?;
    if x.len() < 2 {
        return None;
    }
    let (total, squares, weighted_sum, weight_of) = if weights.is_some() {
        let products: Vec<f64> = x.iter().zip(&w).map(|(&x, &w)| x * w).collect();
        let squares: Vec<f64> = w.iter().map(|&w| w * w).collect();
        (blocked(&w), blocked(&squares), blocked(&products), w)
    } else {
        let n = x.len() as f64;
        let ones = vec![1.0; x.len()];
        let products: Vec<f64> = x.iter().map(|&x| x * 1.0).collect();
        (n, n, blocked(&products), ones)
    };
    if total <= 0.0 {
        return None;
    }
    let divisor = total - squares / total;
    if divisor <= 0.0 {
        return None;
    }
    let mean = weighted_sum / total;
    let deviations: Vec<f64> = x
        .iter()
        .zip(&weight_of)
        .map(|(&x, &w)| w * (x - mean).powi(2))
        .collect();
    Some(blocked(&deviations) / divisor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point_rounds_each_term_and_adds_exactly() {
        // 0.35 times 10 is 3.5 in a double, which rounds to 4 with ties to even.
        assert_eq!(
            fixed_point_sum([0.35], 1).unwrap().to_bits(),
            0.4_f64.to_bits()
        );
        assert_eq!(
            fixed_point_sum([0.1; 10], 15).unwrap().to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(fixed_point_sum([], 6).unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!(fixed_point_sum([1.0], 23), None);
        assert_eq!(fixed_point_sum([f64::NAN], 6), None);
        assert_eq!(fixed_point_sum([1e33], 6), None);
        assert!(fixed_point_sum([1e31], 6).is_some());
    }

    #[test]
    fn variance_with_unit_weights_is_the_sample_variance() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let ones = [1.0; 8];
        assert_eq!(
            variance(&x, None).unwrap().to_bits(),
            variance(&x, Some(&ones)).unwrap().to_bits()
        );
        assert_eq!(
            variance(&x, None).unwrap().to_bits(),
            (32.0_f64 / 7.0).to_bits()
        );
    }

    #[test]
    fn the_scale_of_the_weights_does_not_matter() {
        let x = [1.5, 2.25, 4.0, 7.5, 3.125, 9.0];
        let w = [1.0, 2.5, 0.5, 3.0, 1.25, 2.0];
        let scaled: Vec<f64> = w.iter().map(|w| w * 8.0).collect();
        assert_eq!(
            variance(&x, Some(&w)).unwrap().to_bits(),
            variance(&x, Some(&scaled)).unwrap().to_bits()
        );
        assert_eq!(
            mean(&x, Some(&w)).unwrap().to_bits(),
            mean(&x, Some(&scaled)).unwrap().to_bits()
        );
    }

    #[test]
    fn r_agrees_on_a_small_sample() {
        // R: cov.wt(cbind(x), wt = w / sum(w))$cov and weighted.mean(x, w).
        let x = [1.5, 2.25, 4.0, 7.5, 3.125, 9.0];
        let w = [1.0, 2.5, 0.5, 3.0, 1.25, 2.0];
        let v = variance(&x, Some(&w)).unwrap();
        assert!((v - 10.656_673_569_277_109).abs() <= 1e-14 * v);
        let m = mean(&x, Some(&w)).unwrap();
        assert!((m - 5.222_560_975_609_756).abs() <= 1e-15 * m);
    }

    #[test]
    fn non_positive_weights_drop_the_row_and_bad_input_is_none() {
        let x = [1.0, 2.0, 3.0, 100.0];
        let w = [1.0, 1.0, 1.0, 0.0];
        assert_eq!(mean(&x, Some(&w)).unwrap().to_bits(), 2.0_f64.to_bits());
        assert_eq!(variance(&x, Some(&w)).unwrap().to_bits(), 1.0_f64.to_bits());
        assert_eq!(mean(&x, Some(&w[..3])), None);
        assert_eq!(mean(&[f64::NAN], None), None);
        assert_eq!(mean(&[1.0], Some(&[f64::INFINITY])), None);
        assert_eq!(mean(&[], None), None);
        assert_eq!(mean(&[1.0], Some(&[0.0])), None);
        assert_eq!(variance(&[1.0], None), None);
        assert_eq!(variance(&[1.0, 2.0], Some(&[1.0, 0.0])), None);
    }

    /// The values are what `weighted_mean` and `weighted_variance` of the
    /// `dansk_register` Python package return (polars 1.41.2) for
    /// `x_i = ((7919 i + 13) mod 10007) / 97` and
    /// `w_i = 0.25 + ((104729 i) mod 1009) / 403`, as the bits of the weighted
    /// mean, the unweighted mean, the weighted variance and the unweighted
    /// variance. `_weighted_variance` of `balance.py` gives the same bits.
    #[test]
    fn the_python_package_gives_the_same_bits() {
        let cases: [(usize, [u64; 4]); 8] = [
            (
                2,
                [
                    0x4052_64E9_4087_BFA0,
                    0x4044_7A0F_D5C5_F02A,
                    0x40AA_08F4_76BC_5570,
                    0x40AA_08F4_76BC_556D,
                ],
            ),
            (
                3,
                [
                    0x4051_079F_8408_D396,
                    0x4047_B143_BD24_1198,
                    0x4086_C7D0_0280_9A72,
                    0x409B_F94A_3B83_5098,
                ],
            ),
            (
                127,
                [
                    0x404A_9E47_4899_8D87,
                    0x4049_E1EE_163C_4E63,
                    0x4085_DA89_7DC2_A0A8,
                    0x408C_4443_FBF1_69C2,
                ],
            ),
            (
                128,
                [
                    0x404A_9BD9_177B_E13F,
                    0x4049_E1FA_B8BE_0548,
                    0x4085_8F9E_F211_A5CD,
                    0x408C_0B49_88F9_D87B,
                ],
            ),
            (
                129,
                [
                    0x404A_7CAF_D0B7_B110,
                    0x4049_CCAB_47CA_2714,
                    0x4085_8120_90FF_3F54,
                    0x408B_EFCD_CCFC_6F03,
                ],
            ),
            (
                300,
                [
                    0x404A_2CF2_DED2_4659,
                    0x4049_D604_FF30_CEA1,
                    0x408B_C608_BC97_215C,
                    0x408B_E6C6_42F0_F6F0,
                ],
            ),
            (
                1000,
                [
                    0x4049_E4CE_AB9E_20BB,
                    0x4049_D334_8D1F_3A0B,
                    0x408A_EEB3_0650_1F45,
                    0x408B_C500_683B_7D00,
                ],
            ),
            (
                5000,
                [
                    0x4049_CC0C_FC4B_51E8,
                    0x4049_CA2A_A853_118D,
                    0x408B_8E1F_289A_7EED,
                    0x408B_BBB8_8611_2E6F,
                ],
            ),
        ];
        for (n, expected) in cases {
            let x: Vec<f64> = (0..n)
                .map(|i| ((i * 7919 + 13) % 10007) as f64 / 97.0)
                .collect();
            let w: Vec<f64> = (0..n)
                .map(|i| 0.25 + ((i * 104_729) % 1009) as f64 / 403.0)
                .collect();
            let got = [
                mean(&x, Some(&w)).unwrap(),
                mean(&x, None).unwrap(),
                variance(&x, Some(&w)).unwrap(),
                variance(&x, None).unwrap(),
            ];
            for (g, e) in got.iter().zip(expected) {
                assert_eq!(g.to_bits(), e, "n = {n}");
            }
        }
    }
}
