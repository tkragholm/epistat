//! The weighted mean and the variance with reliability weights, the effective
//! sample size of a set of weights, and limits on weights and probabilities.
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
//! # Effective sample size and limits
//!
//! [`effective_sample_size`] is Kish's `(sum w_i)^2 / sum w_i^2` \[3\], which
//! does not depend on the scale of the weights. The sums run in the order
//! given; when the squares would overflow or lose precision, the weights are
//! divided by the largest first.
//!
//! [`WeightLimits`] cuts weights at fixed values or at R's type 1 quantiles of
//! the weights, as `TrialEmulation`'s `limit_weight` does, and
//! [`ProbabilityBounds`] keeps a fitted probability, such as a propensity
//! score, inside `[lower, upper]` before it is inverted into a weight. Both
//! replace a value below the lower limit by the limit and one above the upper
//! by that, `w.max(lower).min(upper)`.
//!
//! # References
//!
//! 1. Cochran WG. Sampling Techniques. 3rd ed. New York: Wiley; 1977.
//!    Section 2.
//! 2. R Core Team. `cov.wt` in the R package stats, method `"unbiased"`.
//! 3. Kish L. Survey Sampling. New York: Wiley; 1965, p. 427.

use epistat_core::InputError;

use crate::Error;
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

/// Kish's effective sample size, `(sum w)^2 / sum w^2`, of the weights.
///
/// The value does not depend on the scale of the weights: equal weights of any
/// size give their number. A zero weight is a unit that is not in the sample
/// and counts for nothing. The result is `None` when no weight is positive,
/// since there is no sample to have a size.
///
/// # Errors
///
/// [`Error::Input`] for the first weight that is negative or not finite.
pub fn effective_sample_size(weights: &[f64]) -> Result<Option<f64>, Error> {
    let mut largest = 0.0_f64;
    for (row, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() {
            return Err(InputError::NotFinite {
                what: "weights",
                row,
                column: None,
            }
            .into());
        }
        if weight < 0.0 {
            return Err(InputError::Domain {
                what: "weights",
                value: weight,
                domain: "[0, inf)",
                row: None,
            }
            .into());
        }
        largest = largest.max(weight);
    }
    if largest == 0.0 {
        return Ok(None);
    }
    let sums = |scale: f64| {
        weights
            .iter()
            .fold((0.0_f64, 0.0_f64), |(sum, sum_sq), &weight| {
                let weight = weight / scale;
                (sum + weight, sum_sq + weight * weight)
            })
    };
    let (mut sum, mut sum_sq) = sums(1.0);
    // The squares are accurate while the largest one is a normal number, and
    // the sums are finite while they have not overflowed.
    let plain = largest * largest >= f64::MIN_POSITIVE
        && sum.is_finite()
        && sum_sq.is_finite()
        && (sum * sum).is_finite();
    if !plain {
        (sum, sum_sq) = sums(largest);
    }
    Ok(Some((sum * sum) / sum_sq))
}

/// Where weights are cut.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightLimits {
    /// At the `lower` and `upper` quantiles of the weights, R's type 1 (the
    /// inverse of the empirical distribution function), as `TrialEmulation`
    /// takes them. [`WeightLimits::P99`] is its `"p99"`.
    Quantiles {
        /// The probability of the lower quantile.
        lower: f64,
        /// The probability of the upper quantile.
        upper: f64,
    },
    /// At fixed values, as its `"weight_limits"`.
    Values {
        /// The smallest weight kept.
        lower: f64,
        /// The largest weight kept.
        upper: f64,
    },
}

impl WeightLimits {
    /// The 1st and 99th percentiles.
    pub const P99: Self = Self::Quantiles {
        lower: 0.01,
        upper: 0.99,
    };

    /// The weights cut to the limits: each weight below the lower limit set
    /// to it and each above the upper set to it.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for no weights, a weight that is negative or not
    /// finite, quantile probabilities outside `[0, 1]` or limits with the
    /// lower above the upper.
    pub fn apply(self, weights: &[f64]) -> Result<Vec<f64>, Error> {
        if weights.is_empty() {
            return Err(InputError::Empty { what: "weights" }.into());
        }
        if let Some(row) = weights.iter().position(|w| !w.is_finite()) {
            return Err(InputError::NotFinite {
                what: "weights",
                row,
                column: None,
            }
            .into());
        }
        if let Some(&value) = weights.iter().find(|w| **w < 0.0) {
            return Err(InputError::Domain {
                what: "weights",
                value,
                domain: "[0, inf)",
                row: None,
            }
            .into());
        }
        let (lower, upper) = match self {
            Self::Quantiles { lower, upper } => {
                ordered("limits", lower, upper)?;
                for p in [lower, upper] {
                    epistat_core::check::probability("limits", p)?;
                }
                let mut sorted = weights.to_vec();
                sorted.sort_by(f64::total_cmp);
                let at = |p| {
                    crate::quantile::type1(&sorted, p).ok_or(InputError::Empty { what: "weights" })
                };
                (at(lower)?, at(upper)?)
            }
            Self::Values { lower, upper } => {
                ordered("limits", lower, upper)?;
                (lower, upper)
            }
        };
        Ok(weights.iter().map(|&w| w.max(lower).min(upper)).collect())
    }
}

/// `lower <= upper`, neither NaN.
fn ordered(what: &'static str, lower: f64, upper: f64) -> Result<(), InputError> {
    if lower.is_nan() || upper.is_nan() || lower > upper {
        return Err(InputError::Domain {
            what,
            value: lower,
            domain: "at most the upper limit",
            row: None,
        });
    }
    Ok(())
}

/// Bounds a fitted probability is kept within, such as `[0.01, 0.99]` for a
/// propensity score before it is inverted into a weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityBounds {
    lower: f64,
    upper: f64,
}

impl ProbabilityBounds {
    /// The bounds `[lower, upper]`.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] unless `0 <= lower <= upper <= 1`.
    pub fn new(lower: f64, upper: f64) -> Result<Self, Error> {
        epistat_core::check::probability("lower bound", lower)?;
        epistat_core::check::probability("upper bound", upper)?;
        ordered("lower bound", lower, upper)?;
        Ok(Self { lower, upper })
    }

    /// The lower bound.
    #[must_use]
    pub const fn lower(self) -> f64 {
        self.lower
    }

    /// The upper bound.
    #[must_use]
    pub const fn upper(self) -> f64 {
        self.upper
    }

    /// `p` moved into the bounds, `p.max(lower).min(upper)`.
    #[must_use]
    pub fn apply(self, p: f64) -> f64 {
        p.max(self.lower).min(self.upper)
    }
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

    #[test]
    fn kish_effective_sample_size() {
        assert_eq!(effective_sample_size(&[2.0; 5]), Ok(Some(5.0)));
        assert_eq!(effective_sample_size(&[0.0, 0.0]), Ok(None));
        assert_eq!(effective_sample_size(&[]), Ok(None));
        // (1 + 3)^2 / (1 + 9).
        assert_eq!(effective_sample_size(&[1.0, 3.0, 0.0]), Ok(Some(1.6)));
        // Squares that underflow are scaled first.
        assert_eq!(effective_sample_size(&[1e-200, 1e-200]), Ok(Some(2.0)));
        assert!(effective_sample_size(&[1.0, -1.0]).is_err());
        assert!(effective_sample_size(&[1.0, f64::NAN]).is_err());
    }

    #[test]
    fn weight_limits_cut_at_values_and_quantiles() {
        let w = WeightLimits::Values {
            lower: 1.0,
            upper: 3.0,
        }
        .apply(&[5.0, 0.5, 2.0])
        .unwrap();
        assert_eq!(w, vec![3.0, 1.0, 2.0]);
        // quantile(1:10, c(0.1, 0.99), type = 1) in R is 1 and 10.
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        let cut = WeightLimits::Quantiles {
            lower: 0.15,
            upper: 0.85,
        }
        .apply(&x)
        .unwrap();
        assert_eq!(cut[0], 2.0);
        assert_eq!(cut[9], 9.0);
        assert_eq!(WeightLimits::P99.apply(&x).unwrap(), x);
        assert!(
            WeightLimits::Values {
                lower: 2.0,
                upper: 1.0
            }
            .apply(&[1.0])
            .is_err()
        );
        assert!(WeightLimits::P99.apply(&[]).is_err());
        assert!(WeightLimits::P99.apply(&[-1.0]).is_err());
        assert!(
            WeightLimits::Quantiles {
                lower: -0.1,
                upper: 0.5
            }
            .apply(&[1.0])
            .is_err()
        );
    }

    #[test]
    fn probability_bounds_clamp() {
        let b = ProbabilityBounds::new(0.01, 0.99).unwrap();
        assert_eq!(b.apply(0.0), 0.01);
        assert_eq!(b.apply(0.5), 0.5);
        assert_eq!(b.apply(1.0), 0.99);
        assert_eq!((b.lower(), b.upper()), (0.01, 0.99));
        assert!(ProbabilityBounds::new(0.6, 0.4).is_err());
        assert!(ProbabilityBounds::new(-0.1, 0.4).is_err());
        assert!(ProbabilityBounds::new(0.1, f64::NAN).is_err());
    }
}
