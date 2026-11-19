//! The Kitagawa decomposition of a difference between two weighted means into
//! a composition part and a rate part.
//!
//! # Algorithm
//!
//! Two populations have the same strata. In stratum `i`, population A has the
//! share `a_i` of its weight and the rate `r_i`, and population B has the share
//! `b_i` and the rate `q_i`. The shares of each population sum to 1. The gap in
//! the overall rate is
//!
//! ```text
//! gap = sum(a_i r_i) - sum(b_i q_i).
//! ```
//!
//! [`kitagawa`] takes B as the reference. The composition part is the
//! change in shares at the reference rates, and the rate part is the change in
//! rates at A's shares:
//!
//! ```text
//! composition = sum((a_i - b_i) q_i),
//! rate        = sum(a_i (r_i - q_i)).
//! ```
//!
//! The two add to the gap exactly, since `sum((a_i - b_i) q_i) + sum(a_i (r_i -
//! q_i)) = sum(a_i r_i) - sum(b_i q_i)`. This is the decomposition of Kitagawa
//! \[1\] with the rates of B as the standard. [`kitagawa_symmetric`] weights
//! both parts at the average of the two populations,
//!
//! ```text
//! composition = sum((a_i - b_i) (r_i + q_i) / 2),
//! rate        = sum((r_i - q_i) (a_i + b_i) / 2),
//! ```
//!
//! which also add to the gap and do not depend on which population is the
//! reference \[1\], \[2\]. The choice between them changes the split, not the gap.
//! [`shares`] turns weights into shares.
//!
//! # Summation
//!
//! Each sum is the correctly rounded sum of its terms ([`crate::sum::exact`]),
//! so it does not depend on the order of the strata. In [`kitagawa`] a term with
//! a share of exactly 0 is left out, even where its rate is NaN, since a
//! stratum without members has no rate. Any other term that is not finite makes
//! the result `None`.
//!
//! # Edge cases
//!
//! `None` when the four slices differ in length, are empty, or a term used is
//! NaN or infinite. [`shares`] returns `None` when the weights do not have a
//! positive finite total, or one is NaN or negative.
//!
//! # Agreement with other software
//!
//! The tests check hand-computed examples, the exact addition of the parts to the
//! gap, and the reference decomposition of the PHD equity kernel
//! (`kernels::equity`), which forms the same sums.
//!
//! # References
//!
//! 1. Kitagawa EM. Components of a difference between two rates. J Am Stat
//!    Assoc 1955;50:1168-1194. doi:10.1080/01621459.1955.10501299
//! 2. Das Gupta P. Standardization and Decomposition of Rates: A User's Manual.
//!    Washington: US Bureau of the Census; 1993. Current Population Reports,
//!    P23-186.

use crate::sum::exact;

/// A gap split into its two parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decomposition {
    /// The overall rate of A minus that of B.
    pub gap: f64,
    /// The part from the difference in shares.
    pub composition: f64,
    /// The part from the difference in rates.
    pub rate: f64,
}

/// The sum of `left * right` over the strata, skipping a zero `left`, or
/// `None` when a term that is used is not finite.
fn sum_products(left: &[f64], right: &[f64]) -> Option<f64> {
    let mut terms = Vec::with_capacity(left.len());
    for (&a, &b) in left.iter().zip(right) {
        if a == 0.0 {
            continue;
        }
        let term = a * b;
        if !term.is_finite() {
            return None;
        }
        terms.push(term);
    }
    Some(exact(terms))
}

/// The shares `w_i / sum(w)` of non-negative weights, the total taken as a
/// correctly rounded sum. `None` for the cases in the module documentation.
#[must_use]
pub fn shares(weights: &[f64]) -> Option<Vec<f64>> {
    if weights.iter().any(|&w| !(w >= 0.0 && w.is_finite())) {
        return None;
    }
    let total = exact(weights.iter().copied());
    (total > 0.0 && total.is_finite()).then(|| weights.iter().map(|&w| w / total).collect())
}

fn same_length(slices: [&[f64]; 4]) -> bool {
    let n = slices[0].len();
    n > 0 && slices.iter().all(|s| s.len() == n)
}

/// The decomposition of the gap between A and the reference B, with B's rates
/// as the standard: `composition = sum((a - b) q)` and `rate = sum(a (r -
/// q))`. The gap is `sum(a r) - sum(b q)`.
///
/// `None` for the cases in the module documentation.
#[must_use]
pub fn kitagawa(
    shares: &[f64],
    rates: &[f64],
    reference_shares: &[f64],
    reference_rates: &[f64],
) -> Option<Decomposition> {
    if !same_length([shares, rates, reference_shares, reference_rates]) {
        return None;
    }
    let own = sum_products(shares, rates)?;
    let reference = sum_products(reference_shares, reference_rates)?;
    let share_gap: Vec<f64> = shares
        .iter()
        .zip(reference_shares)
        .map(|(a, b)| a - b)
        .collect();
    let rate_gap: Vec<f64> = rates
        .iter()
        .zip(reference_rates)
        .map(|(a, b)| a - b)
        .collect();
    Some(Decomposition {
        gap: own - reference,
        composition: sum_products(&share_gap, reference_rates)?,
        rate: sum_products(shares, &rate_gap)?,
    })
}

/// The decomposition with both parts weighted at the average of A and B:
/// `composition = sum((a - b) (r + q) / 2)` and `rate = sum((r - q) (a + b) /
/// 2)`. The gap is `sum(a r) - sum(b q)`.
///
/// `None` when the slices differ in length or are empty, or any term is not
/// finite.
#[must_use]
#[expect(
    clippy::manual_midpoint,
    reason = "the average is (x + y) / 2, as in the Python it reproduces"
)]
pub fn kitagawa_symmetric(
    shares: &[f64],
    rates: &[f64],
    reference_shares: &[f64],
    reference_rates: &[f64],
) -> Option<Decomposition> {
    if !same_length([shares, rates, reference_shares, reference_rates]) {
        return None;
    }
    let n = shares.len();
    let mut own = Vec::with_capacity(n);
    let mut reference = Vec::with_capacity(n);
    let mut composition = Vec::with_capacity(n);
    let mut rate = Vec::with_capacity(n);
    for i in 0..n {
        let (a, r, b, q) = (shares[i], rates[i], reference_shares[i], reference_rates[i]);
        own.push(a * r);
        reference.push(b * q);
        composition.push((a - b) * ((r + q) / 2.0));
        rate.push((r - q) * ((a + b) / 2.0));
    }
    let finite = |terms: &[f64]| terms.iter().all(|t| t.is_finite());
    if !(finite(&own) && finite(&reference) && finite(&composition) && finite(&rate)) {
        return None;
    }
    Some(Decomposition {
        gap: exact(own) - exact(reference),
        composition: exact(composition),
        rate: exact(rate),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Binary fractions, so every value below is exact.
    const A: [f64; 2] = [0.75, 0.25];
    const R: [f64; 2] = [8.0, 16.0];
    const B: [f64; 2] = [0.5, 0.5];
    const Q: [f64; 2] = [4.0, 12.0];

    #[test]
    fn hand_computed_reference_decomposition() {
        // sum(a r) = 10, sum(b q) = 8, composition = 0.25 * 4 - 0.25 * 12,
        // rate = 0.75 * 4 + 0.25 * 4.
        let d = kitagawa(&A, &R, &B, &Q).unwrap();
        assert_eq!(d.gap.to_bits(), 2.0_f64.to_bits());
        assert_eq!(d.composition.to_bits(), (-2.0_f64).to_bits());
        assert_eq!(d.rate.to_bits(), 4.0_f64.to_bits());
    }

    #[test]
    fn hand_computed_symmetric_decomposition() {
        // composition = 0.25 * 6 - 0.25 * 14, rate = 4 * 0.625 + 4 * 0.375.
        let d = kitagawa_symmetric(&A, &R, &B, &Q).unwrap();
        assert_eq!(d.gap.to_bits(), 2.0_f64.to_bits());
        assert_eq!(d.composition.to_bits(), (-2.0_f64).to_bits());
        assert_eq!(d.rate.to_bits(), 4.0_f64.to_bits());
        // Swapping the populations swaps the signs of the parts.
        let swapped = kitagawa_symmetric(&B, &Q, &A, &R).unwrap();
        assert_eq!(swapped.composition.to_bits(), 2.0_f64.to_bits());
        assert_eq!(swapped.rate.to_bits(), (-4.0_f64).to_bits());
    }

    #[test]
    fn the_parts_add_to_the_gap() {
        let a = shares(&[3.0, 5.0, 2.0, 9.0]).unwrap();
        let b = shares(&[4.0, 1.0, 7.0, 1.0]).unwrap();
        let r = [10.5, 3.25, 8.0, 1.5];
        let q = [9.0, 4.5, 7.25, 2.0];
        for d in [
            kitagawa(&a, &r, &b, &q).unwrap(),
            kitagawa_symmetric(&a, &r, &b, &q).unwrap(),
        ] {
            assert!((d.composition + d.rate - d.gap).abs() <= 1e-14 * d.gap.abs().max(1.0));
        }
    }

    #[test]
    fn a_stratum_without_members_has_no_rate() {
        let a = [0.5, 0.5, 0.0];
        let r = [8.0, 16.0, f64::NAN];
        let b = [0.25, 0.75, 0.0];
        let q = [4.0, 12.0, f64::NAN];
        let d = kitagawa(&a, &r, &b, &q).unwrap();
        // 0.25 * 4 + (-0.25) * 12 and 0.5 * 4 + 0.5 * 4.
        assert_eq!(d.composition.to_bits(), (-2.0_f64).to_bits());
        assert_eq!(d.rate.to_bits(), 4.0_f64.to_bits());
        assert_eq!(kitagawa(&[0.5, 0.5, 0.1], &r, &b, &q), None);
        assert_eq!(kitagawa_symmetric(&a, &r, &b, &q), None);
    }

    /// The sums of `estimates` in `kernels/src/equity.rs`, for one level against
    /// the reference: `sum_products` with `math.fsum`, then the gap and the two
    /// parts.
    fn kernel_copy(
        shares: &[f64],
        excess: &[f64],
        ref_shares: &[f64],
        ref_excess: &[f64],
    ) -> [f64; 3] {
        fn sum_products(left: &[f64], right: &[f64]) -> f64 {
            let mut terms = Vec::with_capacity(left.len());
            for (&a, &b) in left.iter().zip(right) {
                if a == 0.0 {
                    continue;
                }
                terms.push(a * b);
            }
            if terms.iter().all(|t| t.is_finite()) {
                exact(terms)
            } else {
                f64::NAN
            }
        }
        let own = sum_products(shares, excess);
        let own_ref = sum_products(ref_shares, ref_excess);
        let share_gap: Vec<f64> = shares.iter().zip(ref_shares).map(|(a, b)| a - b).collect();
        let excess_gap: Vec<f64> = excess.iter().zip(ref_excess).map(|(a, b)| a - b).collect();
        [
            own - own_ref,
            sum_products(&share_gap, ref_excess),
            sum_products(shares, &excess_gap),
        ]
    }

    #[test]
    fn the_equity_kernel_gives_the_same_bits() {
        let a = shares(&[3.0, 5.0, 0.0, 9.0, 2.5]).unwrap();
        let b = shares(&[4.0, 1.0, 7.0, 1.0, 0.0]).unwrap();
        let r = [10.5, 3.3, f64::NAN, 1.7, 8.1];
        let q = [9.1, 4.5, 7.25, 2.0, f64::NAN];
        // A has no members in stratum 2 and B none in stratum 4, so each has a
        // NaN rate where the other's share is not zero: the product is NaN.
        assert_eq!(kitagawa(&a, &r, &b, &q), None);
        let a = shares(&[3.0, 5.0, 2.0, 9.0, 2.5]).unwrap();
        let b = shares(&[4.0, 1.0, 7.0, 1.0, 6.1]).unwrap();
        let r = [10.5, 3.3, 8.3, 1.7, 8.1];
        let q = [9.1, 4.5, 7.25, 2.0, 0.3];
        let d = kitagawa(&a, &r, &b, &q).unwrap();
        let reference = kernel_copy(&a, &r, &b, &q);
        assert_eq!(d.gap.to_bits(), reference[0].to_bits());
        assert_eq!(d.composition.to_bits(), reference[1].to_bits());
        assert_eq!(d.rate.to_bits(), reference[2].to_bits());
    }

    #[test]
    fn refusals_are_none() {
        assert_eq!(kitagawa(&A, &R, &B, &Q[..1]), None);
        assert_eq!(kitagawa(&[], &[], &[], &[]), None);
        assert_eq!(shares(&[0.0, 0.0]), None);
        assert_eq!(shares(&[1.0, -1.0]), None);
        assert_eq!(shares(&[1.0, f64::NAN]), None);
        assert_eq!(shares(&[1.0, 3.0]).unwrap(), [0.25, 0.75]);
    }
}
