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
//! [`kitagawa_parts`] forms the same three quantities as [`kitagawa`] but
//! reports each one on its own, so a quantity that is undefined does not take
//! the others with it.
//!
//! # Summation
//!
//! Each sum is the correctly rounded sum of its terms ([`crate::sum::exact`]),
//! so it does not depend on the order of the strata. In [`kitagawa`] a term with
//! a share of exactly 0 is left out, even where its rate is NaN, since a
//! stratum without members has no rate. Any other term that is not finite makes
//! the result `None`. [`kitagawa_parts`] applies the same rule to each part:
//! the gap needs the shares and rates of both populations, the composition part
//! needs the share gaps and the reference rates, and the rate part needs the
//! shares and the rate gaps. A part whose terms are all skipped or finite is
//! defined, even when another part is not. With reference rates that are all
//! NaN and shares equal to the reference shares, the composition part is 0
//! while the gap and the rate part are `None`.
//!
//! # Edge cases
//!
//! `None` when the four slices differ in length, are empty, or a term used is
//! NaN or infinite. [`kitagawa_parts`] returns all three parts `None` for
//! slices that differ in length or are empty, and otherwise `None` only for the
//! part with a term that is not finite. [`shares`] returns `None` when the weights do not have a
//! positive finite total, or one is NaN or negative.
//!
//! # Variances
//!
//! With the shares fixed, both parts are linear in the rates, so for
//! independent rates with variances `v_i` (A) and `u_i` (B) their variances
//! and covariance are exact sums, which [`kitagawa_variance`] forms:
//!
//! ```text
//! gap                     sum a^2 v + sum b^2 u
//! reference  composition  sum (a - b)^2 u,     rate  sum a^2 (v + u),
//!            covariance   -sum (a - b) a u
//! symmetric  composition  sum ((a - b)/2)^2 (v + u),   rate  sum ((a + b)/2)^2 (v + u),
//!            covariance   sum ((a - b)/2) ((a + b)/2) (v - u)
//! ```
//!
//! The variance of the composition part plus that of the rate part plus twice
//! their covariance is the variance of the gap. [`kitagawa_poisson`] takes
//! events and person-time by stratum: the shares are those of the person-time,
//! which is taken as fixed, and the counts are Poisson, so `v_i = d_i / y_i^2`
//! times the square of the scale, estimated from the counts. The intervals
//! are normal, `estimate -/+ z se`. Rates whose variance has another source,
//! such as a bootstrap or a binomial risk, go to [`kitagawa_variance`]
//! directly. Shares estimated from a sample add a term of their own, which
//! these variances leave out.
//!
//! # Agreement with other software
//!
//! The symmetric decomposition is Das Gupta's decomposition of a rate into an
//! age structure and a rate factor, and the tests check it against
//! `DasGuptR::dgnpop` with `crossclassified`. They check the decomposition
//! with a reference and the variances against the formulas written out in R,
//! the exact addition of the parts to the gap, and the reference
//! decomposition of the PHD equity kernel (`kernels::equity`), which forms
//! the same sums.
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

/// A gap split into its two parts, each `None` when its own terms are not
/// finite. See [`kitagawa_parts`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecompositionParts {
    /// The overall rate of A minus that of B.
    pub gap: Option<f64>,
    /// The part from the difference in shares.
    pub composition: Option<f64>,
    /// The part from the difference in rates.
    pub rate: Option<f64>,
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

/// The same decomposition as [`kitagawa`], with each of the gap, the
/// composition part and the rate part `None` on its own when a term it uses is
/// not finite. A term with a share or share gap of exactly 0 is left out.
///
/// Where [`kitagawa`] is `Some`, every part here is `Some` and equal to it bit
/// for bit. All three are `None` when the slices differ in length or are empty.
#[must_use]
pub fn kitagawa_parts(
    shares: &[f64],
    rates: &[f64],
    reference_shares: &[f64],
    reference_rates: &[f64],
) -> DecompositionParts {
    if !same_length([shares, rates, reference_shares, reference_rates]) {
        return DecompositionParts {
            gap: None,
            composition: None,
            rate: None,
        };
    }
    let gaps = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
    let gap = sum_products(shares, rates)
        .zip(sum_products(reference_shares, reference_rates))
        .map(|(own, reference)| own - reference);
    DecompositionParts {
        gap,
        composition: sum_products(&gaps(shares, reference_shares), reference_rates),
        rate: sum_products(shares, &gaps(rates, reference_rates)),
    }
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

/// Which split of a gap a variance is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// The rates of the reference as the standard, as [`kitagawa`].
    Reference,
    /// Both parts at the average of the two populations, as
    /// [`kitagawa_symmetric`].
    Symmetric,
}

/// The variances of the gap and the two parts of a decomposition, and the
/// covariance of the parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartVariances {
    /// The variance of the gap.
    pub gap: f64,
    /// The variance of the composition part.
    pub composition: f64,
    /// The variance of the rate part.
    pub rate: f64,
    /// The covariance of the composition part and the rate part.
    pub covariance: f64,
}

/// The sum of `coefficient * variance` over the strata, skipping a zero
/// coefficient, or `None` when a term used is not finite.
fn weighted_variances(coefficients: &[f64], variances: &[f64]) -> Option<f64> {
    sum_products(coefficients, variances)
}

/// The variances of the gap and of the parts of a decomposition with fixed
/// shares and independent rates whose variances are `variances` (A) and
/// `reference_variances` (B). The parts are linear in the rates, so with
/// `v_i` and `u_i` the variances of `r_i` and `q_i`,
///
/// ```text
/// gap                       sum a^2 v + sum b^2 u
/// Reference   composition   sum (a - b)^2 u
///             rate          sum a^2 (v + u)
///             covariance    -sum (a - b) a u
/// Symmetric   composition   sum ((a - b) / 2)^2 (v + u)
///             rate          sum ((a + b) / 2)^2 (v + u)
///             covariance    sum ((a - b) / 2) ((a + b) / 2) (v - u)
/// ```
///
/// A term with a coefficient of exactly 0 is left out, as in [`kitagawa`],
/// and the sums are correctly rounded. `None` when the slices differ in
/// length or are empty, a variance is negative, or a term used is not finite.
#[must_use]
#[expect(
    clippy::manual_midpoint,
    reason = "the average is (x + y) / 2, as in kitagawa_symmetric"
)]
pub fn kitagawa_variance(
    shares: &[f64],
    variances: &[f64],
    reference_shares: &[f64],
    reference_variances: &[f64],
    split: Split,
) -> Option<PartVariances> {
    if !same_length([shares, variances, reference_shares, reference_variances]) {
        return None;
    }
    if variances
        .iter()
        .chain(reference_variances)
        .any(|&v| v < 0.0)
    {
        return None;
    }
    let map = |f: &dyn Fn(f64, f64) -> f64| -> Vec<f64> {
        shares
            .iter()
            .zip(reference_shares)
            .map(|(&a, &b)| f(a, b))
            .collect()
    };
    let both: Vec<f64> = variances
        .iter()
        .zip(reference_variances)
        .map(|(v, u)| v + u)
        .collect();
    let gap = weighted_variances(&map(&|a, _| a * a), variances)?
        + weighted_variances(&map(&|_, b| b * b), reference_variances)?;
    let (composition, rate, covariance) = match split {
        Split::Reference => (
            weighted_variances(&map(&|a, b| (a - b) * (a - b)), reference_variances)?,
            weighted_variances(&map(&|a, _| a * a), &both)?,
            -weighted_variances(&map(&|a, b| (a - b) * a), reference_variances)?,
        ),
        Split::Symmetric => {
            let difference: Vec<f64> = variances
                .iter()
                .zip(reference_variances)
                .map(|(v, u)| v - u)
                .collect();
            let half_gap = |a: f64, b: f64| (a - b) / 2.0;
            let half_sum = |a: f64, b: f64| (a + b) / 2.0;
            (
                weighted_variances(&map(&|a, b| half_gap(a, b) * half_gap(a, b)), &both)?,
                weighted_variances(&map(&|a, b| half_sum(a, b) * half_sum(a, b)), &both)?,
                weighted_variances(&map(&|a, b| half_gap(a, b) * half_sum(a, b)), &difference)?,
            )
        }
    };
    Some(PartVariances {
        gap,
        composition,
        rate,
        covariance,
    })
}

/// One part of a decomposition with its standard error and normal interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Part {
    /// The estimate.
    pub estimate: f64,
    /// Its standard error.
    pub se: f64,
    /// The lower limit, `estimate - z se`.
    pub lower: f64,
    /// The upper limit, `estimate + z se`.
    pub upper: f64,
}

/// The decomposition of a gap between two sets of Poisson rates with
/// standard errors. See [`kitagawa_poisson`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonDecomposition {
    /// The crude rate of A minus that of B.
    pub gap: Part,
    /// The part from the difference in shares of person-time.
    pub composition: Part,
    /// The part from the difference in rates.
    pub rate: Part,
    /// The covariance of the composition part and the rate part.
    pub covariance: f64,
}

/// The Kitagawa decomposition of the gap in crude rates between A and the
/// reference B from events and person-time by stratum, with standard errors
/// and normal intervals at the coverage `level`.
///
/// The shares are each population's shares of its person-time, from
/// [`shares`], and the rates `scale d / y`, so the gap is the difference of
/// the crude rates times `scale`. The estimates are those of [`kitagawa`] or
/// [`kitagawa_symmetric`], bit for bit. The person-time is taken as fixed and
/// the counts as Poisson, so a rate has the variance `scale^2 d / y^2` and the
/// variances are those of [`kitagawa_variance`]; they are exact for the
/// linear parts and estimated from the counts, so a stratum without events
/// adds nothing to them.
///
/// `None` when the slices differ in length or are empty, a count or a
/// person-time is negative or not finite, a population has no person-time, a
/// stratum has events without person-time, `level` is outside `(0, 1)`,
/// `scale` is not positive and finite, or a rate that is used is undefined: a
/// stratum without person-time in one population and with it in the other
/// has no rate where one is needed.
#[must_use]
pub fn kitagawa_poisson(
    group: crate::rates::Strata<'_>,
    reference: crate::rates::Strata<'_>,
    split: Split,
    level: f64,
    scale: f64,
) -> Option<PoissonDecomposition> {
    let n = group.events.len();
    if n == 0
        || [group.time, reference.events, reference.time]
            .iter()
            .any(|s| s.len() != n)
    {
        return None;
    }
    if !(level > 0.0 && level < 1.0 && scale > 0.0 && scale.is_finite()) {
        return None;
    }
    let valid = |events: &[f64], time: &[f64]| {
        events.iter().zip(time).all(|(&d, &y)| {
            d.is_finite() && d >= 0.0 && y.is_finite() && y >= 0.0 && !(y == 0.0 && d > 0.0)
        })
    };
    if !(valid(group.events, group.time) && valid(reference.events, reference.time)) {
        return None;
    }
    let share_a = shares(group.time)?;
    let share_b = shares(reference.time)?;
    let rates = |events: &[f64], time: &[f64]| -> (Vec<f64>, Vec<f64>) {
        events
            .iter()
            .zip(time)
            .map(|(&d, &y)| (scale * d / y, scale * scale * d / (y * y)))
            .unzip()
    };
    let (rate_a, var_a) = rates(group.events, group.time);
    let (rate_b, var_b) = rates(reference.events, reference.time);
    let d = match split {
        Split::Reference => kitagawa(&share_a, &rate_a, &share_b, &rate_b)?,
        Split::Symmetric => kitagawa_symmetric(&share_a, &rate_a, &share_b, &rate_b)?,
    };
    let v = kitagawa_variance(&share_a, &var_a, &share_b, &var_b, split)?;
    let z = crate::normal::two_sided_z(level);
    let part = |estimate: f64, variance: f64| {
        let se = variance.sqrt();
        Part {
            estimate,
            se,
            lower: estimate - z * se,
            upper: estimate + z * se,
        }
    };
    Some(PoissonDecomposition {
        gap: part(d.gap, v.gap),
        composition: part(d.composition, v.composition),
        rate: part(d.rate, v.rate),
        covariance: v.covariance,
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
    fn parts_agree_with_the_whole_and_split_when_undefined() {
        let d = kitagawa(&A, &R, &B, &Q).unwrap();
        let p = kitagawa_parts(&A, &R, &B, &Q);
        assert_eq!(p.gap.unwrap().to_bits(), d.gap.to_bits());
        assert_eq!(p.composition.unwrap().to_bits(), d.composition.to_bits());
        assert_eq!(p.rate.unwrap().to_bits(), d.rate.to_bits());
        // No comparator weight: every rate is NaN. The reference level has a
        // share gap of exactly 0 in each stratum, so its composition part is 0.
        let nan = [f64::NAN; 2];
        let p = kitagawa_parts(&B, &nan, &B, &nan);
        assert_eq!(p.composition.unwrap().to_bits(), 0.0_f64.to_bits());
        assert_eq!((p.gap, p.rate), (None, None));
        // Another level: the composition part needs the reference rates too.
        let p = kitagawa_parts(&A, &nan, &B, &nan);
        assert_eq!(
            p,
            DecompositionParts {
                gap: None,
                composition: None,
                rate: None
            }
        );
        let p = kitagawa_parts(&A, &R, &B, &Q[..1]);
        assert_eq!(p.gap, None);
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
