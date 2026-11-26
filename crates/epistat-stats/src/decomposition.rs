//! Deprecated: this module moved to `epistat_epi::decomposition` in 0.3.0, where its
//! documentation, explainer and tests are. The copy here keeps the items and
//! the results of 0.2 for one release and is removed in 0.4.0.

#![expect(deprecated, reason = "the copy of 0.2 calls its own deprecated items")]

use crate::sum::exact;

/// A gap split into its two parts.
#[derive(Debug, Clone, Copy, PartialEq)]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::decomposition::Decomposition`"
)]
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
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::decomposition::DecompositionParts`"
)]
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
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::decomposition::shares`"
)]
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
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::decomposition::kitagawa`"
)]
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
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::decomposition::kitagawa_parts`"
)]
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
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::decomposition::kitagawa_symmetric`"
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
