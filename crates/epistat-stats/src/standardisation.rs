//! Deprecated: this module moved to `epistat_epi::standardisation` in 0.3.0, where its
//! documentation, explainer and tests are. The copy here keeps the items and
//! the results of 0.2 for one release and is removed in 0.4.0.

#![expect(deprecated, reason = "the copy of 0.2 calls its own deprecated items")]

use crate::sum::exact;

/// The weighted counts of the comparators and the exposed in each stratum, as
/// `[comparators, exposed]`.
///
/// `None` for the cases in the module documentation.
#[must_use]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::standardisation::stratum_counts`"
)]
pub fn stratum_counts(
    stratum: &[u32],
    exposed: &[bool],
    weights: &[f64],
    n_strata: usize,
) -> Option<Vec<[f64; 2]>> {
    if stratum.len() != exposed.len() || stratum.len() != weights.len() {
        return None;
    }
    let mut terms: Vec<[Vec<f64>; 2]> = vec![[Vec::new(), Vec::new()]; n_strata];
    for ((&s, &is_exposed), &g) in stratum.iter().zip(exposed).zip(weights) {
        if !g.is_finite() {
            return None;
        }
        terms.get_mut(s as usize)?[usize::from(is_exposed)].push(g);
    }
    Some(
        terms
            .into_iter()
            .map(|[c, e]| [exact(c), exact(e)])
            .collect(),
    )
}

/// The weight of each person after standardisation to the exposed mix:
/// `g` for the exposed and `g n_E(s) / n_C(s)` for a comparator, 0 where
/// `n_C(s)` is not positive.
///
/// `None` for the cases in the module documentation.
#[must_use]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::standardisation::standardisation_weights`"
)]
pub fn standardisation_weights(
    stratum: &[u32],
    exposed: &[bool],
    weights: &[f64],
    n_strata: usize,
) -> Option<Vec<f64>> {
    let counts = stratum_counts(stratum, exposed, weights, n_strata)?;
    Some(
        stratum
            .iter()
            .zip(exposed)
            .zip(weights)
            .map(|((&s, &is_exposed), &g)| {
                let [comparators, exposed_count] = counts[s as usize];
                if is_exposed {
                    g
                } else if comparators > 0.0 {
                    g * exposed_count / comparators
                } else {
                    0.0
                }
            })
            .collect(),
    )
}
