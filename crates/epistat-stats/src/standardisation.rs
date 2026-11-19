//! Direct standardisation of a comparator group to the stratum mix of an
//! exposed group.
//!
//! # Algorithm
//!
//! Persons belong to a stratum `s` and are exposed or comparators. Each has a
//! weight `g`, for example the draw of a cluster bootstrap, or 1. The weighted
//! counts of stratum `s` are
//!
//! ```text
//! n_E(s) = sum of g over the exposed in s,    n_C(s) = sum of g over the comparators in s.
//! ```
//!
//! [`stratum_counts`] forms them. [`standardisation_weights`] gives an
//! exposed person the weight `g` and a comparator the weight
//!
//! ```text
//! g n_E(s) / n_C(s)    if n_C(s) > 0,    0 otherwise,
//! ```
//!
//! so that the weighted comparators have the exposed group's distribution over
//! the strata \[1\], \[2\]. The product `g n_E` is formed before the division. A
//! stratum without comparator weight gives its comparators weight 0 and the
//! exposed of that stratum have no counterpart. The result is
//! the weights of the exposed mix, and the comparison of weighted means is
//! the effect of exposure standardised to that mix.
//!
//! # Summation
//!
//! The counts are added as fixed-point integers by
//! [`crate::weighted::fixed_point_sum`]'s rule: each term is rounded to
//! `places` decimals, to the nearest with ties to even, and the integers are
//! added, so the counts do not depend on the order of the persons. This
//! is the arithmetic of the weight bootstrap of the PHD kernels, where the
//! counts are recomputed in every replicate.
//!
//! # Edge cases
//!
//! `None` when the slices differ in length, a stratum is `n_strata` or more,
//! `places` is above 22, or a weight is NaN or infinite or too large for the
//! fixed-point sum.
//!
//! # References
//!
//! 1. Sato T, Matsuyama Y. Marginal structural models as a tool for
//!    standardization. Epidemiology 2003;14:680-686.
//! 2. Rothman KJ, Greenland S, Lash TL. Modern Epidemiology. 3rd ed.
//!    Philadelphia: Lippincott Williams and Wilkins; 2008. Chapter 3.

use crate::weighted::{fixed_point, scale_for, total_to_f64};

/// The weighted counts of the comparators and the exposed in each stratum, as
/// `[comparators, exposed]`.
///
/// `None` for the cases in the module documentation.
#[must_use]
pub fn stratum_counts(
    stratum: &[u32],
    exposed: &[bool],
    weights: &[f64],
    n_strata: usize,
    places: u32,
) -> Option<Vec<[f64; 2]>> {
    let scale = scale_for(places)?;
    if stratum.len() != exposed.len() || stratum.len() != weights.len() {
        return None;
    }
    let mut totals = vec![[0i128; 2]; n_strata];
    for ((&s, &is_exposed), &g) in stratum.iter().zip(exposed).zip(weights) {
        let cell = &mut totals.get_mut(s as usize)?[usize::from(is_exposed)];
        *cell = cell.checked_add(fixed_point(g, scale)?)?;
    }
    totals
        .into_iter()
        .map(|[c, e]| Some([total_to_f64(c, scale)?, total_to_f64(e, scale)?]))
        .collect()
}

/// The weight of each person after standardisation to the exposed mix:
/// `g` for the exposed and `g n_E(s) / n_C(s)` for a comparator, 0 where
/// `n_C(s)` is not positive.
///
/// `None` for the cases in the module documentation.
#[must_use]
pub fn standardisation_weights(
    stratum: &[u32],
    exposed: &[bool],
    weights: &[f64],
    n_strata: usize,
    places: u32,
) -> Option<Vec<f64>> {
    let counts = stratum_counts(stratum, exposed, weights, n_strata, places)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    // Stratum 0: two exposed, four comparators. Stratum 1: three exposed,
    // one comparator. Stratum 2: one exposed, no comparator.
    const STRATUM: [u32; 11] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2];
    const EXPOSED: [bool; 11] = [
        true, true, false, false, false, false, true, true, true, false, true,
    ];

    #[test]
    fn counts_and_weights_follow_the_formula() {
        let ones = [1.0; 11];
        let counts = stratum_counts(&STRATUM, &EXPOSED, &ones, 3, 15).unwrap();
        assert_eq!(counts, [[4.0, 2.0], [1.0, 3.0], [0.0, 1.0]]);
        let weights = standardisation_weights(&STRATUM, &EXPOSED, &ones, 3, 15).unwrap();
        let expected: [f64; 11] = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 3.0, 1.0];
        assert_eq!(
            weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|w| w.to_bits()).collect::<Vec<_>>()
        );
        // The weighted comparators have the exposed counts in each stratum
        // that has comparators.
        let mut comparators = [0.0; 3];
        for ((&s, &e), &w) in STRATUM.iter().zip(&EXPOSED).zip(&weights) {
            if !e {
                comparators[s as usize] += w;
            }
        }
        assert_eq!(
            comparators.map(f64::to_bits),
            [2.0_f64, 3.0, 0.0].map(f64::to_bits)
        );
    }

    #[test]
    fn weights_multiply_before_the_division() {
        let g = [0.5, 0.25, 3.0, 0.125, 1.5, 2.0, 1.0, 1.0, 1.0, 0.75, 1.0];
        let counts = stratum_counts(&STRATUM, &EXPOSED, &g, 3, 15).unwrap();
        assert_eq!(
            counts[0].map(f64::to_bits),
            [6.625_f64, 0.75].map(f64::to_bits)
        );
        let weights = standardisation_weights(&STRATUM, &EXPOSED, &g, 3, 15).unwrap();
        assert_eq!(weights[2].to_bits(), (3.0 * 0.75 / 6.625_f64).to_bits());
    }

    #[test]
    fn refusals_are_none() {
        let ones = [1.0; 11];
        assert_eq!(stratum_counts(&STRATUM, &EXPOSED, &ones[..10], 3, 15), None);
        assert_eq!(stratum_counts(&STRATUM, &EXPOSED, &ones, 2, 15), None);
        assert_eq!(stratum_counts(&STRATUM, &EXPOSED, &ones, 3, 23), None);
        let mut bad = ones;
        bad[3] = f64::NAN;
        assert_eq!(
            standardisation_weights(&STRATUM, &EXPOSED, &bad, 3, 15),
            None
        );
    }
}
