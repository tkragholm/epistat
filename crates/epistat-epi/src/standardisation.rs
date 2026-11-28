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
//! Each count is the correctly rounded sum of its weights
//! ([`epistat_stats::sum::exact`]), so it does not depend on the order of the
//! persons, and a count of integer weights is exact.
//!
//! # Edge cases
//!
//! Columns of different lengths are [`InputError::Length`] and a stratum code
//! of `n_strata` or more is [`InputError::Domain`] at its row. The weights
//! are [`CaseWeights`], checked where they were built, and `None` stands for
//! a weight of 1 for every person. A count beyond the range of `f64` is
//! infinite, and a comparator of that stratum then has weight 0 or NaN.
//!
//! ```
//! use epistat_core::CaseWeights;
//! use epistat_epi::standardisation::standardisation_weights;
//!
//! // Stratum 0: one exposed, two comparators; stratum 1: two exposed, one comparator.
//! let stratum = [0, 0, 0, 1, 1, 1];
//! let exposed = [true, false, false, true, true, false];
//! let w = standardisation_weights(&stratum, &exposed, None, 2)?;
//! assert_eq!(w, [1.0, 0.5, 0.5, 1.0, 1.0, 2.0]);
//! let draws = [1.0, 2.0, 0.0, 1.0, 1.0, 1.0];
//! let w = standardisation_weights(&stratum, &exposed, Some(CaseWeights::frequency(&draws)?), 2)?;
//! assert_eq!(w, [1.0, 1.0, 0.0, 1.0, 1.0, 2.0]);
//! assert!(standardisation_weights(&[0, 2], &[true, false], None, 2).is_err());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Agreement with other software
//!
//! With unit weights, the weights of the comparators are those of
//! `MatchIt::matchit(method = "exact", estimand = "ATT")` up to its rescaling
//! of the control weights to sum to the number of matched controls; the tests
//! check them against `MatchIt` and the weighted counts against `tapply` in
//! R.
//!
//! # References
//!
//! 1. Sato T, Matsuyama Y. Marginal structural models as a tool for
//!    standardization. Epidemiology 2003;14:680-686.
//! 2. Rothman KJ, Greenland S, Lash TL. Modern Epidemiology. 3rd ed.
//!    Philadelphia: Lippincott Williams and Wilkins; 2008. Chapter 3.

use std::borrow::Cow;

use epistat_core::{CaseWeights, InputError, check};
use epistat_stats::sum::exact;

use crate::Error;

/// The weights of each person, 1 for `None`, after the columns are checked.
fn checked<'a>(
    stratum: &[u32],
    exposed: &[bool],
    weights: Option<CaseWeights<'a>>,
    n_strata: usize,
) -> Result<Cow<'a, [f64]>, Error> {
    check::same_length("exposed", stratum.len(), exposed.len())?;
    let weights = match weights {
        Some(weights) => {
            weights.check_rows(stratum.len())?;
            Cow::Borrowed(weights.as_slice())
        }
        None => Cow::Owned(vec![1.0; stratum.len()]),
    };
    if let Some(row) = stratum.iter().position(|&s| s as usize >= n_strata) {
        return Err(InputError::Domain {
            what: "stratum",
            value: f64::from(stratum[row]),
            domain: "the codes below n_strata",
            row: Some(row),
        }
        .into());
    }
    Ok(weights)
}

/// The weighted counts of the comparators and the exposed in each stratum, as
/// `[comparators, exposed]`, with every weight 1 for `weights = None`.
///
/// # Errors
///
/// [`InputError::Length`] when `exposed` or the weights differ in length
/// from `stratum`, and [`InputError::Domain`] at the first stratum code of
/// `n_strata` or more.
pub fn stratum_counts(
    stratum: &[u32],
    exposed: &[bool],
    weights: Option<CaseWeights<'_>>,
    n_strata: usize,
) -> Result<Vec<[f64; 2]>, Error> {
    let weights = checked(stratum, exposed, weights, n_strata)?;
    Ok(counts(stratum, exposed, &weights, n_strata))
}

fn counts(stratum: &[u32], exposed: &[bool], weights: &[f64], n_strata: usize) -> Vec<[f64; 2]> {
    let mut terms: Vec<[Vec<f64>; 2]> = vec![[Vec::new(), Vec::new()]; n_strata];
    for ((&s, &is_exposed), &g) in stratum.iter().zip(exposed).zip(weights) {
        terms[s as usize][usize::from(is_exposed)].push(g);
    }
    terms
        .into_iter()
        .map(|[c, e]| [exact(c), exact(e)])
        .collect()
}

/// The weight of each person after standardisation to the exposed mix:
/// `g` for the exposed and `g n_E(s) / n_C(s)` for a comparator, 0 where
/// `n_C(s)` is not positive, with `g` 1 for every person for `weights =
/// None`.
///
/// # Errors
///
/// As [`stratum_counts`].
pub fn standardisation_weights(
    stratum: &[u32],
    exposed: &[bool],
    weights: Option<CaseWeights<'_>>,
    n_strata: usize,
) -> Result<Vec<f64>, Error> {
    let weights = checked(stratum, exposed, weights, n_strata)?;
    let counts = counts(stratum, exposed, &weights, n_strata);
    Ok(stratum
        .iter()
        .zip(exposed)
        .zip(weights.iter())
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
        .collect())
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
        let counts = stratum_counts(&STRATUM, &EXPOSED, None, 3).unwrap();
        assert_eq!(counts, [[4.0, 2.0], [1.0, 3.0], [0.0, 1.0]]);
        let weights = standardisation_weights(&STRATUM, &EXPOSED, None, 3).unwrap();
        assert_eq!(
            weights,
            standardisation_weights(
                &STRATUM,
                &EXPOSED,
                Some(CaseWeights::new(&ones).unwrap()),
                3
            )
            .unwrap()
        );
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
        let g = CaseWeights::new(&g).unwrap();
        let counts = stratum_counts(&STRATUM, &EXPOSED, Some(g), 3).unwrap();
        assert_eq!(
            counts[0].map(f64::to_bits),
            [6.625_f64, 0.75].map(f64::to_bits)
        );
        let weights = standardisation_weights(&STRATUM, &EXPOSED, Some(g), 3).unwrap();
        assert_eq!(weights[2].to_bits(), (3.0 * 0.75 / 6.625_f64).to_bits());
    }

    #[test]
    fn bad_input_is_an_error_that_names_the_row() {
        let ones = [1.0; 11];
        let short = CaseWeights::new(&ones[..10]).unwrap();
        assert!(matches!(
            stratum_counts(&STRATUM, &EXPOSED, Some(short), 3),
            Err(Error::Input(InputError::Length {
                what: "weights",
                ..
            }))
        ));
        assert!(matches!(
            stratum_counts(&STRATUM, &EXPOSED[..10], None, 3),
            Err(Error::Input(InputError::Length {
                what: "exposed",
                ..
            }))
        ));
        assert_eq!(
            standardisation_weights(&STRATUM, &EXPOSED, None, 2),
            Err(Error::Input(InputError::Domain {
                what: "stratum",
                value: 2.0,
                domain: "the codes below n_strata",
                row: Some(10),
            }))
        );
        let mut bad = ones;
        bad[3] = f64::NAN;
        assert!(CaseWeights::new(&bad).is_err());
    }
}
