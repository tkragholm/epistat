//! The Kitagawa decomposition with its variances and the standardisation
//! weights, checked against R 4.6.2.
//!
//! The reference is `tests/data/decomposition.json`, written by
//! `tests/data/decomposition.R`: `DasGuptR::dgnpop` 2.2.0 for the symmetric
//! decomposition, the formulas written out for the decomposition with a
//! reference and for the variances, `MatchIt::matchit` 4.8.2 exact matching
//! for the weights of unweighted comparators and `tapply` for weighted counts.
//!
//! Tolerance. Each part is a sum of a few dozen products, so its rounding
//! error is a few units of `eps` times the sum of the absolute terms, which is
//! below the sum of the crude rates of the two populations; the hand formulas
//! in R are checked to `1e-13` of that scale. `dgnpop` forms the parts as
//! differences of standardised rates that it sums in another order, `1e-12`
//! of the scale. A variance is a sum of positive terms, or for a covariance of
//! terms below the variances, so it is checked to `1e-13` of the sum of the
//! two part variances. The weights are a product and a quotient of two counts
//! that R sums in row order, with a rounding error below `n eps` for `n` = 400
//! persons, `1e-13` relative.

use epistat_stats::decomposition::{Split, kitagawa, kitagawa_poisson, kitagawa_symmetric, shares};
use epistat_stats::rates::Strata;
use epistat_stats::standardisation::{standardisation_weights, stratum_counts};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    kitagawa: Vec<Case>,
    standardisation: Standardisation,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    events1: Vec<f64>,
    time1: Vec<f64>,
    events0: Vec<f64>,
    time0: Vec<f64>,
    scale: f64,
    gap: f64,
    reference: Vec<f64>,
    symmetric: Vec<f64>,
    dasgupta: Vec<f64>,
    var_gap: f64,
    var_reference: Vec<f64>,
    var_symmetric: Vec<f64>,
}

#[derive(Deserialize)]
struct Standardisation {
    stratum: Vec<u32>,
    exposed: Vec<bool>,
    matchit: Vec<f64>,
    matchit_factor: f64,
    g: Vec<f64>,
    comparators: Vec<f64>,
    exposed_counts: Vec<f64>,
    weighted: Vec<f64>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/decomposition.json"
    ))
}

#[test]
fn decompositions_and_their_variances() {
    let mut worst = 0.0_f64;
    for case in reference().kitagawa {
        let name = &case.name;
        let group = Strata {
            events: &case.events1,
            time: &case.time1,
        };
        let reference = Strata {
            events: &case.events0,
            time: &case.time0,
        };
        let crude =
            |d: &[f64], y: &[f64]| case.scale * d.iter().sum::<f64>() / y.iter().sum::<f64>();
        let scale = crude(&case.events1, &case.time1) + crude(&case.events0, &case.time0);
        let mut check = |got: f64, want: f64, bound: f64, of: f64| {
            worst = worst.max((got - want).abs() / (bound * of));
        };
        let a = shares(&case.time1).unwrap();
        let b = shares(&case.time0).unwrap();
        let rate = |d: &[f64], y: &[f64]| -> Vec<f64> {
            d.iter().zip(y).map(|(d, y)| case.scale * d / y).collect()
        };
        let (r, q) = (
            rate(&case.events1, &case.time1),
            rate(&case.events0, &case.time0),
        );
        let k = kitagawa(&a, &r, &b, &q).unwrap();
        let s = kitagawa_symmetric(&a, &r, &b, &q).unwrap();
        check(k.gap, case.gap, 1e-13, scale);
        check(k.composition, case.reference[0], 1e-13, scale);
        check(k.rate, case.reference[1], 1e-13, scale);
        check(s.composition, case.symmetric[0], 1e-13, scale);
        check(s.rate, case.symmetric[1], 1e-13, scale);
        check(s.composition, case.dasgupta[0], 1e-12, scale);
        check(s.rate, case.dasgupta[1], 1e-12, scale);
        for (split, want, d) in [
            (Split::Reference, &case.var_reference, k),
            (Split::Symmetric, &case.var_symmetric, s),
        ] {
            let p = kitagawa_poisson(group, reference, split, 0.95, case.scale).unwrap();
            assert_eq!(p.gap.estimate.to_bits(), d.gap.to_bits(), "{name}");
            assert_eq!(
                p.composition.estimate.to_bits(),
                d.composition.to_bits(),
                "{name}"
            );
            assert_eq!(p.rate.estimate.to_bits(), d.rate.to_bits(), "{name}");
            let of = want[0] + want[1];
            check(p.gap.se * p.gap.se, case.var_gap, 1e-13, case.var_gap);
            check(p.composition.se * p.composition.se, want[0], 1e-13, of);
            check(p.rate.se * p.rate.se, want[1], 1e-13, of);
            check(p.covariance, want[2], 1e-13, of);
            // The gap is the sum of the parts, so its variance is the sum of
            // theirs and twice their covariance.
            let total = p.composition.se.powi(2) + p.rate.se.powi(2) + 2.0 * p.covariance;
            check(total, case.var_gap, 1e-12, of);
        }
    }
    eprintln!("decomposition: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn standardisation_weights_against_matchit_and_tapply() {
    let s = reference().standardisation;
    let n_strata = 8;
    let ones = vec![1.0; s.stratum.len()];
    let w = standardisation_weights(&s.stratum, &s.exposed, &ones, n_strata).unwrap();
    let mut worst = 0.0_f64;
    for ((&got, &m), &e) in w.iter().zip(&s.matchit).zip(&s.exposed) {
        let want = if e { m } else { m / s.matchit_factor };
        worst = worst.max((got - want).abs() / (1e-13 * want.abs().max(1.0)));
    }
    let counts = stratum_counts(&s.stratum, &s.exposed, &s.g, n_strata).unwrap();
    for (k, c) in counts.iter().enumerate() {
        worst = worst.max((c[0] - s.comparators[k]).abs() / (1e-13 * s.comparators[k]));
        worst = worst.max((c[1] - s.exposed_counts[k]).abs() / (1e-13 * s.exposed_counts[k]));
    }
    let w = standardisation_weights(&s.stratum, &s.exposed, &s.g, n_strata).unwrap();
    for (&got, &want) in w.iter().zip(&s.weighted) {
        worst = worst.max((got - want).abs() / (1e-13 * want.abs().max(1e-300)));
    }
    eprintln!("standardisation: worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}
