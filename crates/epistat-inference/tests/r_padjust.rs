//! P-value adjustment checked against R 4.6.2's `stats::p.adjust`.
//!
//! The reference is `tests/data/padjust.json`, written by
//! `tests/data/padjust.R` with 17 significant digits. Bonferroni, Holm,
//! Hochberg, Hommel and Benjamini-Hochberg are the same products and
//! comparisons as R's and must agree to the bit. Benjamini-Yekutieli adds the
//! harmonic sum `q`, which R may accumulate in extended precision, so its
//! values are held to `n` units in the last place of `q`, that is a relative
//! `n * 2.2e-16` with `n` the number of comparisons (at most 3000 here).

use epistat_inference::multiplicity::{Adjustment, adjust};
use epistat_testkit::{Fingerprint, Tolerance, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    p: Vec<f64>,
    n: usize,
    adjusted: Adjusted,
}

#[derive(Deserialize)]
struct Adjusted {
    bonferroni: Vec<f64>,
    holm: Vec<f64>,
    hochberg: Vec<f64>,
    hommel: Vec<f64>,
    #[serde(rename = "BH")]
    bh: Vec<f64>,
    #[serde(rename = "BY")]
    by: Vec<f64>,
}

#[test]
fn matches_p_adjust() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/padjust.json"
    ));
    let mut hash = Fingerprint::new();
    for case in reference.cases {
        let a = &case.adjusted;
        let methods = [
            (Adjustment::Bonferroni, &a.bonferroni, 0),
            (Adjustment::Holm, &a.holm, 0),
            (Adjustment::Hochberg, &a.hochberg, 0),
            (Adjustment::Hommel, &a.hommel, 0),
            (Adjustment::BenjaminiHochberg, &a.bh, 0),
            (Adjustment::BenjaminiYekutieli, &a.by, case.n as u64),
        ];
        for (method, expected, ulps) in methods {
            let got = adjust(&case.p, method, Some(case.n)).unwrap();
            let tolerance = Tolerance::exact().ulps(ulps.max(1) * 2);
            let exact = ulps == 0;
            for (i, (g, e)) in got.iter().zip(expected).enumerate() {
                let ok = if exact {
                    g.to_bits() == e.to_bits()
                } else {
                    tolerance.accepts(*g, *e)
                };
                assert!(ok, "{} {method:?} [{i}]: {g:e} against {e:e}", case.name);
            }
            if case.name == "fifty uniform" {
                hash = hash.f64s(&got);
            }
        }
    }
    eprintln!("p.adjust fingerprint {:#018x}", hash.finish());
    assert_eq!(hash.finish(), 0x865b_8d4e_666b_f96c);
}
