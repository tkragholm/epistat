//! The Abadie-Imbens estimate and its standard errors are bit-identical in
//! rayon pools of 1, 2, 3, 4 and 8 threads, on the simulated cohort of
//! `tests/data/abadie_imbens.json` with the ATE, three matches, the bias
//! adjustment and the conditional variances.

use epistat_matching::Estimand;
use epistat_matching::abadie_imbens::{
    AbadieImbensData, AbadieImbensSpec, DistanceWeight, abadie_imbens,
};
use epistat_testkit::{POOLS, assert_same_on_pools, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    sim: Sim,
}

#[derive(Deserialize)]
struct Sim {
    treat: Vec<f64>,
    y: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    x3: Vec<f64>,
    x4: Vec<f64>,
}

#[test]
fn estimate_does_not_depend_on_the_number_of_threads() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/abadie_imbens.json"
    ));
    let s = r.sim;
    let data = AbadieImbensData::new(
        s.treat.iter().map(|t| *t > 0.5).collect(),
        s.y,
        vec![s.x1, s.x2, s.x3, s.x4],
    )
    .unwrap();
    let spec = AbadieImbensSpec {
        target: Estimand::Ate,
        matches: 3,
        bias_adjust: true,
        var_calc: 2,
        weight: DistanceWeight::Mahalanobis,
        ..AbadieImbensSpec::new()
    };
    assert_same_on_pools(&POOLS, || {
        let e = abadie_imbens(&data, &spec).unwrap();
        let mut bits = vec![
            e.estimate,
            e.estimate_unadjusted,
            e.se,
            e.se_sample,
            e.se_conditional,
            e.se_standard,
        ];
        bits.extend(e.times_matched);
        bits
    });
}
