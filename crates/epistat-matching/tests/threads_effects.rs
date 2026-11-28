//! The bootstrap of matched sets and the effect estimates give bit-identical
//! output in rayon pools of 1, 2, 3, 4 and 8 threads.
//!
//! The sample is the 2:1 match with replacement of `tests/data/effects.json`,
//! with the logistic model and its interactions, so that the fit, the
//! two-way covariance and every replicate enter the comparison.

use epistat_core::Bootstrap;
use epistat_matching::UnitRole;
use epistat_matching::effects::{
    BootstrapSpec, Clustering, EffectSpec, MatchedSample, OutcomeModel, UnitData, bootstrap,
    estimate_effects,
};
use epistat_matching::output::LongMatches;
use epistat_testkit::{POOLS, assert_same_on_pools, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Matches {
    lalonde: Lalonde,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Lalonde {
    treat: Vec<f64>,
    re78: Vec<f64>,
    age: Vec<f64>,
    educ: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    rows: Vec<usize>,
    weight: Vec<f64>,
    subclass: Vec<Option<usize>>,
}

fn sample() -> MatchedSample {
    let m: Matches = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/effects.json"
    ));
    let case = m.cases.iter().find(|c| c.name == "replace_ratio2").unwrap();
    let l = &m.lalonde;
    let units = UnitData::new(
        (1..=l.treat.len()).map(|i| format!("u{i}")).collect(),
        l.treat.iter().map(|t| *t > 0.5).collect(),
        l.re78
            .iter()
            .map(|y| f64::from(u8::from(*y > 0.0)))
            .collect(),
    )
    .unwrap()
    .with_covariate("age", l.age.clone())
    .unwrap()
    .with_covariate("educ", l.educ.clone())
    .unwrap();
    let id: Vec<String> = case.rows.iter().map(|r| format!("u{r}")).collect();
    let long = LongMatches {
        role: vec![UnitRole::Case; id.len()],
        id,
        pair_id: case.subclass.clone(),
        weight: case.weight.clone(),
    };
    MatchedSample::from_long_matches(&long, &units).unwrap()
}

#[test]
fn effects_do_not_depend_on_the_number_of_threads() {
    let sample = sample();
    let mut spec = EffectSpec::new(OutcomeModel::Logistic);
    spec.interactions = true;
    spec.clustering = Clustering::PairAndUnit;
    assert_same_on_pools(&POOLS, || {
        let out = estimate_effects(&sample, &spec).unwrap();
        let mut bits = out.coefficients.clone();
        bits.extend(out.covariance.clone().unwrap());
        bits.push(out.difference.estimate);
        bits.push(out.difference.se.unwrap());
        bits.push(out.odds_ratio.unwrap().se.unwrap());
        let boot = bootstrap(&sample, &spec, &BootstrapSpec::new(Bootstrap::new(60, 7))).unwrap();
        bits.extend(boot.difference.replicates);
        bits.extend(boot.odds_ratio.unwrap().replicates);
        bits
    });
}
