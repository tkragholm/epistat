//! Effect estimates from the engine's own output: a match by the engine, its
//! `match_data`, and the effect on that sample. The R comparisons in
//! `r_effects.rs` start from MatchIt's matched rows; this test checks that the
//! crate's output reaches the estimator with the ids and weights in place.

use chrono::NaiveDate;
use epistat_matching::effects::{
    EffectSpec, MatchedSample, OutcomeModel, UnitData, estimate_effects,
};
use epistat_matching::output::{MatchDataOptions, Units, match_data};
use epistat_matching::prelude::*;

struct Cohort {
    anchors: Vec<BaseRecord>,
    candidates: Vec<BaseRecord>,
    ids: Vec<String>,
    treated: Vec<bool>,
    x: Vec<f64>,
}

/// 40 anchors and 120 candidates with one covariate, all born on the same day
/// so the birth window admits every pair.
fn cohort() -> Cohort {
    let birth = NaiveDate::from_ymd_opt(1990, 6, 1).expect("valid date");
    let (mut anchors, mut candidates) = (Vec::new(), Vec::new());
    let (mut ids, mut treated, mut x) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..160_u32 {
        let id = format!("u{i:03}");
        let is_case = i < 40;
        let record = BaseRecord::new(id.clone(), birth);
        if is_case {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
        ids.push(id);
        treated.push(is_case);
        x.push(f64::from((i * 37) % 101) / 10.0);
    }
    Cohort {
        anchors,
        candidates,
        ids,
        treated,
        x,
    }
}

fn matched(cohort: &Cohort) -> MatchedSample {
    let outcome = MatchJob::new_standard(&cohort.anchors, &cohort.candidates, 11)
        .run()
        .expect("a valid job");
    assert_eq!(outcome.matched_cases, 40, "every anchor finds a candidate");
    let units = Units::new(cohort.ids.clone(), cohort.treated.clone()).expect("unique ids");
    let data = match_data(&outcome, &units, MatchDataOptions::default()).expect("fixed roles");
    let y: Vec<f64> = cohort
        .treated
        .iter()
        .zip(&cohort.x)
        .map(|(t, x)| 2.0 * f64::from(u8::from(*t)) + 0.5 * x + 1.0)
        .collect();
    let units = UnitData::new(cohort.ids.clone(), cohort.treated.clone(), y)
        .expect("valid units")
        .with_covariate("x", cohort.x.clone())
        .expect("valid covariate");
    MatchedSample::from_match_data(&data, &units).expect("every matched id has data")
}

#[test]
fn an_outcome_linear_in_the_covariate_gives_the_effect_exactly() {
    let cohort = cohort();
    let sample = matched(&cohort);
    assert_eq!(sample.len(), 80, "40 pairs");
    let estimates =
        estimate_effects(&sample, &EffectSpec::new(OutcomeModel::Linear)).expect("a linear fit");
    // The outcome is 2 t + 0.5 x + 1 exactly, so the fit has no residual and
    // the effect is 2 to the rounding of a least-squares solve on 80 rows with
    // x up to 10: a few units in the last place of 2.
    assert!(
        (estimates.difference.estimate - 2.0).abs() < 1e-12,
        "{}",
        estimates.difference.estimate
    );
    let se = estimates.difference.se.expect("a defined variance");
    assert!(se < 1e-10, "no residual, so no variance: {se}");
}

#[test]
fn without_covariates_the_effect_is_the_difference_in_matched_means() {
    let cohort = cohort();
    let outcome = MatchJob::new_standard(&cohort.anchors, &cohort.candidates, 11)
        .run()
        .expect("a valid job");
    let units = Units::new(cohort.ids.clone(), cohort.treated.clone()).expect("unique ids");
    let data = match_data(&outcome, &units, MatchDataOptions::default()).expect("fixed roles");
    let y: Vec<f64> = cohort
        .treated
        .iter()
        .zip(&cohort.x)
        .map(|(t, x)| 2.0 * f64::from(u8::from(*t)) + 0.5 * x + 1.0)
        .collect();
    let sample = MatchedSample::from_match_data(
        &data,
        &UnitData::new(cohort.ids.clone(), cohort.treated.clone(), y.clone()).expect("valid units"),
    )
    .expect("every matched id has data");
    let estimates =
        estimate_effects(&sample, &EffectSpec::new(OutcomeModel::Linear)).expect("a linear fit");

    // The same difference from the pairs: 1:1 without replacement gives every
    // matched unit weight 1.
    let row = |id: &str| cohort.ids.iter().position(|u| u == id).expect("known id");
    let (mut treated_sum, mut control_sum) = (0.0, 0.0);
    for pair in &outcome.pairs {
        treated_sum += y[row(&pair.case_id)];
        control_sum += y[row(&pair.control_id)];
    }
    let n = 40.0;
    let by_hand = treated_sum / n - control_sum / n;
    assert!(
        (estimates.difference.estimate - by_hand).abs() < 1e-12,
        "{} against {by_hand}",
        estimates.difference.estimate
    );
}
