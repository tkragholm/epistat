//! Options that a method does not read, scores that are missing or not
//! finite, and time limits that a duration cannot hold are errors, and the
//! error names the option or the unit.

use chrono::NaiveDate;
use epistat_core::Estimand;
use epistat_matching::Error;
use epistat_matching::FineBalance;
use epistat_matching::prelude::*;
use epistat_matching::spec::{
    BinningSpec, CardinalitySpec, CemSpec, CemVariableSpec, CovarianceSpec, CovariatesSpec,
    DistanceSpec, MatchMethod, NearestSpec, TransformSpec, VariableRatioSpec,
};
use epistat_matching::{BalanceRecord, SpecOutput};
use std::collections::BTreeMap;

fn day(day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, day).expect("valid date")
}

fn units() -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let record = |id: &str, birth: u32, x: f64| {
        let mut record = BalanceRecord::builder(id, day(birth))
            .numeric("x", x)
            .build();
        record.core.strata.insert("sex".into(), "1".into());
        record
    };
    let anchors = vec![
        record("a1", 1, 1.0),
        record("a2", 2, 2.0),
        record("a3", 3, 3.0),
    ];
    let candidates = (0..9_u32)
        .map(|i| record(&format!("c{i}"), 1 + i, f64::from(i) * 0.4))
        .collect();
    (anchors, candidates)
}

fn run(spec: &MatchSpec) -> Result<SpecOutput, Error> {
    let (anchors, candidates) = units();
    run_spec(
        spec,
        SpecInput::Standard {
            cases: &anchors,
            controls: &candidates,
        },
    )
}

fn names(error: &Error, option: &str) {
    let text = error.to_string();
    assert!(
        text.contains(option),
        "the error does not name {option}: {text}"
    );
}

fn scores(score: impl Fn(&str) -> Option<f64>) -> BTreeMap<String, f64> {
    let (anchors, candidates) = units();
    anchors
        .iter()
        .chain(&candidates)
        .filter_map(|record| score(&record.core.id).map(|s| (record.core.id.clone(), s)))
        .collect()
}

fn nearest_on(scores: BTreeMap<String, f64>) -> MatchSpec {
    MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Scores {
            scores,
            caliper: None,
        },
        seed: Some(0),
        ..Default::default()
    }
}

fn x_covariates() -> CovariatesSpec {
    CovariatesSpec {
        names: vec!["x".into()],
        ..Default::default()
    }
}

#[test]
fn a_cardinality_time_limit_a_duration_cannot_hold_is_an_error() {
    let spec = MatchSpec {
        method: MatchMethod::Cardinality,
        cardinality: CardinalitySpec {
            covariates: x_covariates(),
            time: Some(1e30),
            ..Default::default()
        },
        seed: Some(0),
        ..Default::default()
    };
    names(
        &spec.validate().expect_err("1e30 seconds"),
        "cardinality.time",
    );
    names(&run(&spec).expect_err("1e30 seconds"), "cardinality.time");
}

#[test]
fn fine_balance_outside_the_optimal_method_is_an_error() {
    let spec = MatchSpec {
        fine_balance: Some(FineBalance::on_keys(["sex"])),
        seed: Some(0),
        ..Default::default()
    };
    names(&spec.validate().expect_err("random method"), "fine_balance");
    let full = MatchSpec {
        method: MatchMethod::Full,
        ..spec
    };
    names(&full.validate().expect_err("full method"), "fine_balance");
}

#[test]
fn a_birth_window_is_an_error_for_methods_that_do_not_read_it() {
    let exact = MatchSpec {
        method: MatchMethod::Exact,
        exact: vec!["sex".into()],
        birth_window_days: 5,
        seed: Some(0),
        ..Default::default()
    };
    names(&exact.validate().expect_err("exact"), "birth_window_days");
    let scores = scores(|_| Some(0.5));
    for method in [MatchMethod::Subclass, MatchMethod::Quick] {
        let spec = MatchSpec {
            method,
            birth_window_days: 5,
            ..nearest_on(scores.clone())
        };
        names(&spec.validate().expect_err("window"), "birth_window_days");
    }
    let cem = MatchSpec {
        method: MatchMethod::Cem,
        cem: CemSpec {
            variables: vec![CemVariableSpec::Numeric {
                name: "x".into(),
                binning: BinningSpec::Bins { n: 3 },
            }],
            ..Default::default()
        },
        birth_window_days: 5,
        seed: Some(0),
        ..Default::default()
    };
    names(&cem.validate().expect_err("cem"), "birth_window_days");
}

#[test]
fn an_estimand_is_an_error_for_methods_that_do_not_read_it() {
    let exact = MatchSpec {
        method: MatchMethod::Exact,
        exact: vec!["sex".into()],
        estimand: Estimand::Ate,
        seed: Some(0),
        ..Default::default()
    };
    names(&exact.validate().expect_err("exact"), "estimand");
    let quick = MatchSpec {
        method: MatchMethod::Quick,
        estimand: Estimand::Atc,
        ..nearest_on(scores(|_| Some(0.5)))
    };
    names(&quick.validate().expect_err("quick"), "estimand");
}

#[test]
fn a_distance_caliper_is_an_error_for_the_subclass_method() {
    let spec = MatchSpec {
        method: MatchMethod::Subclass,
        distance: DistanceSpec::Scores {
            scores: scores(|_| Some(0.5)),
            caliper: Some(0.1),
        },
        seed: Some(0),
        ..Default::default()
    };
    names(&spec.validate().expect_err("caliper"), "distance");
}

#[test]
fn trial_options_outside_trial_mode_are_an_error() {
    let mut spec = MatchSpec {
        seed: Some(0),
        ..MatchSpec::default()
    };
    spec.trials.first_period = Some(3);
    names(&spec.validate().expect_err("standard mode"), "trials");
}

#[test]
fn a_variable_ratio_needs_a_score_distance() {
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Mahalanobis {
            covariates: x_covariates(),
            covariance: CovarianceSpec::default(),
            transform: TransformSpec::default(),
            caliper: None,
        },
        nearest: NearestSpec {
            variable_ratio: Some(VariableRatioSpec {
                average: 1.5,
                min: 1,
                max: 2,
            }),
            ..Default::default()
        },
        seed: Some(0),
        ..Default::default()
    };
    names(
        &spec.validate().expect_err("mahalanobis"),
        "nearest.variable_ratio",
    );
}

#[test]
fn a_score_that_is_not_finite_is_an_error_naming_the_unit() {
    for (id, value) in [
        ("a1", f64::NAN),
        ("c0", f64::INFINITY),
        ("c3", f64::NEG_INFINITY),
    ] {
        let spec = nearest_on(scores(|unit| Some(if unit == id { value } else { 0.5 })));
        names(&run(&spec).expect_err("non-finite score"), id);
        let subclass = MatchSpec {
            method: MatchMethod::Subclass,
            ..spec
        };
        names(&run(&subclass).expect_err("non-finite score"), id);
    }
}

#[test]
fn a_unit_without_a_score_is_an_error_naming_the_unit() {
    let spec = nearest_on(scores(|id| (id != "c1").then_some(0.5)));
    names(&run(&spec).expect_err("missing score"), "c1");
    let subclass = MatchSpec {
        method: MatchMethod::Subclass,
        ..spec
    };
    names(&run(&subclass).expect_err("missing score"), "c1");
}
