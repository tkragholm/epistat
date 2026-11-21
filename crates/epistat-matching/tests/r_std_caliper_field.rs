//! A caliper on a covariate in standard deviations, checked against MatchIt
//! 4.8.2 with `caliper = c(z = width)` and `std.caliper = TRUE`.
//!
//! The reference is `tests/data/matchit_std_caliper_field.json`, written by
//! `tests/data/matchit_std_caliper_field.R`. It holds simulated data, a numeric
//! distance, and the nearest-neighbour pairs (ratio 1, data order, no
//! replacement) for five widths, with `discard = "none"` and `"both"`. MatchIt
//! takes the standard deviation of `z` over the units that remain after
//! trimming, which the script confirms by reproducing each standardised result
//! with an unstandardised caliper of width times that standard deviation.
//!
//! The spec runs the same matching with the distance as scores and
//! `std_caliper` set for the field. The pairs are integers, so they must be
//! equal as sets.

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Deserialize)]
struct Reference {
    treat: Vec<f64>,
    z: Vec<f64>,
    distance: Vec<f64>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    discard: String,
    width: f64,
    treated: Vec<usize>,
    control: Vec<usize>,
}

fn id(row: usize) -> String {
    format!("r{row}")
}

fn pairs_from_spec(reference: &Reference, case: &Case) -> BTreeSet<(usize, usize)> {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    let mut scores = BTreeMap::new();
    for (idx, label) in reference.treat.iter().enumerate() {
        let row = idx + 1;
        let mut record = BalanceRecord::builder(id(row), birth).build();
        record.core.numerics.insert("z".into(), reference.z[idx]);
        scores.insert(id(row), reference.distance[idx]);
        if *label > 0.5 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Scores {
            scores,
            caliper: None,
        },
        nearest: NearestSpec {
            m_order: Some(OrderSpec::Data),
            ..Default::default()
        },
        assignment: Some(AssignmentSpec::Sequential),
        calipers: BTreeMap::from([("z".to_string(), case.width)]),
        std_caliper: StdCaliperSpec::Each(StdCaliperEach {
            distance: false,
            fields: BTreeMap::from([("z".to_string(), true)]),
        }),
        discard: match case.discard.as_str() {
            "both" => Some(DiscardSpec::Both),
            _ => None,
        },
        birth_window_days: 100_000,
        ..Default::default()
    };
    let output = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect("the spec runs");
    let SpecOutput::Pairs { outcome, .. } = output else {
        panic!("nearest matching returns pairs");
    };
    let row = |text: &str| text[1..].parse::<usize>().expect("row id");
    outcome
        .pairs
        .iter()
        .map(|pair| (row(pair.anchor_id()), row(pair.comparator_id())))
        .collect()
}

/// The pairs equal MatchIt's for every width, with and without trimming. With
/// `discard = "both"` the standard deviation is the one over the remaining
/// units, which differs from the one over all units for these data.
#[test]
fn a_standard_deviation_caliper_on_a_field_equals_matchit() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/matchit_std_caliper_field.json"
    ));
    assert!(reference.cases.len() >= 10);
    for case in &reference.cases {
        let expected: BTreeSet<(usize, usize)> = case
            .treated
            .iter()
            .copied()
            .zip(case.control.iter().copied())
            .collect();
        let ours = pairs_from_spec(&reference, case);
        assert_eq!(
            ours, expected,
            "discard {} width {}: pairs differ from MatchIt",
            case.discard, case.width
        );
    }
}
