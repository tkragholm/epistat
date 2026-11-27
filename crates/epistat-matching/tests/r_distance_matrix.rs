//! A user-supplied matrix of pair distances checked against R 4.6's `MatchIt`
//! 4.8.2, `matchit(method = "nearest", distance = <matrix>)`.
//!
//! The reference is `tests/data/matchit_distance_matrix.json`, written by
//! `tests/data/matchit_distance_matrix.R`. It holds simulated data in random
//! row order and two treated-by-control matrices, one continuous and one with
//! forbidden pairs (`Inf`, and every entry of two treated units). The runs
//! cover ratios 1 to 3, matching with replacement, the orders `data`,
//! `closest` and `farthest`, a caliper that equals the distance of one pair,
//! and the estimands ATT and ATC. The distances are continuous, so no two
//! candidates of an anchor tie and MatchIt's tie rule does not enter.
//!
//! The crate matches the same pairs in the same order of assignment in every
//! run. For ATC MatchIt reads the matrix transposed, as the controls are the
//! focal units, and so does the test.
//!
//! Other checks: the matrix is a caliper for the standard strategies, a
//! pair at the caliper is kept, a unit that is not in the matrix is never
//! matched, an order by score is refused, and a matrix with a missing value, a
//! repeated id or the wrong number of values is refused.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_core::Estimand;
use epistat_matching::Error;
use epistat_matching::prelude::*;
use epistat_matching::{DistanceCaliper, DistanceConfig, InvalidCriteriaReason};
use epistat_matching::{MatchOrder, PairDistanceMatrix};

use r_reference::{Versions, assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    treat: Vec<u8>,
    rows_treated: Vec<usize>,
    rows_control: Vec<usize>,
    dist: Vec<Vec<f64>>,
    inf: Vec<Vec<Option<f64>>>,
    caliper_exact: f64,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Run {
    name: String,
    matrix: String,
    estimand: String,
    ratio: usize,
    replace: bool,
    order: Option<String>,
    caliper: Option<f64>,
    matchit: Vec<Rows>,
}

/// R writes a vector of one element as a bare number.
#[derive(Deserialize)]
#[serde(untagged)]
enum Rows {
    Many(Vec<usize>),
    One(usize),
}

impl Rows {
    fn to_vec(&self) -> Vec<usize> {
        match self {
            Self::Many(values) => values.clone(),
            Self::One(value) => vec![*value],
        }
    }
}

fn reference() -> Reference {
    load_reference("matchit_distance_matrix.json")
}

fn record(row: usize) -> BaseRecord {
    BaseRecord::new(
        row.to_string(),
        NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date"),
    )
}

fn values(reference: &Reference, name: &str) -> Vec<Vec<f64>> {
    match name {
        "dist" => reference.dist.clone(),
        "inf" => reference
            .inf
            .iter()
            .map(|row| row.iter().map(|v| v.unwrap_or(f64::INFINITY)).collect())
            .collect(),
        other => panic!("matrix {other}"),
    }
}

/// The matrix with the focal units as rows: R's matrix for ATT, its transpose
/// for ATC.
fn matrix_for(reference: &Reference, run: &Run) -> PairDistanceMatrix {
    let table = values(reference, &run.matrix);
    let ids = |rows: &[usize]| rows.iter().map(ToString::to_string).collect::<Vec<_>>();
    if run.estimand == "ATT" {
        PairDistanceMatrix::new(
            ids(&reference.rows_treated),
            ids(&reference.rows_control),
            table.into_iter().flatten().collect(),
        )
    } else {
        let flat = (0..reference.rows_control.len())
            .flat_map(|column| table.iter().map(move |row| row[column]))
            .collect();
        PairDistanceMatrix::new(
            ids(&reference.rows_control),
            ids(&reference.rows_treated),
            flat,
        )
    }
    .expect("valid matrix")
}

fn order_of(name: &str) -> MatchOrder {
    match name {
        "closest" => MatchOrder::Closest,
        "farthest" => MatchOrder::Farthest,
        other => panic!("order {other}"),
    }
}

fn rows_of(outcome: &MatchOutcome, focal_rows: &[usize]) -> Vec<Vec<usize>> {
    let mut by_anchor: HashMap<usize, Vec<usize>> = HashMap::new();
    for pair in &outcome.pairs {
        by_anchor
            .entry(pair.case_id.as_str().parse().expect("row number"))
            .or_default()
            .push(pair.control_id.as_str().parse().expect("row number"));
    }
    focal_rows
        .iter()
        .map(|row| by_anchor.remove(row).unwrap_or_default())
        .collect()
}

fn units(reference: &Reference, atc: bool) -> (Vec<usize>, Vec<BaseRecord>, Vec<BaseRecord>) {
    let (focal, other) = if atc {
        (&reference.rows_control, &reference.rows_treated)
    } else {
        (&reference.rows_treated, &reference.rows_control)
    };
    (
        focal.clone(),
        focal.iter().map(|row| record(*row)).collect(),
        other.iter().map(|row| record(*row)).collect(),
    )
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let reference = reference();
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
    assert_eq!(
        reference
            .treat
            .iter()
            .map(|t| usize::from(*t == 1))
            .sum::<usize>(),
        reference.rows_treated.len()
    );
}

/// Pairs equal MatchIt's, in order of assignment, in every run.
#[test]
fn nearest_pairs_match_matchit() {
    let reference = reference();
    let mut with_caliper = 0;
    for run in &reference.runs {
        let atc = run.estimand == "ATC";
        let (focal_rows, anchors, candidates) = units(&reference, atc);
        let caliper = run
            .caliper
            .map(|width| DistanceCaliper::new(width).expect("width"));
        let config = DistanceConfig::matrix(matrix_for(&reference, run), caliper);
        let mut job = MatchJob::new_standard(&anchors, &candidates, 0)
            .with_nearest()
            .with_ratio(MatchRatio::new(run.ratio).expect("ratio"))
            .with_replacement(run.replace)
            .with_estimand(if atc { Estimand::Atc } else { Estimand::Att })
            .with_distance_config(config);
        if let Some(order) = &run.order {
            job = job.with_m_order(order_of(order));
        }
        let outcome = job.run().expect("a valid job");
        let expected: Vec<Vec<usize>> = run.matchit.iter().map(Rows::to_vec).collect();
        assert_eq!(rows_of(&outcome, &focal_rows), expected, "{}", run.name);
        with_caliper += usize::from(run.caliper.is_some());
    }
    assert_eq!(with_caliper, 2);
    assert_eq!(reference.runs.len(), 14);
}

/// The caliper is inclusive: a width just below the fifth smallest nearest
/// distance loses that pair, and the width itself keeps it.
#[test]
fn a_pair_at_the_caliper_is_kept() {
    let reference = reference();
    let run = reference
        .runs
        .iter()
        .find(|run| run.name == "att_caliper")
        .expect("run");
    let (_, anchors, candidates) = units(&reference, false);
    let count = |width: f64| {
        let config = DistanceConfig::matrix(
            matrix_for(&reference, run),
            Some(DistanceCaliper::new(width).expect("width")),
        );
        MatchJob::new_standard(&anchors, &candidates, 0)
            .with_nearest()
            .with_distance_config(config)
            .run()
            .expect("a valid job")
            .matched_cases
    };
    assert_eq!(count(reference.caliper_exact), 5);
    assert_eq!(count(reference.caliper_exact * (1.0 - 1e-12)), 4);
}

/// The matrix limits the standard strategies as a caliper: every pair chosen
/// has a finite entry no wider than the caliper.
#[test]
fn matrix_is_a_caliper_for_the_other_strategies() {
    let reference = reference();
    let run = &reference.runs[0];
    let (_, anchors, candidates) = units(&reference, false);
    let width = reference.caliper_exact * 3.0;
    let matrix = matrix_for(&reference, run);
    let config = DistanceConfig::matrix(
        matrix.clone(),
        Some(DistanceCaliper::new(width).expect("width")),
    );
    let outcome = MatchJob::new_standard(&anchors, &candidates, 7)
        .with_distance_config(config)
        .with_birth_window(BirthDateWindowDays::new(10).expect("window"))
        .run()
        .expect("a valid job");
    assert!(!outcome.pairs.is_empty(), "no output, test is vacuous");
    for pair in &outcome.pairs {
        let distance = matrix
            .case_position(pair.case_id.as_str())
            .zip(matrix.control_position(pair.control_id.as_str()))
            .and_then(|(i, j)| matrix.at(i, j))
            .expect("a finite entry");
        assert!(distance <= width, "{distance} against {width}");
    }
}

/// A unit absent from the matrix has no distance and is not matched.
#[test]
fn units_outside_the_matrix_are_not_matched() {
    let matrix = PairDistanceMatrix::new(
        vec!["a".into(), "b".into()],
        vec!["x".into(), "y".into()],
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .expect("valid");
    let anchors = [
        BaseRecord::new("a", NaiveDate::from_ymd_opt(2000, 1, 1).unwrap()),
        BaseRecord::new("z", NaiveDate::from_ymd_opt(2000, 1, 1).unwrap()),
    ];
    let candidates = [
        BaseRecord::new("x", NaiveDate::from_ymd_opt(2000, 1, 1).unwrap()),
        BaseRecord::new("w", NaiveDate::from_ymd_opt(2000, 1, 1).unwrap()),
    ];
    let outcome = MatchJob::new_standard(&anchors, &candidates, 0)
        .with_nearest()
        .with_distance_config(DistanceConfig::matrix(matrix.clone(), None))
        .run()
        .expect("a valid job");
    assert_eq!(outcome.pairs.len(), 1);
    assert_eq!(outcome.pairs[0].case_id.as_str(), "a");
    assert_eq!(outcome.pairs[0].control_id.as_str(), "x");
    let (rows, columns) = matrix.positions_of(&anchors, &candidates);
    assert_eq!(rows, vec![Some(0), None]);
    assert_eq!(columns, vec![Some(0), None]);
}

/// Distances by position are the table's entries, and a forbidden pair has
/// none.
#[test]
fn distance_by_position_reads_the_table() {
    let matrix = PairDistanceMatrix::new(
        vec!["a".into(), "b".into()],
        vec!["x".into(), "y".into(), "z".into()],
        vec![1.0, f64::INFINITY, 3.0, 4.0, 5.0, 6.0],
    )
    .expect("valid");
    let config = DistanceConfig::matrix(matrix, None);
    assert_eq!(config.pair_distance_by_position(0, 0), Some(1.0));
    assert_eq!(config.pair_distance_by_position(0, 1), None);
    assert_eq!(config.pair_distance_by_position(1, 2), Some(6.0));
    assert_eq!(config.pair_distance_by_position(2, 0), None);
    assert_eq!(
        DistanceConfig::date(None).pair_distance_by_position(0, 0),
        None
    );
}

#[test]
fn an_order_by_score_is_refused() {
    let reference = reference();
    let run = &reference.runs[0];
    let (_, anchors, candidates) = units(&reference, false);
    let outcome = MatchJob::new_standard(&anchors, &candidates, 0)
        .with_nearest()
        .with_m_order(MatchOrder::Largest)
        .with_distance_config(DistanceConfig::matrix(matrix_for(&reference, run), None))
        .run();
    assert_eq!(
        outcome.unwrap_err(),
        Error::InvalidCriteria(InvalidCriteriaReason::NearestOptions)
    );
}

#[test]
fn invalid_matrices_are_refused() {
    let ids = |names: &[&str]| names.iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(
        PairDistanceMatrix::new(vec![], ids(&["x"]), vec![]).unwrap_err(),
        Error::Input(epistat_core::InputError::Empty {
            what: "distance matrix ids"
        })
    );
    assert_eq!(
        PairDistanceMatrix::new(ids(&["a"]), ids(&["x", "y"]), vec![1.0]).unwrap_err(),
        Error::Input(epistat_core::InputError::Length {
            what: "distance matrix values",
            expected: 2,
            found: 1
        })
    );
    assert_eq!(
        PairDistanceMatrix::new(ids(&["a", "a"]), ids(&["x"]), vec![1.0, 2.0]).unwrap_err(),
        Error::DuplicateId { id: "a".into() }
    );
    assert_eq!(
        PairDistanceMatrix::new(
            ids(&["a", "b"]),
            ids(&["x", "y"]),
            vec![1.0, 2.0, f64::NAN, 4.0]
        )
        .unwrap_err(),
        Error::DistanceNotANumber {
            case: 1,
            control: 0
        }
    );
    // A negative distance is refused, with its cell named.
    assert_eq!(
        PairDistanceMatrix::new(
            ids(&["a", "b"]),
            ids(&["x", "y"]),
            vec![-5.0, 0.0, 0.0, 1.0]
        )
        .unwrap_err(),
        Error::NegativeDistance {
            case: 0,
            control: 0
        }
    );
    assert_eq!(
        PairDistanceMatrix::new(
            ids(&["a", "b"]),
            ids(&["x", "y"]),
            vec![1.0, 2.0, 3.0, f64::NEG_INFINITY]
        )
        .unwrap_err(),
        Error::NegativeDistance {
            case: 1,
            control: 1
        }
    );
}
