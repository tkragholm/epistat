//! Weighted eQQ distances checked against R 4.6.2 with MatchIt 4.8.2.
//!
//! `tests/data/matchit_eqq.R` calls MatchIt's internal `qqsum()`, which
//! `summary.matchit(standardize = FALSE)` uses for the eQQ mean and maximum, on
//! 400 small random samples. They differ in the sizes of the groups, the number
//! of distinct values (so that both the branch for at most five distinct values
//! and the continuous one are used), ties, and the weights (one, random, integer
//! and with zeros). Each case is compared to
//! `epistat_stats::balance::eqq_distance_weighted`.
//!
//! The two agree except in one situation, where MatchIt has a defect. With at
//! most five distinct values in the pooled sample, MatchIt reads the larger
//! group at its last cumulative weight as the largest pooled value, even when
//! the larger group has no unit at that value. The function reads the larger
//! group's own largest value. The test counts these cases, asserts that the
//! two differ in no other case, and asserts both values for a hand-made
//! sample: group 1 is 1 1 1 2 and group 0 is 2 3, where MatchIt gives a mean of
//! 0.5 and the function 1.
//!
//! The values are read from the data and not computed, so a tolerance only has
//! to absorb the difference between sums formed in R and here in the same order:
//! 1e-12 relative.

#![expect(
    clippy::float_cmp,
    reason = "group indicators and weights in the reference data are exact zeros and ones, and values are compared for equality"
)]

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_stats::balance::eqq_distance_weighted;
use epistat_testkit::{Tolerance, load_json, na_f64};
use r_reference::assert_r_version;
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    x: Vec<f64>,
    group: Vec<f64>,
    weight: Vec<f64>,
    #[serde(deserialize_with = "na_f64")]
    mean: f64,
    #[serde(deserialize_with = "na_f64")]
    max: f64,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    matchit: String,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
    defect: Case,
}

/// Our distances for a case: group 1 is the first argument.
fn ours(case: &Case) -> Option<(f64, f64)> {
    let pick = |wanted: f64| {
        let (mut values, mut weights) = (Vec::new(), Vec::new());
        for ((x, g), w) in case.x.iter().zip(&case.group).zip(&case.weight) {
            if *g == wanted {
                values.push(*x);
                weights.push(*w);
            }
        }
        (values, weights)
    };
    let (a, wa) = pick(1.0);
    let (b, wb) = pick(0.0);
    eqq_distance_weighted(&a, &wa, &b, &wb)
}

/// Whether the few-valued branch of MatchIt reads a value the larger group
/// does not have at its last cumulative weight.
fn hits_the_defect(case: &Case) -> bool {
    let positive = |wanted: f64| {
        case.x
            .iter()
            .zip(&case.group)
            .zip(&case.weight)
            .filter(|((_, g), w)| **g == wanted && **w > 0.0)
            .map(|((x, _), _)| *x)
            .collect::<Vec<_>>()
    };
    let (a, b) = (positive(1.0), positive(0.0));
    let mut pooled = case.x.clone();
    pooled.sort_by(f64::total_cmp);
    pooled.dedup();
    if pooled.len() > 5 || a.len() == b.len() {
        return false;
    }
    let large = if a.len() > b.len() { &a } else { &b };
    let large_max = large.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    large_max < pooled[pooled.len() - 1]
}

#[test]
fn weighted_eqq_agrees_with_matchit() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/matchit_eqq.json"
    ));
    assert_r_version(&reference.versions.r);
    assert_eq!(reference.versions.matchit, "4.8.2");
    let tolerance = Tolerance::exact().rel(1e-12).abs(1e-12);
    let (mut compared, mut defects, mut few_valued) = (0, 0, 0);
    for (idx, case) in reference.cases.iter().enumerate() {
        let (mean, max) = ours(case).unwrap_or_else(|| panic!("case {idx} is undefined"));
        let agrees = tolerance.accepts(mean, case.mean) && tolerance.accepts(max, case.max);
        if hits_the_defect(case) {
            defects += 1;
            continue;
        }
        let mut pooled = case.x.clone();
        pooled.sort_by(f64::total_cmp);
        pooled.dedup();
        few_valued += usize::from(pooled.len() <= 5);
        assert!(
            agrees,
            "case {idx}: ({mean}, {max}) against R ({}, {})",
            case.mean, case.max
        );
        compared += 1;
    }
    assert!(compared > 250, "{compared} cases compared");
    assert!(few_valued > 100, "{few_valued} few-valued cases");
    assert!(defects > 5, "{defects} cases hit the defect");
}

/// MatchIt reads the pooled maximum, 3, as the larger group's largest value
/// and so finds no gap between 3 and 3. The larger group, 1 1 1 2, ends at 2,
/// so the gap is 1.
#[test]
fn few_valued_defect_is_asserted_in_both_values() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/matchit_eqq.json"
    ));
    let case = &reference.defect;
    assert!((case.mean - 0.5).abs() < 1e-15 && (case.max - 1.0).abs() < 1e-15);
    let (mean, max) = ours(case).unwrap();
    assert!((mean - 1.0).abs() < 1e-15 && (max - 1.0).abs() < 1e-15);
    assert!(hits_the_defect(case));
}
