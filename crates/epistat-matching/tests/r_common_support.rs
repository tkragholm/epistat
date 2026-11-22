//! Common-support trimming checked against MatchIt 4.8.2's `discard`.
//!
//! The reference is `tests/data/matchit_discard.json`, written by
//! `tests/data/matchit_discard.R`. For `lalonde`, a simulated data set of
//! discrete covariates and four numeric-vector grids it holds the distance
//! MatchIt used, on the probability and on the linear predictor scale, and the
//! units discarded under `discard = "none"`, `"treated"`, `"control"` and
//! `"both"`. The grids and the discrete data have units exactly at the ends of
//! the other group's range, in both groups, and many units with equal scores.
//!
//! Which units are kept is read from the matching itself. Every kept control is
//! allowed for every kept treated unit, with replacement and a ratio equal to
//! the number of controls, so a unit takes part in a pair exactly when it is
//! kept. The test compares those sets with MatchIt's, and the trimmed counts
//! in the diagnostics with MatchIt's counts.
//!
//! A numeric `distance` in MatchIt is used as given, with no transformation.
//! The scores here are handed to `DistanceConfig::propensity_score_map` the
//! same way, so the test covers the numeric-vector row for trimming too.

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{DeterministicSelection, RapidHashMap};
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    matchit: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    scale: String,
    treat: Vec<f64>,
    distance: Vec<f64>,
    discarded: Discarded,
}

#[derive(Deserialize)]
struct Discarded {
    none: Vec<bool>,
    treated: Vec<bool>,
    control: Vec<bool>,
    both: Vec<bool>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/matchit_discard.json"
    ))
}

fn id(idx: usize) -> String {
    format!("u{idx}")
}

struct Run {
    outcome: MatchOutcome,
    kept_anchors: BTreeSet<usize>,
    kept_candidates: BTreeSet<usize>,
}

fn run(case: &Case, policy: Option<CommonSupport>) -> Run {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    let mut scores = RapidHashMap::default();
    for (idx, label) in case.treat.iter().enumerate() {
        let record = BaseRecord::new(id(idx), birth);
        scores.insert(id(idx), case.distance[idx]);
        if *label > 0.5 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    let mut criteria = MatchingCriteria::builder()
        .match_ratio(candidates.len())
        .allow_replacement(true)
        .build();
    criteria.common_support = policy;
    let config = DistanceConfig::propensity_score_map(scores, None);
    let request = StandardMatchRequest::new(&criteria, DeterministicSelection, &())
        .with_distance_config(&config);
    let outcome = match_standard(&anchors, &candidates, request);
    let index_of = |text: &str| text[1..].parse::<usize>().expect("unit id");
    let kept_anchors = outcome
        .pairs
        .iter()
        .map(|p| index_of(p.anchor_id()))
        .collect();
    let kept_candidates = outcome
        .pairs
        .iter()
        .map(|p| index_of(p.comparator_id()))
        .collect();
    Run {
        outcome,
        kept_anchors,
        kept_candidates,
    }
}

fn expected_kept(case: &Case, discarded: &[bool], group: f64) -> BTreeSet<usize> {
    (0..case.treat.len())
        .filter(|idx| (case.treat[*idx] - group).abs() < 0.5 && !discarded[*idx])
        .collect()
}

fn count_discarded(case: &Case, discarded: &[bool], group: f64) -> usize {
    (0..case.treat.len())
        .filter(|idx| (case.treat[*idx] - group).abs() < 0.5 && discarded[*idx])
        .count()
}

fn range_of(case: &Case, group: f64) -> (f64, f64) {
    let values = case
        .distance
        .iter()
        .zip(&case.treat)
        .filter(|(_, t)| (**t - group).abs() < 0.5)
        .map(|(d, _)| *d);
    values.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), d| {
        (lo.min(d), hi.max(d))
    })
}

/// For every case, scale and policy the kept units, the trimmed counts and the
/// overlap interval equal MatchIt's. `Treated` and `Control` trim one group
/// each, `Both` trims both, and a unit exactly at the end of the other group's
/// range is kept. The cases without overlap are checked in
/// [`no_overlap_is_reported`].
#[test]
fn trimmed_units_match_matchit() {
    let reference = reference();
    println!(
        "{} / MatchIt {}",
        reference.versions.r, reference.versions.matchit
    );
    let mut boundary_kept = HashMap::<&str, usize>::new();
    for case in reference.cases.iter().filter(|c| c.name != "disjoint") {
        let treated_range = range_of(case, 1.0);
        let control_range = range_of(case, 0.0);
        let lower = treated_range.0.max(control_range.0);
        let upper = treated_range.1.min(control_range.1);
        for (policy, discarded) in [
            (None, &case.discarded.none),
            (Some(CommonSupport::Treated), &case.discarded.treated),
            (Some(CommonSupport::Control), &case.discarded.control),
            (Some(CommonSupport::Both), &case.discarded.both),
        ] {
            let label = format!("{} {} {policy:?}", case.name, case.scale);
            let got = run(case, policy);
            assert_eq!(
                got.kept_anchors,
                expected_kept(case, discarded, 1.0),
                "kept treated units, {label}"
            );
            assert_eq!(
                got.kept_candidates,
                expected_kept(case, discarded, 0.0),
                "kept control units, {label}"
            );
            let diagnostics = &got.outcome.diagnostics;
            assert_eq!(
                diagnostics.common_support_trimmed_anchors,
                count_discarded(case, discarded, 1.0),
                "trimmed treated count, {label}"
            );
            assert_eq!(
                diagnostics.common_support_trimmed_candidates,
                count_discarded(case, discarded, 0.0),
                "trimmed control count, {label}"
            );
            // The ids are the units MatchIt discarded, in the order of the data.
            let discarded_ids = |group: f64| -> Vec<String> {
                (0..case.treat.len())
                    .filter(|idx| (case.treat[*idx] - group).abs() < 0.5 && discarded[*idx])
                    .map(id)
                    .collect()
            };
            assert_eq!(
                diagnostics.common_support_trimmed_ids.anchors,
                discarded_ids(1.0),
                "trimmed treated ids, {label}"
            );
            assert_eq!(
                diagnostics.common_support_trimmed_ids.candidates,
                discarded_ids(0.0),
                "trimmed control ids, {label}"
            );
            if policy.is_some() {
                assert_eq!(
                    diagnostics.common_support_overlap,
                    Some((lower, upper)),
                    "{label}"
                );
                assert_eq!(
                    diagnostics.common_support_anchor_score_bounds,
                    Some(treated_range),
                    "{label}"
                );
                assert_eq!(
                    diagnostics.common_support_candidate_score_bounds,
                    Some(control_range),
                    "{label}"
                );
            }
            if policy == Some(CommonSupport::Both) {
                for group in [1.0, 0.0] {
                    let on_end = (0..case.treat.len()).filter(|idx| {
                        (case.treat[*idx] - group).abs() < 0.5
                            && (case.distance[*idx].to_bits() == lower.to_bits()
                                || case.distance[*idx].to_bits() == upper.to_bits())
                            && !discarded[*idx]
                    });
                    *boundary_kept
                        .entry(if group > 0.5 { "treated" } else { "control" })
                        .or_default() += on_end.count();
                }
            }
        }
    }
    // The reference has units of each group exactly at the ends of the overlap
    // that MatchIt keeps, so the strictness of the comparison is exercised.
    assert!(boundary_kept["treated"] > 0 && boundary_kept["control"] > 0);
}

/// MatchIt trims on the distance it is given, so the probability and the linear
/// predictor scale of one data set discard the same units, and so does ours.
#[test]
fn both_scales_trim_the_same_units() {
    let reference = reference();
    let mut pairs = 0;
    for case in reference.cases.iter().filter(|c| c.scale == "probability") {
        let Some(linear) = reference
            .cases
            .iter()
            .find(|c| c.name == case.name && c.scale == "linear")
        else {
            continue;
        };
        pairs += 1;
        assert_eq!(case.discarded.both, linear.discarded.both, "{}", case.name);
        for policy in [
            CommonSupport::Treated,
            CommonSupport::Control,
            CommonSupport::Both,
        ] {
            let a = run(case, Some(policy));
            let b = run(linear, Some(policy));
            assert_eq!(a.kept_anchors, b.kept_anchors, "{} {policy:?}", case.name);
            assert_eq!(
                a.kept_candidates, b.kept_candidates,
                "{} {policy:?}",
                case.name
            );
        }
    }
    assert!(pairs >= 6);
}

/// With ranges that do not overlap MatchIt discards every unit of the trimmed
/// groups and matches nothing. Here the match is refused with the failure
/// reason `NoOverlap`, for each policy, and nothing is paired.
#[test]
fn no_overlap_is_reported() {
    let reference = reference();
    let case = reference
        .cases
        .iter()
        .find(|c| c.name == "disjoint")
        .expect("disjoint case");
    assert_eq!(count_discarded(case, &case.discarded.treated, 1.0), 3);
    assert_eq!(count_discarded(case, &case.discarded.control, 0.0), 3);
    assert_eq!(count_discarded(case, &case.discarded.both, 1.0), 3);
    assert_eq!(count_discarded(case, &case.discarded.both, 0.0), 3);
    for policy in [
        CommonSupport::Treated,
        CommonSupport::Control,
        CommonSupport::Both,
    ] {
        let got = run(case, Some(policy));
        assert!(
            got.outcome.pairs.is_empty(),
            "{policy:?} {:?}",
            got.outcome.pairs
        );
        assert_eq!(
            got.outcome
                .diagnostics
                .exclusion_counts
                .get(&ExclusionReason::CommonSupportFailure(
                    CommonSupportFailureReason::NoOverlap
                ))
                .copied(),
            Some(1),
            "{policy:?}"
        );
    }
}

/// A policy without propensity scores is refused, not ignored.
#[test]
fn policy_without_scores_is_refused() {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let anchors = vec![BaseRecord::new("a", birth)];
    let candidates = vec![BaseRecord::new("c", birth)];
    let mut criteria = MatchingCriteria::builder().build();
    criteria.common_support = Some(CommonSupport::Both);
    let request = StandardMatchRequest::new(&criteria, DeterministicSelection, &());
    let outcome = match_standard(&anchors, &candidates, request);
    assert!(outcome.pairs.is_empty(), "{:?}", outcome.pairs);
    assert_eq!(
        outcome
            .diagnostics
            .exclusion_counts
            .get(&ExclusionReason::CommonSupportFailure(
                CommonSupportFailureReason::RequiresPropensityScoreMap
            ))
            .copied(),
        Some(1)
    );
}
