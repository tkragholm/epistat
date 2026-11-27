//! Standard matching: random streams, unique keys, birth window under a
//! distance, typed exclusion reasons, control counts and fallback ratios.

use chrono::NaiveDate;
use epistat_matching::MatchOrder;
use epistat_matching::prelude::*;
use epistat_matching::{Assignment, DistanceConfig, ExclusionReason, InvalidCriteriaReason};
use rapidhash::RapidHashMap;
use std::collections::HashSet;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).expect("date")
}

fn rec(id: &str, date: NaiveDate) -> BaseRecord {
    BaseRecord::new(id, date)
}

fn ratio(n: usize) -> MatchRatio {
    MatchRatio::new(n).expect("non-zero")
}

fn scores(pairs: &[(&str, f64)]) -> RapidHashMap<String, f64> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

/// With replacement and a random strategy every anchor should draw its own control.
#[test]
fn replacement_random_selection_draws_a_stream_per_anchor() {
    let anchors: Vec<_> = (0..30)
        .map(|i| rec(&format!("a{i:02}"), d(2010, 1, 1)))
        .collect();
    let candidates: Vec<_> = (0..30)
        .map(|i| rec(&format!("c{i:02}"), d(2010, 1, 1)))
        .collect();
    let out = MatchJob::new_standard(&anchors, &candidates, 7)
        .with_ratio(ratio(1))
        .with_replacement(true)
        .run()
        .expect("a valid job");
    let distinct: HashSet<_> = out.pairs.iter().map(|p| p.control_id.clone()).collect();
    assert!(
        distinct.len() > 5,
        "30 anchors, 30 identical candidates, distinct controls used: {}",
        distinct.len()
    );
}

/// Same without replacement, strata: random draws should not repeat the same relative position
/// in every stratum.
#[test]
fn random_selection_differs_between_strata() {
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for s in 0..20 {
        let mut a = rec(&format!("a{s:02}"), d(2010, 1, 1));
        a.strata.insert("s".into(), format!("s{s}"));
        anchors.push(a);
        for j in 0..10 {
            let mut c = rec(&format!("c{s:02}_{j}"), d(2010, 1, 1));
            c.strata.insert("s".into(), format!("s{s}"));
            candidates.push(c);
        }
    }
    let out = MatchJob::new_standard(&anchors, &candidates, 7)
        .with_ratio(ratio(1))
        .with_exact_match("s")
        .run()
        .expect("a valid job");
    let positions: HashSet<_> = out
        .pairs
        .iter()
        .map(|p| p.control_id.rsplit('_').next().unwrap().to_string())
        .collect();
    assert!(
        positions.len() > 2,
        "relative positions drawn across 20 strata: {positions:?}"
    );
}

/// unique_by_key: controls sharing a key are mutually exclusive. Ratio 2, two controls of one
/// family, no other control: at most one may be used.
#[test]
fn unique_key_not_reused_within_one_anchor_default_engine() {
    let anchors = vec![rec("a1", d(2010, 1, 1))];
    let mut c1 = rec("c1", d(2010, 1, 1));
    let mut c2 = rec("c2", d(2010, 1, 1));
    c1.strata.insert("family".into(), "f".into());
    c2.strata.insert("family".into(), "f".into());
    let candidates = vec![c1, c2];
    let out = MatchJob::new_standard(&anchors, &candidates, 1)
        .with_ratio(ratio(2))
        .with_unique_by_key(Some("family".into()))
        .run()
        .expect("a valid job");
    assert_eq!(out.pairs.len(), 1, "pairs: {:?}", out.pairs);
}

#[test]
fn unique_key_not_reused_within_one_anchor_nearest() {
    let anchors = vec![rec("a1", d(2010, 1, 1))];
    let mut c1 = rec("c1", d(2010, 1, 1));
    let mut c2 = rec("c2", d(2010, 1, 1));
    c1.strata.insert("family".into(), "f".into());
    c2.strata.insert("family".into(), "f".into());
    let candidates = vec![c1, c2];
    let out = MatchJob::new_standard(&anchors, &candidates, 1)
        .with_nearest()
        .with_assignment(Assignment::Sequential)
        .with_ratio(ratio(2))
        .with_unique_by_key(Some("family".into()))
        .run()
        .expect("a valid job");
    assert_eq!(out.pairs.len(), 1, "pairs: {:?}", out.pairs);
}

/// A family must be used once across the whole match, also across exact strata.
#[test]
fn unique_key_not_reused_across_exact_strata() {
    let mut a1 = rec("a1", d(2010, 1, 1));
    let mut a2 = rec("a2", d(2010, 1, 1));
    a1.strata.insert("sex".into(), "M".into());
    a2.strata.insert("sex".into(), "F".into());
    let mut c1 = rec("c1", d(2010, 1, 1));
    let mut c2 = rec("c2", d(2010, 1, 1));
    c1.strata.insert("sex".into(), "M".into());
    c2.strata.insert("sex".into(), "F".into());
    c1.strata.insert("family".into(), "f".into());
    c2.strata.insert("family".into(), "f".into());
    let out = MatchJob::new_standard(&[a1, a2][..], &[c1, c2][..], 1)
        .with_ratio(ratio(1))
        .with_exact_match("sex")
        .with_unique_by_key(Some("family".into()))
        .run()
        .expect("a valid job");
    assert_eq!(out.pairs.len(), 1, "pairs: {:?}", out.pairs);
}

/// Closest order: pair taken after another with the same family in the same round.
#[test]
fn unique_key_not_reused_in_closest_order() {
    let mut anchors = Vec::new();
    for i in 0..2 {
        anchors.push(rec(&format!("a{i}"), d(2010, 1, 1)));
    }
    let mut c1 = rec("c1", d(2010, 1, 1));
    let mut c2 = rec("c2", d(2010, 1, 1));
    c1.strata.insert("family".into(), "f".into());
    c2.strata.insert("family".into(), "f".into());
    let candidates = vec![c1, c2];
    let sc = scores(&[("a0", 0.5), ("a1", 0.6), ("c1", 0.5), ("c2", 0.6)]);
    let cfg = DistanceConfig::propensity_score_map(sc, None);
    let out = MatchJob::new_standard(&anchors, &candidates, 1)
        .with_nearest()
        .with_m_order(MatchOrder::Closest)
        .with_distance_config(cfg)
        .with_unique_by_key(Some("family".into()))
        .run()
        .expect("a valid job");
    assert_eq!(out.pairs.len(), 1, "pairs: {:?}", out.pairs);
}

/// with_birth_window documents: candidates must be within the window of the case.
#[test]
fn birth_window_applies_with_a_distance_config() {
    let anchors = vec![rec("a1", d(2010, 1, 1))];
    let candidates = vec![rec("c1", d(2015, 1, 1))];
    let sc = scores(&[("a1", 0.5), ("c1", 0.5)]);
    let out = MatchJob::new_standard(&anchors, &candidates, 1)
        .with_birth_window(BirthDateWindowDays::new(1).expect("ok"))
        .with_distance_config(DistanceConfig::propensity_score_map(sc, None))
        .run()
        .expect("a valid job");
    assert!(
        out.pairs.is_empty(),
        "matched across 5 years with window 1: {:?}",
        out.pairs
    );
}

/// The risk-set path: one case, ratio 2, two controls of one family and no other control.
#[test]
fn unique_key_not_reused_within_one_case_risk_set() {
    use epistat_matching::RoleTransitionRecord;
    let case = RoleTransitionRecord::from_record(rec("a1", d(2010, 1, 1)), Some(d(2012, 1, 1)));
    let mut c1 = rec("c1", d(2010, 1, 1));
    let mut c2 = rec("c2", d(2010, 1, 1));
    c1.strata.insert("family".into(), "f".into());
    c2.strata.insert("family".into(), "f".into());
    let cohort = vec![
        case,
        RoleTransitionRecord::from_record(c1, None),
        RoleTransitionRecord::from_record(c2, None),
    ];
    let out = MatchJob::new_transition(&cohort, 1)
        .with_ratio(ratio(2))
        .with_unique_by_key(Some("family".into()))
        .run()
        .expect("a valid job");
    assert_eq!(out.pairs.len(), 1, "pairs: {:?}", out.pairs);
}

/// Exclusion diagnostics use the typed reason the constraint's text names.
#[test]
fn exclusion_reasons_map_to_typed_variants() {
    use epistat_matching::ConstraintReason;
    let anchors = vec![rec("a1", d(2010, 1, 1)), rec("a2", d(2010, 1, 1))];
    let candidates = vec![rec("c1", d(2010, 1, 1))];
    let out = MatchJob::new_standard(&anchors, &candidates, 1)
        .with_ratio(ratio(1))
        .run()
        .expect("a valid job");
    let keys: Vec<_> = out.diagnostics.exclusion_counts.keys().collect();
    assert!(
        out.diagnostics
            .exclusion_counts
            .contains_key(&ExclusionReason::Constraint(
                ConstraintReason::ReplacementDisallowed
            )),
        "exclusion keys: {keys:?}"
    );
}

/// With replacement the outcome's used_controls counts the distinct controls used.
#[test]
fn used_controls_with_replacement() {
    let anchors = vec![rec("a1", d(2010, 1, 1)), rec("a2", d(2010, 1, 1))];
    let candidates = vec![rec("c1", d(2010, 1, 1))];
    let out = MatchJob::new_standard(&anchors, &candidates, 1)
        .with_ratio(ratio(1))
        .with_replacement(true)
        .run()
        .expect("a valid job");
    assert_eq!(out.pairs.len(), 2);
    assert_eq!(out.used_controls, 1, "used_controls {}", out.used_controls);
}

#[test]
fn used_controls_with_replacement_nearest() {
    let anchors = vec![rec("a1", d(2010, 1, 1)), rec("a2", d(2010, 1, 1))];
    let candidates = vec![rec("c1", d(2010, 1, 1))];
    let out = MatchJob::new_standard(&anchors, &candidates, 1)
        .with_nearest()
        .with_ratio(ratio(1))
        .with_replacement(true)
        .run()
        .expect("a valid job");
    assert_eq!(out.used_controls, 1, "used_controls {}", out.used_controls);
}

/// A fallback ratio above the primary ratio is an invalid job.
#[test]
fn fallback_ratio_above_primary_is_an_error() {
    let anchors = vec![rec("a1", d(2010, 1, 1))];
    let candidates: Vec<_> = (0..10)
        .map(|i| rec(&format!("c{i}"), d(2010, 1, 1)))
        .collect();
    let out = MatchJob::new_standard(&anchors, &candidates, 1)
        .with_ratio(ratio(2))
        .with_ratio_fallback(vec![ratio(4)])
        .run();
    assert_eq!(
        out.unwrap_err(),
        Error::InvalidCriteria(InvalidCriteriaReason::FallbackRatioAbovePrimary)
    );
}

/// An anchor matched on a declared fallback tier is not a shortfall, and is counted as
/// matched on a lower tier.
#[test]
fn fallback_tier_below_primary_is_counted_apart_from_shortfall() {
    let anchors = vec![rec("a1", d(2010, 1, 1))];
    let candidates: Vec<_> = (0..2)
        .map(|i| rec(&format!("c{i}"), d(2010, 1, 1)))
        .collect();
    let out = MatchJob::new_standard(&anchors, &candidates, 1)
        .with_ratio(ratio(4))
        .with_ratio_fallback(vec![ratio(2)])
        .run()
        .expect("a valid job");
    assert_eq!(out.pairs.len(), 2);
    assert_eq!(
        out.diagnostics.anchors_below_required_ratio, 0,
        "{:?}",
        out.diagnostics
    );
    assert_eq!(
        out.diagnostics.anchors_on_lower_tier, 1,
        "{:?}",
        out.diagnostics
    );
}
