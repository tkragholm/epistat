//! Common-support trimming in risk-set matching.
//!
//! A case is trimmed when its score is strictly outside the range of the
//! scores of the people at risk on its event date, whatever the matching
//! constraints are. A candidate is trimmed when its score is outside the range
//! of the cases' scores.
//!
//! The reference is `tests/data/ccwc_support_ranges.json`, written by
//! `tests/data/ccwc_support_ranges.R`. The script takes each case's risk set
//! from `Epi::ccwc` without matching variables (controls = 1000 returns the
//! whole set), checks it against the definition written from the data, and
//! computes the smallest and largest score in each set in R. The file records
//! the versions of R and Epi. The other tests use small cohorts built by hand.

use chrono::{Days, NaiveDate};
use epistat_matching::DeterministicSelection;
use epistat_matching::prelude::*;
use epistat_testkit::{POOLS, assert_same_on_pools, load_json};
use serde::Deserialize;
use std::collections::BTreeSet;

type Rec = RoleTransitionRecord<BaseRecord>;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    entry: Vec<i64>,
    event: Vec<Option<i64>>,
    score: Vec<f64>,
    cases: Vec<usize>,
    set_size: Vec<usize>,
    outside: Vec<bool>,
    case_range: Vec<f64>,
    at_bounds: usize,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    #[serde(rename = "Epi")]
    epi: String,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/ccwc_support_ranges.json"
    ))
}

fn day(days: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1)
        .expect("valid date")
        .checked_add_days(Days::new(u64::try_from(days).expect("after 1970")))
        .expect("in range")
}

fn birth() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date")
}

fn person(id: &str, entry: Option<i64>, event: Option<i64>) -> Rec {
    let mut record =
        RoleTransitionRecord::from_record(BaseRecord::new(id, birth()), event.map(day));
    record.entry_date = entry.map(day);
    record
}

/// Match every case once, with replacement and in a fixed order, with scores
/// and a common-support policy. The ids of the cases that found a control are
/// returned with the outcome.
fn run<P: RiskSetPolicy<Rec> + Send + Sync>(
    cohort: &[Rec],
    scores: &[(String, f64)],
    support: Option<CommonSupport>,
    policy: P,
) -> (BTreeSet<String>, MatchOutcome) {
    let config = DistanceConfig::propensity_score_map(scores.iter().cloned().collect(), None);
    let outcome = MatchJob::new_transition(cohort, 0)
        .with_ratio(MatchRatio::new(1).expect("ratio"))
        .with_birth_window(BirthDateWindowDays::new(30_000).expect("window"))
        .with_age_limit(AgeLimitYears::new(100).expect("age limit"))
        .with_replacement(true)
        .with_distance_config(config)
        .with_common_support(support)
        .with_risk_set_policy(policy)
        .with_strategy(DeterministicSelection)
        .run();
    let matched = outcome.pairs.iter().map(|p| p.case_id.clone()).collect();
    (matched, outcome)
}

fn scores_of(ids: &[&str], values: &[f64]) -> Vec<(String, f64)> {
    ids.iter()
        .map(|id| (*id.to_string()).to_string())
        .zip(values.iter().copied())
        .collect()
}

#[test]
fn trimmed_cases_agree_with_ranges_over_the_ccwc_risk_sets() {
    let reference = reference();
    assert!(reference.versions.r.starts_with("R version"));
    assert_ne!(reference.versions.epi, "");
    assert!(
        reference.at_bounds > 0,
        "the reference has no tie at a bound"
    );
    assert!(reference.outside.iter().any(|&o| o));
    assert!(reference.set_size.iter().all(|&s| s > 0));
    assert!(reference.case_range[0] < reference.case_range[1]);

    let n = reference.score.len();
    let id = |row: usize| format!("p{row:03}");
    let cohort: Vec<Rec> = (0..n)
        .map(|i| person(&id(i + 1), Some(reference.entry[i]), reference.event[i]))
        .collect();
    let scores: Vec<(String, f64)> = (0..n).map(|i| (id(i + 1), reference.score[i])).collect();

    let (matched, outcome) = run(
        &cohort,
        &scores,
        Some(CommonSupport::Treated),
        DefaultRiskSetPolicy,
    );
    let expected: BTreeSet<String> = reference
        .cases
        .iter()
        .zip(&reference.outside)
        .filter(|(_, outside)| !**outside)
        .map(|(row, _)| id(*row))
        .collect();
    assert_eq!(matched, expected);
    let trimmed = reference.outside.iter().filter(|&&o| o).count();
    assert_eq!(outcome.diagnostics.common_support_trimmed_anchors, trimmed);
    assert_eq!(outcome.diagnostics.common_support_trimmed_candidates, 0);

    // The same cohort through a policy that states the default rule itself,
    // which takes the scan over every record instead of the sweep.
    let (scanned, _) = run(&cohort, &scores, Some(CommonSupport::Treated), SameRule);
    assert_eq!(scanned, expected);
}

/// The default at-risk rule, stated as a policy of its own.
struct SameRule;

impl RiskSetPolicy<Rec> for SameRule {
    fn is_control_at_risk(&self, case: &Rec, control: &Rec, date: NaiveDate) -> bool {
        DefaultRiskSetPolicy.is_control_at_risk(case, control, date)
    }
}

/// Case `c1` on day 10 has score 5. Its risk set holds `a` and `b` with scores
/// 1 and 2. Case `c2` enters on day 20, has score 9 on day 100 and then has
/// `a`, `b` and `d` (score 8) at risk, so the cohort as a whole spans 1 to 9 and
/// contains 5.
fn outside_own_risk_set() -> (Vec<Rec>, Vec<(String, f64)>) {
    let cohort = vec![
        person("a", Some(0), None),
        person("b", Some(0), None),
        person("d", Some(50), None),
        person("c1", Some(0), Some(10)),
        person("c2", Some(20), Some(100)),
    ];
    let scores = scores_of(&["a", "b", "d", "c1", "c2"], &[1.0, 2.0, 8.0, 5.0, 9.0]);
    (cohort, scores)
}

#[test]
fn a_case_outside_its_own_risk_set_but_inside_the_cohort_is_trimmed() {
    let (cohort, scores) = outside_own_risk_set();
    let (matched, outcome) = run(&cohort, &scores, None, DefaultRiskSetPolicy);
    assert_eq!(matched, ["c1", "c2"].map(String::from).into());
    assert_eq!(outcome.diagnostics.common_support_trimmed_anchors, 0);

    for policy in [CommonSupport::Treated, CommonSupport::Both] {
        let (matched, outcome) = run(&cohort, &scores, Some(policy), DefaultRiskSetPolicy);
        // c1: 5 is above the range 1 to 2 of those at risk on day 10.
        // c2: 9 is above the range 1 to 8 of those at risk on day 100.
        assert!(matched.is_empty(), "{policy:?}: {matched:?}");
        assert_eq!(outcome.diagnostics.common_support_trimmed_anchors, 2);
    }
}

#[test]
fn a_case_inside_its_own_risk_set_range_is_kept_whatever_the_cohort_holds() {
    // c1 has score 5 on day 10 with a and b at risk, scores 1 and 9. A later
    // case c2 enters after day 10, has score 20 and sets the cohort's range wide, but a case is
    // judged only against the people at risk on its own date.
    let cohort = vec![
        person("a", Some(0), None),
        person("b", Some(0), None),
        person("c1", Some(0), Some(10)),
        person("c2", Some(20), Some(100)),
    ];
    let scores = scores_of(&["a", "b", "c1", "c2"], &[1.0, 9.0, 5.0, 20.0]);
    let (matched, outcome) = run(
        &cohort,
        &scores,
        Some(CommonSupport::Treated),
        DefaultRiskSetPolicy,
    );
    assert_eq!(matched, ["c1"].map(String::from).into());
    assert_eq!(outcome.diagnostics.common_support_trimmed_anchors, 1);
}

#[test]
fn a_score_equal_to_an_end_of_the_range_is_kept() {
    let cohort = vec![
        person("a", Some(0), None),
        person("b", Some(0), None),
        person("lo", Some(0), Some(10)),
        person("hi", Some(0), Some(20)),
    ];
    // lo has the score of a, hi the score of b.
    let scores = scores_of(&["a", "b", "lo", "hi"], &[1.0, 3.0, 1.0, 3.0]);
    let (matched, outcome) = run(
        &cohort,
        &scores,
        Some(CommonSupport::Treated),
        DefaultRiskSetPolicy,
    );
    assert_eq!(matched, ["lo", "hi"].map(String::from).into());
    assert_eq!(outcome.diagnostics.common_support_trimmed_anchors, 0);
}

#[test]
fn people_who_have_not_entered_or_have_left_are_not_in_the_range() {
    // On day 10 only a is at risk: b enters on day 11, e has its event on
    // day 10 and f entered on day 10 (at risk, since entry on the day counts).
    let cohort = vec![
        person("a", Some(0), None),
        person("b", Some(11), None),
        person("e", Some(0), Some(10)),
        person("f", Some(10), None),
        person("c", Some(0), Some(10)),
    ];
    let scores = scores_of(&["a", "b", "e", "f", "c"], &[1.0, 50.0, 40.0, 2.0, 3.0]);
    let (matched, _) = run(
        &cohort,
        &scores,
        Some(CommonSupport::Treated),
        DefaultRiskSetPolicy,
    );
    // Range of a and f is 1 to 2. c (3) and e (40) are cases outside it.
    assert!(matched.is_empty(), "{matched:?}");
}

#[test]
fn a_case_with_nobody_at_risk_is_not_trimmed() {
    let cohort = vec![person("c", Some(0), Some(10)), person("a", Some(20), None)];
    let scores = scores_of(&["c", "a"], &[1.0, 1.0]);
    let (matched, outcome) = run(
        &cohort,
        &scores,
        Some(CommonSupport::Treated),
        DefaultRiskSetPolicy,
    );
    assert!(matched.is_empty());
    assert_eq!(outcome.diagnostics.common_support_trimmed_anchors, 0);
    assert_eq!(outcome.diagnostics.anchors_with_no_candidates, 1);
}

#[test]
fn candidates_outside_the_range_of_the_cases_are_trimmed_under_control() {
    // The cases have scores 2 and 3. Candidate z (score 10) is outside that
    // range, y (score 2) is on its end.
    let cohort = vec![
        person("c1", Some(0), Some(10)),
        person("c2", Some(0), Some(20)),
        person("y", Some(0), None),
        person("z", Some(0), None),
    ];
    let scores = scores_of(&["c1", "c2", "y", "z"], &[2.0, 3.0, 2.0, 10.0]);
    for policy in [CommonSupport::Control, CommonSupport::Both] {
        let (matched, outcome) = run(&cohort, &scores, Some(policy), DefaultRiskSetPolicy);
        assert_eq!(outcome.diagnostics.common_support_trimmed_candidates, 1);
        assert!(outcome.pairs.iter().all(|p| p.control_id != "z"));
        if policy == CommonSupport::Control {
            assert_eq!(matched, ["c1", "c2"].map(String::from).into());
        }
    }
}

#[test]
fn trimming_does_not_depend_on_the_number_of_threads() {
    let reference = reference();
    let n = reference.score.len();
    let id = |row: usize| format!("p{row:03}");
    let cohort: Vec<Rec> = (0..n)
        .map(|i| person(&id(i + 1), Some(reference.entry[i]), reference.event[i]))
        .collect();
    let scores: Vec<(String, f64)> = (0..n).map(|i| (id(i + 1), reference.score[i])).collect();
    assert_same_on_pools(&POOLS, || {
        let (_, outcome) = run(
            &cohort,
            &scores,
            Some(CommonSupport::Both),
            DefaultRiskSetPolicy,
        );
        let mut digest: Vec<f64> = vec![
            outcome.diagnostics.common_support_trimmed_anchors as f64,
            outcome.diagnostics.common_support_trimmed_candidates as f64,
        ];
        for pair in &outcome.pairs {
            digest.push(pair.case_id.len() as f64);
            digest.extend(pair.case_id.bytes().map(f64::from));
            digest.extend(pair.control_id.bytes().map(f64::from));
        }
        digest
    });
}
