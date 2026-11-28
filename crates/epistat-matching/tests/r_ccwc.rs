//! Risk-set matching checked against `Epi::ccwc`.
//!
//! The reference is `tests/data/ccwc_risk_sets.json`, written by
//! `tests/data/ccwc_risk_sets.R` on a simulated cohort of 400 people with two
//! matching variables, staggered entry, deaths, emigrations, tied failure
//! times and people leaving the risk set exactly on a failure day. The file
//! records the versions of R and Epi.
//!
//! `ccwc` samples at random, so individual sets cannot agree. What is compared
//! is what both define:
//!
//! * the risk set of every case, from `ccwc(controls = 1000)`, which returns
//!   the whole set, against the controls the engine returns with a ratio
//!   larger than the cohort and replacement allowed;
//! * the number of cases with fewer than the requested controls and with none;
//! * how often each member of a risk set is drawn when one control is drawn per
//!   case, over a fixed list of seeds, against a chi-square threshold derived
//!   from the level and the number of tests.
//!
//! `ccwc` keeps a person with exit time equal to the failure time in the risk
//! set. The engine drops a person from the day of death, emigration or first
//! event. The reference data therefore gives `ccwc` exit = end of risk minus
//! one day, which makes "at risk at t" the same statement. The test
//! `ccwc_keeps_people_who_leave_on_the_failure_day` checks that the
//! unshifted data does give a different set.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::{Days, NaiveDate};
use epistat_matching::DeterministicSelection;
use epistat_matching::prelude::*;
use epistat_matching::{DefaultRiskSetPolicy, RiskSetPolicy};
use epistat_stats::dist::chi_square::sf as chi_square_sf;

use r_reference::{Versions, assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    age_limit_years: u8,
    k_small: usize,
    reps: usize,
    people: People,
    calendar: Scale,
    age: Scale,
    draws: Vec<Draw>,
    r_chi_square: Vec<f64>,
}

#[derive(Deserialize)]
struct People {
    sex: Vec<String>,
    area: Vec<String>,
    birth: Vec<i64>,
    entry: Vec<i64>,
    event: Vec<Option<i64>>,
    death: Vec<Option<i64>>,
    emigration: Vec<Option<i64>>,
    end: i64,
    in_age: Vec<i64>,
}

#[derive(Deserialize)]
struct Scale {
    cases: Vec<usize>,
    risk_set: Vec<Vec<usize>>,
    ncase_at_time: Vec<usize>,
    raw_extra: Vec<Vec<usize>>,
    k4_warnings: Vec<String>,
    k4_controls_returned: Vec<usize>,
    k4_controls_in_several_sets: usize,
    k4_future_cases_as_controls: usize,
}

#[derive(Deserialize)]
struct Draw {
    case: usize,
    members: Vec<usize>,
    counts: Vec<usize>,
}

fn reference() -> Reference {
    load_reference("ccwc_risk_sets.json")
}

type Rec = RoleTransitionRecord<BaseRecord>;

fn day(days: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1)
        .expect("valid date")
        .checked_add_days(Days::new(u64::try_from(days).expect("after 1970")))
        .expect("in range")
}

/// Row numbers are the 1-based rows of the R data frame; the id is `p` and
/// the row number.
fn id(row: usize) -> String {
    format!("p{row:03}")
}

/// The cohort with no entry date set, for the route through a risk-set policy.
fn records(people: &People) -> Vec<Rec> {
    (0..people.sex.len())
        .map(|i| {
            let mut base = BaseRecord::new(id(i + 1), day(people.birth[i]));
            base.strata.insert("sex".into(), people.sex[i].clone());
            base.strata.insert("area".into(), people.area[i].clone());
            if let Some(d) = people.death[i] {
                base = base.with_death_date(day(d));
            }
            if let Some(d) = people.emigration[i] {
                base = base.with_emigration_date(day(d));
            }
            RoleTransitionRecord::from_record(base, people.event[i].map(day))
        })
        .collect()
}

/// The cohort with each person's entry date on the record.
fn records_with_entry(people: &People) -> Vec<Rec> {
    let mut recs = records(people);
    for (rec, entry) in recs.iter_mut().zip(&people.entry) {
        rec.entry_date = Some(day(*entry));
    }
    recs
}

/// Calendar scale through a policy: a control is at risk when it has entered on
/// or before the case's index date. Leaving through death, emigration or an
/// event is handled by the engine's own rules.
struct EntryPolicy {
    entry: HashMap<String, NaiveDate>,
}

impl RiskSetPolicy<Rec> for EntryPolicy {
    fn is_control_at_risk(&self, case: &Rec, control: &Rec, case_event_date: NaiveDate) -> bool {
        self.entry[control.id()] <= case_event_date
            && DefaultRiskSetPolicy.is_control_at_risk(case, control, case_event_date)
    }
}

/// Age scale: a control is at risk when, on the day it reaches the case's age
/// at the case's index date, it has entered and not yet left. The engine's
/// alive and resident checks are calendar checks, so this policy carries every
/// condition.
struct AgePolicy {
    entry: HashMap<String, NaiveDate>,
    risk_end: HashMap<String, NaiveDate>,
}

impl RiskSetPolicy<Rec> for AgePolicy {
    fn is_control_at_risk(&self, case: &Rec, control: &Rec, case_event_date: NaiveDate) -> bool {
        let age = case_event_date - case.birth_date();
        let at = control.birth_date() + age;
        self.entry[control.id()] <= at && at < self.risk_end[control.id()]
    }
}

fn entry_map(people: &People) -> HashMap<String, NaiveDate> {
    (0..people.sex.len())
        .map(|i| (id(i + 1), day(people.entry[i])))
        .collect()
}

fn risk_end_map(people: &People) -> HashMap<String, NaiveDate> {
    (0..people.sex.len())
        .map(|i| {
            let end = [
                people.event[i],
                people.death[i],
                people.emigration[i],
                Some(people.end + 1),
            ]
            .into_iter()
            .flatten()
            .min()
            .expect("the end of the study is always present");
            (id(i + 1), day(end))
        })
        .collect()
}

fn ratio(k: usize) -> MatchRatio {
    MatchRatio::new(k).expect("non-zero ratio")
}

fn window() -> BirthDateWindowDays {
    // About 82 years: no birth-date restriction on a cohort born in a decade.
    BirthDateWindowDays::new(30_000).expect("non-negative")
}

fn age_limit(reference: &Reference) -> AgeLimitYears {
    AgeLimitYears::new(reference.age_limit_years).expect("positive")
}

/// Controls per case in the order of the case ids.
fn by_case(outcome: &MatchOutcome) -> BTreeMap<String, Vec<String>> {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for pair in &outcome.pairs {
        map.entry(pair.case_id.clone())
            .or_default()
            .push(pair.control_id.clone());
    }
    map
}

/// Calendar scale with the entry date on the records, no policy.
fn run_calendar(
    reference: &Reference,
    recs: &[Rec],
    k: usize,
    replacement: bool,
    seed: Option<u64>,
) -> MatchOutcome {
    let job = MatchJob::new_transition(recs, seed.unwrap_or(0))
        .with_ratio(ratio(k))
        .with_birth_window(window())
        .with_age_limit(age_limit(reference))
        .with_exact_matches(["sex", "area"])
        .with_replacement(replacement)
        .with_alive_check()
        .with_resident_check();
    if seed.is_some() {
        job.run().expect("a valid job")
    } else {
        job.with_strategy(DeterministicSelection)
            .run()
            .expect("a valid job")
    }
}

/// Calendar scale with the entry date in a risk-set policy and none on the
/// records.
fn run_calendar_policy(
    reference: &Reference,
    recs: &[Rec],
    k: usize,
    replacement: bool,
    seed: Option<u64>,
) -> MatchOutcome {
    let policy = EntryPolicy {
        entry: entry_map(&reference.people),
    };
    let job = MatchJob::new_transition(recs, seed.unwrap_or(0))
        .with_ratio(ratio(k))
        .with_birth_window(window())
        .with_age_limit(age_limit(reference))
        .with_exact_matches(["sex", "area"])
        .with_replacement(replacement)
        .with_alive_check()
        .with_resident_check()
        .with_risk_set_policy(policy);
    if seed.is_some() {
        job.run().expect("a valid job")
    } else {
        job.with_strategy(DeterministicSelection)
            .run()
            .expect("a valid job")
    }
}

fn run_age(reference: &Reference, recs: &[Rec], k: usize) -> MatchOutcome {
    let policy = AgePolicy {
        entry: entry_map(&reference.people),
        risk_end: risk_end_map(&reference.people),
    };
    MatchJob::new_transition(recs, 0)
        .with_ratio(ratio(k))
        .with_birth_window(window())
        .with_age_limit(age_limit(reference))
        .with_exact_matches(["sex", "area"])
        .with_replacement(true)
        .with_risk_set_policy(policy)
        .with_strategy(DeterministicSelection)
        .run()
        .expect("a valid job")
}

fn expected_sets(scale: &Scale) -> BTreeMap<String, BTreeSet<String>> {
    scale
        .cases
        .iter()
        .zip(&scale.risk_set)
        .filter(|(_, set)| !set.is_empty())
        .map(|(case, set)| (id(*case), set.iter().map(|m| id(*m)).collect()))
        .collect()
}

fn assert_same_sets(outcome: &MatchOutcome, scale: &Scale, label: &str) {
    let got = by_case(outcome);
    let expected = expected_sets(scale);
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>(),
        "{label}: the cases with a non-empty risk set"
    );
    for (case, controls) in &got {
        let distinct: BTreeSet<String> = controls.iter().cloned().collect();
        assert_eq!(
            distinct.len(),
            controls.len(),
            "{label}: {case} gets a control twice"
        );
        assert_eq!(&distinct, &expected[case], "{label}: risk set of {case}");
    }
}

#[test]
fn versions_are_recorded() {
    let reference = reference();
    assert_r_version(&reference.versions["R"]);
    assert_eq!(reference.versions["Epi"], "2.66");
}

/// With a ratio above every risk set and replacement, the engine returns each
/// case's whole risk set. Those sets equal the ones `ccwc` returns when asked
/// for more controls than there are, on the calendar scale, with the entry
/// date on the records. The data has people who enter on a case's failure day,
/// who are in the set, as `entry <= t` in `ccwc` says.
#[test]
fn calendar_risk_sets_equal_ccwc() {
    let reference = reference();
    let recs = records_with_entry(&reference.people);
    let outcome = run_calendar(&reference, &recs, 10_000, true, None);
    assert_eq!(
        outcome.diagnostics.total_cases_evaluated,
        reference.calendar.cases.len(),
        "the cases the age limit leaves"
    );
    assert_same_sets(&outcome, &reference.calendar, "calendar");
}

/// The risk-set policy route to the same sets gives the same pairs as the
/// entry date on the records, and a cohort without entry dates differs from
/// both, so the entry date is doing the work.
#[test]
fn entry_policy_and_entry_date_give_the_same_risk_sets() {
    let reference = reference();
    let plain = records(&reference.people);
    let with_entry = records_with_entry(&reference.people);
    for (k, replacement, seed) in [
        (10_000, true, None),
        (2, false, Some(5)),
        (1, true, Some(9)),
    ] {
        let native = run_calendar(&reference, &with_entry, k, replacement, seed);
        let policy = run_calendar_policy(&reference, &plain, k, replacement, seed);
        assert_eq!(
            native.pairs, policy.pairs,
            "k={k} replacement={replacement}"
        );
        assert_eq!(native.matched_cases, policy.matched_cases);
        assert_eq!(native.unmatched_cases, policy.unmatched_cases);
    }
    let ignored = run_calendar(&reference, &plain, 10_000, true, None);
    let native = run_calendar(&reference, &with_entry, 10_000, true, None);
    assert_ne!(ignored.pairs, native.pairs);
}

/// The same on the age scale (`origin = date of birth` in `ccwc`), through a
/// risk-set policy: the engine has no native age time scale.
#[test]
fn age_scale_risk_sets_equal_ccwc() {
    let reference = reference();
    let recs = records(&reference.people);
    let outcome = run_age(&reference, &recs, 10_000);
    assert_eq!(
        outcome.diagnostics.total_cases_evaluated,
        reference.age.cases.len()
    );
    assert_same_sets(&outcome, &reference.age, "age");
}

/// The cases are the people whose event falls before the age limit, the same
/// people `ccwc` gets as failures.
#[test]
fn age_limit_selects_the_cases_ccwc_gets_as_failures() {
    let reference = reference();
    let expected: Vec<usize> = (0..reference.people.in_age.len())
        .filter(|i| reference.people.in_age[*i] == 1)
        .map(|i| i + 1)
        .collect();
    assert_eq!(expected, reference.calendar.cases);
    assert!(
        reference
            .people
            .event
            .iter()
            .zip(&reference.people.in_age)
            .any(|(e, a)| e.is_some() && *a == 0),
        "the data has an event after the age limit"
    );
}

/// `ccwc` keeps a person whose exit equals the failure time in the risk set,
/// the engine does not. On the data where exit is the day of death,
/// emigration or an over-age event, the sets differ by exactly those people.
#[test]
fn ccwc_keeps_people_who_leave_on_the_failure_day() {
    let reference = reference();
    let recs = records_with_entry(&reference.people);
    let outcome = run_calendar(&reference, &recs, 10_000, true, None);
    let got = by_case(&outcome);
    let mut differing = 0;
    for (case, extra) in reference
        .calendar
        .cases
        .iter()
        .zip(&reference.calendar.raw_extra)
    {
        for person in extra {
            differing += 1;
            let in_ours = got
                .get(&id(*case))
                .is_some_and(|set| set.contains(&id(*person)));
            assert!(!in_ours, "{person} left on the day of case {case}");
            let leaves = reference.people.death[person - 1]
                .into_iter()
                .chain(reference.people.emigration[person - 1])
                .chain(reference.people.event[person - 1])
                .any(|d| Some(d) == reference.people.event[case - 1]);
            assert!(leaves, "{person} is not leaving on the day of case {case}");
        }
    }
    assert!(differing >= 6, "the data has boundary people: {differing}");
}

fn count_warning(warnings: &[String], tail: &str) -> usize {
    warnings
        .iter()
        .find_map(|w| w.strip_suffix(tail).and_then(|n| n.trim().parse().ok()))
        .unwrap_or(0)
}

/// Cases with too few controls. The engine counts per case, `ccwc` per set:
/// tied cases share one set that must hold `controls` times the number of
/// tied cases. The counts agree for every case that is alone at its time. For
/// the tied set with 4 or more but fewer than 8 people at risk, `ccwc` reports
/// an incomplete set and returns the whole risk set, and the engine gives each
/// of the two cases 4 controls.
#[test]
fn shortfall_counts_follow_ccwc_for_cases_alone_at_their_time() {
    let reference = reference();
    let recs = records_with_entry(&reference.people);
    let k = reference.k_small;
    let scale = &reference.calendar;
    let sizes: Vec<usize> = scale.risk_set.iter().map(Vec::len).collect();
    let p = &reference.people;

    // ccwc's sets: one per (cell, failure day).
    let mut sets: BTreeMap<(String, String, i64), (usize, usize)> = BTreeMap::new();
    for (q, case) in scale.cases.iter().enumerate() {
        let key = (
            p.sex[case - 1].clone(),
            p.area[case - 1].clone(),
            p.event[case - 1].expect("a case has an event"),
        );
        let entry = sets.entry(key).or_insert((0, sizes[q]));
        entry.0 += 1;
        assert_eq!(entry.1, sizes[q], "tied cases share a risk set");
    }
    let ccwc_nomatch: usize = sets.values().filter(|(_, m)| *m == 0).map(|(n, _)| n).sum();
    let ccwc_incomplete = sets.values().filter(|(n, m)| *m > 0 && *m < k * n).count();
    assert_eq!(
        count_warning(&scale.k4_warnings, "cases could not be matched"),
        ccwc_nomatch
    );
    assert_eq!(
        count_warning(&scale.k4_warnings, "case-control sets are incomplete"),
        ccwc_incomplete
    );
    assert!(
        scale
            .k4_warnings
            .iter()
            .any(|w| w == "there were tied failure times")
    );

    let outcome = run_calendar(&reference, &recs, k, true, Some(1));
    let none = sizes.iter().filter(|m| **m == 0).count();
    let below = sizes.iter().filter(|m| **m > 0 && **m < k).count();
    assert_eq!(outcome.diagnostics.cases_with_no_controls, none);
    assert_eq!(outcome.diagnostics.cases_below_required_ratio, below);
    assert_eq!(none, ccwc_nomatch);

    let got = by_case(&outcome);
    // Per ccwc set: the controls the engine gives all its cases together and
    // the controls ccwc returns for the set.
    let mut per_set: BTreeMap<(String, String, i64), (usize, usize)> = BTreeMap::new();
    for (q, case) in scale.cases.iter().enumerate() {
        let ours = got.get(&id(*case)).map_or(0, Vec::len);
        assert_eq!(ours, sizes[q].min(k), "controls of case {case}");
        let key = (
            p.sex[case - 1].clone(),
            p.area[case - 1].clone(),
            p.event[case - 1].expect("a case has an event"),
        );
        let entry = per_set
            .entry(key)
            .or_insert((0, scale.k4_controls_returned[q]));
        entry.0 += ours;
        assert_eq!(entry.1, scale.k4_controls_returned[q]);
        if scale.ncase_at_time[q] == 1 {
            assert_eq!(ours, scale.k4_controls_returned[q], "case {case}");
        }
    }
    // The tied set with 5 at risk: ccwc returns 5 controls for the two cases,
    // the engine 4 for each. Every other set agrees, and ccwc counts one
    // incomplete set where the engine counts no case below the ratio.
    let differing: Vec<_> = per_set.values().filter(|(a, b)| a != b).collect();
    assert_eq!(differing, vec![&(8, 5)]);
    assert_eq!(ccwc_incomplete, below + 1);
}

fn uses(outcome: &MatchOutcome) -> HashMap<&str, usize> {
    let mut count: HashMap<&str, usize> = HashMap::new();
    for pair in &outcome.pairs {
        *count.entry(pair.control_id.as_str()).or_default() += 1;
    }
    count
}

/// Without replacement a control serves one case only. `ccwc` samples every
/// set independently: a person is a control for several cases, and a person
/// who fails later is a control for earlier cases, in both.
#[test]
fn replacement_and_future_cases() {
    let reference = reference();
    let recs = records_with_entry(&reference.people);
    let k = reference.k_small;
    assert!(reference.calendar.k4_controls_in_several_sets > 0);
    assert!(reference.calendar.k4_future_cases_as_controls > 0);

    let case_ids: BTreeSet<String> = reference.calendar.cases.iter().map(|c| id(*c)).collect();
    let with = run_calendar(&reference, &recs, k, true, Some(7));
    assert!(uses(&with).values().any(|n| *n > 1));
    assert!(
        with.pairs.iter().any(|p| case_ids.contains(&p.control_id)),
        "a future case serves as a control"
    );
    let without = run_calendar(&reference, &recs, k, false, Some(7));
    assert!(uses(&without).values().all(|n| *n == 1));
    assert!(
        without
            .pairs
            .iter()
            .any(|p| case_ids.contains(&p.control_id)),
        "a future case serves as a control without replacement too"
    );
    // The engine's default can leave cases short that replacement would fill.
    assert!(without.diagnostics.cases_with_no_controls >= with.diagnostics.cases_with_no_controls);
}

/// Cases tied at one time are in neither's risk set. `ccwc` makes one set for
/// them with distinct controls; the engine draws for each separately, so with
/// replacement they can share a control, without replacement they never do.
#[test]
fn tied_cases() {
    let reference = reference();
    let recs = records_with_entry(&reference.people);
    let scale = &reference.calendar;
    let p = &reference.people;
    let tied: Vec<usize> = (0..scale.cases.len())
        .filter(|q| scale.ncase_at_time[*q] > 1)
        .collect();
    let same_set = |q: usize, o: usize| {
        let (a, b) = (scale.cases[q] - 1, scale.cases[o] - 1);
        q != o && p.event[a] == p.event[b] && p.sex[a] == p.sex[b] && p.area[a] == p.area[b]
    };
    let mut pairs = Vec::new();
    for q in &tied {
        for o in &tied {
            if same_set(*q, *o) {
                assert!(!scale.risk_set[*q].contains(&scale.cases[*o]));
                if q < o {
                    pairs.push((*q, *o));
                }
            }
        }
    }
    assert!(pairs.len() >= 3, "tied pairs: {}", pairs.len());

    // One control each, without replacement: the cases of a tied set never
    // share a control.
    let without = by_case(&run_calendar(&reference, &recs, 1, false, Some(3)));
    let mut checked = 0;
    for (q, o) in &pairs {
        if let (Some(a), Some(b)) = (
            without.get(&id(scale.cases[*q])),
            without.get(&id(scale.cases[*o])),
        ) {
            assert_ne!(a, b);
            checked += 1;
        }
    }
    assert!(checked >= 3);

    // With replacement and the whole risk set requested, both get all of it.
    let full = by_case(&run_calendar(&reference, &recs, 10_000, true, Some(3)));
    for (q, o) in &pairs {
        let fa: BTreeSet<_> = full[&id(scale.cases[*q])].iter().collect();
        let fb: BTreeSet<_> = full[&id(scale.cases[*o])].iter().collect();
        assert_eq!(fa, fb);
    }
}

/// Chi-square critical value for `df` at upper-tail probability `alpha`,
/// found by bisection on the survival function to a relative width of 1e-12.
fn chi_square_critical(df: f64, alpha: f64) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, df + 100.0 * df.sqrt() + 100.0);
    while hi - lo > 1e-12 * hi {
        let mid = f64::midpoint(lo, hi);
        if chi_square_sf(mid, df) > alpha {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

fn chi_square(counts: &[usize]) -> f64 {
    let total: usize = counts.iter().sum();
    let expected = total as f64 / counts.len() as f64;
    counts
        .iter()
        .map(|c| {
            let d = *c as f64 - expected;
            d * d / expected
        })
        .sum()
}

/// Level of the whole family of tests: the chance that a correct sampler
/// fails the per-case or the pooled test on the fixed seeds is below this.
const FAMILY_ALPHA: f64 = 1e-3;

fn assert_uniform(counts_by_case: &[(usize, Vec<usize>)], label: &str) {
    // Each case is tested at FAMILY_ALPHA / (2 * cases) and the pooled
    // statistic, which sums independent chi-squares, at FAMILY_ALPHA / 2, so
    // the Bonferroni bound on the family is FAMILY_ALPHA.
    let n = counts_by_case.len() as f64;
    let mut pooled = 0.0;
    let mut pooled_df = 0.0;
    for (case, counts) in counts_by_case {
        let df = (counts.len() - 1) as f64;
        let stat = chi_square(counts);
        let limit = chi_square_critical(df, FAMILY_ALPHA / (2.0 * n));
        assert!(
            stat < limit,
            "{label}: case {case}, chi-square {stat} with {df} df, limit {limit}"
        );
        pooled += stat;
        pooled_df += df;
    }
    let limit = chi_square_critical(pooled_df, FAMILY_ALPHA / 2.0);
    assert!(
        pooled < limit,
        "{label}: pooled chi-square {pooled} with {pooled_df} df, limit {limit}"
    );
}

/// Every member of a risk set is drawn equally often, in `ccwc` (the counts
/// in the reference, 2000 runs with seeds 20261200 + r) and in the engine
/// (2000 runs with seeds 1 to 2000), for each case that is alone at its time.
/// A run draws one control per case, with replacement across cases. The sums
/// of the counts are checked to be the number of runs, so no run is missing.
#[test]
fn draws_are_uniform_over_the_risk_set_in_both() {
    let reference = reference();
    let recs = records_with_entry(&reference.people);
    assert_eq!(reference.draws.len(), reference.r_chi_square.len());

    let r_counts: Vec<(usize, Vec<usize>)> = reference
        .draws
        .iter()
        .map(|d| {
            assert_eq!(d.counts.iter().sum::<usize>(), reference.reps);
            (d.case, d.counts.clone())
        })
        .collect();
    for (draw, stat) in reference.draws.iter().zip(&reference.r_chi_square) {
        let ours = chi_square(&draw.counts);
        assert!((ours - stat).abs() <= 1e-9 * stat.max(1.0), "{ours} {stat}");
    }
    assert_uniform(&r_counts, "ccwc");

    let mut ours: Vec<HashMap<String, usize>> = vec![HashMap::new(); reference.draws.len()];
    for seed in 1..=reference.reps as u64 {
        let outcome = run_calendar(&reference, &recs, 1, true, Some(seed));
        let got = by_case(&outcome);
        for (q, draw) in reference.draws.iter().enumerate() {
            let chosen = &got[&id(draw.case)];
            assert_eq!(chosen.len(), 1);
            *ours[q].entry(chosen[0].clone()).or_default() += 1;
        }
    }
    let our_counts: Vec<(usize, Vec<usize>)> = reference
        .draws
        .iter()
        .zip(&ours)
        .map(|(draw, seen)| {
            #[expect(
                clippy::iter_over_hash_type,
                reason = "each entry is asserted on its own"
            )]
            for key in seen.keys() {
                assert!(
                    draw.members.iter().any(|m| id(*m) == *key),
                    "drawn outside the risk set"
                );
            }
            let counts: Vec<usize> = draw
                .members
                .iter()
                .map(|m| seen.get(&id(*m)).copied().unwrap_or(0))
                .collect();
            (draw.case, counts)
        })
        .collect();
    assert_uniform(&our_counts, "engine");
}
