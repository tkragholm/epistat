//! Sequential trial emulation checked against the R package TrialEmulation and
//! `Epi::ccwc`.
//!
//! The reference is `tests/data/trial_emulation.json`, written by
//! `tests/data/trial_emulation.R`. It holds three panels (`data_censored` and
//! the first 12 people of `trial_example` of TrialEmulation, and a simulated
//! panel of 316 children with deaths, emigrations and staggered entry), the
//! expansion of each by TrialEmulation 0.0.5 as an intention-to-treat and a
//! per-protocol sequence, and for the simulated panel the risk set of every
//! initiator in every trial from `Epi::ccwc`. The file records the versions of
//! R, TrialEmulation and Epi.
//!
//! What is compared:
//!
//! * the expansion, row for row: for every person and trial, the number of
//!   rows of the intention-to-treat and the per-protocol follow-up, the
//!   assigned treatment and the follow-up time of the outcome. The script
//!   checks that the follow-up times of every person and trial run from 0
//!   without gaps, so the counts stand for every row;
//! * the period of the deviation and the last period of the follow-up that
//!   `match_trials` reports for every matched person, against the rows of the
//!   per-protocol and the intention-to-treat expansions;
//! * the risk set of every initiator in every trial, which is deterministic,
//!   against `Epi::ccwc` (the script checks it equal to a direct definition);
//! * how often each member of a risk set is drawn, for four initiators that
//!   are alone among the initiators of their trial and cell, over 600 seeds in
//!   R and in the engine, against a chi-square threshold derived from the
//!   level and the number of tests.
//!
//! The options that say how people are reused between trials have no
//! counterpart in R and are checked here against the data.

use chrono::{Days, NaiveDate};
use epistat_matching::prelude::*;
use epistat_stats::gamma::chi_square_sf;
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    age_limit_years: u8,
    day0: i64,
    step_days: i64,
    reps: usize,
    data_censored: Case,
    trial_example: Case,
    simulated: Simulated,
    risk_sets: BTreeMap<String, Vec<String>>,
    draws: Vec<Draw>,
    r_chi_square: Vec<f64>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    #[serde(rename = "TrialEmulation")]
    trial_emulation: String,
    #[serde(rename = "Epi")]
    epi: String,
}

#[derive(Deserialize)]
struct Case {
    panel: PanelCols,
    expansion: Expansion,
}

#[derive(Deserialize)]
struct Simulated {
    panel: PanelCols,
    people: People,
    expansion: Expansion,
}

#[derive(Deserialize)]
struct PanelCols {
    id: Vec<i64>,
    period: Vec<i64>,
    eligible: Vec<i64>,
    treatment: Vec<Option<i64>>,
    outcome: Vec<i64>,
}

#[derive(Deserialize)]
struct Expansion {
    id: Vec<i64>,
    trial_period: Vec<i64>,
    itt_rows: Vec<i64>,
    assigned: Vec<i64>,
    outcome_at: Vec<Option<i64>>,
    pp_rows: Vec<i64>,
    pp_outcome_at: Vec<Option<i64>>,
}

#[derive(Deserialize)]
struct People {
    id: Vec<String>,
    sex: Vec<String>,
    area: Vec<String>,
    birth: Vec<i64>,
    entry: Vec<i64>,
    death: Vec<Option<i64>>,
    emigration: Vec<Option<i64>>,
}

#[derive(Deserialize)]
struct Draw {
    key: String,
    members: Vec<String>,
    counts: Vec<usize>,
}

type Panel = Vec<TrialPerson<BalanceRecord>>;

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/trial_emulation.json"
    ))
}

fn day(days: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1)
        .expect("valid date")
        .checked_add_days(Days::new(u64::try_from(days).expect("after 1970")))
        .expect("in range")
}

fn flag(value: i64) -> bool {
    assert!(value == 0 || value == 1, "a 0 or 1 flag, got {value}");
    value == 1
}

/// People with no matching data, for the expansion.
fn plain_panel(cols: &PanelCols) -> Panel {
    let mut order: Vec<i64> = Vec::new();
    let mut rows: HashMap<i64, Vec<TrialPeriod>> = HashMap::new();
    for i in 0..cols.id.len() {
        let id = cols.id[i];
        if !rows.contains_key(&id) {
            order.push(id);
        }
        rows.entry(id).or_default().push(TrialPeriod {
            period: cols.period[i],
            date: day(10_000 + cols.period[i]),
            eligible: flag(cols.eligible[i]),
            treatment: cols.treatment[i].map(flag),
            event: flag(cols.outcome[i]),
        });
    }
    order
        .into_iter()
        .map(|id| TrialPerson {
            record: BalanceRecord::new(id.to_string(), day(1)),
            periods: rows.remove(&id).expect("listed"),
        })
        .collect()
}

/// The simulated panel with the people's strata, entry, death and emigration.
fn simulated_panel(reference: &Reference) -> Panel {
    let p = &reference.simulated.people;
    let cols = &reference.simulated.panel;
    let mut rows: HashMap<i64, Vec<TrialPeriod>> = HashMap::new();
    for i in 0..cols.id.len() {
        rows.entry(cols.id[i]).or_default().push(TrialPeriod {
            period: cols.period[i],
            date: day(reference.day0 + reference.step_days * cols.period[i]),
            eligible: flag(cols.eligible[i]),
            treatment: cols.treatment[i].map(flag),
            event: flag(cols.outcome[i]),
        });
    }
    let mut ids: Vec<i64> = rows.keys().copied().collect();
    ids.sort_unstable();
    ids.into_iter()
        .map(|id| {
            let i = usize::try_from(id - 1).expect("positive id");
            let mut base = BaseRecord::new(p.id[i].clone(), day(p.birth[i]));
            base.strata.insert("sex".into(), p.sex[i].clone());
            base.strata.insert("area".into(), p.area[i].clone());
            base = base.with_entry_date(day(p.entry[i]));
            if let Some(d) = p.death[i] {
                base = base.with_death_date(day(d));
            }
            if let Some(d) = p.emigration[i] {
                base = base.with_emigration_date(day(d));
            }
            let record = BalanceRecord::new(base.id.clone(), base.birth_date);
            TrialPerson {
                record: BalanceRecord {
                    core: base,
                    ..record
                },
                periods: rows.remove(&id).expect("listed"),
            }
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
struct Summary {
    itt_rows: i64,
    assigned: bool,
    outcome_at: Option<i64>,
}

/// Rows, assignment and outcome time of each person and trial, from the
/// expanded rows.
fn summarise(rows: &[ExpandedRow]) -> BTreeMap<(String, i64), Summary> {
    let mut map: BTreeMap<(String, i64), Summary> = BTreeMap::new();
    for row in rows {
        let key = (row.id.clone(), row.trial_period);
        let entry = map.entry(key).or_insert(Summary {
            itt_rows: 0,
            assigned: row.assigned,
            outcome_at: None,
        });
        assert_eq!(
            row.follow_up_time, entry.itt_rows,
            "follow-up time runs from 0 without gaps"
        );
        assert_eq!(entry.assigned, row.assigned);
        entry.itt_rows += 1;
        if row.outcome {
            entry.outcome_at = Some(row.follow_up_time);
        }
    }
    map
}

fn check_expansion(
    panel: &Panel,
    expansion: &Expansion,
    label: &str,
    name: &dyn Fn(i64) -> String,
) {
    let itt = TrialOptions {
        censor_at_deviation: false,
        ..TrialOptions::default()
    };
    let got_itt = summarise(&expand_trials(panel, &itt).expect("valid panel"));
    let got_pp = summarise(&expand_trials(panel, &TrialOptions::default()).expect("valid panel"));
    assert_eq!(got_itt.len(), expansion.id.len(), "{label}: trials");
    assert_eq!(
        got_pp.len(),
        expansion.id.len(),
        "{label}: trials, per protocol"
    );
    let mut cut = 0;
    for i in 0..expansion.id.len() {
        let key = (name(expansion.id[i]), expansion.trial_period[i]);
        let want_itt = Summary {
            itt_rows: expansion.itt_rows[i],
            assigned: flag(expansion.assigned[i]),
            outcome_at: expansion.outcome_at[i],
        };
        let want_pp = Summary {
            itt_rows: expansion.pp_rows[i],
            assigned: flag(expansion.assigned[i]),
            outcome_at: expansion.pp_outcome_at[i],
        };
        assert_eq!(
            got_itt[&key], want_itt,
            "{label}: intention to treat, {key:?}"
        );
        assert_eq!(got_pp[&key], want_pp, "{label}: per protocol, {key:?}");
        if want_pp.itt_rows < want_itt.itt_rows {
            cut += 1;
        }
    }
    assert!(cut > 0, "{label}: the data has deviations");
}

#[test]
fn versions_are_recorded() {
    let reference = reference();
    assert!(reference.versions.r.starts_with("R version"));
    assert_eq!(reference.versions.trial_emulation, "0.0.5");
    assert_eq!(reference.versions.epi, "2.66");
}

/// The expansion of `data_censored`, which has one trial for most people, and
/// of the first 12 people of `trial_example`, which have eligibility over many
/// periods, equals TrialEmulation's, as intention to treat and per protocol.
#[test]
fn expansion_equals_trial_emulation_on_its_example_data() {
    let reference = reference();
    check_expansion(
        &plain_panel(&reference.data_censored.panel),
        &reference.data_censored.expansion,
        "data_censored",
        &|id| id.to_string(),
    );
    check_expansion(
        &plain_panel(&reference.trial_example.panel),
        &reference.trial_example.expansion,
        "trial_example",
        &|id| id.to_string(),
    );
}

/// The same on the simulated panel, whose people enter late, die and emigrate,
/// restart treatment and have an outcome that ends their follow-up.
#[test]
fn expansion_equals_trial_emulation_on_the_simulated_panel() {
    let reference = reference();
    check_expansion(
        &simulated_panel(&reference),
        &reference.simulated.expansion,
        "simulated",
        &|id| format!("p{id:03}"),
    );
}

fn ratio(k: usize) -> MatchRatio {
    MatchRatio::new(k).expect("non-zero ratio")
}

fn run_trial(
    reference: &Reference,
    cohort: &[RoleTransitionRecord<BalanceRecord>],
    k: usize,
    replacement: bool,
    seed: u64,
) -> MatchOutcome {
    MatchJob::new_transition(cohort, seed)
        .with_ratio(ratio(k))
        .with_birth_window(BirthDateWindowDays::new(30_000).expect("non-negative"))
        .with_age_limit(AgeLimitYears::new(reference.age_limit_years).expect("positive"))
        .with_exact_matches(["sex", "area"])
        .with_replacement(replacement)
        .with_alive_check()
        .with_resident_check()
        .run()
}

fn run(
    reference: &Reference,
    panel: &Panel,
    options: &TrialOptions,
    k: usize,
    replacement: bool,
    seed: u64,
) -> TrialOutcome {
    match_trials(panel, options, |period, cohort| {
        Ok::<_, std::convert::Infallible>(run_trial(
            reference,
            cohort,
            k,
            replacement,
            trial_seed(seed, period),
        ))
    })
    .expect("valid panel")
}

fn key(period: i64, id: &str) -> String {
    format!("{period}:{id}")
}

fn controls_by_initiator(outcome: &TrialOutcome) -> BTreeMap<String, Vec<String>> {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for trial in &outcome.trials {
        for pair in &trial.outcome.pairs {
            map.entry(key(trial.period, pair.anchor_id()))
                .or_default()
                .push(pair.comparator_id().to_string());
        }
    }
    for controls in map.values_mut() {
        controls.sort();
    }
    map
}

/// With a ratio above every risk set and replacement, the engine returns the
/// whole risk set of every initiator in every trial. Those sets equal the ones
/// `Epi::ccwc` returns for the same trial with the initiation as the failure.
/// The data has people who enter, die or emigrate around the start of a trial,
/// controls who initiate in a later trial, and initiators older than the age
/// limit, who are neither cases nor controls.
#[test]
fn risk_sets_of_every_trial_equal_ccwc() {
    let reference = reference();
    let panel = simulated_panel(&reference);
    let outcome = run(
        &reference,
        &panel,
        &TrialOptions::default(),
        10_000,
        true,
        0,
    );
    let got = controls_by_initiator(&outcome);
    let want: BTreeMap<String, Vec<String>> = reference
        .risk_sets
        .iter()
        .filter(|(_, set)| !set.is_empty())
        .map(|(k, set)| (k.clone(), set.clone()))
        .collect();
    assert!(want.len() > 100, "the reference has many sets");
    assert_eq!(got.len(), want.len(), "initiators with a set");
    for (initiator, controls) in &want {
        assert_eq!(&got[initiator], controls, "risk set of {initiator}");
    }
    let n_trials = outcome.trials.len();
    assert!(n_trials >= 10, "{n_trials} trials");
    let over_age = reference.risk_sets.len()
        < reference
            .simulated
            .expansion
            .assigned
            .iter()
            .filter(|a| **a == 1)
            .count();
    assert!(over_age, "the data has initiators over the age limit");
}

/// Every matched person has the start of the follow-up of the trial, the
/// period of the deviation and the last period of the follow-up that the
/// per-protocol and intention-to-treat expansions of TrialEmulation have: the
/// deviation is the period after the last per-protocol row when that is
/// before the last intention-to-treat row.
#[test]
fn deviation_and_follow_up_equal_the_expansion() {
    let reference = reference();
    let panel = simulated_panel(&reference);
    let e = &reference.simulated.expansion;
    let truth: HashMap<(String, i64), (i64, i64, bool)> = (0..e.id.len())
        .map(|i| {
            (
                (
                    reference.simulated.people.id[usize::try_from(e.id[i] - 1).expect("id")]
                        .clone(),
                    e.trial_period[i],
                ),
                (e.pp_rows[i], e.itt_rows[i], flag(e.assigned[i])),
            )
        })
        .collect();
    let outcome = run(&reference, &panel, &TrialOptions::default(), 3, true, 4);
    let (mut deviating, mut staying) = (0, 0);
    for trial in &outcome.trials {
        for member in &trial.members {
            let (pp, itt, assigned) = truth[&(member.id.clone(), trial.period)];
            assert_eq!(member.assigned, assigned);
            assert_eq!(member.follow_up_start_period, trial.period);
            assert_eq!(
                member.follow_up_start,
                day(reference.day0 + reference.step_days * trial.period)
            );
            assert_eq!(member.last_period, trial.period + pp - 1);
            if pp < itt {
                deviating += 1;
                assert_eq!(member.deviation_period, Some(trial.period + pp));
                assert_eq!(
                    member.deviation_date,
                    Some(day(
                        reference.day0 + reference.step_days * (trial.period + pp)
                    ))
                );
            } else {
                staying += 1;
                assert_eq!(member.deviation_period, None);
                assert_eq!(member.deviation_date, None);
            }
        }
    }
    assert!(deviating > 0 && staying > 0, "{deviating} {staying}");

    // The deviation is the same without censoring, and the follow-up then runs
    // to the last intention-to-treat row.
    let itt = TrialOptions {
        censor_at_deviation: false,
        ..TrialOptions::default()
    };
    let outcome = run(&reference, &panel, &itt, 3, true, 4);
    for trial in &outcome.trials {
        for member in &trial.members {
            let (pp, itt_rows, _) = truth[&(member.id.clone(), trial.period)];
            assert_eq!(member.last_period, trial.period + itt_rows - 1);
            assert_eq!(
                member.deviation_period,
                (pp < itt_rows).then_some(trial.period + pp)
            );
        }
    }
}

/// With replacement, a control serves in several trials. With
/// `ControlReuse::OncePerRun` a person drawn as a control in one trial is not
/// a control in any later trial, and is still an initiator in later trials.
#[test]
fn control_reuse_across_trials() {
    let reference = reference();
    let panel = simulated_panel(&reference);
    let uses = |outcome: &TrialOutcome| -> HashMap<String, usize> {
        let mut seen: HashMap<String, usize> = HashMap::new();
        for trial in &outcome.trials {
            let controls: BTreeSet<&str> = trial
                .outcome
                .pairs
                .iter()
                .map(MatchedPair::comparator_id)
                .collect();
            for control in controls {
                *seen.entry(control.to_string()).or_default() += 1;
            }
        }
        seen
    };
    let by_default = run(&reference, &panel, &TrialOptions::default(), 2, false, 9);
    assert!(
        uses(&by_default).values().any(|n| *n > 1),
        "a control serves in several trials by default"
    );
    let once = TrialOptions {
        control_reuse: ControlReuse::OncePerRun,
        ..TrialOptions::default()
    };
    let outcome = run(&reference, &panel, &once, 2, false, 9);
    assert!(!outcome.trials.is_empty());
    assert!(uses(&outcome).values().all(|n| *n == 1));
    // A control of an early trial is an initiator of a later one.
    let controls: BTreeSet<String> = outcome
        .trials
        .iter()
        .flat_map(|t| {
            t.outcome
                .pairs
                .iter()
                .map(|p| p.comparator_id().to_string())
        })
        .collect();
    let initiators_later = outcome.trials.iter().any(|t| {
        t.outcome
            .pairs
            .iter()
            .any(|p| controls.contains(p.anchor_id()))
    });
    assert!(initiators_later, "an earlier control initiates later");
}

/// With `initiators_as_earlier_controls = false` nobody who initiates in a
/// later trial is a control, and with the default somebody is.
#[test]
fn initiators_as_earlier_controls_option() {
    let reference = reference();
    let panel = simulated_panel(&reference);
    let e = &reference.simulated.expansion;
    let mut last_initiation: HashMap<String, i64> = HashMap::new();
    for i in 0..e.id.len() {
        if flag(e.assigned[i]) {
            let id =
                reference.simulated.people.id[usize::try_from(e.id[i] - 1).expect("id")].clone();
            let last = last_initiation.entry(id).or_insert(i64::MIN);
            *last = (*last).max(e.trial_period[i]);
        }
    }
    let later_initiator =
        |id: &str, period: i64| last_initiation.get(id).is_some_and(|last| *last > period);
    let by_default = run(&reference, &panel, &TrialOptions::default(), 3, true, 2);
    assert!(
        by_default.trials.iter().any(|t| t
            .outcome
            .pairs
            .iter()
            .any(|p| later_initiator(p.comparator_id(), t.period))),
        "by default a later initiator is a control"
    );
    let options = TrialOptions {
        initiators_as_earlier_controls: false,
        ..TrialOptions::default()
    };
    let outcome = run(&reference, &panel, &options, 3, true, 2);
    assert!(!outcome.trials.is_empty());
    for trial in &outcome.trials {
        for pair in &trial.outcome.pairs {
            assert!(
                !later_initiator(pair.comparator_id(), trial.period),
                "{} controls in trial {} and initiates later",
                pair.comparator_id(),
                trial.period
            );
        }
    }
}

/// The mode of the spec gives the trials of the function, with the seed of
/// each trial derived from the seed of the spec.
#[test]
fn spec_mode_matches_the_function() {
    let reference = reference();
    let panel = simulated_panel(&reference);
    let spec = MatchSpec {
        mode: MatchMode::SequentialTrials,
        method: MatchMethod::Random,
        seed: 31,
        ratio: 2,
        replace: true,
        birth_window_days: 30_000,
        age_limit_years: Some(reference.age_limit_years),
        exact: vec!["sex".into(), "area".into()],
        require_alive: true,
        require_resident: true,
        ..MatchSpec::default()
    };
    let SpecOutput::Trials(from_spec) =
        run_spec(&spec, SpecInput::Trials { panel: &panel }).expect("valid spec")
    else {
        panic!("expected trials");
    };
    let by_hand = run(&reference, &panel, &TrialOptions::default(), 2, true, 31);
    assert_eq!(from_spec.trials.len(), by_hand.trials.len());
    assert!(!by_hand.trials.is_empty());
    for (a, b) in from_spec.trials.iter().zip(&by_hand.trials) {
        assert_eq!(a.period, b.period);
        assert_eq!(a.outcome.pairs, b.outcome.pairs);
        assert_eq!(a.members, b.members);
    }
}

/// The mode needs a panel, takes the options of risk-set matching and rejects
/// a panel with two people of one id.
#[test]
fn spec_mode_needs_a_panel() {
    let reference = reference();
    let panel = simulated_panel(&reference);
    let spec = MatchSpec {
        mode: MatchMode::SequentialTrials,
        ..MatchSpec::default()
    };
    assert!(matches!(
        run_spec::<BalanceRecord>(
            &spec,
            SpecInput::Standard {
                anchors: &[],
                candidates: &[]
            }
        ),
        Err(SpecError::InputMode {
            mode: MatchMode::SequentialTrials
        })
    ));
    let nearest = MatchSpec {
        method: MatchMethod::Nearest,
        ..spec.clone()
    };
    assert!(matches!(
        run_spec(&nearest, SpecInput::Trials { panel: &panel }),
        Err(SpecError::Option {
            option: "method",
            ..
        })
    ));
    let mut twice = panel.clone();
    twice.push(panel[0].clone());
    assert!(matches!(
        run_spec(&spec, SpecInput::Trials { panel: &twice }),
        Err(SpecError::Option {
            option: "panel",
            ..
        })
    ));
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

fn assert_uniform(counts_by_case: &[(String, Vec<usize>)], label: &str) {
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

/// Every member of a risk set is drawn equally often, in R (the counts in the
/// reference, 600 runs of `ccwc` with one control) and in the engine (600 runs
/// with seeds 1 to 600), for the four initiators that are alone in their trial
/// and cell. The engine draws one control per initiator with replacement. The
/// sets include people who initiate in a later trial.
#[test]
fn draws_are_uniform_over_the_risk_set_in_both() {
    let reference = reference();
    let panel = simulated_panel(&reference);
    assert_eq!(reference.draws.len(), 4);
    assert_eq!(reference.draws.len(), reference.r_chi_square.len());
    let r_counts: Vec<(String, Vec<usize>)> = reference
        .draws
        .iter()
        .map(|d| {
            assert_eq!(d.counts.iter().sum::<usize>(), reference.reps);
            assert_eq!(&d.members, &reference.risk_sets[&d.key]);
            (d.key.clone(), d.counts.clone())
        })
        .collect();
    for (draw, stat) in reference.draws.iter().zip(&reference.r_chi_square) {
        let ours = chi_square(&draw.counts);
        assert!((ours - stat).abs() <= 1e-9 * stat.max(1.0), "{ours} {stat}");
    }
    assert_uniform(&r_counts, "ccwc");

    let mut ours: Vec<HashMap<String, usize>> = vec![HashMap::new(); reference.draws.len()];
    for seed in 1..=reference.reps as u64 {
        let got = controls_by_initiator(&run(
            &reference,
            &panel,
            &TrialOptions::default(),
            1,
            true,
            seed,
        ));
        for (q, draw) in reference.draws.iter().enumerate() {
            let chosen = &got[&draw.key];
            assert_eq!(chosen.len(), 1);
            *ours[q].entry(chosen[0].clone()).or_default() += 1;
        }
    }
    let our_counts: Vec<(String, Vec<usize>)> = reference
        .draws
        .iter()
        .zip(&ours)
        .map(|(draw, seen)| {
            for member in seen.keys() {
                assert!(draw.members.contains(member), "drawn outside the risk set");
            }
            let counts: Vec<usize> = draw
                .members
                .iter()
                .map(|m| seen.get(m).copied().unwrap_or(0))
                .collect();
            (draw.key.clone(), counts)
        })
        .collect();
    assert_uniform(&our_counts, "engine");
}
