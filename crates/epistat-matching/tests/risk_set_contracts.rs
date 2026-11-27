//! Risk-set matching, exposure density sampling and trial emulation: common
//! support against a brute-force scan, input checks, thread independence and
//! the options of the trial loop.

use chrono::{Days, NaiveDate};
use epistat_matching::DeterministicSelection;
use epistat_matching::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

type Rec = RoleTransitionRecord<BaseRecord>;

fn day(days: u64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2010, 1, 1)
        .unwrap()
        .checked_add_days(Days::new(days))
        .unwrap()
}

fn birth() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).unwrap()
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % n
    }
}

#[derive(Clone)]
struct P {
    id: String,
    entry: Option<u64>,
    event: Option<u64>,
    death: Option<u64>,
    score: Option<f64>,
}

fn random_cohort(rng: &mut Lcg, n: usize) -> Vec<P> {
    (0..n)
        .map(|i| P {
            id: format!("p{i:03}"),
            entry: (rng.next(3) > 0).then(|| rng.next(25)),
            event: (rng.next(2) == 0).then(|| rng.next(30)),
            death: (rng.next(4) == 0).then(|| rng.next(40)),
            score: (rng.next(10) > 0).then(|| rng.next(7) as f64 / 2.0),
        })
        .collect()
}

fn records(c: &[P]) -> Vec<Rec> {
    c.iter()
        .map(|p| {
            let mut b = BaseRecord::new(&p.id, birth());
            if let Some(d) = p.death {
                b = b.with_death_date(day(d));
            }
            let mut r = RoleTransitionRecord::from_record(b, p.event.map(day));
            r.entry_date = p.entry.map(day);
            r
        })
        .collect()
}

struct SameRule;
impl RiskSetPolicy<Rec> for SameRule {
    fn is_control_at_risk(&self, case: &Rec, control: &Rec, date: NaiveDate) -> bool {
        DefaultRiskSetPolicy.is_control_at_risk(case, control, date)
    }
}

fn trimmed_count(c: &[P], alive: bool, scan: bool, policy: CommonSupport) -> (usize, usize, bool) {
    let cohort = records(c);
    let scores = c
        .iter()
        .filter_map(|p| p.score.map(|s| (p.id.clone(), s)))
        .collect();
    let job = MatchJob::new_transition(&cohort, 0)
        .with_ratio(MatchRatio::new(1).unwrap())
        .with_birth_window(BirthDateWindowDays::new(30_000).unwrap())
        .with_age_limit(AgeLimitYears::new(100).unwrap())
        .with_replacement(true)
        .with_distance_config(DistanceConfig::propensity_score_map(scores, None))
        .with_common_support(Some(policy))
        .with_strategy(DeterministicSelection);
    let o = match (alive, scan) {
        (false, false) => job.run(),
        (true, false) => job.with_alive_check().run(),
        (false, true) => job.with_risk_set_policy(SameRule).run(),
        (true, true) => job.with_alive_check().with_risk_set_policy(SameRule).run(),
    };
    (
        o.diagnostics.common_support_trimmed_anchors,
        o.diagnostics.common_support_trimmed_candidates,
        o.pairs.is_empty() && o.diagnostics.common_support_overlap.is_none(),
    )
}

fn brute(c: &[P], alive: bool, policy: CommonSupport) -> Option<(usize, usize)> {
    let fin = |p: &P| p.score.filter(|s| s.is_finite());
    let cases: Vec<&P> = c.iter().filter(|p| p.event.is_some()).collect();
    let cb = cases
        .iter()
        .filter_map(|p| fin(p))
        .fold(None, |a: Option<(f64, f64)>, s| {
            Some(a.map_or((s, s), |(l, h)| (l.min(s), h.max(s))))
        })?;
    let ab = c
        .iter()
        .filter_map(fin)
        .fold(None, |a: Option<(f64, f64)>, s| {
            Some(a.map_or((s, s), |(l, h)| (l.min(s), h.max(s))))
        })?;
    let lo = cb.0.max(ab.0);
    let hi = cb.1.min(ab.1);
    if lo > hi {
        return None;
    }
    let mut tc = 0;
    if matches!(policy, CommonSupport::Treated | CommonSupport::Both) {
        for case in &cases {
            let d = case.event.unwrap();
            let mut range: Option<(f64, f64)> = None;
            for p in c {
                let Some(s) = fin(p) else { continue };
                let at_risk = p.entry.is_none_or(|e| e <= d)
                    && p.event.is_none_or(|e| e > d)
                    && (!alive || p.death.is_none_or(|x| x > d));
                if at_risk {
                    range = Some(range.map_or((s, s), |(l, h)| (l.min(s), h.max(s))));
                }
            }
            if let Some((l, h)) = range
                && !fin(case).is_some_and(|s| s >= l && s <= h)
            {
                tc += 1;
            }
        }
    }
    let mut tcand = 0;
    if matches!(policy, CommonSupport::Control | CommonSupport::Both) {
        tcand = c
            .iter()
            .filter(|p| !fin(p).is_some_and(|s| s >= lo && s <= hi))
            .count();
    }
    Some((tc, tcand))
}

#[test]
fn common_support_sweep_agrees_with_brute_force() {
    let mut rng = Lcg(7);
    let mut checked = 0;
    for _ in 0..300 {
        let c = random_cohort(&mut rng, 30);
        for policy in [CommonSupport::Treated, CommonSupport::Both] {
            for alive in [false, true] {
                let Some(expected) = brute(&c, alive, policy) else {
                    continue;
                };
                let sweep = trimmed_count(&c, alive, false, policy);
                let scan = trimmed_count(&c, alive, true, policy);
                assert_eq!(
                    (sweep.0, sweep.1),
                    expected,
                    "sweep {policy:?} alive={alive}"
                );
                assert_eq!((scan.0, scan.1), expected, "scan {policy:?} alive={alive}");
                checked += 1;
            }
        }
    }
    assert!(checked > 500);
}

/// A case whose event date is before its birth date is an error under the
/// default policy, and the error names the id.
#[test]
fn event_before_birth_is_an_error_naming_the_id() {
    let early = NaiveDate::from_ymd_opt(1990, 1, 1).unwrap();
    let cohort = vec![
        RoleTransitionRecord::from_record(BaseRecord::new("bad", birth()), Some(early)),
        RoleTransitionRecord::from_record(BaseRecord::new("ctl", birth()), None),
    ];
    let o = MatchJob::new_transition(&cohort, 0)
        .with_ratio(MatchRatio::new(1).unwrap())
        .with_birth_window(BirthDateWindowDays::new(30_000).unwrap())
        .with_age_limit(AgeLimitYears::new(18).unwrap())
        .with_strategy(DeterministicSelection)
        .run();
    assert_eq!(o.pairs, vec![]);
    assert!(
        o.diagnostics
            .exclusion_counts
            .contains_key(&ExclusionReason::EventBeforeBirth("bad".to_string())),
        "{:?}",
        o.diagnostics.exclusion_counts
    );
}

/// Two units with one id make a pair ambiguous, so the default policy refuses
/// the cohort and names the id.
#[test]
fn duplicate_ids_are_an_error_naming_the_id() {
    let cohort = vec![
        RoleTransitionRecord::from_record(BaseRecord::new("a", birth()), Some(day(100))),
        RoleTransitionRecord::from_record(BaseRecord::new("b", birth()), None),
        RoleTransitionRecord::from_record(BaseRecord::new("b", birth()), None),
    ];
    let o = MatchJob::new_transition(&cohort, 0)
        .with_ratio(MatchRatio::new(1).unwrap())
        .with_birth_window(BirthDateWindowDays::new(30_000).unwrap())
        .with_age_limit(AgeLimitYears::new(100).unwrap())
        .run();
    assert_eq!(o.pairs, vec![]);
    assert!(
        o.diagnostics
            .exclusion_counts
            .contains_key(&ExclusionReason::DuplicateId("b".to_string())),
        "{:?}",
        o.diagnostics.exclusion_counts
    );
}

#[test]
fn tied_cases_are_not_each_others_controls_and_without_replacement_is_distinct() {
    let mut rng = Lcg(11);
    for _ in 0..100 {
        let c = random_cohort(&mut rng, 40);
        let cohort = records(&c);
        let o = MatchJob::new_transition(&cohort, 3)
            .with_ratio(MatchRatio::new(2).unwrap())
            .with_birth_window(BirthDateWindowDays::new(30_000).unwrap())
            .with_age_limit(AgeLimitYears::new(100).unwrap())
            .with_replacement(false)
            .with_strategy(DeterministicSelection)
            .run();
        let by: BTreeMap<&str, &P> = c.iter().map(|p| (p.id.as_str(), p)).collect();
        let mut seen = BTreeSet::new();
        for pair in &o.pairs {
            let a = by[pair.anchor_id()];
            let k = by[pair.comparator_id()];
            let d = a.event.unwrap();
            assert!(k.entry.is_none_or(|e| e <= d));
            assert!(k.event.is_none_or(|e| e > d));
            assert!(
                seen.insert(pair.comparator_id().to_string()),
                "control reused"
            );
        }
    }
}

#[test]
fn wide_scan_is_the_same_on_one_and_many_threads() {
    let mut rng = Lcg(5);
    let n = 6000;
    let c: Vec<P> = (0..n)
        .map(|i| P {
            id: format!("p{i:05}"),
            entry: None,
            event: (i % 50 == 0).then(|| 5 + rng.next(100)),
            death: None,
            score: None,
        })
        .collect();
    let cohort = records(&c);
    let run = || {
        MatchJob::new_transition(&cohort, 99)
            .with_ratio(MatchRatio::new(3).unwrap())
            .with_birth_window(BirthDateWindowDays::new(30_000).unwrap())
            .with_age_limit(AgeLimitYears::new(100).unwrap())
            .with_replacement(false)
            .run()
            .pairs
            .iter()
            .map(|p| format!("{}>{}", p.anchor_id(), p.comparator_id()))
            .collect::<Vec<_>>()
    };
    let pool = |t| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .unwrap()
    };
    let one = pool(1).install(run);
    let many = pool(8).install(run);
    assert_ne!(one, Vec::<String>::new());
    assert_eq!(one, many);
}

#[test]
fn exposure_density_invariants() {
    let mut rng = Lcg(21);
    for rep in 0..100 {
        let c = random_cohort(&mut rng, 40);
        let cohort = records(&c);
        let replacement = rep % 2 == 0;
        let b = BirthDateWindowDays::new(30_000).unwrap();
        let o = MatchJob::new_transition(&cohort, 1)
            .with_ratio(MatchRatio::new(2).unwrap())
            .with_birth_window(b)
            .with_age_limit(AgeLimitYears::new(100).unwrap())
            .with_replacement(replacement)
            .run();
        let r = exposure_density_sets(&cohort, o).unwrap();
        let mut total = 0;
        for e in &r.exposed {
            total += e.n_controls;
            assert_eq!(
                r.controls.iter().filter(|k| k.exposed_id == e.id).count(),
                e.n_controls
            );
        }
        assert_eq!(total, r.controls.len());
        for k in &r.controls {
            if let Some(x) = k.exposure_date {
                assert!(x > k.index_date);
            }
        }
        for w in r.exposed.windows(2) {
            assert!((w[0].index_date, &w[0].id) < (w[1].index_date, &w[1].id));
        }
        if !replacement {
            let ids: BTreeSet<_> = r.controls.iter().map(|k| &k.control_id).collect();
            assert_eq!(ids.len(), r.controls.len());
        }
    }
}

fn tp(id: &str, rows: &[(i64, Option<bool>)]) -> TrialPerson<BaseRecord> {
    TrialPerson {
        record: BaseRecord::new(id, birth()),
        periods: rows
            .iter()
            .map(|&(period, treatment)| TrialPeriod {
                period,
                date: day(period.unsigned_abs() * 30),
                eligible: true,
                treatment,
                event: false,
            })
            .collect(),
    }
}

fn run_trials(
    panel: &[TrialPerson<BaseRecord>],
    options: &TrialOptions,
    replacement: bool,
) -> TrialOutcome {
    match_trials(panel, options, |pool| {
        Ok(pool
            .job(trial_seed(1, pool.period()))
            .with_ratio(MatchRatio::new(1).unwrap())
            .with_birth_window(BirthDateWindowDays::new(30_000).unwrap())
            .with_age_limit(AgeLimitYears::new(100).unwrap())
            .with_replacement(replacement)
            .run())
    })
    .unwrap()
}

#[test]
fn trial_reuse_options() {
    let f = Some(false);
    let t = Some(true);
    // i0 initiates at 0, i1 at 1, i2 at 2. c is never treated.
    let panel = vec![
        tp("i0", &[(0, t), (1, t), (2, t)]),
        tp("i1", &[(0, f), (1, t), (2, t)]),
        tp("i2", &[(0, f), (1, f), (2, t)]),
        tp("c", &[(0, f), (1, f), (2, f)]),
    ];
    let default = run_trials(&panel, &TrialOptions::default(), false);
    // Trial 0: initiator i0, pool i0 i1 i2 c.
    assert_eq!(default.trials[0].n_eligible, 4);
    assert_eq!(default.trials[0].n_initiators, 1);
    // Trial 2: initiators i0, i1, i2 all assigned t, only c is a control.
    assert_eq!(default.trials[2].n_initiators, 3);
    let no_future = run_trials(
        &panel,
        &TrialOptions {
            initiators_as_earlier_controls: false,
            ..TrialOptions::default()
        },
        false,
    );
    // Trial 0: i1 and i2 initiate later (i1 at 2 as last), so only i0 and c remain.
    assert_eq!(
        no_future.trials[0].n_eligible, 2,
        "pool {:?}",
        no_future.trials[0].members
    );
    let once = run_trials(
        &panel,
        &TrialOptions {
            control_reuse: ControlReuse::OncePerRun,
            ..TrialOptions::default()
        },
        true,
    );
    let mut first_trial: BTreeMap<String, i64> = BTreeMap::new();
    for t in &once.trials {
        for p in &t.outcome.pairs {
            let at = *first_trial
                .entry(p.comparator_id().to_string())
                .or_insert(t.period);
            assert_eq!(
                at,
                t.period,
                "{} served in trials {at} and {}",
                p.comparator_id(),
                t.period
            );
        }
    }
}
