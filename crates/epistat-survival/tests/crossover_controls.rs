//! `crossover::sample_controls` and `crossover::case_time_control_windows`:
//! who is eligible under each rule, that the draw is uniform, distinct within
//! a case and the same on any number of threads, what is refused, stratified
//! referents refused, and how the windows of cases and controls are put
//! together.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    reason = "small test counts and days"
)]

use epistat_survival::crossover::{
    Candidates, CaseEvents, ControlRule, EventStrata, Observation, Referent, StratumUnit,
    build_windows, build_windows_in_strata, case_time_control_windows, sample_controls,
};
use epistat_testkit::{POOLS, assert_same_on_pools};

const T: i64 = 18_336;

/// Ten candidates, person codes 100 to 109. Candidate k is observed from
/// `T - 100` to `T + 100` except 3 (from `T + 1`) and 4 (to `T - 1`); 5 has
/// its first event at `T`, 6 at `T + 5`, 7 at `T + 40` and 8 at `T - 30`;
/// keys alternate 0 and 1 except that 9 has key 2.
struct Pool {
    person: Vec<u32>,
    start: Vec<i64>,
    end: Vec<i64>,
    first: Vec<Option<i64>>,
    key: Vec<u64>,
}

fn pool() -> Pool {
    let person: Vec<u32> = (100..110).collect();
    let mut start = vec![T - 100; 10];
    let mut end = vec![T + 100; 10];
    start[3] = T + 1;
    end[4] = T - 1;
    let mut first = vec![None; 10];
    first[5] = Some(T);
    first[6] = Some(T + 5);
    first[7] = Some(T + 40);
    first[8] = Some(T - 30);
    let mut key: Vec<u64> = (0..10).map(|k| k % 2).collect();
    key[9] = 2;
    Pool {
        person,
        start,
        end,
        first,
        key,
    }
}

impl Pool {
    fn candidates(&self, keyed: bool) -> Candidates<'_> {
        Candidates {
            person: &self.person,
            start: &self.start,
            end: &self.end,
            first_event: &self.first,
            key: keyed.then_some(&self.key[..]),
        }
    }
}

fn eligible(rule: ControlRule, case_person: u32, key: Option<u64>) -> Vec<u32> {
    let p = pool();
    let keys = key.map(|k| vec![k]);
    let cases = CaseEvents {
        day: &[T],
        person: &[case_person],
        key: keys.as_deref(),
    };
    let c = sample_controls(&cases, &p.candidates(key.is_some()), rule, 10, 1).unwrap();
    let mut got = c.person.clone();
    got.sort_unstable();
    assert_eq!(c.cases_short, vec![0]);
    got
}

#[test]
fn the_rules_choose_the_people_they_name() {
    // Event-free and observed on T: not 3 or 4 (not observed), 5 (event on
    // T) or 8 (event before); 6 and 7 become cases later and are eligible.
    assert_eq!(
        eligible(ControlRule::EventFree, 1, None),
        vec![100, 101, 102, 106, 107, 109]
    );
    // The case's own person is never drawn.
    assert_eq!(
        eligible(ControlRule::EventFree, 101, None),
        vec![100, 102, 106, 107, 109]
    );
    // Key 0 is candidates 0, 2, 4, 6, 8.
    assert_eq!(
        eligible(ControlRule::EventFree, 1, Some(0)),
        vec![100, 102, 106]
    );
    // Future cases: an event after T + gap.
    assert_eq!(
        eligible(ControlRule::FutureCases { gap: 0 }, 1, None),
        vec![106, 107]
    );
    assert_eq!(
        eligible(ControlRule::FutureCases { gap: 5 }, 1, None),
        vec![107]
    );
    // A key no candidate has gives nothing.
    assert_eq!(
        eligible(ControlRule::EventFree, 1, Some(7)),
        Vec::<u32>::new()
    );
}

#[test]
fn controls_are_distinct_within_a_case_and_carry_its_day() {
    let p = pool();
    let days = [T, T, T - 50];
    let cases = CaseEvents {
        day: &days,
        person: &[1, 2, 3],
        key: None,
    };
    let c = sample_controls(&cases, &p.candidates(false), ControlRule::EventFree, 3, 9).unwrap();
    assert_eq!(c.len(), 9);
    for k in 0..3 {
        let mut mine: Vec<u32> = (0..c.len())
            .filter(|&i| c.case[i] == k)
            .map(|i| c.person[i])
            .collect();
        assert_eq!(mine.len(), 3);
        mine.dedup();
        assert_eq!(mine.len(), 3);
    }
    assert!((0..c.len()).all(|i| c.day[i] == days[c.case[i] as usize]));
    assert_eq!(c.cases_short, Vec::<u32>::new());
    // The same seed gives the same controls.
    let again =
        sample_controls(&cases, &p.candidates(false), ControlRule::EventFree, 3, 9).unwrap();
    assert_eq!(c, again);
}

/// The share of draws that picks each eligible candidate, over `n` seeds.
fn frequencies(n: usize, candidates: &Candidates<'_>, eligible: &[u32]) -> Vec<f64> {
    let mut count = vec![0_usize; eligible.len()];
    for seed in 0..n as u64 {
        let c = sample_controls(
            &CaseEvents {
                day: &[T],
                person: &[u32::MAX],
                key: None,
            },
            candidates,
            ControlRule::EventFree,
            1,
            seed,
        )
        .unwrap();
        let k = eligible.iter().position(|&e| e == c.person[0]).unwrap();
        count[k] += 1;
    }
    count.iter().map(|&c| c as f64 / n as f64).collect()
}

#[test]
fn the_draw_is_uniform_by_rejection_and_by_listing() {
    // Each of e eligible candidates is drawn with probability 1/e. Over n
    // seeds its share has standard deviation sqrt((1/e)(1 - 1/e) / n); five of
    // them bound it with a chance below 1e-5 per candidate of failing.
    let n = 20_000;
    let p = pool();
    let eligible = [100, 101, 102, 106, 107, 109];
    let e = eligible.len() as f64;
    let sd = ((1.0 / e) * (1.0 - 1.0 / e) / n as f64).sqrt();
    for share in frequencies(n, &p.candidates(false), &eligible) {
        assert!((share - 1.0 / e).abs() < 5.0 * sd, "{share}");
    }
    // 2,000 candidates of whom four are observed on T: rejection fails and
    // the eligible ones are listed.
    let m = 2_000;
    let person: Vec<u32> = (0..m).collect();
    let start: Vec<i64> = (0..m)
        .map(|k| if k % 500 == 7 { T - 1 } else { T + 1 })
        .collect();
    let end = vec![T + 10; m as usize];
    let first = vec![None; m as usize];
    let wide = Candidates {
        person: &person,
        start: &start,
        end: &end,
        first_event: &first,
        key: None,
    };
    let eligible = [7, 507, 1007, 1507];
    let sd = (0.25 * 0.75 / n as f64).sqrt();
    for share in frequencies(n, &wide, &eligible) {
        assert!((share - 0.25).abs() < 5.0 * sd, "{share}");
    }
}

#[test]
fn sampling_does_not_depend_on_the_number_of_threads() {
    let m = 3_000_usize;
    let person: Vec<u32> = (0..m as u32).collect();
    let start: Vec<i64> = (0..m).map(|k| T - 400 + (k % 97) as i64).collect();
    let end: Vec<i64> = (0..m).map(|k| T + 300 - (k % 89) as i64).collect();
    let first: Vec<Option<i64>> = (0..m)
        .map(|k| (k % 3 == 0).then_some(T - 200 + (k % 401) as i64))
        .collect();
    let key: Vec<u64> = (0..m).map(|k| (k % 5) as u64).collect();
    let n = 2_500_usize;
    let day: Vec<i64> = (0..n).map(|k| T - 100 + (k % 211) as i64).collect();
    let case_person: Vec<u32> = (0..n as u32).collect();
    let case_key: Vec<u64> = (0..n).map(|k| (k % 5) as u64).collect();
    assert_same_on_pools(&POOLS, || {
        let c = sample_controls(
            &CaseEvents {
                day: &day,
                person: &case_person,
                key: Some(&case_key),
            },
            &Candidates {
                person: &person,
                start: &start,
                end: &end,
                first_event: &first,
                key: Some(&key),
            },
            ControlRule::FutureCases { gap: 3 },
            4,
            2026,
        )
        .unwrap();
        let mut out: Vec<f64> = c.person.iter().map(|&p| f64::from(p)).collect();
        out.extend(c.case.iter().map(|&p| f64::from(p)));
        out.extend(c.cases_short.iter().map(|&p| f64::from(p)));
        out
    });
}

#[test]
fn unusable_sampling_input_is_refused() {
    let p = pool();
    let cases = CaseEvents {
        day: &[T],
        person: &[1],
        key: None,
    };
    let run = |cases: &CaseEvents<'_>, cand: &Candidates<'_>, rule, k| {
        sample_controls(cases, cand, rule, k, 1).is_err()
    };
    assert!(run(&cases, &p.candidates(false), ControlRule::EventFree, 0));
    assert!(run(
        &cases,
        &p.candidates(false),
        ControlRule::FutureCases { gap: -1 },
        1
    ));
    // A key on one side only.
    assert!(run(&cases, &p.candidates(true), ControlRule::EventFree, 1));
    let short = CaseEvents {
        day: &[T, T],
        person: &[1],
        key: None,
    };
    assert!(run(&short, &p.candidates(false), ControlRule::EventFree, 1));
    let backwards = Candidates {
        start: &p.end,
        end: &p.start,
        ..p.candidates(false)
    };
    assert!(run(&cases, &backwards, ControlRule::EventFree, 1));
}

#[test]
fn case_and_control_windows_are_built_alike_and_grouped() {
    let p = pool();
    let days = [T, T - 10];
    let persons = [1, 2];
    let c = sample_controls(
        &CaseEvents {
            day: &days,
            person: &persons,
            key: None,
        },
        &p.candidates(false),
        ControlRule::EventFree,
        2,
        5,
    )
    .unwrap();
    let referent = Referent::Unidirectional { n: 2, step: 30 };
    let obs_start = [T - 100, T - 50];
    let obs_end = [T + 10, T + 10];
    let ctc = case_time_control_windows(
        &days,
        &persons,
        Some(Observation {
            start: &obs_start,
            end: &obs_end,
        }),
        &c,
        7,
        0,
        referent,
    )
    .unwrap();
    // The second case is observed from T - 50, so only one referent is left.
    let w = &ctc.windows;
    let sets: Vec<u32> = w.set.iter().copied().filter(|&s| s < 2).collect();
    assert_eq!(sets, vec![0, 0, 0, 1, 1]);
    // Each control's set is the same construction at its case's day.
    for k in 0..c.len() {
        let rows: Vec<usize> = (0..w.len()).filter(|&i| w.set[i] == 2 + k as u32).collect();
        let alone = build_windows(
            &[c.day[k]],
            &[c.person[k]],
            7,
            0,
            referent,
            Some(Observation {
                start: &[c.start[k]],
                end: &[c.end[k]],
            }),
        )
        .unwrap();
        let starts: Vec<i64> = rows.iter().map(|&i| w.start[i]).collect();
        assert_eq!(starts, alone.start);
        assert!(
            rows.iter()
                .all(|&i| !ctc.case_group[i] && w.person[i] == c.person[k])
        );
    }
    assert!((0..w.len()).all(|i| ctc.case_group[i] == (w.set[i] < 2)));
    // Without observation for the cases, nothing of theirs is dropped.
    let open = case_time_control_windows(&days, &persons, None, &c, 7, 0, referent).unwrap();
    assert_eq!(open.windows.set.iter().filter(|&&s| s < 2).count(), 6);
}

#[test]
fn stratified_referents_refuse_what_cannot_be_built() {
    let one = |r: Referent, h: i64| build_windows(&[T], &[1], h, 0, r, None);
    let fixed = |length| StratumUnit::Fixed { origin: 0, length };
    assert!(
        one(
            Referent::Stratified {
                unit: fixed(28),
                spacing: 7
            },
            7
        )
        .is_ok()
    );
    assert!(
        one(
            Referent::Stratified {
                unit: fixed(28),
                spacing: 7
            },
            8
        )
        .is_err()
    );
    assert!(
        one(
            Referent::Stratified {
                unit: fixed(0),
                spacing: 7
            },
            1
        )
        .is_err()
    );
    assert!(
        one(
            Referent::Stratified {
                unit: StratumUnit::Year,
                spacing: 0
            },
            1
        )
        .is_err()
    );
    assert!(one(Referent::TimeStratified { unit: fixed(-3) }, 1).is_err());
    // A 30-day hazard window with referents every 35 days in the year.
    let w = one(
        Referent::Stratified {
            unit: StratumUnit::Year,
            spacing: 35,
        },
        30,
    )
    .unwrap();
    // 2020-03-15: 74 days into the year, 291 to its end, so shifts -70, -35,
    // 35, .., 280.
    assert_eq!(w.len(), 1 + 2 + 8);
    // Strata per event: an event outside its stratum and short columns.
    let strata = |first: &'static [i64], last: &'static [i64]| EventStrata { first, last };
    let st = |s, sp| build_windows_in_strata(&[T], &[1], 1, 0, s, sp, None);
    assert!(st(strata(&[T - 14], &[T + 14]), 7).is_ok());
    assert!(st(strata(&[T + 1], &[T + 14]), 7).is_err());
    assert!(st(strata(&[T - 14, 0], &[T + 14, 0]), 7).is_err());
    assert!(st(strata(&[T - 14], &[T + 14]), 0).is_err());
}
