//! `crossover`: what it refuses, the rules written out on small cases, and
//! exposure intervals that overlap, touch or belong to other people.

use epistat_survival::crossover::{
    ExposureIntervals, Measure, Observation, Referent, StratumUnit, build_windows,
    case_time_control_design, exposure_in_windows,
};

// 2020-03-15 is day 18336, a Sunday.
const T: i64 = 18_336;

fn one(
    referent: Referent,
    h: i64,
    lag: i64,
    obs: Option<(i64, i64)>,
) -> epistat_survival::crossover::Windows {
    let (s, e) = obs.map_or((vec![], vec![]), |(a, b)| (vec![a], vec![b]));
    let o = obs.map(|_| Observation { start: &s, end: &e });
    build_windows(&[T], &[7], h, lag, referent, o).unwrap()
}

#[test]
fn unidirectional_windows_are_the_hazard_window_moved_back() {
    let w = one(Referent::Unidirectional { n: 2, step: 7 }, 3, 1, None);
    assert_eq!(w.start, vec![T - 4, T - 18, T - 11]);
    assert_eq!(w.end, vec![T - 2, T - 16, T - 9]);
    assert_eq!(w.is_case, vec![true, false, false]);
    assert_eq!(w.set, vec![0, 0, 0]);
    assert_eq!(w.person, vec![7, 7, 7]);
}

#[test]
fn a_forward_window_that_holds_the_event_day_is_dropped_and_counted() {
    // h = 7, step = 7: the window after is [T, T + 6], which holds T.
    let w = one(Referent::Bidirectional { n: 1, step: 7 }, 7, 0, None);
    assert_eq!(w.len(), 2);
    assert_eq!(w.n_referents_on_event_day, 1);
}

#[test]
fn time_stratified_referents_are_the_other_same_weekdays_of_the_month() {
    // The Sundays of March 2020: the 1st, 8th, 22nd and 29th besides the 15th.
    let w = one(
        Referent::TimeStratified {
            unit: StratumUnit::Month,
        },
        1,
        0,
        None,
    );
    let days: Vec<i64> = w.start.iter().map(|s| s + 1 - T).collect();
    assert_eq!(days, vec![0, -14, -7, 7, 14]);
}

#[test]
fn observation_drops_referents_then_sets() {
    // Observation from 10 days before the event: the referent 14 days back is
    // outside, the others are inside.
    let w = one(
        Referent::TimeStratified {
            unit: StratumUnit::Month,
        },
        1,
        0,
        Some((T - 10, T + 30)),
    );
    assert_eq!(w.n_referents_outside_observation, 1);
    assert_eq!(w.len(), 4);
    // Nothing inside: the set is dropped.
    let w = one(
        Referent::TimeStratified {
            unit: StratumUnit::Month,
        },
        1,
        0,
        Some((T - 1, T + 3)),
    );
    assert!(w.is_empty());
    assert_eq!(w.sets_without_referent, vec![0]);
    // The hazard window itself outside observation.
    let w = one(
        Referent::TimeStratified {
            unit: StratumUnit::Month,
        },
        3,
        0,
        Some((T - 1, T + 30)),
    );
    assert_eq!(w.sets_case_outside_observation, vec![0]);
    assert_eq!(w.sets_without_referent, Vec::<u32>::new());
}

#[test]
fn unusable_input_is_refused() {
    let ts = Referent::TimeStratified {
        unit: StratumUnit::Year,
    };
    let bad = |h, lag, r, ev: &[i64], pe: &[u32]| build_windows(ev, pe, h, lag, r, None).is_err();
    assert!(bad(0, 0, ts, &[T], &[1]));
    assert!(bad(1, -1, ts, &[T], &[1]));
    assert!(bad(8, 0, ts, &[T], &[1]));
    assert!(bad(1, 0, ts, &[T], &[1, 2]));
    assert!(bad(
        3,
        0,
        Referent::Unidirectional { n: 2, step: 2 },
        &[T],
        &[1]
    ));
    assert!(bad(
        1,
        0,
        Referent::Bidirectional { n: 0, step: 7 },
        &[T],
        &[1]
    ));
    let o = Observation {
        start: &[T + 1],
        end: &[T],
    };
    assert!(build_windows(&[T], &[1], 1, 0, ts, Some(o)).is_err());
    let o = Observation {
        start: &[],
        end: &[],
    };
    assert!(build_windows(&[T], &[1], 1, 0, ts, Some(o)).is_err());
    let w = one(ts, 1, 0, None);
    let bad_exposure = ExposureIntervals {
        person: &[7],
        start: &[5],
        end: &[4],
    };
    assert!(exposure_in_windows(&w, &bad_exposure, Measure::Days).is_err());
    let short = ExposureIntervals {
        person: &[7],
        start: &[],
        end: &[4],
    };
    assert!(exposure_in_windows(&w, &short, Measure::Days).is_err());
    assert!(case_time_control_design(&[1.0], &[true, false]).is_err());
}

#[test]
fn exposure_counts_a_day_once_and_only_for_the_person() {
    let w = one(Referent::Unidirectional { n: 1, step: 10 }, 5, 0, None);
    // Hazard window [T-5, T-1], referent [T-15, T-11].
    let exposure = ExposureIntervals {
        person: &[7, 7, 7, 8, 7],
        start: &[T - 8, T - 4, T - 3, T - 20, T - 30],
        end: &[T - 5, T - 2, T - 1, T, T - 14],
    };
    let days = exposure_in_windows(&w, &exposure, Measure::Days).unwrap();
    // Hazard: T-5, T-4..T-1 are covered (T-8..T-5 gives T-5; T-4..T-2 and T-3..T-1 merge), so 5.
    // Referent: [T-30, T-14] covers T-15..T-14 of the window, 2 days.
    assert_eq!(days, vec![5.0, 2.0]);
    let any = exposure_in_windows(&w, &exposure, Measure::Indicator).unwrap();
    assert_eq!(any, vec![1.0, 1.0]);
    // Intervals that only touch are merged: no day is lost or counted twice.
    let touching = ExposureIntervals {
        person: &[7, 7],
        start: &[T - 5, T - 3],
        end: &[T - 4, T - 1],
    };
    assert_eq!(
        exposure_in_windows(&w, &touching, Measure::Days).unwrap(),
        vec![5.0, 0.0]
    );
}

#[test]
fn the_case_time_control_design_has_the_exposure_and_its_product_with_the_group() {
    let x = case_time_control_design(&[1.0, 0.0, 1.0], &[true, true, false]).unwrap();
    assert_eq!((x.nrows(), x.ncols()), (3, 2));
    assert_eq!([x[(0, 0)], x[(1, 0)], x[(2, 0)]], [1.0, 0.0, 1.0]);
    assert_eq!([x[(0, 1)], x[(1, 1)], x[(2, 1)]], [1.0, 0.0, 0.0]);
}
