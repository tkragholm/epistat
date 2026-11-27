//! `sccs`: the input it refuses, people it drops and counts, the cases the R
//! reference does not cover (a piece of zero length, a window cut by the end
//! of observation, a window that overlaps another, an exposure before the
//! start of observation) and the piece rules written out.

use epistat_linalg::Mat;
use epistat_survival::Level;
use epistat_survival::sccs::{
    ExposureSeries, PieceSpec, SccsData, SccsOptions, WindowSpec, build_pieces, fit,
};

fn one_person<'a>(
    events: &'a [f64],
    exposures: &'a [f64],
    windows: &'a [WindowSpec],
    age_cuts: &'a [f64],
) -> PieceSpec<'a> {
    PieceSpec {
        start: &[0.0],
        end: &[100.0],
        event_offsets: Box::leak(Box::new([0, events.len()])),
        events,
        exposures: Box::leak(Box::new([ExposureSeries {
            offsets: Box::leak(Box::new([0, exposures.len()])),
            dates: exposures,
            windows,
        }])),
        age_cuts,
        calendar: None,
    }
}

const W: WindowSpec = WindowSpec {
    from: 0.0,
    to: 10.0,
};

#[test]
fn a_window_cut_by_the_end_of_observation_is_shorter() {
    let p = build_pieces(&one_person(&[5.0], &[95.0], &[W], &[])).unwrap();
    assert_eq!(p.start, vec![0.0, 95.0]);
    assert_eq!(p.end, vec![95.0, 100.0]);
    assert_eq!(p.window, vec![vec![None, Some(0)]]);
}

#[test]
fn an_exposure_before_observation_gives_the_rest_of_its_window() {
    let p = build_pieces(&one_person(&[5.0], &[-4.0], &[W], &[])).unwrap();
    assert_eq!(p.start, vec![0.0, 6.0]);
    assert_eq!(p.window, vec![vec![Some(0), None]]);
    // An exposure whose window ends before observation starts adds no cut.
    let q = build_pieces(&one_person(&[5.0], &[-40.0], &[W], &[])).unwrap();
    assert_eq!(q.start, vec![0.0]);
    assert_eq!(q.window, vec![vec![None]]);
}

#[test]
fn an_event_on_a_boundary_belongs_to_the_piece_that_starts_there() {
    let p = build_pieces(&one_person(&[10.0, 20.0, 0.0], &[10.0], &[W], &[20.0])).unwrap();
    assert_eq!(p.start, vec![0.0, 10.0, 20.0]);
    assert_eq!(p.events, vec![1.0, 1.0, 1.0]);
    assert_eq!(p.age_band, vec![0, 0, 1]);
}

#[test]
fn the_latest_exposure_date_takes_an_overlapping_piece() {
    let pre = WindowSpec {
        from: -5.0,
        to: 0.0,
    };
    let after = WindowSpec {
        from: 0.0,
        to: 30.0,
    };
    // Doses on 20 and 40: days 35 to 39 are in the pre window of the second
    // dose and in the late window of the first, and take the second's.
    let p = build_pieces(&one_person(&[], &[20.0, 40.0], &[pre, after], &[])).unwrap();
    let at = |t: f64| {
        let i = p.start.iter().rposition(|&s| s <= t).unwrap();
        p.window[0][i]
    };
    assert_eq!(at(30.0), Some(1));
    assert_eq!(at(36.0), Some(0));
    assert_eq!(at(42.0), Some(1));
    assert_eq!(at(71.0), None);
}

#[test]
fn input_that_is_not_usable_is_refused() {
    let bad = |spec: PieceSpec<'_>| assert!(build_pieces(&spec).is_err());
    bad(one_person(&[100.0], &[], &[], &[]));
    bad(one_person(&[-1.0], &[], &[], &[]));
    bad(one_person(&[f64::NAN], &[], &[], &[]));
    bad(one_person(&[], &[f64::NAN], &[W], &[]));
    bad(one_person(&[], &[], &[], &[50.0, 50.0]));
    bad(one_person(&[], &[], &[], &[60.0, 50.0]));
    let overlap = [
        W,
        WindowSpec {
            from: 5.0,
            to: 20.0,
        },
    ];
    bad(one_person(&[], &[], &overlap, &[]));
    bad(one_person(
        &[],
        &[],
        &[WindowSpec { from: 3.0, to: 3.0 }],
        &[],
    ));
    let mut empty = one_person(&[], &[], &[], &[]);
    empty.end = &[0.0];
    bad(empty);
    let mut offsets = one_person(&[1.0], &[], &[], &[]);
    offsets.event_offsets = &[0, 2];
    bad(offsets);
}

fn data_of(person: &[u32], length: &[f64], events: &[f64]) -> (Vec<u32>, Vec<f64>, Vec<f64>) {
    (person.to_vec(), length.to_vec(), events.to_vec())
}

#[test]
fn the_fit_refuses_what_it_cannot_use_and_counts_what_it_drops() {
    let x = Mat::from_fn(6, 1, |i, _| (i % 2) as f64);
    let opts = SccsOptions::default();
    let run = |p: &[u32], l: &[f64], y: &[f64], x: &Mat<f64>| {
        let (p, l, y) = data_of(p, l, y);
        fit(
            &SccsData {
                person: &p,
                length: &l,
                events: &y,
            },
            x.as_ref(),
            &opts,
        )
    };
    let person = [0, 0, 1, 1, 2, 2];
    let length = [1.0, 2.0, 1.5, 1.0, 2.0, 2.0];
    let events = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    // A zero length, a negative event count, a person split in two runs, a
    // design of the wrong height and a column that is not finite are errors.
    let mut zero = length;
    zero[3] = 0.0;
    assert!(run(&person, &zero, &events, &x).is_err());
    let mut negative = events;
    negative[0] = -1.0;
    assert!(run(&person, &length, &negative, &x).is_err());
    assert!(run(&[0, 1, 0, 1, 2, 2], &length, &events, &x).is_err());
    assert!(run(&person, &length, &events[..5], &x).is_err());
    let nan = Mat::from_fn(6, 1, |i, _| if i == 2 { f64::NAN } else { 1.0 });
    assert!(run(&person, &length, &events, &nan).is_err());
    let none = Mat::<f64>::zeros(6, 0);
    assert!(run(&person, &length, &events, &none).is_err());
    // Person 2 has no events and is counted.
    let f = run(&person, &length, &events, &x).unwrap();
    assert_eq!(f.n_persons, 3);
    assert_eq!(f.n_persons_used, 2);
    assert_eq!(f.n_persons_without_events, 1);
    assert_eq!(f.n_persons_one_piece, 0);
    // One piece with an event is dropped and counted.
    let g = run(
        &[0, 0, 1, 1, 2],
        &length[..5],
        &[1.0, 0.0, 0.0, 1.0, 1.0],
        &x.as_ref().get(0..5, ..).to_owned(),
    )
    .unwrap();
    assert_eq!(g.n_persons_one_piece, 1);
    // Nobody left: no estimate and no iterations.
    let h = run(
        &[0, 1],
        &[1.0, 1.0],
        &[1.0, 1.0],
        &x.as_ref().get(0..2, ..).to_owned(),
    )
    .unwrap();
    assert!(
        h.coefficients[0].is_none()
            && h.log_likelihood.is_none()
            && h.null_log_likelihood.is_none()
    );
    assert_eq!(h.iterations, 0);
}

#[test]
fn the_estimate_of_two_pieces_per_person_is_the_log_odds_ratio_of_the_counts() {
    // 20 people with a window piece of the same length as the baseline piece:
    // a events fall in the window, b in the baseline, beta = ln(a / b).
    let n = 20;
    let person: Vec<u32> = (0..n).flat_map(|i| [i, i]).collect();
    let length = vec![5.0; 2 * n as usize];
    let events: Vec<f64> = (0..n)
        .flat_map(|i| if i < 12 { [1.0, 0.0] } else { [0.0, 1.0] })
        .collect();
    let x = Mat::from_fn(2 * n as usize, 1, |i, _| (i % 2) as f64);
    let f = fit(
        &SccsData {
            person: &person,
            length: &length,
            events: &events,
        },
        x.as_ref(),
        &SccsOptions::default(),
    )
    .unwrap();
    // Window rows are the odd ones: 8 events there, 12 in the baseline.
    let beta = f.coefficients[0].unwrap();
    assert!((beta - (8.0f64 / 12.0).ln()).abs() < 1e-9);
    let var = f.covariance[0].unwrap();
    assert!((var - (1.0 / 8.0 + 1.0 / 12.0)).abs() < 1e-9);
    // The interval and test follow from the closed form. The fit stops within
    // 1e-10 relative in the log likelihood, so beta and var are within about
    // 1e-9 of it, and each derived quantity within a few times that.
    let se = (1.0_f64 / 8.0 + 1.0 / 12.0).sqrt();
    assert_eq!(f.standard_errors[0], Some(var.sqrt()));
    let ri = f.relative_incidence(Level::default(), false).unwrap()[0].unwrap();
    let z = 1.959_963_984_540_054;
    assert!((ri.relative_incidence - 8.0 / 12.0).abs() < 1e-8);
    assert!((ri.lower - ((8.0f64 / 12.0).ln() - z * se).exp()).abs() < 1e-8);
    assert!((ri.upper - ((8.0f64 / 12.0).ln() + z * se).exp()).abs() < 1e-8);
    assert!((ri.z - (8.0f64 / 12.0).ln() / se).abs() < 1e-8);
    // R 4.6.2: 2 * pnorm(-abs(log(8/12) / sqrt(1/8 + 1/12))).
    assert!((ri.p_value - 0.374_363_512_135_005).abs() < 1e-8);
    // Every person is observed in and out of the window and has an event.
    assert_eq!(f.n_informative_by_column, vec![20]);
    assert!(f.relative_incidence(Level::default(), true).is_err());
}

#[test]
fn two_exposure_series_have_their_own_windows_and_columns() {
    // Series 0 opens [0, 10) on day 20, series 1 opens [0, 30) and [30, 40)
    // on day 25, so the two overlap on days 25 to 29.
    let first = [WindowSpec {
        from: 0.0,
        to: 10.0,
    }];
    let second = [
        WindowSpec {
            from: 0.0,
            to: 30.0,
        },
        WindowSpec {
            from: 30.0,
            to: 40.0,
        },
    ];
    let series = [
        ExposureSeries {
            offsets: &[0, 1, 1],
            dates: &[20.0],
            windows: &first,
        },
        ExposureSeries {
            offsets: &[0, 1, 1],
            dates: &[25.0],
            windows: &second,
        },
    ];
    let spec = PieceSpec {
        start: &[0.0, 0.0],
        end: &[100.0, 50.0],
        event_offsets: &[0, 2, 3],
        events: &[27.0, 60.0, 10.0],
        exposures: &series,
        ..PieceSpec::default()
    };
    let p = build_pieces(&spec).unwrap();
    assert_eq!(p.start, vec![0.0, 20.0, 25.0, 30.0, 55.0, 65.0, 0.0]);
    assert_eq!(
        p.window,
        vec![
            vec![None, Some(0), Some(0), None, None, None, None],
            vec![None, None, Some(0), Some(0), Some(1), None, None],
        ]
    );
    assert_eq!(p.events, vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    assert_eq!(p.n_windows, vec![1, 2]);
    let x = p.design();
    assert_eq!(x.ncols(), 3);
    // Days 25 to 29 are in both: the two indicators are both 1.
    assert_eq!((x[(2, 0)], x[(2, 1)], x[(2, 2)]), (1.0, 1.0, 0.0));
    assert_eq!(p.exposed_cases(0), 1);
    assert_eq!(p.exposed_cases(1), 1);
    let summary = p.window_summary();
    assert_eq!(summary.len(), 3);
    assert_eq!(
        (summary[0].cases, summary[0].events, summary[0].time),
        (1, 1.0, 10.0)
    );
    assert_eq!(
        (summary[1].cases, summary[1].events, summary[1].time),
        (1, 1.0, 30.0)
    );
    assert_eq!((summary[2].series, summary[2].window), (1, 1));
    assert_eq!((summary[2].events, summary[2].time), (1.0, 10.0));
    assert_eq!(summary[2].informative_cases, 1);
}

#[test]
fn calendar_input_that_is_not_usable_is_refused() {
    use epistat_survival::sccs::{Calendar, SeasonStart};
    let with = |calendar: Calendar<'_>| {
        build_pieces(&PieceSpec {
            start: &[0.0],
            end: &[400.0],
            event_offsets: &[0, 0],
            calendar: Some(calendar),
            ..PieceSpec::default()
        })
    };
    let march = SeasonStart { month: 3, day: 1 };
    let ok = with(Calendar {
        origin: &[0.0],
        cuts: &[],
        season_starts: &[march, SeasonStart { month: 9, day: 1 }],
    });
    assert!(ok.is_ok());
    for starts in [
        &[march][..],
        &[march, SeasonStart { month: 2, day: 29 }][..],
        &[SeasonStart { month: 9, day: 1 }, march][..],
        &[march, SeasonStart { month: 13, day: 1 }][..],
        &[march, SeasonStart { month: 4, day: 31 }][..],
    ] {
        assert!(
            with(Calendar {
                origin: &[0.0],
                cuts: &[],
                season_starts: starts,
            })
            .is_err()
        );
    }
    assert!(
        with(Calendar {
            origin: &[],
            ..Calendar::default()
        })
        .is_err()
    );
    assert!(
        with(Calendar {
            origin: &[f64::NAN],
            ..Calendar::default()
        })
        .is_err()
    );
    assert!(
        with(Calendar {
            origin: &[0.0],
            cuts: &[200.0, 100.0],
            season_starts: &[],
        })
        .is_err()
    );
}

#[test]
fn event_dependent_exposure_drops_later_doses_and_refuses_bad_windows() {
    use epistat_survival::sccs::{EventDependentOptions, EventDependentSpec, fit_event_dependent};
    // Twelve cases: a dose on day 40 and a second on day 80; the event on day
    // 50 + i, so the second dose of the cases with their event before day 80
    // comes after it and is dropped.
    let n = 12;
    let start = vec![0.0; n];
    let end = vec![200.0; n];
    let event: Vec<f64> = (0..n).map(|i| 45.0 + 10.0 * i as f64).collect();
    let dose_offsets: Vec<usize> = (0..=n).map(|i| 2 * i).collect();
    let doses: Vec<f64> = (0..n).flat_map(|_| [40.0, 80.0]).collect();
    let windows = [WindowSpec {
        from: 0.0,
        to: 14.0,
    }];
    let spec = EventDependentSpec {
        start: &start,
        end: &end,
        event: &event,
        dose_offsets: &dose_offsets,
        doses: &doses,
        windows: &windows,
        age_cuts: &[],
        same_dose_parameters: true,
    };
    let f = fit_event_dependent(&spec, &EventDependentOptions::default()).unwrap();
    // Events on days 45, 55, 65, 75 precede the second dose.
    assert_eq!(f.n_doses_after_event, 4);
    assert_eq!(f.n_cases, n);
    // Every case starts unexposed, so all are at level 0, all have dose 1,
    // and the eight that keep the second dose have level 2.
    assert_eq!(f.n_strata_by_level, vec![12, 12, 8]);
    let bad = |w: &[WindowSpec]| {
        fit_event_dependent(
            &EventDependentSpec { windows: w, ..spec },
            &EventDependentOptions::default(),
        )
        .is_err()
    };
    assert!(bad(&[WindowSpec {
        from: -5.0,
        to: 14.0
    }]));
    assert!(bad(&[WindowSpec {
        from: 0.0,
        to: f64::INFINITY
    }]));
    assert!(bad(&[
        WindowSpec { from: 0.0, to: 7.0 },
        WindowSpec {
            from: 8.0,
            to: 14.0
        }
    ]));
    assert!(bad(&[]));
    let late = [300.0; 12];
    assert!(
        fit_event_dependent(
            &EventDependentSpec {
                event: &late,
                ..spec
            },
            &EventDependentOptions::default()
        )
        .is_err()
    );
}
