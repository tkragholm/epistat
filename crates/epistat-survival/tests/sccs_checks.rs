//! `sccs`: the input it refuses, people it drops and counts, the cases the R
//! reference does not cover (a piece of zero length, a window cut by the end
//! of observation, a window that overlaps another, an exposure before the
//! start of observation) and the piece rules written out.

use epistat_linalg::Mat;
use epistat_survival::sccs::{PieceSpec, SccsData, SccsOptions, WindowSpec, build_pieces, fit};

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
        exposure_offsets: Box::leak(Box::new([0, exposures.len()])),
        exposures,
        age_cuts,
        windows,
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
    assert_eq!(p.window, vec![None, Some(0)]);
}

#[test]
fn an_exposure_before_observation_gives_the_rest_of_its_window() {
    let p = build_pieces(&one_person(&[5.0], &[-4.0], &[W], &[])).unwrap();
    assert_eq!(p.start, vec![0.0, 6.0]);
    assert_eq!(p.window, vec![Some(0), None]);
    // An exposure whose window ends before observation starts adds no cut.
    let q = build_pieces(&one_person(&[5.0], &[-40.0], &[W], &[])).unwrap();
    assert_eq!(q.start, vec![0.0]);
    assert_eq!(q.window, vec![None]);
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
        p.window[i]
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
}
