//! Cases of the data tools that R does not accept or accepts differently,
//! and the rules of the module documentation that no reference covers.

use epistat_survival::SurvData;
use epistat_survival::condense::{CondenseOptions, condense};
use epistat_survival::split::{SplitOptions, split};
use epistat_survival::tmerge::{Observations, Spans, Tmerge, TmergeOptions};

#[test]
fn split_without_cuts_returns_the_rows() {
    let stop = [3.0, 5.0, 9.0];
    let status = [1, 0, 1];
    let data = SurvData::builder(&stop, &status)
        .n_codes(1)
        .build()
        .unwrap();
    let got = split(&data, &[], &SplitOptions::default()).unwrap();
    assert_eq!(got.start, [0.0; 3]);
    assert_eq!(got.stop, stop);
    assert_eq!(got.status, status);
    assert_eq!(got.row, [0, 1, 2]);
    assert_eq!(got.episode, [1, 1, 1]);
    assert_eq!(got.ends_at_cut, [false; 3]);
}

#[test]
fn split_refuses_a_cut_that_is_not_finite() {
    let stop = [3.0];
    let status = [1];
    let data = SurvData::builder(&stop, &status)
        .n_codes(1)
        .build()
        .unwrap();
    for bad in [f64::NAN, f64::INFINITY] {
        assert!(split(&data, &[1.0, bad], &SplitOptions::default()).is_err());
    }
}

#[test]
fn split_keeps_one_of_two_cuts_that_the_tie_rule_made_equal() {
    let stop = [10.0];
    let status = [1];
    let data = SurvData::builder(&stop, &status)
        .n_codes(1)
        .build()
        .unwrap();
    let got = split(&data, &[5.0, 5.0 + 1e-13], &SplitOptions::default()).unwrap();
    assert_eq!(got.stop, [5.0, 10.0]);
    assert_eq!(got.cuts, [5.0]);
}

#[test]
fn condense_without_a_merge_returns_every_row() {
    let (start, stop, status, id) = ([0.0, 12.0], [10.0, 25.0], [0, 0], [1, 1]);
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .n_codes(1)
        .build()
        .unwrap();
    let got = condense(
        &data,
        &[1, 1],
        &CondenseOptions {
            merge_over_events: true,
        },
    )
    .unwrap();
    assert_eq!(got.row, [0, 1]);
}

#[test]
fn condense_needs_a_start_and_an_id() {
    let (stop, status, id) = ([10.0], [0], [1]);
    let options = CondenseOptions {
        merge_over_events: true,
    };
    let no_start = SurvData::builder(&stop, &status)
        .id(&id)
        .n_codes(1)
        .build()
        .unwrap();
    assert!(condense(&no_start, &[0], &options).is_err());
    let start = [0.0];
    let no_id = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(1)
        .build()
        .unwrap();
    assert!(condense(&no_id, &[0], &options).is_err());
}

fn one_subject() -> Tmerge {
    Tmerge::new(
        &[7],
        &Spans {
            id: &[7],
            tstart: None,
            tstop: &[10.0],
        },
        TmergeOptions::default(),
    )
    .unwrap()
}

#[test]
fn an_operation_with_no_valid_observation_changes_nothing() {
    let mut tm = one_subject();
    let obs = Observations {
        id: &[99],
        time: &[Some(3.0)],
        value: None,
    };
    tm.event("e", &obs).unwrap();
    assert_eq!(tm.len(), 1);
    assert_eq!(tm.variable("e").unwrap().values, [Some(0.0)]);
    assert_eq!(tm.counts()[0].1.missid, 1);
}

#[test]
fn a_time_that_is_not_finite_is_refused() {
    let mut tm = one_subject();
    let obs = Observations {
        id: &[7],
        time: &[Some(f64::INFINITY)],
        value: None,
    };
    assert!(tm.event("e", &obs).is_err());
}

#[test]
fn an_event_cannot_take_the_name_of_a_covariate() {
    let mut tm = one_subject();
    let obs = Observations {
        id: &[7],
        time: &[Some(3.0)],
        value: None,
    };
    tm.tdc("x", &obs, None).unwrap();
    assert!(tm.event("x", &obs).is_err());
}

#[test]
fn a_first_call_from_events_needs_each_subject_once() {
    let options = TmergeOptions::default();
    let time = [Some(5.0), Some(6.0)];
    assert!(Tmerge::from_first_events(&[1, 2], &[1, 1], &time, options).is_err());
    assert!(Tmerge::from_first_events(&[1, 2], &[1, 2], &time, options).is_ok());
    assert!(Tmerge::from_first_events(&[1, 2], &[1, 2], &[Some(5.0), None], options).is_err());
}

#[test]
fn a_missing_value_in_cumevent_is_refused() {
    let mut tm = one_subject();
    let obs = Observations {
        id: &[7],
        time: &[Some(3.0)],
        value: Some(&[None]),
    };
    let mut tm_keep = Tmerge::new(
        &[7],
        &Spans {
            id: &[7],
            tstart: None,
            tstop: &[10.0],
        },
        TmergeOptions {
            na_rm: false,
            ..TmergeOptions::default()
        },
    )
    .unwrap();
    assert!(tm_keep.cumevent("k", &obs).is_err());
    // With na_rm the observation is dropped and nothing is recorded.
    tm.cumevent("k", &obs).unwrap();
    assert_eq!(tm.variable("k").unwrap().values, [Some(0.0)]);
}

#[test]
fn split_rows_carry_input_columns_and_refit_as_follow_up() {
    let stop = [4.0, 7.0, 2.0];
    let status = [1, 0, 1];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let got = split(&data, &[3.0, 5.0], &SplitOptions::default()).unwrap();
    let age = [61.0, 47.0, 70.0];
    let gathered = got.gather(&age).unwrap();
    let by_hand: Vec<f64> = got.row.iter().map(|&r| age[r as usize]).collect();
    assert_eq!(gathered, by_hand);
    assert!(got.gather(&age[..2]).is_err());
    let buf = got.to_surv_buf().unwrap();
    let back = buf.as_data();
    assert_eq!(back.n_rows(), got.row.len());
    let follow = back.followup().unwrap();
    assert_eq!(follow.stop(), &got.stop[..]);
    assert_eq!(follow.start(), Some(&got.start[..]));
}
