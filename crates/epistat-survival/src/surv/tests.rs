use super::{Outcome, SurvBuf, SurvData};
use crate::check::check_row_count;

fn message<T>(result: crate::Result<T>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.to_string(),
    }
}

fn simple() -> (Vec<f64>, Vec<i32>) {
    (vec![1.0, 2.0, 3.0], vec![0, 1, 0])
}

#[test]
fn a_right_censored_input_builds_with_defaults() {
    let (stop, status) = simple();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let surv = data.followup().unwrap();
    assert!(surv.start().is_none());
    assert_eq!(surv.stop(), &stop[..]);
    assert_eq!(surv.status(), &status[..]);
    assert_eq!(surv.n_codes(), 1);
    assert_eq!(surv.len(), 3);
    assert!(!surv.is_empty());
    assert!(data.weight().is_none() && data.stratum().is_none());
    assert!(data.cluster().is_none() && data.id().is_none() && data.istate().is_none());
    assert_eq!((data.n_rows(), data.n_used()), (3, 3));
    assert_eq!(
        (data.n_strata(), data.n_clusters(), data.n_ids()),
        (0, 0, 0)
    );
    assert!(matches!(data.outcome(), Outcome::Followup(_)));
}

#[test]
fn no_rows_is_a_valid_input() {
    let data = SurvData::builder(&[], &[]).build().unwrap();
    assert_eq!((data.n_rows(), data.n_used()), (0, 0));
    assert!(data.followup().unwrap().is_empty());
}

#[test]
fn columns_of_different_lengths_are_refused() {
    let (stop, status) = simple();
    assert!(message(SurvData::builder(&stop, &status[..2]).build()).contains("status has 2"));
    assert!(
        message(SurvData::builder(&stop, &status).weight(&[1.0]).build()).contains("weight has 1")
    );
    assert!(
        message(
            SurvData::builder(&stop, &status)
                .stratum(&[0, 1, 2, 3])
                .build()
        )
        .contains("stratum has 4")
    );
    SurvData::builder(&stop, &status)
        .weight(&[1.0; 3])
        .build()
        .unwrap();
}

#[test]
#[cfg(target_pointer_width = "64")]
fn more_rows_than_u32_are_refused() {
    check_row_count(u32::MAX as usize).unwrap();
    assert!(check_row_count(u32::MAX as usize + 1).is_err());
}

#[test]
fn n_codes_of_zero_is_refused() {
    let (stop, status) = (vec![1.0], vec![0]);
    assert!(message(SurvData::builder(&stop, &status).n_codes(0).build()).contains("n_codes is 0"));
    SurvData::builder(&stop, &status)
        .n_codes(1)
        .build()
        .unwrap();
}

#[test]
fn non_finite_times_and_weights_are_refused_with_count_and_first_row() {
    let status = [0, 0, 0];
    let stop = [1.0, f64::NAN, f64::INFINITY];
    assert_eq!(
        message(SurvData::builder(&stop, &status).build()),
        "stop: 2 values are not finite; drop or resolve them before building, first at row 1"
    );
    let start = [0.0, f64::NEG_INFINITY, 0.0];
    assert!(
        message(SurvData::builder(&[1.0; 3], &status).start(&start).build())
            .starts_with("start: 1 values are not finite")
    );
    let weight = [1.0, 1.0, f64::NAN];
    assert!(
        message(
            SurvData::builder(&[1.0; 3], &status)
                .weight(&weight)
                .build()
        )
        .ends_with("first at row 2")
    );
    // A very large finite time is fine.
    SurvData::builder(&[f64::MAX, -1e308, 0.0], &status)
        .build()
        .unwrap();
}

#[test]
fn a_negative_weight_is_refused_and_zero_is_not() {
    let (stop, status) = simple();
    assert_eq!(
        message(
            SurvData::builder(&stop, &status)
                .weight(&[1.0, -0.5, -2.0])
                .build()
        ),
        "weight: 2 values are negative, first at row 1"
    );
    SurvData::builder(&stop, &status)
        .weight(&[1.0, 0.0, -0.0])
        .build()
        .unwrap();
}

#[test]
fn weight_zero_rows_count_in_n_rows_and_not_in_n_used() {
    let (stop, status) = simple();
    let data = SurvData::builder(&stop, &status)
        .weight(&[2.0, 0.0, 0.5])
        .build()
        .unwrap();
    assert_eq!((data.n_rows(), data.n_used()), (3, 2));
}

#[test]
fn a_status_outside_zero_to_n_codes_is_refused() {
    let stop = [1.0, 2.0, 3.0];
    assert_eq!(
        message(SurvData::builder(&stop, &[0, 2, 3]).build()),
        "status: 2 values are outside 0..=1, first at row 1"
    );
    assert!(message(SurvData::builder(&stop, &[0, -1, 1]).build()).ends_with("first at row 1"));
    SurvData::builder(&stop, &[0, 1, 0]).build().unwrap();
    SurvData::builder(&stop, &[0, 2, 3])
        .n_codes(3)
        .build()
        .unwrap();
    assert!(message(SurvData::builder(&stop, &[0, 2, 4]).n_codes(3).build()).contains("0..=3"));
}

#[test]
fn an_initial_state_above_n_codes_is_refused_and_zero_is_unknown() {
    let (stop, status) = simple();
    assert_eq!(
        message(
            SurvData::builder(&stop, &status)
                .n_codes(2)
                .istate(&[1, 3, 0])
                .build()
        ),
        "istate: 1 values are above n_codes = 2, first at row 1"
    );
    let data = SurvData::builder(&stop, &status)
        .n_codes(2)
        .istate(&[1, 2, 0])
        .build()
        .unwrap();
    assert_eq!(data.istate(), Some(&[1, 2, 0][..]));
}

#[test]
fn an_interval_with_start_not_below_stop_is_refused() {
    let (stop, status) = simple();
    assert_eq!(
        message(
            SurvData::builder(&stop, &status)
                .start(&[0.0, 2.0, 4.0])
                .build()
        ),
        "start: 2 values are not below their stop, first at row 1"
    );
    let data = SurvData::builder(&stop, &status)
        .start(&[0.0, 1.999, 2.0])
        .build()
        .unwrap();
    assert_eq!(
        data.followup().unwrap().start(),
        Some(&[0.0, 1.999, 2.0][..])
    );
}

#[test]
fn without_start_any_finite_stop_is_allowed() {
    SurvData::builder(&[-5.0, 0.0, 3.0], &[0, 0, 0])
        .build()
        .unwrap();
}

#[test]
fn already_dense_codes_are_borrowed_and_counted() {
    let (stop, status) = simple();
    let codes = [1_u32, 0, 1];
    let data = SurvData::builder(&stop, &status)
        .stratum(&codes)
        .build()
        .unwrap();
    assert_eq!(data.stratum().unwrap().as_ptr(), codes.as_ptr());
    assert_eq!(data.n_strata(), 2);
}

#[test]
fn sparse_codes_become_dense_in_sorted_code_order() {
    let stop = [1.0; 5];
    let status = [0; 5];
    // Small maximum, below the row count, goes through the table.
    let small = [4_u32, 0, 4, 2, 0];
    let data = SurvData::builder(&stop, &status)
        .cluster(&small)
        .build()
        .unwrap();
    assert_eq!(data.cluster().unwrap(), &[2, 0, 2, 1, 0]);
    assert_eq!(data.n_clusters(), 3);
    // A maximum above the row count goes through sorting.
    let large = [4_000_000_000_u32, 7, 4_000_000_000, 100, 7];
    let data = SurvData::builder(&stop, &status)
        .id(&large)
        .build()
        .unwrap();
    assert_eq!(data.id().unwrap(), &[2, 0, 2, 1, 0]);
    assert_eq!(data.n_ids(), 3);
}

#[test]
fn codes_starting_above_zero_are_recoded_even_when_contiguous() {
    let stop = [1.0; 3];
    let status = [0; 3];
    let data = SurvData::builder(&stop, &status)
        .stratum(&[1, 2, 1])
        .build()
        .unwrap();
    assert_eq!(data.stratum().unwrap(), &[0, 1, 0]);
    assert_eq!(data.n_strata(), 2);
}

#[test]
fn a_single_code_gives_one_level() {
    let stop = [1.0; 3];
    let status = [0; 3];
    let data = SurvData::builder(&stop, &status)
        .stratum(&[0, 0, 0])
        .cluster(&[9, 9, 9])
        .build()
        .unwrap();
    assert_eq!((data.n_strata(), data.n_clusters()), (1, 1));
    assert_eq!(data.cluster().unwrap(), &[0, 0, 0]);
}

#[test]
fn the_buffer_checks_like_the_borrowed_builder() {
    assert!(SurvBuf::builder(vec![1.0], vec![2]).build().is_err());
    assert!(
        SurvBuf::builder(vec![1.0, 2.0], vec![0, 1])
            .start(vec![0.0, 2.0])
            .build()
            .is_err()
    );
}

#[test]
fn the_buffer_hands_out_the_same_data_as_the_borrowed_builder() {
    let buf = SurvBuf::builder(vec![1.0, 2.0, 3.0, 4.0], vec![0, 2, 1, 0])
        .start(vec![0.0, 0.5, 0.5, 1.0])
        .n_codes(2)
        .weight(vec![1.0, 0.0, 2.0, 1.0])
        .stratum(vec![5, 3, 5, 3])
        .cluster(vec![0, 1, 1, 0])
        .id(vec![10, 20, 20, 10])
        .istate(vec![1, 1, 2, 0])
        .build()
        .unwrap();
    let data = buf.as_data();
    assert_eq!((data.n_rows(), data.n_used()), (4, 3));
    assert_eq!(data.stratum().unwrap(), &[1, 0, 1, 0]);
    assert_eq!(data.n_strata(), 2);
    assert_eq!(data.id().unwrap(), &[0, 1, 1, 0]);
    // A dense column is kept without a copy.
    assert_eq!(data.cluster().unwrap(), &[0, 1, 1, 0]);
    assert_eq!(data.n_clusters(), 2);
    assert_eq!(data.istate().unwrap(), &[1, 1, 2, 0]);
    let surv = data.followup().unwrap();
    assert_eq!(surv.start().unwrap(), &[0.0, 0.5, 0.5, 1.0]);
    assert_eq!(surv.n_codes(), 2);
    assert_eq!(data.weight().unwrap(), &[1.0, 0.0, 2.0, 1.0]);
}
