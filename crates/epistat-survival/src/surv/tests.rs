use super::{Outcome, SurvBuf, SurvData};
use crate::check::check_row_count;
use crate::timefix::TimeFix;

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

fn is_borrowed(data: &SurvData<'_>) -> bool {
    let surv = data.followup().unwrap();
    matches!(surv.stop, std::borrow::Cow::Borrowed(_))
        && surv
            .start
            .as_ref()
            .is_none_or(|start| matches!(start, std::borrow::Cow::Borrowed(_)))
}

const NEAR: f64 = 1e-12;

#[test]
fn a_new_data_set_has_had_no_tie_policy_applied() {
    let (stop, status) = simple();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    assert_eq!(data.timefix().unwrap(), TimeFix::Exact);
}

#[test]
fn timefix_replaces_only_the_times_and_only_when_a_tie_is_found() {
    let stop = [1.0, 1.0 + NEAR, 5.0, 7.0];
    let start = [0.0, 0.0, 1.0, 5.0 + NEAR];
    let status = [1, 0, 1, 0];
    let weight = [1.0, 2.0, 0.0, 1.0];
    let stratum = [3_u32, 1, 3, 1];
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .weight(&weight)
        .stratum(&stratum)
        .build()
        .unwrap();
    let fixed = data.with_timefix(TimeFix::default()).unwrap();
    let surv = fixed.followup().unwrap();
    assert_eq!(surv.stop(), &[1.0, 1.0, 5.0, 7.0]);
    assert_eq!(surv.start().unwrap(), &[0.0, 0.0, 1.0, 5.0]);
    assert!(!is_borrowed(&fixed));
    assert_eq!(surv.status().as_ptr(), status.as_ptr());
    assert_eq!(fixed.weight().unwrap().as_ptr(), weight.as_ptr());
    assert_eq!(fixed.stratum().unwrap(), &[1, 0, 1, 0]);
    assert_eq!(fixed.timefix().unwrap(), TimeFix::default());
    assert_eq!((fixed.n_rows(), fixed.n_used()), (4, 3));
}

#[test]
fn timefix_without_a_tie_keeps_the_times_borrowed_and_records_the_policy() {
    let stop = [1.0, 2.5, 3.5];
    let status = [0, 1, 0];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let fixed = data.with_timefix(TimeFix::default()).unwrap();
    assert!(is_borrowed(&fixed));
    assert_eq!(fixed.followup().unwrap().stop().as_ptr(), stop.as_ptr());
    assert_eq!(fixed.timefix().unwrap(), TimeFix::default());
}

#[test]
fn integer_day_numbers_pass_through_borrowed() {
    let stop = [100.0, 101.0, 20_000.0];
    let status = [1, 0, 1];
    let fixed = SurvData::builder(&stop, &status)
        .build()
        .unwrap()
        .with_timefix(TimeFix::default())
        .unwrap();
    assert!(is_borrowed(&fixed));
}

#[test]
fn exact_and_a_zero_tolerance_change_nothing_and_are_not_recorded() {
    let stop = [1.0, 1.0 + NEAR];
    let status = [1, 0];
    for policy in [TimeFix::Exact, TimeFix::Aeq { tolerance: 0.0 }] {
        let fixed = SurvData::builder(&stop, &status)
            .build()
            .unwrap()
            .with_timefix(policy)
            .unwrap();
        assert!(is_borrowed(&fixed));
        assert_eq!(fixed.followup().unwrap().stop(), &stop[..]);
        assert_eq!(fixed.timefix().unwrap(), TimeFix::Exact);
    }
}

#[test]
fn exact_does_not_undo_an_applied_policy() {
    let stop = [1.0, 1.0 + NEAR];
    let status = [1, 0];
    let fixed = SurvData::builder(&stop, &status)
        .build()
        .unwrap()
        .with_timefix(TimeFix::default())
        .unwrap()
        .with_timefix(TimeFix::Exact)
        .unwrap();
    assert_eq!(fixed.followup().unwrap().stop(), &[1.0, 1.0]);
    assert_eq!(fixed.timefix().unwrap(), TimeFix::default());
}

#[test]
fn applying_the_same_policy_twice_changes_nothing() {
    let stop = [1.0, 1.0 + NEAR, 5.0];
    let status = [1, 0, 1];
    let once = SurvData::builder(&stop, &status)
        .build()
        .unwrap()
        .with_timefix(TimeFix::default())
        .unwrap();
    let before = once.followup().unwrap().stop().to_vec();
    let twice = once.with_timefix(TimeFix::default()).unwrap();
    assert_eq!(twice.followup().unwrap().stop(), &before[..]);
    assert_eq!(twice.timefix().unwrap(), TimeFix::default());
}

#[test]
fn a_different_policy_is_applied_to_the_times_as_they_stand() {
    let stop = [1.0, 1.0 + 1e-3, 5.0];
    let status = [1, 0, 1];
    let first = SurvData::builder(&stop, &status)
        .build()
        .unwrap()
        .with_timefix(TimeFix::default())
        .unwrap();
    assert_eq!(first.followup().unwrap().stop(), &stop[..]);
    let loose = TimeFix::Aeq { tolerance: 1e-2 };
    let second = first.with_timefix(loose).unwrap();
    assert_eq!(second.followup().unwrap().stop(), &[1.0, 1.0, 5.0]);
    assert_eq!(second.timefix().unwrap(), loose);
}

#[test]
fn timefix_refuses_what_fix_times_refuses() {
    let (stop, status) = simple();
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        let data = SurvData::builder(&stop, &status).build().unwrap();
        assert!(
            message(data.with_timefix(TimeFix::Aeq { tolerance })).contains("tolerance"),
            "{tolerance}"
        );
    }
    let stop = [1.0 + NEAR, 2.0];
    let status = [1, 0];
    let start = [1.0, 0.0];
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .build()
        .unwrap();
    assert!(message(data.with_timefix(TimeFix::default())).contains("effective length 0"));
}

#[test]
fn timefix_works_on_data_from_a_buffer_and_leaves_the_buffer_alone() {
    let buf = SurvBuf::builder(vec![1.0, 1.0 + NEAR, 3.0], vec![1, 0, 1])
        .stratum(vec![9, 4, 9])
        .build()
        .unwrap();
    let fixed = buf.as_data().with_timefix(TimeFix::default()).unwrap();
    assert_eq!(fixed.followup().unwrap().stop(), &[1.0, 1.0, 3.0]);
    assert_eq!(fixed.stratum_levels().unwrap().to_vec(), [4, 9]);
    assert_eq!(
        buf.as_data().followup().unwrap().stop()[1].to_bits(),
        (1.0 + NEAR).to_bits()
    );
}

#[test]
fn already_dense_columns_have_no_level_map() {
    let (stop, status) = simple();
    let data = SurvData::builder(&stop, &status)
        .stratum(&[1, 0, 1])
        .build()
        .unwrap();
    let levels = data.stratum_levels().unwrap();
    assert!(levels.is_identity());
    assert_eq!((levels.len(), levels.is_empty()), (2, false));
    assert_eq!(levels.code(1), Some(1));
    assert_eq!(levels.code(2), None);
    assert_eq!(levels.to_vec(), [0, 1]);
    assert!(data.cluster_levels().is_none() && data.id_levels().is_none());
}

#[test]
fn recoded_columns_map_each_level_back_to_the_callers_code() {
    let stop = [1.0; 5];
    let status = [0; 5];
    // A maximum below the row count is recoded through a table.
    let data = SurvData::builder(&stop, &status)
        .cluster(&[4, 0, 4, 2, 0])
        .build()
        .unwrap();
    let levels = data.cluster_levels().unwrap();
    assert!(!levels.is_identity());
    assert_eq!(levels.to_vec(), [0, 2, 4]);
    assert_eq!(levels.code(2), Some(4));
    assert_eq!(levels.code(3), None);
    // A maximum above it is recoded through sorting.
    let data = SurvData::builder(&stop, &status)
        .id(&[4_000_000_000, 7, 4_000_000_000, 100, 7])
        .stratum(&[2, 3, 2, 3, 3])
        .build()
        .unwrap();
    assert_eq!(data.id_levels().unwrap().to_vec(), [7, 100, 4_000_000_000]);
    assert_eq!(data.stratum_levels().unwrap().to_vec(), [2, 3]);
    // Level k is the k-th smallest code, so the map and the dense codes agree.
    let id = data.id().unwrap();
    let labels = data.id_levels().unwrap();
    let original = [4_000_000_000, 7, 4_000_000_000, 100, 7];
    for (row, &level) in id.iter().enumerate() {
        assert_eq!(labels.code(level as usize), Some(original[row]));
    }
    let mut ascending = labels.iter().collect::<Vec<_>>();
    ascending.sort_unstable();
    assert_eq!(ascending, labels.to_vec());
    assert_eq!(labels.iter().len(), 3);
    assert_eq!(labels.iter().next_back(), Some(4_000_000_000));
}

#[test]
fn the_buffer_keeps_the_level_maps() {
    let buf = SurvBuf::builder(vec![1.0; 4], vec![0; 4])
        .stratum(vec![5, 3, 5, 3])
        .cluster(vec![0, 1, 1, 0])
        .id(vec![10, 20, 20, 10])
        .build()
        .unwrap();
    let data = buf.as_data();
    assert_eq!(data.stratum_levels().unwrap().to_vec(), [3, 5]);
    assert!(data.cluster_levels().unwrap().is_identity());
    assert_eq!(data.id_levels().unwrap().to_vec(), [10, 20]);
}

#[test]
fn used_rows_are_the_rows_of_positive_weight_in_both_directions() {
    let stop = [1.0; 5];
    let status = [0; 5];
    let weight = [1.0, 0.0, 2.0, -0.0, 0.5];
    let data = SurvData::builder(&stop, &status)
        .weight(&weight)
        .build()
        .unwrap();
    let used = data.used_rows();
    assert_eq!(used.len(), 3);
    assert_eq!(used.clone().collect::<Vec<_>>(), [0, 2, 4]);
    assert_eq!(used.rev().collect::<Vec<_>>(), [4, 2, 0]);
    assert_eq!(data.used_index(), Some(vec![0, 2, 4]));
    // Both ends can be consumed together.
    let mut both = data.used_rows();
    assert_eq!(
        (both.next(), both.next_back(), both.len()),
        (Some(0), Some(4), 1)
    );
    assert_eq!(
        (both.next(), both.next(), both.next_back()),
        (Some(2), None, None)
    );
}

#[test]
fn without_unused_rows_there_is_no_index() {
    let (stop, status) = simple();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    assert_eq!(data.used_rows().collect::<Vec<_>>(), [0, 1, 2]);
    assert_eq!(data.used_index(), None);
    let data = SurvData::builder(&stop, &status)
        .weight(&[1.0, 2.0, 3.0])
        .build()
        .unwrap();
    assert_eq!(data.used_index(), None);
    assert_eq!(data.used_rows().len(), 3);
}

#[test]
fn used_rows_handle_all_zero_weights_and_no_rows() {
    let (stop, status) = simple();
    let data = SurvData::builder(&stop, &status)
        .weight(&[0.0; 3])
        .build()
        .unwrap();
    assert_eq!(data.used_rows().len(), 0);
    assert_eq!(data.used_rows().next(), None);
    assert_eq!(data.used_index(), Some(vec![]));
    let empty = SurvData::builder(&[], &[]).build().unwrap();
    assert_eq!(empty.used_rows().next_back(), None);
    assert_eq!(empty.used_index(), None);
}

#[test]
fn a_cause_reads_one_code_as_the_event_without_copying_status() {
    let stop = [1.0; 5];
    let status = [0, 1, 2, 3, 2];
    let data = SurvData::builder(&stop, &status)
        .n_codes(3)
        .weight(&[1.0, 1.0, 0.0, 1.0, 2.0])
        .build()
        .unwrap();
    let cause = data.cause(2).unwrap();
    assert_eq!(cause.code(), 2);
    assert_eq!(cause.status.as_ptr(), status.as_ptr());
    assert_eq!((cause.len(), cause.is_empty()), (5, false));
    assert_eq!(
        cause.iter().collect::<Vec<_>>(),
        [false, false, true, false, true]
    );
    assert!(cause.is_event(2) && !cause.is_event(1));
    assert!(cause.is_competing(1) && cause.is_competing(3));
    assert!(!cause.is_competing(0) && !cause.is_competing(2));
    // The row of weight 0 is not counted.
    assert_eq!(cause.n_events(), 1);
    assert_eq!(data.cause(1).unwrap().n_events(), 1);
    assert_eq!(data.cause(3).unwrap().n_events(), 1);
}

#[test]
fn n_events_without_weights_counts_every_row() {
    let stop = [1.0; 4];
    let data = SurvData::builder(&stop, &[1, 1, 0, 1]).build().unwrap();
    assert_eq!(data.cause(1).unwrap().n_events(), 3);
}

#[test]
fn a_cause_of_zero_or_above_n_codes_is_refused() {
    let (stop, status) = simple();
    let data = SurvData::builder(&stop, &status)
        .n_codes(2)
        .build()
        .unwrap();
    assert!(message(data.cause(0)).contains("0 is censoring"));
    assert_eq!(
        message(data.cause(3)),
        "cause 3 is outside 1..=2; 0 is censoring"
    );
    data.cause(2).unwrap();
}
