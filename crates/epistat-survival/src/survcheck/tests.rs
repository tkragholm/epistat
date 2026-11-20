use super::{Policy, Transition, survcheck};
use crate::surv::SurvData;

fn rows(flag: &super::Flagged) -> Vec<u32> {
    flag.rows.clone()
}

#[test]
fn it_needs_an_id() {
    let data = SurvData::builder(&[1.0], &[1]).build().unwrap();
    let message = survcheck(&data).unwrap_err().to_string();
    assert!(message.contains("id column"), "{message}");
}

#[test]
fn clean_follow_up_has_no_flags_and_chains_the_state() {
    // Subject 0 goes 1 -> 2, subject 1 is censored in state 1.
    let (start, stop) = ([0.0, 3.0, 0.0], [3.0, 7.0, 5.0]);
    let (status, id, istate) = ([2, 0, 0], [0, 0, 1], [1, 2, 1]);
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(2)
        .id(&id)
        .istate(&istate)
        .build()
        .unwrap();
    let check = survcheck(&data).unwrap();
    assert!(check.is_clean());
    assert_eq!(check.istate, [1, 2, 1]);
    assert_eq!(
        check.transitions,
        [
            Transition {
                from: 1,
                to: 0,
                count: 1
            },
            Transition {
                from: 1,
                to: 2,
                count: 1
            },
            Transition {
                from: 2,
                to: 0,
                count: 1
            },
        ]
    );
    assert_eq!(check.n_transitions(), 1);
    assert_eq!(check.n_ids, 2);
}

#[test]
fn each_flag_is_found_on_its_own_row_in_input_order() {
    // One subject whose rows are given out of time order. In time order:
    // (0,2] event 1, (2,4] same state, (5,6] a gap, (6,8] in state 2 after
    // state 1 (teleport), (9,10] in state 2 with the state reached 1 (jump),
    // (9.5,11] an overlap.
    let start = [9.0, 0.0, 6.0, 2.0, 9.5, 5.0];
    let stop = [10.0, 2.0, 8.0, 4.0, 11.0, 6.0];
    let status = [0, 1, 0, 0, 0, 0];
    let istate = [2, 1, 2, 1, 1, 1];
    let id = [7, 7, 7, 7, 7, 7];
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(2)
        .id(&id)
        .istate(&istate)
        .build()
        .unwrap();
    let check = survcheck(&data).unwrap();
    assert_eq!(rows(&check.gap), [5]);
    assert_eq!(rows(&check.teleport), [2]);
    assert_eq!(rows(&check.jump), [0]);
    assert_eq!(rows(&check.overlap), [4]);
    assert_eq!(check.overlap.ids, [0]);
    assert_eq!(check.istate, [1, 1, 1, 1, 1, 1]);
}

#[test]
fn without_istate_a_start_column_gives_no_jumps_or_teleports() {
    let (start, stop, status, id) = ([0.0, 5.0, 4.0], [3.0, 8.0, 9.0], [2, 0, 0], [1, 1, 1]);
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(2)
        .id(&id)
        .build()
        .unwrap();
    let check = survcheck(&data).unwrap();
    // Order by stop: (0,3], (5,8], (4,9]. The third starts before 8.
    assert_eq!(rows(&check.gap), [1]);
    assert_eq!(rows(&check.overlap), [2]);
    assert_eq!(check.jump.count() + check.teleport.count(), 0);
    assert_eq!(check.istate, [0, 2, 2]);
}

#[test]
fn right_censored_rows_of_one_id_overlap_from_time_zero() {
    let (stop, status, id) = ([4.0, 9.0], [1, 0], [3, 3]);
    let data = SurvData::builder(&stop, &status).id(&id).build().unwrap();
    let check = survcheck(&data).unwrap();
    assert_eq!(rows(&check.overlap), [1]);
    assert_eq!(check.teleport.count(), 0);
    assert_eq!(check.istate, [0, 1]);
}

#[test]
fn right_censored_rows_after_an_event_overlap_rather_than_teleport() {
    // Without start every row begins at 0, which never equals a positive
    // stop, so a row after an event is an overlap and not a teleport.
    let (stop, status, id) = ([4.0, 9.0], [2, 0], [3, 3]);
    let data = SurvData::builder(&stop, &status)
        .n_codes(2)
        .id(&id)
        .build()
        .unwrap();
    let check = survcheck(&data).unwrap();
    assert_eq!(rows(&check.overlap), [1]);
    assert_eq!(check.teleport.count(), 0);
}

#[test]
fn the_row_order_of_equal_keys_is_the_input_order() {
    let (start, stop, status, id) = ([0.0, 0.0], [5.0, 5.0], [0, 0], [0, 0]);
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .build()
        .unwrap();
    assert_eq!(rows(&survcheck(&data).unwrap().overlap), [1]);
}

#[test]
fn policy_follows_the_number_of_codes_and_names_the_first_row() {
    let (start, stop, status, id) = ([0.0, 1.0], [2.0, 3.0], [0, 0], [4, 4]);
    let binary = SurvData::builder(&stop, &status)
        .start(&start)
        .id(&id)
        .build()
        .unwrap();
    assert_eq!(Policy::for_data(&binary), Policy::Report);
    let check = survcheck(&binary).unwrap();
    assert_eq!(check.overlap.count(), 1);
    check.enforce(Policy::for_data(&binary)).unwrap();

    let multi = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(2)
        .id(&id)
        .build()
        .unwrap();
    assert_eq!(Policy::for_data(&multi), Policy::Strict);
    let message = survcheck(&multi)
        .unwrap()
        .enforce(Policy::Strict)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("overlap") && message.contains("row 1") && message.contains("id code 0"),
        "{message}"
    );
}

#[test]
fn gaps_alone_are_allowed_under_the_strict_policy() {
    let (start, stop, status, id) = ([0.0, 5.0], [2.0, 6.0], [0, 0], [0, 0]);
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .n_codes(2)
        .id(&id)
        .build()
        .unwrap();
    let check = survcheck(&data).unwrap();
    assert_eq!(check.gap.count(), 1);
    check.enforce(Policy::Strict).unwrap();
}

#[test]
fn the_sparse_table_agrees_with_the_dense_one() {
    let (start, stop, status, id) = ([0.0, 2.0, 0.0], [2.0, 4.0, 3.0], [3, 0, 0], [0, 0, 1]);
    let istate = [1, 3, 1];
    let build = |n_codes: u32| {
        let data = SurvData::builder(&stop, &status)
            .start(&start)
            .n_codes(n_codes)
            .id(&id)
            .istate(&istate)
            .build()
            .unwrap();
        survcheck(&data).unwrap().transitions
    };
    assert_eq!(build(3), build(1 << 20));
}
