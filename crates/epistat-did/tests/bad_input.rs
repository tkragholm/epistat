//! The data views every estimator takes, and the cluster functions that take
//! an influence function directly, meet bad input with an error of kind input
//! and the code of the shared check, never a panic.
//!
//! The estimators take only checked views, so the views are where bad columns
//! are refused. The first column of a view fixes the number of rows: an empty
//! view has the code of an empty argument, and an empty column beside it the
//! code of a length error.

use epistat_did::continuous::ContinuousData;
use epistat_did::triple_did::TripleDidData;
use epistat_did::{StaggeredPanel, SwitchingPanel, did_standard_error};
use epistat_inference::Clusters;
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

const UNIT: [u32; 6] = [0, 0, 1, 1, 2, 2];
const PERIOD: [i64; 6] = [1, 2, 1, 2, 1, 2];
const FIRST: [Option<i64>; 6] = [Some(2), Some(2), None, None, None, None];
const Y: [f64; 6] = [1.0, 2.5, 0.5, 0.75, 1.5, 1.25];
const W: [f64; 6] = [1.0, 1.0, 2.0, 2.0, 0.5, 0.5];

/// The code of each kind of bad column beside a first column of fixed rows.
fn beside(bad: BadInput) -> &'static str {
    match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    }
}

/// The kinds a column of numbers can carry; order does not matter to it.
fn column(bad: BadInput, good: &[f64]) -> Option<Vec<f64>> {
    (bad != BadInput::Unsorted).then(|| bad.column(good))
}

#[test]
fn the_staggered_panel_refuses_bad_columns() {
    assert_rejects_bad_input_with_codes("StaggeredPanel key", BadInput::code, |bad| {
        let reversed: Vec<i64> = PERIOD.iter().rev().copied().collect();
        Some(
            match bad {
                BadInput::Empty => StaggeredPanel::builder(&[], &[], &[], &[]).build(),
                BadInput::Unsorted => {
                    StaggeredPanel::builder(&UNIT, &reversed, &FIRST, &Y).build()
                }
                BadInput::Mismatched => {
                    StaggeredPanel::builder(&UNIT, &PERIOD[..5], &FIRST, &Y).build()
                }
                BadInput::Nan | BadInput::Infinite => return None,
            }
            .map(|panel| panel.len()),
        )
    });
    assert_rejects_bad_input_with_codes("StaggeredPanel outcome", beside, |bad| {
        column(bad, &Y).map(|y| {
            StaggeredPanel::builder(&UNIT, &PERIOD, &FIRST, &y)
                .build()
                .map(|panel| panel.len())
        })
    });
    assert_rejects_bad_input_with_codes("StaggeredPanel weights", beside, |bad| {
        column(bad, &W).map(|w| {
            StaggeredPanel::builder(&UNIT, &PERIOD, &FIRST, &Y)
                .weights(epistat_core::CaseWeights::new(&w).expect("weights"))
                .build()
                .map(|panel| panel.len())
        })
    });
}

#[test]
fn the_switching_panel_refuses_bad_columns() {
    let outcome: Vec<Option<f64>> = Y.iter().copied().map(Some).collect();
    let treatment = [Some(0.0), Some(1.0), Some(0.0), Some(0.0), Some(0.0), Some(1.0)];
    assert_rejects_bad_input_with_codes("SwitchingPanel key", BadInput::code, |bad| {
        let reversed: Vec<i64> = PERIOD.iter().rev().copied().collect();
        Some(
            match bad {
                BadInput::Empty => SwitchingPanel::builder(&[], &[], &[], &[]).build(),
                BadInput::Unsorted => {
                    SwitchingPanel::builder(&UNIT, &reversed, &outcome, &treatment).build()
                }
                BadInput::Mismatched => {
                    SwitchingPanel::builder(&UNIT, &PERIOD[..5], &outcome, &treatment).build()
                }
                BadInput::Nan | BadInput::Infinite => return None,
            }
            .map(|panel| panel.len()),
        )
    });
    assert_rejects_bad_input_with_codes("SwitchingPanel weights", beside, |bad| {
        column(bad, &W).map(|w| {
            SwitchingPanel::builder(&UNIT, &PERIOD, &outcome, &treatment)
                .weights(epistat_core::CaseWeights::new(&w).expect("weights"))
                .build()
                .map(|panel| panel.len())
        })
    });
}

#[test]
fn the_unit_level_views_refuse_bad_columns() {
    let group = [true, true, false, false, true, false];
    let partition = [true, false, true, false, false, true];
    let change = [0.5, -0.25, 1.0, 0.0, 0.75, -1.0];
    let dose = [0.0, 0.5, 1.0, 1.5, 0.0, 2.0];
    assert_rejects_bad_input_with_codes("TripleDidData change", beside, |bad| {
        column(bad, &change).map(|c| {
            TripleDidData::builder(&group, &partition, &c)
                .build()
                .map(|data| data.len())
        })
    });
    assert_rejects_bad_input_with_codes("TripleDidData weights", beside, |bad| {
        column(bad, &W).map(|w| {
            TripleDidData::builder(&group, &partition, &change)
                .weights(epistat_core::CaseWeights::new(&w).expect("weights"))
                .build()
                .map(|data| data.len())
        })
    });
    assert_rejects_bad_input_with_codes("ContinuousData dose", BadInput::code, |bad| {
        column(bad, &dose).map(|d| {
            ContinuousData::builder(&d, &change)
                .build()
                .map(|data| data.len())
        })
    });
    assert_rejects_bad_input_with_codes("ContinuousData change", beside, |bad| {
        column(bad, &change).map(|c| {
            ContinuousData::builder(&dose, &c)
                .build()
                .map(|data| data.len())
        })
    });
    assert_rejects_bad_input_with_codes("ContinuousData weights", beside, |bad| {
        column(bad, &W).map(|w| {
            ContinuousData::builder(&dose, &change)
                .weights(epistat_core::CaseWeights::new(&w).expect("weights"))
                .build()
                .map(|data| data.len())
        })
    });
}

#[test]
fn the_standard_error_refuses_a_bad_influence_function() {
    let psi = [0.5, -1.25, 2.0, 0.75, -2.0, 0.0];
    let clusters = Clusters::from_codes(&[0, 0, 1, 1, 2, 2]);
    assert_rejects_bad_input_with_codes("did_standard_error", BadInput::code, |bad| {
        column(bad, &psi).map(|psi| did_standard_error(&psi, &clusters))
    });
}
