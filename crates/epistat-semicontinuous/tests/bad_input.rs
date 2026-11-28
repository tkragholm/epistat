//! The data views every fit takes meet bad input with an error of kind input
//! and the code of the shared check, never a panic.
//!
//! The fits take only checked views, so the views are where bad columns are
//! refused. The view's first argument fixes the number of rows: an empty
//! design or outcome there has the code of an empty argument, and any other
//! empty column the code of a length error. Order does not matter to a cost
//! model, so `Unsorted` is not tried.

use epistat_core::{CaseWeights, Seed};
use epistat_linalg::Mat;
use epistat_semicontinuous::compare::{CrossValidationOptions, cross_validate, select};
use epistat_semicontinuous::{CostData, Error, LongitudinalData};
use epistat_testkit::{BadInput, assert_rejects_bad_input_with_codes};

const Y: [f64; 6] = [0.0, 1.5, 2.0, 0.0, 3.5, 1.0];
const WEIGHTS: [f64; 6] = [1.0, 0.5, 2.0, 1.0, 1.0, 0.5];
const TIME: [f64; 6] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
const SUBJECT: [u32; 6] = [0, 0, 1, 1, 2, 2];

fn design() -> Mat<f64> {
    Mat::from_fn(6, 2, |i, j| if j == 0 { 1.0 } else { i as f64 / 5.0 })
}

/// The code of each kind of bad column beside a design of fixed rows.
fn code(bad: BadInput) -> &'static str {
    match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    }
}

/// The code of each kind of bad argument that fixes the number of rows.
fn first(bad: BadInput) -> &'static str {
    bad.code()
}

/// The kinds a column of numbers can carry.
fn column(bad: BadInput, good: &[f64]) -> Option<Vec<f64>> {
    (bad != BadInput::Unsorted).then(|| bad.column(good))
}

#[test]
fn cost_data_refuses_bad_columns() {
    let x = design();
    assert_rejects_bad_input_with_codes("CostData outcome", code, |bad| {
        column(bad, &Y).map(|y| CostData::builder(x.as_ref(), &y).build().map(|_| ()))
    });
    // The weights refuse a value that is not finite when they are built,
    // and the view a column of the wrong length.
    assert_rejects_bad_input_with_codes("CostData weights", code, |bad| {
        column(bad, &WEIGHTS).map(|w| -> Result<(), Error> {
            let weights = CaseWeights::new(&w)?;
            CostData::builder(x.as_ref(), &Y)
                .weights(weights)
                .build()
                .map(|_| ())
        })
    });
}

#[test]
fn cost_data_refuses_a_bad_design() {
    assert_rejects_bad_input_with_codes("CostData design", first, |bad| {
        let x = match bad {
            BadInput::Nan | BadInput::Infinite => {
                let value = bad.column(&[0.5])[0];
                Mat::from_fn(6, 2, |i, j| if (i, j) == (3, 1) { value } else { 1.0 })
            }
            BadInput::Empty | BadInput::Mismatched => {
                let rows = if bad == BadInput::Empty { 0 } else { 5 };
                Mat::from_fn(rows, 2, |_, _| 1.0)
            }
            BadInput::Unsorted => return None,
        };
        Some(CostData::builder(x.as_ref(), &Y).build().map(|_| ()))
    });
}

#[test]
fn longitudinal_data_refuses_bad_columns() {
    let x = design();
    assert_rejects_bad_input_with_codes("LongitudinalData outcome", first, |bad| {
        column(bad, &Y).map(|y| {
            LongitudinalData::builder(&y, x.as_ref(), x.as_ref(), &SUBJECT, &TIME)
                .build()
                .map(|_| ())
        })
    });
    assert_rejects_bad_input_with_codes("LongitudinalData time", code, |bad| {
        column(bad, &TIME).map(|time| {
            LongitudinalData::builder(&Y, x.as_ref(), x.as_ref(), &SUBJECT, &time)
                .build()
                .map(|_| ())
        })
    });
}

#[test]
fn the_power_selection_refuses_bad_powers() {
    let x = design();
    let data = CostData::builder(x.as_ref(), &Y).build().unwrap();
    let powers = [1.2, 1.5, 1.8];
    // No powers is a selection among the other models, and the powers have no
    // length to match.
    let powered = |bad: BadInput| matches!(bad, BadInput::Nan | BadInput::Infinite);
    assert_rejects_bad_input_with_codes("select", first, |bad| {
        column(bad, &powers)
            .filter(|_| powered(bad))
            .map(|p| select(&data, &p).map(|_| ()))
    });
    assert_rejects_bad_input_with_codes("cross_validate", first, |bad| {
        column(bad, &powers)
            .filter(|_| powered(bad))
            .map(|p| cross_validate(&data, &p, &CrossValidationOptions::new(Seed(1))).map(|_| ()))
    });
}
