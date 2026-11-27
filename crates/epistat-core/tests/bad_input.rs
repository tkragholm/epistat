//! The checks of this crate refuse bad input with an `Input` error, and its
//! error codes are unique. Uses the helpers of `epistat-testkit`.

use epistat_core::{InputError, check};
use epistat_testkit::{BadInput, assert_codes_unique, assert_rejects_bad_input_with_codes};

#[test]
fn column_checks_reject_bad_columns() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [0.1, 0.2, 0.3, 0.4];
    assert_rejects_bad_input_with_codes("finite and sorted_by", BadInput::code, |bad| {
        Some(match bad {
            BadInput::Empty => check::non_empty("x", bad.column(&x).len()).map(|()| 0),
            BadInput::Mismatched => {
                check::same_length("y", x.len(), bad.column(&y).len()).map(|()| 0)
            }
            BadInput::Nan | BadInput::Infinite => check::finite("x", &bad.column(&x)).map(|()| 0),
            BadInput::Unsorted => {
                check::sorted_by("x", &bad.column(&x), f64::total_cmp).map(|()| 0)
            }
        })
    });
}

#[test]
fn the_codes_of_input_error_are_unique() {
    let errors = [
        InputError::Length {
            what: "a",
            expected: 1,
            found: 0,
        },
        InputError::Empty { what: "a" },
        InputError::NotFinite {
            what: "a",
            row: 0,
            column: None,
        },
        InputError::Domain {
            what: "a",
            value: 0.0,
            domain: "x",
            row: None,
        },
        InputError::Unsorted { what: "a", row: 1 },
        InputError::Duplicate { what: "a", row: 1 },
        InputError::NotConstant { what: "a", unit: 0 },
        InputError::TooLarge {
            what: "a",
            limit: 1,
            found: 2,
        },
        InputError::NotBelow {
            what: "a",
            value: 1.0,
            limit: 1.0,
        },
    ];
    assert_codes_unique("input", &errors);
}
