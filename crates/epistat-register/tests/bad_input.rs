//! The entry points that take floating-point columns meet bad input with an
//! error of kind input, never a panic, and the error codes are unique.

use epistat_core::Classify;
use epistat_register::exposure::{
    Assumption, Dispensings, DoseRule, Periods, Stock, Supply, exposure_periods,
};
use epistat_register::wtd::{Family, fit_wtd};
use epistat_register::{Error, MergedSpans};
use epistat_testkit::{BadInput, assert_codes_unique, assert_rejects_bad_input};

#[test]
fn the_waiting_time_fit_rejects_bad_times() {
    let times = [1.0, 2.0, 4.0, 7.0, 9.0];
    assert_rejects_bad_input("fit_wtd", |bad| match bad {
        BadInput::Empty | BadInput::Nan | BadInput::Infinite => {
            Some(fit_wtd(&bad.column(&times), 30.0, Family::LogNormal))
        }
        _ => None,
    });
}

#[test]
fn exposure_periods_reject_bad_quantities() {
    let offsets = [0, 2, 4];
    let group = [Some(0); 4];
    let day = [0, 20, 5, 30];
    let quantity = [30.0, 30.0, 60.0, 30.0];
    let assumptions = [Assumption {
        dose_default: 1.0,
        dose_min: 0.5,
        dose_max: 2.0,
        max_stock_days: 30.0,
        grace_days: 7.0,
    }];
    let supply = Supply {
        assumptions: &assumptions,
        rule: DoseRule::Default,
        stock: Stock::CarryForward,
        pauses: None,
    };
    assert_rejects_bad_input("exposure_periods", |bad| match bad {
        BadInput::Empty | BadInput::Nan | BadInput::Infinite | BadInput::Mismatched => {
            let quantity = bad.column(&quantity);
            Some(exposure_periods(
                &Dispensings {
                    person_offsets: &offsets,
                    group: &group,
                    day: &day,
                    quantity: &quantity,
                },
                &supply,
            ))
        }
        BadInput::Unsorted => None,
    });
}

#[test]
fn a_missing_period_has_no_end_day_and_bad_spans_are_refused() {
    let periods = Periods::default();
    assert_eq!(periods.end_day(0), None);
    assert_eq!(periods.last_day(0), None);
    let merged = MergedSpans {
        first_row: vec![0],
        last_row: vec![1],
    };
    assert!(merged.bounds(&[0], &[1]).is_err());
}

#[test]
fn the_codes_are_unique() {
    assert_codes_unique(
        "register",
        &[
            Error::Invalid(String::new()),
            Error::MissingReferenceYear(1900),
            Error::MissingYears(vec![1900]),
            Error::Numerical(String::new()),
        ],
    );
    assert_eq!(Error::MissingYears(vec![]).code(), "register.missing_years");
}
