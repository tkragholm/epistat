//! The entry points that take floating-point columns meet bad input with an
//! error of kind input and the shared code of the check, never a panic, and
//! the error codes are unique.

use epistat_core::Classify;
use epistat_register::exposure::{
    Assumption, Dispensings, DoseRule, Periods, Stock, Supply, exposure_periods,
};
use epistat_register::lexis::{Cuts, split_at};
use epistat_register::pre2dup::{GlobalParameters, GroupParameters, Purchases, drug_use_periods};
use epistat_register::spans::merge_sorted_spans;
use epistat_register::wtd::{Family, WaitingTime, fit_wtd, waiting_times};
use epistat_register::{CodeItemProblem, CrosswalkProblem, Error, PedigreeProblem};
use epistat_testkit::{BadInput, assert_codes_unique, assert_rejects_bad_input_with_codes};

#[test]
fn the_waiting_time_fit_rejects_bad_times() {
    let times = [1.0, 2.0, 4.0, 7.0, 9.0];
    // An empty column has fewer than the three times the fit needs.
    let code = |bad: BadInput| match bad {
        BadInput::Empty => "register.too_few",
        other => other.code(),
    };
    assert_rejects_bad_input_with_codes("fit_wtd", code, |bad| match bad {
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
    // The quantity has one value per day, so an empty one is of the wrong
    // length.
    let code = |bad: BadInput| match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    };
    assert_rejects_bad_input_with_codes("exposure_periods", code, |bad| match bad {
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
fn drug_use_periods_reject_bad_packages() {
    let offsets = [0, 2, 4];
    let group = [Some(0); 4];
    let day = [0, 20, 5, 30];
    let ddd = [Some(30.0); 4];
    let packages = [1.0, 1.0, 2.0, 1.0];
    let refill = [None; 4];
    let groups = [GroupParameters {
        lower_ddd_per_day: 0.5,
        common_ddd_per_day: 1.0,
        upper_ddd_per_day: 2.0,
        min_period_days: 14.0,
        max_refill_days: 90.0,
    }];
    let code = |bad: BadInput| match bad {
        BadInput::Empty => "input.length",
        other => other.code(),
    };
    assert_rejects_bad_input_with_codes("drug_use_periods", code, |bad| match bad {
        BadInput::Empty | BadInput::Nan | BadInput::Infinite | BadInput::Mismatched => {
            let packages = bad.column(&packages);
            Some(drug_use_periods(
                &Purchases {
                    person_offsets: &offsets,
                    group: &group,
                    day: &day,
                    ddd: &ddd,
                    packages: &packages,
                    typical_refill_days: &refill,
                },
                &groups,
                &GlobalParameters::TANSKANEN_2015,
                None,
            ))
        }
        BadInput::Unsorted => None,
    });
}

#[test]
fn a_missing_period_has_no_end_day() {
    let periods = Periods::default();
    assert_eq!(periods.end_day(0), None);
    assert_eq!(periods.last_day(0), None);
}

#[test]
fn the_codes_are_unique() {
    let errors = [
        Error::EndBeforeStart { what: "x", row: 0 },
        Error::OutOfRange {
            what: "x",
            row: Some(0),
            value: 1,
            limit: 1,
        },
        Error::FirstRowBeginsNoKey { what: "x" },
        Error::DayOverflow { what: "x", row: 0 },
        Error::CodeItem {
            item: String::new(),
            problem: CodeItemProblem::Empty,
        },
        Error::DuplicateName {
            what: "x",
            name: String::new(),
        },
        Error::UnknownName {
            what: "x",
            name: String::new(),
        },
        Error::Supersedes {
            key: String::new(),
            target: String::new(),
        },
        Error::SharedCode {
            what: "x",
            first: String::new(),
            second: String::new(),
        },
        Error::NoItems { set: String::new() },
        Error::Crosswalk {
            from: String::new(),
            to: String::new(),
            problem: CrosswalkProblem::Repeated,
        },
        Error::Pedigree {
            person: 0,
            problem: PedigreeProblem::SameParents,
        },
        Error::GroupTooLarge {
            what: "x",
            id: 0,
            size: 2,
            limit: 1,
        },
        Error::Required { what: "x", by: "y" },
        Error::TooFew {
            what: "x",
            needed: 3,
            found: 0,
        },
        Error::MissingReferenceYear(1900),
        Error::MissingYears(vec![1900]),
    ];
    assert_codes_unique("register", &errors);
    assert_eq!(Error::MissingYears(vec![]).code(), "register.missing_years");
}

/// A column of days spoiled as `bad` spoils a column; NaN and infinities have
/// no counterpart among days.
fn days(bad: BadInput, good: &[i64]) -> Option<Vec<i64>> {
    let mut column = good.to_vec();
    match bad {
        BadInput::Empty => column.clear(),
        BadInput::Mismatched => {
            column.pop();
        }
        BadInput::Unsorted => column.reverse(),
        BadInput::Nan | BadInput::Infinite => return None,
    }
    Some(column)
}

#[test]
fn the_day_columns_refuse_a_wrong_length() {
    let new_key = [true, false, false, true];
    let start = [0_i64, 10, 20, 5];
    let end = [8_i64, 15, 30, 9];
    let length = |_: BadInput| "input.length";
    assert_rejects_bad_input_with_codes("merge_sorted_spans", length, |bad| {
        matches!(bad, BadInput::Empty | BadInput::Mismatched).then(|| {
            let end = days(bad, &end).expect("a day column");
            merge_sorted_spans(&new_key, &start, &end, 0)
        })
    });
    assert_rejects_bad_input_with_codes("split_at", length, |bad| {
        matches!(bad, BadInput::Empty | BadInput::Mismatched).then(|| {
            let end = days(bad, &end).expect("a day column");
            split_at(&start, &end, &[Cuts::CalendarYears])
        })
    });
}

#[test]
fn waiting_times_refuse_bad_offsets_and_days() {
    let offsets = [0_usize, 2, 4];
    let day = [3_i64, 40, 10, 200];
    assert_rejects_bad_input_with_codes("waiting_times", BadInput::code, |bad| {
        let day = days(bad, &day)?;
        // The days of a person need no order: the first and last in the
        // window are found by scanning.
        matches!(bad, BadInput::Mismatched)
            .then(|| waiting_times(&offsets, &day, 0, 365, WaitingTime::Forward))
    });
}
