//! The event definitions checked against a direct implementation in R
//! (`tests/data/events_reference.R`): first event in a window, prior status,
//! confirmation, civil-year look-backs with both leap rules, and the `heaven`
//! look-back of `look.back * 365.25` days. The results are integers and
//! strings, so agreement is exact.

use epistat_register::LeapBirthday;
use epistat_register::events::{
    CaseDate, ConfirmBy, EventRows, Lookback, PersonWindows, PriorStatus, Rule, evaluate,
};
use epistat_testkit::{Fingerprint, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    first: Vec<Option<f64>>,
    case: Vec<Option<f64>>,
    prior: Vec<String>,
    n: Vec<u32>,
    before: Vec<u32>,
}

#[derive(Deserialize)]
struct Reference {
    entry: Vec<f64>,
    exit: Vec<f64>,
    history: Vec<f64>,
    person: Vec<u32>,
    day: Vec<f64>,
    mask: Vec<u64>,
    rules: Vec<Expected>,
}

/// The rules of the reference script, in order.
fn rules() -> Vec<Rule> {
    let base = |lookback, inclusive| Rule {
        entry_inclusive: inclusive,
        ..Rule::single(lookback)
    };
    let confirm = |within: Option<i64>, date, lookback, inclusive| Rule {
        require: 2,
        confirm_within: within,
        confirm_by: ConfirmBy::Dates,
        date_of_case: date,
        ..base(lookback, inclusive)
    };
    vec![
        base(Lookback::Days(365), false),
        base(Lookback::Years(5, LeapBirthday::March1), true),
        base(Lookback::Years(1, LeapBirthday::February28), false),
        base(Lookback::Unlimited, false),
        confirm(Some(30), CaseDate::FirstEvent, Lookback::None, false),
        // heaven: date >= index - 5 * 365.25 and date <= index, for whole days
        // floor(5 * 365.25) = 1826 days back, index day included.
        base(Lookback::Days(1826), true),
        confirm(Some(10), CaseDate::ConfirmingEvent, Lookback::None, false),
        confirm(None, CaseDate::FirstEvent, Lookback::Days(100), true),
    ]
}

#[expect(clippy::cast_possible_truncation, reason = "whole days from R")]
fn ints(values: &[f64]) -> Vec<i64> {
    values.iter().map(|&v| v as i64).collect()
}

fn status(name: &str) -> PriorStatus {
    match name {
        "NotAssessed" => PriorStatus::NotAssessed,
        "NoneObserved" => PriorStatus::NoneObserved,
        "Prevalent" => PriorStatus::Prevalent,
        "HistoryTooShort" => PriorStatus::HistoryTooShort,
        other => panic!("unknown status {other}"),
    }
}

fn run(r: &Reference) -> epistat_register::events::Outcomes {
    let persons = r.entry.len();
    let mut offsets = vec![0_usize; persons + 1];
    for &p in &r.person {
        offsets[p as usize + 1] += 1;
    }
    for p in 0..persons {
        offsets[p + 1] += offsets[p];
    }
    let day = ints(&r.day);
    let rows = EventRows {
        person_offsets: &offsets,
        day: &day,
        mask: &r.mask,
        contact: None,
    };
    let (entry, exit, history) = (ints(&r.entry), ints(&r.exit), ints(&r.history));
    let windows = PersonWindows {
        entry: &entry,
        exit: &exit,
        history_start: &history,
    };
    evaluate(&rows, &windows, &rules()).unwrap()
}

#[test]
fn every_definition_agrees_with_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/events_reference.json"
    ));
    let got = run(&r);
    let mut fp = Fingerprint::new();
    let mut seen = [0_usize; 4];
    for (k, (g, e)) in got.iter().zip(&r.rules).enumerate() {
        for (p, o) in g.iter().enumerate() {
            let label = format!("rule {k} person {p}");
            let day = |v: Option<f64>| v.map(|x| ints(&[x])[0]);
            assert_eq!(o.first_in_window, day(e.first[p]), "{label} first");
            assert_eq!(o.case_day, day(e.case[p]), "{label} case");
            assert_eq!(o.prior, status(&e.prior[p]), "{label} prior");
            assert_eq!(o.n_events_in_window, e.n[p], "{label} n");
            assert_eq!(o.n_before_history, e.before[p], "{label} before history");
            fp = fp
                .u64(o.first_in_window.map_or(u64::MAX, i64::cast_unsigned))
                .u64(o.case_day.map_or(u64::MAX, i64::cast_unsigned))
                .u64(o.prior as u64)
                .u64(u64::from(o.n_events_in_window));
            seen[0] += usize::from(o.prior == PriorStatus::Prevalent);
            seen[1] += usize::from(o.prior == PriorStatus::HistoryTooShort);
            seen[2] += usize::from(o.case_day.is_some() && o.case_day != o.first_in_window);
            seen[3] += usize::from(o.n_before_history > 0);
        }
    }
    // The sample reaches every branch, or agreement would say little.
    assert!(seen.iter().all(|&c| c > 20), "{seen:?}");
    assert_eq!(fp.finish(), 2_443_825_616_837_089_056, "golden hash");
}
