use super::*;

const MARCH1: LeapBirthday = LeapBirthday::March1;

fn one_person(days: &[i64], masks: &[u64]) -> (Vec<usize>, Vec<i64>, Vec<u64>) {
    (vec![0, days.len()], days.to_vec(), masks.to_vec())
}

fn run(
    days: &[i64],
    masks: &[u64],
    entry: i64,
    exit: i64,
    history: i64,
    rules: &[Rule],
) -> Vec<Outcome> {
    let (offsets, day, mask) = one_person(days, masks);
    let rows = EventRows {
        person_offsets: &offsets,
        day: &day,
        mask: &mask,
        contact: None,
    };
    let windows = PersonWindows {
        entry: &[entry],
        exit: &[exit],
        history_start: &[history],
    };
    evaluate(&rows, &windows, rules)
        .unwrap()
        .into_iter()
        .map(|per_person| per_person[0])
        .collect()
}

#[test]
fn first_event_in_window_includes_entry_and_exit_days() {
    let rule = Rule::single(Lookback::None);
    let o = &run(&[5, 10, 20, 30], &[1; 4], 10, 20, 0, &[rule])[0];
    assert_eq!(o.first_in_window, Some(10));
    assert_eq!(o.n_events_in_window, 2);
    assert_eq!(o.prior, PriorStatus::NotAssessed);
    let o = &run(&[21], &[1], 10, 20, 0, &[rule])[0];
    assert_eq!(o.first_in_window, None);
    assert_eq!(o.case_day, None);
}

#[test]
fn an_event_on_the_entry_day_is_prior_or_in_window_never_both() {
    let inclusive = Rule {
        entry_inclusive: true,
        ..Rule::single(Lookback::Days(40))
    };
    let exclusive = Rule::single(Lookback::Days(40));
    let o = run(&[50], &[0b11], 50, 80, 0, &[inclusive, exclusive]);
    assert_eq!(
        (o[0].prior, o[0].first_in_window),
        (PriorStatus::Prevalent, None)
    );
    assert_eq!(
        (o[1].prior, o[1].first_in_window),
        (PriorStatus::NoneObserved, Some(50))
    );
}

#[test]
fn prior_status_separates_none_observed_from_too_short_history() {
    let rule = Rule::single(Lookback::Days(30));
    // Window [70, 100) inside the history.
    assert_eq!(
        run(&[], &[], 100, 200, 0, &[rule])[0].prior,
        PriorStatus::NoneObserved
    );
    // Window starts before the history.
    assert_eq!(
        run(&[], &[], 100, 200, 80, &[rule])[0].prior,
        PriorStatus::HistoryTooShort
    );
    // An event found settles it either way.
    assert_eq!(
        run(&[85], &[1], 100, 200, 80, &[rule])[0].prior,
        PriorStatus::Prevalent
    );
    // The window start is inclusive: day 70 is in, day 69 is not.
    assert_eq!(
        run(&[70], &[1], 100, 200, 0, &[rule])[0].prior,
        PriorStatus::Prevalent
    );
    assert_eq!(
        run(&[69], &[1], 100, 200, 0, &[rule])[0].prior,
        PriorStatus::NoneObserved
    );
    let unlimited = Rule::single(Lookback::Unlimited);
    assert_eq!(
        run(&[], &[], 100, 200, 80, &[unlimited])[0].prior,
        PriorStatus::NoneObserved
    );
    assert_eq!(
        run(&[81], &[1], 100, 200, 80, &[unlimited])[0].prior,
        PriorStatus::Prevalent
    );
}

#[test]
fn events_before_the_history_are_counted_and_not_used() {
    let rule = Rule::single(Lookback::Unlimited);
    let o = &run(&[10, 20], &[1, 1], 100, 200, 50, &[rule])[0];
    assert_eq!(o.n_before_history, 2);
    assert_eq!(o.prior, PriorStatus::NoneObserved);
    // An entry before the history start: the window starts at the history.
    let o = &run(
        &[10, 60],
        &[1, 1],
        20,
        200,
        50,
        &[Rule::single(Lookback::None)],
    )[0];
    assert_eq!(o.first_in_window, Some(60));
    assert_eq!(o.n_before_history, 1);
}

#[test]
fn look_back_in_years_follows_the_leap_rule() {
    let entry = days_from_civil(2024, 2, 29);
    let by = |y, leap| {
        Lookback::Years(y, leap)
            .window_start(entry, 0)
            .unwrap()
            .unwrap()
    };
    assert_eq!(by(4, MARCH1), days_from_civil(2020, 2, 29));
    assert_eq!(by(1, MARCH1), days_from_civil(2023, 3, 1));
    assert_eq!(
        by(1, LeapBirthday::February28),
        days_from_civil(2023, 2, 28)
    );
    assert_eq!(by(5, MARCH1), days_from_civil(2019, 3, 1));
    let ordinary = days_from_civil(2024, 6, 15);
    assert_eq!(
        Lookback::Years(5, MARCH1)
            .window_start(ordinary, 0)
            .unwrap(),
        Some(days_from_civil(2019, 6, 15))
    );
}

#[test]
fn confirmation_by_a_second_date_within_the_limit() {
    let rule = |within, date| Rule {
        require: 2,
        confirm_within: within,
        date_of_case: date,
        ..Rule::single(Lookback::None)
    };
    let days = [10, 10, 20, 50, 51];
    let mask = [1; 5];
    let first = |w| run(&days, &mask, 0, 100, 0, &[rule(w, CaseDate::FirstEvent)])[0].case_day;
    let confirming = |w| {
        run(
            &days,
            &mask,
            0,
            100,
            0,
            &[rule(w, CaseDate::ConfirmingEvent)],
        )[0]
        .case_day
    };
    // Two events on one day do not confirm each other by date.
    assert_eq!(first(Some(9)), Some(50));
    assert_eq!(first(Some(10)), Some(10));
    assert_eq!(confirming(Some(10)), Some(20));
    assert_eq!(confirming(Some(1)), Some(51));
    assert_eq!(first(Some(0)), None);
    assert_eq!(first(None), Some(10));
    // One event is never confirmed.
    assert_eq!(
        run(&[10], &[1], 0, 100, 0, &[rule(None, CaseDate::FirstEvent)])[0].case_day,
        None
    );
    // The confirming event must be inside the follow-up window.
    assert_eq!(
        run(
            &[10, 101],
            &[1, 1],
            0,
            100,
            0,
            &[rule(None, CaseDate::FirstEvent)]
        )[0]
        .case_day,
        None
    );
}

#[test]
fn confirmation_by_contacts_accepts_two_contacts_on_one_day() {
    let offsets = [0, 3];
    let day = [10, 10, 40];
    let mask = [1_u64; 3];
    let contact = [7_u32, 7, 7];
    let rule = Rule {
        require: 2,
        confirm_by: ConfirmBy::Contacts,
        confirm_within: Some(5),
        ..Rule::single(Lookback::None)
    };
    let windows = PersonWindows {
        entry: &[0],
        exit: &[100],
        history_start: &[0],
    };
    let eval = |contact: &[u32]| {
        let rows = EventRows {
            person_offsets: &offsets,
            day: &day,
            mask: &mask,
            contact: Some(contact),
        };
        evaluate(&rows, &windows, &[rule]).unwrap()[0][0].case_day
    };
    assert_eq!(eval(&contact), None);
    assert_eq!(eval(&[7, 8, 7]), Some(10));
    assert_eq!(eval(&[7, 7, 8]), None);
    let rows = EventRows {
        person_offsets: &offsets,
        day: &day,
        mask: &mask,
        contact: None,
    };
    assert!(evaluate(&rows, &windows, &[rule]).is_err());
}

#[test]
fn many_definitions_share_one_pass() {
    let rules = [
        Rule::single(Lookback::Days(10)),
        Rule::single(Lookback::Days(10)),
    ];
    let o = run(&[95, 100, 120], &[0b01, 0b10, 0b10], 100, 150, 0, &rules);
    assert_eq!(
        (o[0].prior, o[0].first_in_window),
        (PriorStatus::Prevalent, None)
    );
    assert_eq!(
        (o[1].prior, o[1].first_in_window),
        (PriorStatus::NoneObserved, Some(100))
    );
    assert_eq!(o[1].n_events_in_window, 2);
}

#[test]
fn refusals() {
    let rule = Rule::single(Lookback::None);
    let rows = EventRows {
        person_offsets: &[0, 2],
        day: &[5, 3],
        mask: &[1, 1],
        contact: None,
    };
    let windows = PersonWindows {
        entry: &[0],
        exit: &[10],
        history_start: &[0],
    };
    assert!(evaluate(&rows, &windows, &[rule]).is_err(), "unsorted");
    let rows = EventRows {
        day: &[3, 5],
        ..rows
    };
    assert!(evaluate(&rows, &windows, &[]).is_err());
    assert!(evaluate(&rows, &windows, &[Rule { require: 3, ..rule }]).is_err());
    assert!(evaluate(&rows, &windows, &[Rule::single(Lookback::Days(-1))]).is_err());
    assert!(
        evaluate(
            &rows,
            &windows,
            &[Rule {
                confirm_within: Some(-1),
                ..rule
            }]
        )
        .is_err()
    );
    let late = PersonWindows {
        entry: &[11],
        ..windows
    };
    assert!(evaluate(&rows, &late, &[rule]).is_err());
    let short = PersonWindows {
        exit: &[],
        ..windows
    };
    assert!(evaluate(&rows, &short, &[rule]).is_err());
    let bad = EventRows {
        person_offsets: &[0, 1],
        ..rows
    };
    assert!(evaluate(&bad, &windows, &[rule]).is_err());
}
