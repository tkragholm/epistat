use chrono::NaiveDate;
use epistat_matching::MatchJob;

fn main() {
    let records = vec![epistat_matching::RoleTransitionRecord::from_record(
        epistat_matching::BaseRecord::new(
            "a",
            NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date"),
        ),
        None,
    )];

    let _ = MatchJob::new_transition(&records, 0).with_age_limit(6).run();
}
