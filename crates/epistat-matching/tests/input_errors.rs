//! Invalid input is an error with the kind `Input`, not a panic and not an
//! empty outcome: duplicate ids and events before birth in risk-set matching,
//! caliper windows that are negative or not finite, and the columns of the
//! estimators that take them. The codes of the crate's error are unique.

use chrono::NaiveDate;
use epistat_core::Estimand;
use epistat_core::{Classify, ErrorKind, InputError};
use epistat_matching::abadie_imbens::{AbadieImbensData, AbadieImbensSpec, abadie_imbens};
use epistat_matching::constraints::caliper_on_field;
use epistat_matching::{
    AgeLimitYears, BaseRecord, BirthDateWindowDays, Error, MatchJob, MatchRatio, MatchWeightMethod,
    PairDistanceMatrix, RoleTransitionRecord,
};
use epistat_testkit::{
    BadInput, assert_codes_unique, assert_input_error, assert_rejects_bad_input,
};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("a valid date")
}

fn member(id: &str, birth: NaiveDate, event: Option<NaiveDate>) -> RoleTransitionRecord {
    RoleTransitionRecord::from_record(BaseRecord::new(id, birth), event)
}

fn cohort() -> Vec<RoleTransitionRecord> {
    vec![
        member("case", date(2010, 1, 1), Some(date(2012, 1, 1))),
        member("a", date(2010, 1, 5), None),
        member("b", date(2010, 1, 9), None),
    ]
}

fn job(
    records: &[RoleTransitionRecord],
) -> Result<epistat_matching::MatchOutcome, epistat_matching::Error> {
    MatchJob::new_transition(records, 1)
        .with_ratio(MatchRatio::new(1).expect("ratio"))
        .with_birth_window(BirthDateWindowDays::new(30).expect("window"))
        .with_age_limit(AgeLimitYears::new(6).expect("age limit"))
        .run()
}

#[test]
fn a_valid_cohort_matches() {
    let outcome = job(&cohort()).expect("a valid cohort");
    assert_eq!(outcome.pairs.len(), 1);
}

#[test]
fn a_duplicate_id_is_an_error() {
    let mut records = cohort();
    records.push(member("a", date(2010, 1, 7), None));
    assert_input_error("duplicate id", || job(&records));
    assert_eq!(
        job(&records).unwrap_err(),
        Error::DuplicateId { id: "a".into() }
    );
}

#[test]
fn an_event_before_birth_is_an_error() {
    let mut records = cohort();
    records.push(member("early", date(2011, 1, 1), Some(date(2010, 6, 1))));
    assert_input_error("event before birth", || job(&records));
    assert_eq!(
        job(&records).unwrap_err(),
        Error::EventBeforeBirth { id: "early".into() }
    );
}

#[test]
fn a_caliper_window_must_be_finite_and_not_negative() {
    let records = cohort();
    for window in [f64::NAN, f64::INFINITY, -1.0] {
        let run = || {
            MatchJob::new_transition(&records, 1)
                .with_ratio(MatchRatio::new(1).expect("ratio"))
                .with_constraint(caliper_on_field("x", window))
                .run()
        };
        assert_input_error(&format!("caliper window {window}"), run);
        assert!(matches!(
            run().unwrap_err(),
            Error::Input(InputError::Domain {
                what: "caliper window",
                ..
            })
        ));
    }
    // A window of 0 is a valid caliper.
    MatchJob::new_transition(&records, 1)
        .with_ratio(MatchRatio::new(1).expect("ratio"))
        .with_constraint(caliper_on_field("x", 0.0))
        .run()
        .expect("a zero window");
}

#[test]
fn an_estimand_no_matcher_targets_is_an_error() {
    let records = cohort();
    let refused = MatchJob::new_transition(&records, 1)
        .with_estimand(Estimand::Ato)
        .run()
        .unwrap_err();
    assert_eq!(
        refused,
        Error::UnsupportedEstimand {
            estimand: Estimand::Ato,
            method: "matching",
        }
    );
    assert_eq!(refused.kind(), ErrorKind::Unsupported);
    assert!(matches!(
        MatchWeightMethod::for_estimand(Estimand::Ato),
        Err(Error::UnsupportedEstimand { .. })
    ));
    let data = AbadieImbensData::new(
        vec![true, false, true, false],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![vec![0.5, 1.5, 2.0, 3.5]],
    )
    .expect("valid columns");
    for target in [Estimand::Atm, Estimand::Ato] {
        let spec = AbadieImbensSpec {
            target,
            ..AbadieImbensSpec::new()
        };
        assert!(matches!(
            abadie_imbens(&data, &spec),
            Err(Error::UnsupportedEstimand { .. })
        ));
    }
}

#[test]
fn the_abadie_imbens_data_refuse_bad_columns() {
    let outcome = [1.0, 2.0, 3.0, 4.0];
    let treated = vec![true, false, true, false];
    let x = vec![0.5, 1.5, 2.0, 3.5];
    assert_rejects_bad_input("AbadieImbensData::new outcome", |bad| {
        (bad != BadInput::Unsorted)
            .then(|| AbadieImbensData::new(treated.clone(), bad.column(&outcome), vec![x.clone()]))
    });
    assert_rejects_bad_input("AbadieImbensData::new covariate", |bad| {
        (bad != BadInput::Unsorted && bad != BadInput::Empty)
            .then(|| AbadieImbensData::new(treated.clone(), outcome.to_vec(), vec![bad.column(&x)]))
    });
}

#[test]
fn the_distance_matrix_refuses_bad_values() {
    let ids = |names: &[&str]| names.iter().map(ToString::to_string).collect::<Vec<_>>();
    let values = [1.0, 2.0, 3.0, 4.0];
    // An infinite distance forbids a pair and is valid.
    assert_rejects_bad_input("PairDistanceMatrix::new", |bad| {
        matches!(bad, BadInput::Nan | BadInput::Mismatched).then(|| {
            PairDistanceMatrix::new(ids(&["a", "b"]), ids(&["x", "y"]), bad.column(&values))
        })
    });
}

#[test]
fn every_variant_kind_and_code_is_stable() {
    let errors = [
        Error::DuplicateId { id: "a".into() },
        Error::EventBeforeBirth { id: "a".into() },
        Error::NegativeBirthDateWindow,
        Error::ZeroMatchRatio,
        Error::Option {
            option: "ratio",
            message: String::new(),
        },
        Error::NoSharedStratum,
        Error::NoMatches,
        Error::SingularCovariance,
        Error::NotConverged,
        Error::ReestimateAcross,
        Error::Trial {
            period: 1,
            source: Box::new(Error::NoMatches),
        },
    ];
    let kinds: Vec<ErrorKind> = errors.iter().map(Classify::kind).collect();
    assert_eq!(
        kinds,
        [
            ErrorKind::Input,
            ErrorKind::Input,
            ErrorKind::Input,
            ErrorKind::Input,
            ErrorKind::Input,
            ErrorKind::Input,
            ErrorKind::Numerical,
            ErrorKind::Numerical,
            ErrorKind::Numerical,
            ErrorKind::Unsupported,
            ErrorKind::Numerical,
        ]
    );
    // The trial's error has the code of its source, so it is left out here.
    assert_codes_unique("matching", &errors[..errors.len() - 1]);
}
