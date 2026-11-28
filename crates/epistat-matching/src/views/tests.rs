use super::*;
use crate::RoleIndexedRecord;
use crate::constraints::ResidenceSpell;
use crate::matching::CovariateRecord;
use crate::types::{BaseRecord, CovariateRef};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("a valid date")
}

fn unix(y: i64, m: i64, d: i64) -> i64 {
    Day::from_civil(y, m, d).expect("a date").unix_days()
}

/// The rows of a view are the records the same values make through
/// `BaseRecord`: the id, the stratum and the numeric field of each column,
/// a null as a value the record does not have.
#[test]
fn the_rows_of_a_view_equal_base_records_of_the_same_values() {
    let ids = ["b", "a", "c"];
    let x = [Some(1.5), None, Some(-2.0)];
    let s = [Some("F"), Some("M"), None];
    let data = TreatmentData::builder(&ids, &[true, false, false])
        .numeric("x", &x[..])
        .categorical("s", &s[..])
        .build()
        .expect("a valid view");
    let birth = date(1970, 1, 1);
    for (row, record) in data.cohort().rows().enumerate() {
        let mut base = BaseRecord::new(ids[row], birth);
        if let Some(value) = x[row] {
            base = base.with_numeric("x", value);
        }
        if let Some(value) = s[row] {
            base.strata.insert("s".to_string(), value.to_string());
        }
        assert_eq!(record.id(), base.id());
        assert_eq!(record.birth_date(), base.birth_date());
        assert_eq!(record.stratum("s"), base.stratum("s"));
        assert_eq!(record.numeric("x"), base.numeric("x"));
        let covariate = x[row].map_or(CovariateRef::Missing, CovariateRef::Numeric);
        assert_eq!(record.covariate("x"), Some(covariate));
    }
    assert_eq!(data.n_treated(), 1);
    assert_eq!(data.treated(), &[true, false, false]);
    assert_eq!(
        data.columns().collect::<Vec<_>>(),
        [("x", ColumnKind::Numeric), ("s", ColumnKind::Categorical)]
    );

    let birth = [unix(2010, 1, 1), unix(2011, 6, 30), unix(2012, 2, 29)];
    let index = [Some(unix(2013, 1, 1)), None, Some(unix(2014, 1, 1))];
    let death = [None, Some(unix(2020, 1, 1)), None];
    let entry = [None, None, Some(unix(2012, 3, 1))];
    let cohort = RiskSetCohort::builder(&ids, Col::unix_days(&birth[..]))
        .index_date(Col::unix_days_nullable(&index[..]))
        .death_date(Col::unix_days_nullable(&death[..]))
        .entry_date(Col::unix_days_nullable(&entry[..]))
        .can_be_case(&[true, true, false])
        .build()
        .expect("a valid cohort");
    let naive = |days: Option<i64>| {
        days.map(|days| {
            let (y, m, d) = Day::from_unix_days(days).expect("a day").to_civil();
            date(
                i32::try_from(y).expect("a year"),
                u32::try_from(m).expect("a month"),
                u32::try_from(d).expect("a day"),
            )
        })
    };
    for (row, record) in cohort.cohort().risk_set_rows().iter().enumerate() {
        assert_eq!(record.id(), ids[row]);
        assert_eq!(Some(record.birth_date()), naive(Some(birth[row])));
        assert_eq!(record.event_date(), naive(index[row]));
        assert_eq!(record.death_date(), naive(death[row]));
        assert_eq!(record.entry_date(), naive(entry[row]));
    }
    // c's index date ends its time at risk, and does not make it a case.
    assert_eq!(cohort.n_cases(), 1);
    assert_eq!(cohort.id_order().by_id(), &[1, 0, 2]);
    assert_eq!(cohort.can_be_case(), &[true, true, false]);
    assert_eq!(cohort.comparator_until(), None);
    assert!(!cohort.has_residence());
}

/// The order of a view's rows, from the codes of its ids, is the order the
/// engine finds by comparing string ids, and numeric for integer ids.
#[test]
fn the_order_from_the_codes_of_string_ids_is_the_engines() {
    let mut state = 7_u64;
    let ids: Vec<String> = (0..2_000)
        .map(|k| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            format!("{}-{k}", (state >> 40) % 1_000)
        })
        .collect();
    let treated = vec![false; ids.len()];
    let data = TreatmentData::builder(&ids, &treated)
        .build()
        .expect("distinct ids");
    let birth = date(1970, 1, 1);
    let records: Vec<BaseRecord> = ids.iter().map(|id| BaseRecord::new(id, birth)).collect();
    let engine = IdOrder::of(&records).expect("fits");
    assert_eq!(data.id_order().by_id(), engine.by_id());
    for idx in 0..ids.len() {
        assert_eq!(data.id_order().rank(idx), engine.rank(idx));
    }
    assert!(data.id_order().distinct());

    let numbers = [10_u64, 9, 100];
    let data = TreatmentData::builder(&numbers, &[true, false, false])
        .build()
        .expect("distinct ids");
    assert_eq!(data.id_order().by_id(), &[1, 0, 2]);
    let text: Vec<BaseRecord> = numbers
        .iter()
        .map(|id| BaseRecord::new(id.to_string(), birth))
        .collect();
    assert_eq!(IdOrder::of(&text).expect("fits").by_id(), &[0, 2, 1]);
}

#[test]
fn residence_becomes_the_engines_spells() {
    let ids = ["p1", "p2", "p3"];
    let birth = [unix(2000, 1, 1); 3];
    let person = ["p3", "p1", "p3"];
    let from = [None, Some(unix(2001, 1, 1)), Some(unix(2005, 1, 1))];
    let to = [Some(unix(2002, 12, 31)), None, Some(unix(2005, 1, 1))];
    let cohort = |end| {
        RiskSetCohort::builder(&ids, Col::unix_days(&birth[..]))
            .residence(Residence::spells(
                &person,
                Col::unix_days_nullable(&from[..]),
                Col::unix_days_nullable(&to[..]),
                end,
            ))
            .build()
    };
    let inclusive = cohort(SpellEnd::Inclusive).expect("valid spells");
    assert_eq!(
        inclusive.residence().expect("residence"),
        &[
            Some(vec![ResidenceSpell {
                from: Some(date(2001, 1, 1)),
                to: None,
            }]),
            None,
            Some(vec![
                ResidenceSpell {
                    from: None,
                    to: Some(date(2003, 1, 1)),
                },
                ResidenceSpell {
                    from: Some(date(2005, 1, 1)),
                    to: Some(date(2005, 1, 2)),
                },
            ]),
        ]
    );
    let exclusive = cohort(SpellEnd::Exclusive).expect("valid spells");
    assert_eq!(
        exclusive.residence().expect("residence")[2]
            .as_deref()
            .map(|spells| spells[0].to),
        Some(Some(date(2002, 12, 31)))
    );

    let emigration = [None, Some(unix(2004, 5, 1)), None];
    let cohort = RiskSetCohort::builder(&ids, Col::unix_days(&birth[..]))
        .residence(Residence::until_emigration(Col::unix_days_nullable(
            &emigration[..],
        )))
        .build()
        .expect("valid dates");
    assert_eq!(
        cohort.residence().expect("residence"),
        &[
            None,
            Some(vec![ResidenceSpell {
                from: None,
                to: Some(date(2004, 5, 1)),
            }]),
            None,
        ]
    );
}

/// A label type whose distinct values print alike.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Blurred(u8);

impl fmt::Display for Blurred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unit {}", self.0 / 10)
    }
}

impl Label for Blurred {}

#[test]
fn a_view_refuses_bad_ids_and_columns() {
    let ids = ["a", "b", "a"];
    assert_eq!(
        TreatmentData::builder(&ids, &[true, false, true])
            .build()
            .unwrap_err(),
        Error::DuplicateIdRows {
            id: "a".into(),
            first: 0,
            row: 2,
        }
    );
    assert!(matches!(
        TreatmentData::builder::<&str>(&[], &[]).build(),
        Err(Error::Input(epistat_core::InputError::Empty {
            what: "ids"
        }))
    ));
    assert_eq!(
        TreatmentData::builder(&[Blurred(11), Blurred(12)], &[true, false])
            .build()
            .unwrap_err(),
        Error::AmbiguousLabel {
            column: "ids".into(),
            text: "unit 1".into(),
        }
    );

    let ids = ["a", "b", "c"];
    let treated = [true, false, false];
    let base = || TreatmentData::builder(&ids, &treated);
    assert_eq!(
        TreatmentData::builder(&ids, &treated[..2])
            .build()
            .unwrap_err(),
        Error::ColumnLength {
            column: "treated".into(),
            found: 2,
            expected: 3,
        }
    );
    assert_eq!(
        base().numeric("x", &[1.0, 2.0][..]).build().unwrap_err(),
        Error::ColumnLength {
            column: "x".into(),
            found: 2,
            expected: 3,
        }
    );
    assert_eq!(
        base()
            .numeric("x", &[1.0, f64::NAN, 2.0][..])
            .build()
            .unwrap_err(),
        Error::NonFiniteValue {
            column: "x".into(),
            row: 1,
            id: "b".into(),
        }
    );
    assert_eq!(
        base()
            .numeric("x", &[1.0, 2.0, 3.0][..])
            .categorical("x", &["u", "v", "w"][..])
            .build()
            .unwrap_err(),
        Error::DuplicateColumn { column: "x".into() }
    );
    assert_eq!(
        base()
            .numeric("x", Col::with_validity(&[1.0, 2.0, 3.0], &[true, false]))
            .build()
            .unwrap_err(),
        Error::ColumnLength {
            column: "x validity".into(),
            found: 2,
            expected: 3,
        }
    );
    assert_eq!(
        base()
            .categorical("g", &[Blurred(1), Blurred(2), Blurred(1)][..])
            .build()
            .unwrap_err(),
        Error::AmbiguousLabel {
            column: "g".into(),
            text: "unit 0".into(),
        }
    );
    assert_eq!(
        base()
            .date("d", Col::unix_days(&[0_i64, i64::MAX, 0][..]))
            .build()
            .unwrap_err(),
        Error::DateOutOfRange {
            column: "d".into(),
            row: 1,
        }
    );
}

#[test]
fn a_named_column_is_checked_against_the_data() {
    let sex = [Some("F"), None, Some("M")];
    let data = TreatmentData::builder(&["a", "b", "c"], &[true, false, false])
        .categorical("sex", &sex[..])
        .numeric("age", &[30.0, 40.0, 50.0][..])
        .date("visit", Col::unix_days(&[1_i64, 2, 3][..]))
        .build()
        .expect("a valid view");
    assert_eq!(data.column_kind("visit"), Ok(ColumnKind::Date));
    let unknown = data.column_kind("gender").unwrap_err();
    assert_eq!(
        unknown,
        Error::UnknownColumn {
            name: "gender".into(),
            known: vec!["sex".into(), "age".into(), "visit".into()],
        }
    );
    assert_eq!(
        unknown.to_string(),
        "no column `gender`; the data have sex, age, visit"
    );
    assert_eq!(
        data.check_exact_key("sex", ExactMissing::Refuse),
        Err(Error::NullKey {
            column: "sex".into(),
            row: 1,
            id: "b".into(),
        })
    );
    assert_eq!(data.check_exact_key("sex", ExactMissing::OwnLevel), Ok(()));
    assert_eq!(
        data.check_exact_key("age", ExactMissing::OwnLevel),
        Err(Error::WrongColumnKind {
            name: "age".into(),
            expected: ColumnKind::Categorical,
            found: ColumnKind::Numeric,
        })
    );
    assert!(matches!(
        data.check_exact_key("gender", ExactMissing::OwnLevel),
        Err(Error::UnknownColumn { .. })
    ));
}

#[test]
fn a_risk_set_cohort_refuses_bad_dates_and_spells() {
    let ids = ["a", "b"];
    let birth = [unix(2010, 1, 1), unix(2011, 1, 1)];
    let base = || RiskSetCohort::builder(&ids, Col::unix_days(&birth[..]));
    assert_eq!(
        base()
            .index_date(Col::unix_days_nullable(&[None, Some(unix(2010, 6, 1))][..]))
            .build()
            .unwrap_err(),
        Error::IndexBeforeBirth {
            id: "b".into(),
            row: 1,
        }
    );
    let nulls = [Some(birth[0]), None];
    assert_eq!(
        RiskSetCohort::builder(&ids, Col::unix_days_nullable(&nulls[..]))
            .build()
            .unwrap_err(),
        Error::NullValue {
            column: "birth date".into(),
            row: 1,
        }
    );
    // A day of a Day, far outside the dates of the engine.
    let far = [birth[0], Day::MAX.unix_days()];
    assert_eq!(
        RiskSetCohort::builder(&ids, Col::unix_days(&far[..]))
            .build()
            .unwrap_err(),
        Error::DateOutOfRange {
            column: "birth date".into(),
            row: 1,
        }
    );
    assert_eq!(
        base().can_be_case(&[true]).build().unwrap_err(),
        Error::ColumnLength {
            column: "can be case".into(),
            found: 1,
            expected: 2,
        }
    );
    let spells = |person: &[&str], from: i64, to: i64, end| {
        let (from, to) = ([Some(from)], [Some(to)]);
        base()
            .residence(Residence::spells(
                person,
                Col::unix_days_nullable(&from[..]),
                Col::unix_days_nullable(&to[..]),
                end,
            ))
            .build()
    };
    assert_eq!(
        spells(&["z"], 1, 2, SpellEnd::Exclusive).unwrap_err(),
        Error::UnknownSpellPerson {
            person: "z".into(),
            row: 0,
        }
    );
    assert_eq!(
        spells(&["a"], 5, 4, SpellEnd::Inclusive).unwrap_err(),
        Error::SpellOrder {
            person: "a".into(),
            row: 0,
        }
    );
    assert!(spells(&["a"], 5, 5, SpellEnd::Inclusive).is_ok());
    assert!(spells(&["a"], 5, 5, SpellEnd::Exclusive).is_ok());
    assert_eq!(
        spells(&["a"], 5, 4, SpellEnd::Exclusive).unwrap_err(),
        Error::SpellOrder {
            person: "a".into(),
            row: 0,
        }
    );
    assert_eq!(
        base()
            .residence(Residence::until_emigration(Col::unix_days_nullable(
                &[None::<i64>][..]
            )))
            .build()
            .unwrap_err(),
        Error::ColumnLength {
            column: "emigration date".into(),
            found: 1,
            expected: 2,
        }
    );
}
