//! Records built from Arrow arrays equal the records built directly, and a
//! match on them gives the same output, bit for bit, as the same match on
//! the directly built records. The cohort and the job are those of the
//! `nearest` case of `tests/golden.rs`, whose hash is asserted again here.
//!
//! The two timing tests are `#[ignore]`: run them in a release build with
//! `cargo test --release --features arrow --test arrow -- --ignored --nocapture`.

#![cfg(feature = "arrow")]
#![expect(
    clippy::unreadable_literal,
    reason = "the hashes are pasted as the test prints them"
)]

mod common;

use std::sync::Arc;

use arrow_array::types::{Int8Type, UInt16Type};
use arrow_array::{
    Array, ArrayRef, Date32Array, DictionaryArray, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use chrono::NaiveDate;
use common::{Level, Person, Rec, cohort, records};
use epistat_matching::arrow::{ArrowInputError, ArrowRecords, column};
use epistat_matching::prelude::*;
use epistat_matching::{
    Cohort, CohortError, CovariateRef, CovariateValue, NearestBirthDateSelection,
};

fn epoch() -> NaiveDate {
    NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
}

fn d32(date: NaiveDate) -> i32 {
    i32::try_from((date - epoch()).num_days()).unwrap()
}

fn dates(values: impl Iterator<Item = Option<NaiveDate>>) -> ArrayRef {
    Arc::new(Date32Array::from(
        values.map(|d| d.map(d32)).collect::<Vec<_>>(),
    ))
}

fn strings<'a>(values: impl Iterator<Item = Option<&'a str>>) -> ArrayRef {
    Arc::new(StringArray::from(values.collect::<Vec<_>>()))
}

fn same_base(a: &BaseRecord, b: &BaseRecord) {
    assert_eq!(a.id, b.id);
    assert_eq!(a.birth_date, b.birth_date);
    assert_eq!(a.strata, b.strata);
    assert_eq!(a.numerics.len(), b.numerics.len());
    for (name, value) in &a.numerics {
        assert_eq!(value.to_bits(), b.numerics[name].to_bits(), "{name}");
    }
    assert_eq!(a.unique_key, b.unique_key);
    assert_eq!(a.death_date, b.death_date);
    assert_eq!(a.emigration_date, b.emigration_date);
    assert_eq!(a.entry_date, b.entry_date);
}

fn same_covariate(a: &CovariateValue, b: &CovariateValue) {
    match (a, b) {
        (CovariateValue::Numeric(x), CovariateValue::Numeric(y)) => {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        _ => assert_eq!(a, b),
    }
}

/// The columns of a cohort of people as Arrow arrays.
struct Columns {
    id: ArrayRef,
    birth: ArrayRef,
    sex: ArrayRef,
    municipality: ArrayRef,
    parent_year: ArrayRef,
    death: ArrayRef,
    emigration: ArrayRef,
    event: ArrayRef,
}

fn columns(people: &[Person]) -> Columns {
    Columns {
        id: strings(people.iter().map(|p| Some(p.id.as_str()))),
        birth: dates(people.iter().map(|p| Some(p.birth))),
        sex: strings(people.iter().map(|p| Some(p.sex))),
        municipality: strings(people.iter().map(|p| Some(p.municipality.as_str()))),
        parent_year: Arc::new(Float64Array::from(
            people.iter().map(|p| p.parent_year).collect::<Vec<_>>(),
        )),
        death: dates(people.iter().map(|p| p.death)),
        emigration: dates(people.iter().map(|p| p.emigration)),
        event: dates(people.iter().map(|p| p.first_date)),
    }
}

fn arrow_risk_set(c: &Columns, with_caliper: bool) -> Vec<Rec> {
    let mut builder = ArrowRecords::new(&*c.id, &*c.birth)
        .strata("sex", &*c.sex)
        .strata("municipality", &*c.municipality)
        .death_date(&*c.death)
        .emigration_date(&*c.emigration)
        .event_date(&*c.event);
    if with_caliper {
        builder = builder.numeric("parent_year", &*c.parent_year);
    }
    builder.risk_set_records().unwrap()
}

#[test]
fn risk_set_records_equal_the_direct_records() {
    let people = cohort(2_000, 100, 10, 14);
    let c = columns(&people);
    for with_caliper in [false, true] {
        let direct = records(&people, Level::Municipality, false, with_caliper);
        let arrow = arrow_risk_set(&c, with_caliper);
        assert_eq!(direct.len(), arrow.len());
        assert!(arrow.iter().any(|r| r.transition_date.is_some()));
        assert!(arrow.iter().any(|r| r.death_date.is_some()));
        assert!(arrow.iter().any(|r| r.emigration_date.is_some()));
        for (a, b) in direct.iter().zip(&arrow) {
            same_base(&a.record, &b.record);
            assert_eq!(a.transition_date, b.transition_date);
        }
        // The caliper variant has records with and without the numeric field.
        assert_eq!(arrow.iter().any(|r| !r.numerics.is_empty()), with_caliper);
        assert!(arrow.iter().any(|r| r.numerics.is_empty()));
    }
}

#[test]
fn base_records_carry_every_field() {
    let birth = NaiveDate::from_ymd_opt(2001, 5, 6).unwrap();
    let before_epoch = NaiveDate::from_ymd_opt(1960, 2, 29).unwrap();
    let id = strings([Some("a"), Some("b")].into_iter());
    let born = dates([Some(birth), Some(before_epoch)].into_iter());
    let key = strings([Some("k"), None].into_iter());
    let death = dates([None, Some(birth)].into_iter());
    let emigration = dates([Some(before_epoch), None].into_iter());
    let entry = dates([Some(birth), Some(birth)].into_iter());
    let x = Float64Array::from(vec![Some(1.5), None]);
    let got = ArrowRecords::new(&*id, &*born)
        .unique_key(&*key)
        .death_date(&*death)
        .emigration_date(&*emigration)
        .entry_date(&*entry)
        .numeric("x", &x)
        .base_records()
        .unwrap();
    let want = [
        BaseRecord::new("a", birth)
            .with_emigration_date(before_epoch)
            .with_entry_date(birth)
            .with_numeric("x", 1.5),
        BaseRecord::new("b", before_epoch)
            .with_death_date(birth)
            .with_entry_date(birth),
    ];
    let mut want = want;
    want[0].unique_key = Some("k".to_string());
    for (a, b) in want.iter().zip(&got) {
        same_base(a, b);
    }
}

#[test]
fn balance_records_equal_the_builder() {
    let birth = NaiveDate::from_ymd_opt(2001, 5, 6).unwrap();
    let id = strings([Some("a"), Some("b"), Some("c")].into_iter());
    let born = dates(std::iter::repeat_n(Some(birth), 3));
    let age = Float64Array::from(vec![Some(3.5), None, Some(f64::NAN)]);
    let inf = Float64Array::from(vec![Some(f64::INFINITY), Some(2.0), Some(-0.0)]);
    let region = strings([Some("N"), None, Some("S")].into_iter());
    let got = ArrowRecords::new(&*id, &*born)
        .numeric_covariate("age", &age)
        .numeric_covariate("inf", &inf)
        .categorical_covariate("region", &*region)
        .balance_records()
        .unwrap();
    let region_values = [Some("N"), None, Some("S")];
    let ages = [Some(3.5), None, Some(f64::NAN)];
    let infs = [Some(f64::INFINITY), Some(2.0), Some(-0.0)];
    for (i, name) in ["a", "b", "c"].into_iter().enumerate() {
        let want = BalanceRecord::builder(name, birth)
            .numeric("age", ages[i])
            .numeric("inf", infs[i])
            .categorical("region", region_values[i].map(str::to_string))
            .build();
        same_base(&want.core, &got[i].core);
        assert_eq!(want.covariates.len(), got[i].covariates.len());
        for (key, value) in &want.covariates {
            same_covariate(value, &got[i].covariates[key]);
        }
    }
    assert_eq!(got[1].covariates["age"], CovariateValue::Missing);
    assert_eq!(got[2].covariates["age"], CovariateValue::Missing);
    assert_eq!(got[0].covariates["inf"], CovariateValue::Missing);
    assert_eq!(
        got[2].covariates["region"],
        CovariateValue::Categorical("S".into())
    );
}

/// Every row of `cohort` reads as `records` do: id, dates, every stratum,
/// numeric field and covariate the record has, and none it does not.
fn rows_equal_records(cohort: &Cohort, records: &[BalanceRecord], names: &[&str]) {
    assert_eq!(cohort.len(), records.len());
    for (row, record) in cohort.rows().zip(records) {
        assert_eq!(row.id(), record.id());
        assert_eq!(row.birth_date(), record.birth_date());
        assert_eq!(row.unique_key(), record.unique_key());
        assert_eq!(row.death_date(), record.death_date());
        assert_eq!(row.emigration_date(), record.emigration_date());
        assert_eq!(row.entry_date(), record.entry_date());
        for name in names {
            assert_eq!(row.stratum(name), record.stratum(name), "{name}");
            assert_eq!(
                row.numeric(name).map(f64::to_bits),
                record.numeric(name).map(f64::to_bits),
                "{name}"
            );
            match (row.covariate(name), record.covariate(name)) {
                (Some(CovariateRef::Numeric(a)), Some(CovariateRef::Numeric(b))) => {
                    assert_eq!(a.to_bits(), b.to_bits(), "{name}");
                }
                (a, b) => assert_eq!(a, b, "{name}"),
            }
        }
        let mut row_names: Vec<&str> = row.covariate_names().collect();
        let mut record_names: Vec<&str> = record.covariate_names().collect();
        row_names.sort_unstable();
        record_names.sort_unstable();
        assert_eq!(row_names, record_names);
    }
    assert!(!cohort.has_built_record_maps());
}

#[test]
fn a_cohort_holds_the_balance_records() {
    let birth = NaiveDate::from_ymd_opt(2001, 5, 6).unwrap();
    let before_epoch = NaiveDate::from_ymd_opt(1960, 2, 29).unwrap();
    let id = Int64Array::from(vec![7, 8, 9]);
    let born = dates([Some(birth), Some(before_epoch), Some(birth)].into_iter());
    let key = strings([Some("k"), None, Some("k")].into_iter());
    let sex = strings([Some("1"), Some("2"), None].into_iter());
    let death = dates([None, Some(birth), None].into_iter());
    let emigration = dates([Some(before_epoch), None, None].into_iter());
    let entry = dates([Some(birth), Some(birth), None].into_iter());
    let event = dates([None, None, Some(birth)].into_iter());
    let age = Float64Array::from(vec![Some(3.5), None, Some(f64::NAN)]);
    let other = Float64Array::from(vec![Some(f64::INFINITY), Some(2.0), Some(-0.0)]);
    let region: ArrayRef = Arc::new(
        [Some("N"), None, Some("S")]
            .into_iter()
            .collect::<DictionaryArray<Int8Type>>(),
    );
    let columns = ArrowRecords::new(&id, &*born)
        .strata("sex", &*sex)
        .numeric("age", &age)
        .numeric_covariate("age", &age)
        .numeric("other", &other)
        .numeric_covariate("inf", &other)
        .categorical_covariate("region", &*region)
        .unique_key(&*key)
        .death_date(&*death)
        .emigration_date(&*emigration)
        .entry_date(&*entry)
        .event_date(&*event);
    let records = columns.balance_records().unwrap();
    let cohort = columns.cohort().unwrap();
    rows_equal_records(
        &cohort,
        &records,
        &["sex", "age", "other", "inf", "region", "absent"],
    );
    assert_eq!(cohort.numeric_names(), ["age", "other"]);
    let risk = cohort.risk_set_rows();
    let direct = columns.risk_set_records().unwrap();
    for (a, b) in risk.iter().zip(&direct) {
        assert_eq!(a.transition_date, b.transition_date);
    }
    // Two strata of one name are an error naming it.
    let twice = ArrowRecords::new(&id, &*born)
        .strata("sex", &*sex)
        .strata("sex", &*key)
        .cohort();
    assert!(matches!(
        twice,
        Err(ArrowInputError::Cohort(CohortError::DuplicateName { column })) if column == "sex"
    ));
    // A null id is an error, as for the records.
    let ids = strings([Some("a"), None, Some("c")].into_iter());
    assert_eq!(
        failure(ArrowRecords::new(&*ids, &*born).cohort()),
        ArrowInputError::NullValue {
            column: "id".into(),
            row: 1
        }
    );
}

#[test]
fn a_match_on_cohort_rows_is_bit_identical() {
    let people = cohort(4_000, 100, 10, 15);
    let c = columns(&people);
    let built = ArrowRecords::new(&*c.id, &*c.birth)
        .strata("sex", &*c.sex)
        .strata("municipality", &*c.municipality)
        .numeric("parent_year", &*c.parent_year)
        .death_date(&*c.death)
        .emigration_date(&*c.emigration)
        .event_date(&*c.event)
        .cohort()
        .unwrap();
    let rows = built.risk_set_rows();
    let direct = records(&people, Level::Municipality, false, true);
    let on_rows = MatchJob::new_transition(&rows, 99)
        .with_ratio(MatchRatio::new(3).unwrap())
        .with_birth_window(BirthDateWindowDays::new(60).unwrap())
        .with_age_limit(AgeLimitYears::new(6).unwrap())
        .with_exact_matches(["sex", "municipality"])
        .with_alive_check()
        .with_strategy(NearestBirthDateSelection)
        .run();
    assert_ne!(on_rows.matched_cases, 0);
    assert_eq!(format!("{:?}", run(&direct)), format!("{on_rows:?}"));
    assert!(!built.has_built_record_maps());
}

#[test]
fn id_and_string_types() {
    let birth = NaiveDate::from_ymd_opt(2001, 5, 6).unwrap();
    let born = dates(std::iter::repeat_n(Some(birth), 3));
    let ids: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["7", "8", "9"])),
        Arc::new(LargeStringArray::from(vec!["7", "8", "9"])),
        Arc::new(Int32Array::from(vec![7, 8, 9])),
        Arc::new(Int64Array::from(vec![7, 8, 9])),
        Arc::new(UInt32Array::from(vec![7, 8, 9])),
        Arc::new(UInt64Array::from(vec![7, 8, 9])),
        Arc::new(DictionaryArray::<Int8Type>::from_iter(["7", "8", "9"])),
    ];
    for id in &ids {
        let got = ArrowRecords::new(&**id, &*born).base_records().unwrap();
        let names: Vec<&str> = got.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(names, ["7", "8", "9"], "{}", id.data_type());
    }
    // Strata as every text type, including a dictionary with a null key and a
    // null value.
    let dictionary: DictionaryArray<UInt16Type> =
        [Some("m"), None, Some("m")].into_iter().collect();
    let large = LargeStringArray::from(vec![Some("m"), None, Some("m")]);
    let plain = StringArray::from(vec![Some("m"), None, Some("m")]);
    for stratum in [&dictionary as &dyn Array, &large, &plain] {
        let got = ArrowRecords::new(&*ids[0], &*born)
            .strata("g", stratum)
            .base_records()
            .unwrap();
        assert_eq!(got[0].strata["g"], "m");
        assert!(got[1].strata.is_empty(), "{}", stratum.data_type());
        assert_eq!(got[2].strata["g"], "m");
    }
}

#[test]
fn sliced_arrays_read_the_rows_they_show() {
    let birth = NaiveDate::from_ymd_opt(2001, 5, 6).unwrap();
    let id = StringArray::from(vec!["a", "b", "c", "d"]);
    let born = dates(std::iter::repeat_n(Some(birth), 4));
    let got = ArrowRecords::new(&id.slice(1, 2), &*born.slice(1, 2))
        .base_records()
        .unwrap();
    let names: Vec<&str> = got.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(names, ["b", "c"]);
}

fn failure(result: Result<impl Sized, ArrowInputError>) -> ArrowInputError {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error,
    }
}

#[test]
fn nulls_and_types_are_errors_that_name_the_column() {
    let birth = NaiveDate::from_ymd_opt(2001, 5, 6).unwrap();
    let id = StringArray::from(vec![Some("a"), Some("b")]);
    let born = dates(std::iter::repeat_n(Some(birth), 2));

    let null_id = StringArray::from(vec![Some("a"), None]);
    let error = failure(ArrowRecords::new(&null_id, &*born).base_records());
    assert_eq!(
        error,
        ArrowInputError::NullValue {
            column: "id".into(),
            row: 1
        }
    );
    assert!(error.to_string().contains("column id has a null at row 1"));

    let null_birth = dates([Some(birth), None].into_iter());
    let error = failure(ArrowRecords::new(&id, &*null_birth).base_records());
    assert_eq!(
        error,
        ArrowInputError::NullValue {
            column: "birth_date".into(),
            row: 1
        }
    );

    let bad_birth = Int32Array::from(vec![1, 2]);
    let error = failure(ArrowRecords::new(&id, &bad_birth).base_records());
    assert!(matches!(&error, ArrowInputError::WrongType { column, .. } if column == "birth_date"));
    assert!(error.to_string().contains("Int32"));

    let bad_id = arrow_array::BooleanArray::from(vec![true, false]);
    let error = failure(ArrowRecords::new(&bad_id, &*born).base_records());
    assert!(matches!(&error, ArrowInputError::WrongType { column, .. } if column == "id"));

    let float_stratum = Float64Array::from(vec![1.0, 2.0]);
    let error = failure(
        ArrowRecords::new(&id, &*born)
            .strata("sex", &float_stratum)
            .base_records(),
    );
    assert!(matches!(&error, ArrowInputError::WrongType { column, .. } if column == "sex"));

    let text_numeric = StringArray::from(vec!["1", "2"]);
    let error = failure(
        ArrowRecords::new(&id, &*born)
            .numeric("year", &text_numeric)
            .base_records(),
    );
    assert!(matches!(&error, ArrowInputError::WrongType { column, .. } if column == "year"));

    let int_dictionary: DictionaryArray<Int8Type> = DictionaryArray::new(
        arrow_array::Int8Array::from(vec![0, 0]),
        Arc::new(Int32Array::from(vec![5])),
    );
    let error = failure(
        ArrowRecords::new(&id, &*born)
            .strata("g", &int_dictionary)
            .base_records(),
    );
    assert!(matches!(&error, ArrowInputError::WrongType { column, .. } if column == "g"));

    let short = Float64Array::from(vec![1.0]);
    let error = failure(
        ArrowRecords::new(&id, &*born)
            .numeric("year", &short)
            .base_records(),
    );
    assert_eq!(
        error,
        ArrowInputError::LengthMismatch {
            column: "year".into(),
            found: 1,
            expected: 2
        }
    );

    let bad_event = StringArray::from(vec!["x", "y"]);
    let error = failure(
        ArrowRecords::new(&id, &*born)
            .event_date(&bad_event)
            .risk_set_records(),
    );
    assert!(matches!(&error, ArrowInputError::WrongType { column, .. } if column == "event_date"));

    let far = Date32Array::from(vec![i32::MAX, 0]);
    let error = failure(ArrowRecords::new(&id, &far).base_records());
    assert_eq!(
        error,
        ArrowInputError::DateOutOfRange {
            column: "birth_date".into(),
            row: 0
        }
    );
}

#[test]
fn missing_column_in_a_batch_lists_the_columns_present() {
    let batch =
        RecordBatch::try_from_iter([("id", Arc::new(StringArray::from(vec!["a"])) as ArrayRef)])
            .unwrap();
    assert!(column(&batch, "id").is_ok());
    let error = failure(column(&batch, "birth"));
    assert!(
        error
            .to_string()
            .contains("birth is not in the record batch; it has id")
    );
}

fn fnv(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn run(records: &[Rec]) -> MatchOutcome {
    MatchJob::new_transition(records, 99)
        .with_ratio(MatchRatio::new(3).unwrap())
        .with_birth_window(BirthDateWindowDays::new(60).unwrap())
        .with_age_limit(AgeLimitYears::new(6).unwrap())
        .with_exact_matches(["sex", "municipality"])
        .with_alive_check()
        .with_strategy(NearestBirthDateSelection)
        .run()
}

fn weights_hash(outcome: &MatchOutcome) -> u64 {
    let method = MatchWeightMethod::AnchorUnitCandidateFractional;
    let unit = match_weights_from_pairs(&outcome.pairs, method).to_table();
    let pair = pair_weights_from_pairs(&outcome.pairs, method).to_table();
    fnv(&format!("{unit:?}{pair:?}"))
}

#[test]
fn a_match_on_arrow_records_is_bit_identical() {
    let people = cohort(4_000, 100, 10, 14);
    let direct = run(&records(&people, Level::Municipality, false, false));
    let arrow = run(&arrow_risk_set(&columns(&people), false));
    assert_ne!(direct.matched_cases, 0);
    assert_eq!(format!("{direct:?}"), format!("{arrow:?}"));
    // The golden hashes of the `nearest` case in tests/golden.rs.
    assert_eq!(
        (fnv(&format!("{arrow:?}")), weights_hash(&arrow)),
        (0x2c4ad6fd67c312f7, 0x585c1833c1e022cf)
    );
}

#[test]
fn a_match_with_a_caliper_on_nullable_numerics_is_bit_identical() {
    let people = cohort(4_000, 100, 10, 15);
    let run_with = |records: &[Rec]| {
        MatchJob::new_transition(records, 5)
            .with_ratio(MatchRatio::new(2).unwrap())
            .with_birth_window(BirthDateWindowDays::new(60).unwrap())
            .with_age_limit(AgeLimitYears::new(6).unwrap())
            .with_exact_matches(["sex", "municipality"])
            .with_constraints(vec![
                epistat_matching::constraints::caliper_on_field::<Rec>("parent_year", 5.0),
            ])
            .run()
    };
    let direct = run_with(&records(&people, Level::Municipality, false, true));
    let arrow = run_with(&arrow_risk_set(&columns(&people), true));
    assert_ne!(direct.matched_cases, 0);
    assert_eq!(format!("{direct:?}"), format!("{arrow:?}"));
}

/// Time and size of building `n` records from arrays made without per-row
/// structs.
fn time_build(n: usize) {
    use std::time::Instant;
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state >> 33
    };
    let id = Int64Array::from_iter_values((0..n).map(|i| i64::try_from(i).unwrap() * 7 + 1));
    let birth = Date32Array::from_iter_values(
        (0..n).map(|_| 11_000 + i32::try_from(next() % 5_800).unwrap()),
    );
    let sex =
        StringArray::from_iter_values((0..n).map(|_| if next() % 2 == 0 { "1" } else { "2" }));
    let labels: Vec<String> = (0..98).map(|m| format!("{:03}", 100 + m)).collect();
    let municipality = DictionaryArray::<UInt16Type>::from_iter(
        (0..n).map(|_| labels[(next() % 98) as usize].as_str()),
    );
    let year = Float64Array::from_iter(
        (0..n).map(|_| (next() % 1000 >= 20).then(|| 1960.0 + (next() % 40) as f64)),
    );
    let death = Date32Array::from_iter(
        (0..n).map(|i| (next() % 1000 < 20).then(|| 12_000 + i32::try_from(i % 4_000).unwrap())),
    );
    let event =
        Date32Array::from_iter((0..n).map(|_| {
            (next() % 1000 < 100).then(|| 14_000 + i32::try_from(next() % 3_000).unwrap())
        }));
    let start = Instant::now();
    let built = ArrowRecords::new(&id, &birth)
        .strata("sex", &sex)
        .strata("municipality", &municipality)
        .numeric("parent_year", &year)
        .death_date(&death)
        .event_date(&event)
        .risk_set_records()
        .unwrap();
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "ARROW_BUILD rows={n} seconds={seconds:.2} per_million={:.2}",
        seconds * 1e6 / n as f64
    );
    assert_eq!(built.len(), n);
    if n <= 1_000_000 {
        let start = Instant::now();
        let outcome = run(&built);
        println!(
            "ARROW_MATCH rows={n} seconds={:.2} pairs={}",
            start.elapsed().as_secs_f64(),
            outcome.pairs.len()
        );
    }
    std::hint::black_box(&built);
}

#[test]
#[ignore = "timing, run in release"]
fn time_one_million_records() {
    time_build(1_000_000);
}

#[test]
#[ignore = "timing, run in release"]
fn time_ten_million_records() {
    time_build(10_000_000);
}
