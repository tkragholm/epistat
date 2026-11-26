//! The pieces of `lexis::split_at` against `Epi::splitLexis` and
//! `popEpi::splitMulti` (`tests/data/lexis_reference.R`), with calendar years
//! and with ages on the 365.25-day rule, where the R day scales are exact.
//! The pieces are integers, so agreement is exact. The labels of the pieces
//! are checked against `Epi::timeBand(x, scale, "left")`.

use epistat_register::lexis::{Cuts, labels, split_at};
use epistat_register::{AgeRule, LeapBirthday};
use epistat_testkit::{Fingerprint, assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Pieces {
    id: Vec<f64>,
    start: Vec<f64>,
    dur: Vec<f64>,
    year: Vec<Option<i64>>,
    age: Vec<Option<i64>>,
}

#[derive(Deserialize)]
struct Reference {
    birth: Vec<f64>,
    start: Vec<f64>,
    end: Vec<f64>,
    ages: Vec<u32>,
    epi: Pieces,
}

#[expect(clippy::cast_possible_truncation, reason = "whole days from R")]
fn ints(values: &[f64]) -> Vec<i64> {
    values.iter().map(|&v| v as i64).collect()
}

#[test]
fn pieces_agree_with_epi_and_popepi() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/lexis_reference.json"
    ));
    let (birth, start, end) = (ints(&r.birth), ints(&r.start), ints(&r.end));
    let cuts = [
        Cuts::CalendarYears,
        Cuts::AgeYears {
            birth: &birth,
            ages: &r.ages,
            rule: AgeRule::DaysOver365_25,
        },
    ];
    let split = split_at(&start, &end, &cuts).unwrap();
    // Each piece's labels against Epi::timeBand(x, scale, "left").
    let label = labels(&split, &cuts).unwrap();
    let mut got_labels: Vec<(i64, i64, Option<i64>, Option<i64>)> = (0..split.start.len())
        .map(|i| {
            (
                i64::from(split.source_row[i]),
                split.start[i],
                label[0][i],
                label[1][i],
            )
        })
        .collect();
    got_labels.sort_unstable();
    let want_labels: Vec<(i64, i64, Option<i64>, Option<i64>)> = (0..r.epi.id.len())
        .map(|i| {
            (
                ints(&r.epi.id)[i],
                ints(&r.epi.start)[i],
                r.epi.year[i],
                r.epi.age[i],
            )
        })
        .collect();
    assert_eq!(got_labels, want_labels);
    assert!(
        want_labels.iter().any(|w| w.3.is_none()),
        "a piece below the first age"
    );
    // R drops the rows that are empty, and numbers the others from 0 in the
    // reference; ours keeps the source row, so the same rows appear.
    let mut got: Vec<(i64, i64, i64)> = (0..split.start.len())
        .map(|i| {
            (
                i64::from(split.source_row[i]),
                split.start[i],
                split.end[i] - split.start[i],
            )
        })
        .collect();
    got.sort_unstable();
    let ids: Vec<i64> = ints(&r.epi.id);
    // R's ids are the original row numbers (from 1) minus 1.
    let want: Vec<(i64, i64, i64)> = (0..ids.len())
        .map(|i| (ids[i], ints(&r.epi.start)[i], ints(&r.epi.dur)[i]))
        .collect();
    assert_eq!(got.len(), want.len(), "piece count");
    assert_eq!(got, want);
    assert_same_on_threads(|| {
        let again = split_at(&start, &end, &[Cuts::CalendarYears]).unwrap();
        again.start.iter().map(|&s| s as f64).collect()
    });
    let mut fp = Fingerprint::new();
    for (row, s, d) in &got {
        fp = fp
            .u64(row.cast_unsigned())
            .u64(s.cast_unsigned())
            .u64(d.cast_unsigned());
    }
    assert!(got.len() > 3000);
    assert_eq!(fp.finish(), 18_086_303_607_271_229_656, "golden hash");
    // A civil rule gives different cuts from the 365.25-day rule for some
    // people, which is why R's scale cannot check it.
    let civil = split_at(
        &start,
        &end,
        &[Cuts::AgeYears {
            birth: &birth,
            ages: &r.ages,
            rule: AgeRule::CivilYears(LeapBirthday::March1),
        }],
    )
    .unwrap();
    assert_ne!(civil.start.len(), 0);
}
