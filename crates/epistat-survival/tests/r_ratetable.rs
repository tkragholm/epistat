//! Expected counts from a rate table compared with `survival::pyears(expect =
//! "event")` on `survexp.us`.
//!
//! The reference is `tests/data/ratetable.json`, written by
//! `tests/data/ratetable.R`: the table, two cohorts with delayed entry, the
//! total expected count of each, and the expected count of the first 150 rows
//! one by one. R uses the first and last cell of a dimension for a value
//! outside its cut points, so the cuts given to `RateTable` are those of R with
//! the first and last moved far out, which gives the same reach. The second
//! cohort has entry dates before the first calendar year and the first cohort
//! has ages and dates beyond the last cell.
//!
//! Tolerance. A row crosses at most about 60 cells, and its expected count is a
//! sum of that many products, so its relative error is bounded by `60 eps`,
//! which is 1.3e-14. R adds the pieces of 2000 rows in turn in plain double
//! precision, so its total carries an error of up to `1e5 eps`, 2.2e-11, which
//! is the bound for the totals. The checks print the largest deviation in units
//! of its bound.

use epistat_survival::SurvData;
use epistat_survival::lexis::cells::CellKey;
use epistat_survival::lexis::{LexisOptions, TimeScale, split_cells};
use epistat_survival::ratetable::{RateDim, RateTable};
use epistat_testkit::{Fingerprint, assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    age_cuts: Vec<f64>,
    year_cuts: Vec<f64>,
    rates: Vec<f64>,
    cohorts: Vec<Cohort>,
}

#[derive(Deserialize)]
struct Cohort {
    name: String,
    start: Vec<f64>,
    stop: Vec<f64>,
    age0: Vec<f64>,
    date0: Vec<f64>,
    sex: Vec<u32>,
    expected: f64,
    per_row: Vec<f64>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/ratetable.json"
    ))
}

fn table(r: &Reference) -> RateTable {
    let mut age = r.age_cuts.clone();
    age.push(1e6);
    let mut year = r.year_cuts.clone();
    year[0] = -1e6;
    year.push(1e6);
    RateTable::new(
        vec![
            RateDim::Time { cuts: age },
            RateDim::Factor { levels: 2 },
            RateDim::Time { cuts: year },
        ],
        r.rates.clone(),
        1.0,
    )
    .unwrap()
}

#[test]
fn expected_counts_match_pyears() {
    let r = reference();
    let t = table(&r);
    let mut worst_total = 0.0_f64;
    let mut worst_row = 0.0_f64;
    for c in &r.cohorts {
        let status = vec![0; c.stop.len()];
        let data = SurvData::builder(&c.stop, &status)
            .start(&c.start)
            .build()
            .unwrap();
        let e = t.expected(&data, &[&c.age0, &c.date0], &[&c.sex]).unwrap();
        assert!(e.offtable.abs() < 1e-300, "{}", c.name);
        let total = (e.total - c.expected).abs() / (2.2e-11 * c.expected);
        worst_total = worst_total.max(total);
        for (i, want) in c.per_row.iter().enumerate() {
            let ratio = (e.per_row[i] - want).abs() / (1.3e-14 * want);
            worst_row = worst_row.max(ratio);
        }
        eprintln!("{}: total {:e} against {:e}", c.name, e.total, c.expected);
    }
    eprintln!("total {worst_total:.3}, per row {worst_row:.3} of the bound");
    assert!(worst_total <= 1.0 && worst_row <= 1.0);
}

#[test]
fn follow_up_before_the_first_cut_is_reported() {
    let r = reference();
    // The table as given, with its own outer cuts: calendar years from 1940 and
    // ages to 110 years, with an end for the last year.
    let mut age = r.age_cuts.clone();
    age.push(120.0 * 365.25);
    let mut year = r.year_cuts.clone();
    year.push(year[80] + 366.0);
    let t = RateTable::new(
        vec![
            RateDim::Time { cuts: age },
            RateDim::Factor { levels: 2 },
            RateDim::Time { cuts: year.clone() },
        ],
        r.rates.clone(),
        1.0,
    )
    .unwrap();
    let c = &r.cohorts[1];
    let status = vec![0; c.stop.len()];
    let data = SurvData::builder(&c.stop, &status)
        .start(&c.start)
        .build()
        .unwrap();
    let e = t.expected(&data, &[&c.age0, &c.date0], &[&c.sex]).unwrap();
    let before: f64 = (0..c.stop.len())
        .map(|i| ((c.stop[i].min(year[0] - c.date0[i])) - c.start[i]).max(0.0))
        .sum();
    assert!(before > 0.0);
    assert!(
        (e.offtable - before).abs() <= 1e-9 * before,
        "{} {before}",
        e.offtable
    );
    let extended = table(&r)
        .expected(&data, &[&c.age0, &c.date0], &[&c.sex])
        .unwrap();
    assert!(e.total < extended.total);
}

#[test]
fn cells_from_a_split_agree_on_any_number_of_threads() {
    let r = reference();
    let t = table(&r);
    let c = &r.cohorts[0];
    let status = vec![0; c.stop.len()];
    let data = SurvData::builder(&c.stop, &status)
        .start(&c.start)
        .build()
        .unwrap();
    let age_entry: Vec<f64> = c.age0.iter().zip(&c.start).map(|(a, s)| a + s).collect();
    let year_entry: Vec<f64> = c.date0.iter().zip(&c.start).map(|(a, s)| a + s).collect();
    let mut age_breaks = r.age_cuts.clone();
    age_breaks.push(1e6);
    let mut year_breaks = r.year_cuts.clone();
    year_breaks[0] = -1e6;
    year_breaks.push(1e6);
    let run = || {
        let cells = split_cells(
            &data,
            &[
                TimeScale {
                    entry: &age_entry,
                    breaks: &age_breaks,
                },
                TimeScale {
                    entry: &year_entry,
                    breaks: &year_breaks,
                },
            ],
            &[CellKey {
                codes: &c.sex,
                levels: 2,
            }],
            None,
            &LexisOptions::default(),
        )
        .unwrap();
        t.expected_keyed(
            &[&cells.codes[0], &cells.codes[2], &cells.codes[1]],
            &cells.length,
        )
        .unwrap()
    };
    let k = run();
    assert!(k.offtable.abs() < 1e-300);
    // The split sums pieces in blocks, the walk row by row, so the totals agree
    // to rounding of the sums, not to the bit.
    let want = c.expected;
    assert!(
        (k.total - want).abs() <= 1e-10 * want,
        "{} against {want}",
        k.total
    );
    assert_same_on_threads(|| {
        let k = run();
        let mut v = k.per_cell.clone();
        v.push(k.total);
        v
    });
}

#[test]
fn fingerprint_of_the_expected_counts() {
    let r = reference();
    let t = table(&r);
    let c = &r.cohorts[0];
    let status = vec![0; c.stop.len()];
    let data = SurvData::builder(&c.stop, &status)
        .start(&c.start)
        .build()
        .unwrap();
    let e = t.expected(&data, &[&c.age0, &c.date0], &[&c.sex]).unwrap();
    let hash = Fingerprint::new()
        .f64(e.total)
        .f64s(&e.per_row[..150])
        .f64s(&e.cells[..500])
        .finish();
    eprintln!("ratetable fingerprint {hash:#018x}");
    assert_eq!(hash, 0x11da_ad80_8536_f81c);
}
