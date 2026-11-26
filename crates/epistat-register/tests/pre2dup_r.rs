//! Drug use periods against PRE2DUP written out in base R from Tanskanen et
//! al. 2015 (`tests/data/pre2dup_reference.R`), with the readings of the paper
//! that the module documents.
//!
//! The starts and purchase counts are integers and compared exactly. The ends
//! and doses are computed with the same operations in both, but R sums the
//! hospital days and the DDD with `sum`, which adds in extended precision on
//! some platforms, so they are compared to a relative `1e-13`: a handful of
//! additions of numbers below `10^4`, each off by at most half an ulp,
//! `10 * eps` relative.

use epistat_register::exposure::Pauses;
use epistat_register::pre2dup::{
    GlobalParameters, GroupParameters, Purchases, drug_use_periods,
};
use epistat_testkit::{Tolerance, assert_close, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Group {
    lower: f64,
    common: f64,
    upper: f64,
    min_period: f64,
    max_refill: f64,
}

#[derive(Deserialize)]
struct Rows {
    person: Vec<u32>,
    day: Vec<i64>,
    ddd: Vec<f64>,
    packages: Vec<f64>,
    refill: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct Stays {
    person: Vec<u32>,
    start: Vec<i64>,
    end: Vec<i64>,
}

#[derive(Deserialize)]
struct Periods {
    person: Vec<u32>,
    start: Vec<i64>,
    end: Vec<f64>,
    n: Vec<u32>,
    ddd: Vec<f64>,
    hd: Vec<f64>,
    per_day: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    group: Group,
    rows: Rows,
    stays: Stays,
    periods: Periods,
}

fn offsets(person: &[u32], persons: usize) -> Vec<usize> {
    let mut o = vec![0_usize; persons + 1];
    for &p in person {
        o[p as usize + 1] += 1;
    }
    for p in 0..persons {
        o[p + 1] += o[p];
    }
    o
}

#[test]
fn periods_agree_with_the_method_written_in_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pre2dup_reference.json"
    ));
    let persons = 150;
    let o = offsets(&r.rows.person, persons);
    let so = offsets(&r.stays.person, persons);
    let n = r.rows.day.len();
    let ddd: Vec<Option<f64>> = r.rows.ddd.iter().map(|&d| Some(d)).collect();
    let p = Purchases {
        person_offsets: &o,
        group: &vec![Some(0); n],
        day: &r.rows.day,
        ddd: &ddd,
        packages: &r.rows.packages,
        typical_refill_days: &r.rows.refill,
    };
    let stays = Pauses {
        person_offsets: &so,
        start: &r.stays.start,
        end: &r.stays.end,
    };
    let g = GroupParameters {
        lower_ddd_per_day: r.group.lower,
        common_ddd_per_day: r.group.common,
        upper_ddd_per_day: r.group.upper,
        min_period_days: r.group.min_period,
        max_refill_days: r.group.max_refill,
    };
    let got = drug_use_periods(&p, &[g], &GlobalParameters::TANSKANEN_2015, Some(&stays))
        .unwrap();
    assert_eq!(got.person, r.periods.person);
    assert_eq!(got.start, r.periods.start);
    assert_eq!(got.n_purchases, r.periods.n);
    let tol = Tolerance::exact().rel(1e-13);
    assert_close(&got.end, &r.periods.end, tol);
    let total: Vec<f64> = got.ddd.iter().map(|d| d.unwrap()).collect();
    assert_close(&total, &r.periods.ddd, tol);
    assert_close(&got.hospital_days, &r.periods.hd, tol);
    let per_day: Vec<f64> = got.ddd_per_day.iter().map(|d| d.unwrap()).collect();
    assert_close(&per_day, &r.periods.per_day, tol);
    // The data reach the stockpiling test, the lower limit, hospital days
    // and lone purchases.
    assert!(r.periods.hd.iter().any(|&h| h > 0.0));
    assert!(r.periods.n.iter().any(|&k| k == 1));
    assert!(r.periods.n.iter().any(|&k| k >= 8));
}
