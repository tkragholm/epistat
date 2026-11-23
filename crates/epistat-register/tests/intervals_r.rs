//! The interval algebra against logical day arrays in base R
//! (`tests/data/intervals_reference.R`). The results are integers, so
//! agreement is exact.

use epistat_register::intervals::{SpanSet, clip, complement_within, intersect, subtract};
use epistat_testkit::{Fingerprint, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Flat {
    start: Vec<i64>,
    end: Vec<i64>,
    n: Vec<usize>,
}

#[derive(Deserialize)]
struct Reference {
    days: i64,
    lo: Vec<i64>,
    hi: Vec<i64>,
    a: Flat,
    b: Flat,
    intersect: Flat,
    subtract: Flat,
    complement: Flat,
    clip: Flat,
}

fn set(f: &Flat) -> SpanSet {
    let mut offsets = vec![0_usize];
    for k in &f.n {
        offsets.push(offsets.last().unwrap() + k);
    }
    SpanSet::new(f.start.clone(), f.end.clone(), offsets).unwrap()
}

fn same(got: &SpanSet, want: &Flat, label: &str, fp: &mut Fingerprint) {
    assert_eq!(got.start, want.start, "{label} starts");
    assert_eq!(got.end, want.end, "{label} ends");
    let n: Vec<usize> = got.key_offsets.windows(2).map(|w| w[1] - w[0]).collect();
    assert_eq!(n, want.n, "{label} per key");
    let mut f = std::mem::take(fp);
    for (s, e) in got.start.iter().zip(&got.end) {
        f = f.u64(s.cast_unsigned()).u64(e.cast_unsigned());
    }
    *fp = f;
}

#[test]
fn every_operation_agrees_with_day_arrays() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/intervals_reference.json"
    ));
    let (a, b) = (set(&r.a), set(&r.b));
    let mut fp = Fingerprint::new();
    same(
        &intersect(&a, &b).unwrap(),
        &r.intersect,
        "intersect",
        &mut fp,
    );
    same(&subtract(&a, &b).unwrap(), &r.subtract, "subtract", &mut fp);
    let n = a.n_keys();
    let zero = vec![0_i64; n];
    let end = vec![r.days; n];
    same(
        &complement_within(&zero, &end, &a).unwrap(),
        &r.complement,
        "complement",
        &mut fp,
    );
    same(&clip(&a, &r.lo, &r.hi).unwrap(), &r.clip, "clip", &mut fp);
    assert!(a.len() > 3000);
    assert_eq!(fp.finish(), 11_974_849_007_494_031_700, "golden hash");
}
