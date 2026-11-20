//! The structural checks compared with `survival::survcheck`: every flag with
//! its rows and subjects, every transition count and the resolved state of
//! every row, exactly.

use epistat_survival::SurvData;
use epistat_survival::survcheck::{Flagged, survcheck};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    r_version: String,
    survival_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Flag {
    count: usize,
    rows: Vec<u32>,
    ids: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    k: u32,
    id: Vec<f64>,
    stop: Vec<f64>,
    start: Option<Vec<f64>>,
    status: Vec<i32>,
    istate: Option<Vec<u32>>,
    overlap: Flag,
    gap: Flag,
    jump: Flag,
    teleport: Flag,
    from: Vec<u32>,
    to: Vec<u32>,
    count: Vec<usize>,
    resolved: Vec<u32>,
}

/// The id column as `u32` codes: the rank of each given id, so the codes are
/// not 0..k in the order of appearance but still preserve the id order.
fn codes(id: &[f64]) -> (Vec<u32>, Vec<f64>) {
    let mut sorted = id.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    let codes = id
        .iter()
        .map(|v| {
            u32::try_from(
                sorted
                    .iter()
                    .position(|s| s.to_bits() == v.to_bits())
                    .unwrap(),
            )
            .unwrap()
        })
        .collect();
    (codes, sorted)
}

fn assert_flag(case: &str, name: &str, got: &Flagged, expected: &Flag, sorted_ids: &[f64]) {
    assert_eq!(got.count(), expected.count, "{case}: {name} count");
    // R counts rows from 1.
    let rows: Vec<u32> = expected.rows.iter().map(|r| r - 1).collect();
    assert_eq!(got.rows, rows, "{case}: {name} rows");
    // R lists the ids in order of first appearance, this crate by code.
    let mut ids: Vec<u32> = expected
        .ids
        .iter()
        .map(|v| {
            u32::try_from(
                sorted_ids
                    .iter()
                    .position(|s| s.to_bits() == v.to_bits())
                    .unwrap(),
            )
            .unwrap()
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(got.ids, ids, "{case}: {name} ids");
}

#[test]
fn every_case_matches_survcheck_exactly() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/survcheck.json"
    ));
    assert!(reference.r_version.starts_with("R version 4.6"));
    assert!(reference.survival_version.starts_with("3.8"));
    let mut flagged = [0_usize; 4];
    for case in &reference.cases {
        let (id, sorted_ids) = codes(&case.id);
        let mut builder = SurvData::builder(&case.stop, &case.status)
            .n_codes(case.k)
            .id(&id);
        if let Some(start) = &case.start {
            builder = builder.start(start);
        }
        if let Some(istate) = &case.istate {
            builder = builder.istate(istate);
        }
        let data = builder.build().unwrap();
        let got = survcheck(&data).unwrap();
        let name = case.name.as_str();
        assert_flag(name, "overlap", &got.overlap, &case.overlap, &sorted_ids);
        assert_flag(name, "gap", &got.gap, &case.gap, &sorted_ids);
        assert_flag(name, "jump", &got.jump, &case.jump, &sorted_ids);
        assert_flag(name, "teleport", &got.teleport, &case.teleport, &sorted_ids);
        for (total, flag) in
            flagged
                .iter_mut()
                .zip([&got.overlap, &got.gap, &got.jump, &got.teleport])
        {
            *total += flag.count();
        }
        let mut expected: Vec<(u32, u32, usize)> = case
            .from
            .iter()
            .zip(&case.to)
            .zip(&case.count)
            .map(|((&from, &to), &count)| (from, to, count))
            .collect();
        expected.sort_unstable();
        let found: Vec<(u32, u32, usize)> = got
            .transitions
            .iter()
            .map(|t| (t.from, t.to, t.count))
            .collect();
        assert_eq!(found, expected, "{name}: transitions");
        assert_eq!(got.istate, case.resolved, "{name}: resolved istate");
    }
    // Every flag is exercised by the reference.
    assert!(flagged.iter().all(|&count| count > 0), "{flagged:?}");
}
