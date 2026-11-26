//! The Danish Charlson map against `heaven::findCondition` and
//! `heaven::charlsonIndex` given the same list, written out in R from
//! Thygesen et al. 2011 and Christensen et al. 2011
//! (`tests/data/comorbidity_denmark_reference.R`).
//!
//! Flags and weights are integers, so agreement is exact.

use std::collections::HashMap;

use epistat_register::codes::{Dictionary, normalize_any};
use epistat_register::comorbidity::{
    ComorbidityMap, IndexTable, charlson_denmark_weights, flags_at_index,
};
use epistat_register::events::{EventRows, Lookback};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Universe {
    icd10: usize,
    icd8: usize,
}

#[derive(Deserialize)]
struct Masks {
    raw: Vec<String>,
    mask: Vec<u64>,
}

#[derive(Deserialize)]
struct Persons {
    person: Vec<u32>,
    code: Vec<String>,
    day: Vec<i64>,
    index: Vec<i64>,
    flags: Vec<u64>,
    score: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    keys: Vec<String>,
    universe: Universe,
    masks: Masks,
    persons: Persons,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/comorbidity_denmark_reference.json"
    ))
}

/// The raw codes of the R universe: ICD-10 with the register's D, ICD-8 as
/// digits, three and five of them.
fn universe() -> Vec<String> {
    let mut out = Vec::new();
    for width in [2_usize, 3] {
        for number in 0..10_u32.pow(u32::try_from(width).unwrap()) {
            for letter in 'A'..='Z' {
                out.push(format!("D{letter}{number:0width$}"));
            }
        }
    }
    for (width, count) in [(3_usize, 1000_u32), (5, 100_000)] {
        for number in 0..count {
            out.push(format!("{number:0width$}"));
        }
    }
    out
}

#[test]
fn every_code_has_the_conditions_heaven_finds() {
    let r = reference();
    let map = ComorbidityMap::charlson_denmark();
    let keys: Vec<&str> = map.conditions().iter().map(|c| c.key.as_str()).collect();
    assert_eq!(keys, r.keys);
    let codes = universe();
    assert_eq!(codes.len(), r.universe.icd10 + r.universe.icd8);
    let want: HashMap<&str, u64> = r
        .masks
        .raw
        .iter()
        .map(String::as_str)
        .zip(r.masks.mask.iter().copied())
        .collect();
    let mut matched = 0;
    for raw in &codes {
        let (_, normalized) = normalize_any(raw).unwrap();
        let got = map.mask(&normalized);
        assert_eq!(got, want.get(raw.as_str()).copied().unwrap_or(0), "{raw}");
        matched += usize::from(got != 0);
    }
    assert_eq!(matched, r.masks.raw.len());
    // Every condition is reached by some code of each revision.
    for bit in 0..keys.len() {
        for icd8 in [true, false] {
            assert!(
                r.masks
                    .raw
                    .iter()
                    .zip(&r.masks.mask)
                    .any(|(c, m)| m >> bit & 1 == 1 && c.starts_with('D') != icd8),
                "condition {} in {}",
                keys[bit],
                if icd8 { "ICD-8" } else { "ICD-10" }
            );
        }
    }
}

#[test]
fn flags_and_scores_equal_charlson_index() {
    let r = reference();
    let map = ComorbidityMap::charlson_denmark();
    let weights = charlson_denmark_weights();
    let p = &r.persons;
    let n = p.index.len();
    // Rows sorted by person and day, as flags_at_index takes them.
    let mut order: Vec<usize> = (0..p.person.len()).collect();
    order.sort_by_key(|&i| (p.person[i], p.day[i], i));
    let normalized: Vec<String> = order
        .iter()
        .map(|&i| normalize_any(&p.code[i]).unwrap().1)
        .collect();
    let refs: Vec<&str> = normalized.iter().map(String::as_str).collect();
    let dict = Dictionary::new(&refs).unwrap();
    let mask = dict.gather(&map.classify(dict.distinct()).masks);
    let day: Vec<i64> = order.iter().map(|&i| p.day[i]).collect();
    let mut offsets = vec![0_usize; n + 1];
    for &i in &order {
        offsets[p.person[i] as usize + 1] += 1;
    }
    for k in 0..n {
        offsets[k + 1] += offsets[k];
    }
    let rows = EventRows {
        person_offsets: &offsets,
        day: &day,
        mask: &mask,
        contact: None,
    };
    let persons: Vec<u32> = (0..u32::try_from(n).unwrap()).collect();
    let index = IndexTable {
        person: &persons,
        day: &p.index,
        history_start: &vec![i64::MIN / 4; n],
    };
    // heaven keeps the dates d >= index - 5 * 365.25, which for whole days is
    // d >= index - 1826, and the index day itself.
    let found = flags_at_index(&rows, &index, Lookback::Days(1826), true).unwrap();
    for k in 0..n {
        let flags = map.apply_hierarchy(found[k].flags);
        assert_eq!(flags, p.flags[k], "person {k}");
        let score = found[k].score(&map, &weights, false).unwrap().unwrap();
        assert_eq!(score.to_bits(), p.score[k].to_bits(), "person {k}");
    }
    assert!(p.score.iter().any(|&s| s >= 8.0));
    assert!(p.flags.iter().any(|&f| f.count_ones() >= 4));
}
