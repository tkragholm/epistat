//! The Charlson and Elixhauser maps, hierarchies and weights checked against
//! the R package comorbidity 1.1.0, run as a black box by
//! `tests/data/comorbidity_reference.R` on every ICD-10 category, each with one
//! subdivision digit, and the codes around every code the maps name.
//!
//! The maps are transcribed from Quan et al. 2005, and where the package and
//! the tables disagree the tables are followed and the code is listed here as
//! an exception. On the 29,315 codes of the reference they agree on every
//! flag of both maps, so there is no exception.
//!
//! Flags are integers and weights are integers, so agreement is exact.

use epistat_register::codes::Dictionary;
use epistat_register::comorbidity::{
    ComorbidityMap, Weights, charlson_original, charlson_quan2011, elixhauser_van_walraven,
};
use epistat_testkit::{Fingerprint, assert_same_on_threads, load_json};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize)]
struct Nonzero {
    conditions: Vec<String>,
    code: Vec<String>,
    mask: Vec<u64>,
}

#[derive(Deserialize)]
struct Scored {
    mask: Vec<u64>,
    #[serde(default)]
    charlson: Vec<f64>,
    #[serde(default)]
    quan: Vec<f64>,
    #[serde(default)]
    vw: Vec<f64>,
}

#[derive(Deserialize)]
struct Single {
    charlson: Nonzero,
    elixhauser: Nonzero,
}

#[derive(Deserialize)]
struct Persons {
    codes: Vec<Vec<String>>,
    charlson_raw: Scored,
    charlson: Scored,
    elixhauser_raw: Scored,
    elixhauser: Scored,
}

#[derive(Deserialize)]
struct Reference {
    package_version: String,
    codes: Vec<String>,
    single: Single,
    persons: Persons,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/comorbidity_reference.json"
    ))
}

/// Every letter followed by two digits, and by three digits.
fn alphabet() -> Vec<String> {
    let mut out = Vec::new();
    for letter in 'A'..='Z' {
        for n in 0..100 {
            out.push(format!("{letter}{n:02}"));
        }
        for n in 0..1000 {
            out.push(format!("{letter}{n:03}"));
        }
    }
    out
}

/// Every code of the reference has the package's mask, and the fingerprint
/// of the masks over [`alphabet`].
fn check_masks(map: &ComorbidityMap, nonzero: &Nonzero, codes: &[String], label: &str) -> u64 {
    let keys: Vec<&str> = map.conditions().iter().map(|c| c.key.as_str()).collect();
    assert_eq!(nonzero.conditions, keys, "{label} conditions");
    let expected: HashMap<&str, u64> = nonzero
        .code
        .iter()
        .map(String::as_str)
        .zip(nonzero.mask.iter().copied())
        .collect();
    for code in codes {
        let want = expected.get(code.as_str()).copied().unwrap_or(0);
        assert_eq!(map.mask(code), want, "{label} {code}");
    }
    alphabet()
        .iter()
        .fold(Fingerprint::new(), |fp, code| fp.u64(map.mask(code)))
        .finish()
}

#[test]
fn every_code_of_the_alphabet_has_the_flags_of_the_package() {
    let r = reference();
    assert_eq!(r.package_version, "1.1.0");
    // Every category and every category with one subdivision digit.
    assert!(r.codes.len() >= 26 * 1100);
    let charlson = ComorbidityMap::charlson_quan_icd10();
    let elixhauser = ComorbidityMap::elixhauser_quan_icd10();
    let h1 = check_masks(&charlson, &r.single.charlson, &r.codes, "charlson");
    let h2 = check_masks(&elixhauser, &r.single.elixhauser, &r.codes, "elixhauser");
    assert_eq!(h1, 2_341_647_204_595_256_857, "Charlson mask hash");
    assert_eq!(h2, 9_071_002_199_122_471_612, "Elixhauser mask hash");
}

/// The keys of the conditions of `map` that hold `code`.
fn conditions_of<'a>(map: &'a ComorbidityMap, code: &str) -> Vec<&'a str> {
    map.conditions()
        .iter()
        .filter(|c| c.codes.contains(code))
        .map(|c| c.key.as_str())
        .collect()
}

#[test]
fn range_ends_wildcards_and_short_codes_are_read_as_the_tables_print_them() {
    let charlson = ComorbidityMap::charlson_quan_icd10();
    let elixhauser = ComorbidityMap::elixhauser_quan_icd10();
    for (code, expected) in [
        // I42.5–I42.9 and I42.0 in congestive heart failure; I42 itself is
        // shorter than the items and is in nothing.
        ("I424", &[][..]),
        ("I425", &["chf"][..]),
        ("I429", &["chf"]),
        ("I42", &[]),
        // I43.x is the whole category.
        ("I43", &["chf"]),
        ("I438", &["chf"]),
        // E10.2–E10.5 and E10.7 with chronic complications, E10.6 without.
        ("E101", &["diab"]),
        ("E102", &["diabwc"]),
        ("E105", &["diabwc"]),
        ("E106", &["diab"]),
        ("E107", &["diabwc"]),
        ("E10", &[]),
        // C37.x–C41.x within any malignancy, with nothing between it and
        // C30.x–C34.x; C44 (skin) is left out.
        ("C349", &["canc"]),
        ("C369", &[]),
        ("C37", &["canc"]),
        ("C419", &["canc"]),
        ("C44", &[]),
        ("C779", &["metacanc"]),
    ] {
        assert_eq!(conditions_of(&charlson, code), expected, "Charlson {code}");
    }
    for (code, expected) in [
        ("I425", &["chf"][..]),
        ("I426", &["chf", "alcohol"][..]),
        // D65–D68.x with the wildcard on one end only.
        ("D64", &[]),
        ("D65", &["coag"]),
        ("D689", &["coag"]),
        ("D690", &[]),
        ("D691", &["coag"]),
        // R64, F10 and E52 printed without a decimal are categories.
        ("R64", &["wloss"]),
        ("R649", &["wloss"]),
        ("F10", &["alcohol"]),
        ("F102", &["alcohol"]),
        ("E52", &["alcohol"]),
        ("E102", &["diabc"]),
        ("E108", &["diabc"]),
        ("E109", &["diabunc"]),
        ("E10", &[]),
    ] {
        assert_eq!(
            conditions_of(&elixhauser, code),
            expected,
            "Elixhauser {code}"
        );
    }
}

fn person_masks(map: &ComorbidityMap, codes: &[Vec<String>]) -> Vec<u64> {
    let flat: Vec<&str> = codes.iter().flatten().map(String::as_str).collect();
    let dict = Dictionary::new(&flat).unwrap();
    let per_code = map.classify(dict.distinct()).masks;
    let per_row = dict.gather(&per_code);
    let mut at = 0;
    codes
        .iter()
        .map(|person| {
            let mask = per_row[at..at + person.len()].iter().fold(0, |a, m| a | m);
            at += person.len();
            mask
        })
        .collect()
}

fn check_persons(
    map: &ComorbidityMap,
    codes: &[Vec<String>],
    raw: &Scored,
    hier: &Scored,
    weights: &[(&Weights, &[f64])],
    label: &str,
) {
    let flags = person_masks(map, codes);
    assert_eq!(flags, raw.mask, "{label} raw flags");
    let after: Vec<u64> = flags.iter().map(|&f| map.apply_hierarchy(f)).collect();
    assert_eq!(after, hier.mask, "{label} flags after the hierarchy");
    for (w, expected) in weights {
        let scores: Vec<f64> = after.iter().map(|&f| map.score(f, w).unwrap()).collect();
        assert_eq!(&scores, expected, "{label} score {}", w.name);
    }
}

#[test]
fn persons_have_the_flags_and_scores_of_the_package() {
    let r = reference();
    let p = &r.persons;
    let charlson = ComorbidityMap::charlson_quan_icd10();
    let (original, quan) = (charlson_original(), charlson_quan2011());
    check_persons(
        &charlson,
        &p.codes,
        &p.charlson_raw,
        &p.charlson,
        &[(&original, &p.charlson.charlson), (&quan, &p.charlson.quan)],
        "charlson",
    );
    let elixhauser = ComorbidityMap::elixhauser_quan_icd10();
    let vw = elixhauser_van_walraven();
    check_persons(
        &elixhauser,
        &p.codes,
        &p.elixhauser_raw,
        &p.elixhauser,
        &[(&vw, &p.elixhauser.vw)],
        "elixhauser",
    );
    // The hierarchy changes something in the sample, or the check says nothing.
    assert_ne!(p.charlson_raw.mask, p.charlson.mask);
    assert_ne!(p.elixhauser_raw.mask, p.elixhauser.mask);
}

#[test]
fn scores_do_not_depend_on_the_thread_count() {
    let r = reference();
    let charlson = ComorbidityMap::charlson_quan_icd10();
    let quan = charlson_quan2011();
    assert_same_on_threads(|| {
        person_masks(&charlson, &r.persons.codes)
            .iter()
            .map(|&f| charlson.score(charlson.apply_hierarchy(f), &quan).unwrap())
            .collect()
    });
}
