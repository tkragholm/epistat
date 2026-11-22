//! The Charlson and Elixhauser maps, hierarchies and weights checked against
//! the R package comorbidity 1.1.0 (`tests/data/comorbidity_reference.R`).
//!
//! Flags are integers and weights are integers, so agreement is exact.

use epistat_register::codes::Dictionary;
use epistat_register::comorbidity::{
    ComorbidityMap, Weights, charlson_original, charlson_quan2011, elixhauser_van_walraven,
};
use epistat_register::normalize_diagnosis_code;
use epistat_testkit::{Fingerprint, assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Nonzero {
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
struct Tidy {
    raw: Vec<String>,
    tidy: Vec<String>,
}

#[derive(Deserialize)]
struct Reference {
    package_version: String,
    zero_around: Vec<String>,
    single: Single,
    persons: Persons,
    tidy: Tidy,
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

fn check_masks(
    map: &ComorbidityMap,
    nonzero: &Nonzero,
    zero_around: &[String],
    label: &str,
) -> u64 {
    let mut expected: std::collections::HashMap<&str, u64> = nonzero
        .code
        .iter()
        .map(String::as_str)
        .zip(nonzero.mask.iter().copied())
        .collect();
    let codes = alphabet();
    let mut fp = Fingerprint::new();
    for code in &codes {
        let want = expected.remove(code.as_str()).unwrap_or(0);
        let got = map.mask(code);
        assert_eq!(got, want, "{label} {code}");
        fp = fp.u64(got);
    }
    // The rest of the reference holds codes of other lengths, around prefixes.
    for (code, want) in expected {
        assert_eq!(map.mask(code), want, "{label} {code}");
    }
    for code in zero_around {
        assert_eq!(map.mask(code), 0, "{label} {code}");
    }
    fp.finish()
}

#[test]
fn every_code_of_the_alphabet_has_the_flags_of_the_package() {
    let r = reference();
    assert_eq!(r.package_version, "1.1.0");
    let charlson = ComorbidityMap::charlson_quan_icd10();
    let elixhauser = ComorbidityMap::elixhauser_quan_icd10();
    let h1 = check_masks(&charlson, &r.single.charlson, &r.zero_around, "charlson");
    let h2 = check_masks(
        &elixhauser,
        &r.single.elixhauser,
        &r.zero_around,
        "elixhauser",
    );
    assert_eq!(h1, 2_341_647_204_595_256_857, "Charlson mask hash");
    assert_eq!(h2, 9_071_002_199_122_471_612, "Elixhauser mask hash");
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
fn the_package_tidies_codes_as_normalize_diagnosis_code_does() {
    let r = reference();
    for (raw, tidy) in r.tidy.raw.iter().zip(&r.tidy.tidy) {
        let ours = normalize_diagnosis_code(raw).unwrap().full_code;
        // The package keeps the Danish register prefix, which is not part of
        // ICD-10, where the crate removes it.
        let expected = if raw.eq_ignore_ascii_case("dj45") {
            tidy.strip_prefix('D').unwrap()
        } else {
            tidy
        };
        assert_eq!(ours, expected, "{raw}");
    }
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
