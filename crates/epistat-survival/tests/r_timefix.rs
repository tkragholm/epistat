//! The tie policy checked against `survival::aeqSurv`, bit for bit. The
//! outputs are values taken from the input, so the two must agree exactly.

use epistat_survival::timefix::{TimeFix, fix_times};
use epistat_testkit::{assert_same_on_pools, bits, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    r_version: String,
    survival_version: String,
    cases: Vec<Case>,
    infinite: Infinite,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    tolerance: f64,
    start: Option<Vec<f64>>,
    stop: Vec<f64>,
    error: Option<String>,
    out_start: Option<Vec<f64>>,
    out_stop: Option<Vec<f64>>,
    changed: Option<bool>,
}

#[derive(Deserialize)]
struct Infinite {
    right_censored_output: Vec<String>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/timefix.json"
    ))
}

#[test]
fn every_case_matches_aeqsurv_bit_for_bit() {
    let reference = reference();
    assert!(reference.r_version.starts_with("R version 4.6"));
    assert!(reference.survival_version.starts_with("3.8"));
    let (mut changed, mut refused) = (0, 0);
    for case in &reference.cases {
        let policy = TimeFix::Aeq {
            tolerance: case.tolerance,
        };
        let got = fix_times(case.start.as_deref(), &case.stop, policy);
        if case.error.is_some() {
            assert!(
                got.unwrap_err().to_string().contains("effective length 0"),
                "{}",
                case.name
            );
            refused += 1;
            continue;
        }
        let expected_stop = case.out_stop.as_ref().unwrap();
        if case.changed == Some(false) {
            assert!(got.unwrap().is_none(), "{}", case.name);
            assert_eq!(bits(expected_stop), bits(&case.stop), "{}", case.name);
            continue;
        }
        changed += 1;
        let fixed = got
            .unwrap()
            .unwrap_or_else(|| panic!("{}: no tie", case.name));
        assert_eq!(bits(&fixed.stop), bits(expected_stop), "{}", case.name);
        match (&fixed.start, &case.out_start) {
            (Some(got), Some(expected)) => {
                assert_eq!(bits(got), bits(expected), "{}", case.name);
            }
            (None, None) => {}
            _ => panic!("{}: start present on one side only", case.name),
        }
    }
    assert!(
        changed >= 12 && refused == 2,
        "{changed} changed, {refused} refused"
    );
}

#[test]
fn the_default_tolerance_is_the_one_of_aeqsurv() {
    let reference = reference();
    let TimeFix::Aeq { tolerance } = TimeFix::default() else {
        panic!("default is not Aeq");
    };
    let default_cases = reference
        .cases
        .iter()
        .filter(|c| bits(&[c.tolerance]) == bits(&[tolerance]))
        .count();
    assert!(default_cases >= 15);
}

#[test]
fn infinite_times_are_refused_where_r_maps_them_to_a_finite_value() {
    let reference = reference();
    // R's right-censored output replaces the infinite time by the largest
    // finite one.
    assert_eq!(reference.infinite.right_censored_output[2], "3");
    let stop = [1.0, 1.0 + 1e-12, f64::INFINITY, 3.0];
    assert!(fix_times(None, &stop, TimeFix::default()).is_err());
    assert!(fix_times(Some(&[f64::NEG_INFINITY]), &[1.0], TimeFix::default()).is_err());
}

#[test]
fn the_result_does_not_depend_on_the_number_of_threads() {
    // Noisy copies of day numbers, as intervals.
    let n = 200_000_usize;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        #[expect(clippy::cast_precision_loss, reason = "a unit-interval draw")]
        let u = (state >> 11) as f64 / (1_u64 << 53) as f64;
        u
    };
    let mut start = Vec::with_capacity(n);
    let mut stop = Vec::with_capacity(n);
    for _ in 0..n {
        let a = (next() * 20_000.0).floor();
        let b = a + 1.0 + (next() * 3_000.0).floor();
        start.push(a + a * (next() - 0.5) * 1e-12);
        stop.push(b + b * (next() - 0.5) * 1e-12);
    }
    assert_same_on_pools(&[1, 2, 3, 8, 1], || {
        let fixed = fix_times(Some(&start), &stop, TimeFix::default())
            .unwrap()
            .expect("noisy values tie");
        let mut out = fixed.start.unwrap();
        out.extend(fixed.stop);
        out
    });
}
