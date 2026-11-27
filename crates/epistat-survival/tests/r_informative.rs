//! Informative matched sets and strata against a count in base R.
//!
//! The reference is `tests/data/informative.json`, written by
//! `tests/data/informative.R`: 150 matched sets and a stratified data set with
//! weights, counted with `tapply`-style loops over the sets. The counts are
//! integers, so they must agree exactly. A conditional logistic fit on the
//! informative sets alone must equal the fit on all sets, which is what the
//! definition claims; the tolerance is the Newton stopping rule of `coxph`,
//! `1e-9` on the coefficients.

use epistat_linalg::Mat;
use epistat_survival::clogit::{ClogitOptions, clogit};
use epistat_survival::informative::{InformativeSets, informative_sets, informative_strata};
use epistat_survival::surv::SurvData;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    clogit: Data,
    cox: Data,
}

#[derive(Deserialize)]
struct Data {
    set: Vec<u32>,
    case: Vec<bool>,
    x: Vec<Vec<f64>>,
    weight: Option<Vec<f64>>,
    time: Option<Vec<f64>>,
    status: Option<Vec<i32>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    labels: Vec<u32>,
    rows: Vec<usize>,
    cases: Vec<usize>,
    varies: Vec<bool>,
    n_sets: usize,
    n_informative_sets: usize,
    n_rows: usize,
    n_informative_rows: usize,
    n_informative_cases: usize,
    by_column: Vec<usize>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/informative.json"
    ))
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn assert_matches(got: &InformativeSets, want: &Expected) {
    assert_eq!(got.n_sets, want.n_sets);
    assert_eq!(got.n_informative_sets, want.n_informative_sets);
    assert_eq!(got.n_rows, want.n_rows);
    assert_eq!(got.n_informative_rows, want.n_informative_rows);
    assert_eq!(got.n_informative_cases, want.n_informative_cases);
    assert_eq!(got.informative_by_column, want.by_column);
    for (i, s) in got.sets.iter().enumerate() {
        assert_eq!(s.set, want.labels[i], "set {i}");
        assert_eq!(s.rows, want.rows[i], "set {i}");
        assert_eq!(s.cases, want.cases[i], "set {i}");
        assert_eq!(s.varies, want.varies[i], "set {i}");
    }
    assert_eq!(got.sets.len(), want.labels.len());
}

#[test]
fn matched_sets_agree_with_the_count_in_r() {
    let d = reference().clogit;
    let x = matrix(&d.x);
    let got = informative_sets(&d.case, &d.set, x.as_ref(), None).unwrap();
    assert_matches(&got, &d.expected);
    assert!(got.n_informative_sets < got.n_sets);
}

#[test]
fn strata_with_weights_agree_with_the_count_in_r() {
    let d = reference().cox;
    let x = matrix(&d.x);
    let weight = d.weight.as_ref().unwrap();
    let got = informative_sets(&d.case, &d.set, x.as_ref(), Some(weight)).unwrap();
    assert_matches(&got, &d.expected);
    let time = d.time.as_ref().unwrap();
    let status = d.status.as_ref().unwrap();
    let data = SurvData::builder(time, status)
        .stratum(&d.set)
        .weight(weight)
        .build()
        .unwrap();
    // The data set numbers the strata by first appearance, in place of the
    // labels given.
    let from_data = informative_strata(&data, x.as_ref()).unwrap();
    assert_eq!(from_data.sets.len(), got.sets.len());
    for (a, b) in from_data.sets.iter().zip(&got.sets) {
        assert_eq!((a.rows, a.cases, a.varies), (b.rows, b.cases, b.varies));
    }
    assert_eq!(from_data.n_informative_sets, got.n_informative_sets);
    assert_eq!(from_data.informative_by_column, got.informative_by_column);
    assert!(got.n_rows < d.set.len(), "rows of weight 0 are left out");
}

#[test]
fn a_fit_on_the_informative_sets_equals_the_fit_on_all() {
    let d = reference().clogit;
    let x = matrix(&d.x);
    let info = informative_sets(&d.case, &d.set, x.as_ref(), None).unwrap();
    let keep: Vec<usize> = (0..d.set.len())
        .filter(|&r| {
            info.sets
                .iter()
                .any(|s| s.set == d.set[r] && s.is_informative())
        })
        .collect();
    assert_eq!(keep.len(), info.n_informative_rows);
    let case: Vec<bool> = keep.iter().map(|&r| d.case[r]).collect();
    let set: Vec<u32> = keep.iter().map(|&r| d.set[r]).collect();
    let sub = Mat::from_fn(keep.len(), 2, |i, j| x[(keep[i], j)]);
    let all = clogit(&d.case, &d.set, x.as_ref(), &ClogitOptions::default()).unwrap();
    let part = clogit(&case, &set, sub.as_ref(), &ClogitOptions::default()).unwrap();
    for (a, b) in all.coefficients.iter().zip(&part.coefficients) {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "{a} {b}");
    }
    for (a, b) in all.se.iter().zip(&part.se) {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert!((a - b).abs() <= 1e-9 * a, "{a} {b}");
    }
    // A set with both outcome levels and a constant covariate adds the log of
    // the reciprocal number of ways to choose its cases, whatever the
    // coefficients, which is the whole difference of the two log likelihoods.
    let constants: f64 = info
        .sets
        .iter()
        .filter(|s| s.has_both_outcomes() && !s.is_informative())
        .map(|s| -ln_choose(s.rows, s.cases))
        .sum();
    let (la, lb) = (
        all.log_partial_likelihood.unwrap(),
        part.log_partial_likelihood.unwrap(),
    );
    assert!(
        (la - lb - constants).abs() <= 1e-9 * la.abs(),
        "{la} {lb} {constants}"
    );
}

fn ln_choose(n: usize, k: usize) -> f64 {
    (1..=k).map(|i| ((n + 1 - i) as f64 / i as f64).ln()).sum()
}

#[test]
fn invalid_input_is_refused() {
    let x = Mat::from_fn(3, 1, |i, _| i as f64);
    let case = [true, false, true];
    let set = [1, 1, 2];
    assert!(informative_sets(&case, &set, x.as_ref(), None).is_ok());
    assert!(informative_sets(&case[..2], &set, x.as_ref(), None).is_err());
    assert!(informative_sets(&case, &set, Mat::<f64>::zeros(3, 0).as_ref(), None).is_err());
    let bad = Mat::from_fn(3, 1, |i, _| if i == 1 { f64::NAN } else { 0.0 });
    assert!(informative_sets(&case, &set, bad.as_ref(), None).is_err());
    assert!(informative_sets(&case, &set, x.as_ref(), Some(&[1.0, -1.0, 1.0])).is_err());
    // A set whose rows are all cases, and one whose covariate is constant.
    let got = informative_sets(
        &[true, true, false, true],
        &[1, 1, 2, 2],
        Mat::from_fn(4, 1, |_, _| 1.0).as_ref(),
        None,
    )
    .unwrap();
    assert_eq!(got.n_informative_sets, 0);
    assert!(!got.sets[0].varies && !got.sets[0].has_both_outcomes());
}
