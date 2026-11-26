//! The exact interval of a rate ratio against `stats::poisson.test` in R.
//!
//! The reference is `tests/data/rate_ratio_exact.json`, written by
//! `tests/data/rate_ratio_exact.R`: 12 pairs of counts, 5 pairs of person-times
//! and 4 levels, with counts from 0 to 10^5.
//!
//! Tolerance. The limits are Clopper-Pearson limits, quantiles of a beta
//! distribution that agree with `qbeta` to `1e-11` for shapes up to 1000 (see
//! `r_beta_quantile.rs`) and to `1e-8` above, then scaled by `p / (1 - p)`,
//! which multiplies the relative error by `1 / (1 - p)`. With `p` below 0.99
//! in the grid that is a factor 100. The bound is therefore `1e-9` for counts up
//! to 1000 and `1e-6` above. The estimate and the p-value are the same
//! computations as in `r_rates.rs`, with bounds of `1e-13` and `1e-11`, the
//! p-value `1e-9` above 1000 counts, where it is a sum of up to 10^5 point
//! probabilities, each accurate to `1e-14`. The
//! test prints the largest deviation in units of its bound.

use epistat_core::Level;
use epistat_epi::rates::rate_ratio_exact;
use epistat_testkit::load_json;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    e1: f64,
    e0: f64,
    t1: f64,
    t0: f64,
    level: f64,
    lower: f64,
    upper: Value,
    estimate: Value,
    p: f64,
}

fn number(v: &Value) -> Option<f64> {
    match v {
        Value::String(s) => {
            assert_eq!(s, "Inf");
            None
        }
        other => other.as_f64(),
    }
}

fn ratio(worst: &mut f64, got: f64, want: f64, bound: f64) {
    *worst = worst.max((got - want).abs() / (bound * want.abs().max(1e-300)));
}

#[test]
fn agrees_with_poisson_test() {
    let r: Reference = load_json(format!(
        "{}/tests/data/rate_ratio_exact.json",
        env!("CARGO_MANIFEST_DIR")
    ));
    let mut worst = 0.0_f64;
    for c in &r.cases {
        let label = format!("{} {} {} {} {}", c.e1, c.e0, c.t1, c.t0, c.level);
        let level = Level::confidence(c.level).unwrap();
        let got = rate_ratio_exact(c.e1, c.t1, c.e0, c.t0, level)
            .unwrap()
            .unwrap();
        let limit_bound = if c.e1.max(c.e0) > 1000.0 { 1e-6 } else { 1e-9 };
        if c.lower == 0.0 {
            assert_eq!(got.lower(), Some(0.0), "{label}");
        } else {
            ratio(&mut worst, got.lower().unwrap(), c.lower, limit_bound);
        }
        match number(&c.upper) {
            Some(want) => ratio(&mut worst, got.upper().unwrap(), want, limit_bound),
            None => assert_eq!(got.upper(), None, "{label}"),
        }
        match number(&c.estimate) {
            Some(want) => ratio(&mut worst, got.estimate, want, 1e-13),
            None => assert_eq!(got.estimate, f64::INFINITY, "{label}"),
        }
        let p_bound = if c.e1.max(c.e0) > 1000.0 { 1e-9 } else { 1e-11 };
        ratio(&mut worst, got.p_value.unwrap(), c.p, p_bound);
    }
    eprintln!("worst deviation {worst:.3} of its bound");
    assert!(worst <= 1.0, "{worst}");
}

#[test]
fn undefined_and_invalid_input() {
    let level = Level::default();
    assert_eq!(rate_ratio_exact(0.0, 1.0, 0.0, 1.0, level).unwrap(), None);
    assert_eq!(rate_ratio_exact(1.0, 0.0, 2.0, 1.0, level).unwrap(), None);
    assert!(rate_ratio_exact(1.5, 1.0, 2.0, 1.0, level).is_err());
    assert!(rate_ratio_exact(-1.0, 1.0, 2.0, 1.0, level).is_err());
}
