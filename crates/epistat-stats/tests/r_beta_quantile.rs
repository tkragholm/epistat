//! `beta_quantile` against R's `qbeta` on a grid of 1,210 cases.
//!
//! The reference is `tests/data/beta_quantile.json`, written by
//! `tests/data/beta_quantile.R`: shapes from 1e-3 to 1e4 and probabilities
//! from 1e-12 to 1 - 1e-12.

use epistat_stats::student::{beta_quantile, regularised_beta};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    p: f64,
    a: f64,
    b: f64,
    q: f64,
    warned: bool,
}

/// The deviation of `got` from R's `want`, in units of the smaller of `x` and
/// `1 - x`, after allowing two units in the last place of a result near 1,
/// where the rounding of `x` is all the information there is about `1 - x`.
fn deviation(got: f64, want: f64) -> f64 {
    ((got - want).abs() - 2.0 * f64::EPSILON).max(0.0) / want.min(1.0 - want).max(f64::EPSILON)
}

/// Shapes up to 1000: the incomplete beta function agrees with R to about
/// `1e-13` in the tail it computes, and the quantile inherits that divided by
/// the slope of `ln F` in `ln x` (or `ln y`), which is below 1 only for a tail
/// `1 - F` formed as a complement, so the bound is `1e-11`. Shapes of 1e4 have
/// `ln B` of size 1e5 and a log-gamma difference that loses `eps` times that,
/// so the bound there is `1e-8`.
fn bound(c: &Case) -> f64 {
    if c.a.max(c.b) > 1000.0 { 1e-8 } else { 1e-11 }
}

#[test]
fn grid() {
    let r: Reference = load_json(format!(
        "{}/tests/data/beta_quantile.json",
        env!("CARGO_MANIFEST_DIR")
    ));
    let mut compared = 0;
    let mut worst = 0.0_f64;
    let mut worst_label = String::new();
    let mut warned = 0;
    for c in &r.cases {
        let got = beta_quantile(c.p, c.a, c.b);
        let label = format!(
            "p {:e} a {:e} b {:e}: ours {got:?}, R {:e}",
            c.p, c.a, c.b, c.q
        );
        let got = got.expect(&label);
        if c.warned {
            // R warned that it did not reach full precision, and it did not:
            // the quantile is below the smallest positive number or its value
            // of 1 - x is below `eps`, where R's answer is off by an ulp or
            // more. Ours must still be a valid quantile.
            warned += 1;
            assert!((0.0..=1.0).contains(&got), "{label}");
            continue;
        }
        let ratio = deviation(got, c.q) / bound(c);
        if ratio > worst {
            worst = ratio;
            worst_label = label;
        }
        compared += 1;
    }
    eprintln!("compared {compared}, R warned on {warned}");
    eprintln!("worst {worst:.3} of its bound at {worst_label}");
    assert!(worst <= 1.0, "{worst_label}");
}

#[test]
fn the_quantile_inverts_the_distribution_function() {
    for (a, b) in [(0.3, 0.7), (2.0, 5.0), (40.0, 3.0), (0.2, 50.0)] {
        for p in [1e-9, 0.01, 0.4, 0.5, 0.8, 0.99] {
            let x = beta_quantile(p, a, b).unwrap();
            let back = regularised_beta(a, b, x, 1.0 - x);
            assert!(
                (back - p).abs() <= 1e-11 * p.min(1.0 - p).max(1e-300),
                "{a} {b} {p}"
            );
        }
    }
}

#[test]
fn edges_and_invalid_input() {
    assert_eq!(beta_quantile(0.0, 2.0, 3.0), Some(0.0));
    assert_eq!(beta_quantile(1.0, 2.0, 3.0), Some(1.0));
    for (p, a, b) in [
        (-0.1, 1.0, 1.0),
        (1.1, 1.0, 1.0),
        (f64::NAN, 1.0, 1.0),
        (0.5, 0.0, 1.0),
        (0.5, 1.0, -1.0),
        (0.5, f64::INFINITY, 1.0),
        (0.5, 1.0, f64::NAN),
    ] {
        assert_eq!(beta_quantile(p, a, b), None, "{p} {a} {b}");
    }
}
