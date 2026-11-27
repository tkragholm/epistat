//! `dist::beta::regularised` against mpmath at 50 digits.
//!
//! The reference is `tests/data/beta_mpmath.json`, written by
//! `tests/data/beta_mpmath.py`: shapes from 1e-3 to 1e7 and `x` from the far
//! tails to the centre of each distribution, with both `I_x(a, b)` and its
//! complement `I_y(b, a)`.

use epistat_stats::dist::beta::regularised;
use epistat_stats::dist::gamma::ln_gamma;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    a: f64,
    b: f64,
    x: f64,
    y: f64,
    value: f64,
    complement: f64,
}

const EPS: f64 = f64::EPSILON;

/// The shape from which the function forms `x^a y^b / B(a, b)` about the
/// mean (`CENTRED_FROM` in `dist/beta.rs`).
const CENTRED_FROM: f64 = 10.0;

/// `z - ln(1 + z)`, accurate enough for a bound.
fn gap(z: f64) -> f64 {
    z - z.ln_1p()
}

/// The relative error allowed in the tail that the function computes
/// directly, `x^a y^b / (a B(a, b))` times the continued fraction.
///
/// The factor is an exponential, so its relative error is the absolute error
/// of the exponent. Below [`CENTRED_FROM`] the exponent is
/// `a ln x + b ln y - ln B(a, b)`, and each term carries a relative error of a
/// few eps: `ln_gamma` about `1e-15`, which is below `5 eps`, and the products
/// one rounding each, so the coefficient is 8 on the sum of their magnitudes.
/// From a shape of 1000 on, `ln B(a, b)` comes from a Stirling difference
/// whose terms are `ln Gamma(small)` and about `small ln(large)`.
///
/// From [`CENTRED_FROM`] on, the exponent is
/// `-[a g(u) + b g(v)]` plus Stirling corrections below `1 / 60`, with
/// `u = (a + b) e / a`, `v = -(a + b) e / b` and `(a + b) e = b x - a y`
/// formed with an absolute error of `eps (a + b) max(x, y)`, which is
/// `eps (a |u| + b |v| + a + b)` at most. The derivative of `a g(u)` in `u` is
/// `a u / (1 + u)`, of size `a` at most where `u` is not small, so the error
/// of the formed `u` and `v` enters as `eps (a |u| + b |v|)` up to a factor 2,
/// and `g` itself is formed to a few eps relative. The coefficient is 8 on
/// `a g(u) + b g(v) + a |u| + b |v| + 1`.
///
/// The continued fraction takes of the order of `sqrt(a + b)` terms near the
/// switching point `(a + 1) / (a + b + 2)`, each step with a few roundings in
/// the product of its ratios, so `64 eps + 4 sqrt(a + b) eps`.
fn direct_bound(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let (small, large) = if a < b { (a, b) } else { (b, a) };
    let fraction = 64.0 * EPS + 4.0 * (a + b).sqrt() * EPS;
    let terms = if small >= CENTRED_FROM {
        let s = a + b;
        let scaled = if x <= y { s * x - a } else { b - s * y };
        let (u, v) = (scaled / a, -scaled / b);
        a * gap(u) + b * gap(v) + a * u.abs() + b * v.abs() + 1.0
    } else {
        let (ln_x, ln_y) = if x <= y {
            (x.ln(), (-x).ln_1p())
        } else {
            ((-y).ln_1p(), y.ln())
        };
        let ln_beta = if large >= 1000.0 {
            ln_gamma(small).abs() + small * large.ln() + small
        } else {
            ln_gamma(a).abs() + ln_gamma(b).abs() + ln_gamma(a + b).abs()
        };
        (a * ln_x).abs() + (b * ln_y).abs() + ln_beta
    };
    8.0 * EPS * terms + fraction
}

/// Whether `regularised(a, b, x, y)` evaluates `I_x(a, b)` by its continued
/// fraction (rather than as `1 - I_y(b, a)`).
fn direct(a: f64, b: f64, x: f64) -> bool {
    x < (a + 1.0) / (a + b + 2.0)
}

/// The deviation of `regularised(a, b, x, y)` from `want`, in units of its
/// bound. A directly computed tail is held to [`direct_bound`] relative; a
/// tail formed as `1 - other` to that bound times `other`, plus the rounding
/// of the subtraction.
fn deviation(a: f64, b: f64, x: f64, y: f64, want: f64, other: f64) -> f64 {
    let got = regularised(a, b, x, y);
    let bound = direct_bound(a, b, x, y);
    if direct(a, b, x) {
        ((got - want) / want).abs() / (bound + 2.0 * EPS)
    } else {
        (got - want).abs() / (bound * other + 2.0 * EPS * want)
    }
}

#[test]
fn grid() {
    let r: Reference = load_json(format!(
        "{}/tests/data/beta_mpmath.json",
        env!("CARGO_MANIFEST_DIR")
    ));
    assert!(r.cases.len() > 1000);
    let mut worst = 0.0_f64;
    let mut worst_label = String::new();
    for c in &r.cases {
        for (name, units) in [
            (
                "I_x(a, b)",
                deviation(c.a, c.b, c.x, c.y, c.value, c.complement),
            ),
            (
                "I_y(b, a)",
                deviation(c.b, c.a, c.y, c.x, c.complement, c.value),
            ),
        ] {
            let label = format!("{name} at a = {}, b = {}, x = {:e}", c.a, c.b, c.x);
            assert!(units <= 1.0, "{label}: {units:.3} of the bound");
            if units > worst {
                worst = units;
                worst_label = label;
            }
        }
    }
    println!("worst deviation from mpmath, in units of its bound: {worst:.3} ({worst_label})");
}
