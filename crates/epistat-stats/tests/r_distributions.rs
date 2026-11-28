//! The distribution objects of `dist` against R's d, p and q functions.
//!
//! The reference is `tests/data/distributions.json`, written by
//! `tests/data/distributions.R`: for the normal, t, chi-square, F, gamma,
//! beta, Poisson and binomial distributions at several parameters, the
//! density and its log, both tails and their logs at a set of points, and
//! the quantiles of both tails at probabilities from `1e-300` to
//! `1 - 1e-10`. The objects call the free functions, so this checks those
//! too.

#![expect(
    clippy::float_cmp,
    reason = "equal values, infinities included, are compared exactly"
)]

use epistat_stats::dist::{Beta, Binomial, ChiSquared, FisherF, Gamma, Normal, Poisson, StudentT};
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::HashMap;

/// A number as jsonlite writes it: a JSON number, or `"Inf"`, `"-Inf"` and
/// `"NaN"` as strings.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
enum Number {
    Finite(f64),
    Text(String),
}

impl Number {
    fn value(&self) -> f64 {
        match self {
            Self::Finite(v) => *v,
            Self::Text(text) => match text.as_str() {
                "Inf" => f64::INFINITY,
                "-Inf" => f64::NEG_INFINITY,
                other => {
                    assert!(other == "NaN" || other == "NA", "unexpected {other}");
                    f64::NAN
                }
            },
        }
    }
}

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    params: HashMap<String, f64>,
    points: Vec<Point>,
    quantiles: Vec<Quantile>,
}

#[derive(Deserialize)]
struct Point {
    x: Number,
    d: Number,
    log_d: Number,
    lower: Number,
    upper: Number,
    log_lower: Option<Number>,
    log_upper: Option<Number>,
}

#[derive(Deserialize)]
struct Quantile {
    p: Number,
    lower: Number,
    upper: Number,
}

/// The methods of one distribution, as closures over the object.
struct Methods {
    pdf: Box<dyn Fn(f64) -> f64>,
    ln_pdf: Box<dyn Fn(f64) -> f64>,
    cdf: Box<dyn Fn(f64) -> f64>,
    sf: Box<dyn Fn(f64) -> f64>,
    ln_cdf: Box<dyn Fn(f64) -> f64>,
    ln_sf: Box<dyn Fn(f64) -> f64>,
    quantile: Box<dyn Fn(f64) -> f64>,
    isf: Box<dyn Fn(f64) -> f64>,
    discrete: bool,
}

macro_rules! methods {
    ($d:expr, $density:ident, $ln_density:ident, $discrete:expr) => {{
        let d = $d;
        Methods {
            pdf: Box::new(move |x| d.$density(x)),
            ln_pdf: Box::new(move |x| d.$ln_density(x)),
            cdf: Box::new(move |x| d.cdf(x)),
            sf: Box::new(move |x| d.sf(x)),
            ln_cdf: Box::new(move |x| d.ln_cdf(x)),
            ln_sf: Box::new(move |x| d.ln_sf(x)),
            quantile: Box::new(move |p| d.quantile(p)),
            isf: Box::new(move |p| d.isf(p)),
            discrete: $discrete,
        }
    }};
}

fn build(case: &Case) -> Methods {
    let p = |name: &str| case.params[name];
    match case.name.as_str() {
        "normal" => methods!(Normal::new(p("mean"), p("sd")).unwrap(), pdf, ln_pdf, false),
        "t" => methods!(StudentT::new(p("df")).unwrap(), pdf, ln_pdf, false),
        "chisq" => methods!(ChiSquared::new(p("df")).unwrap(), pdf, ln_pdf, false),
        "f" => methods!(
            FisherF::new(p("df1"), p("df2")).unwrap(),
            pdf,
            ln_pdf,
            false
        ),
        "gamma" => methods!(
            Gamma::with_rate(p("shape"), p("rate")).unwrap(),
            pdf,
            ln_pdf,
            false
        ),
        "beta" => methods!(
            Beta::new(p("shape1"), p("shape2")).unwrap(),
            pdf,
            ln_pdf,
            false
        ),
        "poisson" => methods!(Poisson::new(p("lambda")).unwrap(), pmf, ln_pmf, true),
        "binomial" => {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the sizes are small whole numbers"
            )]
            let trials = p("size") as u64;
            methods!(Binomial::new(trials, p("prob")).unwrap(), pmf, ln_pmf, true)
        }
        other => panic!("unknown distribution {other}"),
    }
}

/// The relative error of `got` against R's `want`, with values below the
/// smallest normal number compared absolutely and equal infinities as 0.
fn relative(got: f64, want: f64) -> f64 {
    if got == want || (got.is_nan() && want.is_nan()) {
        return 0.0;
    }
    if !want.is_finite() || !got.is_finite() {
        return f64::INFINITY;
    }
    (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
}

/// The error of a log against R's, absolute below 1 in magnitude.
fn logarithmic(got: f64, want: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    if !want.is_finite() || !got.is_finite() {
        return f64::INFINITY;
    }
    (got - want).abs() / want.abs().max(1.0)
}

/// The largest error found for one kind of value, with where it was.
#[derive(Default)]
struct Worst {
    error: f64,
    at: String,
    compared: usize,
}

impl Worst {
    fn record(&mut self, error: f64, at: impl FnOnce() -> String) {
        self.compared += 1;
        if error > self.error || error.is_nan() {
            self.error = error;
            self.at = at();
        }
    }

    fn check(&self, what: &str, bound: f64) {
        assert!(self.compared > 0, "{what}: nothing compared");
        assert!(
            self.error <= bound,
            "{what}: error {:e} above {bound:e} at {}",
            self.error,
            self.at
        );
    }
}

fn reference() -> Reference {
    load_json(format!(
        "{}/tests/data/distributions.json",
        env!("CARGO_MANIFEST_DIR")
    ))
}

/// Where the tail's log is below this, the tail underflows an `f64` and
/// only the forms computed on the log scale (the normal's) can match R.
const UNDERFLOW_LOG: f64 = -700.0;

/// The errors of one point: density, both tails and their logs.
fn compare_point(case: &Case, m: &Methods, point: &Point, worst: &mut [Worst; 7]) {
    let [density, ln_density, tails, ln_tails, _, _, large_df2] = worst;
    let x = point.x.value();
    let label = |what: &str| format!("{} {:?} {what} at {x:e}", case.name, case.params);
    let (d, log_d) = (point.d.value(), point.log_d.value());
    density.record(relative((m.pdf)(x), d), || label("density"));
    if log_d.is_finite() {
        ln_density.record(logarithmic((m.ln_pdf)(x), log_d), || label("log density"));
    }
    // From 1e7 denominator degrees of freedom the F tail is the chi-square
    // tail with one correction term, whose error is of the order
    // c^4 / df2^2 (documented in `dist::f::sf`).
    let approximated = case.name == "f" && case.params["df2"] >= 1e7;
    let tail_bucket = if approximated { large_df2 } else { tails };
    tail_bucket.record(relative((m.cdf)(x), point.lower.value()), || label("cdf"));
    tail_bucket.record(relative((m.sf)(x), point.upper.value()), || label("sf"));
    for (got, want, what) in [
        ((m.ln_cdf)(x), &point.log_lower, "ln_cdf"),
        ((m.ln_sf)(x), &point.log_upper, "ln_sf"),
    ] {
        let Some(want) = want.as_ref().map(Number::value) else {
            continue;
        };
        if approximated || want < UNDERFLOW_LOG && case.name != "normal" {
            continue;
        }
        ln_tails.record(logarithmic(got, want), || label(what));
    }
}

/// The errors of the quantiles of both tails at one probability.
fn compare_quantile(case: &Case, m: &Methods, q: &Quantile, worst: &mut [Worst; 7]) {
    let [_, _, _, _, quantiles, counts, _] = worst;
    let p = q.p.value();
    let label = |what: &str| format!("{} {:?} {what} at {p:e}", case.name, case.params);
    for (got, want, tail, what) in [
        ((m.quantile)(p), q.lower.value(), &m.cdf, "quantile"),
        ((m.isf)(p), q.upper.value(), &m.sf, "isf"),
    ] {
        if m.discrete {
            let error = if got == want { 0.0 } else { 1.0 };
            counts.record(error, || format!("{} ({got} against {want})", label(what)));
            continue;
        }
        // The t quantile searches a bracket up to 1e300 and is NaN beyond
        // it (documented in `dist::t::quantile`).
        if want.abs() > 1e299 {
            continue;
        }
        // Where R's quantile misses p by more than ours does, as R's qt does
        // with fewer than one degree of freedom far out, ours is held to its
        // own tail instead of to R's quantile.
        let ours = relative(tail(got), p);
        let error = if ours <= relative(tail(want), p) {
            ours.min(relative(got, want))
        } else {
            relative(got, want)
        };
        quantiles.record(error, || label(what));
    }
}

#[test]
fn every_method_agrees_with_r() {
    let r = reference();
    let mut worst: [Worst; 7] = Default::default();
    for case in &r.cases {
        let m = build(case);
        for point in &case.points {
            compare_point(case, &m, point, &mut worst);
        }
        for q in &case.quantiles {
            compare_quantile(case, &m, q, &mut worst);
        }
    }
    let [
        density,
        ln_density,
        tails,
        ln_tails,
        quantiles,
        counts,
        large_df2,
    ] = &worst;
    density.check("density", 1e-12);
    ln_density.check("log density", 1e-13);
    tails.check("tails", 1e-12);
    ln_tails.check("log tails", 1e-12);
    quantiles.check("quantiles", 1e-11);
    counts.check("discrete quantiles", 0.0);
    large_df2.check("F tails with df2 of 1e7 and more", 1e-6);
}
