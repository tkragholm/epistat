//! `metrics::royston` checked against `survival::royston` in survival 3.8-12.
//!
//! Six data sets, each with ties = TRUE and FALSE and with adjust = FALSE and
//! TRUE: the lung data with a predictor that hardly ties, with a predictor of
//! four values, and with weights and strata; simulated data with tied times and
//! a predictor on a coarse grid; and the same with delayed entry. `royston`
//! refits on `fit$y` and ignores weights and strata, so those cases check that
//! the Rust function ignores them too.
//!
//! Two programs are compared: `estimate` on R's own linear predictor, and
//! `from_fit` on a fit of `cox::fit`.
//!
//! # Tolerances
//!
//! Write `eps` for the machine epsilon, `n` for the rows and `kappa` for the
//! condition number of the information of the fit that produced the predictor.
//!
//! - The refit has one covariate, so its information has condition number 1.
//!   Two implementations that take the same Newton steps differ in the
//!   coefficient by the error of the solve, 8 eps, plus the rounding of the
//!   sums of the score and the information, at most `2 n` eps each, plus the
//!   few eps of each normal score and of a mean over ties: relative
//!   `b = (8 + 4 n + 12) eps`. `D` and `se(D)` (a square root of a variance of
//!   the same relative error) are tested at `b` relative, `R.D` at `3 b`.
//! - The adjusted `D` is a square root of `t = (1 + b^2 - r) / r`, which
//!   amplifies the relative error of the coefficient by `a = 2 b^2 / |1 + b^2 -
//!   r|`, so it is tested at `b (1 + a)`, and `se(D)` at `b (2 + a)`. The
//!   adjusted `R.D = 1 - r / (1 + b^2)` moves by at most `r b_rel / 2`.
//! - `R.KO` is a variance from a two-pass sum of `n` terms: `4 (n + 4) eps`
//!   relative. `C.GH` is a mean of `n (n - 1) / 2` terms in (0, 1), each with
//!   a relative error of `(5 + range) eps`, where `range` is the range of the
//!   predictor (the exponential of a difference carries the rounding of the
//!   difference times its size), summed in rows of at most `n` terms: relative
//!   `2 (n + 5 + range) eps` between the two programs.
//! - From a fit of `cox::fit` the predictor differs from R's by the error of
//!   the solve for the coefficients of the model, `8 kappa eps` relative to the
//!   coefficients, so each value of the predictor moves by at most
//!   `delta = 8 kappa eps sum_j |x_ij b_j|`, taken twice for the centring, which
//!   the test asserts. The ranks do not change, and `D` is compared at the bound
//!   above, so a swapped rank shows up as a failure. `C.GH` then moves by at most `delta / 2` and
//!   `R.KO` by at most `4 delta (sd + delta)` absolutely.
//! - `R.N` is a function of two log likelihoods, each a sum of at most `n` terms
//!   of size up to `|l|` and so off by `n eps |l|` in each program. Its
//!   numerator `1 - exp(-x)` with `x = LR / n` and its denominator
//!   `1 - exp(2 l_0 / n)` move by the derivative times that, and the bound is
//!   computed in the test from the values.
//!
//! The bounds are not changed after a run. The worst deviation in units of its
//! bound is printed for each output.

use std::collections::BTreeMap;

use epistat_survival::cox::predict::{FittedModel, PredictOptions, PredictType, predict};
use epistat_survival::cox::{CoxOptions, fit};
use epistat_survival::metrics::royston::{
    Adjustment, FitOptions, RoystonOptions, estimate, from_fit,
};
use epistat_survival::{SurvData, timefix::TimeFix};
use epistat_testkit::load_json;
use faer::Mat;
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Values {
    #[serde(rename = "D")]
    d: f64,
    #[serde(rename = "se(D)")]
    se_d: f64,
    #[serde(rename = "R.D")]
    r_d: f64,
    #[serde(rename = "R.KO")]
    r_ko: f64,
    #[serde(rename = "R.N")]
    r_n: f64,
    #[serde(rename = "C.GH")]
    c_gh: f64,
}

#[derive(Deserialize)]
struct Result {
    ties: bool,
    adjust: bool,
    values: Values,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    stop: Vec<f64>,
    status: Vec<i32>,
    start: Option<Vec<f64>>,
    weight: Option<Vec<f64>>,
    stratum: Option<Vec<u32>>,
    x: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    eta: Vec<f64>,
    nevent: f64,
    loglik: Vec<f64>,
    n: usize,
    information_condition: f64,
    results: Vec<Result>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Default)]
struct Worst(BTreeMap<String, f64>);

impl Worst {
    fn put(&mut self, name: &str, got: f64, want: f64, bound: f64) {
        let ratio = (got - want).abs() / bound;
        let e = self.0.entry(name.to_string()).or_insert(0.0);
        *e = e.max(ratio);
        assert!(
            ratio <= 1.0,
            "{name}: got {got:e}, R {want:e}, {ratio:.3} times the bound"
        );
    }
}

fn data_of(c: &Case) -> SurvData<'_> {
    let mut b = SurvData::builder(&c.stop, &c.status);
    if let Some(s) = &c.start {
        b = b.start(s);
    }
    if let Some(w) = &c.weight {
        b = b.weight(w);
    }
    if let Some(s) = &c.stratum {
        b = b.stratum(s);
    }
    b.build().unwrap()
}

/// The relative bound of the Cox refit on the normal scores.
fn cox_bound(n: usize) -> f64 {
    (8.0 + 4.0 * n as f64 + 12.0) * f64::EPSILON
}

#[test]
fn estimate_matches_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/royston.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        r.versions.r, r.versions.survival, r.versions.jsonlite
    );
    let mut worst = Worst::default();
    for c in &r.cases {
        let data = data_of(c);
        let n = c.stop.len();
        let range = c.eta.iter().copied().fold(f64::MIN, f64::max)
            - c.eta.iter().copied().fold(f64::MAX, f64::min);
        for res in &c.results {
            let got = estimate(
                &data,
                &c.eta,
                &RoystonOptions {
                    ties: res.ties,
                    adjust: res.adjust.then_some(Adjustment {
                        n_events: c.nevent,
                        n_coef: c.coefficients.len(),
                    }),
                    ..RoystonOptions::default()
                },
            )
            .unwrap();
            assert_eq!(got.n_rows, n);
            assert_eq!(got.n_used, n);
            assert!(got.r_n.is_none());
            compare(&mut worst, &got, res, c, range, 0.0, 0.0);
        }
    }
    for (k, v) in &worst.0 {
        eprintln!("estimate: worst {k}: {v:.3} of its bound");
    }
}

fn compare(
    worst: &mut Worst,
    got: &epistat_survival::metrics::royston::Royston,
    res: &Result,
    c: &Case,
    range: f64,
    delta: f64,
    sd: f64,
) {
    let n = c.stop.len();
    let b = cox_bound(n);
    let want = &res.values;
    let beta = got.beta.unwrap();
    let (d_bound, se_bound, rd_bound) = if res.adjust {
        let r = c.nevent / (c.nevent - c.coefficients.len() as f64);
        let a = 2.0 * beta * beta / (1.0 + beta * beta - r).abs();
        (
            b * (1.0 + a) * want.d.abs(),
            b * (2.0 + a) * want.se_d,
            r * b / 2.0 + 4.0 * f64::EPSILON,
        )
    } else {
        (b * want.d.abs(), b * want.se_d, 3.0 * b * want.r_d)
    };
    worst.put("D", got.d.unwrap(), want.d, d_bound);
    worst.put("se(D)", got.se_d.unwrap(), want.se_d, se_bound);
    worst.put("R.D", got.r_d.unwrap(), want.r_d, rd_bound);
    let n_f = n as f64;
    worst.put(
        "R.KO",
        got.r_ko.unwrap(),
        want.r_ko,
        4.0 * (n_f + 4.0) * f64::EPSILON * want.r_ko + 4.0 * delta * (sd + delta),
    );
    worst.put(
        "C.GH",
        got.c_gh.unwrap(),
        want.c_gh,
        2.0 * (n_f + 5.0 + range) * f64::EPSILON * want.c_gh + delta / 2.0,
    );
}

#[test]
fn from_fit_matches_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/royston.json"
    ));
    let mut worst = Worst::default();
    for c in &r.cases {
        let data = data_of(c);
        let n = c.stop.len();
        let p = c.x.len();
        let x = Mat::from_fn(n, p, |i, j| c.x[j][i]);
        let cox = CoxOptions::default();
        let model = fit(&data, x.as_ref(), &cox).unwrap();
        // The predictor of the fit against R's, after R's centring.
        let eta_ours = predict(
            &FittedModel {
                data: &data,
                x: x.as_ref(),
                fit: &model,
                options: &cox,
            },
            PredictType::LinearPredictor,
            None,
            &PredictOptions::default(),
        )
        .unwrap()
        .fit;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
        let shift = (0..n)
            .map(|i| (eta_ours[i] - c.eta[i]).abs())
            .fold(0.0, f64::max);
        let scale = (0..n)
            .map(|i| {
                (0..p)
                    .map(|j| (x[(i, j)] * c.coefficients[j]).abs())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let delta = 8.0 * c.information_condition * f64::EPSILON * scale * 2.0;
        assert!(shift <= delta, "{}: predictor moved by {shift:e}", c.name);
        let range = c.eta.iter().copied().fold(f64::MIN, f64::max)
            - c.eta.iter().copied().fold(f64::MAX, f64::min);
        let sd = {
            let m = mean(&c.eta);
            (c.eta.iter().map(|e| (e - m) * (e - m)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
        };
        for res in &c.results {
            let got = from_fit(
                &data,
                x.as_ref(),
                &model,
                &cox,
                &FitOptions {
                    ties: res.ties,
                    adjust: res.adjust,
                    gonen_heller: true,
                },
            )
            .unwrap();
            assert_eq!(got.n_rows, n);
            compare(&mut worst, &got, res, c, range, delta, sd);
            // Nagelkerke from the log likelihoods.
            let (l0, l1) = (c.loglik[0], c.loglik[1]);
            let nf = c.n as f64;
            let x_lr = -2.0 * (l0 - l1) / nf;
            let e = |l: f64| 2.0 * nf * f64::EPSILON * l.abs();
            let dx = 2.0 * (e(l0) + e(l1)) / nf;
            let dy = 2.0 * e(l0) / nf;
            let num = 1.0 - (-x_lr).exp();
            let den = 1.0 - (2.0 * l0 / nf).exp();
            let rel = dx * (-x_lr).exp() / num + dy * (2.0 * l0 / nf).exp() / den;
            worst.put(
                "R.N",
                got.r_n.unwrap(),
                res.values.r_n,
                rel * res.values.r_n + 4.0 * f64::EPSILON,
            );
        }
    }
    for (k, v) in &worst.0 {
        eprintln!("from_fit: worst {k}: {v:.3} of its bound");
    }
}

#[test]
fn timefix_is_applied_to_the_times() {
    // Times that differ by rounding error tie after the fix, and the result is
    // that of the data with exact ties.
    let stop = [1.0, 1.0 + 1e-13, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0];
    let tied = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0];
    let status = [1, 1, 0, 1, 1, 0, 1, 1];
    let eta = [0.3, -0.2, 0.5, 0.1, 0.9, -0.4, 0.0, 0.7];
    let a = SurvData::builder(&stop, &status).build().unwrap();
    let b = SurvData::builder(&tied, &status).build().unwrap();
    let opts = RoystonOptions::default();
    assert_eq!(
        estimate(&a, &eta, &opts).unwrap(),
        estimate(&b, &eta, &opts).unwrap()
    );
    let exact = RoystonOptions {
        timefix: TimeFix::Exact,
        ..opts
    };
    assert_ne!(
        estimate(&a, &eta, &exact).unwrap().beta,
        estimate(&a, &eta, &opts).unwrap().beta
    );
}
