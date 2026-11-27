//! The influence function of the Brier score with a Cox censoring model, with
//! tied times, against central differences of the score refitted with perturbed
//! case weights (`tests/data/metrics_censoring_ties.R`).
//!
//! The score is a function of the case weights `w` through the mean and through
//! the weighted Breslow fit of the censoring model. Its derivative with respect
//! to the weight of row `k`, times `N`, is the influence function of row `k`.
//! `riskRegression`'s influence function counts a tied time once, so it is not
//! the reference.
//!
//! # Tolerance
//!
//! The difference quotient at step `h` has the truncation error `c h^2`, which
//! the two steps `h` and `2h` give as `(D(2h) - D(h)) / 3` for the quotient at
//! `h`; the extrapolated `(4 D(h) - D(2h)) / 3` is the reference and the
//! correction size is the bound. The roundoff of the quotient is `N (eps_B + b_B)
//! / h`, with `eps_B` the change of the score when the fit is made at a looser
//! convergence (0 here) and `b_B = 4 (n + m + 3) eps w` the rounding of a score
//! of `n` terms, `m` distinct censoring times and weights at most `w`, which is
//! the largest `1 / G` at the end of follow-up for the covariate value of the
//! largest risk score. The Rust fit is run at the same tight convergence as R's.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties};
use epistat_survival::metrics::score::{Censoring, CoxCensoring, ScoreOptions, brier};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    time: Vec<f64>,
    status: Vec<i32>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    times: Vec<f64>,
    n_tied_times: usize,
    risk: Vec<Vec<f64>>,
    brier: Vec<f64>,
    wmax: f64,
    fit_error: f64,
    h1: f64,
    d1: Vec<Vec<f64>>,
    d2: Vec<Vec<f64>>,
}

#[test]
fn influence_with_tied_censoring_times_against_numerical_derivatives() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/metrics_censoring_ties.json"
    ));
    assert!(r.n_tied_times > 20, "the data must have tied times");
    let n = r.time.len();
    let t = r.times.len();
    let nf = n as f64;
    let data = SurvData::builder(&r.time, &r.status).build().unwrap();
    let x = Mat::from_fn(n, 2, |i, j| if j == 0 { r.x1[i] } else { r.x2[i] });
    let risk = Mat::from_fn(n, t, |i, j| r.risk[i][j]);
    let opts = ScoreOptions {
        keep_influence: true,
        censoring: Censoring::Cox(CoxCensoring {
            x: x.as_ref(),
            options: CoxOptions {
                ties: Ties::Breslow,
                tolerance: 1e-14,
                ..CoxOptions::default()
            },
        }),
        ..ScoreOptions::default()
    };
    let score = brier(&data, risk.as_ref(), &r.times, &opts).unwrap();
    let influence = score.influence.as_ref().unwrap();
    let eps = f64::EPSILON;
    let m = r.status.iter().filter(|&&s| s == 0).count() as f64;
    let b_b = 4.0 * (nf + m + 3.0) * eps * r.wmax;
    let mut worst = 0.0f64;
    let mut worst_value = 0.0f64;
    for j in 0..t {
        // The score itself is the same as the refit's at unit weights.
        let dev = (score.brier[j] - r.brier[j]).abs();
        assert!(dev <= b_b, "brier {dev}");
        worst_value = worst_value.max(dev);
        for k in 0..n {
            let (d1, d2) = (r.d1[k][j], r.d2[k][j]);
            let reference = (4.0 * d1 - d2) / 3.0;
            let bound = (d2 - d1).abs() / 3.0 + nf * (r.fit_error + b_b) / r.h1;
            let dev = (influence[(k, j)] - reference).abs() / bound;
            assert!(
                dev <= 1.0,
                "row {k}, time {}: influence {} against {reference}, bound {bound}",
                r.times[j],
                influence[(k, j)]
            );
            worst = worst.max(dev);
        }
    }
    eprintln!(
        "worst influence deviation: {worst:.3} of its bound; worst score deviation {worst_value:e}"
    );
}
