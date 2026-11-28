//! `rmst::rmst2_adjusted` checked against `survRM2::rmst2` 1.0.4 with
//! `covariates`: the coefficients, standard errors, z statistics, p-values and
//! limits of the regression of the restricted mean on the arm and covariates for
//! the difference, the ratio and the ratio of times lost, on `survival::lung` and
//! simulated sets with tied times, one and two covariates and an arm that ends in
//! a censoring. See `data/rmst_covariates.R`.
//!
//! # Tolerances
//!
//! With `n` rows, `p` coefficients and `kappa` the condition number of the matrix
//! the fit solves with, as recorded by R (the 2-norm of `X'WX`, or of
//! `X'W diag(mu) X` for the log link):
//!
//! - A coefficient is the solution of normal equations whose entries are sums of
//!   `n` terms, so its error is at most `16 kappa (n + p) eps` times the larger
//!   of 1 and the largest coefficient, in either program. The ratio and loss
//!   ratio fits stop on a change of the deviance of `1e-8` and are not at the
//!   root of their equation. R reports its own distance from the root as
//!   `stop_error`, measured by refitting its weights with `epsilon = 1e-14`,
//!   and the fit here follows the same iteration from the same start, so each
//!   program is allowed that distance: `2 stop_error` is added.
//! - A standard error is the square root of a diagonal entry of
//!   `A^-1 S A^-1`, a product of sums over `n` rows with the conditioning of
//!   `A`, so its relative error is at most `64 kappa (n + p) eps`. For the log
//!   link, `A` and `S` depend on the coefficients through `exp(x' beta)`, whose
//!   relative change is at most `p x_max` times the change of a coefficient,
//!   so `4 p x_max` times the allowance of the coefficients is added to it.
//! - `z` has the sum of the relative errors of its numerator and denominator.
//! - A p-value is `2 Phi(-|z|)`, with the derivative `-2 phi` bounded by `0.8` in
//!   `z`, plus `8 eps` for `Phi`. `rmst2` forms `1 - pchisq(z^2, 1)`, which loses
//!   `eps` absolutely to the subtraction.
//! - A limit is the coefficient less `z_alpha` standard errors, so adds the error
//!   of the standard error times `z_alpha` and `4 eps` for the quantile, and a
//!   limit of a ratio is the exponential, whose relative error is that
//!   absolute error.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    reason = "row counts, status and arm codes are small whole numbers"
)]

use std::collections::HashMap;

use epistat_linalg::Mat;
use epistat_survival::Level;
use epistat_survival::SurvData;
use epistat_survival::rmst::{RmstOptions, RmstRegression, Tau, rmst2_adjusted};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: HashMap<String, Dataset>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
    #[serde(rename = "survRM2")]
    survrm2: String,
}

#[derive(Deserialize)]
struct Dataset {
    time: Vec<f64>,
    status: Vec<f64>,
    arm: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    fits: Vec<Fit>,
}

#[derive(Deserialize)]
struct Fit {
    covs: Vec<String>,
    tau_given: Option<f64>,
    tau: f64,
    alpha: f64,
    difference: Row,
    ratio: Row,
    loss_ratio: Row,
}

#[derive(Deserialize)]
struct Row {
    coef: Vec<f64>,
    se: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    ratio: Option<Vec<f64>>,
    stop_error: f64,
    cond: f64,
}

#[derive(Default)]
struct Worst(HashMap<String, (f64, String)>);

impl Worst {
    #[track_caller]
    fn see(&mut self, output: &str, label: &str, got: f64, want: f64, tolerance: f64) {
        let ratio = (got - want).abs() / tolerance;
        assert!(
            ratio <= 1.0,
            "{output} at {label}: got {got:e}, R {want:e}, deviation {:e} against tolerance {tolerance:e}",
            (got - want).abs()
        );
        let entry = self
            .0
            .entry(output.to_string())
            .or_insert((0.0, String::new()));
        if ratio >= entry.0 {
            *entry = (ratio, label.to_string());
        }
    }
}

fn check(
    w: &mut Worst,
    label: &str,
    got: &RmstRegression,
    want: &Row,
    log_link: bool,
    n: f64,
    xmax: f64,
    z_alpha: f64,
) {
    let p = want.coef.len();
    assert_eq!(got.coefficients.len(), p, "{label}: width");
    let big = want.coef.iter().fold(1.0f64, |m, b| m.max(b.abs()));
    let stop = if log_link { 2.0 * want.stop_error } else { 0.0 };
    let tc = 16.0 * want.cond * (n + p as f64) * EPS * big + stop;
    let rel_se = 64.0 * want.cond * (n + p as f64) * EPS
        + if log_link {
            4.0 * p as f64 * xmax * tc
        } else {
            0.0
        };
    assert_eq!(got.ratio.is_some(), log_link, "{label}: ratio present");
    for j in 0..p {
        let l = format!("{label} coefficient {j}");
        w.see("coef", &l, got.coefficients[j], want.coef[j], tc);
        let se_tol = rel_se * want.se[j];
        w.see("se", &l, got.se[j], want.se[j], se_tol);
        let rel_z = tc / want.coef[j].abs() + rel_se;
        w.see(
            "z",
            &l,
            got.z[j],
            want.z[j],
            rel_z * want.z[j].abs() + 4.0 * EPS,
        );
        let p_tol = 0.8 * want.z[j].abs() * rel_z + 8.0 * EPS + EPS;
        w.see("p", &l, got.p_value[j], want.p[j], p_tol);
        let limit_tol = tc + z_alpha * se_tol + 4.0 * EPS * z_alpha * want.se[j] + 4.0 * EPS * big;
        let scale = |v: f64| if log_link { v.exp() } else { 1.0 };
        let (lo, hi) = (want.lower[j], want.upper[j]);
        w.see(
            "lower",
            &l,
            got.lower[j],
            lo,
            limit_tol * if log_link { lo.abs() } else { 1.0 },
        );
        w.see(
            "upper",
            &l,
            got.upper[j],
            hi,
            limit_tol * if log_link { hi.abs() } else { 1.0 },
        );
        if let (Some(g), Some(r)) = (&got.ratio, &want.ratio) {
            w.see("ratio", &l, g[j], r[j], tc * scale(want.coef[j]));
        }
    }
}

#[test]
fn rmst_covariates_agree_with_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/rmst_covariates.json"
    ));
    eprintln!(
        "{}, survival {}, survRM2 {}",
        reference.versions.r, reference.versions.survival, reference.versions.survrm2
    );
    let mut worst = Worst::default();
    for case in &reference.cases {
        let d = &reference.datasets[&case.name];
        let n = d.time.len();
        let status: Vec<i32> = d.status.iter().map(|&s| s as i32).collect();
        let arm: Vec<u32> = d.arm.iter().map(|&a| a as u32).collect();
        let data = SurvData::builder(&d.time, &status).build().unwrap();
        for fit in &case.fits {
            let cols: Vec<&Vec<f64>> = fit
                .covs
                .iter()
                .map(|c| if c == "x1" { &d.x1 } else { &d.x2 })
                .collect();
            let x = Mat::from_fn(n, cols.len(), |i, j| cols[j][i]);
            let xmax = (0..n)
                .flat_map(|i| (0..cols.len()).map(move |j| (i, j)))
                .map(|(i, j)| x[(i, j)].abs())
                .fold(1.0f64, f64::max);
            let mut opts = RmstOptions::new(fit.tau_given.map_or(Tau::Largest, Tau::Fixed));
            opts.level = Level::confidence(1.0 - fit.alpha).unwrap();
            let label = format!(
                "{} {:?} tau {:?} alpha {}",
                case.name, fit.covs, fit.tau_given, fit.alpha
            );
            let got = rmst2_adjusted(&data, &arm, Some(x.as_ref()), &opts)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                got.difference.tau.to_bits(),
                fit.tau.to_bits(),
                "{label}: tau"
            );
            let z_alpha = epistat_stats::dist::normal::quantile(opts.level.upper_probability());
            // The arm is the stratum, with `xmax` at least 1 for it.
            let (nf, xm) = (n as f64, xmax);
            check(
                &mut worst,
                &format!("{label} difference"),
                &got.difference,
                &fit.difference,
                false,
                nf,
                xm,
                z_alpha,
            );
            check(
                &mut worst,
                &format!("{label} ratio"),
                &got.ratio,
                &fit.ratio,
                true,
                nf,
                xm,
                z_alpha,
            );
            check(
                &mut worst,
                &format!("{label} loss ratio"),
                &got.loss_ratio,
                &fit.loss_ratio,
                true,
                nf,
                xm,
                z_alpha,
            );
        }
    }
    let mut rows: Vec<_> = worst.0.into_iter().collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (output, (ratio, label)) in rows {
        eprintln!("worst {output}: {ratio:.4} of tolerance at {label}");
    }
}

#[test]
fn rmst_regression_refuses_what_rmst2_does_not_take() {
    use epistat_survival::rmst::{RmstEffect, rmst2_regression};
    let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let status = [1, 0, 1, 1, 0, 1, 1, 1];
    let arm = [0u32, 0, 0, 0, 1, 1, 1, 1];
    let opts = RmstOptions::new(Tau::Fixed(4.0));
    let weight = [1.0; 8];
    let start = [0.0; 8];
    let data = SurvData::builder(&time, &status)
        .weight(&weight)
        .build()
        .unwrap();
    assert!(rmst2_regression(&data, &arm, None, RmstEffect::Difference, &opts).is_err());
    let data = SurvData::builder(&time, &status)
        .start(&start)
        .build()
        .unwrap();
    assert!(rmst2_regression(&data, &arm, None, RmstEffect::Difference, &opts).is_err());
    let data = SurvData::builder(&time, &status).build().unwrap();
    let short = Mat::from_fn(7, 1, |i, _| i as f64);
    assert!(rmst2_regression(&data, &arm, Some(short.as_ref()), RmstEffect::Ratio, &opts).is_err());
    // A covariate that repeats the arm is aliased.
    let repeat = Mat::from_fn(8, 1, |i, _| f64::from(arm[i]));
    assert!(
        rmst2_regression(
            &data,
            &arm,
            Some(repeat.as_ref()),
            RmstEffect::Difference,
            &opts
        )
        .is_err()
    );
    assert!(rmst2_regression(&data, &arm, None, RmstEffect::Difference, &opts).is_ok());
}
