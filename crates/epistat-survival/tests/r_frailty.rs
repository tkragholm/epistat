//! `frailty::fit` against `survival::coxph` with `frailty(cluster)`
//! (`tests/data/frailty.R`), and the EM route against the penalised one.
//!
//! Deviations of the coefficients and the log frailties are measured in
//! standard errors: R's for the coefficients, and for a log frailty the square
//! root of `frailty_variances`. Each kind of reference has its own bound `b`
//! in those units, with `eps = 1e-12` the stopping rule R ran with, a relative
//! change of the log likelihood `l`:
//!
//! * `sparse = FALSE`: R's Newton steps use the whole information. The step
//!   before the last changed `l` by at most `eps |l|`, so it was at most
//!   `sqrt(2 eps |l|)` standard errors long, and the quadratic convergence of
//!   Newton's method leaves an error of the order of its square,
//!   `b = 2 eps (1 + |l|)`. Our own stopping rule, a decrement below `1e-10`
//!   before a last full step, adds `1e-9`.
//! * `sparse = TRUE`: R's steps use the diagonal of the frailty block and
//!   converge linearly. With a rate of at most a half the distance left in `l`
//!   is at most the last change, `eps |l|`, so `b = sqrt(2 eps (1 + |l|))`.
//! * The transcription for Efron ties with weights and entry times, where
//!   `coxph` does not converge (see the R script): its gradient is below `2e-12`
//!   and its covariance comes from `optimHess`, central differences of the
//!   gradient with steps of `1e-3`, whose relative error is of the order of the
//!   square of the step, so `b = 1e-6` and `1e-5` for the covariance.
//!
//! A covariance entry is compared relative to the largest diagonal entry, with
//! the same bound, since the information at a point `b` standard errors away
//! differs by about `b` relative, plus `1e-10` for the conjugate gradient
//! solve. The partial log likelihood is not stationary in the log frailties: its
//! gradient there is the penalty's, `nu (e^w - 1)`, so it may differ by `b` times
//! the sum over clusters of `nu |e^w - 1|` times a standard error, plus
//! rounding of `4e-16 |l|`. The penalised likelihood is stationary, so the
//! corrected likelihood `c.loglik` differs by at most `b^2 (p + G) / 2` from the
//! parameters, plus `nu G` times twice the machine epsilon for R's
//! `log(nu / (nu + d))` near 1, plus rounding.

use epistat_linalg::Mat;
use epistat_survival::SurvBuf;
use epistat_survival::cox::Ties;
use epistat_survival::frailty::{FrailtyFit, FrailtyOptions, Method, Theta, ThetaStatus, fit};
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::BTreeMap;

const R_EPS: f64 = 1e-12;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
}

#[derive(Deserialize)]
struct Dataset {
    time: Vec<f64>,
    start: Vec<f64>,
    status: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    cluster: Vec<f64>,
    stratum: Vec<f64>,
    weight: Vec<f64>,
    offset: Vec<f64>,
}

#[derive(Deserialize)]
struct RFit {
    name: String,
    dataset: String,
    ties: String,
    sparse: bool,
    fixed: bool,
    #[serde(default)]
    transcribed: bool,
    theta: f64,
    coef: Vec<f64>,
    var: Vec<f64>,
    var2: Option<Vec<f64>>,
    frail: Vec<f64>,
    fvar: Option<Vec<f64>>,
    #[serde(deserialize_with = "epistat_testkit::na_f64_vec")]
    loglik: Vec<f64>,
    c_loglik: Option<f64>,
    history: Option<Vec<Vec<f64>>>,
}

#[derive(Deserialize)]
struct RProfile {
    name: String,
    dataset: String,
    ties: String,
    theta: f64,
    c_loglik: f64,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: BTreeMap<String, Dataset>,
    fits: Vec<RFit>,
    profiles: Vec<RProfile>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/frailty.json"
    ))
}

fn ties(name: &str) -> Ties {
    if name == "breslow" {
        Ties::Breslow
    } else {
        Ties::Efron
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "statuses, clusters and strata are small whole numbers written by R as doubles"
)]
fn run(d: &Dataset, ties: Ties, theta: Theta, method: Method) -> FrailtyFit {
    let n = d.time.len();
    let status: Vec<i32> = d.status.iter().map(|&s| s as i32).collect();
    let mut b = SurvBuf::builder(d.time.clone(), status)
        .cluster(d.cluster.iter().map(|&c| c as u32).collect());
    if d.start.iter().any(|&s| s > 0.0) {
        b = b.start(d.start.clone());
    }
    if d.stratum.iter().any(|&s| s != 0.0) {
        b = b.stratum(d.stratum.iter().map(|&s| s as u32).collect());
    }
    if d.weight.iter().any(|&w| w - 1.0 != 0.0) {
        b = b.weight(d.weight.clone());
    }
    let buf = b.build().unwrap();
    let x = Mat::from_fn(n, 2, |i, j| [d.x1[i], d.x2[i]][j]);
    let has_offset = d.offset.iter().any(|&o| o != 0.0);
    let opts = FrailtyOptions {
        ties,
        theta,
        method,
        offset: has_offset.then_some(&d.offset[..]),
        theta_tolerance: 1e-9,
        ..FrailtyOptions::default()
    };
    fit(&buf.as_data(), x.as_ref(), &opts).unwrap()
}

/// The worst ratio of a deviation to its bound over the checks of one fit.
struct Worst {
    name: String,
    ratio: f64,
    what: &'static str,
}

impl Worst {
    fn check(&mut self, what: &'static str, deviation: f64, bound: f64) {
        let ratio = deviation / bound;
        assert!(
            ratio <= 1.0,
            "{}: {what} deviates by {deviation:e}, bound {bound:e}",
            self.name
        );
        if ratio > self.ratio {
            self.ratio = ratio;
            self.what = what;
        }
    }
}

#[expect(clippy::cast_precision_loss, reason = "parameter counts are small")]
fn compare_fit(f: &RFit, got: &FrailtyFit) -> Worst {
    let mut worst = Worst {
        name: f.name.clone(),
        ratio: 0.0,
        what: "",
    };
    let l = f.loglik[1].abs();
    let b = if f.transcribed {
        1e-6
    } else if f.sparse {
        (2.0 * R_EPS * (1.0 + l)).sqrt()
    } else {
        2.0 * R_EPS * (1.0 + l) + 1e-9
    };
    let (p, g) = (2usize, f.frail.len());
    for a in 0..p {
        let se = f.var[a * p + a].sqrt();
        worst.check(
            "coefficient",
            (got.coefficients[a].unwrap() - f.coef[a]).abs() / se,
            b,
        );
    }
    let frail_se: Vec<f64> = got
        .frailty_variances
        .iter()
        .map(|v| v.unwrap().sqrt())
        .collect();
    for ((w, &want), se) in got.frailties.iter().zip(&f.frail).zip(&frail_se) {
        worst.check("log frailty", (w.unwrap() - want).abs() / se, b);
    }
    let weighted = got.marginal_log_likelihood.is_none();
    let (cov, cov2) = if f.sparse {
        (&got.diagonal_covariance, &got.diagonal_covariance_sandwich)
    } else {
        (&got.covariance, &got.covariance_sandwich)
    };
    let scale = f.var[0].abs().max(f.var[3].abs());
    let var_bound = if f.transcribed { 1e-5 } else { b + 1e-10 };
    for (c, &want) in cov.iter().zip(&f.var) {
        worst.check("var", (c.unwrap() - want).abs() / scale, var_bound);
    }
    if let Some(var2) = &f.var2 {
        let scale2 = var2[0].abs().max(var2[3].abs());
        for (c, &want) in cov2.iter().zip(var2) {
            worst.check("var2", (c.unwrap() - want).abs() / scale2, b + 1e-10);
        }
    }
    if let Some(fvar) = &f.fvar {
        for (v, &want) in got.frailty_variances.iter().zip(fvar) {
            worst.check("fvar", (v.unwrap() - want).abs() / want, b + 1e-10);
        }
    }
    let nu = 1.0 / f.theta;
    let slope: f64 = got
        .frailties
        .iter()
        .zip(&frail_se)
        .map(|(w, se)| nu * (w.unwrap().exp() - 1.0).abs() * se)
        .sum();
    worst.check(
        "partial log likelihood",
        (got.log_partial_likelihood - f.loglik[1]).abs(),
        b * slope + 4e-16 * l + 1e-12,
    );
    if let (Some(c), false) = (f.c_loglik, weighted) {
        let rounding = 2.0 * f64::EPSILON * nu * g as f64 + 4e-16 * l;
        worst.check(
            "c.loglik",
            (got.marginal_log_likelihood.unwrap() - c).abs(),
            0.5 * b * b * (p + g) as f64 + rounding + 1e-12,
        );
    }
    if let Some(h) = &f.history {
        assert_eq!(h.len(), got.history.len(), "{}: search length", f.name);
        for (row, step) in h.iter().zip(&got.history) {
            let nu = if row[0] > 0.0 { 1.0 / row[0] } else { 0.0 };
            let rounding = 2.0 * f64::EPSILON * nu * g as f64 + 4e-16 * l;
            worst.check(
                "search c.loglik",
                (step.marginal_log_likelihood - row[2]).abs(),
                0.5 * b * b * (p + g) as f64 + rounding + 1e-12,
            );
            worst.check(
                "search theta",
                (step.theta - row[0]).abs(),
                1e-6 * row[0] + 1e-15,
            );
        }
    }
    worst
}

#[test]
fn fits_equal_coxph() {
    let r = reference();
    assert_eq!(r.versions.survival, "3.8.12");
    println!("{}, survival {}", r.versions.r, r.versions.survival);
    for f in &r.fits {
        let d = &r.datasets[&f.dataset];
        let theta = if f.fixed {
            Theta::Fixed(f.theta)
        } else {
            Theta::Coxph {
                eps: 1e-5,
                outer_max: 30,
            }
        };
        let got = run(d, ties(&f.ties), theta, Method::Penalised);
        assert!(got.converged, "{}", f.name);
        let worst = compare_fit(f, &got);
        println!("{:36} worst {:.3} ({})", f.name, worst.ratio, worst.what);
    }
}

/// The search of `coxph` stops early; the profile maximum agrees with
/// `optimize` run on `c.loglik` to `1e-9` on `log theta`. Near the maximum `c`
/// is quadratic in `t = log theta` and flat, so the two maxima agree in `c` to
/// the c.loglik bound of the sparse fits, and in `t` only to about the square
/// root of that bound over the curvature, `1e-3` here.
#[test]
fn profile_equals_optimize() {
    let r = reference();
    for f in &r.profiles {
        let d = &r.datasets[&f.dataset];
        let got = run(d, ties(&f.ties), Theta::Profile, Method::Penalised);
        assert_eq!(got.theta_status, ThetaStatus::Interior, "{}", f.name);
        let m = got.marginal_log_likelihood.unwrap();
        let bound = 2.0 * R_EPS * (1.0 + m.abs()) + 1e-10;
        assert!(
            (m - f.c_loglik).abs() <= bound,
            "{}: {m} {}",
            f.name,
            f.c_loglik
        );
        let t = (got.theta.ln() - f.theta.ln()).abs();
        assert!(t <= 1e-3, "{}: theta {} {}", f.name, got.theta, f.theta);
        let lr = got.likelihood_ratio.unwrap();
        let m0 = got.cox_log_likelihood.unwrap();
        assert!((lr.statistic - 2.0 * (m - m0)).abs() <= 1e-9 * (1.0 + m.abs()));
        println!(
            "{:36} theta {:.9} vs {:.9}, c {:.3e}",
            f.name,
            got.theta,
            f.theta,
            m - f.c_loglik
        );
    }
}

/// With no heterogeneity in the data the profile rises towards theta = 0 and
/// the fit is the Cox model.
#[test]
fn profile_on_the_boundary() {
    let r = reference();
    let d = &r.datasets["sim50_none"];
    let got = run(d, Ties::Breslow, Theta::Profile, Method::Penalised);
    assert_eq!(got.theta_status, ThetaStatus::Boundary);
    assert_eq!(got.theta.to_bits(), 0.0_f64.to_bits());
    assert!(got.frailties.iter().all(|w| w == &Some(0.0)));
    assert_eq!(
        got.likelihood_ratio.unwrap().p_value.to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        got.marginal_log_likelihood.unwrap().to_bits(),
        got.cox_log_likelihood.unwrap().to_bits()
    );
}

/// EM reaches the penalised solution with Breslow ties, and its marginal
/// likelihood, computed from the gamma integral, equals the corrected
/// likelihood of the penalised fit. EM converges linearly at an unknown rate,
/// stopping when the marginal likelihood changes by less than `1e-14`
/// relative; with a rate up to `0.999` the parameters are then within
/// `sqrt(2e-14 |m| / 0.001)` standard errors, and the marginal likelihood,
/// stationary at the solution, within `1e-9`.
#[test]
fn em_equals_penalised() {
    let r = reference();
    let em = Method::Em {
        max_iterations: 20_000,
        tolerance: 1e-14,
    };
    for (dataset, theta) in [("kidney", 0.5), ("sim50", 0.6), ("sim50_entry_unit", 0.6)] {
        let d = &r.datasets[dataset];
        let a = run(d, Ties::Breslow, Theta::Fixed(theta), Method::Penalised);
        let b = run(d, Ties::Breslow, Theta::Fixed(theta), em);
        assert!(b.converged, "{dataset}: EM in {} steps", b.iterations);
        let bound = (2e-14 * a.marginal_log_likelihood.unwrap().abs() / 1e-3).sqrt();
        let mut worst: f64 = 0.0;
        for k in 0..2 {
            let dev =
                (a.coefficients[k].unwrap() - b.coefficients[k].unwrap()).abs() / a.se[k].unwrap();
            worst = worst.max(dev);
        }
        for ((wa, wb), v) in a
            .frailties
            .iter()
            .zip(&b.frailties)
            .zip(&a.frailty_variances)
        {
            worst = worst.max((wa.unwrap() - wb.unwrap()).abs() / v.unwrap().sqrt());
        }
        assert!(worst <= bound, "{dataset}: {worst:e} against {bound:e}");
        let (ma, mb) = (
            a.marginal_log_likelihood.unwrap(),
            b.marginal_log_likelihood.unwrap(),
        );
        assert!((ma - mb).abs() <= 1e-9, "{dataset}: {ma} {mb}");
        println!(
            "{dataset}: EM {} steps, worst {worst:.2e} of bound {bound:.2e}, marginal {:.2e}",
            b.iterations,
            ma - mb
        );
    }
}
