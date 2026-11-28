//! Inverse probability of treatment weights and weighted Cox fits checked against
//! R 4.6.2: `WeightIt::weightit` and its summary, `ipw::ipwpoint`,
//! `WeightIt::coxph_weightit` for the variance that accounts for the estimated
//! weights, `survival::coxph(robust = TRUE)` for the variance that does not, and
//! finite differences of `coxph` refits for counting-process data.
//!
//! # The tolerances
//!
//! Both sides run IRLS from the same start with the same stopping rule, so they
//! reach the same coefficients up to the error of the solves, about the
//! condition number `kappa` of the information matrix on its scaled columns
//! times the machine epsilon. A coefficient differs by `bound = 8 kappa eps`
//! relative to `max(|c|, 1)`, as in `r_cox.rs`. The error of the linear
//! predictor of subject `i` is then at most `eta_i = bound sum_j |z_ij| max(|c_j|, 1)`.
//! Every weight has `|d log w / d eta| <= 1`, so its relative error is at most
//! `eta_i` to first order, and `2 eta_i + 4 eps` with the second order and the
//! rounding of the formula. A fitted probability differs by at most `eta_i / 4`.
//! A sum of `n` weights adds `n eps` in relative terms, and the summaries that
//! are functions of the sums add their relative errors.
//!
//! The Cox covariances are sums over subjects of products of score residuals,
//! each with a relative error of at most `2 (bound_cox X + omega)`, where `X` is
//! `sum_j max_i |x_ij| max(|beta_j|, 1)` and `omega` the largest relative error of a
//! weight. A covariance entry is a quadratic form in two such quantities
//! and the inverse information, so the tolerance is
//! `16 (bound_cox X + omega) + 16 n eps` relative to `sqrt(V_ii V_jj)`. The factor
//! 16 allows two factors of 2 for the two residuals, a factor 2 for the two
//! inverse informations and a factor 2 for the cross block of the stacked
//! system. Where the reference is a finite difference, its error is added: the
//! difference between the covariance from Richardson-extrapolated derivatives
//! and from the plain central differences, which is an upper bound for the
//! extrapolated one.

use epistat_core::Estimand;
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties};
use epistat_survival::weighting::Weights;
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::iptw::{self, IptwOptions};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    #[serde(rename = "WeightIt")]
    weightit: String,
    ipw: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Summary {
    n: usize,
    sum: f64,
    min: f64,
    max: f64,
    mean: f64,
    cv: f64,
    ess: f64,
}

#[derive(Deserialize)]
struct RIptw {
    name: String,
    estimand: String,
    stabilise: bool,
    weights: Vec<f64>,
    propensity: Vec<f64>,
    coefficients: Vec<f64>,
    treated: Summary,
    control: Summary,
    weightit_ess: Vec<f64>,
    weightit_cv: Vec<f64>,
    kappa: f64,
    #[serde(default)]
    ipw_weights: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct RCox {
    name: String,
    iptw: String,
    ties: String,
    stop: Vec<f64>,
    status: Vec<i32>,
    #[serde(default)]
    start: Option<Vec<f64>>,
    #[serde(default)]
    subject: Option<Vec<u32>>,
    #[serde(default)]
    cluster: Option<Vec<u32>>,
    x: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    robust_covariance: Vec<Vec<f64>>,
    #[serde(default)]
    exact_covariance: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    shortcut_covariance: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    covariance_plain: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    robust_covariance_plain: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    kappa: Option<f64>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    treat: Vec<u8>,
    design: Vec<Vec<f64>>,
    iptw: Vec<RIptw>,
    cox: Vec<RCox>,
    cox_fd: Vec<RCox>,
}

fn load() -> Reference {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/weighting.json"
    ));
    assert!(reference.versions.r.starts_with("R version 4.6"));
    assert_eq!(reference.versions.survival, "3.8.12");
    assert_eq!(reference.versions.weightit, "2.1.0");
    assert_eq!(reference.versions.ipw, "1.3.2");
    assert_ne!(reference.versions.jsonlite, "");
    reference
}

fn estimand(name: &str) -> Estimand {
    match name {
        "ATE" => Estimand::Ate,
        "ATT" => Estimand::Att,
        "ATC" => Estimand::Atc,
        other => panic!("estimand {other}"),
    }
}

/// The error bound of the linear predictors, one per subject, and the bound on
/// a coefficient.
fn eta_bounds(design: &Mat<f64>, coefficients: &[f64], kappa: f64) -> Vec<f64> {
    let bound = 8.0 * kappa * EPS;
    (0..design.nrows())
        .map(|i| {
            bound
                * (0..design.ncols())
                    .map(|j| design[(i, j)].abs() * coefficients[j].abs().max(1.0))
                    .sum::<f64>()
        })
        .collect()
}

fn design(reference: &Reference) -> (Mat<f64>, Vec<bool>) {
    let n = reference.treat.len();
    let p = reference.design[0].len();
    (
        Mat::from_fn(n, p, |i, j| reference.design[i][j]),
        reference.treat.iter().map(|&a| a == 1).collect(),
    )
}

#[test]
fn weights_and_summaries_match_weightit() {
    let reference = load();
    let (z, treat) = design(&reference);
    let n = treat.len();
    for r in &reference.iptw {
        let opts = IptwOptions {
            estimand: estimand(&r.estimand),
            stabilise: r.stabilise,
            optimal_subset: None,
            ..Default::default()
        };
        let w = iptw::fit(&treat, z.as_ref(), &opts).unwrap();
        let eta = eta_bounds(&z, &r.coefficients, r.kappa);
        let bound = 8.0 * r.kappa * EPS;
        let mut worst = [0.0f64; 3];
        for (i, &eta_i) in eta.iter().enumerate() {
            let tol_w = r.weights[i] * (2.0 * eta_i + 4.0 * EPS);
            let tol_p = eta_i / 4.0 + 4.0 * EPS;
            let dw = (w.weights[i] - r.weights[i]).abs() / tol_w;
            let dp = (w.propensity[i] - r.propensity[i]).abs() / tol_p;
            assert!(
                dw <= 1.0,
                "{} weight {i}: {} against {}",
                r.name,
                w.weights[i],
                r.weights[i]
            );
            assert!(dp <= 1.0, "{} propensity {i}", r.name);
            worst[0] = worst[0].max(dw);
            worst[1] = worst[1].max(dp);
        }
        for (j, (c, rc)) in w.coefficients.iter().zip(&r.coefficients).enumerate() {
            let d = (c - rc).abs() / (bound * rc.abs().max(1.0));
            assert!(d <= 1.0, "{} coefficient {j}", r.name);
            worst[2] = worst[2].max(d);
        }
        // Summaries, from the weights of this crate against R's.
        let omega = (0..n).map(|i| 2.0 * eta[i] + 4.0 * EPS).fold(0.0, f64::max);
        let rel = 4.0 * omega + 16.0 * n as f64 * EPS;
        for (mine, theirs, label) in [
            (w.summary.treated.unwrap(), &r.treated, "treated"),
            (w.summary.control.unwrap(), &r.control, "control"),
        ] {
            assert_eq!(mine.n, theirs.n);
            for (m, t, what) in [
                (mine.sum, theirs.sum, "sum"),
                (mine.mean, theirs.mean, "mean"),
                (mine.min, theirs.min, "min"),
                (mine.max, theirs.max, "max"),
                (mine.coefficient_of_variation.unwrap(), theirs.cv, "cv"),
                (mine.effective_sample_size, theirs.ess, "ess"),
            ] {
                assert!(
                    (m - t).abs() <= rel * t.abs().max(1e-300) + 4.0 * EPS,
                    "{} {label} {what}: {m} against {t}",
                    r.name
                );
            }
        }
        // The summary of R's own weights is R's summary to the rounding of the sums.
        let rw = |treated: bool| {
            epistat_survival::weighting::WeightSummary::new(
                r.weights
                    .iter()
                    .zip(&treat)
                    .filter(|&(_, &a)| a == treated)
                    .map(|(&w, _)| w),
            )
            .unwrap()
        };
        let tight = 16.0 * n as f64 * EPS;
        for (mine, theirs) in [(rw(true), &r.treated), (rw(false), &r.control)] {
            for (m, t) in [
                (mine.sum, theirs.sum),
                (mine.mean, theirs.mean),
                (mine.min, theirs.min),
                (mine.max, theirs.max),
                (mine.coefficient_of_variation.unwrap(), theirs.cv),
                (mine.effective_sample_size, theirs.ess),
            ] {
                assert!(
                    (m - t).abs() <= tight * t.abs(),
                    "{} summary of R's weights",
                    r.name
                );
            }
        }
        // What summary.weightit prints: effective sample size and coefficient
        // of variation, controls then treated for the first and treated then
        // controls for the second.
        let tight_ess = tight;
        let (ess_t, ess_c) = (
            rw(true).effective_sample_size,
            rw(false).effective_sample_size,
        );
        assert!((ess_c - r.weightit_ess[0]).abs() <= tight_ess * ess_c);
        assert!((ess_t - r.weightit_ess[1]).abs() <= tight_ess * ess_t);
        let (cv_t, cv_c) = (
            rw(true).coefficient_of_variation.unwrap(),
            rw(false).coefficient_of_variation.unwrap(),
        );
        assert!((cv_t - r.weightit_cv[0]).abs() <= tight * cv_t);
        assert!((cv_c - r.weightit_cv[1]).abs() <= tight * cv_c);
        if let Some(ipw_weights) = &r.ipw_weights {
            for i in 0..n {
                let tol = ipw_weights[i] * (2.0 * eta[i] + 4.0 * EPS) * 4.0;
                assert!(
                    (w.weights[i] - ipw_weights[i]).abs() <= tol,
                    "stabilised weight {i} against ipwpoint"
                );
            }
        }
        println!(
            "{}: worst deviation in tolerances: weights {:.3e}, propensity {:.3e}, coefficients {:.3e}",
            r.name, worst[0], worst[1], worst[2]
        );
    }
}

fn cox_case(reference: &Reference, c: &RCox) -> (f64, f64) {
    let (z, treat) = design(reference);
    let ri = reference.iptw.iter().find(|r| r.name == c.iptw).unwrap();
    let opts = IptwOptions {
        estimand: estimand(&ri.estimand),
        stabilise: ri.stabilise,
        optimal_subset: None,
        ..Default::default()
    };
    let w = iptw::fit(&treat, z.as_ref(), &opts).unwrap();
    let rows = c.stop.len();
    let p = c.x[0].len();
    let x = Mat::from_fn(rows, p, |i, j| c.x[i][j]);
    let mut builder = SurvData::builder(&c.stop, &c.status);
    if let Some(start) = &c.start {
        builder = builder.start(start);
    }
    if let Some(cluster) = &c.cluster {
        builder = builder.cluster(cluster);
    }
    let data = builder.build().unwrap();
    let ties = if c.ties == "efron" {
        Ties::Efron
    } else {
        Ties::Breslow
    };
    let mut cox_opts = CoxOptions {
        ties,
        ..CoxOptions::default()
    };
    if c.start.is_some() || c.kappa.is_none() {
        // The finite-difference references were fitted to a tight tolerance.
        cox_opts.tolerance = 1e-13;
        cox_opts.max_iterations = 100;
    }
    let fit = wcox::fit(
        &data,
        x.as_ref(),
        &cox_opts,
        &Weights::iptw(&w, c.subject.as_deref()),
    )
    .unwrap();

    let eta = eta_bounds(&z, &ri.coefficients, ri.kappa);
    let omega = eta.iter().map(|e| 2.0 * e + 4.0 * EPS).fold(0.0, f64::max);
    // Kappa of the information of the Cox fit on the scaled columns: from R
    // when it recorded one, otherwise from the covariance of the fit.
    let kappa = c.kappa.unwrap_or_else(|| {
        let v = &fit.fit.covariance;
        let diag: Vec<f64> = (0..p).map(|i| v.get(i, i).unwrap().sqrt()).collect();
        let scaled = Mat::from_fn(p, p, |i, j| v.get(i, j).unwrap() / (diag[i] * diag[j]));
        let inv = epistat_linalg::inverse(scaled.as_ref(), &epistat_linalg::Attempt::SYMMETRIC)
            .unwrap()
            .x;
        let norm = |m: &Mat<f64>| {
            (0..p)
                .map(|i| (0..p).map(|j| m[(i, j)].abs()).sum::<f64>())
                .fold(0.0, f64::max)
        };
        norm(&scaled) * norm(&inv)
    });
    let bound = 8.0 * kappa * EPS;
    let big_x: f64 = (0..p)
        .map(|j| {
            (0..rows).map(|i| x[(i, j)].abs()).fold(0.0, f64::max)
                * c.coefficients[j].abs().max(1.0)
        })
        .sum();
    let rel = 16.0 * (bound * big_x + omega) + 16.0 * rows as f64 * EPS;
    let mut worst = (0.0f64, 0.0f64);
    for j in 0..p {
        let d = (fit.coefficients[j] - c.coefficients[j]).abs()
            / (bound * c.coefficients[j].abs().max(1.0) + 2.0 * omega * c.coefficients[j].abs());
        assert!(
            d <= 1.0,
            "{} coefficient {j}: {} against {}",
            c.name,
            fit.coefficients[j],
            c.coefficients[j]
        );
    }
    // The adjusted covariance is compared with the exact one, a numerical
    // Jacobian of the stacked equations or finite differences of refits.
    // numDeriv's Richardson derivatives add a relative error below 1e-7, which
    // is far above what they show against the finite differences.
    let adjusted_reference = c.exact_covariance.as_ref().unwrap_or(&c.covariance);
    let numeric = if c.exact_covariance.is_some() {
        1e-7
    } else {
        0.0
    };
    for (mine, theirs, plain, label, slot) in [
        (
            &fit.covariance,
            adjusted_reference,
            &c.covariance_plain,
            "adjusted",
            0,
        ),
        (
            &fit.robust_covariance,
            &c.robust_covariance,
            &c.robust_covariance_plain,
            "robust",
            1,
        ),
    ] {
        for i in 0..p {
            for j in 0..p {
                let scale = (theirs[i][i] * theirs[j][j]).sqrt();
                let fd = plain
                    .as_ref()
                    .map_or(0.0, |q| (q[i][j] - theirs[i][j]).abs());
                let tol = rel * scale + 2.0 * fd + if slot == 0 { numeric * scale } else { 0.0 };
                let dev = (mine.get(i, j).unwrap() - theirs[i][j]).abs() / tol;
                assert!(
                    dev <= 1.0,
                    "{} {label} covariance ({i},{j}): {} against {} (tolerance {tol:e})",
                    c.name,
                    mine.get(i, j).unwrap(),
                    theirs[i][j]
                );
                if slot == 0 {
                    worst.0 = worst.0.max(dev);
                } else {
                    worst.1 = worst.1.max(dev);
                }
            }
        }
    }
    // WeightIt's covariance is the one with the Cox score at unit weights in the
    // cross block (times G / (G - 1) with G clusters, the factor it applies).
    if let Some(shortcut) = &c.shortcut_covariance {
        let g = c.cluster.as_ref().map_or(1.0, |cl| {
            let mut v = cl.clone();
            v.sort_unstable();
            v.dedup();
            v.len() as f64
        });
        let factor = if c.cluster.is_some() {
            g / (g - 1.0)
        } else {
            1.0
        };
        let mut worst_gap = 0.0f64;
        for (i, row) in shortcut.iter().enumerate() {
            for (j, &entry) in row.iter().enumerate() {
                let weightit = c.covariance[i][j];
                let explained = entry * factor;
                let scale = (c.covariance[i][i] * c.covariance[j][j]).sqrt();
                assert!(
                    (weightit - explained).abs() <= 1e-7 * scale,
                    "{}: WeightIt's covariance is not the unit-weight form ({weightit} against {explained})",
                    c.name
                );
                worst_gap =
                    worst_gap.max((fit.covariance.get(i, j).unwrap() - weightit).abs() / scale);
            }
        }
        println!(
            "{}: WeightIt's own value differs from the exact one by up to {:.2e} of the standard error scale",
            c.name, worst_gap
        );
    }
    println!(
        "{}: worst deviation in tolerances: adjusted covariance {:.3e}, robust covariance {:.3e}",
        c.name, worst.0, worst.1
    );
    // The adjustment is not the identity: the adjusted and the robust variance differ.
    assert_ne!(fit.covariance, fit.robust_covariance);
    worst
}

#[test]
fn cox_variances_match_weightit_and_coxph() {
    let reference = load();
    for c in &reference.cox {
        cox_case(&reference, c);
    }
}

#[test]
fn variances_match_finite_differences_of_refits() {
    let reference = load();
    for c in &reference.cox_fd {
        cox_case(&reference, c);
    }
}
