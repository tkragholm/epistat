//! Overlap, matching and optimal-subset weights checked against R 4.6.2:
//! `WeightIt::weightit` with `estimand` ATO, ATM and ATOS, the weights of
//! `PSweight::SumStat`, and the variance of a weighted Cox fit that accounts for
//! the estimated propensity model, against central differences of refits. See
//! `data/overlap.R` for the construction.
//!
//! # The tolerances
//!
//! As in `r_weighting.rs`: both fits run IRLS from the same start with the same
//! stopping rule, so a coefficient differs by `bound = 8 kappa eps` relative to
//! `max(|c|, 1)` and the linear predictor of subject `i` by at most `eta_i =
//! bound sum_j |z_ij| max(|c_j|, 1)`. Every weight here has `|d log w / d eta| <=
//! 1`, so its relative error is at most `2 eta_i + 4 eps`, and a fitted probability
//! differs by at most `eta_i / 4 + 4 eps`.
//!
//! - The balance gap `sum a_i w_i z_ij - sum (1 - a_i) w_i z_ij` of the overlap
//!   weights is `sum (a_i - p_i) z_ij`, which is 0 at the maximum likelihood
//!   estimate, so it is at most `sum |z_ij| (eta_i / 4 + 4 eps)` plus the
//!   rounding of the sum, `4 n eps sum |z_ij|`.
//! - PSweight divides the weights by their sum in the group, so its weights agree
//!   to `4 omega + 16 n eps` relative, with `omega` the largest relative error of
//!   a weight.
//! - The cut-off of the optimal subset is a function of sorted probabilities and
//!   is compared exactly in which subjects it removes, and to `1e-9` in value.
//! - The covariances are sums over `n` subjects of products of influences, so
//!   they agree to `16 (bound X + omega) + 16 n eps` relative to `sqrt(V_ii
//!   V_jj)`, plus twice the difference between the Richardson-extrapolated and
//!   the plain central-difference covariance of the reference.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::iptw::{self, Estimand, IptwOptions};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    #[serde(rename = "WeightIt")]
    weightit: String,
    #[serde(rename = "PSweight")]
    psweight: String,
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
struct RWeights {
    name: String,
    estimand: String,
    weights: Vec<f64>,
    propensity: Vec<f64>,
    coefficients: Vec<f64>,
    treated: Summary,
    control: Summary,
    kappa: f64,
    alpha: Option<f64>,
    balance_gap: f64,
    #[serde(default)]
    psweight: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct RCox {
    estimand: String,
    coefficients: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    robust_covariance: Vec<Vec<f64>>,
    covariance_plain: Vec<Vec<f64>>,
    robust_covariance_plain: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    design: Vec<Vec<f64>>,
    treat: Vec<u8>,
    stop: Vec<f64>,
    status: Vec<i32>,
    age: Vec<f64>,
    weights: Vec<RWeights>,
    cox: Vec<RCox>,
}

fn load() -> Reference {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/overlap.json"
    ));
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.survival, "3.8.12");
    assert_eq!(r.versions.weightit, "2.1.0");
    assert_eq!(r.versions.psweight, "2.1.2");
    r
}

fn estimand(name: &str) -> Estimand {
    match name {
        "ATO" => Estimand::Ato,
        "ATM" => Estimand::Atm,
        "ATOS" => Estimand::Atos,
        other => panic!("estimand {other}"),
    }
}

fn eta_bounds(z: &Mat<f64>, coefficients: &[f64], kappa: f64) -> Vec<f64> {
    let bound = 8.0 * kappa * EPS;
    (0..z.nrows())
        .map(|i| {
            bound
                * (0..z.ncols())
                    .map(|j| z[(i, j)].abs() * coefficients[j].abs().max(1.0))
                    .sum::<f64>()
        })
        .collect()
}

fn inputs(r: &Reference) -> (Mat<f64>, Vec<bool>) {
    let n = r.treat.len();
    (
        Mat::from_fn(n, r.design[0].len(), |i, j| r.design[i][j]),
        r.treat.iter().map(|&a| a == 1).collect(),
    )
}

#[test]
fn weights_match_weightit_and_psweight() {
    let reference = load();
    let (z, treat) = inputs(&reference);
    let n = treat.len();
    for r in &reference.weights {
        let opts = IptwOptions {
            estimand: estimand(&r.estimand),
            stabilise: false,
        };
        let w = iptw::fit(&treat, z.as_ref(), &opts).unwrap();
        let eta = eta_bounds(&z, &r.coefficients, r.kappa);
        let mut worst = 0.0f64;
        for i in 0..n {
            if r.weights[i] == 0.0 {
                assert_eq!(w.weights[i], 0.0, "{} weight {i}", r.name);
                continue;
            }
            let tol = r.weights[i] * (2.0 * eta[i] + 4.0 * EPS);
            let d = (w.weights[i] - r.weights[i]).abs() / tol;
            assert!(d <= 1.0, "{} weight {i}: {} against {}", r.name, w.weights[i], r.weights[i]);
            worst = worst.max(d);
            let dp = (w.propensity[i] - r.propensity[i]).abs() / (eta[i] / 4.0 + 4.0 * EPS);
            assert!(dp <= 1.0, "{} propensity {i}", r.name);
        }
        if let Some(alpha) = r.alpha {
            let mine = w.alpha.unwrap();
            assert!((mine - alpha).abs() <= 1e-9, "{} alpha {mine} against {alpha}", r.name);
        } else {
            assert!(w.alpha.is_none());
        }
        let omega = eta.iter().map(|e| 2.0 * e + 4.0 * EPS).fold(0.0, f64::max);
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
                    (m - t).abs() <= rel * t.abs() + 4.0 * EPS,
                    "{} {label} {what}: {m} against {t}",
                    r.name
                );
            }
        }
        if let Some(ps) = &r.psweight {
            for group in [true, false] {
                let sum: f64 = (0..n).filter(|&i| treat[i] == group).map(|i| w.weights[i]).sum();
                for i in (0..n).filter(|&i| treat[i] == group) {
                    let mine = w.weights[i] / sum;
                    assert!(
                        (mine - ps[i]).abs() <= rel * ps[i],
                        "{} PSweight weight {i}: {mine} against {}",
                        r.name,
                        ps[i]
                    );
                }
            }
        }
        if r.name == "ato" {
            for j in 0..z.ncols() {
                let gap: f64 = (0..n)
                    .map(|i| z[(i, j)] * w.weights[i] * if treat[i] { 1.0 } else { -1.0 })
                    .sum();
                let tol: f64 = (0..n)
                    .map(|i| z[(i, j)].abs() * (eta[i] / 4.0 + 4.0 * EPS + 4.0 * n as f64 * EPS))
                    .sum();
                assert!(gap.abs() <= tol, "overlap balance of column {j}: {gap}");
            }
            assert!(r.balance_gap.is_finite());
        }
        println!("{}: worst deviation in tolerances: weights {worst:.3e}", r.name);
    }
}

#[test]
fn caller_alpha_trims_the_optimal_subset() {
    let reference = load();
    let (z, treat) = inputs(&reference);
    let crump = iptw::fit_atos(&treat, z.as_ref(), None).unwrap();
    let same = iptw::fit_atos(&treat, z.as_ref(), crump.alpha).unwrap();
    assert_eq!(crump.weights, same.weights);
    let wide = iptw::fit_atos(&treat, z.as_ref(), Some(0.05)).unwrap();
    for (i, &p) in wide.propensity.iter().enumerate() {
        let inside = (0.05..=0.95).contains(&p);
        assert_eq!(wide.weights[i] != 0.0, inside);
    }
    assert!(iptw::fit_atos(&treat, z.as_ref(), Some(0.5)).is_err());
    assert!(iptw::fit_atos(&treat, z.as_ref(), Some(-0.1)).is_err());
}

#[test]
fn cox_variances_match_finite_differences_of_refits() {
    let reference = load();
    let (z, treat) = inputs(&reference);
    let n = treat.len();
    let p = 2;
    for c in &reference.cox {
        let ri = reference
            .weights
            .iter()
            .find(|r| r.estimand == c.estimand)
            .unwrap();
        let opts = IptwOptions {
            estimand: estimand(&c.estimand),
            stabilise: false,
        };
        let w = iptw::fit(&treat, z.as_ref(), &opts).unwrap();
        // Rows of weight 0 do not enter the fit.
        let keep: Vec<usize> = (0..n).filter(|&i| w.weights[i] > 0.0).collect();
        let stop: Vec<f64> = keep.iter().map(|&i| reference.stop[i]).collect();
        let status: Vec<i32> = keep.iter().map(|&i| reference.status[i]).collect();
        let subject: Vec<u32> = keep.iter().map(|&i| i as u32).collect();
        let x = Mat::from_fn(keep.len(), p, |k, j| {
            if j == 0 {
                f64::from(u8::from(treat[keep[k]]))
            } else {
                reference.age[keep[k]]
            }
        });
        let data = SurvData::builder(&stop, &status).build().unwrap();
        let cox_opts = CoxOptions {
            eps: 1e-13,
            iter_max: 100,
            ..CoxOptions::default()
        };
        let fit = wcox::fit(&data, x.as_ref(), &cox_opts, &w, Some(&subject)).unwrap();
        let eta = eta_bounds(&z, &ri.coefficients, ri.kappa);
        let omega = eta.iter().map(|e| 2.0 * e + 4.0 * EPS).fold(0.0, f64::max);
        let v = &fit.fit.covariance;
        let diag: Vec<f64> = (0..p).map(|i| v[i * p + i].unwrap().sqrt()).collect();
        let scaled = Mat::from_fn(p, p, |i, j| v[i * p + j].unwrap() / (diag[i] * diag[j]));
        let inv = epistat_linalg::inverse(scaled.as_ref(), &epistat_linalg::Attempt::SYMMETRIC)
            .unwrap()
            .x;
        let norm = |m: &Mat<f64>| {
            (0..p)
                .map(|i| (0..p).map(|j| m[(i, j)].abs()).sum::<f64>())
                .fold(0.0, f64::max)
        };
        let kappa = norm(&scaled) * norm(&inv);
        let bound = 8.0 * kappa * EPS;
        let big_x: f64 = (0..p)
            .map(|j| {
                (0..keep.len()).map(|i| x[(i, j)].abs()).fold(0.0, f64::max)
                    * c.coefficients[j].abs().max(1.0)
            })
            .sum();
        let rel = 16.0 * (bound * big_x + omega) + 16.0 * n as f64 * EPS;
        for j in 0..p {
            let d = (fit.coefficients[j] - c.coefficients[j]).abs()
                / (bound * c.coefficients[j].abs().max(1.0) + 2.0 * omega * c.coefficients[j].abs());
            assert!(d <= 1.0, "{} coefficient {j}", c.estimand);
        }
        let mut worst = (0.0f64, 0.0f64);
        for (mine, theirs, plain, slot) in [
            (&fit.covariance, &c.covariance, &c.covariance_plain, 0),
            (&fit.robust_covariance, &c.robust_covariance, &c.robust_covariance_plain, 1),
        ] {
            for i in 0..p {
                for j in 0..p {
                    let scale = (theirs[i][i] * theirs[j][j]).sqrt();
                    let tol = rel * scale + 2.0 * (plain[i][j] - theirs[i][j]).abs();
                    let dev = (mine[i * p + j] - theirs[i][j]).abs() / tol;
                    assert!(
                        dev <= 1.0,
                        "{} covariance {slot} ({i},{j}): {} against {} (tolerance {tol:e})",
                        c.estimand,
                        mine[i * p + j],
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
        assert_ne!(fit.covariance, fit.robust_covariance);
        println!(
            "{}: worst deviation in tolerances: adjusted {:.3e}, robust {:.3e}",
            c.estimand, worst.0, worst.1
        );
    }
}
