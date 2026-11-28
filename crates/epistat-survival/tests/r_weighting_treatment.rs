//! Pooled logistic weights for a treatment that changes over time, checked against
//! R 4.6.2: `ipw::ipwtm(type = "first" and "all", family = "binomial")` with and
//! without a numerator model and with `trunc`, and the variance of a Cox fit
//! weighted by them that accounts for the estimated models, against the
//! infinitesimal jackknife of the whole procedure found by numerical derivatives
//! of refits (`glm` and `coxph` with case weights, the truncation limits taken
//! again from each refit's weights). See `data/weighting_treatment.R`.
//!
//! # The tolerances
//!
//! The weights follow `r_weighting_pooled.rs`: both sides run IRLS from the same
//! start with the same stopping rule, so a coefficient differs by `bound = 8 kappa
//! eps` relative to `max(|c|, 1)`, the linear predictor of a row by at most
//! `eta_j = bound sum_k |z_jk| max(|c_k|, 1)`, the log of a factor `f_n / f_d` by
//! at most `eta^den_j + eta^num_j`, since the derivative of `log p` and of `log
//! (1 - p)` with respect to the linear predictor is at most 1 in size, and the
//! weight of a row, a product over the rows of its subject up to it, by a
//! relative `2 L + 4 m eps` with `L` the sum of those bounds. A truncated weight
//! is a weight or an interpolation between two, so it has the largest of those
//! bounds over all rows.
//!
//! The covariances follow `r_weighting_estimated.rs`: an entry is a sum over `n`
//! subjects of products of influences, so an error `eta` in every influence of a
//! column moves it by at most `sqrt(n)(eta_j sqrt(V_ii) + eta_i sqrt(V_jj))`; the
//! reference has the truncation of its finite differences (twice the difference
//! between the Richardson-extrapolated and the plain covariance) and the
//! rounding of its refits (`3 * 1e-13 / h` per influence), and the weights of this
//! crate differ from R's by a relative `omega`, which moves the covariance by at
//! most `16 omega sqrt(V_ii V_jj)`; the sums of the crate add `64 rows eps`.

use epistat_linalg::Mat;
use epistat_stats::weighted::ProbabilityBounds;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::pooled::PooledOptions;
use epistat_survival::weighting::pooled::{PooledEvent, TreatmentHistory};
use epistat_survival::weighting::{OutcomeCensoring, WeightAt, Weights};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;
const H: f64 = 1e-4;
const FIT_TOLERANCE: f64 = 1e-13;

#[derive(Deserialize)]
struct Versions {
    r: String,
    ipw: String,
    survival: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    stabilise: bool,
    trunc: Option<f64>,
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    treated: Vec<u8>,
    x1: Vec<f64>,
    design: Vec<Vec<f64>>,
    numerator_design: Option<Vec<Vec<f64>>>,
    ipw_weights: Vec<f64>,
    ipw_truncated: Option<Vec<f64>>,
    estimate: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    covariance_plain: Vec<Vec<f64>>,
    known: Vec<Vec<f64>>,
    known_plain: Vec<Vec<f64>>,
    weights: Vec<f64>,
    weights_before: Vec<f64>,
    truncated: Vec<f64>,
    truncated_before: Vec<f64>,
    active: Vec<u8>,
    probability: Vec<f64>,
    coefficients: Vec<f64>,
    numerator_coefficients: Option<Vec<f64>>,
    kappa_den: f64,
    kappa_num: Option<f64>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
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

fn load() -> Reference {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/weighting_treatment.json"
    ));
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.ipw, "1.3.2");
    assert_eq!(r.versions.survival, "3.8.12");
    r
}

fn history(c: &Case) -> TreatmentHistory {
    if c.kind == "first" {
        TreatmentHistory::First
    } else {
        TreatmentHistory::All
    }
}

#[test]
fn weights_match_ipwtm() {
    let reference = load();
    for c in &reference.cases {
        let n = c.stop.len();
        let z = matrix(&c.design);
        let zn = c.numerator_design.as_ref().map(|m| matrix(m));
        let data = SurvData::builder(&c.stop, &c.status)
            .start(&c.start)
            .id(&c.id)
            .build()
            .unwrap();
        let treated: Vec<bool> = c.treated.iter().map(|&a| a == 1).collect();
        let trunc = c.trunc.map(|t| (t, 1.0 - t));
        let w = epistat_survival::weighting::pooled::fit(
            &data,
            PooledEvent::Treatment {
                treated: &treated,
                history: history(c),
            },
            z.as_ref(),
            zn.as_ref().map(Mat::as_ref),
            &PooledOptions {
                truncation: trunc.map(|(l, u): (f64, f64)| ProbabilityBounds::new(l, u).unwrap()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(c.stabilise, zn.is_some());
        assert_eq!(w.treated.as_ref(), Some(&treated));
        assert_eq!(
            w.active.iter().map(|&a| u8::from(a)).collect::<Vec<_>>(),
            c.active
        );
        let eta_d = eta_bounds(&z, &c.coefficients, c.kappa_den);
        let eta_n = zn.as_ref().map_or_else(
            || vec![0.0; n],
            |m| {
                eta_bounds(
                    m,
                    c.numerator_coefficients.as_ref().unwrap(),
                    c.kappa_num.unwrap(),
                )
            },
        );
        let untruncated = w.untruncated.as_ref().unwrap_or(&w.weight);
        let mut worst = [0.0f64; 4];
        let (mut l, mut m_rows, mut max_rel) = (0.0f64, 0usize, 0.0f64);
        let mut bounds = vec![(0.0, 0usize, 0.0, 0usize); n];
        for i in 0..n {
            if i == 0 || c.id[i] != c.id[i - 1] {
                l = 0.0;
                m_rows = 0;
            }
            let (before_l, before_m) = (l, m_rows);
            if c.active[i] == 1 {
                l += eta_d[i] + eta_n[i];
                m_rows += 1;
            }
            bounds[i] = (l, m_rows, before_l, before_m);
            max_rel = max_rel.max(2.0 * l + 4.0 * (m_rows as f64 + 1.0) * EPS);
        }
        for i in 0..n {
            let (l, rows, before_l, before_m) = bounds[i];
            for (mine, theirs, big_l, k, slot) in [
                (untruncated[i], c.weights[i], l, rows, 0),
                (untruncated[i], c.ipw_weights[i], l, rows, 1),
            ] {
                let tol = theirs * (2.0 * big_l + 4.0 * (k as f64 + 1.0) * EPS);
                let dev = (mine - theirs).abs() / tol;
                assert!(
                    dev <= 1.0,
                    "{} row {i} slot {slot}: {mine} against {theirs}",
                    c.name
                );
                worst[slot] = worst[slot].max(dev);
            }
            if c.trunc.is_none() {
                let theirs = c.weights_before[i];
                let tol = theirs * (2.0 * before_l + 4.0 * (before_m as f64 + 1.0) * EPS);
                let dev = (w.weight_before[i] - theirs).abs() / tol;
                assert!(dev <= 1.0, "{} before {i}", c.name);
                worst[2] = worst[2].max(dev);
            }
            let tol = eta_d[i] / 4.0 + 4.0 * EPS;
            let dev = (w.probability[i] - c.probability[i]).abs() / tol;
            assert!(dev <= 1.0, "{} probability {i}", c.name);
            worst[3] = worst[3].max(dev);
        }
        if let Some(t) = &c.ipw_truncated {
            assert!(w.untruncated.is_some());
            for (i, &ipw) in t.iter().enumerate() {
                for (mine, theirs, what) in [
                    (w.weight[i], ipw, "against ipwtm"),
                    (w.weight[i], c.truncated[i], "against the by-hand"),
                    (w.weight_before[i], c.truncated_before[i], "before"),
                ] {
                    let tol = theirs * (max_rel + 2.0 * EPS);
                    assert!(
                        (mine - theirs).abs() <= tol,
                        "{} truncated row {i} {what}: {mine} against {theirs}",
                        c.name
                    );
                }
            }
            let (lo, hi) = (
                w.weight.iter().copied().fold(f64::INFINITY, f64::min),
                w.weight.iter().copied().fold(0.0, f64::max),
            );
            assert!(
                untruncated.iter().any(|&u| u < lo) || untruncated.iter().any(|&u| u > hi),
                "{}: truncation did nothing",
                c.name
            );
        } else {
            assert!(w.untruncated.is_none());
        }
        println!(
            "{}: worst deviation in tolerances: weight {:.3e}, against ipwtm {:.3e}, before {:.3e}, probability {:.3e}",
            c.name, worst[0], worst[1], worst[2], worst[3]
        );
    }
}

#[test]
fn variances_match_the_jackknife_of_refits() {
    let reference = load();
    for c in &reference.cases {
        let rows = c.stop.len();
        let z = matrix(&c.design);
        let zn = c.numerator_design.as_ref().map(|m| matrix(m));
        let data = SurvData::builder(&c.stop, &c.status)
            .start(&c.start)
            .id(&c.id)
            .build()
            .unwrap();
        let treated: Vec<bool> = c.treated.iter().map(|&a| a == 1).collect();
        let trunc = c.trunc.map(|t| (t, 1.0 - t));
        let w = epistat_survival::weighting::pooled::fit(
            &data,
            PooledEvent::Treatment {
                treated: &treated,
                history: history(c),
            },
            z.as_ref(),
            zn.as_ref().map(Mat::as_ref),
            &PooledOptions {
                truncation: trunc.map(|(l, u): (f64, f64)| ProbabilityBounds::new(l, u).unwrap()),
                ..Default::default()
            },
        )
        .unwrap();
        let x = Mat::from_fn(rows, 2, |i, j| {
            if j == 0 {
                f64::from(c.treated[i])
            } else {
                c.x1[i]
            }
        });
        let opts = CoxOptions {
            tolerance: FIT_TOLERANCE,
            max_iterations: 100,
            ..CoxOptions::default()
        };
        let estimated = Weights {
            iptw: None,
            subject: None,
            censoring: Some(OutcomeCensoring::Pooled {
                fit: &w,
                at: WeightAt::Stop,
            }),
        };
        let fit = wcox::fit(&data, x.as_ref(), &opts, &estimated).unwrap();
        let n = fit.n_units as f64;
        let used = c.ipw_truncated.as_ref().unwrap_or(&c.ipw_weights);
        let omega = w
            .weight
            .iter()
            .zip(used)
            .map(|(a, b)| (a - b).abs() / b)
            .fold(0.0, f64::max);
        let m = 2;
        let flat = |v: &epistat_survival::Covariance| -> Vec<Vec<f64>> {
            (0..m)
                .map(|i| (0..m).map(|j| v.get(i, j).unwrap()).collect())
                .collect()
        };
        let mut worst = [0.0f64; 3];
        for (slot, (mine, theirs, plain)) in [
            (flat(&fit.covariance), &c.covariance, &c.covariance_plain),
            (flat(&fit.robust_covariance), &c.known, &c.known_plain),
        ]
        .into_iter()
        .enumerate()
        {
            for i in 0..m {
                for j in 0..m {
                    let scale = (theirs[i][i] * theirs[j][j]).sqrt();
                    let eta = |k: usize| 3.0 * FIT_TOLERANCE / H * c.estimate[k].abs().max(1.0);
                    let tol = 2.0 * (plain[i][j] - theirs[i][j]).abs()
                        + n.sqrt() * (eta(j) * theirs[i][i].sqrt() + eta(i) * theirs[j][j].sqrt())
                        + (16.0 * omega + 64.0 * rows as f64 * EPS) * scale;
                    let dev = (mine[i][j] - theirs[i][j]).abs() / tol;
                    assert!(
                        dev <= 1.0,
                        "{} covariance slot {slot} ({i},{j}): {} against {} (tolerance {tol:e})",
                        c.name,
                        mine[i][j],
                        theirs[i][j]
                    );
                    worst[slot] = worst[slot].max(dev);
                }
            }
        }
        for i in 0..m {
            let tol = 16.0 * omega * (n * c.known[i][i]).sqrt() + 64.0 * rows as f64 * EPS;
            let dev = (fit.coefficients[i] - c.estimate[i]).abs() / tol;
            assert!(dev <= 1.0, "{} estimate {i}", c.name);
            worst[2] = worst[2].max(dev);
        }
        let differs = (0..m).any(|i| {
            (fit.covariance.get(i, i).unwrap() - fit.robust_covariance.get(i, i).unwrap()).abs()
                > 1e-6 * fit.robust_covariance.get(i, i).unwrap()
        });
        assert!(
            differs,
            "{}: adjusted and known variance are the same",
            c.name
        );
        println!(
            "{}: omega {omega:.2e}; worst deviation in tolerances: adjusted {:.3e}, known {:.3e}, estimate {:.3e}",
            c.name, worst[0], worst[1], worst[2]
        );
    }
}
