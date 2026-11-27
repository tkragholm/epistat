//! The variance of weighted Cox fits and Kaplan-Meier curves with estimated
//! censoring weights (pooled logistic, Kaplan-Meier, Cox) and with treatment and
//! censoring weights together, checked against the infinitesimal jackknife of the
//! whole procedure found by numerical derivatives of refits in R 4.6.2 (glm,
//! survfit and coxph with case weights, central differences with Richardson
//! extrapolation). See `data/weighting_estimated.R` for the construction.
//!
//! # The tolerances
//!
//! An entry `V_ij` of a covariance is a sum over `n` subjects of products of
//! influences, `IF_ui IF_uj`, so an error `eta_i` in every influence of
//! column `i` changes it by at most `sqrt(n)(eta_j sqrt(V_ii) + eta_i sqrt(V_jj))`
//! (Cauchy-Schwarz with `sum_u IF_ui^2 = V_ii`). The reference has three errors.
//!
//! - Truncation of the finite difference: twice the difference between the
//!   covariance from the Richardson-extrapolated derivatives and from the plain
//!   central differences with the larger step, which bounds the error of the
//!   extrapolated one (as in `r_weighting.rs`).
//! - Rounding of the refits: each refit is converged to a relative `1e-13`, and a
//!   central difference with step `h = 1e-4` divides the difference of two refits
//!   by `2 h`. After Richardson's `(4 D(h/2) - D(h)) / 3` the influence has an error
//!   of at most `3 * 1e-13 / h` times `max(|estimate|, 1)`.
//! - The weights of this crate differ from those of R by a relative `omega`,
//!   measured against the weights R printed. The covariance is moved by at most
//!   `16 omega` times `sqrt(V_ii V_jj)`, the factor of `r_weighting.rs`.
//!
//! The crate's own error is the rounding of sums of `n` terms, `64 n eps`
//! relative to `sqrt(V_ii V_jj)`.

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties};
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::ipcw::{self, CensoringModel, IpcwOptions};
use epistat_survival::weighting::iptw::{self, Estimand, IptwOptions};
use epistat_survival::weighting::km::{self, KmOptions};
use epistat_survival::weighting::pooled;
use epistat_survival::weighting::pooled::PooledEvent;
use epistat_survival::weighting::pooled::PooledOptions;
use epistat_survival::weighting::{Censoring, WeightAt, Weights};
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;
/// The step of the finite differences and the convergence of the refits in R.
const H: f64 = 1e-4;
const FIT_TOLERANCE: f64 = 1e-13;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
}

#[derive(Deserialize)]
struct IptwSpec {
    stab: bool,
}

#[derive(Deserialize)]
struct CensSpec {
    model: String,
    stab: bool,
    at: String,
    by_group: Option<bool>,
    xcols: Option<Vec<String>>,
    ties: Option<String>,
}

#[derive(Deserialize)]
struct Spec {
    iptw: Option<IptwSpec>,
    cens: Option<CensSpec>,
    outcome: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    spec: Spec,
    n_units: usize,
    cluster: Option<Vec<u32>>,
    id: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    a: Vec<u32>,
    x1: Vec<f64>,
    l: Vec<f64>,
    weights: Vec<f64>,
    estimate: Vec<f64>,
    covariance: Vec<Vec<f64>>,
    covariance_plain: Vec<Vec<f64>>,
    known: Vec<Vec<f64>>,
    known_plain: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    times: Vec<f64>,
    z_den: Vec<Vec<f64>>,
    z_num: Vec<Vec<f64>>,
    z_iptw: Vec<Vec<f64>>,
    treat: Vec<u8>,
    cases: Vec<Case>,
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn column(c: &Case, name: &str) -> Vec<f64> {
    match name {
        "a" => c.a.iter().map(|&v| f64::from(v)).collect(),
        "x1" => c.x1.clone(),
        "l" => c.l.clone(),
        other => panic!("column {other}"),
    }
}

/// The covariances of an estimate: with the weights estimated, and as known, as
/// matrices over the entries of the reference.
struct Output {
    estimate: Vec<f64>,
    adjusted: Vec<Vec<f64>>,
    known: Vec<Vec<f64>>,
    weights: Vec<f64>,
}

fn run(reference: &Reference, c: &Case) -> Output {
    let rows = c.stop.len();
    let z = matrix(&reference.z_den);
    let zn = matrix(&reference.z_num);
    let zi = matrix(&reference.z_iptw);
    let treat: Vec<bool> = reference.treat.iter().map(|&a| a == 1).collect();
    let stratum: Vec<u32> = c.a.clone();
    let km_outcome = c.spec.outcome == "km";
    let mut builder = SurvData::builder(&c.stop, &c.status)
        .start(&c.start)
        .id(&c.id);
    if km_outcome {
        builder = builder.stratum(&stratum);
    }
    if let Some(cl) = &c.cluster {
        builder = builder.cluster(cl);
    }
    let data = builder.build().unwrap();

    let iptw_fit = c.spec.iptw.as_ref().map(|s| {
        iptw::fit(
            &treat,
            zi.as_ref(),
            &IptwOptions {
                estimand: Estimand::Ate,
                stabilise: s.stab,
                alpha: None,
            },
        )
        .unwrap()
    });
    let pooled_fit = c
        .spec
        .cens
        .as_ref()
        .filter(|s| s.model == "pooled")
        .map(|s| {
            pooled::fit(
                &data,
                PooledEvent::Censoring { censored: None },
                z.as_ref(),
                s.stab.then(|| zn.as_ref()),
                &PooledOptions::default(),
            )
            .unwrap()
        });
    let cens_cols: Vec<Vec<f64>> = c
        .spec
        .cens
        .as_ref()
        .and_then(|s| s.xcols.as_ref())
        .map(|cols| cols.iter().map(|name| column(c, name)).collect())
        .unwrap_or_default();
    let x_cens = Mat::from_fn(rows, cens_cols.len(), |i, j| cens_cols[j][i]);
    let ipcw_opts = c.spec.cens.as_ref().map(|s| IpcwOptions {
        stabilise: s.stab,
        ties: if s.ties.as_deref() == Some("breslow") {
            Ties::Breslow
        } else {
            Ties::Efron
        },
        ..IpcwOptions::default()
    });
    let at = |s: &CensSpec| {
        if s.at == "stop" {
            WeightAt::Stop
        } else {
            WeightAt::Start
        }
    };
    // The censoring model of the hazard is stratified as the outcome data is, for
    // the Kaplan-Meier outcome; the cases name which models use the strata.
    let hazard_data = if c
        .spec
        .cens
        .as_ref()
        .is_some_and(|s| s.by_group == Some(true))
    {
        SurvData::builder(&c.stop, &c.status)
            .start(&c.start)
            .id(&c.id)
            .stratum(&stratum)
            .build()
            .unwrap()
    } else {
        SurvData::builder(&c.stop, &c.status)
            .start(&c.start)
            .id(&c.id)
            .build()
            .unwrap()
    };
    assert!(
        c.spec
            .cens
            .as_ref()
            .is_none_or(|s| s.model == "pooled" || (s.by_group == Some(true)) == km_outcome),
        "{}: the outcome data has the strata of the hazard model",
        c.name
    );
    let censoring = c.spec.cens.as_ref().map(|s| match s.model.as_str() {
        "pooled" => Censoring::Pooled {
            fit: pooled_fit.as_ref().unwrap(),
            at: at(s),
        },
        "km" => Censoring::Hazard {
            model: CensoringModel::KaplanMeier,
            options: ipcw_opts.as_ref().unwrap(),
            at: at(s),
        },
        _ => Censoring::Hazard {
            model: CensoringModel::Cox { x: x_cens.as_ref() },
            options: ipcw_opts.as_ref().unwrap(),
            at: at(s),
        },
    });
    let subject: Vec<u32> = c.id.clone();
    let estimated = Weights {
        iptw: iptw_fit.as_ref(),
        subject: iptw_fit.as_ref().map(|_| subject.as_slice()),
        censoring,
    };
    // The weights: the product of those of the parts.
    let mut weights = vec![1.0; rows];
    if let Some(w) = &iptw_fit {
        for (i, x) in weights.iter_mut().enumerate() {
            *x *= w.weights[c.id[i] as usize];
        }
    }
    if let Some(p) = &pooled_fit {
        let s = c.spec.cens.as_ref().unwrap();
        for (i, x) in weights.iter_mut().enumerate() {
            *x *= if s.at == "stop" {
                p.weight[i]
            } else {
                p.weight_before[i]
            };
        }
    }
    if let Some(s) = c.spec.cens.as_ref().filter(|s| s.model != "pooled") {
        let model = if s.model == "km" {
            CensoringModel::KaplanMeier
        } else {
            CensoringModel::Cox { x: x_cens.as_ref() }
        };
        let w = ipcw::fit(&hazard_data, model, ipcw_opts.as_ref().unwrap()).unwrap();
        for (i, x) in weights.iter_mut().enumerate() {
            *x *= if s.at == "stop" {
                w.weight_at_stop[i]
            } else {
                w.weight_at_start[i]
            };
        }
    }
    if km_outcome {
        let t = reference.times.len();
        let est = km::estimate(&data, &reference.times, &KmOptions::default(), &estimated).unwrap();
        assert_eq!(est.groups, vec![0, 1]);
        let m = 2 * t;
        let mut adjusted = vec![vec![f64::NAN; m]; m];
        let mut known = vec![vec![f64::NAN; m]; m];
        for ti in 0..t {
            for g in 0..2 {
                for h in 0..2 {
                    adjusted[g * t + ti][h * t + ti] =
                        est.covariance[(ti * 2 + g) * 2 + h].unwrap();
                    known[g * t + ti][h * t + ti] =
                        est.covariance_known[(ti * 2 + g) * 2 + h].unwrap();
                }
            }
        }
        Output {
            estimate: est.surv.clone(),
            adjusted,
            known,
            weights,
        }
    } else {
        let x = Mat::from_fn(rows, 3, |i, j| match j {
            0 => f64::from(c.a[i]),
            1 => c.x1[i],
            _ => c.l[i],
        });
        let opts = CoxOptions {
            eps: FIT_TOLERANCE,
            iter_max: 100,
            ..CoxOptions::default()
        };
        let fit = wcox::fit(&data, x.as_ref(), &opts, &estimated).unwrap();
        let p = 3;
        let flat = |v: &[f64]| {
            (0..p)
                .map(|i| (0..p).map(|j| v[i * p + j]).collect())
                .collect()
        };
        Output {
            estimate: fit.coefficients.clone(),
            adjusted: flat(&fit.covariance),
            known: flat(&fit.robust_covariance),
            weights,
        }
    }
}

#[test]
fn variances_with_estimated_weights_match_the_jackknife_of_refits() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/weighting_estimated.json"
    ));
    assert!(reference.versions.r.starts_with("R version 4.6"));
    assert_eq!(reference.versions.survival, "3.8.12");
    for c in &reference.cases {
        let n = c.n_units as f64;
        let out = run(&reference, c);
        let omega = out
            .weights
            .iter()
            .zip(&c.weights)
            .map(|(a, b)| (a - b).abs() / b)
            .fold(0.0, f64::max);
        let m = c.estimate.len();
        assert_eq!(out.estimate.len(), m);
        let rows = c.stop.len() as f64;
        let mut worst = [0.0f64; 3];
        for (slot, (mine, theirs, plain)) in [
            (&out.adjusted, &c.covariance, &c.covariance_plain),
            (&out.known, &c.known, &c.known_plain),
        ]
        .into_iter()
        .enumerate()
        {
            for i in 0..m {
                for j in 0..m {
                    if mine[i][j].is_nan() {
                        continue;
                    }
                    let scale = (theirs[i][i] * theirs[j][j]).sqrt();
                    let eta = |k: usize| 3.0 * FIT_TOLERANCE / H * c.estimate[k].abs().max(1.0);
                    let tol = 2.0 * (plain[i][j] - theirs[i][j]).abs()
                        + n.sqrt() * (eta(j) * theirs[i][i].sqrt() + eta(i) * theirs[j][j].sqrt())
                        + (16.0 * omega + 64.0 * rows * EPS) * scale;
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
        // The estimate itself moves with the weights by at most the sum of the
        // absolute influences of the units times omega, and that is at most
        // sqrt(n V_ii) omega by Cauchy-Schwarz.
        for i in 0..m {
            let tol = 16.0 * omega * (n * c.known[i][i]).sqrt() + 64.0 * rows * EPS;
            let dev = (out.estimate[i] - c.estimate[i]).abs() / tol;
            assert!(
                dev <= 1.0,
                "{} estimate {i}: {} against {} (tolerance {tol:e})",
                c.name,
                out.estimate[i],
                c.estimate[i]
            );
            worst[2] = worst[2].max(dev);
        }
        println!(
            "{}: omega {:.2e}; worst deviation in tolerances: adjusted covariance {:.3e}, known covariance {:.3e}, estimate {:.3e}",
            c.name, omega, worst[0], worst[1], worst[2]
        );
        // Estimating the weights changes the variance, except for curves weighted by
        // the Kaplan-Meier censoring survival of their own strata at the start of
        // the rows, where the two variances are the same in R as well: the weighted
        // curve is then the Kaplan-Meier curve of the outcome.
        assert!(
            c.name == "km_km"
                || (0..m)
                    .any(|i| (out.adjusted[i][i] - out.known[i][i]).abs() > 1e-6 * out.known[i][i]),
            "{}: the adjusted and the known variance are the same",
            c.name
        );
    }
}
