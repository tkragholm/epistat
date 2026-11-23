//! `flexpar::fit`, `predict` and `contrast` against R: `survreg` for the
//! one-degree-of-freedom models, `flexsurv::flexsurvspline`, `rstpm2::stpm2`
//! and a transcription of the likelihood (`tests/data/flexpar.R`). Each
//! quantity is compared to a stated bound, and the worst ratio of deviation to
//! bound of each kind is printed with `--nocapture`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "statuses and clusters are small whole numbers written by R as doubles"
)]

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::flexpar::{
    Contrast, FlexParFit, FlexParOptions, Predict, PredictOptions, Scale, TimeVarying, contrast,
    fit, predict,
};
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    flexsurv: String,
    rstpm2: String,
    #[serde(rename = "numDeriv")]
    num_deriv: String,
    #[serde(rename = "Hmisc")]
    hmisc: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Dataset {
    time: Vec<f64>,
    start: Vec<f64>,
    event: Vec<f64>,
    weight: Vec<f64>,
    cluster: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    x4: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Tvc {
    column: usize,
    df: usize,
}

#[derive(Deserialize)]
struct Band {
    #[serde(rename = "Estimate")]
    estimate: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Curves {
    Plain(Vec<Vec<f64>>),
    Bands(Vec<Band>),
}

impl Curves {
    fn estimate(&self, profile: usize) -> &[f64] {
        match self {
            Self::Plain(v) => &v[profile],
            Self::Bands(v) => &v[profile].estimate,
        }
    }
    fn band(&self, profile: usize) -> Option<(&[f64], &[f64])> {
        match self {
            Self::Plain(_) => None,
            Self::Bands(v) => Some((&v[profile].lower, &v[profile].upper)),
        }
    }
}

#[derive(Deserialize)]
struct Quantity {
    estimate: Vec<f64>,
    se: Option<Vec<f64>>,
    se_log: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct RFit {
    name: String,
    source: String,
    dataset: String,
    scale: String,
    df: usize,
    weighted: bool,
    knots: Option<Vec<f64>>,
    covariates: Vec<String>,
    tvc: Vec<Tvc>,
    loglik: f64,
    gamma: Option<Vec<f64>>,
    beta: Option<Vec<f64>>,
    beta_se: Option<Vec<f64>>,
    surv: Option<Curves>,
    hazard: Option<Curves>,
    cumhaz: Option<Curves>,
    hr_profile: Option<Vec<f64>>,
    hr: Option<Band>,
    theta: Option<Vec<f64>>,
    cov: Option<Vec<f64>>,
    robust_cov: Option<Vec<f64>>,
    profiles: Option<Vec<Vec<f64>>>,
    predictions: Option<HashMap<String, Quantity>>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    times: Vec<f64>,
    profiles: Vec<Vec<f64>>,
    data: HashMap<String, Dataset>,
    fits: Vec<RFit>,
}

fn scale_of(name: &str) -> Scale {
    match name {
        "hazard" => Scale::Hazard,
        "odds" => Scale::Odds,
        "probit" => Scale::Probit,
        other => panic!("{other}"),
    }
}

/// The worst ratio of deviation to bound for each kind of comparison.
#[derive(Default)]
struct Worst(BTreeMap<String, (f64, String)>, String, String);

impl Worst {
    fn note(&mut self, kind: &str, deviation: f64, bound: f64) {
        assert!(deviation.is_finite(), "{kind}: {deviation}");
        let name = self.1.clone();
        let kind = kind.replacen(&self.2.split(' ').next().unwrap().to_string(), &self.2, 1);
        let e = self.0.entry(kind).or_insert((0.0, name.clone()));
        if deviation / bound > e.0 {
            *e = (deviation / bound, name);
        }
    }
    /// `|a - b| / max(|b|, floor)`.
    fn rel(&mut self, kind: &str, a: f64, b: f64, floor: f64, bound: f64) {
        self.note(kind, (a - b).abs() / b.abs().max(floor), bound);
    }
}

struct Run {
    fit: FlexParFit,
    x: Mat<f64>,
}

fn run(reference: &Reference, r: &RFit) -> Run {
    let d = &reference.data[&r.dataset];
    let n = d.time.len();
    let status: Vec<i32> = d.event.iter().map(|&e| e as i32).collect();
    let cluster: Vec<u32> = d.cluster.iter().map(|&c| c as u32).collect();
    let mut builder = SurvData::builder(&d.time, &status);
    if r.dataset != "A" {
        builder = builder.start(&d.start);
    }
    if r.weighted {
        builder = builder.weight(&d.weight);
    }
    if r.source == "transcription" {
        builder = builder.cluster(&cluster);
    }
    let data = builder.build().unwrap();
    let columns: Vec<&Vec<f64>> = r
        .covariates
        .iter()
        .map(|c| match c.as_str() {
            "x1" => &d.x1,
            "x2" => &d.x2,
            "x4" => d.x4.as_ref().unwrap(),
            other => panic!("{other}"),
        })
        .collect();
    let x = Mat::from_fn(n, columns.len(), |i, j| columns[j][i]);
    let tvc: Vec<TimeVarying> = r
        .tvc
        .iter()
        .map(|t| TimeVarying {
            column: t.column,
            df: t.df,
        })
        .collect();
    let options = FlexParOptions {
        scale: scale_of(&r.scale),
        df: r.df,
        knots: r.knots.as_deref(),
        tvc: &tvc,
        ..FlexParOptions::default()
    };
    let f = fit(&data, x.as_ref(), &options).unwrap();
    assert!(f.converged, "{} did not converge", r.name);
    Run { fit: f, x }
}

fn profile_matrix(profiles: &[Vec<f64>], p: usize) -> Mat<f64> {
    Mat::from_fn(1, profiles[p].len(), |_, j| profiles[p][j])
}

#[test]
fn flexpar_against_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/flexpar.json"
    ));
    let v = &reference.versions;
    assert!(v.r.starts_with("R version 4.6.2"), "{}", v.r);
    assert_eq!(
        (v.survival.as_str(), v.flexsurv.as_str(), v.rstpm2.as_str()),
        ("3.8.12", "2.3.2", "1.7.1"),
        "numDeriv {} Hmisc {} jsonlite {}",
        v.num_deriv,
        v.hmisc,
        v.jsonlite
    );
    let mut worst = Worst::default();
    let options = PredictOptions::default();
    for r in &reference.fits {
        worst.1.clone_from(&r.name);
        worst.2 = format!(
            "{}{}",
            r.source,
            if !r.tvc.is_empty() {
                " tvc"
            } else if r.dataset == "C" {
                " counting"
            } else {
                ""
            }
        );
        let out = run(&reference, r);
        let f = &out.fit;
        let q = f.dimension();
        let cov = f.covariance.as_ref().unwrap();
        let se = |k: usize| cov[k * q + k].unwrap().sqrt();
        // the log likelihood
        // The packages stop their optimisers at a relative change of 1e-14 in the
        // log likelihood, which fixes it to about 1e-11 relative. flexsurv's probit
        // fits stop earlier, about 1e-7 from the maximum, so their bounds are
        // 10^4 times wider. The transcription is solved to a score of 1e-10.
        let loose = if r.source == "flexsurv" && r.scale == "probit" {
            1e4
        } else {
            1.0
        };
        let tol_ll = if r.source == "transcription" {
            1e-10
        } else {
            1e-10 * loose
        };
        worst.rel(
            &format!("{} loglik", r.source),
            f.log_likelihood,
            r.loglik,
            1.0,
            tol_ll,
        );
        // the coefficients of the covariates, where they do not depend on the basis
        if let (Some(b), Some(bse)) = (&r.beta, &r.beta_se) {
            for (j, (&bj, &sj)) in b.iter().zip(bse).enumerate() {
                let got = f.beta[j].unwrap();
                worst.note(
                    &format!("{} beta in se", r.source),
                    (got - bj).abs() / sj,
                    1e-5 * loose,
                );
                worst.rel(
                    &format!("{} beta se", r.source),
                    se(f.beta_index(j).unwrap()),
                    sj,
                    1e-300,
                    1e-4 * loose,
                );
            }
        }
        if let (Some(b), Some(g)) = (&r.beta, &r.gamma) {
            // survreg: gamma_1 = 1 / scale, beta = -coef / scale
            worst.rel("survreg gamma1", f.gamma[1], g[1], 1.0, 1e-6);
            worst.rel("survreg gamma0", f.gamma[0], g[0], 1.0, 1e-6);
            for (j, &bj) in b.iter().enumerate() {
                worst.rel("survreg beta", f.beta[j].unwrap(), bj, 1.0, 1e-6);
            }
        }
        // the survivor function and other curves against the packages
        let times = &reference.times;
        let profiles = &reference.profiles;
        for p in 0..2 {
            let x = profile_matrix(profiles, p);
            if let Some(c) = &r.surv {
                let got = predict(f, x.as_ref(), times, Predict::Survival, &options).unwrap();
                for (i, pr) in got[0].iter().enumerate() {
                    worst.note(
                        &format!("{} survival", r.source),
                        (pr.estimate - c.estimate(p)[i]).abs(),
                        1e-7 * loose,
                    );
                    if let Some((lo, hi)) = c.band(p) {
                        worst.note(
                            &format!("{} survival limits", r.source),
                            (pr.lower.unwrap() - lo[i])
                                .abs()
                                .max((pr.upper.unwrap() - hi[i]).abs()),
                            1e-6,
                        );
                    }
                }
            }
            for (name, what, curves) in [
                ("hazard", Predict::Hazard, &r.hazard),
                ("cumhaz", Predict::CumulativeHazard, &r.cumhaz),
            ] {
                if let Some(c) = curves {
                    let got = predict(f, x.as_ref(), times, what, &options).unwrap();
                    for (i, pr) in got[0].iter().enumerate() {
                        worst.rel(
                            &format!("{} {name}", r.source),
                            pr.estimate,
                            c.estimate(p)[i],
                            1e-6,
                            1e-6 * loose,
                        );
                        if let Some((lo, hi)) = c.band(p) {
                            worst.rel(
                                &format!("{} {name} limits", r.source),
                                pr.lower.unwrap(),
                                lo[i],
                                1e-6,
                                1e-5,
                            );
                            worst.rel(
                                &format!("{} {name} limits", r.source),
                                pr.upper.unwrap(),
                                hi[i],
                                1e-6,
                                1e-5,
                            );
                        }
                    }
                }
            }
        }
        if let (Some(hr), Some(profile)) = (&r.hr, &r.hr_profile) {
            let x1 = Mat::from_fn(1, 2, |_, j| if j == 0 { profile[0] } else { 1.0 });
            let x0 = Mat::from_fn(1, 2, |_, j| profile[j]);
            let got = contrast(
                f,
                x1.as_ref(),
                x0.as_ref(),
                times,
                Contrast::HazardRatio,
                &options,
            )
            .unwrap();
            for (i, pr) in got[0].iter().enumerate() {
                worst.rel(
                    "rstpm2 hazard ratio",
                    pr.estimate,
                    hr.estimate[i],
                    1e-6,
                    1e-6,
                );
                worst.rel(
                    "rstpm2 hazard ratio limits",
                    pr.lower.unwrap(),
                    hr.lower[i],
                    1e-6,
                    1e-6,
                );
                worst.rel(
                    "rstpm2 hazard ratio limits",
                    pr.upper.unwrap(),
                    hr.upper[i],
                    1e-6,
                    1e-6,
                );
            }
        }
        // the transcription: the coefficients themselves, the covariances and the delta method
        if r.source == "transcription" {
            let theta = r.theta.as_ref().unwrap();
            let mine = full_theta(f);
            for (k, (&a, &b)) in mine.iter().zip(theta).enumerate() {
                worst.note("transcription theta in se", (a - b).abs() / se(k), 1e-6);
            }
            let rc = r.cov.as_ref().unwrap();
            let scale_cov: f64 = (0..q).map(|k| rc[k * q + k]).fold(0.0, f64::max);
            for k in 0..q * q {
                worst.note(
                    "transcription covariance",
                    (cov[k].unwrap() - rc[k]).abs() / scale_cov,
                    1e-7,
                );
            }
            let rob = f.robust_covariance.as_ref().unwrap();
            let rr = r.robust_cov.as_ref().unwrap();
            let scale_rob: f64 = (0..q).map(|k| rr[k * q + k]).fold(0.0, f64::max);
            for k in 0..q * q {
                worst.note(
                    "transcription robust covariance",
                    (rob[k].unwrap() - rr[k]).abs() / scale_rob,
                    1e-7,
                );
            }
            delta_checks(&mut worst, f, r, &reference.times, &options);
        }
        let _ = &out.x;
    }
    let mut failed = false;
    for (kind, (ratio, name)) in &worst.0 {
        eprintln!("{kind}: {ratio:.3} of the bound (at {name})");
        failed |= *ratio > 1.0;
    }
    assert!(!failed, "a deviation exceeds its bound");
}

fn full_theta(f: &FlexParFit) -> Vec<f64> {
    let mut t = f.gamma.clone();
    t.extend(f.beta.iter().map(|b| b.unwrap()));
    for c in &f.tvc_coefficients {
        t.extend(c);
    }
    t
}

fn delta_checks(
    worst: &mut Worst,
    f: &FlexParFit,
    r: &RFit,
    times: &[f64],
    options: &PredictOptions,
) {
    let profiles = r.profiles.as_ref().unwrap();
    let preds = r.predictions.as_ref().unwrap();
    for p in 0..2 {
        let x = profile_matrix(profiles, p);
        for (key, what) in [
            ("surv", Predict::Survival),
            ("cumhaz", Predict::CumulativeHazard),
            ("hazard", Predict::Hazard),
        ] {
            let reference = &preds[&format!("{key}_{}", p + 1)];
            let got = predict(f, x.as_ref(), times, what, options).unwrap();
            for (i, pr) in got[0].iter().enumerate() {
                worst.rel(
                    &format!("delta {key} estimate"),
                    pr.estimate,
                    reference.estimate[i],
                    1e-6,
                    1e-9,
                );
                worst.rel(
                    &format!("delta {key} se"),
                    pr.standard_error,
                    reference.se.as_ref().unwrap()[i],
                    1e-6,
                    1e-6,
                );
            }
        }
    }
    let x1 = profile_matrix(profiles, 0);
    let x0 = profile_matrix(profiles, 1);
    for (key, what) in [
        ("hazard_ratio", Contrast::HazardRatio),
        ("survival_difference", Contrast::SurvivalDifference),
        ("hazard_difference", Contrast::HazardDifference),
    ] {
        let reference = &preds[key];
        let got = contrast(f, x1.as_ref(), x0.as_ref(), times, what, options).unwrap();
        for (i, pr) in got[0].iter().enumerate() {
            worst.rel(
                &format!("delta {key} estimate"),
                pr.estimate,
                reference.estimate[i],
                1e-6,
                1e-9,
            );
            if let Some(se_log) = &reference.se_log {
                worst.rel(
                    "delta hazard_ratio se_log",
                    pr.standard_error / pr.estimate,
                    se_log[i],
                    1e-6,
                    1e-6,
                );
            } else {
                worst.rel(
                    &format!("delta {key} se"),
                    pr.standard_error,
                    reference.se.as_ref().unwrap()[i],
                    1e-6,
                    1e-6,
                );
            }
        }
    }
}
