//! `pseudo::pseudo_ipcw` and the exact jackknife checked against `eventglm`
//! 1.4.5's pseudo-observation functions: `model.censoring = "coxph"` with the
//! Binder and Hajek forms and the censoring covariates `x1 + x2` and `x1`,
//! `"independent"` (the exact jackknife of the whole sample, which is
//! `Method::Jackknife`) and `"stratified"` (the exact jackknife within the strata
//! of a covariate), for the incidence of two causes, the restricted time lost,
//! and on single-event data the survival and the restricted mean. See
//! `data/pseudo_ipcw.R`.
//!
//! # Tolerances
//!
//! With `n` rows, `p` censoring covariates, `T` distinct exit times, `kappa` the
//! condition number of the information matrix of the censoring model, `beta` its
//! coefficients, `x_max` the largest absolute covariate, `xb` the largest
//! `2 sum_a |x_a beta_a|` over the rows and `H = -ln g_min` the largest cumulative
//! hazard of the censoring that a row is weighted by, with `g_min` recorded by R:
//!
//! - Both programs run the same Newton iteration on the same partial likelihood,
//!   so the coefficients differ by the rounding of its solves,
//!   `16 kappa (n + p) eps max(1, |beta|)`.
//! - The cumulative hazard of a row is a baseline of at most `T` increments,
//!   relative error `8 eps (T + p + 4 + xb)` as in `r_cox_predict`, and the
//!   coefficients move it by `p x_max` times their error. The probability of not
//!   being censored is `exp(-theta H_0)`, so its relative error is `H` times
//!   the sum, and so is that of the weighted outcome `X = V I / G`.
//! - The Binder form of `eventglm` is `theta + (n - 1)(theta - mean of the
//!   others)`, with the mean of `n - 1` values formed by summing them, so it
//!   adds `4 (n + 8) eps` times the largest `|X|` to the value. The Hajek form
//!   forms `n theta - (n - 1) theta(-i)` from sums of `n` terms with the
//!   relative error `(n + 8) eps` each, and adds `4 n (n + 8) eps` times the
//!   larger of 1 and the largest `|X|`.
//! - The exact jackknife is `n theta - (n - 1) theta(-i)` of two estimates that
//!   are products or sums of at most `T` factors with three roundings each, so
//!   the bound is `2 n 12 (T + 4) eps` times the scale of the target, as in
//!   `r_pseudo`.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "labels and counts are whole numbers"
)]

use std::collections::HashMap;

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::pseudo::{
    IpcwMethod, IpcwOptions, Method, PseudoOptions, Target, pseudo, pseudo_ipcw,
};
use epistat_testkit::{load_json, na_f64_vec};
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    data: Data,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    survival: String,
    eventglm: String,
}

#[derive(Deserialize)]
struct Data {
    time: Vec<f64>,
    status: Vec<f64>,
    x1: Vec<f64>,
    x2: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    model: String,
    target: String,
    cens: String,
    method: String,
    times: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    po: Vec<f64>,
    cens_columns: Option<Vec<String>>,
    gmin: Option<f64>,
    beta: Option<Vec<f64>>,
    condition: Option<f64>,
}

#[derive(Default)]
struct Worst(HashMap<String, (f64, String)>);

impl Worst {
    #[track_caller]
    fn see(&mut self, output: &str, label: &str, got: f64, want: f64, tolerance: f64) {
        let ratio = (got - want).abs() / tolerance;
        assert!(
            ratio <= 1.0,
            "{output} at {label}: got {got:e}, R {want:e}, tolerance {tolerance:e}"
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

fn target_of(name: &str) -> Target {
    match name {
        "cif1" => Target::CumulativeIncidence { cause: 1 },
        "cif2" => Target::CumulativeIncidence { cause: 2 },
        "rmtl1" => Target::RestrictedTimeLost { cause: 1 },
        "surv" => Target::Survival,
        "rmean" => Target::RestrictedMean,
        _ => unreachable!(),
    }
}

#[test]
fn eventglm_pseudo_observations_agree() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/pseudo_ipcw.json"
    ));
    eprintln!(
        "{}, survival {}, eventglm {}",
        reference.versions.r, reference.versions.survival, reference.versions.eventglm
    );
    let d = &reference.data;
    let n = d.time.len();
    let competing: Vec<i32> = d.status.iter().map(|&s| s as i32).collect();
    let single: Vec<i32> = d.status.iter().map(|&s| i32::from(s > 0.0)).collect();
    let stratum: Vec<u32> = d.x1.iter().map(|&x| x as u32).collect();
    let mut distinct = d.time.clone();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    let t_count = distinct.len() as f64;
    let mut worst = Worst::default();
    let mut weighted_cases = 0;
    let mut undefined = 0;
    for case in &reference.cases {
        let target = target_of(&case.target);
        let single_event = matches!(target, Target::Survival | Target::RestrictedMean);
        let (status, codes) = if single_event {
            (&single, 1)
        } else {
            (&competing, 2)
        };
        let k = case.times.len();
        let label = format!(
            "{} {} cens {} {}",
            case.model, case.target, case.cens, case.method
        );
        let scale_of = |t: f64| match target {
            Target::RestrictedMean | Target::RestrictedTimeLost { .. } => t.max(1.0),
            _ => 1.0,
        };
        if case.model == "coxph" {
            weighted_cases += 1;
            let columns = case.cens_columns.as_ref().unwrap();
            let p = columns.len();
            let x = Mat::from_fn(
                n,
                p,
                |i, j| {
                    if columns[j] == "x1" { d.x1[i] } else { d.x2[i] }
                },
            );
            let data = SurvData::builder(&d.time, status)
                .n_codes(codes)
                .build()
                .unwrap();
            let mut opts = IpcwOptions::new(target);
            opts.method = if case.method == "hajek" {
                IpcwMethod::Hajek
            } else {
                IpcwMethod::Binder
            };
            let got = pseudo_ipcw(&data, x.as_ref(), &case.times, &opts)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(got.values.len(), n * k, "{label}: shape");
            let beta = case.beta.as_ref().unwrap();
            let big = beta.iter().fold(1.0f64, |m, b| m.max(b.abs()));
            let xmax = (0..n)
                .flat_map(|i| (0..p).map(move |j| (i, j)))
                .map(|(i, j)| x[(i, j)].abs())
                .fold(1.0f64, f64::max);
            let xb = (0..n)
                .map(|i| {
                    (0..p)
                        .map(|j| 2.0 * (x[(i, j)] * beta[j]).abs())
                        .sum::<f64>()
                })
                .fold(0.0f64, f64::max);
            let dbeta = 16.0 * case.condition.unwrap() * (n + p) as f64 * EPS * big;
            let rel_h = 8.0 * EPS * (t_count + p as f64 + 4.0 + xb) + p as f64 * xmax * dbeta;
            let h = -case.gmin.unwrap().ln();
            let rel_x = rel_h * h + EPS;
            let times_before_all = case.times[0] < distinct[0];
            for i in 0..n {
                for (slot, &t) in case.times.iter().enumerate() {
                    let (mine, theirs) = (got.values[i * k + slot], case.po[i * k + slot]);
                    let at = format!("{label} row {i} time {t}");
                    if times_before_all && slot == 0 {
                        // eventglm takes the probability of not being censored at
                        // the first exit time, which is below 1 where someone
                        // is censored there. Before any censoring it is 1.
                        if case.target == "surv" {
                            assert_eq!(mine.to_bits(), 1.0f64.to_bits(), "{at}");
                            assert!(theirs > 1.0 + 1e-3, "{at}: R gives {theirs}");
                        } else {
                            assert_eq!(mine.to_bits(), t.to_bits(), "{at}");
                            assert!(theirs > t + 1e-3, "{at}: R gives {theirs}");
                        }
                        continue;
                    }
                    let big_x = case.po.iter().map(|v| v.abs()).fold(1.0f64, f64::max);
                    let tol = if case.method == "hajek" {
                        mine.abs().max(1.0) * rel_x * n as f64
                            + 4.0 * n as f64 * (n as f64 + 8.0) * EPS * big_x
                    } else {
                        theirs.abs() * rel_x + 4.0 * (n as f64 + 8.0) * EPS * big_x
                    };
                    worst.see(&format!("ipcw {}", case.method), &at, mine, theirs, tol);
                }
            }
        } else {
            let mut strata_col: Option<&[u32]> = None;
            if case.model == "stratified" {
                strata_col = Some(&stratum);
            }
            let mut builder = SurvData::builder(&d.time, status).n_codes(codes);
            if let Some(s) = strata_col {
                builder = builder.stratum(s);
            }
            let data = builder.build().unwrap();
            let mut opts = PseudoOptions::new(target);
            opts.method = Method::Jackknife;
            let got = pseudo(&data, &case.times, &opts).unwrap_or_else(|e| panic!("{label}: {e}"));
            // The subjects are in the order of the strata and then the rows, and
            // eventglm's are in the order of the rows.
            let mut position = vec![0usize; n];
            for (r, &s) in got.subject.iter().enumerate() {
                position[s as usize] = r;
            }
            for (i, &row) in position.iter().enumerate() {
                for (slot, &t) in case.times.iter().enumerate() {
                    let mine = got.values[row * k + slot];
                    let theirs = case.po[i * k + slot];
                    if theirs.is_nan() {
                        // eventglm's jackknife is undefined beyond the last time of
                        // the data, where the curve is held at its last value here.
                        assert!(t > distinct[distinct.len() - 1], "R is undefined at {t}");
                        assert!(mine.is_finite());
                        undefined += 1;
                        continue;
                    }
                    let tol = 2.0 * n as f64 * 12.0 * (t_count + 4.0) * EPS * scale_of(t);
                    worst.see(
                        &case.model,
                        &format!("{label} row {i} time {t}"),
                        mine,
                        theirs,
                        tol,
                    );
                }
            }
        }
    }
    assert_eq!(weighted_cases, 22);
    assert_eq!(
        undefined,
        6 * n,
        "values R leaves undefined beyond the data"
    );
    let mut rows: Vec<_> = worst.0.into_iter().collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (output, (ratio, label)) in rows {
        eprintln!("worst {output}: {ratio:.4} of tolerance at {label}");
    }
}

#[test]
fn ipcw_refuses_what_eventglm_does_not_take() {
    let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let status = [1, 0, 1, 1, 0, 1, 0, 1];
    let z = Mat::from_fn(8, 1, |i, _| [0.3, -0.5, 0.9, 0.1, -0.2, 0.7, -0.8, 0.4][i]);
    let ok = IpcwOptions::new(Target::Survival);
    let data = SurvData::builder(&time, &status).build().unwrap();
    pseudo_ipcw(&data, z.as_ref(), &[3.0], &ok).unwrap();
    assert!(pseudo_ipcw(&data, z.as_ref(), &[], &ok).is_err());
    assert!(pseudo_ipcw(&data, z.as_ref(), &[3.0, 2.0], &ok).is_err());
    // A hazard is not the outcome of a subject.
    let hazard = IpcwOptions::new(Target::CumulativeHazard);
    assert!(pseudo_ipcw(&data, z.as_ref(), &[3.0], &hazard).is_err());
    // The covariates have the wrong number of rows.
    let short = Mat::from_fn(7, 1, |i, _| i as f64);
    assert!(pseudo_ipcw(&data, short.as_ref(), &[3.0], &ok).is_err());
    // Delayed entry, weights and strata.
    let start = [0.0; 8];
    let delayed = SurvData::builder(&time, &status)
        .start(&start)
        .build()
        .unwrap();
    assert!(pseudo_ipcw(&delayed, z.as_ref(), &[3.0], &ok).is_err());
    let weight = [1.0; 8];
    let weighted = SurvData::builder(&time, &status)
        .weight(&weight)
        .build()
        .unwrap();
    assert!(pseudo_ipcw(&weighted, z.as_ref(), &[3.0], &ok).is_err());
    let stratum = [0u32, 1, 0, 1, 0, 1, 0, 1];
    let stratified = SurvData::builder(&time, &status)
        .stratum(&stratum)
        .build()
        .unwrap();
    assert!(pseudo_ipcw(&stratified, z.as_ref(), &[3.0], &ok).is_err());
    // The survival needs a single event code, and a cause must exist.
    let causes = [1, 0, 2, 1, 0, 2, 0, 1];
    let competing = SurvData::builder(&time, &causes)
        .n_codes(2)
        .build()
        .unwrap();
    assert!(pseudo_ipcw(&competing, z.as_ref(), &[3.0], &ok).is_err());
    let missing = IpcwOptions::new(Target::CumulativeIncidence { cause: 3 });
    assert!(pseudo_ipcw(&competing, z.as_ref(), &[3.0], &missing).is_err());
    let present = IpcwOptions::new(Target::CumulativeIncidence { cause: 2 });
    pseudo_ipcw(&competing, z.as_ref(), &[3.0], &present).unwrap();
    // A censoring covariate that repeats another is aliased.
    let twice = Mat::from_fn(8, 2, |i, _| z[(i, 0)]);
    assert!(pseudo_ipcw(&data, twice.as_ref(), &[3.0], &ok).is_err());
}
