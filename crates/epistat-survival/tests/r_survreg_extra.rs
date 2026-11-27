//! The options of `survreg::fit` added after the first reference set, checked
//! against R 4.6.2's `survival::survreg` (survival 3.8-12): offsets, the t
//! distribution, robust and clustered variance, the dfbeta and dfbetas
//! residuals, and `predict(type = "terms")`. The data and the R code are in
//! `tests/data/survreg_extra.R`, which also says which results of R are wrong
//! and how the tests treat them.
//!
//! # Tolerances
//!
//! The iteration is the same as R's, so a result that follows from the final
//! point differs by what the solves of the Newton iteration amplify: each is
//! off by about the condition number `kappa` of the information on the scale
//! of the iteration times the machine epsilon, and each information is a sum
//! of `n` terms, so `bound = (8 + n) kappa eps`, relative to the size of the
//! value or to 1. The coefficients, log scales, variance and linear
//! predictors use it as in `r_survreg.rs`.
//!
//! The t distribution takes the same bound: its distribution function is an
//! incomplete beta function and its density has a log-gamma constant, and the
//! results agree with R's to far below it. The one exception is derived where
//! it applies, the working residual of an exact time.
//!
//! The robust variance is `V M V` with `V` the variance, off by `q * bound`
//! in relative terms, and `M` a sum of cluster totals of rows, each off by
//! the same `bound`; it takes `2 q bound`. A dfbeta residual is a row of
//! scores times `V`, and takes `(p + 1) bound`. A collapsed residual is a
//! sum of `m` rows, each off by that much, so its error is relative to the
//! sum of the absolute values of the rows it sums.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "strata, clusters and statuses are small whole numbers written by R as doubles"
)]
#![expect(
    clippy::float_cmp,
    reason = "event and status codes are written by R as exact 0, 1, 2 and 3"
)]
#![expect(clippy::type_complexity, reason = "a local table of names and numbers")]
#![expect(
    clippy::struct_field_names,
    reason = "the fields are the names of R's dfbeta types"
)]
#![expect(
    clippy::struct_excessive_bools,
    reason = "the fields are the flags the R script writes for each fit"
)]

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::survreg::{
    DfbetaOptions, Distribution, PredictOptions, PredictType, ResidualKind, SurvregOptions,
    TermsReference, fit,
};
use epistat_testkit::{NanRule, load_json, na_f64_vec, none_as_nan, worst_deviation};
use serde::Deserialize;
use std::collections::HashMap;

/// A matrix whose NaN entries R wrote as null.
#[derive(Deserialize)]
struct NaMatrix(Vec<Vec<Option<f64>>>);

impl std::ops::Deref for NaMatrix {
    type Target = Vec<Vec<Option<f64>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The matrix with NaN for each null.
fn dense(m: &NaMatrix) -> Vec<Vec<f64>> {
    m.iter()
        .map(|row| row.iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        .collect()
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Dataset {
    #[serde(deserialize_with = "na_f64_vec")]
    time: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    event: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    time2: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    status: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    stratum: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    weight: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    off: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    cluster: Vec<f64>,
}

#[derive(Deserialize)]
struct Residuals {
    response: Vec<Option<f64>>,
    deviance: Vec<Option<f64>>,
    working: Vec<Option<f64>>,
    ldcase: Vec<Option<f64>>,
    ldresp: Vec<Option<f64>>,
    ldshape: Vec<Option<f64>>,
}

#[derive(Deserialize)]
struct Dfbetas {
    dfbeta: NaMatrix,
    dfbetas: NaMatrix,
    weighted: NaMatrix,
    collapsed: NaMatrix,
    collapsed_s: NaMatrix,
}

#[derive(Deserialize)]
struct Terms {
    groups: Vec<Vec<usize>>,
    means: Vec<f64>,
    fit: NaMatrix,
    se_r: NaMatrix,
    se_delta: NaMatrix,
}

#[derive(Deserialize)]
struct RFit {
    name: String,
    dataset: String,
    distribution: String,
    parms: f64,
    use_strata: bool,
    use_weights: bool,
    fixed_scale: f64,
    use_offset: bool,
    robust: String,
    x: Vec<Vec<f64>>,
    coefficients: Vec<Option<f64>>,
    log_scale: Vec<f64>,
    variance: NaMatrix,
    loglik: f64,
    loglik_null: f64,
    iterations: usize,
    df: usize,
    linear_predictors: Vec<f64>,
    information_condition: f64,
    predict_rows: usize,
    #[serde(deserialize_with = "na_f64_vec")]
    lp: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    lp_se: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    response: Vec<f64>,
    #[serde(deserialize_with = "na_f64_vec")]
    response_se: Vec<f64>,
    quantile: Vec<Vec<Option<f64>>>,
    quantile_se: Vec<Vec<Option<f64>>>,
    lp_se_corrected: Option<Vec<f64>>,
    lp_new: Option<Vec<Option<f64>>>,
    quantile_new: Option<NaMatrix>,
    residuals: Residuals,
    interval_scale_defect: bool,
    dfbeta: Dfbetas,
    dfbeta_corrected: Dfbetas,
    robust_variance: Option<NaMatrix>,
    robust_variance_corrected: Option<NaMatrix>,
    n_clusters: Option<usize>,
    terms: Option<Terms>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: HashMap<String, Dataset>,
    fits: Vec<RFit>,
}

fn distribution(name: &str) -> Distribution {
    match name {
        "weibull" => Distribution::Weibull,
        "gaussian" => Distribution::Gaussian,
        "logistic" => Distribution::Logistic,
        "lognormal" => Distribution::LogNormal,
        "t" => Distribution::StudentT,
        other => panic!("{other}"),
    }
}

struct Columns {
    time: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    status: Vec<i32>,
    stratum: Vec<u32>,
    cluster: Vec<u32>,
    weight: Vec<f64>,
    offset: Vec<f64>,
}

fn columns(dataset: &str, d: &Dataset) -> Columns {
    let mut c = Columns {
        time: d.time.clone(),
        lower: Vec::new(),
        upper: Vec::new(),
        status: Vec::new(),
        stratum: d
            .stratum
            .iter()
            .map(|&s| if s.is_nan() { 0 } else { s as u32 })
            .collect(),
        cluster: d.cluster.iter().map(|&s| s as u32).collect(),
        weight: d.weight.clone(),
        offset: d.off.clone(),
    };
    for i in 0..d.time.len() {
        let (status, lower, upper) = match dataset {
            "lung" | "right" => {
                let e = i32::from(d.event[i] == 1.0);
                (e, d.time[i], if e == 1 { d.time[i] } else { f64::INFINITY })
            }
            "left" => {
                if d.event[i] == 1.0 {
                    (1, d.time[i], d.time[i])
                } else {
                    (2, f64::NEG_INFINITY, d.time[i])
                }
            }
            _ => match d.status[i] as i32 {
                0 => (0, d.time[i], f64::INFINITY),
                1 => (1, d.time[i], d.time[i]),
                2 => (2, f64::NEG_INFINITY, d.time[i]),
                _ => (3, d.time[i], d.time2[i]),
            },
        };
        c.status.push(status);
        c.lower.push(lower);
        c.upper.push(upper);
    }
    c
}

fn build<'a>(dataset: &str, c: &'a Columns, r: &RFit) -> SurvData<'a> {
    if dataset == "lung" || dataset == "right" {
        let mut b = SurvData::builder(&c.time, &c.status);
        if r.use_strata {
            b = b.stratum(&c.stratum);
        }
        if r.use_weights {
            b = b.weight(&c.weight);
        }
        if r.robust == "cluster" {
            b = b.cluster(&c.cluster);
        }
        if r.use_offset {
            b = b.offset(&c.offset);
        }
        b.build().unwrap()
    } else {
        let mut b = SurvData::interval_builder(&c.lower, &c.upper);
        if r.use_strata {
            b = b.stratum(&c.stratum);
        }
        if r.use_weights {
            b = b.weight(&c.weight);
        }
        if r.robust == "cluster" {
            b = b.cluster(&c.cluster);
        }
        if r.use_offset {
            b = b.offset(&c.offset);
        }
        b.build().unwrap()
    }
}

fn deviation_with(got: &[f64], expected: &[f64], bound: f64, scale: impl Fn(usize) -> f64) -> f64 {
    assert_eq!(got.len(), expected.len());
    got.iter()
        .zip(expected)
        .enumerate()
        .map(|(i, (g, e))| (g - e).abs() / (bound * scale(i)))
        .fold(0.0, f64::max)
}

/// The columns of a row-major result that are not aliased, as a flat vector
/// of the entries of `want` (rows of R) and `got` (row-major, `width` wide).
fn pairs(
    got: &[Option<f64>],
    want: &[Vec<f64>],
    width: usize,
    aliased: &[bool],
) -> (Vec<f64>, Vec<f64>) {
    let (mut g, mut w) = (Vec::new(), Vec::new());
    for (i, row) in want.iter().enumerate() {
        for b in 0..width {
            if b < aliased.len() && aliased[b] {
                assert!(got[i * width + b].is_none());
                continue;
            }
            g.push(got[i * width + b].unwrap());
            w.push(row[b]);
        }
    }
    (g, w)
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one pass per fit, each output in turn"
)]
fn extra_options_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/survreg_extra.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let mut report: Vec<(String, f64, Vec<(String, f64)>)> = Vec::new();
    for r in &reference.fits {
        let d = &reference.datasets[&r.dataset];
        let c = columns(&r.dataset, d);
        let x = Mat::from_fn(r.x.len(), r.x[0].len(), |i, j| r.x[i][j]);
        let is_t = r.distribution == "t";
        let options = SurvregOptions {
            distribution: distribution(&r.distribution),
            df: is_t.then_some(r.parms),
            scale: (r.fixed_scale > 0.0).then_some(r.fixed_scale),
            robust: r.robust == "robust",
            ..SurvregOptions::default()
        };
        let data = build(&r.dataset, &c, r);
        let f = fit(&data, x.as_ref(), &options).unwrap_or_else(|e| panic!("{}: {e}", r.name));
        if r.dataset == "interval" {
            // The same data as bounds and as a status give the same fit.
            let mut b = SurvData::interval_status_builder(&c.time, &d.time2, &c.status);
            if r.use_strata {
                b = b.stratum(&c.stratum);
            }
            if r.use_weights {
                b = b.weight(&c.weight);
            }
            if r.robust == "cluster" {
                b = b.cluster(&c.cluster);
            }
            if r.use_offset {
                b = b.offset(&c.offset);
            }
            let by_status = fit(&b.build().unwrap(), x.as_ref(), &options).unwrap();
            assert_eq!(f, by_status, "{}: the two interval routes differ", r.name);
        }
        let n = f.n_used as f64;
        let p = r.coefficients.len();
        let q = p + if f.scale_estimated { f.n_strata } else { 0 };
        let aliased: Vec<bool> = r.coefficients.iter().map(Option::is_none).collect();
        let rank_deficient = aliased.iter().any(|&a| a);
        let count = 8.0 + n;
        let bound = count * r.information_condition * f64::EPSILON;
        assert!(f.converged, "{}", r.name);
        assert_eq!(f.iterations, r.iterations, "{}", r.name);
        assert_eq!(f.df() + (p - f.rank), r.df, "{}", r.name);
        for (a, b) in r.coefficients.iter().zip(&f.coefficients) {
            assert_eq!(a.is_none(), b.is_none(), "{} aliased", r.name);
        }
        let mut worst: Vec<(String, f64)> = Vec::new();
        let mut note = |what: &str, units: f64| {
            worst.push((what.to_owned(), units));
        };
        note(
            "coefficient",
            worst_deviation(
                &none_as_nan(&f.coefficients),
                &none_as_nan(&r.coefficients),
                bound,
                1.0,
                NanRule::Coincide,
            ),
        );
        note(
            "log scale",
            worst_deviation(&f.log_scale, &r.log_scale, bound, 1.0, NanRule::Coincide),
        );
        let variance_units = |got: &[Option<f64>], want: &[Vec<f64>], factor: f64| -> f64 {
            assert_eq!(want.len(), q, "{}", r.name);
            let mut worst = 0.0f64;
            for a in 0..q {
                for b in 0..q {
                    let dead = (a < p && aliased[a]) || (b < p && aliased[b]);
                    let value = got[a * q + b];
                    if dead {
                        assert!(value.is_none(), "{} variance alias", r.name);
                    } else {
                        let scale = (want[a][a] * want[b][b]).sqrt();
                        worst = worst
                            .max((value.unwrap() - want[a][b]).abs() / (factor * bound * scale));
                    }
                }
            }
            worst
        };
        note(
            "variance",
            variance_units(&f.variance, &dense(&r.variance), q as f64),
        );
        let ll_bound = 4.0 * n * f64::EPSILON;
        note(
            "loglik",
            worst_deviation(&[f.loglik], &[r.loglik], ll_bound, 1.0, NanRule::Coincide),
        );
        note(
            "null loglik",
            worst_deviation(
                &[f.loglik_null],
                &[r.loglik_null],
                ll_bound,
                1.0,
                NanRule::Coincide,
            ),
        );
        note(
            "linear predictor",
            worst_deviation(
                &f.linear_predictors,
                &r.linear_predictors,
                bound * p as f64,
                1.0,
                NanRule::Coincide,
            ),
        );

        // Robust variance.
        assert_eq!(
            f.robust_variance.is_some(),
            r.robust != "none",
            "{}",
            r.name
        );
        if let (Some(got), Some(native), Some(corrected)) = (
            &f.robust_variance,
            r.robust_variance.as_ref().map(dense),
            r.robust_variance_corrected.as_ref().map(dense),
        ) {
            let (native, corrected) = (&native, &corrected);
            let factor = 2.0 * q as f64;
            note("robust variance", variance_units(got, corrected, factor));
            if r.interval_scale_defect {
                // R's own robust variance differs from the one that follows
                // from its definition.
                let rel = (0..q)
                    .flat_map(|a| (0..q).map(move |b| (a, b)))
                    .filter(|&(a, b)| !((a < p && aliased[a]) || (b < p && aliased[b])))
                    .map(|(a, b)| {
                        (native[a][b] - corrected[a][b]).abs()
                            / (corrected[a][a] * corrected[b][b]).sqrt()
                    })
                    .fold(0.0, f64::max);
                assert!(rel > 1e-3, "{}: R's robust variance agrees, {rel}", r.name);
                note("R defect, robust variance (relative)", rel);
            } else {
                note(
                    "robust variance vs native",
                    variance_units(got, native, factor),
                );
            }
            assert_eq!(f.n_clusters, r.n_clusters, "{}", r.name);
        }

        // Predictions of the fitted rows, offset included.
        if !rank_deficient {
            let m = r.predict_rows;
            let xm = Mat::from_fn(m, p, |i, j| r.x[i][j]);
            let strata: Vec<u32> = (0..m).map(|i| c.stratum[i]).collect();
            let offset: Vec<f64> = c.offset[..m].to_vec();
            let run = |kind: PredictType, with_offset: bool| {
                f.predict(
                    xm.as_ref(),
                    &PredictOptions {
                        kind,
                        se_fit: true,
                        strata: r.use_strata.then(|| strata.clone()),
                        offset: (r.use_offset && with_offset).then(|| offset.clone()),
                        robust: r.robust != "none",
                    },
                )
                .unwrap()
            };
            let pb = bound * (p as f64 + 1.0);
            let lp = run(PredictType::LinearPredictor, true);
            let resp = run(PredictType::Response, true);
            let quant = run(
                PredictType::Quantile {
                    p: vec![0.1, 0.5, 0.9],
                },
                true,
            );
            let eq: Vec<f64> = r.quantile.iter().flatten().map(|v| v.unwrap()).collect();
            let eqs: Vec<f64> = r.quantile_se.iter().flatten().map(|v| v.unwrap()).collect();
            let qb = pb;
            // Where R's robust variance is wrong, so is every standard error
            // from it; the linear predictor's is compared with the one from
            // the corrected variance and the others are left out.
            let affected = r.lp_se_corrected.is_some();
            let lp_se_want = r.lp_se_corrected.as_ref().unwrap_or(&r.lp_se);
            let mut checks = vec![
                ("lp", &lp.fit, &r.lp, pb),
                ("lp se", lp.se.as_ref().unwrap(), lp_se_want, pb),
                ("response", &resp.fit, &r.response, pb),
                ("quantile", &quant.fit, &eq, qb),
            ];
            if !affected {
                checks.push(("response se", resp.se.as_ref().unwrap(), &r.response_se, pb));
                checks.push(("quantile se", quant.se.as_ref().unwrap(), &eqs, qb));
            }
            for (what, got, want, b) in checks {
                note(
                    what,
                    worst_deviation(got, want, b, 1.0, NanRule::Coincide) * b,
                );
                assert!(
                    worst_deviation(got, want, b, 1.0, NanRule::Coincide) <= 1.0,
                    "{} {what}",
                    r.name
                );
            }
            if let (Some(lp_new), Some(quantile_new)) = (&r.lp_new, &r.quantile_new) {
                let lp_new: Vec<f64> = lp_new.iter().map(|v| v.unwrap()).collect();
                let quantile_new = dense(quantile_new);
                // R's predictions from new data take the offset as 0.
                let lp0 = run(PredictType::LinearPredictor, false);
                let q0 = run(
                    PredictType::Quantile {
                        p: vec![0.1, 0.5, 0.9],
                    },
                    false,
                );
                let flat_q: Vec<f64> = quantile_new.iter().flatten().copied().collect();
                let u1 = worst_deviation(&lp0.fit, &lp_new, pb, 1.0, NanRule::Coincide);
                let u2 = worst_deviation(&q0.fit, &flat_q, qb, 1.0, NanRule::Coincide);
                assert!(u1 <= 1.0 && u2 <= 1.0, "{}: no-offset predictions", r.name);
                let gap = lp_new
                    .iter()
                    .zip(&r.lp)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(gap > 1e-3, "{}: R's new-data lp has the offset", r.name);
                note("R defect, lp from new data (absolute gap)", gap);
            }
        }

        // Residuals, as in r_survreg.rs.
        let kinds = [
            (
                ResidualKind::Response,
                &r.residuals.response,
                "response residual",
            ),
            (
                ResidualKind::Deviance,
                &r.residuals.deviance,
                "deviance residual",
            ),
            (
                ResidualKind::Working,
                &r.residuals.working,
                "working residual",
            ),
            (ResidualKind::LdCase, &r.residuals.ldcase, "ldcase"),
            (ResidualKind::LdResp, &r.residuals.ldresp, "ldresp"),
            (ResidualKind::LdShape, &r.residuals.ldshape, "ldshape"),
        ];
        for (kind, want, what) in kinds {
            let got = f.residuals(&data, x.as_ref(), kind, false).unwrap();
            let rb = bound * (p as f64 + 1.0);
            let keep: Vec<usize> = (0..got.len())
                .filter(|&i| {
                    let interval = c.status[i] == 3;
                    let displaced = match kind {
                        ResidualKind::LdShape => interval,
                        ResidualKind::LdCase | ResidualKind::LdResp => {
                            interval && f.scale_estimated
                        }
                        // R's deviance function of the t distribution is
                        // wrong for an interval: NaN, and a wrong centre.
                        ResidualKind::Deviance | ResidualKind::Response => interval && is_t,
                        ResidualKind::Working => false,
                    };
                    !displaced
                })
                .collect();
            if is_t && matches!(kind, ResidualKind::Deviance) {
                for i in (0..got.len()).filter(|&i| c.status[i] == 3) {
                    assert!(want[i].is_none_or(f64::is_nan), "{}", r.name);
                    assert!(got[i].is_some(), "{}", r.name);
                }
            }
            if is_t && matches!(kind, ResidualKind::Response) {
                let gap = (0..got.len())
                    .filter(|&i| c.status[i] == 3)
                    .map(|i| (got[i].unwrap() - want[i].unwrap()).abs())
                    .fold(0.0, f64::max);
                if c.status.contains(&3) {
                    assert!(gap > 1e-3, "{}: R's interval centre agrees, {gap}", r.name);
                }
            }
            let squared = matches!(kind, ResidualKind::Deviance);
            let pick = |v: &[Option<f64>]| -> Vec<f64> {
                keep.iter()
                    .map(|&i| {
                        let value = none_as_nan(v)[i];
                        if squared { value * value.abs() } else { value }
                    })
                    .collect()
            };
            // The working residual of an exact time under the t distribution
            // divides by a curvature `-(nu + 1)(nu - z^2) / (nu + z^2)^2`
            // that changes sign at `z^2 = nu`. A relative error `delta` in z
            // changes `nu - z^2` by `2 z^2 delta` relative to `|nu - z^2|`,
            // so the residual's relative error is that of the other
            // quantities times `(nu + z^2) / |nu - z^2|`, and its tolerance
            // grows by that factor. R's own evaluation has the same
            // cancellation.
            let curvature_condition = |i: usize| -> f64 {
                if is_t && matches!(kind, ResidualKind::Working) && c.status[i] == 1 {
                    let stratum = if r.use_strata {
                        c.stratum[i] as usize
                    } else {
                        0
                    };
                    let z = (c.time[i] - f.linear_predictors[i]) / f.log_scale[stratum].exp();
                    (r.parms + z * z) / (r.parms - z * z).abs()
                } else {
                    1.0
                }
            };
            let (g, w) = (pick(&got), pick(want));
            let u = deviation_with(&g, &w, rb, |k| {
                w[k].abs().max(1.0) * curvature_condition(keep[k])
            });
            assert!(u <= 1.0, "{} {what}: {u} of the bound", r.name);
            note(what, u * rb);
        }

        // dfbeta, dfbetas, weighted and collapsed.
        let rb = bound * (p as f64 + 1.0);
        let by_cluster = DfbetaOptions {
            collapse: Some(&c.cluster),
            ..DfbetaOptions::default()
        };
        let plain = f
            .dfbeta(&data, x.as_ref(), &DfbetaOptions::default())
            .unwrap();
        let standard = f
            .dfbeta(
                &data,
                x.as_ref(),
                &DfbetaOptions {
                    standardise: true,
                    ..DfbetaOptions::default()
                },
            )
            .unwrap();
        let weighted = f
            .dfbeta(
                &data,
                x.as_ref(),
                &DfbetaOptions {
                    weighted: true,
                    ..DfbetaOptions::default()
                },
            )
            .unwrap();
        let collapsed = f
            .dfbeta(
                &data,
                x.as_ref(),
                &DfbetaOptions {
                    weighted: true,
                    ..by_cluster
                },
            )
            .unwrap();
        let collapsed_s = f
            .dfbeta(
                &data,
                x.as_ref(),
                &DfbetaOptions {
                    standardise: true,
                    ..by_cluster
                },
            )
            .unwrap();
        assert_eq!(plain.width, q);
        assert_eq!(plain.n_rows, c.time.len());
        let (native_tables, corrected_tables) = (&r.dfbeta, &r.dfbeta_corrected);
        let n_groups = corrected_tables.collapsed.len();
        assert_eq!(collapsed.n_rows, n_groups);
        let mut levels = c.cluster.clone();
        levels.sort_unstable();
        levels.dedup();
        assert_eq!(collapsed.groups.as_ref().unwrap(), &levels);
        let tables: [(
            &str,
            &epistat_survival::survreg::Dfbeta,
            &NaMatrix,
            &NaMatrix,
        ); 5] = [
            (
                "dfbeta",
                &plain,
                &corrected_tables.dfbeta,
                &native_tables.dfbeta,
            ),
            (
                "dfbetas",
                &standard,
                &corrected_tables.dfbetas,
                &native_tables.dfbetas,
            ),
            (
                "dfbeta weighted",
                &weighted,
                &corrected_tables.weighted,
                &native_tables.weighted,
            ),
            (
                "dfbeta collapsed",
                &collapsed,
                &corrected_tables.collapsed,
                &native_tables.collapsed,
            ),
            (
                "dfbetas collapsed",
                &collapsed_s,
                &corrected_tables.collapsed_s,
                &native_tables.collapsed_s,
            ),
        ];
        for (what, got, corrected, native) in tables {
            let (corrected, native) = (dense(corrected), dense(native));
            let (g, w) = pairs(&got.values, &corrected, q, &aliased);
            // The error of a sum of rows is relative to the sum of their
            // absolute values.
            let collapsing = what.contains("collapsed");
            let scale = |k: usize| -> f64 {
                if !collapsing {
                    return w[k].abs().max(1.0);
                }
                let live: Vec<usize> = (0..q).filter(|&b| !(b < p && aliased[b])).collect();
                let (row, col) = (k / live.len(), live[k % live.len()]);
                let level = levels[row];
                let uncollapsed = dense(if what == "dfbeta collapsed" {
                    &corrected_tables.weighted
                } else {
                    &corrected_tables.dfbetas
                });
                (0..c.cluster.len())
                    .filter(|&i| c.cluster[i] == level)
                    .map(|i| uncollapsed[i][col].abs())
                    .sum::<f64>()
                    .max(1.0)
            };
            let u = deviation_with(&g, &w, rb, scale);
            assert!(u <= 1.0, "{} {what}: {u} of the bound", r.name);
            note(what, u * rb);
            if r.interval_scale_defect {
                let (_, nat) = pairs(&got.values, &native, q, &aliased);
                let gap = nat
                    .iter()
                    .zip(&w)
                    .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                    .fold(0.0, f64::max);
                assert!(gap > 1e-3, "{} {what}: R's value agrees, {gap}", r.name);
            } else {
                let (_, nat) = pairs(&got.values, &native, q, &aliased);
                let u = deviation_with(&g, &nat, rb, scale);
                assert!(
                    u <= 1.0,
                    "{} {what} against R's own: {u} of the bound",
                    r.name
                );
            }
        }

        // Terms.
        if let Some(t) = &r.terms {
            let m = r.predict_rows;
            let xm = Mat::from_fn(m, p, |i, j| r.x[i][j]);
            let pred = f
                .predict(
                    xm.as_ref(),
                    &PredictOptions {
                        kind: PredictType::Terms {
                            groups: t.groups.clone(),
                            reference: TermsReference::Sample,
                        },
                        se_fit: true,
                        robust: r.robust != "none",
                        ..PredictOptions::default()
                    },
                )
                .unwrap();
            assert_eq!(pred.width, t.groups.len());
            let pb = bound * (p as f64 + 1.0);
            let want: Vec<f64> = dense(&t.fit).into_iter().flatten().collect();
            let se_delta: Vec<f64> = dense(&t.se_delta).into_iter().flatten().collect();
            let se_r: Vec<f64> = dense(&t.se_r).into_iter().flatten().collect();
            let means = worst_deviation(
                &f.column_means,
                &t.means,
                f64::EPSILON * n,
                1.0,
                NanRule::Coincide,
            );
            note("column means", means * f64::EPSILON * n);
            let u = worst_deviation(&pred.fit, &want, pb, 1.0, NanRule::Coincide);
            let us = worst_deviation(
                pred.se.as_ref().unwrap(),
                &se_delta,
                pb,
                1.0,
                NanRule::Coincide,
            );
            assert!(u <= 1.0 && us <= 1.0 && means <= 1.0, "{} terms", r.name);
            note("terms", u * pb);
            note("terms se (delta method)", us * pb);
            // R's standard error is a different quantity.
            let gap = pred
                .se
                .as_ref()
                .unwrap()
                .iter()
                .zip(&se_r)
                .map(|(a, b)| (a - b).abs() / a.abs().max(1e-300))
                .fold(0.0, f64::max);
            assert!(gap > 1e-3, "{}: R's term se agrees, {gap}", r.name);
            note("R defect, terms se (relative)", gap);
        }
        report.push((r.name.clone(), r.information_condition, worst));
    }
    for (name, kappa, worst) in &report {
        let cells: Vec<String> = worst.iter().map(|(w, v)| format!("{w} {v:.1e}")).collect();
        eprintln!("{name:32} kappa {kappa:9.2e}  {}", cells.join("  "));
    }
}
