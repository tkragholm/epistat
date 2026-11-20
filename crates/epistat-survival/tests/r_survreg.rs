//! `survreg::fit`, `predict` and `residuals` checked against R 4.6.2's
//! `survival::survreg` on the lung data and on simulated right-, left- and
//! interval-censored data.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "strata and statuses are small whole numbers written by R as doubles"
)]
#![expect(
    clippy::float_cmp,
    reason = "event and status codes are written by R as exact 0, 1, 2 and 3"
)]
#![expect(clippy::type_complexity, reason = "a local table of names and numbers")]

use epistat_survival::SurvData;
use epistat_survival::survreg::{
    Distribution, PredictOptions, PredictType, ResidualKind, SurvregFit, SurvregOptions, fit,
};
use epistat_testkit::{NanRule, load_json, na_f64_vec, none_as_nan, worst_deviation};
use faer::Mat;
use serde::Deserialize;
use std::collections::HashMap;

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
struct RFit {
    name: String,
    dataset: String,
    distribution: String,
    use_strata: bool,
    use_weights: bool,
    fixed_scale: f64,
    x: Vec<Vec<f64>>,
    coefficients: Vec<Option<f64>>,
    log_scale: Vec<f64>,
    variance: Vec<Vec<f64>>,
    loglik: f64,
    loglik_null: f64,
    iterations: u32,
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
    residuals: Residuals,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: HashMap<String, Dataset>,
    fits: Vec<RFit>,
}

/// The sort of data each dataset is, and how the interval ones are written.
#[derive(Clone, Copy)]
enum Route {
    Right,
    Left,
    IntervalStatus,
    IntervalBounds,
}

fn distribution(name: &str) -> Distribution {
    match name {
        "weibull" => Distribution::Weibull,
        "exponential" => Distribution::Exponential,
        "rayleigh" => Distribution::Rayleigh,
        "extreme" => Distribution::ExtremeValue,
        "gaussian" => Distribution::Gaussian,
        "logistic" => Distribution::Logistic,
        "lognormal" => Distribution::LogNormal,
        "loglogistic" => Distribution::LogLogistic,
        other => panic!("{other}"),
    }
}

struct Columns {
    time: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    status: Vec<i32>,
    time2: Vec<f64>,
    stratum: Vec<u32>,
    weight: Vec<f64>,
}

fn columns(dataset: &str, d: &Dataset) -> Columns {
    let n = d.time.len();
    let mut c = Columns {
        time: d.time.clone(),
        lower: Vec::new(),
        upper: Vec::new(),
        status: Vec::new(),
        time2: d.time2.clone(),
        stratum: d
            .stratum
            .iter()
            .map(|&s| if s.is_nan() { 0 } else { s as u32 })
            .collect(),
        weight: d.weight.clone(),
    };
    for i in 0..n {
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
            _ => {
                let s = d.status[i] as i32;
                match s {
                    0 => (0, d.time[i], f64::INFINITY),
                    1 => (1, d.time[i], d.time[i]),
                    2 => (2, f64::NEG_INFINITY, d.time[i]),
                    _ => (3, d.time[i], d.time2[i]),
                }
            }
        };
        c.status.push(status);
        c.lower.push(lower);
        c.upper.push(upper);
    }
    c
}

fn build<'a>(c: &'a Columns, route: Route, r: &RFit) -> SurvData<'a> {
    match route {
        Route::Right => {
            let mut b = SurvData::builder(&c.time, &c.status);
            if r.use_strata {
                b = b.stratum(&c.stratum);
            }
            if r.use_weights {
                b = b.weight(&c.weight);
            }
            b.build().unwrap()
        }
        Route::Left | Route::IntervalBounds => {
            let mut b = SurvData::interval_builder(&c.lower, &c.upper);
            if r.use_strata {
                b = b.stratum(&c.stratum);
            }
            if r.use_weights {
                b = b.weight(&c.weight);
            }
            b.build().unwrap()
        }
        Route::IntervalStatus => {
            let mut b = SurvData::interval_status_builder(&c.time, &c.time2, &c.status);
            if r.use_strata {
                b = b.stratum(&c.stratum);
            }
            if r.use_weights {
                b = b.weight(&c.weight);
            }
            b.build().unwrap()
        }
    }
}

fn design(r: &RFit) -> Mat<f64> {
    Mat::from_fn(r.x.len(), r.x[0].len(), |i, j| r.x[i][j])
}

#[test]
fn fits_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/survreg.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    let mut report: Vec<(String, f64, Vec<(&str, f64)>)> = Vec::new();
    for r in &reference.fits {
        let d = &reference.datasets[&r.dataset];
        let c = columns(&r.dataset, d);
        let x = design(r);
        let routes: Vec<Route> = match r.dataset.as_str() {
            "interval" | "interval0" => vec![Route::IntervalStatus, Route::IntervalBounds],
            "left" => vec![Route::Left],
            _ => vec![Route::Right],
        };
        let options = SurvregOptions {
            distribution: distribution(&r.distribution),
            scale: (r.fixed_scale > 0.0).then_some(r.fixed_scale),
            ..SurvregOptions::default()
        };
        let mut fits: Vec<SurvregFit> = Vec::new();
        for route in &routes {
            let data = build(&c, *route, r);
            fits.push(
                fit(&data, x.as_ref(), &options).unwrap_or_else(|e| panic!("{}: {e}", r.name)),
            );
        }
        if fits.len() == 2 {
            assert_eq!(
                fits[0], fits[1],
                "{}: the two interval routes differ",
                r.name
            );
        }
        let f = &fits[0];
        let data = build(&c, routes[0], r);
        let n = f.n_used as f64;
        let p = r.coefficients.len();
        let q = f.coefficients.len() + if f.scale_estimated { f.n_strata } else { 0 };
        let rank_deficient = r.coefficients.iter().any(Option::is_none);
        // The Newton iteration is a sequence of solves with the information
        // matrix, each off by about its condition number times the machine
        // epsilon, and the information is a sum of n terms, each off by an
        // epsilon, so a value that follows from the final point differs from
        // R's by (8 + n) kappa eps relative to its size or to 1.
        let bound = (8.0 + n) * r.information_condition * f64::EPSILON;
        assert!(f.converged, "{}", r.name);
        assert_eq!(f.iterations, r.iterations, "{}", r.name);
        assert_eq!(f.n_rows, r.x.len());
        assert_eq!(f.df() + (p - f.rank), r.df, "{}", r.name);
        for (a, b) in r.coefficients.iter().zip(&f.coefficients) {
            assert_eq!(a.is_none(), b.is_none(), "{} aliased", r.name);
        }
        let coefficient = worst_deviation(
            &none_as_nan(&f.coefficients),
            &none_as_nan(&r.coefficients),
            bound,
            1.0,
            NanRule::Unchecked,
        );
        let log_scale = worst_deviation(&f.log_scale, &r.log_scale, bound, 1.0, NanRule::Unchecked);
        let variance_r: Vec<Vec<f64>> = r.variance.clone();
        assert_eq!(variance_r.len(), q, "{}", r.name);
        let mut variance = 0.0f64;
        for a in 0..q {
            for b in 0..q {
                let got = f.variance[a * q + b];
                let e = variance_r[a][b];
                let aliased = (a < p && r.coefficients[a].is_none())
                    || (b < p && r.coefficients[b].is_none());
                if aliased {
                    assert!(got.is_none() && e == 0.0, "{} variance alias", r.name);
                } else {
                    let scale = (variance_r[a][a] * variance_r[b][b]).sqrt();
                    variance = variance.max((got.unwrap() - e).abs() / (q as f64 * bound * scale));
                }
            }
        }
        // A sum of n terms, each with a few ulps of error.
        let ll_bound = 4.0 * n * f64::EPSILON;
        let loglik = worst_deviation(&[f.loglik], &[r.loglik], ll_bound, 1.0, NanRule::Unchecked);
        let null = worst_deviation(
            &[f.loglik_null],
            &[r.loglik_null],
            ll_bound,
            1.0,
            NanRule::Unchecked,
        );
        let lp = worst_deviation(
            &f.linear_predictors,
            &r.linear_predictors,
            bound * p as f64,
            1.0,
            NanRule::Unchecked,
        );
        let mut worst = vec![
            ("coefficient", coefficient * bound),
            ("log scale", log_scale * bound),
            ("variance", variance * bound),
            ("loglik", loglik),
            ("null loglik", null),
            ("linear predictor", lp * bound),
        ];
        let mut units = vec![coefficient, log_scale, variance, loglik, null, lp];
        // Predictions, which R leaves undefined for an aliased column.
        if !rank_deficient {
            let m = r.predict_rows;
            let xm = Mat::from_fn(m, p, |i, j| r.x[i][j]);
            let strata: Vec<u32> = (0..m).map(|i| c.stratum[i]).collect();
            let run = |kind: PredictType| {
                f.predict(
                    xm.as_ref(),
                    &PredictOptions {
                        kind,
                        se_fit: true,
                        strata: r.use_strata.then(|| strata.clone()),
                        ..PredictOptions::default()
                    },
                )
                .unwrap()
            };
            let pb = bound * (p as f64 + 1.0);
            let lp_pred = run(PredictType::LinearPredictor);
            let resp = run(PredictType::Response);
            let quant = run(PredictType::Quantile {
                p: vec![0.1, 0.5, 0.9],
            });
            let eq: Vec<f64> = r.quantile.iter().flatten().map(|v| v.unwrap()).collect();
            let eqs: Vec<f64> = r.quantile_se.iter().flatten().map(|v| v.unwrap()).collect();
            for (what, got, want) in [
                ("lp", &lp_pred.fit, &r.lp),
                ("lp se", lp_pred.se.as_ref().unwrap(), &r.lp_se),
                ("response", &resp.fit, &r.response),
                ("response se", resp.se.as_ref().unwrap(), &r.response_se),
                ("quantile", &quant.fit, &eq),
                ("quantile se", quant.se.as_ref().unwrap(), &eqs),
            ] {
                let u = worst_deviation(got, want, pb, 1.0, NanRule::Unchecked);
                worst.push((what, u * pb));
                units.push(u);
            }
        }
        // Residuals.
        let ones: Vec<f64> = vec![1.0; r.x.len()];
        let _ = &ones;
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
            // The rows R does not compute correctly, see the module
            // documentation of the test: the derivative of an interval row in
            // the log scale has the wrong sign in `residuals.survreg`, which
            // reaches the displacements through the scale, and for the shape
            // displacement also without it. And R's residuals take an interval
            // that starts at 0 as it was given, with a lower bound of minus
            // infinity once logged, where the fit has made it left-censored.
            let keep: Vec<usize> = (0..got.len())
                .filter(|&i| {
                    let interval = c.status[i] == 3;
                    let zero_lower = interval && c.time[i] == 0.0;
                    let displaced = match kind {
                        ResidualKind::LdShape => interval,
                        ResidualKind::LdCase | ResidualKind::LdResp => {
                            interval && f.scale_estimated
                        }
                        _ => false,
                    };
                    !(zero_lower || displaced)
                })
                .collect();
            // The deviance residual is a square root, whose error grows without
            // bound as the radicand goes to zero, so its square is compared,
            // with the sign kept.
            let squared = matches!(kind, ResidualKind::Deviance);
            let pick = |v: &[Option<f64>]| -> Vec<f64> {
                keep.iter()
                    .map(|&i| {
                        let value = none_as_nan(v)[i];
                        if squared { value * value.abs() } else { value }
                    })
                    .collect()
            };
            let u = worst_deviation(&pick(&got), &pick(want), rb, 1.0, NanRule::Unchecked);
            worst.push((what, u * rb));
            units.push(u);
        }
        report.push((r.name.clone(), r.information_condition, worst));
        for (i, u) in units.iter().enumerate() {
            assert!(*u <= 1.0, "{} output {i}: {u} of the bound", r.name);
        }
    }
    for (name, kappa, worst) in &report {
        let cells: Vec<String> = worst.iter().map(|(w, v)| format!("{w} {v:.1e}")).collect();
        eprintln!("{name:26} kappa {kappa:9.2e}  {}", cells.join("  "));
    }
}
