//! Cox fits with time-transform covariates checked against R 4.6.2's
//! `survival::coxph(..., tt = )` on the lung and veteran data, with ties,
//! strata, delayed entry, weights and a cluster.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "strata, clusters and events are small non-negative integers written as doubles"
)]

use std::collections::HashMap;

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties, TimeTransform, TtColumn, TtOptions, fit_tt};
use epistat_testkit::{NanRule, load_json, worst_deviation};
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    survival: String,
    jsonlite: String,
}

#[derive(Deserialize)]
struct Dataset {
    stop: Vec<f64>,
    event: Vec<f64>,
    start: Vec<f64>,
    weight: Vec<f64>,
    columns: HashMap<String, Vec<f64>>,
}

#[derive(Deserialize)]
struct RFit {
    name: String,
    data: String,
    vars: Vec<String>,
    tts: Vec<String>,
    ties: String,
    use_start: bool,
    use_weights: bool,
    strata: Option<Vec<f64>>,
    cluster: Option<Vec<f64>>,
    coefficients: Vec<f64>,
    se: Vec<f64>,
    robust_se: Option<Vec<f64>>,
    loglik: Vec<f64>,
    iterations: usize,
    n_expanded: usize,
    n_risk_sets: usize,
    nevent: f64,
    kappa: f64,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: HashMap<String, Dataset>,
    fits: Vec<RFit>,
}

/// The two fits take the same Newton steps on the same expanded data, so they
/// differ by the error of each solve, about the condition number of the
/// information matrix (scaled to unit diagonal, which the script records)
/// times the machine epsilon, 8 times that as in `r_cox.rs`. The risk sets
/// are visited in another order than R's, which moves the sums by the
/// rounding error of a sum over the rows of the expansion, so `n_expanded`
/// units of the epsilon are added to the bound.
#[test]
fn fits_match_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_tt.json"
    ));
    eprintln!(
        "reference: {}, survival {}, jsonlite {}",
        reference.versions.r, reference.versions.survival, reference.versions.jsonlite
    );
    for r in &reference.fits {
        let d = &reference.datasets[&r.data];
        let rows = d.stop.len();
        let x = Mat::from_fn(rows, r.vars.len(), |i, j| d.columns[&r.vars[j]][i]);
        let status: Vec<i32> = d.event.iter().map(|&e| e as i32).collect();
        let stratum: Option<Vec<u32>> = r
            .strata
            .as_ref()
            .map(|s| s.iter().map(|&v| v as u32).collect());
        let cluster: Option<Vec<u32>> = r
            .cluster
            .as_ref()
            .map(|s| s.iter().map(|&v| v as u32).collect());
        let mut builder = SurvData::builder(&d.stop, &status);
        if r.use_start {
            builder = builder.start(&d.start);
        }
        if r.use_weights {
            builder = builder.weight(&d.weight);
        }
        if let Some(s) = &stratum {
            builder = builder.stratum(s);
        }
        if let Some(c) = &cluster {
            builder = builder.cluster(c);
        }
        let data = builder.build().unwrap();
        let transforms: Vec<TtColumn<'_>> = r
            .tts
            .iter()
            .enumerate()
            .filter(|(_, t)| *t != "none")
            .map(|(column, t)| TtColumn {
                column,
                transform: match t.as_str() {
                    "log" => TimeTransform::LogProduct,
                    "time" => TimeTransform::Product,
                    "step100" => TimeTransform::Step { cut: 100.0 },
                    "obrien" => TimeTransform::OBrien,
                    other => panic!("transform {other}"),
                },
            })
            .collect();
        let opts = TtOptions {
            cox: CoxOptions {
                ties: if r.ties == "efron" {
                    Ties::Efron
                } else {
                    Ties::Breslow
                },
                ..CoxOptions::default()
            },
            max_expanded_rows: None,
        };
        let got = fit_tt(&data, x.as_ref(), &transforms, &opts).unwrap();
        let model = &got.fit;
        let name = &r.name;
        assert_eq!(got.n_expanded, r.n_expanded, "{name}");
        assert_eq!(got.n_risk_sets, r.n_risk_sets, "{name}");
        assert_eq!(model.n_rows, rows, "{name}");
        // R counts the events, the fit sums their weights.
        if !r.use_weights {
            assert!((model.n_events - r.nevent).abs() < 0.5, "{name}");
        }
        assert!(model.converged, "{name}");
        assert_eq!(model.iterations, r.iterations, "{name}");
        let bound = 8.0 * r.kappa * f64::EPSILON + got.n_expanded as f64 * f64::EPSILON;
        let flat = |v: &[Option<f64>]| v.iter().map(|x| x.unwrap()).collect::<Vec<_>>();
        let coefficient = worst_deviation(
            &flat(&model.coefficients),
            &r.coefficients,
            bound,
            1.0,
            NanRule::Coincide,
        );
        let se = worst_deviation(&flat(&model.se), &r.se, bound, 1.0, NanRule::Coincide);
        let robust = match (&model.robust_se, &r.robust_se) {
            (Some(got), Some(expected)) => {
                worst_deviation(&flat(got), expected, bound, 1.0, NanRule::Coincide)
            }
            (None, None) => 0.0,
            _ => panic!("{name}: robust standard errors"),
        };
        let loglik = worst_deviation(
            &[
                model.null_log_partial_likelihood.unwrap(),
                model.log_partial_likelihood.unwrap(),
            ],
            &r.loglik,
            bound,
            1.0,
            NanRule::Coincide,
        );
        eprintln!(
            "{name:30} rows {:6}  coefficient {coefficient:.2e}  se {se:.2e}  robust {robust:.2e}  loglik {loglik:.2e}",
            got.n_expanded
        );
        for (what, v) in [
            ("coefficient", coefficient),
            ("se", se),
            ("robust se", robust),
            ("loglik", loglik),
        ] {
            assert!(v <= 1.0, "{name} {what}: {v} of the bound\n{model:?}");
        }
    }
}
