//! `standardise::attributable::af_from_standardised` checked against `AFcoxph`
//! of the R package AF 0.1.5, run from its source because AF was archived from
//! CRAN. `tests/data/af_cox.R` makes the data and the reference and records
//! the versions.
//!
//! The model is a Cox model with Breslow ties, `X + Z + X:Z`, and the exposure
//! `X` set to 0 in the unexposed level, where the interaction moves with it.
//! `AFcoxph` evaluates the survival as the step function of the Breslow
//! baseline, as `standardise_cox` does without a product limit.
//!
//! The risks and `AF(t)` are compared with `AFcoxph` relative to `1e-9`: both
//! are sums over 800 persons of functions of the same coefficients, which agree
//! to the stopping rule of `coxph` (`eps = 1e-12`, ours the same).
//!
//! The variances of the risks are not compared with `AFcoxph`, whose are too
//! small: its score for the Breslow baseline has the event part of the
//! influence function of the cumulative hazard and not the compensator of the
//! persons at risk, so it falls short by more with time (about 9% at the first
//! time and 40% at the last on these data). Its variance of `AF(t)` is within
//! 7% of ours here, since the term is largely common to the two risks. The
//! reference is the delete-one jackknife of the
//! standardised risks and of `AF(t)`, refitted without each person in R. The
//! jackknife and the influence function differ by terms of order `1/n`; with
//! 800 persons and a baseline of up to 800 jumps the bound is 3% relative, and
//! the largest deviation seen is about 2%. The observed deviation is reported.

use epistat_glm::attributable::IntervalScale;
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_survival::standardise::attributable::af_from_standardised;
use epistat_survival::standardise::{Counterfactual, Level, StandardiseOptions, standardise_cox};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    #[serde(rename = "AF")]
    af: String,
    survival: String,
}

#[derive(Deserialize)]
struct Case {
    label: String,
    time: Vec<f64>,
    status: Vec<i32>,
    design: Vec<Vec<f64>>,
    design0: Vec<Vec<f64>>,
    times: Vec<f64>,
    coef: Vec<f64>,
    af: Vec<f64>,
    af_var: Vec<f64>,
    s: Vec<f64>,
    s0: Vec<f64>,
    jk_s_var: Vec<f64>,
    jk_s0_var: Vec<f64>,
    jk_af_var: Vec<f64>,
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

#[test]
fn the_attributable_fraction_function_matches_afcoxph() {
    let reference: Reference = load_json("tests/data/af_cox.json");
    assert!(reference.versions.r.starts_with("R version 4.6"));
    assert!(reference.versions.af.starts_with("0.1.5"));
    assert_eq!(reference.versions.survival, "3.8.12");
    for c in &reference.cases {
        let data = SurvData::builder(&c.time, &c.status).build().unwrap();
        let x = matrix(&c.design);
        let x0 = matrix(&c.design0);
        let options = CoxOptions {
            ties: Ties::Breslow,
            eps: 1e-12,
            iter_max: 100,
            ..CoxOptions::default()
        };
        let model = fit(&data, x.as_ref(), &options).unwrap();
        for (j, b) in c.coef.iter().enumerate() {
            let got = model.coefficients[j].unwrap();
            assert!(
                (got - b).abs() <= 1e-8 * model.standard_errors[j].unwrap(),
                "{}",
                c.label
            );
        }
        let observed = Level::single(Counterfactual::Columns(&[]));
        let unexposed = Level::single(Counterfactual::Design(x0.as_ref()));
        let opts = StandardiseOptions {
            std_err: true,
            ..StandardiseOptions::default()
        };
        let s = standardise_cox(
            &data,
            x.as_ref(),
            &model,
            &options,
            &[unexposed, observed],
            &c.times,
            &opts,
        )
        .unwrap();
        let curve = af_from_standardised(
            &s,
            &c.times,
            0,
            1,
            epistat_survival::Level::default(),
            IntervalScale::Identity,
        )
        .unwrap();
        let t = c.times.len();
        let mut worst = [0.0f64; 4];
        let mut af_package_ratio = (f64::INFINITY, 0.0f64);
        let cov = s.covariance.as_ref().unwrap();
        for k in 0..t {
            let f0 = s.risk[k].unwrap();
            let f1 = s.risk[t + k].unwrap();
            worst[0] = worst[0].max(((1.0 - f1) - c.s[k]).abs() / c.s[k]);
            worst[0] = worst[0].max(((1.0 - f0) - c.s0[k]).abs() / c.s0[k]);
            let af = curve.af[k].unwrap();
            worst[0] = worst[0].max((af - c.af[k]).abs() / c.af[k].abs());
            let var = curve.se[k].unwrap().powi(2);
            worst[1] = worst[1].max((var - c.jk_af_var[k]).abs() / c.jk_af_var[k]);
            let v1 = cov[k * 4 + 3].unwrap();
            let v0 = cov[k * 4].unwrap();
            worst[2] = worst[2].max((v1 - c.jk_s_var[k]).abs() / c.jk_s_var[k]);
            worst[3] = worst[3].max((v0 - c.jk_s0_var[k]).abs() / c.jk_s0_var[k]);
            let ratio = c.af_var[k] / var;
            af_package_ratio = (af_package_ratio.0.min(ratio), af_package_ratio.1.max(ratio));
        }
        assert!(worst[0] <= 1e-9, "{}: risks and AF {:e}", c.label, worst[0]);
        for (name, w) in ["AF variance", "risk variance", "unexposed risk variance"]
            .iter()
            .zip(&worst[1..])
        {
            assert!(
                *w <= 0.03,
                "{}: {name} against the jackknife {w:e}",
                c.label
            );
        }
        eprintln!(
            "{}: risks and AF {:e}; against the jackknife AF variance {:e}, risk variances {:e} and {:e}; AFcoxph variance over ours {:.3} to {:.3}",
            c.label, worst[0], worst[1], worst[2], worst[3], af_package_ratio.0, af_package_ratio.1
        );
    }
}
