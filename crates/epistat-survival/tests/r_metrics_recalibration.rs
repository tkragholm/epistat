//! `metrics::recalibration` against `riskRegression::plotCalibration(method =
//! "nne")` with the pseudo-values of `Score(cens.method = "pseudo")`, and against
//! `geepack::geese.fit` on `prodlim::jackknife` pseudo-values, on data whose
//! times tie (`tests/data/metrics_recalibration.R`).
//!
//! # Tolerances
//!
//! The pseudo-value `y_i = n F - (n - 1) F_(-i)` of a product-limit estimate of
//! at most `n` factors differs between two programs by at most `delta_y = 4 n (n
//! + 3) eps`: each estimate by `2 (n + 3) eps`, multiplied by `n` and `n - 1`.
//!
//! - A point of the smoothed curve is a mean of pseudo-values, so it differs by
//!   at most `delta_y`, and the mean predicted risk by `(n + 2) eps` for the sum.
//!   The groups and their sizes are integers and must agree exactly.
//! - The coefficients solve `sum_i d_i x_i (y_i - mu_i) = 0`. A change `delta_y`
//!   of the responses moves the solution by `A^-1 sum_i d_i x_i delta_y`, at
//!   most `kappa / ||A||_1 n d_max x_max delta_y` in the largest coefficient,
//!   with `kappa` the condition number and `||A||_1` the 1-norm of the
//!   information, `d_max = 1/4` for the logit link and `0.368` for the
//!   complementary log-log link, and `x_max` the largest design entry. The
//!   iteration adds `2e-12` times the larger of 1 and the largest coefficient,
//!   from stopping both programs at `1e-13` with a rate of at most 0.9, and the
//!   solve `16 kappa (n + p) eps` times the largest coefficient, as in
//!   `r_pseudo_glm.rs`.
//! - The covariance is a smooth function of the coefficients and the responses,
//!   a product of sums over `n` rows: `(32 kappa (n + p) eps + 8 t_c)
//!   sqrt(V_ii V_jj)` with `t_c` the allowance of the coefficients, as there.
//!
//! The bounds are not changed after a run.

use epistat_survival::SurvData;
use epistat_survival::metrics::recalibration::{
    Bandwidth, RecalibrationOptions, RecalibrationScale, SmoothOptions, slope_intercept,
    smoothed_curve,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Data {
    time: Vec<f64>,
    cr: Vec<i32>,
    any: Vec<i32>,
    times: Vec<f64>,
    risk_s: Vec<Vec<f64>>,
    risk_cr: Vec<Vec<f64>>,
    risk_few: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Nne {
    label: String,
    kind: String,
    j: usize,
    bandwidth: f64,
    round: i32,
    pred: Vec<f64>,
    obs: Vec<f64>,
    r_bandwidth: f64,
}

#[derive(Deserialize)]
struct Slope {
    kind: String,
    j: usize,
    link: String,
    coef: Vec<f64>,
    vcov: Vec<f64>,
    converged: bool,
}

#[derive(Deserialize)]
struct Reference {
    versions: std::collections::BTreeMap<String, String>,
    data: Data,
    nne: Vec<Nne>,
    slopes: Vec<Slope>,
}

#[test]
fn recalibration_against_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/metrics_recalibration.json"
    ));
    eprintln!("reference: {:?}", r.versions);
    let d = &r.data;
    let n = d.time.len();
    let nf = n as f64;
    let eps = f64::EPSILON;
    let delta_y = 4.0 * nf * (nf + 3.0) * eps;
    let surv = SurvData::builder(&d.time, &d.any).build().unwrap();
    let comp = SurvData::builder(&d.time, &d.cr)
        .n_codes(2)
        .build()
        .unwrap();
    let column = |m: &[Vec<f64>], j: usize| -> Vec<f64> { m.iter().map(|row| row[j]).collect() };
    let mut worst_obs = 0.0f64;
    for case in &r.nne {
        let (data, risk) = match case.label.as_str() {
            l if l.contains("few") => (&comp, column(&d.risk_few, case.j)),
            _ if case.kind == "survival" => (&surv, column(&d.risk_s, case.j)),
            _ => (&comp, column(&d.risk_cr, case.j)),
        };
        let mut opts = SmoothOptions::new(if (case.bandwidth - nf.powf(-0.25)).abs() < 1e-15 {
            Bandwidth::Smooth
        } else {
            Bandwidth::Fixed(case.bandwidth)
        });
        opts.rounding = (case.round > 0).then_some(case.round);
        let got = smoothed_curve(data, &risk, d.times[case.j], &opts).unwrap();
        assert!(
            (got.bandwidth - case.r_bandwidth).abs() <= 4.0 * eps * case.r_bandwidth,
            "{}: bandwidth {} against {}",
            case.label,
            got.bandwidth,
            case.r_bandwidth
        );
        assert_eq!(got.points.len(), case.pred.len(), "{}: points", case.label);
        for (k, point) in got.points.iter().enumerate() {
            // The rounded predictions are equal exactly; the mean of all of
            // them, for a bandwidth of 1, is a sum of n terms in [0, 1].
            if case.bandwidth >= 1.0 {
                assert!(
                    (point.predicted - case.pred[k]).abs() <= (nf + 2.0) * eps,
                    "{}: mean predicted {} against {}",
                    case.label,
                    point.predicted,
                    case.pred[k]
                );
            } else {
                assert_eq!(
                    point.predicted.to_bits(),
                    case.pred[k].to_bits(),
                    "{}: predicted {k}",
                    case.label
                );
            }
            let dev = (point.observed - case.obs[k]).abs() / delta_y;
            assert!(
                dev <= 1.0,
                "{}: observed {k} is {}, R {}, bound {delta_y}",
                case.label,
                point.observed,
                case.obs[k]
            );
            worst_obs = worst_obs.max(dev);
        }
    }
    eprintln!("worst smoothed observed risk: {worst_obs:.3} of its bound");

    let (mut worst_c, mut worst_v) = (0.0f64, 0.0f64);
    for case in &r.slopes {
        assert!(case.converged);
        let (data, risk) = if case.kind == "survival" {
            (&surv, column(&d.risk_s, case.j))
        } else {
            (&comp, column(&d.risk_cr, case.j))
        };
        let (scale, d_max) = if case.link == "logit" {
            (RecalibrationScale::Logit, 0.25)
        } else {
            (RecalibrationScale::Cloglog, 0.368)
        };
        let mut opts = RecalibrationOptions::new(scale);
        opts.tolerance = 1e-13;
        opts.max_iterations = 200;
        let got = slope_intercept(data, &risk, d.times[case.j], &opts).unwrap();
        assert!(got.converged);
        let kappa = got.information_condition;
        let norm_a = (got.information[0].abs() + got.information[2].abs())
            .max(got.information[1].abs() + got.information[3].abs());
        let x_max = risk
            .iter()
            .map(|&p| match scale {
                RecalibrationScale::Logit => (p / (1.0 - p)).ln().abs(),
                RecalibrationScale::Cloglog => (-(-p).ln_1p()).ln().abs(),
            })
            .fold(1.0f64, f64::max);
        let big = case.coef.iter().fold(1.0f64, |m, b| m.max(b.abs()));
        let t_c = kappa / norm_a * nf * d_max * x_max * delta_y
            + 2e-12 * big
            + 16.0 * kappa * (nf + 2.0) * eps * big;
        let coef = [got.intercept, got.slope];
        for (j, &value) in coef.iter().enumerate() {
            let dev = (value - case.coef[j]).abs() / t_c;
            assert!(
                dev <= 1.0,
                "{} {} {}: coefficient {j} {} against {}, bound {t_c}",
                case.kind,
                case.link,
                case.j,
                value,
                case.coef[j]
            );
            worst_c = worst_c.max(dev);
        }
        for i in 0..2 {
            for j in 0..2 {
                let tv = (32.0 * kappa * (nf + 2.0) * eps + 8.0 * t_c)
                    * (case.vcov[3 * i] * case.vcov[3 * j]).sqrt();
                let dev = (got.covariance[i * 2 + j] - case.vcov[i * 2 + j]).abs() / tv;
                assert!(
                    dev <= 1.0,
                    "{} {} {}: covariance ({i}, {j}) {} against {}, bound {tv}",
                    case.kind,
                    case.link,
                    case.j,
                    got.covariance[i * 2 + j],
                    case.vcov[i * 2 + j]
                );
                worst_v = worst_v.max(dev);
            }
        }
    }
    eprintln!("worst coefficient {worst_c:.3}, worst covariance {worst_v:.3} of their bounds");
}
