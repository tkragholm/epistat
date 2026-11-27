//! Statistical properties of the pooled estimates over simulated studies.
//!
//! 1. A linear regression with a covariate missing at random: with `norm` the
//!    pooled coefficient is unbiased and its 95 per cent interval covers the
//!    true value at the nominal rate, and with `pmm` it agrees with mice. Both
//!    agree with the same simulations imputed by mice (`tests/data/impute_linear.R`).
//! 2. A Cox model with a covariate missing at random given the outcome: with the
//!    event indicator and the Nelson-Aalen cumulative hazard in the imputation
//!    model the pooled log hazard ratio is close to the truth, and with
//!    neither it is biased towards 0 (White and Royston 2009). Both are
//!    compared with the same simulations imputed by mice
//!    (`tests/data/impute_cox.R`), which generate their own random numbers.
//!
//! Bounds. Over 500 studies the mean estimate has standard error
//! `sd / sqrt(500)`, and bias is tested at 4 standard errors. Coverage is a
//! proportion with standard error `sqrt(0.95 * 0.05 / 500) = 0.0097`. It is
//! tested at 4 standard errors, which also leaves room for the small
//! departure from nominal that Rubin's rules with `m = 10` are known to have.
//! The White-Royston imputation is a linear approximation that is accurate for
//! small effects: at a log hazard ratio of 0.3 mice's pooled estimate is about
//! 0.02 low, and the test records that rather than demanding exact
//! unbiasedness.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "test code: codes and counts are small whole numbers, and exact comparisons are intended"
)]

use epistat_impute::{
    Column, ImputeOptions, ImputeSpec, Incomplete, Method, Ties, impute, nelson_aalen,
};
use epistat_inference::pooling::{Pooled, dfcom_events, dfcom_residual};
use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, fit};
use epistat_testkit::{SplitMix64, assert_within_standard_errors, load_json};
use rayon::prelude::*;
use serde::Deserialize;

mod common;

const STUDIES: usize = 500;

fn normal(rng: &mut SplitMix64) -> f64 {
    let (u, v) = (1.0 - rng.unit(), rng.unit());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

struct Summary {
    mean: f64,
    se: f64,
    coverage: f64,
}

fn summarise(results: &[(f64, bool)]) -> Summary {
    let q: Vec<f64> = results.iter().map(|r| r.0).collect();
    Summary {
        mean: common::mean(&q),
        se: (common::variance(&q) / q.len() as f64).sqrt(),
        coverage: results.iter().filter(|r| r.1).count() as f64 / results.len() as f64,
    }
}

fn covers(pooled: &Pooled, truth: f64) -> bool {
    let interval = pooled.interval(epistat_core::Level::alpha(0.05).unwrap());
    let (lo, hi) = (interval.lower.unwrap(), interval.upper.unwrap());
    lo <= truth && truth <= hi
}

fn linear_study(study: usize, method: &Method) -> (f64, bool) {
    let n = 200;
    let mut rng = SplitMix64::new(1_000_003 * (study as u64 + 1));
    let truth = 0.5;
    let (mut x, mut z, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        x[i] = normal(&mut rng);
        z[i] = normal(&mut rng);
        y[i] = 1.0 + truth * x[i] + 0.5 * z[i] + normal(&mut rng);
    }
    let missing: Vec<bool> = (0..n)
        .map(|i| rng.unit() < expit(-0.8 + 0.7 * z[i] + 0.5 * y[i]))
        .collect();
    let data = Incomplete::new(
        vec![
            Column::Continuous(x),
            Column::Continuous(z),
            Column::Continuous(y),
        ],
        &[missing, vec![false; n], vec![false; n]],
    )
    .unwrap();
    let mut spec = ImputeSpec::default_for(&data);
    spec.methods[0] = Some(method.clone());
    let imputed = impute(
        &data,
        &spec,
        &ImputeOptions {
            m: 10,
            max_iterations: 5,
            seed: 7_000_001 + study as u64,
        },
    )
    .unwrap();
    let pooled = imputed
        .pool(&data, Some(dfcom_residual(n, 3)), |c| {
            let (coef, var) = common::lm(&[&c[0], &c[1]], &c[2]);
            Ok::<_, std::convert::Infallible>((coef[1], var[1]))
        })
        .unwrap();
    (pooled.estimate, covers(&pooled, truth))
}

#[derive(Deserialize)]
struct LinearEstimates {
    estimate: Vec<f64>,
    covered: Vec<f64>,
}

#[derive(Deserialize)]
struct LinearResults {
    norm: LinearEstimates,
    pmm: LinearEstimates,
}

#[derive(Deserialize)]
struct LinearReference {
    results: LinearResults,
}

fn coverage_se(p: f64) -> f64 {
    (p * (1.0 - p) / STUDIES as f64).sqrt()
}

#[test]
fn linear_regression_is_unbiased_with_nominal_coverage() {
    let r: LinearReference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/impute_linear.json"
    ));
    for (label, method, reference) in [
        ("norm", Method::Norm, &r.results.norm),
        ("pmm", Method::pmm(), &r.results.pmm),
    ] {
        let results: Vec<(f64, bool)> = (0..STUDIES)
            .into_par_iter()
            .map(|s| linear_study(s, &method))
            .collect();
        let s = summarise(&results);
        let theirs = Summary {
            mean: common::mean(&reference.estimate),
            se: (common::variance(&reference.estimate) / STUDIES as f64).sqrt(),
            coverage: common::mean(&reference.covered),
        };
        eprintln!(
            "linear {label}: mean {:.4} (se {:.4}), coverage {:.3}; mice mean {:.4} (se {:.4}), coverage {:.3}",
            s.mean, s.se, s.coverage, theirs.mean, theirs.se, theirs.coverage
        );
        // Both implementations give the same mean and coverage.
        assert_within_standard_errors(s.mean, s.se, theirs.mean, theirs.se, 4.0);
        assert_within_standard_errors(
            s.coverage,
            coverage_se(s.coverage),
            theirs.coverage,
            coverage_se(theirs.coverage),
            4.0,
        );
        if label == "norm" {
            // Normal imputation of a normal covariate is correct: unbiased and nominal.
            assert_within_standard_errors(s.mean, s.se, 0.5, 0.0, 4.0);
            assert_within_standard_errors(s.coverage, coverage_se(0.95), 0.95, 0.0, 4.0);
        } else {
            // Predictive mean matching with 5 donors under-covers at this size
            // (Morris, White and Royston 2014), as mice's does; the bias is small.
            assert!(s.coverage > 0.88, "{}", s.coverage);
        }
    }
}

const BETA: f64 = 0.3;

fn cox_study(study: usize, with_outcome: bool) -> (f64, bool) {
    let n = 300;
    let mut rng = SplitMix64::new(2_000_003 * (study as u64 + 1));
    let (mut x, mut z) = (vec![0.0; n], vec![0.0; n]);
    let (mut time, mut status) = (vec![0.0; n], vec![0_i32; n]);
    for i in 0..n {
        x[i] = normal(&mut rng);
        z[i] = normal(&mut rng);
        let rate = 0.1 * (BETA * x[i] + 0.3 * z[i]).exp();
        let t_event = -(1.0 - rng.unit()).ln() / rate;
        let t_cens = -(1.0 - rng.unit()).ln() / 0.05;
        time[i] = t_event.min(t_cens);
        status[i] = i32::from(t_event <= t_cens);
    }
    let missing: Vec<bool> = (0..n)
        .map(|i| rng.unit() < expit(-1.6 + f64::from(status[i]) + 0.5 * z[i]))
        .collect();
    let hazard = nelson_aalen(&time, &status, Ties::Efron).unwrap();
    let data = Incomplete::new(
        vec![
            Column::Continuous(x),
            Column::Continuous(z),
            Column::Binary(status.iter().map(|&s| s as u8).collect()),
            Column::Continuous(hazard),
            Column::Continuous(time.clone()),
        ],
        &[
            missing,
            vec![false; n],
            vec![false; n],
            vec![false; n],
            vec![false; n],
        ],
    )
    .unwrap();
    let mut spec = ImputeSpec::default_for(&data);
    spec.methods[0] = Some(Method::Norm);
    spec.predictors[0] = if with_outcome { vec![1, 2, 3] } else { vec![1] };
    let imputed = impute(
        &data,
        &spec,
        &ImputeOptions {
            m: 10,
            max_iterations: 5,
            seed: 9_000_001 + study as u64,
        },
    )
    .unwrap();
    let events = status.iter().filter(|&&s| s == 1).count();
    let pooled = imputed
        .pool(&data, Some(dfcom_events(events, 2)), |c| {
            let surv = SurvData::builder(&time, &status).build()?;
            let x = Mat::from_fn(n, 2, |i, j| c[j][i]);
            let model = fit(&surv, x.as_ref(), &CoxOptions::default())?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>((
                model.coefficients[0].unwrap(),
                model.covariance.get(0, 0).unwrap(),
            ))
        })
        .unwrap();
    (pooled.estimate, covers(&pooled, BETA))
}

#[derive(Deserialize)]
struct Estimates {
    estimate: Vec<f64>,
}

#[derive(Deserialize)]
struct Results {
    with: Estimates,
    without: Estimates,
}

#[derive(Deserialize)]
struct CoxReference {
    beta: f64,
    n: usize,
    results: Results,
}

#[test]
fn cox_log_hazard_ratio_needs_the_outcome_in_the_imputation_model() {
    let r: CoxReference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/impute_cox.json"
    ));
    assert!((r.beta - BETA).abs() == 0.0 && r.n == 300);
    let run = |with_outcome: bool| -> Summary {
        let results: Vec<(f64, bool)> = (0..STUDIES)
            .into_par_iter()
            .map(|s| cox_study(s, with_outcome))
            .collect();
        summarise(&results)
    };
    let (with, without) = (run(true), run(false));
    let reference = |e: &Estimates| {
        (
            common::mean(&e.estimate),
            (common::variance(&e.estimate) / e.estimate.len() as f64).sqrt(),
        )
    };
    let (rw, ro) = (reference(&r.results.with), reference(&r.results.without));
    eprintln!(
        "cox with outcome: {:.4} (se {:.4}) coverage {:.3}; mice {:.4} (se {:.4})",
        with.mean, with.se, with.coverage, rw.0, rw.1
    );
    eprintln!(
        "cox without outcome: {:.4} (se {:.4}) coverage {:.3}; mice {:.4} (se {:.4})",
        without.mean, without.se, without.coverage, ro.0, ro.1
    );
    // The two implementations agree on both estimates.
    assert_within_standard_errors(with.mean, with.se, rw.0, rw.1, 4.0);
    assert_within_standard_errors(without.mean, without.se, ro.0, ro.1, 4.0);
    // With the outcome the estimate is within a tenth of the truth, without it
    // is biased towards 0 by more than a fifth and more than the other.
    assert!((with.mean - BETA).abs() < 0.1 * BETA, "with: {}", with.mean);
    assert!(
        BETA - without.mean > 0.2 * BETA,
        "without: {}",
        without.mean
    );
    assert!(BETA - without.mean > 2.0 * (BETA - with.mean));
    // With the outcome the interval covers at about the nominal rate, without it
    // clearly less often.
    assert!(with.coverage > 0.92, "with: {}", with.coverage);
    assert!(
        without.coverage < with.coverage - 0.04,
        "without: {}",
        without.coverage
    );
}
