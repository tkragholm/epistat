//! Robust standard errors against references that do not share this crate's code.
//!
//! The fixtures below are checked against model-based and HC0 standard errors
//! computed in R from `glm` fits on the same rows (tests/r/robust_se_reference.R),
//! and a clustered fit with one row per cluster must reproduce the unclustered
//! sandwich.
//!
//! The R model-based standard errors are at a dispersion fixed at 1. The crate
//! scales the model-based covariance of a gamma or Tweedie fit by the Pearson
//! dispersion `phi`, as `summary.glm` does, so its standard errors divided by
//! `sqrt(phi)` are compared with them.

use epistat_linalg::Mat;
use epistat_rng::{Stream, replicate_seed};
use epistat_semicontinuous::tweedie::{TweedieModel, TweedieOptions};
use epistat_semicontinuous::two_part::{PartFit, TwoPartOptions};
use epistat_semicontinuous::{CostData, Variance, tweedie, two_part};
use epistat_testkit::{Tolerance, assert_close};

const N_ROWS: usize = 20_000;

/// Intercept, a uniform covariate and a binary one. `Pr(y > 0)` follows the
/// logit `0.2 + 0.8 x1 - 0.5 x2`, and a positive `y` is exponential with mean
/// `exp(1 + 0.3 x1 + 0.2 x2)`, so both parts are correctly specified with unit
/// dispersion.
fn two_part_fixture() -> (Mat<f64>, Vec<f64>) {
    let mut stream = Stream::seeded(20_261_002);
    let mut x = Mat::<f64>::zeros(N_ROWS, 3);
    let mut y = vec![0.0; N_ROWS];
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        x[(i, 2)] = if stream.uniform() < 0.3 { 1.0 } else { 0.0 };
        let eta_logit = 0.8f64.mul_add(x[(i, 1)], 0.2) - 0.5 * x[(i, 2)];
        let prob = 1.0 / (1.0 + (-eta_logit).exp());
        let positive = stream.uniform() < prob;
        let draw = -(1.0 - stream.uniform()).ln();
        if positive {
            let mean = 0.3f64.mul_add(x[(i, 1)], 1.0 + 0.2 * x[(i, 2)]).exp();
            y[i] = mean * draw;
        }
    }
    (x, y)
}

fn positive_rows(x: &Mat<f64>, y: &[f64]) -> (Mat<f64>, Vec<f64>) {
    let rows: Vec<usize> = (0..y.len()).filter(|&i| y[i] > 0.0).collect();
    (
        Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)]),
        rows.iter().map(|&i| y[i]).collect(),
    )
}

/// The standard errors of a fit, each defined.
fn standard_errors(se: &[Option<f64>]) -> Vec<f64> {
    se.iter().map(|se| se.expect("standard error")).collect()
}

/// The standard errors of a part at unit dispersion.
fn unit_dispersion_se(part: &PartFit) -> Vec<f64> {
    let root = part.dispersion.expect("dispersion").sqrt();
    standard_errors(&part.se).iter().map(|se| se / root).collect()
}

/// The model-based standard errors of a Tweedie fit at unit dispersion.
fn tweedie_unit_dispersion_se(model: &TweedieModel) -> Vec<f64> {
    let root = model.dispersion.expect("dispersion").sqrt();
    standard_errors(&model.se).iter().map(|se| se / root).collect()
}

/// HC0 standard errors from R 4 on the rows of `two_part_fixture`:
/// `glm(pos ~ x1 + x2, binomial)` for the logit part and
/// `glm(y ~ x1 + x2, Gamma(link = "log"), subset = y > 0)` for the positive part,
/// each with `B %*% crossprod(X * r) %*% B`, `B = solve(crossprod(X * sqrt(w)))`,
/// `w` the working weights and `r` the score residual.
const R_HC0_LOGIT: [f64; 3] = [0.017_497_413_181, 0.025_828_785_236, 0.031_797_081_066];
const R_HC0_GAMMA: [f64; 3] = [0.011_792_762_166, 0.018_088_586_845, 0.022_420_902_035];

/// Model-based standard errors with dispersion fixed at 1, from the same R fits
/// (R 4.6.2, statmod 1.5.2).
const R_MODEL_LOGIT: [f64; 3] = [0.017_465_963_733, 0.025_869_203_396, 0.031_874_374_141];
const R_MODEL_GAMMA: [f64; 3] = [0.011_592_833_511, 0.017_520_811_121, 0.022_584_796_660];

/// Model-based and HC0 standard errors of `glm(y ~ x1 + x2, tweedie(var.power = 1.5, link.power = 0))`
/// on the rows of the power 1.5 test below.
const R_MODEL_TWEEDIE_1_5: [f64; 3] = [0.005_147_139_437, 0.007_259_768_307, 0.008_985_563_551];
const R_HC0_TWEEDIE_1_5: [f64; 3] = [0.005_147_950_007, 0.007_327_893_604, 0.008_974_668_756];

/// Convergence tolerance of the Tweedie fits compared with R. The default 1e-6
/// stops on a small step and leaves the standard errors about 1e-7 from R's.
const TIGHT_TOLERANCE: f64 = 1e-12;

/// Relative tolerance against R. Both sides solve the same estimating equations
/// to a step near 1e-12, and a standard error varies with the coefficients at
/// order one, so the fits agree to about 1e-9. R reads the
/// fixture rows with its own decimal parser, a further 1e-16. The R values are
/// printed to twelve digits. 1e-8 leaves a margin of a factor ten over both.
const R_TOLERANCE: f64 = 1e-8;

#[test]
fn two_part_standard_errors_match_r() {
    let (x, y) = two_part_fixture();
    let data = CostData::builder(x.as_ref(), &y).build().expect("data");
    let model_based = two_part::fit(&data, &TwoPartOptions::default(), None).expect("fit");
    let robust_options = TwoPartOptions {
        variance: Variance::Sandwich,
        ..TwoPartOptions::default()
    };
    let robust = two_part::fit(&data, &robust_options, None).expect("fit");

    // The logit part has dispersion 1.
    assert_eq!(model_based.logit.dispersion, Some(1.0));
    let model_se = standard_errors(&model_based.logit.se);
    let robust_se = standard_errors(&robust.logit.se);
    assert_close(
        &model_se,
        &R_MODEL_LOGIT,
        Tolerance::exact().rel(R_TOLERANCE),
    );
    assert_close(
        &robust_se,
        &R_HC0_LOGIT,
        Tolerance::exact().rel(R_TOLERANCE),
    );
    assert_close(
        &standard_errors(&robust.gamma.se),
        &R_HC0_GAMMA,
        Tolerance::exact().rel(R_TOLERANCE),
    );
    assert_close(
        &unit_dispersion_se(&model_based.gamma),
        &R_MODEL_GAMMA,
        Tolerance::exact().rel(R_TOLERANCE),
    );
}

#[test]
fn two_part_singleton_clusters_reproduce_the_unclustered_sandwich() {
    let (x, y) = two_part_fixture();
    let clusters: Vec<u32> = (0..N_ROWS)
        .map(|i| u32::try_from(i).expect("small"))
        .collect();
    let options = TwoPartOptions {
        variance: Variance::Sandwich,
        ..TwoPartOptions::default()
    };
    let data = CostData::builder(x.as_ref(), &y).build().expect("data");
    let plain = two_part::fit(&data, &options, None).expect("unclustered fit");
    let clustered_data = CostData::builder(x.as_ref(), &y)
        .clusters(&clusters)
        .build()
        .expect("data");
    let clustered = two_part::fit(&clustered_data, &options, None).expect("clustered fit");
    assert_eq!(clustered.report.cluster_count, Some(N_ROWS));
    assert_close(
        &standard_errors(&clustered.logit.se),
        &standard_errors(&plain.logit.se),
        Tolerance::exact().rel(1e-9),
    );
    assert_close(
        &standard_errors(&clustered.gamma.se),
        &standard_errors(&plain.gamma.se),
        Tolerance::exact().rel(1e-9),
    );
}

#[test]
fn tweedie_gamma_standard_errors_match_r() {
    let (x, y) = two_part_fixture();
    let (x_pos, y_pos) = positive_rows(&x, &y);
    let data = CostData::builder(x_pos.as_ref(), &y_pos)
        .build()
        .expect("data");
    let options = TweedieOptions {
        variance: Variance::Sandwich,
        tolerance: TIGHT_TOLERANCE,
        ..TweedieOptions::default()
    };
    let model = tweedie::fit(&data, 2.0, &options, None, None).expect("fit");
    let model_based = TweedieOptions {
        variance: Variance::ModelBased,
        ..options
    };
    let fitted = tweedie::fit(&data, 2.0, &model_based, None, None).expect("fit");

    let robust_se = standard_errors(&model.se);
    let model_se = tweedie_unit_dispersion_se(&fitted);
    assert_close(
        &model_se,
        &R_MODEL_GAMMA,
        Tolerance::exact().rel(R_TOLERANCE),
    );
    assert_close(
        &robust_se,
        &R_HC0_GAMMA,
        Tolerance::exact().rel(R_TOLERANCE),
    );
}

#[test]
fn tweedie_standard_errors_match_r_at_power_one_and_a_half() {
    // Gamma(shape = sqrt(mu), scale = sqrt(mu)) has mean mu and variance mu^1.5,
    // the Tweedie variance function at power 1.5 with unit dispersion.
    let mut stream = Stream::seeded(20_261_003);
    let mut rng = Stream::seeded(replicate_seed(20_261_003, 1));
    let mut x = Mat::<f64>::zeros(N_ROWS, 3);
    let mut y = vec![0.0; N_ROWS];
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        x[(i, 2)] = if stream.uniform() < 0.3 { 1.0 } else { 0.0 };
        let mu = 0.3f64.mul_add(x[(i, 1)], 2.0 + 0.2 * x[(i, 2)]).exp();
        y[i] = rng.gamma(mu.sqrt(), mu.sqrt()).expect("gamma parameters");
    }
    let data = CostData::builder(x.as_ref(), &y).build().expect("data");
    let options = TweedieOptions {
        variance: Variance::Sandwich,
        tolerance: TIGHT_TOLERANCE,
        ..TweedieOptions::default()
    };
    let model = tweedie::fit(&data, 1.5, &options, None, None).expect("fit");
    let model_based = TweedieOptions {
        variance: Variance::ModelBased,
        ..options
    };
    let fitted = tweedie::fit(&data, 1.5, &model_based, None, None).expect("fit");
    let robust_se = standard_errors(&model.se);
    let model_se = tweedie_unit_dispersion_se(&fitted);
    assert_close(
        &model_se,
        &R_MODEL_TWEEDIE_1_5,
        Tolerance::exact().rel(R_TOLERANCE),
    );
    assert_close(
        &robust_se,
        &R_HC0_TWEEDIE_1_5,
        Tolerance::exact().rel(R_TOLERANCE),
    );

    let clusters: Vec<u32> = (0..N_ROWS)
        .map(|i| u32::try_from(i).expect("small"))
        .collect();
    let clustered_data = CostData::builder(x.as_ref(), &y)
        .clusters(&clusters)
        .build()
        .expect("data");
    let clustered =
        tweedie::fit(&clustered_data, 1.5, &options, None, None).expect("clustered fit");
    assert_close(
        &standard_errors(&clustered.se),
        &robust_se,
        Tolerance::exact().rel(1e-9),
    );
}
