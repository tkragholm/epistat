//! Robust standard errors against references that do not share this crate's code.
//!
//! The fixtures below are checked against model-based and HC0 standard errors
//! computed in R from `glm` fits on the same rows (tests/r/robust_se_reference.R),
//! and a clustered fit with one row per cluster must reproduce the unclustered
//! sandwich.

use epistat_linalg::{Attempt, Mat, Method, inverse};
use epistat_rng::{Stream, replicate_seed};
use epistat_semicontinuous::{
    FitOptions, ModelInput, TweedieOptions, fit_tweedie_input, fit_two_part_input,
};

const N_ROWS: usize = 20_000;

/// Intercept, a uniform covariate and a binary one. `Pr(y > 0)` follows the
/// logit `0.2 + 0.8 x1 - 0.5 x2`, and a positive `y` is exponential with mean
/// `exp(1 + 0.3 x1 + 0.2 x2)`, so both parts are correctly specified with unit
/// dispersion.
fn two_part_fixture() -> (Mat<f64>, Mat<f64>) {
    let mut stream = Stream::seeded(20_261_002);
    let mut x = Mat::<f64>::zeros(N_ROWS, 3);
    let mut y = Mat::<f64>::zeros(N_ROWS, 1);
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
            y[(i, 0)] = mean * draw;
        }
    }
    (x, y)
}

fn positive_rows(x: &Mat<f64>, y: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let rows: Vec<usize> = (0..y.nrows()).filter(|&i| y[(i, 0)] > 0.0).collect();
    (
        Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)]),
        Mat::from_fn(rows.len(), 1, |i, _| y[(rows[i], 0)]),
    )
}

/// Model-based standard errors of a log-link Tweedie fit with unit dispersion,
/// `sqrt(diag((X' diag(mu^(2 - power)) X)^-1))`. The crate reports no model-based
/// covariance for Tweedie fits, so the test forms it here.
fn tweedie_model_based_se(x: &Mat<f64>, beta: &Mat<f64>, power: f64) -> Vec<f64> {
    let p = x.ncols();
    let mut information = Mat::<f64>::zeros(p, p);
    for i in 0..x.nrows() {
        let eta: f64 = (0..p).map(|j| x[(i, j)] * beta[(j, 0)]).sum();
        let weight = eta.exp().powf(2.0 - power);
        for j in 0..p {
            for k in 0..p {
                information[(j, k)] += weight * x[(i, j)] * x[(i, k)];
            }
        }
    }
    let inverse = inverse(information.as_ref(), &[Attempt::plain(Method::Cholesky)])
        .expect("positive-definite information")
        .x;
    (0..p).map(|j| inverse[(j, j)].sqrt()).collect()
}

fn column(m: &Mat<f64>) -> Vec<f64> {
    (0..m.nrows()).map(|i| m[(i, 0)]).collect()
}

fn assert_close(label: &str, got: &[f64], want: &[f64], rel_tol: f64) {
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            ((g - w) / w).abs() < rel_tol,
            "{label}[{k}]: got {g}, want {w} (tolerance {rel_tol})"
        );
    }
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
    let input = ModelInput::new(x, y);
    let (_, model_based) = fit_two_part_input(&input, FitOptions::default()).expect("fit");
    let (_, robust) =
        fit_two_part_input(&input, FitOptions::builder().robust_se(true).build()).expect("fit");

    let model_se = column(model_based.se_logit.as_ref().expect("model-based SE"));
    let robust_se = column(robust.se_logit.as_ref().expect("robust SE"));
    assert_close("logit model-based vs R", &model_se, &R_MODEL_LOGIT, R_TOLERANCE);
    assert_close("logit robust vs R", &robust_se, &R_HC0_LOGIT, R_TOLERANCE);
    assert_close(
        "gamma robust vs R",
        &column(robust.se_gamma.as_ref().expect("robust SE")),
        &R_HC0_GAMMA,
        R_TOLERANCE,
    );
    assert_close(
        "gamma model-based vs R",
        &column(model_based.se_gamma.as_ref().expect("model-based SE")),
        &R_MODEL_GAMMA,
        R_TOLERANCE,
    );
}

#[test]
fn two_part_singleton_clusters_reproduce_the_unclustered_sandwich() {
    let (x, y) = two_part_fixture();
    let clusters: Vec<u64> = (0..N_ROWS as u64).collect();
    let options = FitOptions::builder().robust_se(true).build();
    let (_, plain) = fit_two_part_input(&ModelInput::new(x.clone(), y.clone()), options)
        .expect("unclustered fit");
    let (_, clustered) =
        fit_two_part_input(&ModelInput::new(x, y).with_cluster_ids(clusters), options)
            .expect("clustered fit");
    assert_close(
        "logit",
        &column(clustered.se_logit.as_ref().expect("SE")),
        &column(plain.se_logit.as_ref().expect("SE")),
        1e-9,
    );
    assert_close(
        "gamma",
        &column(clustered.se_gamma.as_ref().expect("SE")),
        &column(plain.se_gamma.as_ref().expect("SE")),
        1e-9,
    );
}

#[test]
fn tweedie_gamma_standard_errors_match_r() {
    let (x, y) = two_part_fixture();
    let (x_pos, y_pos) = positive_rows(&x, &y);
    let input = ModelInput::new(x_pos, y_pos);
    let options = TweedieOptions::builder()
        .robust_se(true)
        .tolerance(TIGHT_TOLERANCE)
        .build();
    let (model, report) = fit_tweedie_input(&input, 2.0, options).expect("fit");

    let robust_se = column(report.se.as_ref().expect("robust SE"));
    let model_se = tweedie_model_based_se(&input.design_matrix, &model.beta, 2.0);
    assert_close(
        "tweedie(2) model-based vs R",
        &model_se,
        &R_MODEL_GAMMA,
        R_TOLERANCE,
    );
    assert_close(
        "tweedie(2) robust vs R",
        &robust_se,
        &R_HC0_GAMMA,
        R_TOLERANCE,
    );
}

#[test]
fn tweedie_standard_errors_match_r_at_power_one_and_a_half() {
    // Gamma(shape = sqrt(mu), scale = sqrt(mu)) has mean mu and variance mu^1.5,
    // the Tweedie variance function at power 1.5 with unit dispersion.
    let mut stream = Stream::seeded(20_261_003);
    let mut rng = Stream::seeded(replicate_seed(20_261_003, 1));
    let mut x = Mat::<f64>::zeros(N_ROWS, 3);
    let mut y = Mat::<f64>::zeros(N_ROWS, 1);
    for i in 0..N_ROWS {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        x[(i, 2)] = if stream.uniform() < 0.3 { 1.0 } else { 0.0 };
        let mu = 0.3f64.mul_add(x[(i, 1)], 2.0 + 0.2 * x[(i, 2)]).exp();
        y[(i, 0)] = rng.gamma(mu.sqrt(), mu.sqrt()).expect("gamma parameters");
    }
    let input = ModelInput::new(x, y);
    let options = TweedieOptions::builder()
        .robust_se(true)
        .tolerance(TIGHT_TOLERANCE)
        .build();
    let (model, report) = fit_tweedie_input(&input, 1.5, options).expect("fit");
    let robust_se = column(report.se.as_ref().expect("robust SE"));
    let model_se = tweedie_model_based_se(&input.design_matrix, &model.beta, 1.5);
    assert_close(
        "tweedie(1.5) model-based vs R",
        &model_se,
        &R_MODEL_TWEEDIE_1_5,
        R_TOLERANCE,
    );
    assert_close(
        "tweedie(1.5) robust vs R",
        &robust_se,
        &R_HC0_TWEEDIE_1_5,
        R_TOLERANCE,
    );

    let clusters: Vec<u64> = (0..N_ROWS as u64).collect();
    let (_, clustered) =
        fit_tweedie_input(&input.with_cluster_ids(clusters), 1.5, options).expect("clustered fit");
    assert_close(
        "tweedie(1.5) singleton clusters",
        &column(clustered.se.as_ref().expect("SE")),
        &robust_se,
        1e-9,
    );
}
