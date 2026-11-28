//! The compact fits against the dense fits, and the cases where a compact fit refuses.
//!
//! Compact and dense fits take the same steps with the arithmetic in another order, so
//! they are compared within a tolerance: `AGREEMENT` for the gamma fits, relative to
//! `1 + |dense value|`, and `LOGIT_AGREEMENT` for the logit part. The dense path with
//! caller-held buffers is compared bit for bit with the dense path without them.

use epistat_core::{CaseWeights, Covariance, InputError};
use epistat_linalg::Mat;
use epistat_semicontinuous::prepared::{
    PreparedGammaDesign, PreparedPredictionDesign, PreparedScratch, PreparedTwoPartDesign,
};
use epistat_semicontinuous::tweedie::{TweedieOptions, TweedieScratch};
use epistat_semicontinuous::two_part::{
    Regularization, Separation, TwoPartOptions, TwoPartWarmStart,
};
use epistat_semicontinuous::{CostData, Error, FitStrategy, Variance, tweedie, two_part};

/// Largest accepted move between the compact and the dense fit of one weight column.
const AGREEMENT: f64 = 1e-8;

/// The same for the fitted logit linear predictors. The logit's last step is the size of
/// the convergence tolerance and its deviance change is the size of the rounding, so
/// either path can halve that step while the other takes it whole.
const LOGIT_AGREEMENT: f64 = 1e-5;

struct Stream(u64);

impl Stream {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn exponential(&mut self) -> f64 {
        -self.uniform().max(1e-12).ln()
    }
}

/// Shaped like a matched cost cohort: an intercept, a calendar year, an age with two
/// spline-like terms, dummies from common to rare, and a heavy-tailed cost.
fn design(n: usize, zero_share: f64, seed: u64) -> (Mat<f64>, Vec<f64>) {
    let mut stream = Stream(seed);
    let p = 16;
    let mut x = Mat::<f64>::zeros(n, p);
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        x[(i, 0)] = 1.0;
        let age = (stream.uniform() * stream.uniform() * 215.0).floor();
        x[(i, 1)] = age;
        x[(i, 2)] = (age / 60.0).powi(3);
        x[(i, 3)] = ((age - 12.0).max(0.0) / 60.0).powi(3);
        x[(i, 4)] = 2000.0 + (stream.uniform() * 19.0).floor();
        for (j, share) in (5..p).zip([
            0.6, 0.45, 0.3, 0.2, 0.15, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005,
        ]) {
            x[(i, j)] = f64::from(stream.uniform() < share);
        }
        let eta = 9.5 + 0.2 * x[(i, 5)] - 0.3 * x[(i, 6)] + 0.5 * x[(i, 12)] - 0.004 * age
            + 0.01 * (x[(i, 4)] - 2010.0);
        let noise = (1.2 * (stream.uniform() - 0.5) + 0.5 * (stream.exponential() - 1.0)).exp();
        y[i] = if stream.uniform() < zero_share {
            0.0
        } else {
            eta.exp() * noise
        };
    }
    (x, y)
}

fn replicate_weights(n: usize, seed: u64) -> Vec<f64> {
    let mut stream = Stream(seed);
    (0..n).map(|_| stream.exponential()).collect()
}

fn close(label: &str, compact: f64, dense: f64) {
    let relative = (compact - dense).abs() / (1.0 + dense.abs());
    assert!(
        relative <= AGREEMENT,
        "{label}: compact {compact} against dense {dense}, {relative:e} apart"
    );
}

fn close_columns(label: &str, compact: &[f64], dense: &[f64]) {
    assert_eq!(compact.len(), dense.len(), "{label}");
    for (j, (&compact, &dense)) in compact.iter().zip(dense).enumerate() {
        close(&format!("{label}[{j}]"), compact, dense);
    }
}

/// The fitted linear predictors over the rows of `x`. The dense logit runs on the raw
/// design, whose intercept and calendar-year coefficients trade off against each other
/// by far more than either moves a prediction.
fn close_predictors(
    label: &str,
    x: &Mat<f64>,
    compact: &[f64],
    dense: &[f64],
    tolerance: f64,
) {
    let mut largest = 0.0_f64;
    for i in 0..x.nrows() {
        let difference: f64 = (0..x.ncols())
            .map(|j| x[(i, j)] * (compact[j] - dense[j]))
            .sum();
        largest = largest.max(difference.abs());
    }
    assert!(
        largest <= tolerance,
        "{label}: linear predictors {largest:e} apart"
    );
}

fn gamma_options() -> TweedieOptions {
    TweedieOptions::default()
}

fn two_part_options() -> TwoPartOptions {
    TwoPartOptions {
        variance: Variance::None,
        ..TwoPartOptions::default()
    }
}

fn case_weights(weights: &[f64]) -> CaseWeights<'_> {
    CaseWeights::new(weights).expect("weights")
}

fn data<'a>(x: &'a Mat<f64>, y: &'a [f64]) -> CostData<'a> {
    CostData::builder(x.as_ref(), y).build().expect("data")
}

fn weighted<'a>(x: &'a Mat<f64>, y: &'a [f64], weights: &'a [f64]) -> CostData<'a> {
    CostData::builder(x.as_ref(), y)
        .weights(case_weights(weights))
        .build()
        .expect("data")
}

fn gamma_warm(x: &Mat<f64>, y: &[f64]) -> Vec<f64> {
    tweedie::fit(&data(x, y), 2.0, &gamma_options(), None, None)
        .expect("point fit")
        .coefficients
}

fn two_part_warm(x: &Mat<f64>, y: &[f64], options: &TwoPartOptions) -> TwoPartWarmStart {
    let model = two_part::fit(&data(x, y), options, None).expect("point");
    TwoPartWarmStart::from(&model)
}

#[test]
fn a_gamma_fit_agrees_with_the_dense_fit() {
    let (x, y) = design(20_000, 0.0, 1);
    let warm = gamma_warm(&x, &y);
    let prepared = PreparedGammaDesign::new(x.as_ref(), &y).expect("design");
    assert_eq!((prepared.nrows(), prepared.ncols()), (20_000, 16));
    let mut scratch = PreparedScratch::default();
    for seed in 0..12 {
        let weights = replicate_weights(20_000, 100 + seed);
        let dense = tweedie::fit(
            &weighted(&x, &y, &weights),
            2.0,
            &gamma_options(),
            Some(&warm),
            None,
        )
        .expect("dense");
        let compact = prepared
            .fit(case_weights(&weights), &warm, &gamma_options(), &mut scratch)
            .expect("compact");
        close_columns("beta", &compact.coefficients, &dense.coefficients);
        assert!(compact.iterations.abs_diff(dense.report.meta.iterations) <= 1);
    }
}

#[test]
fn a_ridge_gamma_fit_agrees_with_the_dense_fit() {
    let (x, y) = design(8_000, 0.0, 2);
    let warm = gamma_warm(&x, &y);
    let options = TweedieOptions {
        l2_penalty: 1e-2,
        ..gamma_options()
    };
    let prepared = PreparedGammaDesign::new(x.as_ref(), &y).expect("design");
    let weights = replicate_weights(8_000, 7);
    let dense = tweedie::fit(
        &weighted(&x, &y, &weights),
        2.0,
        &options,
        Some(&warm),
        None,
    )
    .expect("dense");
    let compact = prepared
        .fit(
            case_weights(&weights),
            &warm,
            &options,
            &mut PreparedScratch::default(),
        )
        .expect("compact");
    close_columns("beta", &compact.coefficients, &dense.coefficients);
}

#[test]
fn a_two_part_fit_agrees_with_the_dense_fit() {
    for (zero_share, n) in [(0.35, 20_000), (0.02, 20_000)] {
        let (x, y) = design(n, zero_share, 3);
        let warm = two_part_warm(&x, &y, &two_part_options());
        let prepared = PreparedTwoPartDesign::new(x.as_ref(), &y).expect("design");
        assert_eq!(prepared.nrows(), n);
        assert!(prepared.positive_nrows() < n);
        let mut scratch = PreparedScratch::default();
        for seed in 0..8 {
            let weights = replicate_weights(n, 200 + seed);
            let dense = two_part::fit(
                &weighted(&x, &y, &weights),
                &two_part_options(),
                Some(&warm),
            )
            .expect("dense");
            let compact = prepared
                .fit(
                    case_weights(&weights),
                    &warm,
                    &two_part_options(),
                    &mut scratch,
                )
                .expect("compact");
            close_predictors(
                "logit",
                &x,
                &compact.logit,
                &dense.logit.coefficients,
                LOGIT_AGREEMENT,
            );
            close_predictors(
                "gamma",
                &x,
                &compact.gamma,
                &dense.gamma.coefficients,
                AGREEMENT,
            );
            assert!(
                compact
                    .iterations_logit
                    .abs_diff(dense.logit.meta.iterations)
                    <= 1
            );
            assert!(
                compact
                    .iterations_gamma
                    .abs_diff(dense.gamma.meta.iterations)
                    <= 1
            );
        }
    }
}

#[test]
fn a_rare_outcome_class_gives_the_intercept_only_logit_of_the_dense_fit() {
    let (x, mut y) = design(6_000, 0.0, 4);
    // Fewer zeros than columns, so the logit is separable.
    for i in 0..5 {
        y[i * 100] = 0.0;
    }
    let options = TwoPartOptions {
        separation: Separation::InterceptOnly,
        ..two_part_options()
    };
    let warm = two_part_warm(&x, &y, &options);
    let prepared = PreparedTwoPartDesign::new(x.as_ref(), &y).expect("design");
    let weights = replicate_weights(6_000, 9);
    let dense = two_part::fit(&weighted(&x, &y, &weights), &options, Some(&warm)).expect("dense");
    let compact = prepared
        .fit(
            case_weights(&weights),
            &warm,
            &options,
            &mut PreparedScratch::default(),
        )
        .expect("compact");
    assert_eq!(compact.iterations_logit, 0);
    assert_eq!(compact.logit_intercept_only, dense.report.logit_intercept_only);
    close_columns("logit", &compact.logit, &dense.logit.coefficients);
    close_predictors(
        "gamma",
        &x,
        &compact.gamma,
        &dense.gamma.coefficients,
        AGREEMENT,
    );

    // Both paths refuse the separable logit by default.
    let refused = prepared
        .fit(
            case_weights(&weights),
            &warm,
            &two_part_options(),
            &mut PreparedScratch::default(),
        )
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(refused, Error::Separation { .. }));
    assert_eq!(
        two_part::fit(&weighted(&x, &y, &weights), &two_part_options(), None).unwrap_err(),
        refused
    );
}

#[test]
fn a_ridge_two_part_fit_agrees_with_the_dense_fit() {
    let (x, y) = design(8_000, 0.3, 5);
    let warm = two_part_warm(&x, &y, &two_part_options());
    let options = TwoPartOptions {
        regularization: Regularization::Ridge {
            lambda: 1e-2,
            exclude_intercept: true,
        },
        ..two_part_options()
    };
    let prepared = PreparedTwoPartDesign::new(x.as_ref(), &y).expect("design");
    let weights = replicate_weights(8_000, 11);
    let dense = two_part::fit(&weighted(&x, &y, &weights), &options, Some(&warm)).expect("dense");
    let compact = prepared
        .fit(
            case_weights(&weights),
            &warm,
            &options,
            &mut PreparedScratch::default(),
        )
        .expect("compact");
    close_predictors(
        "logit",
        &x,
        &compact.logit,
        &dense.logit.coefficients,
        LOGIT_AGREEMENT,
    );
    close_predictors(
        "gamma",
        &x,
        &compact.gamma,
        &dense.gamma.coefficients,
        AGREEMENT,
    );
}

#[test]
fn predictions_average_as_the_models_do() {
    let (x, y) = design(9_000, 0.25, 6);
    let gamma_y: Vec<f64> = y.iter().map(|v| v.max(1.0)).collect();
    let beta = gamma_warm(&x, &gamma_y);
    let gamma_model = tweedie::fit(
        &data(&x, &gamma_y),
        2.0,
        &gamma_options(),
        Some(&beta),
        None,
    )
    .expect("gamma");
    let two_part_model = two_part::fit(&data(&x, &y), &two_part_options(), None).expect("two-part");

    let prediction = PreparedPredictionDesign::new(x.as_ref()).expect("design");
    assert_eq!(prediction.nrows(), 9_000);
    let mut stream = Stream(13);
    let weights: Vec<f64> = (0..9_000)
        .map(|i| {
            if i % 3 == 0 {
                0.0
            } else {
                stream.exponential()
            }
        })
        .collect();
    close(
        "gamma",
        prediction.mean_gamma(&gamma_model.coefficients, &weights),
        gamma_model
            .predict_mean(x.as_ref(), Some(&weights))
            .expect("mean"),
    );
    close(
        "two-part",
        prediction.mean_two_part(
            &two_part_model.logit.coefficients,
            &two_part_model.gamma.coefficients,
            &weights,
        ),
        two_part_model
            .predict_mean(x.as_ref(), Some(&weights))
            .expect("mean"),
    );
}

#[test]
fn a_prediction_design_may_hold_a_constant_column() {
    let (mut x, y) = design(2_000, 0.0, 7);
    let model = tweedie::fit(&data(&x, &y), 2.0, &gamma_options(), None, None).expect("fit");
    for i in 0..2_000 {
        x[(i, 5)] = 1.0;
    }
    assert!(PreparedGammaDesign::new(x.as_ref(), &y).is_none());
    let weights = vec![1.0; 2_000];
    close(
        "mean",
        PreparedPredictionDesign::new(x.as_ref())
            .expect("design")
            .mean_gamma(&model.coefficients, &weights),
        model
            .predict_mean(x.as_ref(), Some(&weights))
            .expect("mean"),
    );
}

#[test]
fn a_design_the_compact_form_cannot_represent_is_refused() {
    let (x, y) = design(1_000, 0.2, 8);
    let y_gamma: Vec<f64> = y.iter().map(|v| v.max(1.0)).collect();
    let y_two_part = &y[..];
    let gamma = |x: &Mat<f64>| PreparedGammaDesign::new(x.as_ref(), &y_gamma).is_some();
    let two_part = |x: &Mat<f64>| PreparedTwoPartDesign::new(x.as_ref(), y_two_part).is_some();
    let prediction = |x: &Mat<f64>| PreparedPredictionDesign::new(x.as_ref()).is_some();
    assert!(gamma(&x) && two_part(&x) && prediction(&x));

    // A constant column stays uncentred in the dense fit, and the algebra cannot.
    let mut constant = x.clone();
    for i in 0..1_000 {
        constant[(i, 3)] = 7.0;
    }
    assert!(!gamma(&constant) && !two_part(&constant) && prediction(&constant));

    // Column 0 is not the intercept.
    let mut no_intercept = x.clone();
    no_intercept[(10, 0)] = 2.0;
    assert!(!gamma(&no_intercept) && !two_part(&no_intercept) && !prediction(&no_intercept));

    // A column's number has to fit a byte.
    let wide = |p: usize| {
        Mat::from_fn(600, p, |i, j| {
            if j == 0 {
                1.0
            } else {
                f64::from(((i * 2_654_435_761 + j * 40_503) >> 7) % 5 == 0)
            }
        })
    };
    let y_wide = vec![1.0; 600];
    assert!(PreparedGammaDesign::new(wide(255).as_ref(), &y_wide).is_some());
    assert!(PreparedGammaDesign::new(wide(256).as_ref(), &y_wide).is_none());
    assert!(PreparedPredictionDesign::new(wide(255).as_ref()).is_some());
    assert!(PreparedPredictionDesign::new(wide(256).as_ref()).is_none());

    // The gamma deviance needs positive outcomes.
    assert!(PreparedGammaDesign::new(x.as_ref(), y_two_part).is_none());
    // A two-part design needs a positive row, and no column constant among those rows.
    let none_positive = vec![0.0; 1_000];
    assert!(PreparedTwoPartDesign::new(x.as_ref(), &none_positive).is_none());
    let mut constant_on_positive = x.clone();
    for i in 0..1_000 {
        constant_on_positive[(i, 3)] = if y_two_part[i] > 0.0 { 1.0 } else { 0.0 };
    }
    assert!(!two_part(&constant_on_positive));

    // Empty input and a length mismatch.
    let empty = Mat::<f64>::zeros(0, 3);
    assert!(PreparedPredictionDesign::new(empty.as_ref()).is_none());
    assert!(PreparedGammaDesign::new(x.as_ref(), &y_gamma[..10]).is_none());
    assert!(PreparedTwoPartDesign::new(x.as_ref(), &y_gamma[..10]).is_none());
}

#[test]
fn a_fit_refuses_what_it_cannot_answer_reliably() {
    let (x, y) = design(6_000, 0.0, 9);
    let warm = gamma_warm(&x, &y);
    let mut scratch = PreparedScratch::default();
    let prepared = PreparedGammaDesign::new(x.as_ref(), &y).expect("design");
    let good = vec![1.0; 6_000];
    let error = |result: Result<(), Error>| result.unwrap_err();

    // A weight of 0 takes its row out of the fit, as in the dense fit.
    let mut zero = good.clone();
    zero[3] = 0.0;
    let compact = prepared
        .fit(case_weights(&zero), &warm, &gamma_options(), &mut scratch)
        .expect("a zero weight");
    let dense = tweedie::fit(
        &weighted(&x, &y, &zero),
        2.0,
        &gamma_options(),
        Some(&warm),
        None,
    )
    .expect("dense");
    close_columns("beta", &compact.coefficients, &dense.coefficients);
    assert_eq!(
        error(
            prepared
                .fit(
                    case_weights(&good[1..]),
                    &warm,
                    &gamma_options(),
                    &mut scratch
                )
                .map(|_| ())
        ),
        Error::Input(InputError::Length {
            what: "weights",
            expected: 6_000,
            found: 5_999,
        })
    );
    assert_eq!(
        error(
            prepared
                .fit(
                    case_weights(&good),
                    &[0.0; 3],
                    &gamma_options(),
                    &mut scratch
                )
                .map(|_| ())
        ),
        Error::Input(InputError::Length {
            what: "warm",
            expected: 16,
            found: 3,
        })
    );

    // No iterations allowed: the fit cannot have converged.
    let no_iterations = TweedieOptions {
        max_iterations: 0,
        ..gamma_options()
    };
    assert_eq!(
        error(
            prepared
                .fit(case_weights(&good), &warm, &no_iterations, &mut scratch)
                .map(|_| ())
        ),
        Error::NotConverged { iterations: 0 }
    );

    // Two identical columns: the pivot of the second is rounding only.
    let mut duplicated = x.clone();
    for i in 0..6_000 {
        duplicated[(i, 9)] = duplicated[(i, 8)];
    }
    let warm_zero = {
        let mut beta = vec![0.0; 16];
        beta[0] = 9.0;
        beta
    };
    let prepared_duplicated = PreparedGammaDesign::new(duplicated.as_ref(), &y).expect("design");
    assert_eq!(
        error(
            prepared_duplicated
                .fit(
                    case_weights(&good),
                    &warm_zero,
                    &gamma_options(),
                    &mut scratch
                )
                .map(|_| ())
        ),
        Error::IllConditioned
    );

    // A dummy that holds a 1 on all but one row, whose weight then almost vanishes.
    let mut nearly_all = x.clone();
    for i in 0..6_000 {
        nearly_all[(i, 9)] = f64::from(i != 17);
    }
    let prepared_nearly_all = PreparedGammaDesign::new(nearly_all.as_ref(), &y).expect("design");
    let mut tiny = good.clone();
    tiny[17] = 1e-9;
    assert_eq!(
        error(
            prepared_nearly_all
                .fit(
                    case_weights(&tiny),
                    &warm_zero,
                    &gamma_options(),
                    &mut scratch
                )
                .map(|_| ())
        ),
        Error::IllConditioned
    );

    // An outcome far beyond the log-link bound.
    let hot: Vec<f64> = y.iter().map(|v| v * 1e20).collect();
    let prepared_hot = PreparedGammaDesign::new(x.as_ref(), &hot).expect("design");
    assert_eq!(
        error(
            prepared_hot
                .fit(
                    case_weights(&good),
                    &warm_zero,
                    &gamma_options(),
                    &mut scratch
                )
                .map(|_| ())
        ),
        Error::Saturated
    );
}

#[test]
fn a_two_part_fit_refuses_what_it_cannot_answer_reliably() {
    let (x, y) = design(6_000, 0.3, 10);
    let warm = two_part_warm(&x, &y, &two_part_options());
    let mut scratch = PreparedScratch::default();
    let prepared = PreparedTwoPartDesign::new(x.as_ref(), &y).expect("design");
    let good = vec![1.0; 6_000];
    let code = |options: TwoPartOptions,
                weights: &[f64],
                warm: &TwoPartWarmStart,
                scratch: &mut PreparedScratch| {
        prepared
            .fit(case_weights(weights), warm, &options, scratch)
            .map(|_| ())
            .unwrap_err()
    };

    assert_eq!(
        code(
            TwoPartOptions {
                regularization: Regularization::ElasticNet {
                    lambda: 1e-3,
                    alpha: 0.5,
                    exclude_intercept: true
                },
                ..two_part_options()
            },
            &good,
            &warm,
            &mut scratch
        ),
        Error::Unsupported {
            what: "an elastic net or a ridge on the intercept in a compact design"
        }
    );
    assert_eq!(
        code(
            TwoPartOptions {
                regularization: Regularization::Ridge {
                    lambda: 1e-3,
                    exclude_intercept: false
                },
                ..two_part_options()
            },
            &good,
            &warm,
            &mut scratch
        ),
        Error::Unsupported {
            what: "an elastic net or a ridge on the intercept in a compact design"
        }
    );
    // A weight that is not finite is refused when the weights are built.
    let mut bad = good.clone();
    bad[0] = f64::NAN;
    assert_eq!(
        CaseWeights::new(&bad),
        Err(InputError::NotFinite {
            what: "weights",
            row: 0,
            column: None
        })
    );
    assert_eq!(
        code(two_part_options(), &good[1..], &warm, &mut scratch),
        Error::Input(InputError::Length {
            what: "weights",
            expected: 6_000,
            found: 5_999,
        })
    );
    assert_eq!(
        code(
            two_part_options(),
            &good,
            &TwoPartWarmStart {
                logit: vec![0.0; 2],
                gamma: vec![0.0; 16],
            },
            &mut scratch
        ),
        Error::Input(InputError::Length {
            what: "warm",
            expected: 16,
            found: 2
        })
    );
    assert_eq!(
        code(
            TwoPartOptions {
                max_iterations: 0,
                ..two_part_options()
            },
            &good,
            &warm,
            &mut scratch
        ),
        Error::NotConverged { iterations: 0 }
    );
}

#[test]
fn a_compact_result_does_not_depend_on_what_the_scratch_held() {
    let (x, y) = design(6_000, 0.0, 11);
    let (x_two, y_two) = design(7_000, 0.3, 12);
    let warm = gamma_warm(&x, &y);
    let warm_two = two_part_warm(&x_two, &y_two, &two_part_options());
    let gamma = PreparedGammaDesign::new(x.as_ref(), &y).expect("gamma");
    let two_part = PreparedTwoPartDesign::new(x_two.as_ref(), &y_two).expect("two");
    let weights = replicate_weights(6_000, 300);
    let weights_two = replicate_weights(7_000, 301);

    let fresh_gamma = gamma
        .fit(
            case_weights(&weights),
            &warm,
            &gamma_options(),
            &mut PreparedScratch::default(),
        )
        .expect("gamma");
    let fresh_two = two_part
        .fit(
            case_weights(&weights_two),
            &warm_two,
            &two_part_options(),
            &mut PreparedScratch::default(),
        )
        .expect("two-part");

    let mut shared = PreparedScratch::default();
    for _ in 0..2 {
        let used_two = two_part
            .fit(
                case_weights(&weights_two),
                &warm_two,
                &two_part_options(),
                &mut shared,
            )
            .expect("two-part");
        let used_gamma = gamma
            .fit(case_weights(&weights), &warm, &gamma_options(), &mut shared)
            .expect("gamma");
        assert_same_bits("gamma", &fresh_gamma.coefficients, &used_gamma.coefficients);
        assert_same_bits("logit", &fresh_two.logit, &used_two.logit);
        assert_same_bits("two-part gamma", &fresh_two.gamma, &used_two.gamma);
    }
}

fn assert_same_bits(label: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{label}");
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{label} [{i}]");
    }
}

fn assert_same_covariance(label: &str, a: Option<&Covariance>, b: Option<&Covariance>) {
    match (a, b) {
        (Some(a), Some(b)) => {
            let bits = |c: &Covariance| -> Vec<Option<u64>> {
                c.as_slice().iter().map(|v| v.map(f64::to_bits)).collect()
            };
            assert_eq!(bits(a), bits(b), "{label}");
        }
        (None, None) => {}
        _ => panic!("{label}: one fit has a covariance"),
    }
}

/// Every power on the path the scratch touches, against the fit that allocates, with one
/// scratch shared across fits of different shapes.
#[test]
fn a_tweedie_fit_with_a_scratch_is_bit_identical_to_one_without() {
    let (x, y) = design(7_000, 0.2, 13);
    let (x_small, y_small) = design(1_500, 0.2, 14);
    let positive: Vec<f64> = y.iter().map(|v| v.max(1.0)).collect();
    let weights = replicate_weights(7_000, 400);
    let weights_small = replicate_weights(1_500, 401);
    let relaxed = FitStrategy::Relaxed {
        fallback_lambda: 1e-3,
        max_retries: 3,
    };
    let mut scratch = TweedieScratch::default();
    for power in [1.0_f64, 1.2, 1.5, 1.8, 2.0] {
        let outcome = if (power - 2.0).abs() < 1e-12 {
            &positive
        } else {
            &y
        };
        for (label, options) in [
            ("default", gamma_options()),
            (
                "relaxed",
                TweedieOptions {
                    strategy: relaxed,
                    ..gamma_options()
                },
            ),
            (
                "ridge/robust",
                TweedieOptions {
                    l2_penalty: 1e-3,
                    variance: Variance::Sandwich,
                    ..gamma_options()
                },
            ),
        ] {
            let input = weighted(&x, outcome, &weights);
            let plain = tweedie::fit(&input, power, &options, None, None).expect("plain");
            let held =
                tweedie::fit(&input, power, &options, None, Some(&mut scratch)).expect("held");
            let name = format!("power {power} {label}");
            assert_same_bits(&name, &plain.coefficients, &held.coefficients);
            assert_eq!(
                plain.report.meta.iterations, held.report.meta.iterations,
                "{name}"
            );
            assert_same_covariance(&name, plain.covariance.as_ref(), held.covariance.as_ref());

            // Another shape reallocates; the next fit is unaffected by it.
            let outcome_small: Vec<f64> = y_small
                .iter()
                .map(|&v| {
                    if (power - 2.0).abs() < 1e-12 {
                        v.max(1.0)
                    } else {
                        v
                    }
                })
                .collect();
            let small = weighted(&x_small, &outcome_small, &weights_small);
            let plain_small = tweedie::fit(&small, power, &options, None, None).expect("plain");
            let held_small =
                tweedie::fit(&small, power, &options, None, Some(&mut scratch)).expect("held");
            assert_same_bits(&name, &plain_small.coefficients, &held_small.coefficients);
        }
    }
}

/// A design with no intercept is fitted uncentred, and leaves the scratch's centred
/// matrix untouched.
#[test]
fn a_scratch_serves_an_uncentred_fit_after_a_centred_one() {
    let (x, y) = design(3_000, 0.0, 15);
    let shifted = Mat::from_fn(3_000, x.ncols(), |i, j| {
        if j == 0 {
            1.0 + (i % 3) as f64 * 0.5
        } else {
            x[(i, j)]
        }
    });
    let mut scratch = TweedieScratch::default();
    tweedie::fit(
        &data(&x, &y),
        2.0,
        &gamma_options(),
        None,
        Some(&mut scratch),
    )
    .expect("centred");
    let uncentred = data(&shifted, &y);
    let plain = tweedie::fit(&uncentred, 2.0, &gamma_options(), None, None);
    let held = tweedie::fit(&uncentred, 2.0, &gamma_options(), None, Some(&mut scratch));
    match (plain, held) {
        (Ok(a), Ok(b)) => assert_same_bits("uncentred", &a.coefficients, &b.coefficients),
        (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
        (a, b) => panic!("one fit failed: {:?} {:?}", a.err(), b.err()),
    }
}
