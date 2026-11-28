//! A view whose weights come from `with_weights` gives the same bits as a view built with
//! those weights, `with_weights` refuses the weights `build` refuses, and the mean-only
//! predictions give the same bits as the mean of the full predictions.
//!
//! These tests compare two paths of the same binary, so unlike `fit_golden` they hold on
//! every target.

use epistat_core::{CaseWeights, Covariance, InputError};
use epistat_linalg::Mat;
use epistat_semicontinuous::lognormal::LogNormalOptions;
use epistat_semicontinuous::tweedie::{TweedieModel, TweedieOptions};
use epistat_semicontinuous::two_part::{TwoPartModel, TwoPartOptions, TwoPartWarmStart};
use epistat_semicontinuous::{CostData, Error, Model, Variance, lognormal, tweedie, two_part};

const N_COLS: usize = 8;

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
}

/// Intercept, five dummies and two continuous columns, with about 40% zeros.
fn sample(n: usize, seed: u64) -> (Mat<f64>, Vec<f64>) {
    let mut stream = Stream(seed);
    let mut x = Mat::<f64>::zeros(n, N_COLS);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        for j in 1..6 {
            x[(i, j)] = f64::from(stream.uniform() < 0.1 + 0.07 * j as f64);
        }
        for j in 6..N_COLS {
            x[(i, j)] = 2.0f64.mul_add(stream.uniform(), -1.0);
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let eta = 0.4f64.mul_add(x[(i, 1)], 0.2 * x[(i, 2)] - 0.1 * x[(i, 6)]);
        let is_zero = stream.uniform() < 0.4 - 0.1 * x[(i, 3)];
        let noise = -stream.uniform().max(1e-12).ln() - stream.uniform().max(1e-12).ln();
        if !is_zero {
            y[i] = (6.0f64 + eta).exp() / 2.0 * noise + 1.0;
        }
    }
    (x, y)
}

/// Weights of a fractional-weight replicate: positive and different for each replicate.
fn replicate_weights(n: usize, replicate: u64) -> Vec<f64> {
    let mut stream = Stream(7_000 + replicate);
    (0..n).map(|_| 0.2 + 2.0 * stream.uniform()).collect()
}

/// Checked weights of a column known to be valid.
fn case_weights(weights: &[f64]) -> CaseWeights<'_> {
    CaseWeights::new(weights).expect("weights")
}

fn same(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "entry {i}");
    }
}

fn same_opt(a: &[Option<f64>], b: &[Option<f64>]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits), "entry {i}");
    }
}

fn same_cov(a: Option<&Covariance>, b: Option<&Covariance>) {
    match (a, b) {
        (Some(a), Some(b)) => same_opt(a.as_slice(), b.as_slice()),
        (None, None) => {}
        _ => panic!("one side has the covariance and the other does not"),
    }
}

fn same_two_part(a: &TwoPartModel, b: &TwoPartModel) {
    for (a, b) in [(&a.logit, &b.logit), (&a.gamma, &b.gamma)] {
        same(&a.coefficients, &b.coefficients);
        same_opt(&a.se, &b.se);
        same_cov(a.covariance.as_ref(), b.covariance.as_ref());
        assert_eq!(a.meta.iterations, b.meta.iterations);
        assert_eq!(
            a.dispersion.map(f64::to_bits),
            b.dispersion.map(f64::to_bits)
        );
    }
    assert_eq!(a.report.variance, b.report.variance);
    assert_eq!(a.report.cluster_count, b.report.cluster_count);
}

fn same_tweedie(a: &TweedieModel, b: &TweedieModel) {
    same(&a.coefficients, &b.coefficients);
    same_opt(&a.se, &b.se);
    same_cov(a.covariance.as_ref(), b.covariance.as_ref());
    assert_eq!(a.report.meta.iterations, b.report.meta.iterations);
    assert_eq!(a.report.variance, b.report.variance);
    assert_eq!(a.report.cluster_count, b.report.cluster_count);
}

/// Clusters of three consecutive rows.
fn clusters(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| u32::try_from(i / 3).expect("small"))
        .collect()
}

/// The three views of one replicate: built with its weights, and derived with
/// `with_weights` from a view without weights and from a view with other weights.
fn views<'a>(
    x: &'a Mat<f64>,
    y: &'a [f64],
    weights: &'a [f64],
    other: &'a [f64],
    clusters: Option<&'a [u32]>,
) -> [CostData<'a>; 3] {
    let builder = |weights: Option<&'a [f64]>| {
        let mut builder = CostData::builder(x.as_ref(), y);
        if let Some(weights) = weights {
            builder = builder.weights(case_weights(weights));
        }
        if let Some(clusters) = clusters {
            builder = builder.clusters(clusters);
        }
        builder.build().expect("data")
    };
    let built = builder(Some(weights));
    let derived = builder(None)
        .with_weights(case_weights(weights))
        .expect("weights");
    let reweighted = builder(Some(other))
        .with_weights(case_weights(weights))
        .expect("weights");
    [built, derived, reweighted]
}

#[test]
fn two_part_reweighted_view_matches_built_view() {
    let n = 6_000;
    let (x, y) = sample(n, 11);
    let cluster_ids = clusters(n);
    let options = [
        Variance::ModelBased,
        Variance::Sandwich,
        Variance::None,
    ]
    .map(|variance| TwoPartOptions {
        variance,
        ..TwoPartOptions::default()
    });
    let full_weights = replicate_weights(n, 0);
    let full = two_part::fit(
        &CostData::builder(x.as_ref(), &y)
            .weights(case_weights(&full_weights))
            .build()
            .expect("data"),
        &TwoPartOptions::default(),
        None,
    )
    .expect("full-sample fit");
    let warm = TwoPartWarmStart::from(&full);

    for options in options {
        for replicate in 1..=3_u64 {
            let weights = replicate_weights(n, replicate);
            for with_clusters in [false, true] {
                let clusters = with_clusters.then_some(cluster_ids.as_slice());
                let [built, derived, reweighted] = views(&x, &y, &weights, &full_weights, clusters);
                for warm in [None, Some(&warm)] {
                    let a = two_part::fit(&built, &options, warm).expect("built");
                    let b = two_part::fit(&derived, &options, warm).expect("derived");
                    let c = two_part::fit(&reweighted, &options, warm).expect("reweighted");
                    same_two_part(&a, &b);
                    same_two_part(&a, &c);
                }
            }
        }
    }
}

#[test]
fn two_part_view_without_weights_matches_unweighted_view() {
    let n = 5_000;
    let (x, y) = sample(n, 12);
    let weights = replicate_weights(n, 1);
    let plain = CostData::builder(x.as_ref(), &y).build().expect("data");
    let weighted = CostData::builder(x.as_ref(), &y)
        .weights(case_weights(&weights))
        .build()
        .expect("data");
    let options = TwoPartOptions::default();
    let a = two_part::fit(&plain, &options, None).expect("unweighted");
    let b = two_part::fit(&weighted.without_weights(), &options, None).expect("without weights");
    same_two_part(&a, &b);
}

#[test]
fn tweedie_reweighted_view_matches_built_view() {
    let n = 6_000;
    let (x, y) = sample(n, 13);
    let cluster_ids = clusters(n);
    let full_weights = replicate_weights(n, 0);
    let full = tweedie::fit(
        &CostData::builder(x.as_ref(), &y)
            .weights(case_weights(&full_weights))
            .build()
            .expect("data"),
        2.0,
        &TweedieOptions::default(),
        None,
        None,
    )
    .expect("full-sample fit");
    for power in [1.5, 2.0] {
        for options in [
            TweedieOptions::default(),
            TweedieOptions {
                variance: Variance::Sandwich,
                ..TweedieOptions::default()
            },
        ] {
            for replicate in 1..=3_u64 {
                let weights = replicate_weights(n, replicate);
                for with_clusters in [false, true] {
                    let clusters = with_clusters.then_some(cluster_ids.as_slice());
                    let [built, derived, reweighted] =
                        views(&x, &y, &weights, &full_weights, clusters);
                    for warm in [None, Some(full.coefficients.as_slice())] {
                        let a = tweedie::fit(&built, power, &options, warm, None).expect("built");
                        let b =
                            tweedie::fit(&derived, power, &options, warm, None).expect("derived");
                        let c = tweedie::fit(&reweighted, power, &options, warm, None)
                            .expect("reweighted");
                        same_tweedie(&a, &b);
                        same_tweedie(&a, &c);
                    }
                }
            }
        }
    }
}

/// A comparator subset gathered once and fitted with the weights of each replicate gives
/// the fit of the subset view rebuilt for each replicate.
#[test]
fn row_subset_gathered_once_matches_rebuilt_subset() {
    let n = 5_000;
    let (x, y) = sample(n, 14);
    let rows: Vec<usize> = (0..n).filter(|&i| x[(i, 6)] < 0.5).collect();
    let x_subset = Mat::from_fn(rows.len(), N_COLS, |i, j| x[(rows[i], j)]);
    let y_subset: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
    let design = CostData::builder(x_subset.as_ref(), &y_subset)
        .build()
        .expect("data");
    for replicate in 1..=3_u64 {
        let all = replicate_weights(n, replicate);
        let weights: Vec<f64> = rows.iter().map(|&i| all[i]).collect();
        let rebuilt = CostData::builder(x_subset.as_ref(), &y_subset)
            .weights(case_weights(&weights))
            .build()
            .expect("data");
        let a = two_part::fit(&rebuilt, &TwoPartOptions::default(), None).expect("rebuilt");
        let b = two_part::fit(
            &design
                .with_weights(case_weights(&weights))
                .expect("weights"),
            &TwoPartOptions::default(),
            None,
        )
        .expect("reweighted");
        same_two_part(&a, &b);
    }
}

#[test]
fn with_weights_checks_the_weights_and_build_the_clusters() {
    let n = 200;
    let (x, y) = sample(n, 15);
    let data = CostData::builder(x.as_ref(), &y).build().expect("data");
    let mut weights = replicate_weights(n, 1);
    // A weight of 0 takes its row out of the fit and is accepted.
    weights[3] = 0.0;
    let zero = data
        .with_weights(case_weights(&weights))
        .expect("a zero weight");
    assert_eq!(zero.weights().map(|w| w.n_positive()), Some(n - 1));
    // The weights refuse a negative or non-finite value when they are built.
    weights[3] = -1.0;
    assert!(matches!(
        CaseWeights::new(&weights),
        Err(InputError::Domain {
            what: "weights",
            row: Some(3),
            ..
        })
    ));
    weights[3] = f64::NAN;
    assert!(matches!(
        CaseWeights::new(&weights),
        Err(InputError::NotFinite {
            what: "weights",
            row: 3,
            column: None
        })
    ));
    // The view refuses weights of another length.
    weights[3] = 1.0;
    assert!(matches!(
        data.with_weights(case_weights(&weights[..n - 1])),
        Err(Error::Input(InputError::Length {
            what: "weights",
            ..
        }))
    ));
    let short = vec![1_u32; n - 1];
    assert!(matches!(
        CostData::builder(x.as_ref(), &y).clusters(&short).build(),
        Err(Error::Input(InputError::Length {
            what: "clusters",
            ..
        }))
    ));
}

#[test]
fn build_refuses_each_kind_of_bad_column() {
    let (x, y) = sample(50, 16);
    let mut bad_y = y.clone();
    bad_y[2] = -1.0;
    let mut nan_x = x.clone();
    nan_x[(4, 3)] = f64::NAN;
    let weights: Vec<f64> = (0..50).map(|i| if i == 7 { 0.0 } else { 1.0 }).collect();
    let long: Vec<f64> = vec![1.0; 51];
    let short = vec![1_u32; 49];
    let cases = [
        CostData::builder(x.as_ref(), &y).build(),
        CostData::builder(x.as_ref(), &bad_y).build(),
        CostData::builder(nan_x.as_ref(), &y).build(),
        CostData::builder(x.as_ref(), &y)
            .weights(case_weights(&weights))
            .build(),
        CostData::builder(x.as_ref(), &y)
            .weights(case_weights(&long))
            .build(),
        CostData::builder(x.as_ref(), &y).clusters(&short).build(),
    ];
    let expected: [Result<(), InputError>; 6] = [
        Ok(()),
        Err(InputError::Domain {
            what: "y",
            value: -1.0,
            domain: "[0, inf)",
            row: None,
        }),
        Err(InputError::NotFinite {
            what: "x",
            row: 4,
            column: None,
        }),
        // A weight of 0 is allowed: its row leaves the fit.
        Ok(()),
        Err(InputError::Length {
            what: "weights",
            expected: 50,
            found: 51,
        }),
        Err(InputError::Length {
            what: "clusters",
            expected: 50,
            found: 49,
        }),
    ];
    for (case, expected) in cases.into_iter().zip(expected) {
        let found = match case {
            Ok(_) => Ok(()),
            Err(Error::Input(error)) => Err(error),
            Err(other) => panic!("expected an input error, got {other:?}"),
        };
        assert_eq!(found, expected);
    }
}

fn weights_with_zeros(n: usize) -> Vec<f64> {
    let mut stream = Stream(99);
    (0..n)
        .map(|_| {
            let u = stream.uniform();
            if u < 0.3 { 0.0 } else { 3.0 * u }
        })
        .collect()
}

/// The mean as the downstream takes it: a plain sum over the rows, or the sum of the
/// weighted rows with the nonzero weights, in row order.
fn reference_mean(values: &[f64], weights: Option<&[f64]>) -> f64 {
    match weights {
        None => {
            let sum: f64 = values.iter().sum();
            sum / values.len() as f64
        }
        Some(weights) => {
            let (sum, total) = (0..values.len())
                .filter(|&i| weights[i] != 0.0)
                .fold((0.0, 0.0), |(sum, total), i| {
                    (sum + weights[i] * values[i], total + weights[i])
                });
            sum / total
        }
    }
}

#[test]
fn predict_mean_matches_the_mean_of_the_full_prediction() {
    let (x_fit, y_fit) = sample(3_000, 21);
    let data = CostData::builder(x_fit.as_ref(), &y_fit)
        .build()
        .expect("data");
    let two_part = two_part::fit(&data, &TwoPartOptions::default(), None).expect("fit");
    let tweedie = tweedie::fit(&data, 1.5, &TweedieOptions::default(), None, None).expect("fit");
    let lognormal = lognormal::fit(&data, &LogNormalOptions::default()).expect("fit");

    // Sizes around and across the block of 4096 rows the linear predictor is formed in.
    for n in [1, 2, 7, 100, 4_095, 4_096, 4_097, 8_192, 8_193, 25_000] {
        let (x, _) = sample(n, 1_000 + n as u64);
        let x = x.as_ref();
        let weights = weights_with_zeros(n);
        for weights in [None, Some(weights.as_slice())] {
            assert_eq!(
                two_part.predict_mean(x, weights).expect("mean").to_bits(),
                reference_mean(
                    &two_part.predict(x).expect("prediction").mean,
                    weights
                )
                .to_bits(),
                "two-part, n = {n}"
            );
            assert_eq!(
                tweedie.predict_mean(x, weights).expect("mean").to_bits(),
                reference_mean(&tweedie.predict(x).expect("prediction").mean, weights).to_bits(),
                "tweedie, n = {n}"
            );
            assert_eq!(
                lognormal.predict_mean(x, weights).expect("mean").to_bits(),
                reference_mean(
                    &lognormal.predict(x).expect("prediction").mean_positive,
                    weights
                )
                .to_bits(),
                "log-normal, n = {n}"
            );
        }
    }
}

#[test]
fn predict_mean_rejects_weights_of_the_wrong_length() {
    let (x, y) = sample(300, 22);
    let data = CostData::builder(x.as_ref(), &y).build().expect("data");
    let model = tweedie::fit(&data, 2.0, &TweedieOptions::default(), None, None).expect("fit");
    assert!(matches!(
        model.predict_mean(x.as_ref(), Some(&[1.0; 5])),
        Err(Error::Input(InputError::Length {
            what: "weights",
            expected: 300,
            found: 5
        }))
    ));
}
