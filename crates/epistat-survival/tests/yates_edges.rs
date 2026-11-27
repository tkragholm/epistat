//! The edges of `yates`: refused input, weights against repeated rows, a
//! population of set columns against the same rows as a matrix, and the tests
//! of a standardised risk.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "small counts"
)]

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::{self, CoxOptions};
use epistat_survival::standardise::yates::{
    InverseLink, YatesLevel, YatesOptions, YatesTest, wald_tests, yates,
};
use epistat_survival::standardise::{
    Counterfactual, ExposureLevel, StandardiseOptions, standardise_cox,
};

fn design() -> Mat<f64> {
    Mat::from_fn(6, 3, |i, j| match j {
        0 => 1.0,
        1 => (i % 2) as f64,
        _ => i as f64 * 0.5,
    })
}

#[test]
fn refused_input() {
    let x = design();
    let beta = [0.1, 0.2, 0.3];
    let var = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01];
    let level = YatesLevel {
        population: Counterfactual::Design(x.as_ref()),
        base: None,
        weights: None,
    };
    let opts = YatesOptions::default();
    assert!(yates(&beta, &var, &[], &opts).is_err(), "no levels");
    assert!(
        yates(&[0.1, f64::NAN, 0.3], &var, &[level, level], &opts).is_err(),
        "an aliased coefficient"
    );
    assert!(yates(&beta, &var[..8], &[level, level], &opts).is_err());
    let mut asym = var;
    asym[1] = 0.5;
    assert!(yates(&beta, &asym, &[level, level], &opts).is_err());
    let narrow = Mat::<f64>::zeros(6, 2);
    let bad = YatesLevel {
        population: Counterfactual::Design(narrow.as_ref()),
        ..level
    };
    assert!(yates(&beta, &var, &[level, bad], &opts).is_err());
    let negative = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0];
    let bad = YatesLevel {
        weights: Some(&negative),
        ..level
    };
    assert!(yates(&beta, &var, &[level, bad], &opts).is_err());
    let zeros = [0.0; 6];
    let bad = YatesLevel {
        weights: Some(&zeros),
        ..level
    };
    assert!(yates(&beta, &var, &[level, bad], &opts).is_err());
    let columns_without_base = YatesLevel {
        population: Counterfactual::Columns(&[(1, 1.0)]),
        base: None,
        weights: None,
    };
    assert!(yates(&beta, &var, &[level, columns_without_base], &opts).is_err());
    assert!(
        yates(&beta, &var, &[level], &opts).is_err(),
        "a test needs two levels"
    );
    let centre = [0.0; 2];
    assert!(
        yates(
            &beta,
            &var,
            &[level, level],
            &YatesOptions {
                centre: Some(&centre),
                ..opts
            }
        )
        .is_err()
    );
    // An inverse link that is not finite at a row.
    let zero_eta = Mat::<f64>::zeros(6, 3);
    let bad = YatesLevel {
        population: Counterfactual::Design(zero_eta.as_ref()),
        ..level
    };
    assert!(
        yates(
            &beta,
            &var,
            &[level, bad],
            &YatesOptions {
                link: InverseLink::Inverse,
                ..opts
            }
        )
        .is_err()
    );
}

fn by_columns<'a>(
    x: &'a Mat<f64>,
    set: &'a [(usize, f64)],
    weights: Option<&'a [f64]>,
) -> YatesLevel<'a> {
    YatesLevel {
        population: Counterfactual::Columns(set),
        base: Some(x.as_ref()),
        weights,
    }
}

#[test]
fn weights_columns_and_matrices_agree() {
    let x = design();
    let beta = [0.1, 0.2, 0.3];
    let var = [0.01, 0.002, 0.0, 0.002, 0.02, 0.001, 0.0, 0.001, 0.03];
    let weights = [2.0, 1.0, 1.0, 3.0, 1.0, 1.0];
    // The rows with the second column set to 1 and to 0.
    let set1 = [(1, 1.0)];
    let set0 = [(1, 0.0)];
    let by_columns = |set, w| by_columns(&x, set, w);
    let opts = YatesOptions {
        link: InverseLink::Exp,
        ..YatesOptions::default()
    };
    let weighted = yates(
        &beta,
        &var,
        &[
            by_columns(&set0, Some(&weights)),
            by_columns(&set1, Some(&weights)),
        ],
        &opts,
    )
    .unwrap();
    // The same rows repeated by their weights, as matrices.
    let rows: Vec<usize> = weights
        .iter()
        .enumerate()
        .flat_map(|(i, w)| std::iter::repeat_n(i, *w as usize))
        .collect();
    let m0 = Mat::from_fn(
        rows.len(),
        3,
        |i, j| if j == 1 { 0.0 } else { x[(rows[i], j)] },
    );
    let m1 = Mat::from_fn(
        rows.len(),
        3,
        |i, j| if j == 1 { 1.0 } else { x[(rows[i], j)] },
    );
    let repeated = yates(
        &beta,
        &var,
        &[
            YatesLevel {
                population: Counterfactual::Design(m0.as_ref()),
                base: None,
                weights: None,
            },
            YatesLevel {
                population: Counterfactual::Design(m1.as_ref()),
                base: None,
                weights: None,
            },
        ],
        &opts,
    )
    .unwrap();
    // Both are means of at most 8 values with a gradient of 3 terms: a few
    // `(n + p) eps` relative.
    let bound = 8.0 * (8.0 + 3.0 + 4.0) * f64::EPSILON;
    for l in 0..2 {
        assert!(
            (weighted.estimate[l] - repeated.estimate[l]).abs() <= bound * repeated.estimate[l]
        );
        assert!((weighted.se[l] - repeated.se[l]).abs() <= bound * repeated.se[l]);
    }
    for (a, b) in weighted
        .covariance
        .as_slice()
        .iter()
        .zip(repeated.covariance.as_slice())
    {
        let (a, b) = (a.unwrap(), b.unwrap());
        assert!((a - b).abs() <= bound * b.abs());
    }
}

#[test]
fn global_test_of_two_levels_is_the_squared_contrast_over_its_variance() {
    let estimate = [1.0, 3.0];
    let cov = [0.5, 0.1, 0.1, 0.4];
    let tests = wald_tests(&estimate, &cov, YatesTest::Global).unwrap();
    assert_eq!(tests.len(), 1);
    assert_eq!(tests[0].label, "global");
    assert_eq!(tests[0].df, 1);
    let want = 4.0 / (0.5 + 0.4 - 0.2);
    assert!((tests[0].chi_square - want).abs() <= 8.0 * f64::EPSILON * want);
    // Pairwise with two levels is one test labelled as the global one, and
    // three levels give three pairs and three tests against the mean.
    let pair = wald_tests(&estimate, &cov, YatesTest::Pairwise).unwrap();
    assert_eq!(pair[0].label, "global");
    assert_eq!(pair[0].chi_square.to_bits(), tests[0].chi_square.to_bits());
    let e3 = [1.0, 2.0, 4.0];
    let c3 = [0.3, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3];
    let labels: Vec<String> = wald_tests(&e3, &c3, YatesTest::Pairwise)
        .unwrap()
        .into_iter()
        .map(|t| t.label)
        .collect();
    assert_eq!(labels, ["1 vs 2", "1 vs 3", "2 vs 3"]);
    let mean = wald_tests(&e3, &c3, YatesTest::VersusMean).unwrap();
    assert_eq!(mean[2].label, "3 vs mean");
    // 3 vs mean: the contrast is 4 - 7/3 = 5/3 and its variance 0.3 (4/9 + 1/9 + 1/9).
    let want = (5.0f64 / 3.0).powi(2) / (0.3 * (4.0 / 9.0 + 1.0 / 9.0 + 1.0 / 9.0));
    assert!((mean[2].chi_square - want).abs() <= 8.0 * f64::EPSILON * want);
    assert!(wald_tests(&[1.0], &[1.0], YatesTest::Global).is_err());
    // A covariance of rank 0 has no degrees of freedom and no p-value.
    let flat = wald_tests(&estimate, &[0.0; 4], YatesTest::Global).unwrap();
    assert_eq!(flat[0].df, 0);
    assert_eq!(flat[0].p_value, None);
}

#[test]
fn standardised_risks_are_tested_with_their_covariance() {
    let n = 200;
    let mut stop = Vec::new();
    let mut status = Vec::new();
    let mut cols = Vec::new();
    let mut rng = epistat_rng::Stream::seeded(5);
    for _ in 0..n {
        let x = rng.normal();
        let a = rng.below(3);
        let t = rng.exp1() / (0.1 * (0.4 * x + 0.3 * a as f64).exp());
        let c = 2.0 + 10.0 * rng.uniform();
        stop.push(t.min(c));
        status.push(i32::from(t <= c));
        cols.push([f64::from(u8::from(a == 1)), f64::from(u8::from(a == 2)), x]);
    }
    let design = Mat::from_fn(n, 3, |i, j| cols[i][j]);
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let options = CoxOptions::default();
    let fit = cox::fit(&data, design.as_ref(), &options).unwrap();
    let sets = [
        vec![(0, 0.0), (1, 0.0)],
        vec![(0, 1.0), (1, 0.0)],
        vec![(0, 0.0), (1, 1.0)],
    ];
    let levels: Vec<ExposureLevel<'_>> = sets
        .iter()
        .map(|s| ExposureLevel::single(Counterfactual::Columns(s)))
        .collect();
    let times = [2.0, 5.0];
    let with = standardise_cox(
        &data,
        design.as_ref(),
        &fit,
        &options,
        &levels,
        &times,
        &StandardiseOptions {
            se: true,
            ..StandardiseOptions::default()
        },
    )
    .unwrap();
    let global = with.tests(1, YatesTest::Global).unwrap();
    assert_eq!(global[0].df, 2);
    assert!(global[0].chi_square.is_finite() && global[0].chi_square >= 0.0);
    assert_eq!(with.tests(1, YatesTest::Pairwise).unwrap().len(), 3);
    assert!(with.tests(5, YatesTest::Global).is_err());
    let without = standardise_cox(
        &data,
        design.as_ref(),
        &fit,
        &options,
        &levels,
        &times,
        &StandardiseOptions::default(),
    )
    .unwrap();
    assert!(without.tests(0, YatesTest::Global).is_err());
}
