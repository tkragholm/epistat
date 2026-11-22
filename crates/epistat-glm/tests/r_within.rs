//! The within-family estimator checked against `fixest::feols` (fixest 0.14 or
//! later, see the R script for the version used) and against `lm` with one
//! dummy variable per family.

use epistat_glm::{DfRule, Within, between_within};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    fam: Vec<usize>,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    coef: Vec<Option<f64>>,
    lm_coef: Vec<Option<f64>>,
    cluster_vcov: Vec<Option<f64>>,
    iid_vcov: Vec<Option<f64>>,
    r2_within: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/within.json"
    ))
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn worst(got: &[Option<f64>], want: &[Option<f64>], label: &str) -> f64 {
    assert_eq!(got.len(), want.len(), "{label}");
    let scale = want.iter().flatten().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let mut worst = 0.0_f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        match (g, w) {
            (None, None) => {}
            (Some(g), Some(w)) => worst = worst.max((g - w).abs() / scale),
            _ => panic!("{label} entry {i}: {g:?} against {w:?}"),
        }
    }
    worst
}

#[test]
fn within_estimates_match_fixest_and_dummies() {
    let mut overall = 0.0_f64;
    for case in &reference().cases {
        let x = matrix(&case.x);
        let clusters = Clusters::from_labels(&case.fam);
        let fit = Within::new()
            .fit(x.as_ref(), &case.y, &clusters, case.weights.as_deref())
            .unwrap();
        // The coefficients equal the dummy-variable regression to the
        // condition of the demeaned Gram matrix times epsilon, about 1e-12
        // here; 1e-10 allows for the conditioning of lm's own QR. The
        // covariances are sums of at most 600 terms: 1e-12 of the largest entry.
        let d = worst(&fit.coefficients, &case.coef, &case.name);
        assert!(d <= 1e-10, "{} coefficients {d:e}", case.name);
        // With a column constant within families lm keeps that column and
        // drops a dummy instead, so there is nothing to compare.
        if case.name != "constant_column" {
            let d = worst(&fit.coefficients, &case.lm_coef, &case.name);
            assert!(d <= 1e-10, "{} against lm {d:e}", case.name);
        }
        let d1 = worst(&fit.covariance, &case.iid_vcov, &case.name);
        let d2 = worst(&fit.robust_covariance, &case.cluster_vcov, &case.name);
        assert!(d1 <= 1e-12, "{} classical {d1:e}", case.name);
        assert!(d2 <= 1e-12, "{} clustered {d2:e}", case.name);
        overall = overall.max(d1).max(d2);
        assert!(
            (fit.r2_within.unwrap() - case.r2_within).abs() <= 1e-10,
            "{} r2",
            case.name
        );

        // Counts by hand.
        // Families are numbered from 1.
        let g = *case.fam.iter().max().unwrap();
        let mut size = vec![0usize; g + 1];
        for &f in &case.fam {
            size[f] += 1;
        }
        let singles = size.iter().filter(|&&s| s == 1).count();
        assert_eq!(fit.n_clusters, g, "{}", case.name);
        assert_eq!(fit.n_singleton_clusters, singles, "{}", case.name);
        let informative = (1..=g)
            .filter(|&f| {
                let rows: Vec<usize> = (0..case.fam.len()).filter(|&i| case.fam[i] == f).collect();
                rows.len() >= 2
                    && (0..3).any(|j| {
                        rows.iter()
                            .any(|&i| case.x[i][j].to_bits() != case.x[rows[0]][j].to_bits())
                    })
            })
            .count();
        assert_eq!(fit.n_informative_clusters, informative, "{}", case.name);
        if case.name == "constant_column" {
            assert_eq!(fit.dropped_columns, vec![2]);
        } else {
            assert!(fit.dropped_columns.is_empty(), "{}", case.name);
        }
    }
    eprintln!("worst covariance deviation over the largest entry: {overall:e}");
}

#[test]
fn the_unadjusted_rule_has_no_small_sample_factor() {
    let reference = reference();
    let case = &reference.cases[1];
    let x = matrix(&case.x);
    let clusters = Clusters::from_labels(&case.fam);
    let nested = Within::new()
        .fit(x.as_ref(), &case.y, &clusters, None)
        .unwrap();
    let plain = Within {
        df_rule: DfRule::Unadjusted,
    }
    .fit(x.as_ref(), &case.y, &clusters, None)
    .unwrap();
    let g = 150.0;
    let n = nested.n_rows as f64;
    let factor = g / (g - 1.0) * (n - 1.0) / (n - 3.0 - 1.0);
    for k in [0, 4, 8] {
        let ratio = nested.robust_covariance[k].unwrap() / plain.robust_covariance[k].unwrap();
        assert!((ratio - factor).abs() < 1e-12, "{ratio} {factor}");
    }
}

#[test]
fn pairs_equal_first_differences() {
    let reference = reference();
    let case = &reference.cases[0];
    let x = matrix(&case.x);
    let clusters = Clusters::from_labels(&case.fam);
    let fit = Within::new()
        .fit(x.as_ref(), &case.y, &clusters, None)
        .unwrap();
    // Regress the differences within each pair on the differences of the
    // covariates, through the normal equations.
    let pairs = case.y.len() / 2;
    let mut xtx = [[0.0; 3]; 3];
    let mut xty = [0.0; 3];
    for k in 0..pairs {
        let (a, b) = (2 * k, 2 * k + 1);
        assert_eq!(case.fam[a], case.fam[b]);
        let dx: Vec<f64> = (0..3).map(|j| case.x[a][j] - case.x[b][j]).collect();
        let dy = case.y[a] - case.y[b];
        for i in 0..3 {
            xty[i] += dx[i] * dy;
            for j in 0..3 {
                xtx[i][j] += dx[i] * dx[j];
            }
        }
    }
    let solution = epistat_linalg::solve(
        Mat::from_fn(3, 3, |i, j| xtx[i][j]).as_ref(),
        Mat::from_fn(3, 1, |i, _| xty[i]).as_ref(),
        &epistat_linalg::Attempt::SYMMETRIC,
    )
    .unwrap();
    for j in 0..3 {
        assert!((fit.coefficients[j].unwrap() - solution.x[(j, 0)]).abs() < 1e-10);
    }
}

#[test]
fn between_within_splits_each_column_into_family_mean_and_deviation() {
    let x = Mat::from_fn(5, 2, |i, j| {
        [[1.0, 4.0], [3.0, 4.0], [5.0, 4.0], [2.0, 9.0], [4.0, 9.0]][i][j]
    });
    let clusters = Clusters::from_labels(&[7, 7, 7, 9, 9]);
    let (means, deviations) = between_within(x.as_ref(), &clusters).unwrap();
    assert_eq!((means.nrows(), means.ncols()), (2, 2));
    assert!((means[(0, 0)] - 3.0).abs() < 1e-15 && (means[(1, 0)] - 3.0).abs() < 1e-15);
    assert!((means[(0, 1)] - 4.0).abs() < 1e-15);
    // A column that is constant within a family has deviation exactly 0.
    assert_eq!(deviations[(0, 1)].to_bits(), 0.0_f64.to_bits());
    assert!((deviations[(0, 0)] + 2.0).abs() < 1e-15 && (deviations[(4, 0)] - 1.0).abs() < 1e-15);
}

fn fit_bits() -> Vec<f64> {
    let reference = reference();
    let case = &reference.cases[3];
    let x = matrix(&case.x);
    let clusters = Clusters::from_labels(&case.fam);
    let fit = Within::new()
        .fit(x.as_ref(), &case.y, &clusters, case.weights.as_deref())
        .unwrap();
    let mut out: Vec<f64> = fit.coefficients.iter().map(|c| c.unwrap()).collect();
    out.extend(fit.robust_covariance.iter().map(|v| v.unwrap()));
    out.extend(fit.covariance.iter().map(|v| v.unwrap()));
    out.push(fit.r2_within.unwrap());
    out
}

#[test]
fn the_fit_does_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(fit_bits);
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn the_fit_has_its_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&fit_bits())
        .finish();
    assert_eq!(hash, 13_498_158_460_805_771_828, "recorded hash");
}
