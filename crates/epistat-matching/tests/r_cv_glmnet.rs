//! The cross-validated elastic net checked against R's `cv.glmnet`.
//!
//! The reference is `tests/data/cv_glmnet.json`, written by
//! `tests/data/cv_glmnet.R`. `cv.glmnet` draws its folds with R's generator,
//! which cannot be reproduced, so the folds come from the Rust fit. The test
//! `write_folds` writes the fold of every row, from `epistat-rng` and a seed,
//! to `tests/data/cv_folds.json`, the R script gives them to `cv.glmnet` as
//! `foldid`, and `folds_are_the_ones_the_reference_used` checks that the Rust
//! fit still draws the same folds. Both sides then fit the same models on the
//! same rows, and the test compares the `lambda` path, the cross-validated
//! deviance and its standard error at every `lambda`, the two chosen
//! penalties and the coefficients at `lambda.1se`.
//!
//! The penalised columns are standardised, as `glmnet` does by default and
//! MatchIt leaves it, except in the ridge case, which sets
//! `standardize = FALSE`.

#[path = "common/r_reference.rs"]
mod r_reference;

use epistat_matching::prelude::*;
use epistat_testkit::load_json;
use r_reference::{Versions, covariate_unit, split_by_group, unit_id as id};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Folds {
    cases: Vec<FoldCase>,
}

#[derive(Serialize, Deserialize)]
struct FoldCase {
    name: String,
    n: usize,
    alpha: f64,
    nfolds: usize,
    seed: u64,
    weighted: bool,
    standardize: bool,
    foldid: Vec<usize>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    alpha: f64,
    nfolds: usize,
    standardize: bool,
    seed: u64,
    x: Vec<Vec<f64>>,
    covariate_names: Vec<String>,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    foldid: Vec<usize>,
    lambda: Vec<f64>,
    cvm: Vec<f64>,
    cvsd: Vec<f64>,
    index_min: usize,
    index_1se: usize,
    coefficients_1se: Vec<f64>,
}

/// The cases: name, rows, alpha, folds, seed, weighted, standardised.
const CASES: [(&str, usize, f64, usize, u64, bool, bool); 4] = [
    ("lasso_unweighted", 300, 1.0, 10, 1, false, true),
    ("elasticnet_weighted", 400, 0.5, 5, 2, true, true),
    ("ridge_raw", 250, 0.0, 10, 3, false, false),
    ("lalonde_elasticnet", 614, 0.5, 10, 4, false, true),
];

fn folds_path() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/cv_folds.json").to_string()
}

fn records(case: &Case) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    split_by_group(case.y.iter().enumerate().map(|(idx, label)| {
        let covariates = case
            .covariate_names
            .iter()
            .zip(&case.x[idx])
            .map(|(name, value)| (name.clone(), CovariateValue::Numeric(*value)));
        (*label > 0.5, covariate_unit(id(idx), covariates))
    }))
}

fn fit_case(case: &Case, rule: LambdaRule) -> PropensityDistancePreparation {
    let (anchors, candidates) = records(case);
    // The reference puts the treated rows first, so the design order of the
    // fit is the order of the data.
    assert!(
        case.y.windows(2).all(|pair| pair[0] >= pair[1]),
        "{}: treated rows first",
        case.name
    );
    let config = PropensityScoreConfig {
        estimator: PropensityEstimator::ElasticNetCv(
            ElasticNetCvConfig::builder()
                .seed(case.seed)
                .alpha(case.alpha)
                .nfolds(case.nfolds)
                .standardize(case.standardize)
                .rule(rule)
                .build(),
        ),
        sampling_weights: case.weights.as_ref().map(|w| {
            w.iter()
                .enumerate()
                .map(|(idx, value)| (id(idx), *value))
                .collect()
        }),
        ..PropensityScoreConfig::default()
    };
    prepare_propensity_distance_config(&anchors, &candidates, &config)
        .expect("cross-validation succeeds")
}

/// Writes the folds that `cv_glmnet.R` passes to `cv.glmnet`.
#[test]
#[ignore = "writes tests/data/cv_folds.json for cv_glmnet.R"]
fn write_folds() {
    let cases = CASES
        .iter()
        .map(
            |&(name, n, alpha, nfolds, seed, weighted, standardize)| FoldCase {
                name: name.to_string(),
                n,
                alpha,
                nfolds,
                seed,
                weighted,
                standardize,
                foldid: cross_validation_folds(n, nfolds, seed).unwrap(),
            },
        )
        .collect();
    let json = serde_json::to_string(&Folds { cases }).expect("serialises");
    std::fs::write(folds_path(), json).expect("writes");
}

/// The folds in `cv_folds.json`, which `cv.glmnet` was given, are the ones the
/// Rust fit draws.
#[test]
fn folds_are_the_ones_the_reference_used() {
    let folds: Folds = load_json(folds_path());
    assert_eq!(folds.cases.len(), CASES.len());
    for (case, &(name, n, _, nfolds, seed, _, _)) in folds.cases.iter().zip(&CASES) {
        assert_eq!(case.name, name);
        assert_eq!(
            case.foldid,
            cross_validation_folds(n, nfolds, seed).unwrap(),
            "{name}"
        );
        // The folds are balanced: sizes differ by at most one.
        let mut sizes = vec![0usize; nfolds];
        for fold in &case.foldid {
            sizes[*fold] += 1;
        }
        let (low, high) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
        assert!(high - low <= 1, "{name}");
    }
}

/// The path of penalties, the cross-validated deviance and its standard error
/// at every penalty, the two choices and the coefficients at `lambda.1se`
/// agree with `cv.glmnet` given the same folds.
///
/// R's `cv.glmnet` is given no `lambda`, as in MatchIt, so each fold has a path
/// of its own and the folds are lined up with the master path by interpolation;
/// the Rust fit does the same. The path is computed by a formula, so it agrees
/// to rounding: the tolerance on its relative difference is 1e-12, a few
/// hundred times the observed 1e-14. The deviance, its standard error and the
/// coefficients come from fits whose coordinate descent stops at a threshold,
/// and the reference's was `thresh = 1e-20`. Running the reference at
/// `thresh = 1e-20` and `1e-22` moves the deviance by 8e-13, and the observed
/// differences from the Rust fit are 1e-11 at most, so the tolerance is 1e-9.
#[test]
fn cross_validation_matches_cv_glmnet() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cv_glmnet.json"
    ));
    println!(
        "{} / glmnet {}",
        reference.versions["r"], reference.versions["glmnet"]
    );
    let folds: Folds = load_json(folds_path());
    for case in &reference.cases {
        let given = folds
            .cases
            .iter()
            .find(|c| c.name == case.name)
            .expect("folds exist");
        assert_eq!(case.foldid, given.foldid, "{}", case.name);
        let prepared = fit_case(case, LambdaRule::OneStandardError);
        let cv = prepared
            .estimate
            .cross_validation
            .as_ref()
            .expect("cross-validation ran");
        assert_eq!(cv.fold_ids, case.foldid, "{}", case.name);
        assert_eq!(cv.unconverged_fits, 0, "{}", case.name);

        let worst = |ours: &[f64], theirs: &[f64], relative: bool| {
            assert_eq!(ours.len(), theirs.len(), "{}", case.name);
            ours.iter()
                .zip(theirs)
                .map(|(a, b)| {
                    if relative {
                        ((a - b) / b).abs()
                    } else {
                        (a - b).abs()
                    }
                })
                .fold(0.0_f64, f64::max)
        };
        let lambda_dev = worst(&cv.lambda, &case.lambda, true);
        let cvm_dev = worst(&cv.deviance, &case.cvm, false);
        let cvsd_dev = worst(&cv.deviance_se, &case.cvsd, false);
        println!(
            "{}: lambda {lambda_dev:e}, cvm {cvm_dev:e}, cvsd {cvsd_dev:e}",
            case.name
        );
        assert!(lambda_dev < 1e-12, "{} lambda path", case.name);
        assert!(cvm_dev < 1e-9, "{} cvm", case.name);
        assert!(cvsd_dev < 1e-9, "{} cvsd", case.name);
        assert_eq!(cv.index_min, case.index_min, "{} lambda.min", case.name);
        assert_eq!(cv.index_one_se, case.index_1se, "{} lambda.1se", case.name);
        assert_eq!(
            cv.index_chosen, case.index_1se,
            "{} default rule",
            case.name
        );
        assert_eq!(prepared.estimate.lambda, Some(cv.lambda_chosen()));

        // Coefficients at lambda.1se: glmnet's first entry is the intercept.
        let names = &prepared.estimate.column_names;
        let ours: Vec<f64> = std::iter::once("intercept".to_string())
            .chain(case.covariate_names.iter().cloned())
            .map(|name| {
                let position = names.iter().position(|n| *n == name).expect("column");
                prepared.estimate.coefficients[position].expect("fitted")
            })
            .collect();
        let coef_dev = worst(&ours, &case.coefficients_1se, false);
        println!("{}: coefficients at lambda.1se {coef_dev:e}", case.name);
        assert!(coef_dev < 1e-9, "{} coefficients", case.name);

        // `lambda.min` is the other rule.
        let min_fit = fit_case(case, LambdaRule::Min);
        let min_cv = min_fit.estimate.cross_validation.as_ref().expect("ran");
        assert_eq!(min_cv.index_chosen, case.index_min, "{}", case.name);
    }
}

/// The same seed gives the same folds and the same choice, and another seed
/// gives other folds.
#[test]
fn folds_follow_the_seed() {
    let a = cross_validation_folds(200, 10, 7).unwrap();
    assert_eq!(a, cross_validation_folds(200, 10, 7).unwrap());
    assert_ne!(a, cross_validation_folds(200, 10, 8).unwrap());
}
