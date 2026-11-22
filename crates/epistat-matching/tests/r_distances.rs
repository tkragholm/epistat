//! Pair distances checked against R 4.6's `MatchIt` 4.8.2.
//!
//! The reference is `tests/data/matchit_distances.json`, written by
//! `tests/data/matchit_distances.R`. For three data sets it holds the matrix
//! of distances between every treated unit and every control that MatchIt's
//! `mahalanobis_dist()`, `robust_mahalanobis_dist()`, `euclidean_dist()` and
//! `scaled_euclidean_dist()` return. The data are a sample of `lalonde` with
//! the factor `race`, the same sample with eight units discarded, and a
//! simulated set with ties, a binary column, a three-level factor and a
//! constant column.
//!
//! What the tests check:
//!
//! * `PooledByDegreesOfFreedom` is MatchIt's default covariance, and every pair
//!   distance agrees. `PooledWithinGroups`, the plain mean of the two group
//!   covariances, differs on these unequal groups and the test asserts that.
//! * `FullSample` agrees with MatchIt called with `var` set to the full-sample
//!   covariance.
//! * `RankTieAdjusted` with the full-sample covariance agrees with
//!   `robust_mahalanobis_dist()`, and on the tied data it differs from `Rank`.
//! * `EuclideanScale::None` and `PooledWithinGroupSd` agree with
//!   `euclidean_dist()` and `scaled_euclidean_dist()`.
//! * With units discarded, MatchIt takes the covariance, the ranks and the
//!   standard deviations from the kept units. Handing the crate only the kept
//!   units gives the same distances between them.
//! * A constant column is left out of every distance, as in MatchIt. When every
//!   column is constant MatchIt returns zero for every pair and the crate
//!   returns an error, since no distance is defined (decision 59). A column
//!   with no variance within either group gives NaN in MatchIt's scaled
//!   Euclidean distance and an error here.
//!
//! Tolerances. The Euclidean distances are sums of a few squares, so the
//! relative tolerance is 1e-13, a few hundred times the rounding of the
//! columns' scaling. The Mahalanobis distances invert a covariance matrix, and
//! the relative error of a quadratic form in the inverse is bounded by the
//! condition number of the correlation matrix times the rounding unit. The
//! script records that condition number, and the tolerance is 64 times the
//! product, plus 1e-13 of the largest distance for the 15 digits of the
//! reference file. The crate adds no ridge to these matrices, and the plain
//! factorisation succeeds.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{
    EuclideanPreparationConfig, EuclideanScale, IdMapEuclideanDistance, IdMapMahalanobisDistance,
    prepare_euclidean_distance_config,
};

use r_reference::{assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
    constant_only: ConstantOnly,
    constant_within_groups: WithinGroups,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    matchit: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    treat: Vec<u8>,
    numerics: BTreeMap<String, Vec<f64>>,
    factors: BTreeMap<String, Vec<String>>,
    distances: BTreeMap<String, Vec<Vec<f64>>>,
    cond: BTreeMap<String, f64>,
}

#[derive(Deserialize)]
struct ConstantOnly {
    error: bool,
    distances: Vec<f64>,
}

#[derive(Deserialize)]
struct WithinGroups {
    scaled_euclidean: Vec<serde_json::Value>,
}

fn reference() -> Reference {
    load_reference("matchit_distances.json")
}

fn record(id: String, covariates: Vec<(String, CovariateValue)>) -> BalanceRecord {
    let mut record = BalanceRecord::new(id, NaiveDate::from_ymd_opt(2000, 1, 1).unwrap());
    record.covariates.extend(covariates);
    record
}

/// Treated units as anchors and controls as candidates, in data order.
fn records(case: &Case) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (row, treat) in case.treat.iter().enumerate() {
        let mut covariates: Vec<(String, CovariateValue)> = case
            .numerics
            .iter()
            .map(|(name, values)| (name.clone(), CovariateValue::Numeric(values[row])))
            .collect();
        covariates.extend(case.factors.iter().map(|(name, values)| {
            (
                name.clone(),
                CovariateValue::Categorical(values[row].clone()),
            )
        }));
        if *treat == 1 {
            anchors.push(record(format!("t{row}"), covariates));
        } else {
            candidates.push(record(format!("c{row}"), covariates));
        }
    }
    (anchors, candidates)
}

/// The largest deviation from the reference matrix in units of its tolerance
/// `relative * distance + absolute`.
fn worst_ratio(
    got: impl Fn(&BalanceRecord, &BalanceRecord) -> Option<f64>,
    anchors: &[BalanceRecord],
    candidates: &[BalanceRecord],
    expected: &[Vec<f64>],
    relative: f64,
) -> f64 {
    assert_eq!(expected.len(), anchors.len());
    let largest = expected
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, value| acc.max(*value));
    let mut worst = 0.0_f64;
    for (anchor, row) in anchors.iter().zip(expected) {
        assert_eq!(row.len(), candidates.len());
        for (candidate, want) in candidates.iter().zip(row) {
            let value = got(anchor, candidate).expect("a distance for every pair");
            let tolerance = relative * want + 1e-13 * largest;
            worst = worst.max((value - want).abs() / tolerance);
        }
    }
    worst
}

fn mahalanobis_ratio(
    case: &Case,
    key: &str,
    strategy: MahalanobisCovarianceStrategy,
    transform: MahalanobisTransform,
) -> f64 {
    let (anchors, candidates) = records(case);
    let config = MahalanobisPreparationConfig::builder()
        .covariance_strategy(strategy)
        .transform(transform)
        .build();
    let prepared = prepare_mahalanobis_distance_config(&anchors, &candidates, None, &config)
        .expect("preparation");
    let metric = IdMapMahalanobisDistance::new(&prepared.vectors, &prepared.inverse_covariance);
    assert!(
        prepared.covariance_rank == prepared.dimension
            || transform == MahalanobisTransform::RankTieAdjusted,
        "{}: the covariance was singular",
        case.name
    );
    let relative = 64.0 * case.cond[key] * f64::EPSILON;
    worst_ratio(
        |a, c| metric.distance(a, c),
        &anchors,
        &candidates,
        &case.distances[key],
        relative,
    )
}

#[test]
fn versions_are_recorded() {
    let reference = reference();
    assert_r_version(&reference.versions.r);
    assert_eq!(reference.versions.matchit, "4.8.2");
}

#[test]
fn mahalanobis_pooled_covariance_is_matchits_default() {
    for case in reference().cases {
        let ratio = mahalanobis_ratio(
            &case,
            "mahalanobis",
            MahalanobisCovarianceStrategy::PooledByDegreesOfFreedom,
            MahalanobisTransform::Raw,
        );
        assert!(ratio <= 1.0, "{}: {ratio} times the tolerance", case.name);
    }
}

#[test]
fn mean_of_group_covariances_is_not_matchits_pooled_covariance() {
    for case in reference().cases {
        let ratio = mahalanobis_ratio(
            &case,
            "mahalanobis",
            MahalanobisCovarianceStrategy::PooledWithinGroups,
            MahalanobisTransform::Raw,
        );
        assert!(ratio > 1e6, "{}: {ratio}", case.name);
    }
}

#[test]
fn mahalanobis_full_sample_covariance() {
    for case in reference().cases {
        let ratio = mahalanobis_ratio(
            &case,
            "mahalanobis_full",
            MahalanobisCovarianceStrategy::FullSample,
            MahalanobisTransform::Raw,
        );
        assert!(ratio <= 1.0, "{}: {ratio} times the tolerance", case.name);
    }
}

#[test]
fn robust_mahalanobis_with_tie_adjustment() {
    for case in reference().cases {
        let ratio = mahalanobis_ratio(
            &case,
            "robust",
            MahalanobisCovarianceStrategy::FullSample,
            MahalanobisTransform::RankTieAdjusted,
        );
        assert!(ratio <= 1.0, "{}: {ratio} times the tolerance", case.name);
    }
}

#[test]
fn rank_without_adjustment_differs_on_tied_data() {
    let reference = reference();
    let case = reference
        .cases
        .iter()
        .find(|case| case.name == "sim_ties")
        .unwrap();
    let ratio = mahalanobis_ratio(
        case,
        "robust",
        MahalanobisCovarianceStrategy::FullSample,
        MahalanobisTransform::Rank,
    );
    assert!(ratio > 1e6, "{ratio}");
}

#[test]
fn euclidean_and_scaled_euclidean() {
    for case in reference().cases {
        let (anchors, candidates) = records(&case);
        for (key, scale) in [
            ("euclidean", EuclideanScale::None),
            ("scaled_euclidean", EuclideanScale::PooledWithinGroupSd),
        ] {
            let Some(expected) = case.distances.get(key) else {
                continue;
            };
            let config = EuclideanPreparationConfig::builder().scale(scale).build();
            let prepared =
                prepare_euclidean_distance_config(&anchors, &candidates, None, &config).unwrap();
            let metric = IdMapEuclideanDistance::new(&prepared.vectors);
            let ratio = worst_ratio(
                |a, c| metric.distance(a, c),
                &anchors,
                &candidates,
                expected,
                1e-13,
            );
            assert!(
                ratio <= 1.0,
                "{} {key}: {ratio} times the tolerance",
                case.name
            );
        }
    }
}

#[test]
fn scaled_euclidean_of_matchit_check_is_not_vacuous() {
    // Unscaled and scaled distances differ on every data set.
    for case in reference().cases {
        let Some(unscaled) = case.distances.get("euclidean") else {
            continue;
        };
        let scaled = &case.distances["scaled_euclidean"];
        assert!(
            (unscaled[0][1] - scaled[0][1]).abs() > 1e-3,
            "{}",
            case.name
        );
    }
}

#[test]
fn constant_columns_are_left_out() {
    let reference = reference();
    let case = reference
        .cases
        .iter()
        .find(|case| case.name == "sim_ties")
        .unwrap();
    assert!(case.numerics.contains_key("k"));
    let (anchors, candidates) = records(case);
    let prepared = prepare_euclidean_distance_config(
        &anchors,
        &candidates,
        None,
        &EuclideanPreparationConfig::default(),
    )
    .unwrap();
    assert!(!prepared.column_names.iter().any(|name| name == "k"));
    let prepared = prepare_mahalanobis_distance_config(
        &anchors,
        &candidates,
        None,
        &MahalanobisPreparationConfig::default(),
    )
    .unwrap();
    assert!(!prepared.column_names.iter().any(|name| name == "k"));
}

#[test]
fn only_constant_columns_are_an_error() {
    let reference = reference();
    assert!(!reference.constant_only.error);
    assert!(reference.constant_only.distances.iter().all(|d| *d == 0.0));
    let constant = |id: &str| {
        record(
            id.to_string(),
            vec![("k".to_string(), CovariateValue::Numeric(3.0))],
        )
    };
    let anchors = [constant("a1"), constant("a2")];
    let candidates = [constant("c1"), constant("c2"), constant("c3")];
    assert!(matches!(
        prepare_euclidean_distance_config(
            &anchors,
            &candidates,
            None,
            &EuclideanPreparationConfig::default()
        ),
        Err(EstimationError::NoCovariatesAvailable)
    ));
    assert!(matches!(
        prepare_mahalanobis_distance_config(
            &anchors,
            &candidates,
            None,
            &MahalanobisPreparationConfig::default()
        ),
        Err(EstimationError::NoCovariatesAvailable)
    ));
}

#[test]
fn column_without_variance_within_groups_cannot_be_scaled() {
    let reference = reference();
    assert!(
        reference
            .constant_within_groups
            .scaled_euclidean
            .iter()
            .all(|value| value == "NaN")
    );
    let unit = |id: &str, a: f64, b: f64| {
        record(
            id.to_string(),
            vec![
                ("a".to_string(), CovariateValue::Numeric(a)),
                ("b".to_string(), CovariateValue::Numeric(b)),
            ],
        )
    };
    let anchors = [unit("t1", 1.0, 5.0), unit("t2", 2.0, 5.0)];
    let candidates = [
        unit("c1", 3.0, 9.0),
        unit("c2", 1.0, 9.0),
        unit("c3", 2.0, 9.0),
    ];
    let scaled = EuclideanPreparationConfig::builder()
        .scale(EuclideanScale::PooledWithinGroupSd)
        .build();
    assert!(matches!(
        prepare_euclidean_distance_config(&anchors, &candidates, None, &scaled),
        Err(EstimationError::ZeroWithinGroupVariance { column }) if column == "b"
    ));
    assert!(
        prepare_euclidean_distance_config(
            &anchors,
            &candidates,
            None,
            &EuclideanPreparationConfig::default()
        )
        .is_ok()
    );
}
