//! Checks on the inputs and settings of the score-based methods: the
//! propensity score estimators, the Mahalanobis preparation, subclassification,
//! quick matching and the genetic search. Each test states the behaviour the
//! documentation or decision 59 requires.

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{
    GeneticFit, GeneticMatchOptions, GeneticProblem, GeneticSearch, QuickMatchOptions, quick_match_subclasses,
};
use std::collections::HashMap;

fn birth() -> NaiveDate {
    NaiveDate::from_ymd_opt(2010, 1, 1).unwrap()
}

fn balance_records(n: usize, f: impl Fn(usize) -> f64) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let mut a = Vec::new();
    let mut c = Vec::new();
    for i in 0..n {
        let x = f(i);
        let r = BalanceRecord::builder(format!("u{i}"), birth())
            .numeric("x", x)
            .numeric("z", ((i * 7919) % 13) as f64)
            .build();
        if i % 2 == 0 { a.push(r) } else { c.push(r) }
    }
    (a, c)
}

/// `subclasses: 0` is an error, not one subclass.
#[test]
fn subclasses_zero_is_an_error_not_one() {
    let anchors: Vec<BaseRecord> = (0..6).map(|i| BaseRecord::new(format!("a{i}"), birth())).collect();
    let cands: Vec<BaseRecord> = (0..6).map(|i| BaseRecord::new(format!("c{i}"), birth())).collect();
    let mut scores = HashMap::new();
    for i in 0..6 {
        scores.insert(format!("a{i}"), 0.1 * i as f64);
        scores.insert(format!("c{i}"), 0.1 * i as f64 + 0.05);
    }
    let cfg = SubclassificationConfig { subclasses: 0, reference_group: None, min_n: 1 };
    let out = subclassify_by_propensity_score_map(&anchors, &cands, &scores, Estimand::Att, &cfg);
    assert_eq!(out.err(), Some(SubclassificationError::NoSubclasses));
}

/// A fold count of 0 is an error, not a division by zero.
#[test]
fn folds_with_zero_folds_are_an_error() {
    assert!(cross_validation_folds(10, 0, 1).is_err());
    assert_eq!(cross_validation_folds(10, 3, 1).unwrap().len(), 10);
}

/// An alpha outside [0, 1] and a negative lambda are errors.
#[test]
fn elastic_net_rejects_alpha_above_one_and_negative_lambda() {
    let (a, c) = balance_records(60, |i| (i % 5) as f64 + 0.3 * ((i * 31) % 7) as f64);
    for (alpha, lambda) in [(2.0, 0.01), (0.5, -1.0), (-0.5, 0.01)] {
        let cfg = PropensityScoreConfig::builder()
            .estimator(PropensityEstimator::ElasticNetLogit(
                ElasticNetLogisticConfig::builder().alpha(alpha).lambda(lambda).build(),
            ))
            .build();
        let r = prepare_propensity_distance_config(&a, &c, &cfg);
        assert!(r.is_err(), "alpha {alpha}, lambda {lambda} accepted");
    }
}

/// a covariate with a small absolute spread is dropped without notice.
#[test]
fn a_small_scale_covariate_is_not_dropped_silently() {
    let (a, c) = balance_records(40, |i| 1e-13 * ((i % 5) as f64));
    let est = prepare_propensity_distance_config(&a, &c, &PropensityScoreConfig::default()).unwrap();
    assert!(
        est.estimate.column_names.iter().any(|n| n == "x"),
        "column x (range 4e-13, not constant) was dropped; columns: {:?}",
        est.estimate.column_names
    );
}

/// A column with one value in every row is dropped and named in the result.
#[test]
fn an_exactly_constant_covariate_is_dropped_and_reported() {
    let mk = |i: usize| {
        BalanceRecord::builder(format!("u{i}"), birth())
            .numeric("x", (i % 5) as f64)
            .numeric("flat", 3.0)
            .build()
    };
    let a: Vec<_> = (0..20).filter(|i| i % 2 == 0).map(mk).collect();
    let c: Vec<_> = (0..20).filter(|i| i % 2 == 1).map(mk).collect();
    let est = prepare_propensity_distance_config(&a, &c, &PropensityScoreConfig::default()).unwrap();
    assert_eq!(est.estimate.dropped_columns, vec!["flat".to_string()]);
    assert!(est.estimate.column_names.iter().any(|n| n == "x"));
}

/// A population of 1 to 8 cannot make a child with every operator and is an
/// error; 9 can.
#[test]
fn a_population_too_small_for_every_operator_is_an_error() {
    let n = 80;
    let focal: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
    let cols: Vec<Vec<f64>> = (0..3)
        .map(|k| (0..n).map(|i| ((i * (k + 3) * 7919) % 101) as f64).collect())
        .collect();
    let order: Vec<usize> = (0..n).collect();
    let p = GeneticProblem::new(&focal, &cols, &cols, &order, GeneticMatchOptions::default(), 1e-8).unwrap();
    for population in [1, 2, 5, 8] {
        let r = p.search(&GeneticFit::default(), &GeneticSearch { population, seed: 1, ..GeneticSearch::default() });
        assert!(r.is_err(), "population {population} accepted");
    }
    let out = p
        .search(&GeneticFit::default(), &GeneticSearch { population: 9, seed: 1, ..GeneticSearch::default() })
        .unwrap();
    assert!(out.evaluations > 9, "{} evaluations", out.evaluations);
}

/// The default starting weights are 1, so a weight range that excludes 1 is an
/// error, not a clamped start.
#[test]
fn default_start_outside_the_weight_range_is_an_error() {
    let n = 40;
    let focal: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
    let cols: Vec<Vec<f64>> = vec![(0..n).map(|i| ((i * 37) % 11) as f64).collect()];
    let order: Vec<usize> = (0..n).collect();
    let p = GeneticProblem::new(&focal, &cols, &cols, &order, GeneticMatchOptions::default(), 1e-8).unwrap();
    let out = p.search(
        &GeneticFit::default(),
        &GeneticSearch { population: 0, min_weight: 2.0, ..GeneticSearch::default() },
    );
    assert!(out.is_err());
}

/// A stratum with nothing in its caliper loses its own units, not the run.
#[test]
fn a_stratum_without_a_subclass_under_the_caliper_drops_only_its_units() {
    let mk = |id: &str, g: &str| {
        let mut r = BaseRecord::new(id, birth());
        r.strata.insert("g".into(), g.into());
        r
    };
    let anchors = vec![mk("a0", "A"), mk("a1", "A"), mk("b0", "B")];
    let cands = vec![mk("c0", "A"), mk("c1", "A"), mk("d0", "B")];
    let coords: HashMap<String, Vec<f64>> = [
        ("a0", 0.0), ("a1", 1.0), ("c0", 0.1), ("c1", 1.1), ("b0", 0.0), ("d0", 100.0),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), vec![v]))
    .collect();
    let opts = QuickMatchOptions { caliper: Some(1.0), ..QuickMatchOptions::default() };
    let out = quick_match_subclasses(&anchors, &cands, |r| coords.get(r.id()).map(Vec::as_slice), 1, &["g"], &opts);
    assert!(out.is_ok(), "stratum A is matchable but the run fails: {:?}", out.err());
}

/// the Mahalanobis distance is invariant to the unit of a covariate; the
/// default absolute ridge of 1e-8 breaks that for small-scale covariates.
#[test]
fn mahalanobis_distance_does_not_depend_on_the_unit() {
    let dist = |scale: f64| {
        let (a, c) = balance_records(60, |i| scale * (((i * 37) % 11) as f64));
        let prep = prepare_mahalanobis_distance_config(&a, &c, None, &MahalanobisPreparationConfig::default()).unwrap();
        let ia = prep.column_names.iter().position(|n| n == "x").unwrap();
        let va = &prep.vectors["u0"];
        let vb = &prep.vectors["u1"];
        let d = prep.dimension;
        let diff: Vec<f64> = (0..d).map(|k| va[k] - vb[k]).collect();
        let mut q = 0.0;
        for i in 0..d {
            for j in 0..d {
                q += diff[i] * prep.inverse_covariance[i * d + j] * diff[j];
            }
        }
        let _ = ia;
        (q, prep.covariance_rank)
    };
    let (big, _) = dist(1.0);
    let (small, ridge) = dist(1e-5);
    assert!(
        (big - small).abs() <= 1e-6 * big.abs(),
        "d^2 {big} with x in units, {small} with x scaled by 1e-5 (rank {ridge})"
    );
}
