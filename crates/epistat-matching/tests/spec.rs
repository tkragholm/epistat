//! A `MatchSpec` gives the outcome of the typed job it describes, bit for bit.
//!
//! Each test builds the job by hand with the builder, runs the spec, and
//! compares the debug rendering of the two outcomes, which prints every float
//! in full. The cohorts and options are those of `tests/golden.rs`, and the
//! spec's outcomes are also compared with its pinned hashes, so a spec that
//! drifted from the engine would fail there as well.
//!
//! The error tests check that every invalid option is an error that names the
//! option, and that the typed job's way of reporting an invalid configuration,
//! a count in the diagnostics of an empty outcome, is an error from the spec.

#![expect(
    clippy::unreadable_literal,
    reason = "the hashes are pasted as the test prints them"
)]

mod common;

use common::{Level, Person, cohort, records};
use epistat_matching::prelude::*;
use std::collections::BTreeMap;

fn fnv(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn hashes(outcome: &MatchOutcome) -> (u64, u64) {
    let unit = match_weights_from_pairs(
        &outcome.pairs,
        MatchWeightMethod::AnchorUnitCandidateFractional,
    )
    .to_table();
    let pair = pair_weights_from_pairs(
        &outcome.pairs,
        MatchWeightMethod::AnchorUnitCandidateFractional,
    )
    .to_table();
    (
        fnv(&format!("{outcome:?}")),
        fnv(&format!("{unit:?}{pair:?}")),
    )
}

fn balance(record: &BaseRecord) -> BalanceRecord {
    BalanceRecord {
        core: record.clone(),
        covariates: std::collections::HashMap::new(),
    }
}

/// The anchors and candidates of the golden standard-matching tests.
fn standard_inputs(seed: u64) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let people = cohort(5_000, 200, 8, seed);
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for p in &people {
        let mut r = BaseRecord::new(&p.id, p.birth);
        r.strata.insert("sex".to_string(), p.sex.to_string());
        r.strata
            .insert("municipality".to_string(), p.municipality.clone());
        r.strata.insert("family".to_string(), p.family.clone());
        if p.exposed {
            anchors.push(balance(&r));
        } else {
            candidates.push(balance(&r));
        }
    }
    for a in anchors.iter().take(20) {
        let mut twin = a.clone();
        twin.core.birth_date += chrono::TimeDelta::days(1);
        candidates.push(twin);
    }
    (anchors, candidates)
}

fn transition_cohort(
    people: &[Person],
    with_family: bool,
) -> Vec<RoleTransitionRecord<BalanceRecord>> {
    records(people, Level::Municipality, with_family, false)
        .into_iter()
        .map(|row| RoleTransitionRecord::from_record(balance(&row.record), row.transition_date))
        .collect()
}

fn run_pairs(
    spec: &MatchSpec,
    anchors: &[BalanceRecord],
    candidates: &[BalanceRecord],
) -> MatchOutcome {
    match run_spec(
        spec,
        SpecInput::Standard {
            anchors,
            candidates,
        },
    )
    .expect("valid spec")
    {
        SpecOutput::Pairs { outcome, .. } => *outcome,
        other => panic!("expected pairs, got {other:?}"),
    }
}

fn same(left: &MatchOutcome, right: &MatchOutcome) {
    assert!(!left.pairs.is_empty(), "no pairs, test is vacuous");
    assert_eq!(format!("{left:?}"), format!("{right:?}"));
}

#[test]
fn spec_reproduces_golden_standard_without_replacement_with_fallback() {
    let (anchors, candidates) = standard_inputs(18);
    let spec = MatchSpec {
        seed: 4,
        ratio: 5,
        ratio_fallback: vec![3, 1],
        birth_window_days: 20,
        exact: vec!["sex".into(), "municipality".into()],
        ..MatchSpec::default()
    };
    let by_spec = run_pairs(&spec, &anchors, &candidates);
    let by_hand = MatchJob::new_standard(&anchors, &candidates, 4)
        .with_ratio(MatchRatio::new(5).unwrap())
        .with_ratio_fallback(vec![
            MatchRatio::new(3).unwrap(),
            MatchRatio::new(1).unwrap(),
        ])
        .with_birth_window(BirthDateWindowDays::new(20).unwrap())
        .with_exact_matches(["sex", "municipality"])
        .run();
    same(&by_spec, &by_hand);
    assert_eq!(hashes(&by_spec), (0x54babf39ce9e9dba, 0x7e322fddce9b9854));
}

#[test]
fn spec_reproduces_golden_standard_with_replacement() {
    let (anchors, candidates) = standard_inputs(19);
    let spec = MatchSpec {
        seed: 8,
        ratio: 2,
        birth_window_days: 30,
        exact: vec!["sex".into(), "municipality".into()],
        replace: true,
        ..MatchSpec::default()
    };
    let by_spec = run_pairs(&spec, &anchors, &candidates);
    assert_eq!(hashes(&by_spec), (0x19c9eab8807cd43f, 0x45fb7072aa50bd57));
}

#[test]
fn spec_reproduces_golden_standard_unique_by_key() {
    let (anchors, candidates) = standard_inputs(20);
    let spec = MatchSpec {
        seed: 6,
        ratio: 3,
        birth_window_days: 40,
        exact: vec!["sex".into()],
        unique_by_key: Some("family".into()),
        ..MatchSpec::default()
    };
    let by_spec = run_pairs(&spec, &anchors, &candidates);
    assert_eq!(hashes(&by_spec), (0xd90cf67cee081244, 0x0ad234b2f7937bb0));
}

#[test]
fn spec_reproduces_golden_lowest_index() {
    let (anchors, candidates) = standard_inputs(21);
    let spec = MatchSpec {
        method: MatchMethod::LowestIndex,
        seed: 6,
        ratio: 2,
        birth_window_days: 10,
        ..MatchSpec::default()
    };
    let by_spec = run_pairs(&spec, &anchors, &candidates);
    assert_eq!(hashes(&by_spec), (0xfbe1e86e4ad4936d, 0x1a8b1221f7c395a3));
}

#[test]
fn spec_reproduces_golden_risk_set_nearest_birth_date() {
    let people = cohort(4_000, 100, 10, 14);
    let cohort_records = transition_cohort(&people, false);
    let spec = MatchSpec {
        mode: MatchMode::RiskSet,
        method: MatchMethod::NearestBirthDate,
        seed: 99,
        ratio: 3,
        birth_window_days: 60,
        age_limit_years: Some(6),
        exact: vec!["sex".into(), "municipality".into()],
        require_alive: true,
        ..MatchSpec::default()
    };
    let output = run_spec(
        &spec,
        SpecInput::RiskSet {
            cohort: &cohort_records,
        },
    )
    .expect("valid spec");
    let SpecOutput::Pairs { outcome, .. } = output else {
        panic!("expected pairs");
    };
    let by_hand = MatchJob::new_transition(&cohort_records, 99)
        .with_ratio(MatchRatio::new(3).unwrap())
        .with_birth_window(BirthDateWindowDays::new(60).unwrap())
        .with_age_limit(AgeLimitYears::new(6).unwrap())
        .with_exact_matches(["sex", "municipality"])
        .with_alive_check()
        .with_strategy(NearestBirthDateSelection)
        .run();
    same(&outcome, &by_hand);
    assert_eq!(hashes(&outcome), (0x959f7432847ef008, 0x585c1833c1e022cf));
}

#[test]
fn spec_reproduces_golden_risk_set_unique_by_family() {
    let people = cohort(4_000, 100, 10, 15);
    let cohort_records = transition_cohort(&people, true);
    let spec = MatchSpec {
        mode: MatchMode::RiskSet,
        seed: 99,
        ratio: 3,
        birth_window_days: 60,
        age_limit_years: Some(6),
        exact: vec!["sex".into(), "municipality".into()],
        unique_by_key: Some("family".into()),
        require_alive: true,
        ..MatchSpec::default()
    };
    let output = run_spec(
        &spec,
        SpecInput::RiskSet {
            cohort: &cohort_records,
        },
    )
    .expect("valid spec");
    let SpecOutput::Pairs { outcome, .. } = output else {
        panic!("expected pairs");
    };
    assert_eq!(hashes(&outcome), (0x65b45674334aeb7e, 0x9e58c677dd7580ca));
}

#[test]
fn spec_with_sex_antiexact_and_caliper_matches_the_typed_job() {
    let (mut anchors, mut candidates) = standard_inputs(22);
    for (index, record) in anchors.iter_mut().chain(candidates.iter_mut()).enumerate() {
        record.core.numerics.insert("x".into(), (index % 17) as f64);
    }
    let spec = MatchSpec {
        seed: 3,
        ratio: 2,
        birth_window_days: 25,
        exact: vec!["municipality".into()],
        match_sex: true,
        antiexact: vec!["family".into()],
        calipers: BTreeMap::from([("x".to_string(), 6.0)]),
        ..MatchSpec::default()
    };
    let by_spec = run_pairs(&spec, &anchors, &candidates);
    let by_hand = MatchJob::new_standard(&anchors, &candidates, 3)
        .with_ratio(MatchRatio::new(2).unwrap())
        .with_birth_window(BirthDateWindowDays::new(25).unwrap())
        .with_exact_matches(["municipality"])
        .with_gender_match()
        .with_antiexact("family")
        .with_constraint(epistat_matching::constraints::caliper_on_field::<
            BalanceRecord,
        >("x", 6.0))
        .run();
    same(&by_spec, &by_hand);
}

#[test]
fn spec_with_rounds_and_each_selection_matches_the_typed_job() {
    let (anchors, candidates) = standard_inputs(23);
    let base = MatchSpec {
        seed: 11,
        ratio: 3,
        birth_window_days: 40,
        exact: vec!["sex".into()],
        assignment: Some(AssignmentSpec::Rounds),
        ..MatchSpec::default()
    };
    let job = || {
        MatchJob::new_standard(&anchors, &candidates, 11)
            .with_ratio(MatchRatio::new(3).unwrap())
            .with_birth_window(BirthDateWindowDays::new(40).unwrap())
            .with_exact_matches(["sex"])
    };
    let random = run_pairs(&base, &anchors, &candidates);
    same(&random, &job().with_round_assignment().run());
    let lowest = run_pairs(
        &MatchSpec {
            method: MatchMethod::LowestIndex,
            ..base.clone()
        },
        &anchors,
        &candidates,
    );
    same(
        &lowest,
        &job()
            .with_strategy(DeterministicSelection)
            .with_round_assignment()
            .run(),
    );
    let birth = run_pairs(
        &MatchSpec {
            method: MatchMethod::NearestBirthDate,
            ..base
        },
        &anchors,
        &candidates,
    );
    same(
        &birth,
        &job()
            .with_strategy(NearestBirthDateSelection)
            .with_round_assignment()
            .run(),
    );
}

/// Numeric and categorical covariates on every unit, and a score.
fn with_covariates(rows: &mut [BalanceRecord], offset: usize, treated: bool) {
    for (index, row) in rows.iter_mut().enumerate() {
        let k = index + offset;
        let age = 20.0 + (k * 7 % 23) as f64 + if treated { 3.0 } else { 0.0 };
        let income = 100.0 + (k * 13 % 41) as f64 + if treated { 5.0 } else { 0.0 };
        row.covariates
            .insert("age".into(), CovariateValue::Numeric(age));
        row.covariates
            .insert("income".into(), CovariateValue::Numeric(income));
        row.covariates.insert(
            "region".into(),
            CovariateValue::Categorical(format!("r{}", k % 3)),
        );
        row.core.numerics.insert("age".into(), age);
    }
}

fn covariate_inputs() -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let (anchors, candidates) = standard_inputs(24);
    let mut anchors: Vec<_> = anchors.into_iter().take(60).collect();
    let mut candidates: Vec<_> = candidates.into_iter().take(240).collect();
    for row in anchors.iter_mut().chain(candidates.iter_mut()) {
        row.core.strata.clear();
        row.core.birth_date = chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
    }
    with_covariates(&mut anchors, 0, true);
    with_covariates(&mut candidates, 1000, false);
    (anchors, candidates)
}

fn covariates(names: &[&str]) -> CovariatesSpec {
    CovariatesSpec {
        names: names.iter().map(|name| (*name).to_string()).collect(),
        missing: MissingSpec::Error,
    }
}

fn wide_window() -> i32 {
    36_500
}

#[test]
fn spec_nearest_on_a_propensity_score_matches_the_typed_job() {
    let (anchors, candidates) = covariate_inputs();
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Propensity {
            covariates: covariates(&["age", "income", "region"]),
            link: LinkSpec::Logit,
            estimator: EstimatorSpec::Glm,
            scale: ScaleSpec::Probability,
            sampling_weights: None,
            caliper: Some(0.2),
        },
        nearest: NearestSpec {
            m_order: Some(OrderSpec::Smallest),
            mahvars: None,
        },
        std_caliper: true,
        ratio: 2,
        birth_window_days: wide_window(),
        seed: 5,
        ..MatchSpec::default()
    };
    let SpecOutput::Pairs {
        outcome,
        propensity,
    } = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect("valid spec")
    else {
        panic!("expected pairs");
    };
    assert!(propensity.is_some());

    let config = PropensityScoreConfig::builder()
        .encoding(
            CovariateEncodingConfig::builder()
                .covariate_keys(vec!["age".into(), "income".into(), "region".into()])
                .include_intercept(true)
                .build(),
        )
        .caliper(DistanceCaliper::new(0.2).unwrap())
        .build();
    let prepared = prepare_propensity_distance_config(&anchors, &candidates, &config).unwrap();
    let by_hand = MatchJob::new_standard(&anchors, &candidates, 5)
        .with_ratio(MatchRatio::new(2).unwrap())
        .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
        .with_distance_config(prepared.distance_config)
        .with_nearest()
        .with_m_order(MatchOrder::Smallest)
        .with_std_caliper(true)
        .run();
    same(&outcome, &by_hand);
}

#[test]
fn spec_nearest_on_mahalanobis_with_replacement_matches_the_typed_job() {
    let (anchors, candidates) = covariate_inputs();
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Mahalanobis {
            covariates: covariates(&["age", "income"]),
            covariance: CovarianceSpec::PooledByDegreesOfFreedom,
            transform: TransformSpec::Raw,
            caliper: None,
        },
        nearest: NearestSpec {
            m_order: Some(OrderSpec::Random),
            mahvars: None,
        },
        replace: true,
        birth_window_days: wide_window(),
        seed: 9,
        ..MatchSpec::default()
    };
    let by_spec = run_pairs(&spec, &anchors, &candidates);
    let prepared = prepare_mahalanobis_distance_config(
        &anchors,
        &candidates,
        None,
        &MahalanobisPreparationConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".into(), "income".into()])
                    .build(),
            )
            .build(),
    )
    .unwrap();
    let by_hand = MatchJob::new_standard(&anchors, &candidates, 9)
        .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
        .with_replacement(true)
        .with_distance_config(prepared.distance_config)
        .with_nearest()
        .with_m_order(MatchOrder::Random)
        .run();
    same(&by_spec, &by_hand);
}

#[test]
fn spec_nearest_with_mahvars_and_euclidean_match_the_typed_jobs() {
    let (anchors, candidates) = covariate_inputs();
    let propensity = DistanceSpec::Propensity {
        covariates: covariates(&["age", "income"]),
        link: LinkSpec::Probit,
        estimator: EstimatorSpec::Glm,
        scale: ScaleSpec::Linear,
        sampling_weights: None,
        caliper: Some(1.0),
    };
    let mahalanobis = DistanceSpec::Mahalanobis {
        covariates: covariates(&["age", "income"]),
        covariance: CovarianceSpec::PooledByDegreesOfFreedom,
        transform: TransformSpec::Raw,
        caliper: None,
    };
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: propensity,
        nearest: NearestSpec {
            m_order: None,
            mahvars: Some(mahalanobis),
        },
        birth_window_days: wide_window(),
        ..MatchSpec::default()
    };
    let by_spec = run_pairs(&spec, &anchors, &candidates);
    let score = prepare_propensity_distance_config(
        &anchors,
        &candidates,
        &PropensityScoreConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".into(), "income".into()])
                    .include_intercept(true)
                    .build(),
            )
            .logistic(
                LogisticRegressionConfig::builder()
                    .link(PropensityLink::Probit)
                    .build(),
            )
            .output_scale(PropensityScoreOutputScale::LinearPredictor)
            .caliper(DistanceCaliper::new(1.0).unwrap())
            .build(),
    )
    .unwrap();
    let ranking = prepare_mahalanobis_distance_config(
        &anchors,
        &candidates,
        None,
        &MahalanobisPreparationConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".into(), "income".into()])
                    .build(),
            )
            .build(),
    )
    .unwrap();
    let by_hand = MatchJob::new_standard(&anchors, &candidates, 0)
        .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
        .with_distance_config(score.distance_config)
        .with_nearest()
        .with_mahvars(ranking.distance_config)
        .run();
    same(&by_spec, &by_hand);

    let euclidean = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Euclidean {
            covariates: covariates(&["age", "income", "region"]),
            scaled: true,
            caliper: None,
        },
        birth_window_days: wide_window(),
        ..MatchSpec::default()
    };
    let by_spec = run_pairs(&euclidean, &anchors, &candidates);
    let prepared = prepare_euclidean_distance_config(
        &anchors,
        &candidates,
        None,
        &EuclideanPreparationConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".into(), "income".into(), "region".into()])
                    .build(),
            )
            .scale(EuclideanScale::PooledWithinGroupSd)
            .build(),
    )
    .unwrap();
    let by_hand = MatchJob::new_standard(&anchors, &candidates, 0)
        .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
        .with_distance_config(prepared.distance_config)
        .with_nearest()
        .run();
    same(&by_spec, &by_hand);
}

#[test]
fn spec_exact_and_subclass_match_the_functions_they_wrap() {
    let (anchors, candidates) = standard_inputs(25);
    let spec = MatchSpec {
        method: MatchMethod::Exact,
        exact: vec!["sex".into(), "municipality".into()],
        ..MatchSpec::default()
    };
    let SpecOutput::Exact(outcome) = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect("valid spec") else {
        panic!("expected exact output");
    };
    let by_hand = exact_match_subclasses(&anchors, &candidates, &["sex", "municipality"]).unwrap();
    assert!(!by_hand.assignments.is_empty());
    assert_eq!(
        format!("{:?}", outcome.assignments_sorted()),
        format!("{:?}", by_hand.assignments_sorted())
    );
    assert_eq!(
        format!("{:?}", outcome.att_weights_sorted()),
        format!("{:?}", by_hand.att_weights_sorted())
    );
    assert_eq!(outcome.dropped_ids, by_hand.dropped_ids);

    let (anchors, candidates) = covariate_inputs();
    let spec = MatchSpec {
        method: MatchMethod::Subclass,
        distance: DistanceSpec::Propensity {
            covariates: covariates(&["age", "income"]),
            link: LinkSpec::Logit,
            estimator: EstimatorSpec::Glm,
            scale: ScaleSpec::Probability,
            sampling_weights: None,
            caliper: None,
        },
        subclass: SubclassSpec {
            n: 5,
            min_n: 1,
            reference: None,
        },
        ..MatchSpec::default()
    };
    let SpecOutput::Subclass { outcome, .. } = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect("valid spec") else {
        panic!("expected subclass output");
    };
    let prepared = prepare_propensity_distance_config(
        &anchors,
        &candidates,
        &PropensityScoreConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".into(), "income".into()])
                    .include_intercept(true)
                    .build(),
            )
            .build(),
    )
    .unwrap();
    let by_hand = subclassify_by_propensity_score_map(
        &anchors,
        &candidates,
        &prepared.estimate.distance_scores,
        Estimand::Att,
        &SubclassificationConfig {
            subclasses: 5,
            reference_group: None,
            min_n: 1,
        },
    );
    assert!(!by_hand.assignments.is_empty());
    assert_eq!(
        format!("{:?}", outcome.assignments_sorted()),
        format!("{:?}", by_hand.assignments_sorted())
    );
    assert_eq!(
        format!("{:?}", outcome.weights_sorted()),
        format!("{:?}", by_hand.weights_sorted())
    );
}

#[test]
fn a_spec_round_trips_through_json() {
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Propensity {
            covariates: covariates(&["age"]),
            link: LinkSpec::Cloglog,
            estimator: EstimatorSpec::ElasticNetCv {
                alpha: 0.5,
                nfolds: 5,
                nlambda: 50,
                rule: LambdaSpec::Min,
                standardize: true,
            },
            scale: ScaleSpec::Linear,
            sampling_weights: Some(BTreeMap::from([("a".to_string(), 1.5)])),
            caliper: Some(0.1),
        },
        calipers: BTreeMap::from([("age".to_string(), 2.0)]),
        estimand: Estimand::Atc,
        ..MatchSpec::default()
    };
    let text = serde_json::to_string(&spec).unwrap();
    let back: MatchSpec = serde_json::from_str(&text).unwrap();
    assert_eq!(spec, back);
    let defaults: MatchSpec = serde_json::from_str("{}").unwrap();
    assert_eq!(defaults, MatchSpec::default());
    let named: MatchSpec = serde_json::from_str(
        r#"{"method": "nearest", "distance": {"kind": "mahalanobis", "covariates": {"names": ["a"]}},
            "nearest": {"m_order": "closest"}, "ratio": 2}"#,
    )
    .unwrap();
    assert_eq!(named.ratio, 2);
    assert_eq!(named.nearest.m_order, Some(OrderSpec::Closest));
    assert!(serde_json::from_str::<MatchSpec>(r#"{"ratoi": 2}"#).is_err());
}

fn message(spec: &MatchSpec) -> String {
    let (anchors, candidates) = standard_inputs(26);
    match run_spec(
        spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    ) {
        Ok(_) => panic!("the spec should be refused"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn invalid_specs_are_errors_that_name_the_option() {
    let defaults = MatchSpec::default;
    assert!(
        message(&MatchSpec {
            ratio: 0,
            ..defaults()
        })
        .contains("ratio is 0")
    );
    assert!(
        message(&MatchSpec {
            birth_window_days: -1,
            ..defaults()
        })
        .contains("negative")
    );
    assert!(
        message(&MatchSpec {
            ratio_fallback: vec![2, 0],
            ..defaults()
        })
        .contains("ratio_fallback")
    );
    assert!(
        message(&MatchSpec {
            calipers: BTreeMap::from([("x".to_string(), -1.0)]),
            ..defaults()
        })
        .contains("calipers")
    );
    assert!(
        message(&MatchSpec {
            nearest: NearestSpec {
                m_order: Some(OrderSpec::Data),
                mahvars: None,
            },
            ..defaults()
        })
        .contains("set the method to nearest")
    );
    assert!(
        message(&MatchSpec {
            method: MatchMethod::Nearest,
            assignment: Some(AssignmentSpec::Sequential),
            nearest: NearestSpec {
                m_order: Some(OrderSpec::Closest),
                mahvars: None,
            },
            ..defaults()
        })
        .contains("set assignment to rounds")
    );
    assert!(
        message(&MatchSpec {
            method: MatchMethod::Nearest,
            ratio_fallback: vec![1],
            ..defaults()
        })
        .contains("no fallback ratios")
    );
    assert!(
        message(&MatchSpec {
            method: MatchMethod::Exact,
            ..defaults()
        })
        .contains("strata keys")
    );
    assert!(
        message(&MatchSpec {
            method: MatchMethod::Exact,
            exact: vec!["sex".into()],
            ratio: 2,
            ..defaults()
        })
        .contains("`ratio`")
    );
    assert!(
        message(&MatchSpec {
            method: MatchMethod::Subclass,
            ..defaults()
        })
        .contains("propensity")
    );
    assert!(
        message(&MatchSpec {
            require_alive: true,
            ..defaults()
        })
        .contains("risk_set")
    );
    assert!(
        message(&MatchSpec {
            mode: MatchMode::RiskSet,
            ..defaults()
        })
        .contains("one cohort")
    );
}

#[test]
fn a_covariate_that_is_missing_from_the_units_is_a_distance_error() {
    let (anchors, candidates) = covariate_inputs();
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Euclidean {
            covariates: covariates(&["age", "weight"]),
            scaled: false,
            caliper: None,
        },
        ..MatchSpec::default()
    };
    let error = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect_err("weight is not a covariate of the units");
    assert!(matches!(error, SpecError::Distance(_)), "{error}");
    assert!(error.to_string().contains("distance could not be built"));
}

#[test]
fn the_typed_jobs_invalid_outcome_is_an_error_from_the_spec() {
    // A fixed order of anchors by score cannot be had on a Mahalanobis
    // distance. The spec refuses it before the job runs; the job reports it
    // as a count in an empty outcome.
    let (anchors, candidates) = covariate_inputs();
    let mahalanobis = || DistanceSpec::Mahalanobis {
        covariates: covariates(&["age", "income"]),
        covariance: CovarianceSpec::PooledByDegreesOfFreedom,
        transform: TransformSpec::Raw,
        caliper: None,
    };
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: mahalanobis(),
        nearest: NearestSpec {
            m_order: Some(OrderSpec::Largest),
            mahvars: None,
        },
        ..MatchSpec::default()
    };
    let error = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect_err("refused");
    assert!(error.to_string().contains("nearest.m_order"));

    let prepared = prepare_mahalanobis_distance_config(
        &anchors,
        &candidates,
        None,
        &MahalanobisPreparationConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".into(), "income".into()])
                    .build(),
            )
            .build(),
    )
    .unwrap();
    let outcome = MatchJob::new_standard(&anchors, &candidates, 0)
        .with_distance_config(prepared.distance_config)
        .with_nearest()
        .with_m_order(MatchOrder::Largest)
        .run();
    assert_eq!(outcome.pairs.len(), 0);
    assert!(
        outcome
            .diagnostics
            .exclusion_counts
            .keys()
            .any(|reason| matches!(reason, ExclusionReason::InvalidCriteria(_)))
    );
}

#[test]
fn error_types_display_and_are_errors() {
    fn is_error<E: std::error::Error>(_: &E) {}
    let criteria = CriteriaValidationError::ZeroMatchRatio;
    is_error(&criteria);
    assert!(criteria.to_string().contains("ratio"));
    let negative = CriteriaValidationError::NegativeBirthDateWindow;
    assert!(negative.to_string().contains("window"));
    let estimation = EstimationError::NoCovariatesAvailable;
    is_error(&estimation);
    assert!(estimation.to_string().contains("covariate"));
    let mahalanobis = MahalanobisError::ZeroDimension;
    is_error(&mahalanobis);
    assert!(mahalanobis.to_string().contains("covariates"));
    let spec_error = SpecError::InputMode {
        mode: MatchMode::Standard,
    };
    is_error(&spec_error);
}

#[test]
fn a_configuration_the_matcher_refuses_is_an_error_not_an_empty_match() {
    // A caliper on a distance with no score cannot be combined with another
    // ranking. Validation does not look for it, the matcher refuses the job,
    // and the spec turns that into an error.
    let (anchors, candidates) = covariate_inputs();
    let mahalanobis = |caliper| DistanceSpec::Mahalanobis {
        covariates: covariates(&["age", "income"]),
        covariance: CovarianceSpec::PooledByDegreesOfFreedom,
        transform: TransformSpec::Raw,
        caliper,
    };
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: mahalanobis(Some(5.0)),
        nearest: NearestSpec {
            m_order: None,
            mahvars: Some(mahalanobis(None)),
        },
        birth_window_days: wide_window(),
        ..MatchSpec::default()
    };
    let error = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect_err("refused by the matcher");
    assert!(
        matches!(
            error,
            SpecError::Rejected(ExclusionReason::InvalidCriteria(_))
        ),
        "{error}"
    );
    assert!(error.to_string().contains("nearest_options"));
}

#[test]
fn spec_risk_set_with_resident_check_and_caliper_matches_the_typed_job() {
    let people = cohort(3_000, 100, 8, 31);
    let mut cohort_records = transition_cohort(&people, false);
    for (index, row) in cohort_records.iter_mut().enumerate() {
        row.record
            .core
            .numerics
            .insert("y".into(), (index % 9) as f64);
    }
    let spec = MatchSpec {
        mode: MatchMode::RiskSet,
        seed: 7,
        ratio: 2,
        ratio_fallback: vec![1],
        birth_window_days: 45,
        exact: vec!["sex".into()],
        calipers: BTreeMap::from([("y".to_string(), 4.0)]),
        require_alive: true,
        require_resident: true,
        ..MatchSpec::default()
    };
    let SpecOutput::Pairs { outcome, .. } = run_spec(
        &spec,
        SpecInput::RiskSet {
            cohort: &cohort_records,
        },
    )
    .expect("valid spec") else {
        panic!("expected pairs");
    };
    let by_hand = MatchJob::new_transition(&cohort_records, 7)
        .with_ratio(MatchRatio::new(2).unwrap())
        .with_ratio_fallback(vec![MatchRatio::new(1).unwrap()])
        .with_birth_window(BirthDateWindowDays::new(45).unwrap())
        .with_exact_matches(["sex"])
        .with_constraints(vec![epistat_matching::constraints::caliper_on_field::<
            RoleTransitionRecord<BalanceRecord>,
        >("y", 4.0)])
        .with_alive_check()
        .with_resident_check()
        .run();
    same(&outcome, &by_hand);
}
