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
use epistat_matching::{
    CardinalityError, CardinalityLimits, CardinalityRequest, CbpsConfig, CbpsEstimand, CbpsMethod,
    GeneticFit, GeneticMatchOptions, GeneticProblem, GeneticSearch, QuickMatchOptions,
    match_cardinality, quick_match_subclasses,
};
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
    assert_eq!(hashes(&by_spec), (0x1520dffbb17aef3e, 0x08198f3ecb4ed043));
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
    assert_eq!(hashes(&by_spec), (0xc088ea6cbef91878, 0x74d78bbc0371530c));
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
    assert_eq!(hashes(&by_spec), (0x0182c20be256ce19, 0x4a4e53faf807cda6));
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
    assert_eq!(hashes(&by_spec), (0xd72f85f8c74df990, 0x1a8b1221f7c395a3));
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
    assert_eq!(hashes(&outcome), (0x2c4ad6fd67c312f7, 0x585c1833c1e022cf));
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
    assert_eq!(hashes(&outcome), (0x2bbae3fb40e82c17, 0x790c1ec42bf6ecb7));
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
            ..NearestSpec::default()
        },
        std_caliper: true.into(),
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
            ..NearestSpec::default()
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
            ..NearestSpec::default()
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
    assert!(by_hand.retained_units() > 0);
    assert_eq!(
        format!("{:?}", outcome.assignments_sorted()),
        format!("{:?}", by_hand.assignments_sorted())
    );
    assert_eq!(
        format!("{:?}", outcome.att_weights_sorted()),
        format!("{:?}", by_hand.att_weights_sorted())
    );
    assert_eq!(outcome.dropped_ids(), by_hand.dropped_ids());

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
        &prepared.estimate.score_table().to_map(),
        Estimand::Att,
        &SubclassificationConfig {
            subclasses: 5,
            reference_group: None,
            min_n: 1,
        },
    )
    .expect("valid subclassification settings");
    assert!(by_hand.retained_units() > 0);
    assert_eq!(
        format!("{:?}", outcome.assignments_sorted()),
        format!("{:?}", by_hand.assignments_sorted())
    );
    assert_eq!(
        format!("{:?}", outcome.weights_sorted()),
        format!("{:?}", by_hand.weights_sorted())
    );
}

/// The quick method through a spec equals the function on the same scores, its
/// weights feed the balance report, and its options are validated.
#[test]
fn spec_quick_gives_subclasses_and_weights_for_balance() {
    let (anchors, candidates) = covariate_inputs();
    let spec = MatchSpec {
        method: MatchMethod::Quick,
        distance: DistanceSpec::Propensity {
            covariates: covariates(&["age", "income"]),
            link: LinkSpec::Logit,
            estimator: EstimatorSpec::Glm,
            scale: ScaleSpec::Probability,
            sampling_weights: None,
            caliper: None,
        },
        ..MatchSpec::default()
    };
    let SpecOutput::Quick(outcome) = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect("valid spec") else {
        panic!("expected quick output");
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
    let by_hand = quick_match_subclasses::<_, &str, _>(
        &anchors,
        &candidates,
        |row: &BalanceRecord| {
            prepared
                .estimate
                .position(row.id())
                .map(|position| std::slice::from_ref(&prepared.estimate.distance_scores[position]))
        },
        1,
        &[],
        &QuickMatchOptions::default(),
    )
    .unwrap();
    assert!(outcome.subclasses.len() > 1);
    assert_eq!(outcome.assignments_sorted(), by_hand.assignments_sorted());
    assert_eq!(outcome.subclass, by_hand.subclass);
    assert_eq!(outcome.dropped_ids(), by_hand.dropped_ids());
    assert_eq!(outcome.retained_units(), anchors.len() + candidates.len());

    let weights = BalanceWeights::new(
        Estimand::Att,
        outcome.unit_weights(Estimand::Att).expect("weights"),
    );
    weights.validate().unwrap();
    let report = balance_report_on_weights(
        &anchors,
        &candidates,
        &weights,
        &BalanceReportOptions::default(),
    )
    .unwrap();
    assert!(!report.numeric.is_empty());

    let message = |spec: MatchSpec| {
        run_spec(
            &spec,
            SpecInput::Standard {
                anchors: &anchors,
                candidates: &candidates,
            },
        )
        .unwrap_err()
        .to_string()
    };
    assert!(
        message(MatchSpec {
            method: MatchMethod::Quick,
            ..MatchSpec::default()
        })
        .contains("distance")
    );
    let mut with_size = MatchSpec::default();
    with_size.quick.size = Some(5);
    assert!(message(with_size).contains("quick"));
    assert!(
        message(MatchSpec {
            method: MatchMethod::Quick,
            distance: spec.distance.clone(),
            ratio: 2,
            ..MatchSpec::default()
        })
        .contains("`ratio`")
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
                lambda_min_ratio: Some(0.01),
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
                ..NearestSpec::default()
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
                ..NearestSpec::default()
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
            ..NearestSpec::default()
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
            ..NearestSpec::default()
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

/// Covariate inputs with the region also as a strata key, which coarsened exact
/// matching reads.
fn cem_inputs() -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let (mut anchors, mut candidates) = covariate_inputs();
    for row in anchors.iter_mut().chain(candidates.iter_mut()) {
        if let Some(CovariateValue::Categorical(region)) = row.covariates.get("region") {
            let region = region.clone();
            row.core.strata.insert("region".into(), region);
        }
    }
    (anchors, candidates)
}

fn cem_variables() -> Vec<CemVariableSpec> {
    vec![
        CemVariableSpec::Numeric {
            name: "age".into(),
            binning: BinningSpec::Bins { n: 4 },
        },
        CemVariableSpec::Categorical {
            name: "region".into(),
            groups: vec![],
        },
    ]
}

#[test]
fn spec_cem_matches_the_functions_it_wraps() {
    let (anchors, candidates) = cem_inputs();
    let input = SpecInput::Standard {
        anchors: &anchors,
        candidates: &candidates,
    };
    let by_hand_variables = [
        CemVariable::numeric("age", CemBinning::Bins(4)),
        CemVariable::categorical("region", vec![]),
    ];

    let spec = MatchSpec {
        method: MatchMethod::Cem,
        cem: CemSpec {
            variables: cem_variables(),
            k2k: None,
        },
        ..MatchSpec::default()
    };
    let SpecOutput::Exact(outcome) = run_spec(&spec, input.clone()).expect("valid spec") else {
        panic!("expected strata");
    };
    let by_hand = cem_subclasses(&anchors, &candidates, &by_hand_variables).unwrap();
    assert!(by_hand.retained_units() > 0);
    assert_eq!(
        format!("{:?}", outcome.assignments_sorted()),
        format!("{:?}", by_hand.assignments_sorted())
    );
    assert_eq!(
        format!("{:?}", outcome.ate_weights_sorted()),
        format!("{:?}", by_hand.ate_weights_sorted())
    );

    let spec = MatchSpec {
        method: MatchMethod::Cem,
        cem: CemSpec {
            variables: cem_variables(),
            k2k: Some(K2kSpec {
                distance: Some(DistanceSpec::Mahalanobis {
                    covariates: covariates(&["age", "income"]),
                    covariance: CovarianceSpec::default(),
                    transform: TransformSpec::default(),
                    caliper: None,
                }),
                m_order: Some(OrderSpec::Closest),
            }),
        },
        ..MatchSpec::default()
    };
    let SpecOutput::Pairs { outcome, .. } = run_spec(&spec, input.clone()).expect("valid spec")
    else {
        panic!("expected pairs");
    };
    let strata = cem_strata(&anchors, &candidates, &by_hand_variables).unwrap();
    let prepared = prepare_mahalanobis_distance_config(
        &anchors,
        &candidates,
        None,
        &MahalanobisPreparationConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".into(), "income".into()])
                    .include_intercept(false)
                    .build(),
            )
            .build(),
    )
    .unwrap();
    let by_hand = cem_k2k_pairs(
        &anchors,
        &candidates,
        &strata,
        Some(&prepared.distance_config),
        MatchOrder::Closest,
        0,
        Estimand::Att,
    )
    .unwrap();
    assert!(!by_hand.pairs.is_empty(), "no output, test is vacuous");
    same(&outcome, &by_hand);
}

#[test]
fn spec_nearest_on_a_distance_matrix_matches_the_typed_job() {
    let (anchors, candidates) = covariate_inputs();
    let age = |record: &BalanceRecord| record.core.numerics["age"];
    let rows: Vec<Vec<Option<f64>>> = anchors
        .iter()
        .map(|a| {
            candidates
                .iter()
                .map(|c| {
                    let distance = (age(a) - age(c)).abs();
                    (distance < 6.0).then_some(distance)
                })
                .collect()
        })
        .collect();
    let ids = |records: &[BalanceRecord]| records.iter().map(|r| r.core.id.clone()).collect();
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Matrix {
            anchor_ids: ids(&anchors),
            candidate_ids: ids(&candidates),
            values: rows.clone(),
            caliper: Some(4.0),
        },
        nearest: NearestSpec {
            m_order: Some(OrderSpec::Closest),
            mahvars: None,
            ..NearestSpec::default()
        },
        ratio: 2,
        birth_window_days: wide_window(),
        ..MatchSpec::default()
    };
    let SpecOutput::Pairs { outcome, .. } = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect("valid spec") else {
        panic!("expected pairs");
    };
    let matrix = PairDistanceMatrix::new(
        ids(&anchors),
        ids(&candidates),
        rows.into_iter()
            .flatten()
            .map(|v| v.unwrap_or(f64::INFINITY))
            .collect(),
    )
    .unwrap();
    let by_hand = MatchJob::new_standard(&anchors, &candidates, 0)
        .with_nearest()
        .with_m_order(MatchOrder::Closest)
        .with_ratio(MatchRatio::new(2).unwrap())
        .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
        .with_distance_config(DistanceConfig::matrix(matrix, DistanceCaliper::new(4.0)))
        .run();
    assert!(!by_hand.pairs.is_empty(), "no output, test is vacuous");
    same(&outcome, &by_hand);
}

#[test]
fn invalid_cem_and_matrix_specs_are_errors_that_name_the_option() {
    let defaults = MatchSpec::default;
    let cem = |cem: CemSpec| MatchSpec {
        method: MatchMethod::Cem,
        cem,
        ..defaults()
    };
    assert!(message(&cem(CemSpec::default())).contains("cem.variables"));
    assert!(
        message(&MatchSpec {
            cem: CemSpec {
                variables: cem_variables(),
                k2k: None,
            },
            ..defaults()
        })
        .contains("cem")
    );
    assert!(
        message(&MatchSpec {
            exact: vec!["sex".into()],
            ..cem(CemSpec {
                variables: cem_variables(),
                k2k: None,
            })
        })
        .contains("exact")
    );
    assert!(
        message(&cem(CemSpec {
            variables: cem_variables(),
            k2k: Some(K2kSpec {
                distance: None,
                m_order: Some(OrderSpec::Largest),
            }),
        }))
        .contains("cem.k2k.m_order")
    );
    assert!(
        message(&cem(CemSpec {
            variables: vec![CemVariableSpec::Numeric {
                name: "age".into(),
                binning: BinningSpec::Bins { n: 0 },
            }],
            k2k: None,
        }))
        .contains("0 bins")
    );
    let ragged = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Matrix {
            anchor_ids: vec!["a".into()],
            candidate_ids: vec!["x".into(), "y".into()],
            values: vec![vec![Some(1.0)]],
            caliper: None,
        },
        ..defaults()
    };
    assert!(message(&ragged).contains("one value for each candidate"));
    let scored = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Matrix {
            anchor_ids: vec!["a".into()],
            candidate_ids: vec!["x".into()],
            values: vec![vec![Some(1.0)]],
            caliper: None,
        },
        nearest: NearestSpec {
            m_order: Some(OrderSpec::Largest),
            mahvars: None,
            ..NearestSpec::default()
        },
        ..defaults()
    };
    assert!(message(&scored).contains("nearest.m_order"));
}

#[test]
fn cem_and_matrix_specs_round_trip_through_json() {
    let spec: MatchSpec = serde_json::from_str(
        r#"{"method": "cem", "cem": {"variables": [
              {"kind": "numeric", "name": "age", "binning": {"kind": "quantiles", "n": 4}},
              {"kind": "categorical", "name": "region", "groups": [["r0", "r1"]]}],
            "k2k": {"distance": {"kind": "matrix", "anchor_ids": ["a"],
                                 "candidate_ids": ["x"], "values": [[null]]},
                    "m_order": "farthest"}}}"#,
    )
    .unwrap();
    assert_eq!(spec.method, MatchMethod::Cem);
    assert_eq!(spec.cem.variables.len(), 2);
    let again: MatchSpec = serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
    assert_eq!(spec, again);
}

/// A spec for the optimal and full methods gives what the typed calls give on
/// the same distance, and refuses options the methods do not use.
#[test]
fn optimal_and_full_specs_equal_the_typed_calls() {
    use epistat_matching::{FullMatchConfig, OptimalRequest, match_full, match_optimal};

    let (anchors, candidates) = covariate_inputs();
    let distance = DistanceSpec::Propensity {
        covariates: covariates(&["age", "income"]),
        link: LinkSpec::Logit,
        estimator: EstimatorSpec::Glm,
        scale: ScaleSpec::Probability,
        sampling_weights: None,
        caliper: None,
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
    let input = SpecInput::Standard {
        anchors: &anchors,
        candidates: &candidates,
    };
    let base = MatchSpec {
        distance,
        birth_window_days: 36_500,
        ..MatchSpec::default()
    };
    let criteria = MatchingCriteria::builder()
        .birth_date_window_days(36_500)
        .match_ratio(2)
        .build();
    let request =
        OptimalRequest::new(&criteria, &()).with_distance_config(&prepared.distance_config);

    let spec = MatchSpec {
        method: MatchMethod::Optimal,
        ratio: 2,
        ..base.clone()
    };
    let SpecOutput::Pairs { outcome, .. } = run_spec(&spec, input.clone()).expect("valid spec")
    else {
        panic!("expected pairs");
    };
    let by_hand = match_optimal(&anchors, &candidates, &request).expect("feasible");
    assert_eq!(outcome.pairs, by_hand.pairs);

    let spec = MatchSpec {
        method: MatchMethod::Full,
        full: FullSpec {
            min_controls: 1.0,
            max_controls: Some(4.0),
            ..FullSpec::default()
        },
        ..base.clone()
    };
    let SpecOutput::Full(outcome) = run_spec(&spec, input.clone()).expect("valid spec") else {
        panic!("expected full output");
    };
    let config = FullMatchConfig {
        min_controls: 1.0,
        max_controls: 4.0,
        ..FullMatchConfig::default()
    };
    let by_hand = match_full(&anchors, &candidates, &request, &config).expect("feasible");
    assert_eq!(outcome.assignments, by_hand.assignments);

    for bad in [
        MatchSpec {
            method: MatchMethod::Full,
            ratio: 2,
            ..base.clone()
        },
        MatchSpec {
            method: MatchMethod::Optimal,
            replace: true,
            ..base.clone()
        },
        MatchSpec {
            method: MatchMethod::Optimal,
            full: FullSpec {
                min_controls: 1.0,
                ..FullSpec::default()
            },
            ..base.clone()
        },
    ] {
        assert!(matches!(
            run_spec(&bad, input.clone()),
            Err(SpecError::Option { .. })
        ));
    }
}

fn propensity_spec(caliper: Option<f64>) -> DistanceSpec {
    DistanceSpec::Propensity {
        covariates: covariates(&["age", "income", "region"]),
        link: LinkSpec::Logit,
        estimator: EstimatorSpec::Glm,
        scale: ScaleSpec::Probability,
        sampling_weights: None,
        caliper,
    }
}

fn propensity_config_by_hand(caliper: Option<f64>) -> PropensityScoreConfig {
    PropensityScoreConfig::builder()
        .encoding(
            CovariateEncodingConfig::builder()
                .covariate_keys(vec!["age".into(), "income".into(), "region".into()])
                .include_intercept(true)
                .build(),
        )
        .maybe_caliper(caliper.map(|width| DistanceCaliper::new(width).unwrap()))
        .build()
}

#[test]
fn spec_nearest_with_a_variable_ratio_matches_the_typed_job() {
    let (anchors, candidates) = covariate_inputs();
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: propensity_spec(None),
        nearest: NearestSpec {
            variable_ratio: Some(VariableRatioSpec {
                average: 2.5,
                min: 1,
                max: 4,
            }),
            ..NearestSpec::default()
        },
        birth_window_days: wide_window(),
        seed: 5,
        ..MatchSpec::default()
    };
    let by_spec = run_pairs(&spec, &anchors, &candidates);
    let prepared =
        prepare_propensity_distance_config(&anchors, &candidates, &propensity_config_by_hand(None))
            .unwrap();
    let by_hand = MatchJob::new_standard(&anchors, &candidates, 5)
        .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
        .with_distance_config(prepared.distance_config)
        .with_nearest()
        .with_variable_ratio(VariableRatio::new(2.5, 1, 4).unwrap())
        .run();
    same(&by_spec, &by_hand);
    let counts: std::collections::BTreeSet<usize> = by_spec
        .pairs
        .iter()
        .fold(BTreeMap::<&str, usize>::new(), |mut counts, pair| {
            *counts.entry(pair.anchor_id()).or_default() += 1;
            counts
        })
        .into_values()
        .collect();
    assert!(counts.len() > 1, "the ratio varies between anchors");
}

#[test]
fn spec_discard_matches_the_typed_job_with_common_support() {
    let (anchors, candidates) = covariate_inputs();
    let mut trimmed_somewhere = false;
    for (spec_policy, typed_policy) in [
        (DiscardSpec::Treated, CommonSupport::Treated),
        (DiscardSpec::Control, CommonSupport::Control),
        (DiscardSpec::Both, CommonSupport::Both),
    ] {
        for method in [MatchMethod::Nearest, MatchMethod::LowestIndex] {
            let spec = MatchSpec {
                method,
                distance: propensity_spec(None),
                discard: Some(spec_policy),
                birth_window_days: wide_window(),
                seed: 5,
                ..MatchSpec::default()
            };
            let by_spec = run_pairs(&spec, &anchors, &candidates);
            let prepared = prepare_propensity_distance_config(
                &anchors,
                &candidates,
                &propensity_config_by_hand(None),
            )
            .unwrap();
            let job = MatchJob::new_standard(&anchors, &candidates, 5)
                .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
                .with_common_support(Some(typed_policy))
                .with_distance_config(prepared.distance_config);
            let by_hand = if method == MatchMethod::Nearest {
                job.with_nearest().run()
            } else {
                job.with_strategy(DeterministicSelection).run()
            };
            same(&by_spec, &by_hand);
            trimmed_somewhere |= by_spec.diagnostics.common_support_trimmed_anchors
                + by_spec.diagnostics.common_support_trimmed_candidates
                > 0;
        }
    }
    assert!(trimmed_somewhere, "some policy trims units");
}

#[test]
fn spec_reestimate_matches_estimate_propensity_and_match() {
    let (anchors, candidates) = covariate_inputs();
    for (spec_policy, typed_policy) in [
        (DiscardSpec::Treated, CommonSupport::Treated),
        (DiscardSpec::Both, CommonSupport::Both),
    ] {
        for rounds in [false, true] {
            let spec = MatchSpec {
                method: MatchMethod::LowestIndex,
                distance: propensity_spec(None),
                discard: Some(spec_policy),
                reestimate: true,
                assignment: rounds.then_some(AssignmentSpec::Rounds),
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
            let criteria = MatchingCriteria::builder()
                .match_ratio(2)
                .birth_date_window_days(wide_window())
                .common_support(typed_policy)
                .build();
            let config = PropensityScoreConfig::builder()
                .encoding(
                    CovariateEncodingConfig::builder()
                        .covariate_keys(vec!["age".into(), "income".into(), "region".into()])
                        .include_intercept(true)
                        .build(),
                )
                .reestimate(true)
                .build();
            let by_hand = if rounds {
                estimate_propensity_and_match(
                    &anchors,
                    &candidates,
                    &criteria,
                    Rounds(DeterministicSelection),
                    &[],
                    &(),
                    &config,
                )
            } else {
                estimate_propensity_and_match(
                    &anchors,
                    &candidates,
                    &criteria,
                    DeterministicSelection,
                    &[],
                    &(),
                    &config,
                )
            }
            .unwrap();
            same(&outcome, &by_hand.outcome);
            let propensity = propensity.expect("the refitted model is reported");
            let by_id = |estimate: &PropensityScoreEstimate| {
                estimate
                    .ids
                    .iter()
                    .zip(&estimate.probabilities)
                    .map(|(id, value)| (id.clone(), value.to_bits()))
                    .collect::<BTreeMap<_, _>>()
            };
            assert_eq!(by_id(&propensity), by_id(&by_hand.propensity));
            assert_eq!(
                propensity.discarded_anchors,
                by_hand.propensity.discarded_anchors
            );
            assert_eq!(
                propensity.discarded_candidates,
                by_hand.propensity.discarded_candidates
            );
            assert!(
                !propensity.discarded_anchors.is_empty()
                    || !propensity.discarded_candidates.is_empty(),
                "the policy trims units"
            );
        }
    }
}

#[test]
fn spec_elastic_net_with_a_penalty_ratio_matches_the_typed_job() {
    let (anchors, candidates) = covariate_inputs();
    let estimator = |ratio: Option<f64>| EstimatorSpec::ElasticNetCv {
        alpha: 0.5,
        nfolds: 4,
        nlambda: 20,
        lambda_min_ratio: ratio,
        rule: LambdaSpec::Min,
        standardize: true,
    };
    let mut paths = Vec::new();
    for ratio in [None, Some(0.05)] {
        let spec = MatchSpec {
            method: MatchMethod::Nearest,
            distance: DistanceSpec::Propensity {
                covariates: covariates(&["age", "income", "region"]),
                link: LinkSpec::Logit,
                estimator: estimator(ratio),
                scale: ScaleSpec::Probability,
                sampling_weights: None,
                caliper: None,
            },
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
        let config = PropensityScoreConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".into(), "income".into(), "region".into()])
                    .include_intercept(true)
                    .build(),
            )
            .estimator(PropensityEstimator::ElasticNetCv(
                ElasticNetCvConfig::builder()
                    .seed(5)
                    .alpha(0.5)
                    .nfolds(4)
                    .nlambda(20)
                    .maybe_lambda_min_ratio(ratio)
                    .rule(LambdaRule::Min)
                    .build(),
            ))
            .build();
        let prepared = prepare_propensity_distance_config(&anchors, &candidates, &config).unwrap();
        let by_hand = MatchJob::new_standard(&anchors, &candidates, 5)
            .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
            .with_distance_config(prepared.distance_config)
            .with_nearest()
            .run();
        same(&outcome, &by_hand);
        paths.push(format!("{:?}", propensity.expect("reported").probabilities));
    }
    assert_ne!(paths[0], paths[1], "the penalty ratio changes the fit");
}

#[test]
fn spec_cbps_matches_the_typed_job() {
    let (anchors, candidates) = covariate_inputs();
    for (over, estimand, method, cbps_estimand) in [
        (true, Estimand::Att, CbpsMethod::Over, CbpsEstimand::Att),
        (false, Estimand::Ate, CbpsMethod::Exact, CbpsEstimand::Ate),
    ] {
        let spec = MatchSpec {
            method: MatchMethod::Nearest,
            distance: DistanceSpec::Propensity {
                covariates: covariates(&["age", "income"]),
                link: LinkSpec::Logit,
                estimator: EstimatorSpec::Cbps { over },
                scale: ScaleSpec::Probability,
                sampling_weights: None,
                caliper: None,
            },
            birth_window_days: wide_window(),
            seed: 5,
            estimand,
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
        let config = PropensityScoreConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".into(), "income".into()])
                    .include_intercept(true)
                    .build(),
            )
            .estimator(PropensityEstimator::Cbps(
                CbpsConfig::builder()
                    .estimand(cbps_estimand)
                    .method(method)
                    .build(),
            ))
            .build();
        let prepared = prepare_propensity_distance_config(&anchors, &candidates, &config).unwrap();
        let by_hand = MatchJob::new_standard(&anchors, &candidates, 5)
            .with_estimand(estimand)
            .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
            .with_distance_config(prepared.distance_config)
            .with_nearest()
            .run();
        same(&outcome, &by_hand);
        let reported = propensity.expect("reported");
        assert!(reported.converged);
        assert!(reported.cbps.is_some());
    }
    let json = r#"{"kind": "cbps"}"#;
    assert_eq!(
        serde_json::from_str::<EstimatorSpec>(json).unwrap(),
        EstimatorSpec::Cbps { over: true }
    );
}

/// The covariate balancing propensity score takes its estimand from the spec,
/// and has none for the ATC or the matched sample.
#[test]
fn spec_cbps_refuses_an_estimand_it_does_not_define() {
    let (anchors, candidates) = covariate_inputs();
    let spec = MatchSpec {
        method: MatchMethod::Nearest,
        distance: DistanceSpec::Propensity {
            covariates: covariates(&["age", "income"]),
            link: LinkSpec::Logit,
            estimator: EstimatorSpec::Cbps { over: true },
            scale: ScaleSpec::Probability,
            sampling_weights: None,
            caliper: None,
        },
        birth_window_days: wide_window(),
        estimand: Estimand::Atc,
        ..MatchSpec::default()
    };
    let result = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    );
    assert!(result.is_err());
}

#[test]
fn spec_std_caliper_can_be_set_for_each_caliper() {
    let (anchors, candidates) = covariate_inputs();
    let sd = epistat_matching::constraints::numeric_field_sd(&anchors, &candidates, "age")
        .expect("age varies");
    let field =
        |width: f64| epistat_matching::constraints::caliper_on_field::<BalanceRecord>("age", width);
    let by_hand = |std_distance: bool, field_width: f64| {
        let prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &propensity_config_by_hand(Some(0.3)),
        )
        .unwrap();
        MatchJob::new_standard(&anchors, &candidates, 5)
            .with_ratio(MatchRatio::new(2).unwrap())
            .with_birth_window(BirthDateWindowDays::new(wide_window()).unwrap())
            .with_distance_config(prepared.distance_config)
            .with_constraint(field(field_width))
            .with_nearest()
            .with_std_caliper(std_distance)
            .run()
    };
    let spec = |std_caliper: StdCaliperSpec| MatchSpec {
        method: MatchMethod::Nearest,
        distance: propensity_spec(Some(0.3)),
        calipers: BTreeMap::from([("age".to_string(), 0.4)]),
        std_caliper,
        ratio: 2,
        birth_window_days: wide_window(),
        seed: 5,
        ..MatchSpec::default()
    };
    let each = |distance: bool, age: bool| {
        StdCaliperSpec::Each(StdCaliperEach {
            distance,
            fields: BTreeMap::from([("age".to_string(), age)]),
        })
    };
    let only_distance = run_pairs(&spec(each(true, false)), &anchors, &candidates);
    same(&only_distance, &by_hand(true, 0.4));
    let only_field = run_pairs(&spec(each(false, true)), &anchors, &candidates);
    same(&only_field, &by_hand(false, 0.4 * sd));
    let both = run_pairs(&spec(true.into()), &anchors, &candidates);
    same(&both, &by_hand(true, 0.4 * sd));
    let neither = run_pairs(&spec(false.into()), &anchors, &candidates);
    same(&neither, &by_hand(false, 0.4));
    let rendered: std::collections::BTreeSet<String> =
        [&only_distance, &only_field, &both, &neither]
            .iter()
            .map(|outcome| format!("{:?}", outcome.pairs))
            .collect();
    assert!(
        rendered.len() >= 3,
        "the four readings give different matches"
    );
}

#[test]
fn the_std_caliper_flag_and_the_named_form_both_read_from_json() {
    let flag: MatchSpec = serde_json::from_str(r#"{"std_caliper": true}"#).unwrap();
    assert_eq!(flag.std_caliper, StdCaliperSpec::All(true));
    let named: MatchSpec = serde_json::from_str(
        r#"{"calipers": {"age": 1.0}, "std_caliper": {"distance": true, "fields": {"age": false}}}"#,
    )
    .unwrap();
    assert!(named.std_caliper.distance());
    assert!(!named.std_caliper.field("age"));
    let text = serde_json::to_string(&named).unwrap();
    assert_eq!(serde_json::from_str::<MatchSpec>(&text).unwrap(), named);
    assert!(serde_json::from_str::<MatchSpec>(r#"{"std_caliper": {"dist": true}}"#).is_err());
}

#[test]
fn invalid_support_and_ratio_options_are_errors_that_name_the_option() {
    let (anchors, candidates) = covariate_inputs();
    let run = |spec: MatchSpec| {
        run_spec(
            &spec,
            SpecInput::Standard {
                anchors: &anchors,
                candidates: &candidates,
            },
        )
        .expect_err("invalid spec")
        .to_string()
    };
    let nearest = MatchSpec {
        method: MatchMethod::Nearest,
        distance: propensity_spec(None),
        ..MatchSpec::default()
    };
    let named = |name: &str, text: String| {
        assert!(text.contains(name), "{name} is not named in: {text}");
    };
    named(
        "reestimate",
        run(MatchSpec {
            reestimate: true,
            ..nearest.clone()
        }),
    );
    named(
        "discard",
        run(MatchSpec {
            discard: Some(DiscardSpec::Both),
            distance: DistanceSpec::default(),
            ..nearest.clone()
        }),
    );
    for method in [
        MatchMethod::Exact,
        MatchMethod::Cem,
        MatchMethod::Cardinality,
    ] {
        named(
            "discard",
            run(MatchSpec {
                discard: Some(DiscardSpec::Both),
                method,
                ..nearest.clone()
            }),
        );
    }
    named(
        "std_caliper",
        run(MatchSpec {
            std_caliper: StdCaliperSpec::Each(StdCaliperEach {
                distance: false,
                fields: BTreeMap::from([("age".to_string(), true)]),
            }),
            ..nearest.clone()
        }),
    );
    named(
        "nearest.variable_ratio",
        run(MatchSpec {
            nearest: NearestSpec {
                variable_ratio: Some(VariableRatioSpec {
                    average: 1.0,
                    min: 1,
                    max: 3,
                }),
                ..NearestSpec::default()
            },
            ..nearest.clone()
        }),
    );
    named(
        "ratio",
        run(MatchSpec {
            ratio: 2,
            nearest: NearestSpec {
                variable_ratio: Some(VariableRatioSpec {
                    average: 2.0,
                    min: 1,
                    max: 3,
                }),
                ..NearestSpec::default()
            },
            ..nearest.clone()
        }),
    );
    named(
        "nearest",
        run(MatchSpec {
            nearest: NearestSpec {
                variable_ratio: Some(VariableRatioSpec {
                    average: 2.0,
                    min: 1,
                    max: 3,
                }),
                ..NearestSpec::default()
            },
            method: MatchMethod::Random,
            ..nearest
        }),
    );
}

#[test]
fn a_risk_set_outcome_has_a_long_form_with_a_role_in_each_pair() {
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
    let ids: Vec<String> = cohort_records
        .iter()
        .map(|row| row.id().to_string())
        .collect();
    let units = Units::without_fixed_roles(ids).expect("unique ids");
    let long = get_matches(&outcome, &units, false).expect("long form of a risk-set match");

    let cases: std::collections::BTreeSet<&str> =
        outcome.pairs.iter().map(MatchedPair::anchor_id).collect();
    let anchor_rows = long.role.iter().filter(|r| **r == UnitRole::Anchor).count();
    assert_eq!(anchor_rows, cases.len(), "one pair per case");
    assert_eq!(long.len(), cases.len() + outcome.pairs.len());
    assert_eq!(long.pair_id.last().copied().flatten(), Some(cases.len()));

    // A case can be a control in the pairs of earlier cases.
    let controls: std::collections::BTreeSet<&str> = outcome
        .pairs
        .iter()
        .map(MatchedPair::comparator_id)
        .collect();
    let both: Vec<&str> = cases.intersection(&controls).copied().collect();
    assert!(!both.is_empty(), "some case is also a control");
    let person: &str = both.first().expect("one");
    let roles_of_person: std::collections::BTreeSet<_> = long
        .id
        .iter()
        .zip(&long.role)
        .filter(|(id, _)| id.as_str() == person)
        .map(|(_, role)| format!("{role:?}"))
        .collect();
    assert_eq!(roles_of_person.len(), 2, "both roles for {person}");

    // Each pair is its anchor first, then its controls, weighing 1 together.
    let mut start = 0;
    while start < long.len() {
        let pair = long.pair_id[start];
        let end = (start..long.len())
            .find(|row| long.pair_id[*row] != pair)
            .unwrap_or(long.len());
        assert_eq!(long.role[start], UnitRole::Anchor);
        assert!(
            long.role[start + 1..end]
                .iter()
                .all(|r| *r == UnitRole::Candidate)
        );
        let controls: f64 = long.weight[start + 1..end].iter().sum();
        assert!((controls - 1.0).abs() < 1e-12);
        start = end;
    }
    // The controls of each case are the pairs of the outcome, as a set.
    let mut from_long: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut current = "";
    for (id, role) in long.id.iter().zip(&long.role) {
        if *role == UnitRole::Anchor {
            current = id;
        } else {
            from_long.entry(current).or_default().push(id);
        }
    }
    let mut from_outcome: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for pair in &outcome.pairs {
        from_outcome
            .entry(pair.anchor_id())
            .or_default()
            .push(pair.comparator_id());
    }
    for controls in from_long.values_mut().chain(from_outcome.values_mut()) {
        controls.sort_unstable();
    }
    assert_eq!(from_long, from_outcome);
}

/// A spec with fine balance gives what the typed call gives, balances the
/// categories of the anchors, and is refused with other methods.
#[test]
fn fine_balance_spec_equals_the_typed_call() {
    use epistat_matching::{FineBalance, OptimalRequest, match_fine_balance};

    let (anchors, candidates) = standard_inputs(24);
    let anchors: Vec<_> = anchors.into_iter().take(40).collect();
    let candidates: Vec<_> = candidates.into_iter().take(300).collect();
    let input = SpecInput::Standard {
        anchors: &anchors,
        candidates: &candidates,
    };
    let balance = FineBalance::on_keys(["sex", "municipality"]);
    let spec = MatchSpec {
        method: MatchMethod::Optimal,
        birth_window_days: 36_500,
        ratio: 2,
        fine_balance: Some(balance.clone()),
        ..MatchSpec::default()
    };
    let SpecOutput::Pairs { outcome, .. } = run_spec(&spec, input.clone()).expect("valid spec")
    else {
        panic!("expected pairs");
    };
    let criteria = MatchingCriteria::builder()
        .birth_date_window_days(36_500)
        .match_ratio(2)
        .build();
    let request = OptimalRequest::new(&criteria, &());
    let by_hand = match_fine_balance(&anchors, &candidates, &request, &balance).expect("feasible");
    assert_eq!(outcome.pairs, by_hand.outcome.pairs);
    assert_eq!(by_hand.outcome.pairs.len(), 80);

    let bad = MatchSpec {
        method: MatchMethod::Full,
        fine_balance: Some(balance),
        ..MatchSpec::default()
    };
    assert!(matches!(
        run_spec(&bad, input),
        Err(SpecError::Option { .. })
    ));
    let again: MatchSpec = serde_json::from_str(&serde_json::to_string(&spec).unwrap()).unwrap();
    assert_eq!(spec, again);
}

#[test]
fn spec_discard_in_risk_set_mode_matches_the_typed_job_with_common_support() {
    let people = cohort(1_500, 100, 10, 14);
    let records = transition_cohort(&people, false);
    let scores: BTreeMap<String, f64> = records
        .iter()
        .enumerate()
        .map(|(i, r)| (r.id().to_string(), ((i * 37) % 101) as f64 / 10.0))
        .collect();
    let spec = MatchSpec {
        mode: MatchMode::RiskSet,
        method: MatchMethod::LowestIndex,
        distance: DistanceSpec::Scores {
            scores: scores.clone(),
            caliper: None,
        },
        discard: Some(DiscardSpec::Both),
        seed: 3,
        ratio: 2,
        birth_window_days: 60,
        age_limit_years: Some(6),
        exact: vec!["sex".into()],
        require_alive: true,
        ..MatchSpec::default()
    };
    let SpecOutput::Pairs { outcome, .. } =
        run_spec(&spec, SpecInput::RiskSet { cohort: &records }).expect("valid spec")
    else {
        panic!("expected pairs");
    };
    let by_hand = MatchJob::new_transition(&records, 3)
        .with_ratio(MatchRatio::new(2).unwrap())
        .with_birth_window(BirthDateWindowDays::new(60).unwrap())
        .with_age_limit(AgeLimitYears::new(6).unwrap())
        .with_exact_matches(["sex"])
        .with_alive_check()
        .with_distance_config(DistanceConfig::propensity_score_map(
            scores.into_iter().collect(),
            None,
        ))
        .with_common_support(Some(CommonSupport::Both))
        .with_strategy(DeterministicSelection)
        .run();
    same(&outcome, &by_hand);
    assert!(
        outcome.diagnostics.common_support_trimmed_anchors
            + outcome.diagnostics.common_support_trimmed_candidates
            > 0,
        "nothing was trimmed, test is vacuous"
    );

    // A propensity model is not a distance in risk-set mode.
    let propensity = MatchSpec {
        distance: DistanceSpec::Propensity {
            covariates: CovariatesSpec {
                names: vec!["age".into()],
                missing: MissingSpec::Error,
            },
            link: LinkSpec::Logit,
            estimator: EstimatorSpec::Glm,
            scale: ScaleSpec::Probability,
            sampling_weights: None,
            caliper: None,
        },
        ..spec
    };
    let error = run_spec(&propensity, SpecInput::RiskSet { cohort: &records })
        .expect_err("not available in risk-set mode");
    assert!(error.to_string().contains("discard"), "{error}");
}

fn option_named(error: &SpecError) -> &str {
    match error {
        SpecError::Option { option, .. } => option,
        other => panic!("expected an option error, got {other}"),
    }
}

fn numeric_column(anchors: &[BalanceRecord], candidates: &[BalanceRecord], key: &str) -> Vec<f64> {
    anchors
        .iter()
        .chain(candidates)
        .map(|row| match row.covariates[key] {
            CovariateValue::Numeric(value) => value,
            CovariateValue::Categorical(_) | CovariateValue::Missing => {
                panic!("{key} is numeric")
            }
        })
        .collect()
}

fn age_income_propensity() -> DistanceSpec {
    DistanceSpec::Propensity {
        covariates: covariates(&["age", "income"]),
        link: LinkSpec::Logit,
        estimator: EstimatorSpec::Glm,
        scale: ScaleSpec::Probability,
        sampling_weights: None,
        caliper: None,
    }
}

/// A search that stops after a few small generations, so a test runs quickly.
fn short_genetic() -> GeneticSpec {
    GeneticSpec {
        covariates: covariates(&["age", "income"]),
        pop_size: 10,
        max_generations: 3,
        wait_generations: 2,
        hard_generation_limit: true,
        ..GeneticSpec::default()
    }
}

/// Genetic matching through the spec is the search and the pairs of
/// `GeneticProblem` on the same variables, the score, the order and the seed.
#[test]
fn spec_genetic_equals_the_direct_search() {
    let (anchors, candidates) = covariate_inputs();
    let spec = MatchSpec {
        method: MatchMethod::Genetic,
        distance: age_income_propensity(),
        genetic: short_genetic(),
        seed: 5,
        ..MatchSpec::default()
    };
    let SpecOutput::Genetic { outcome, search } = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect("valid spec") else {
        panic!("expected genetic output");
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
    let ids: Vec<&str> = anchors
        .iter()
        .chain(&candidates)
        .map(|row| row.core.id.as_str())
        .collect();
    let score: Vec<f64> = ids
        .iter()
        .map(|id| prepared.estimate.distance_score(id).unwrap())
        .collect();
    let age = numeric_column(&anchors, &candidates, "age");
    let income = numeric_column(&anchors, &candidates, "income");
    let focal: Vec<bool> = (0..ids.len()).map(|i| i < anchors.len()).collect();
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    let problem = GeneticProblem::new(
        &focal,
        &[age.clone(), income.clone(), score],
        &[age, income],
        &order,
        GeneticMatchOptions::default(),
        f64::EPSILON.sqrt(),
    )
    .unwrap();
    let by_hand = problem
        .search(
            &GeneticFit::default(),
            &GeneticSearch {
                population: 10,
                max_generations: 3,
                wait_generations: 2,
                hard_generation_limit: true,
                seed: 5,
                ..GeneticSearch::default()
            },
        )
        .unwrap();

    assert_eq!(search.weights, by_hand.weights);
    assert_eq!(search.matches, by_hand.matches);
    assert_eq!(search.generations, by_hand.generations);
    let pairs: Vec<(String, String)> = by_hand
        .matches
        .focal
        .iter()
        .zip(&by_hand.matches.control)
        .map(|(f, c)| (ids[*f].to_string(), ids[*c].to_string()))
        .collect();
    let from_spec: Vec<(String, String)> = outcome
        .pairs
        .iter()
        .map(|pair| (pair.case_id.clone(), pair.control_id.clone()))
        .collect();
    assert_eq!(from_spec, pairs);
    assert_eq!(
        outcome.matched_cases + outcome.unmatched_cases,
        anchors.len()
    );
    assert_eq!(
        outcome.matched_cases,
        anchors.len(),
        "60 anchors, 240 candidates"
    );

    // A repeat run is the same, and the seed is read.
    let again = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .unwrap();
    let SpecOutput::Genetic { search: repeat, .. } = again else {
        panic!("expected genetic output");
    };
    assert_eq!(repeat.weights, search.weights);
}

/// Exact keys, calipers on a covariate and antiexact keys reach the problem:
/// every pair agrees on the key, differs on the other and keeps to the caliper.
#[test]
fn spec_genetic_constraints_hold_in_the_pairs() {
    let (mut anchors, mut candidates) = covariate_inputs();
    for (k, row) in anchors.iter_mut().chain(candidates.iter_mut()).enumerate() {
        row.core.strata.insert("g".into(), format!("g{}", k % 2));
        row.core.strata.insert("h".into(), format!("h{}", k % 5));
    }
    let spec = MatchSpec {
        method: MatchMethod::Genetic,
        genetic: short_genetic(),
        exact: vec!["g".into()],
        antiexact: vec!["h".into()],
        calipers: BTreeMap::from([("age".to_string(), 6.0)]),
        ..MatchSpec::default()
    };
    let SpecOutput::Genetic { outcome, .. } = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect("valid spec") else {
        panic!("expected genetic output");
    };
    assert!(outcome.matched_cases > 0);
    let find = |id: &str| {
        anchors
            .iter()
            .chain(&candidates)
            .find(|row| row.core.id == id)
            .unwrap()
    };
    for pair in &outcome.pairs {
        let (a, c) = (find(&pair.case_id), find(&pair.control_id));
        assert_eq!(a.core.strata["g"], c.core.strata["g"]);
        assert_ne!(a.core.strata["h"], c.core.strata["h"]);
        assert!((a.core.numerics["age"] - c.core.numerics["age"]).abs() <= 6.0);
    }
}

#[test]
fn spec_genetic_rejects_invalid_options() {
    let (anchors, candidates) = covariate_inputs();
    let input = SpecInput::Standard {
        anchors: &anchors,
        candidates: &candidates,
    };
    let base = MatchSpec {
        method: MatchMethod::Genetic,
        genetic: short_genetic(),
        ..MatchSpec::default()
    };
    let with = |edit: &dyn Fn(&mut MatchSpec)| {
        let mut spec = base.clone();
        edit(&mut spec);
        spec
    };
    let cases: Vec<(&str, MatchSpec)> = vec![
        ("genetic.pop_size", with(&|s| s.genetic.pop_size = 1)),
        (
            "genetic.max_generations",
            with(&|s| s.genetic.max_generations = 0),
        ),
        (
            "genetic.wait_generations",
            with(&|s| s.genetic.wait_generations = 0),
        ),
        ("genetic.tolerance", with(&|s| s.genetic.tolerance = 0.0)),
        (
            "genetic.m_order",
            with(&|s| s.genetic.m_order = Some(OrderSpec::Largest)),
        ),
        (
            "genetic.m_order",
            with(&|s| s.genetic.m_order = Some(OrderSpec::Closest)),
        ),
        ("estimand", with(&|s| s.estimand = Estimand::Atc)),
        (
            "distance",
            with(&|s| {
                s.distance = DistanceSpec::Euclidean {
                    covariates: CovariatesSpec::default(),
                    scaled: false,
                    caliper: None,
                }
            }),
        ),
        (
            "unique_by_key",
            with(&|s| s.unique_by_key = Some("k".into())),
        ),
        ("birth_window_days", with(&|s| s.birth_window_days = 10)),
        ("discard", with(&|s| s.discard = Some(DiscardSpec::Both))),
        ("method", with(&|s| s.mode = MatchMode::RiskSet)),
        (
            "calipers",
            with(&|s| {
                s.calipers.insert("region".into(), 1.0);
            }),
        ),
        ("exact", with(&|s| s.exact = vec!["nokey".into()])),
        (
            "genetic",
            MatchSpec {
                method: MatchMethod::Random,
                genetic: short_genetic(),
                ..MatchSpec::default()
            },
        ),
    ];
    for (option, spec) in cases {
        let error = run_spec(&spec, input.clone()).expect_err(option);
        assert_eq!(option_named(&error), option, "{error}");
    }
    // Covariates that cannot be encoded are the distance's error.
    let missing = with(&|s| s.genetic.covariates = covariates(&["nosuch"]));
    assert!(run_spec(&missing, input).is_err());
}

/// Genetic matching through the spec reaches `estimate_effects`, and the
/// balance report takes its pairs.
#[test]
fn spec_genetic_feeds_effects_and_balance() {
    use epistat_matching::effects::{
        EffectSpec, MatchedSample, OutcomeModel, UnitData, estimate_effects,
    };
    let (anchors, candidates) = covariate_inputs();
    let spec = MatchSpec {
        method: MatchMethod::Genetic,
        distance: age_income_propensity(),
        genetic: short_genetic(),
        ..MatchSpec::default()
    };
    let SpecOutput::Genetic { outcome, .. } = run_spec(
        &spec,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .unwrap() else {
        panic!("expected genetic output");
    };
    let ids: Vec<String> = anchors
        .iter()
        .chain(&candidates)
        .map(|row| row.core.id.clone())
        .collect();
    let treated: Vec<bool> = (0..ids.len()).map(|i| i < anchors.len()).collect();
    let units = Units::new(ids.clone(), treated.clone()).unwrap();
    let data = match_data(&outcome, &units, MatchDataOptions::default()).unwrap();
    assert_eq!(data.len(), 2 * anchors.len(), "60 pairs, one control each");
    let age = numeric_column(&anchors, &candidates, "age");
    let y: Vec<f64> = treated
        .iter()
        .zip(&age)
        .map(|(t, a)| 3.0 * f64::from(u8::from(*t)) + 0.25 * a)
        .collect();
    let unit_data = UnitData::new(ids, treated, y)
        .unwrap()
        .with_covariate("age", age)
        .unwrap();
    let sample = MatchedSample::from_match_data(&data, &unit_data).unwrap();
    let estimates = estimate_effects(&sample, &EffectSpec::new(OutcomeModel::Linear)).unwrap();
    assert!(
        (estimates.difference.estimate - 3.0).abs() < 1e-9,
        "{}",
        estimates.difference.estimate
    );

    let weights = BalanceWeights::from_outcome(&outcome);
    weights.validate().unwrap();
    let report = balance_report_on_weights(
        &anchors,
        &candidates,
        &weights,
        &BalanceReportOptions::default(),
    )
    .unwrap();
    assert!(!report.numeric.is_empty());
}

fn cardinality_spec() -> MatchSpec {
    MatchSpec {
        method: MatchMethod::Cardinality,
        cardinality: CardinalitySpec {
            covariates: covariates(&["age", "income"]),
            tols: vec![0.25],
            time: None,
            nodes: Some(2_000),
            ..CardinalitySpec::default()
        },
        ..MatchSpec::default()
    }
}

/// Cardinality matching through the spec is `match_cardinality` on the same
/// columns and limits, in strata and without, with pairing and without.
#[test]
fn spec_cardinality_equals_the_direct_call() {
    let (mut anchors, mut candidates) = covariate_inputs();
    for (k, row) in anchors.iter_mut().chain(candidates.iter_mut()).enumerate() {
        row.core.strata.insert("g".into(), format!("g{}", k % 2));
    }
    let treated: Vec<bool> = (0..anchors.len() + candidates.len())
        .map(|i| i < anchors.len())
        .collect();
    let age = numeric_column(&anchors, &candidates, "age");
    let income = numeric_column(&anchors, &candidates, "income");
    let columns = vec![age, income];
    let input = SpecInput::Standard {
        anchors: &anchors,
        candidates: &candidates,
    };
    let limits = CardinalityLimits {
        time: None,
        nodes: Some(2_000),
    };

    // One stratum, 1:1.
    let SpecOutput::Cardinality(found) = run_spec(&cardinality_spec(), input.clone()).unwrap()
    else {
        panic!("expected cardinality output");
    };
    let mut request = CardinalityRequest::new(&treated, &columns);
    request.tols = vec![0.25];
    request.limits = limits;
    let by_hand = match_cardinality(&request).unwrap();
    assert_eq!(found.outcome.weights, by_hand.weights);
    assert_eq!(found.outcome.selected, by_hand.selected);
    assert_eq!(found.outcome.strata, by_hand.strata);
    assert!(found.outcome.selected.iter().any(|s| *s));
    assert_eq!(found.ids.len(), treated.len());
    assert_eq!(found.anchor, treated);

    // Exact strata, 1:2, for the effect in the whole sample, paired.
    let mut spec = cardinality_spec();
    spec.exact = vec!["g".into()];
    spec.estimand = Estimand::Ate;
    spec.cardinality.ratio = Some(2.0);
    spec.cardinality.tols = vec![0.5];
    spec.cardinality.mahvars = vec!["age".into(), "income".into()];
    let SpecOutput::Cardinality(found) = run_spec(&spec, input.clone()).unwrap() else {
        panic!("expected cardinality output");
    };
    let exact: Vec<u64> = (0..treated.len()).map(|k| (k % 2) as u64).collect();
    let mut request = CardinalityRequest::new(&treated, &columns);
    request.estimand = Estimand::Ate;
    request.ratio = Some(2.0);
    request.tols = vec![0.5];
    request.exact = Some(&exact);
    request.pair_on = &columns;
    request.limits = limits;
    let by_hand = match_cardinality(&request).unwrap();
    assert_eq!(found.outcome.weights, by_hand.weights);
    assert_eq!(found.outcome.pairs, by_hand.pairs);
    assert!(found.outcome.pairs.is_some());
    let long = found.get_matches(false).expect("a paired selection");
    assert!(!long.is_empty());
}

/// Profile matching, sampling weights and the errors of the method.
#[test]
fn spec_cardinality_profile_and_errors() {
    let (anchors, candidates) = covariate_inputs();
    let input = SpecInput::Standard {
        anchors: &anchors,
        candidates: &candidates,
    };
    let mut profile = cardinality_spec();
    profile.cardinality.ratio = None;
    profile.cardinality.tols = vec![0.5];
    profile.cardinality.sampling_weights = Some(
        anchors
            .iter()
            .chain(&candidates)
            .map(|row| (row.core.id.clone(), 1.0))
            .collect(),
    );
    let SpecOutput::Cardinality(found) = run_spec(&profile, input.clone()).unwrap() else {
        panic!("expected cardinality output");
    };
    // The treated are all kept for the effect in the treated.
    let kept_anchors = found
        .outcome
        .selected
        .iter()
        .zip(&found.anchor)
        .filter(|(selected, anchor)| **selected && **anchor)
        .count();
    assert_eq!(kept_anchors, anchors.len());
    assert!(
        found.get_matches(false).is_err(),
        "no pairs without mahvars"
    );

    let base = cardinality_spec();
    let with = |edit: &dyn Fn(&mut MatchSpec)| {
        let mut spec = base.clone();
        edit(&mut spec);
        spec
    };
    let cases: Vec<(&str, MatchSpec)> = vec![
        ("ratio", with(&|s| s.ratio = 2)),
        ("replace", with(&|s| s.replace = true)),
        ("estimand", with(&|s| s.estimand = Estimand::Atm)),
        ("distance", with(&|s| s.distance = age_income_propensity())),
        (
            "calipers",
            with(&|s| {
                s.calipers.insert("age".into(), 1.0);
            }),
        ),
        (
            "cardinality.ratio",
            with(&|s| s.cardinality.ratio = Some(-1.0)),
        ),
        (
            "cardinality.tols",
            with(&|s| s.cardinality.tols = Vec::new()),
        ),
        (
            "cardinality.tols",
            with(&|s| s.cardinality.tols = vec![-0.1]),
        ),
        (
            "cardinality.std_tols",
            with(&|s| s.cardinality.std_tols = Vec::new()),
        ),
        (
            "cardinality.time",
            with(&|s| s.cardinality.time = Some(0.0)),
        ),
        (
            "cardinality.nodes",
            with(&|s| s.cardinality.nodes = Some(0)),
        ),
        (
            "cardinality.sampling_weights",
            with(&|s| s.cardinality.sampling_weights = Some(BTreeMap::new())),
        ),
        (
            "cardinality.mahvars",
            with(&|s| {
                s.cardinality.ratio = Some(1.5);
                s.cardinality.mahvars = vec!["age".into()];
            }),
        ),
        ("method", with(&|s| s.mode = MatchMode::RiskSet)),
        ("exact", with(&|s| s.exact = vec!["nokey".into()])),
        (
            "cardinality",
            MatchSpec {
                cardinality: cardinality_spec().cardinality,
                ..MatchSpec::default()
            },
        ),
    ];
    for (option, spec) in cases {
        let error = run_spec(&spec, input.clone()).expect_err(option);
        assert_eq!(option_named(&error), option, "{error}");
    }

    // A tolerance that no subset meets is the solver's error.
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (i, (value, treated)) in [(0.0, true), (0.0, true), (10.0, false), (10.0, false)]
        .into_iter()
        .enumerate()
    {
        let mut row = BalanceRecord {
            core: BaseRecord::new(
                format!("u{i}"),
                chrono::NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
            ),
            covariates: std::collections::HashMap::new(),
        };
        row.covariates
            .insert("x".into(), CovariateValue::Numeric(value));
        if treated {
            anchors.push(row);
        } else {
            candidates.push(row);
        }
    }
    let infeasible = MatchSpec {
        method: MatchMethod::Cardinality,
        estimand: Estimand::Ate,
        cardinality: CardinalitySpec {
            covariates: covariates(&["x"]),
            ratio: None,
            tols: vec![0.1],
            std_tols: vec![false],
            time: None,
            ..CardinalitySpec::default()
        },
        ..MatchSpec::default()
    };
    let error = run_spec(
        &infeasible,
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .expect_err("no subset meets the tolerance");
    assert!(
        matches!(
            error,
            SpecError::Cardinality(CardinalityError::Infeasible { key: 0 })
        ),
        "{error}"
    );
}

/// Cardinality matching through the spec reaches `estimate_effects` through
/// `match_data`, and the balance report takes its weights.
#[test]
fn spec_cardinality_feeds_effects_and_balance() {
    use epistat_matching::effects::{
        Clustering, EffectSpec, MatchedSample, OutcomeModel, UnitData, estimate_effects,
    };
    let (anchors, candidates) = covariate_inputs();
    let SpecOutput::Cardinality(found) = run_spec(
        &cardinality_spec(),
        SpecInput::Standard {
            anchors: &anchors,
            candidates: &candidates,
        },
    )
    .unwrap() else {
        panic!("expected cardinality output");
    };
    let data = found.match_data(MatchDataOptions::default()).unwrap();
    let selected = found.outcome.selected.iter().filter(|s| **s).count();
    assert_eq!(data.len(), selected);
    assert!(data.weight.iter().all(|w| *w > 0.0));
    let age = numeric_column(&anchors, &candidates, "age");
    let y: Vec<f64> = found
        .anchor
        .iter()
        .zip(&age)
        .map(|(t, a)| 3.0 * f64::from(u8::from(*t)) + 0.25 * a)
        .collect();
    let unit_data = UnitData::new(found.ids.clone(), found.anchor.clone(), y)
        .unwrap()
        .with_covariate("age", age)
        .unwrap();
    let sample = MatchedSample::from_match_data(&data, &unit_data).unwrap();
    // A selection that is not paired has no pairs to cluster by.
    let effect = EffectSpec {
        clustering: Clustering::Unit,
        ..EffectSpec::new(OutcomeModel::Linear)
    };
    let estimates = estimate_effects(&sample, &effect).unwrap();
    assert!(
        (estimates.difference.estimate - 3.0).abs() < 1e-9,
        "{}",
        estimates.difference.estimate
    );

    let weights = BalanceWeights::from_unit_weights(
        &anchors,
        &candidates,
        &found.unit_weights(),
        Estimand::Att,
    );
    weights.validate().unwrap();
    let report = balance_report_on_weights(
        &anchors,
        &candidates,
        &weights,
        &BalanceReportOptions::default(),
    )
    .unwrap();
    assert!(!report.numeric.is_empty());
}

/// The new methods read from JSON with MatchIt's names, and an explicit null
/// ratio is profile matching.
#[test]
fn genetic_and_cardinality_specs_read_from_json() {
    let spec: MatchSpec = serde_json::from_str(
        r#"{"method": "genetic",
            "genetic": {"pop_size": 20, "fit_func": "qqmean.mean", "loss": "minimum"}}"#,
    )
    .unwrap();
    assert_eq!(spec.method, MatchMethod::Genetic);
    assert_eq!(spec.genetic.pop_size, 20);
    assert_eq!(spec.genetic.fit_func, FitFuncSpec::QqMeanMean);
    assert_eq!(spec.genetic.loss, LossSpec::Minimum);
    assert_eq!(spec.genetic.wait_generations, 4);
    let spec: MatchSpec = serde_json::from_str(
        r#"{"method": "cardinality", "cardinality": {"ratio": null, "tols": [0.1, 0.2]}}"#,
    )
    .unwrap();
    assert_eq!(spec.cardinality.ratio, None);
    assert_eq!(spec.cardinality.std_tols, vec![true]);
    assert_eq!(spec.cardinality.time, Some(120.0));
    let text = serde_json::to_string(&spec).unwrap();
    assert_eq!(serde_json::from_str::<MatchSpec>(&text).unwrap(), spec);
    assert!(serde_json::from_str::<MatchSpec>(r#"{"genetic": {"popsize": 1}}"#).is_err());
}
