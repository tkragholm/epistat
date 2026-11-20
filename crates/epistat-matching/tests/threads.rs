//! Every public entry point that computes something gives bit-identical
//! output in rayon pools of 1, 2, 3, 4 and 8 threads.
//!
//! Each check reduces the whole output to a list of 64-bit values: pairs in
//! order, counts, weights, every balance statistic and every exclusion count.
//! Text and integers are folded into FNV-1a hashes and floats are taken by
//! their bits, so two outputs agree only when they agree in every bit. Hash
//! maps are sorted by key first, since their iteration order is not part of
//! the result. The inputs hold a few thousand units, enough for rayon to split
//! the work across the largest pool.

mod common;

use chrono::NaiveDate;
use common::{Level, cohort, records, two_pass};
use epistat_matching::prelude::*;
use epistat_matching::{
    DeterministicSelection, NearestBirthDateSelection, ReconstructedEpisode,
    ReconstructedMatchingOptions, SubclassificationConfig, balance_report_with_options,
    estimate_propensity_and_match, prepare_mahalanobis_distance_config,
    prepare_propensity_distance_config, reconstruct_case_control_pairs,
    subclassify_by_propensity_score_map,
};
use epistat_testkit::{POOLS, assert_same_on_pools};
use std::collections::HashMap;

/// A list of 64-bit values read back as floats so that the pool check can
/// compare them by bits.
#[derive(Default)]
struct Digest(Vec<f64>);

impl Digest {
    fn bits(&mut self, bits: u64) {
        self.0.push(f64::from_bits(bits));
    }

    fn float(&mut self, value: f64) {
        self.bits(value.to_bits());
    }

    fn int(&mut self, value: usize) {
        self.bits(value as u64);
    }

    /// Folds the text into one value (FNV-1a).
    fn text(&mut self, text: &str) {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in text.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        self.bits(hash);
    }

    /// Folds a value through its debug text, which shows every float digit.
    fn debug(&mut self, value: &impl std::fmt::Debug) {
        self.text(&format!("{value:?}"));
    }

    fn floats(&mut self, values: &HashMap<String, f64, impl std::hash::BuildHasher>) {
        let mut entries: Vec<_> = values.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        self.int(entries.len());
        for (key, value) in entries {
            self.text(key);
            self.float(*value);
        }
    }

    fn outcome(&mut self, outcome: &MatchOutcome) {
        self.debug(outcome);
        self.int(outcome.pairs.len());
        self.int(outcome.matched_cases);
        self.int(outcome.unmatched_cases);
        self.int(outcome.used_controls);
        self.float(outcome.avg_controls_per_case);
        for (reason, count) in &outcome.diagnostics.exclusion_counts {
            self.debug(reason);
            self.int(*count);
        }
    }

    fn weights(&mut self, outcome: &MatchOutcome) {
        for method in [
            MatchWeightMethod::PairCount,
            MatchWeightMethod::AnchorUnitCandidateFractional,
            MatchWeightMethod::CandidateUnitAnchorFractional,
        ] {
            let unit = match_weights_from_pairs(&outcome.pairs, method);
            self.floats(&unit.anchor);
            self.floats(&unit.candidate);
            self.debug(&unit.to_table());
            self.debug(&pair_weights_from_pairs(&outcome.pairs, method).to_table());
        }
    }

    fn estimate(&mut self, estimate: &PropensityScoreEstimate) {
        self.floats(&estimate.probabilities);
        self.floats(&estimate.linear_predictor);
        self.floats(&estimate.distance_scores);
        for coefficient in &estimate.coefficients {
            self.float(*coefficient);
        }
        self.debug(&estimate.column_names);
        self.text(estimate.estimator);
        self.int(usize::from(estimate.converged));
        self.int(estimate.iterations);
        self.debug(&estimate.ridge);
    }
}

fn date(offset: i64) -> NaiveDate {
    NaiveDate::from_ymd_opt(2005, 1, 1).unwrap() + chrono::TimeDelta::days(offset)
}

/// Anchors and candidates with strata, a numeric and a categorical covariate
/// that differ between the groups, and a few missing values.
fn balance_inputs(
    anchors: usize,
    candidates: usize,
    seed: u64,
) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let mut rng = common::Lcg::new(seed);
    let mut make = |prefix: &str, i: usize, exposed: bool| {
        let shift = if exposed { 0.6 } else { 0.0 };
        let age = 30.0 + rng.below(400) as f64 / 10.0 + shift * 5.0;
        let income = (rng.below(1000) as f64 / 100.0 + shift).ln_1p();
        let education = match rng.below(if exposed { 3 } else { 4 }) {
            0 => "basic",
            1 => "upper",
            2 => "tertiary",
            _ => "other",
        };
        let missing_income = rng.below(40) == 0;
        let mut record = BalanceRecord::builder(
            format!("{prefix}{i:06}"),
            date(i64::try_from(rng.below(900)).unwrap()),
        )
        .numeric("age", age)
        .numeric("income", if missing_income { None } else { Some(income) })
        .categorical("education", Some(education.to_string()))
        .build();
        record
            .core
            .strata
            .insert("sex".to_string(), (rng.below(2) + 1).to_string());
        record
            .core
            .strata
            .insert("region".to_string(), format!("r{}", rng.below(8)));
        record
    };
    let a = (0..anchors).map(|i| make("a", i, true)).collect();
    let c = (0..candidates).map(|i| make("c", i, false)).collect();
    (a, c)
}

fn criteria(ratio: usize, replacement: bool) -> MatchingCriteria {
    MatchingCriteria::builder()
        .birth_date_window_days(45)
        .match_ratio(ratio)
        .required_strata(vec!["sex".to_string(), "region".to_string()])
        .allow_replacement(replacement)
        .build()
}

fn base_pair(
    n_anchors: usize,
    n_candidates: usize,
    seed: u64,
) -> (Vec<BaseRecord>, Vec<BaseRecord>) {
    let (a, c) = balance_inputs(n_anchors, n_candidates, seed);
    (
        a.into_iter().map(|r| r.core).collect(),
        c.into_iter().map(|r| r.core).collect(),
    )
}

#[test]
fn standard_matching_and_weights() {
    let (anchors, candidates) = base_pair(1_500, 6_000, 1);
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        for replacement in [false, true] {
            let job = || {
                MatchJob::new_standard(&anchors, &candidates, 11)
                    .with_ratio(MatchRatio::new(3).unwrap())
                    .with_ratio_fallback(vec![MatchRatio::new(1).unwrap()])
                    .with_birth_window(BirthDateWindowDays::new(45).unwrap())
                    .with_exact_matches(["sex", "region"])
                    .with_replacement(replacement)
            };
            for outcome in [
                job().run(),
                job().with_strategy(NearestBirthDateSelection).run(),
                job().with_strategy(DeterministicSelection).run(),
            ] {
                assert!(!outcome.pairs.is_empty(), "no output, test is vacuous");
                d.outcome(&outcome);
                d.weights(&outcome);
            }
        }
        d.0
    });
}

#[test]
fn standard_matching_with_unique_key_and_without_strata() {
    let (mut anchors, mut candidates) = base_pair(1_000, 4_000, 2);
    for (i, r) in anchors.iter_mut().chain(candidates.iter_mut()).enumerate() {
        r.unique_key = Some(format!("u{}", i % 1_500));
    }
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        let out = MatchJob::new_standard(&anchors, &candidates, 5)
            .with_ratio(MatchRatio::new(2).unwrap())
            .with_birth_window(BirthDateWindowDays::new(10).unwrap())
            .run();
        assert!(!out.pairs.is_empty(), "no output, test is vacuous");
        d.outcome(&out);
        let out = MatchJob::new_standard(&anchors, &candidates, 5)
            .with_ratio(MatchRatio::new(2).unwrap())
            .with_birth_window(BirthDateWindowDays::new(10).unwrap())
            .with_exact_matches(["sex", "region"])
            .with_unique_by_key(Some("unique".to_string()))
            .run();
        d.outcome(&out);
        d.0
    });
}

#[test]
fn transition_matching() {
    let people = cohort(6_000, 90, 12, 31);
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        let recs = records(&people, Level::Municipality, true, false);
        let job = || {
            MatchJob::new_transition(&recs, 99)
                .with_ratio(MatchRatio::new(3).unwrap())
                .with_birth_window(BirthDateWindowDays::new(60).unwrap())
                .with_age_limit(AgeLimitYears::new(6).unwrap())
                .with_exact_matches(["sex", "municipality"])
                .with_alive_check()
        };
        let out = job().run();
        assert!(!out.pairs.is_empty(), "no output, test is vacuous");
        d.outcome(&out);
        d.weights(&out);
        let out = job().with_strategy(NearestBirthDateSelection).run();
        d.outcome(&out);
        let out = job().with_strategy(DeterministicSelection).run();
        d.outcome(&out);
        let (pass1, pass2) = two_pass(&people, 7, true);
        d.outcome(&pass1);
        d.outcome(&pass2);
        d.0
    });
}

#[test]
fn run_with_balance_and_balance_reports() {
    let (cases, controls) = balance_inputs(800, 3_200, 3);
    let anchors: Vec<BaseRecord> = cases.iter().map(|r| r.core.clone()).collect();
    let candidates: Vec<BaseRecord> = controls.iter().map(|r| r.core.clone()).collect();
    let mut supplemental = SupplementalBalanceCovariates::default();
    for (i, r) in cases.iter().enumerate() {
        supplemental.cases.insert(
            r.core.id.clone(),
            HashMap::from([(
                "score".to_string(),
                CovariateValue::Numeric(i as f64 % 17.0),
            )]),
        );
    }
    for (i, r) in controls.iter().enumerate() {
        supplemental.controls.insert(
            r.core.id.clone(),
            HashMap::from([
                (
                    "score".to_string(),
                    CovariateValue::Numeric(i as f64 % 13.0),
                ),
                (
                    "group".to_string(),
                    CovariateValue::Categorical(format!("g{}", i % 4)),
                ),
            ]),
        );
    }
    let first = MatchJob::new_standard(&anchors, &candidates, 3)
        .with_ratio(MatchRatio::new(2).unwrap())
        .with_birth_window(BirthDateWindowDays::new(45).unwrap())
        .with_exact_matches(["sex", "region"])
        .run();

    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        let job = || {
            MatchJob::new_standard(&anchors, &candidates, 3)
                .with_ratio(MatchRatio::new(2).unwrap())
                .with_birth_window(BirthDateWindowDays::new(45).unwrap())
                .with_exact_matches(["sex", "region"])
        };
        let (outcome, report) = job().run_with_balance(&cases, &controls);
        d.outcome(&outcome);
        d.debug(&report);
        let (outcome, report) = job()
            .with_strategy(NearestBirthDateSelection)
            .run_with_balance(&cases, &controls);
        d.outcome(&outcome);
        d.debug(&report);
        for transform in [
            NumericBalanceTransform::None,
            NumericBalanceTransform::Squares,
            NumericBalanceTransform::SquaresAndPairwiseInteractions,
        ] {
            for with_supplement in [false, true] {
                let options = BalanceReportOptions::builder()
                    .numeric_transforms(transform)
                    .supplemental_covariates(if with_supplement {
                        supplemental.clone()
                    } else {
                        SupplementalBalanceCovariates::default()
                    })
                    .build();
                let report = balance_report_with_options(&cases, &controls, &first, &options);
                assert!(!report.numeric.is_empty(), "no output, test is vacuous");
                d.debug(&report);
            }
        }
        let diagnostics = balance_diagnostics(
            &anchors,
            &candidates,
            &first,
            &["sex".to_string(), "region".to_string()],
        );
        d.float(diagnostics.match_rate);
        d.float(diagnostics.avg_controls_per_case);
        let mut strata: Vec<_> = diagnostics.strata_counts.iter().collect();
        strata.sort();
        d.debug(&strata);
        d.0
    });
}

#[test]
fn transition_run_with_balance() {
    let people = cohort(5_000, 100, 10, 32);
    let balance: Vec<BalanceRecord> = people
        .iter()
        .map(|p| {
            BalanceRecord::builder(p.id.clone(), p.birth)
                .numeric("parent_year", p.parent_year)
                .categorical("municipality", Some(p.municipality.clone()))
                .build()
        })
        .collect();
    let (cases, controls): (Vec<_>, Vec<_>) = {
        let mut cases = Vec::new();
        let mut controls = Vec::new();
        for (p, b) in people.iter().zip(balance) {
            if p.exposed {
                cases.push(b);
            } else {
                controls.push(b);
            }
        }
        (cases, controls)
    };
    assert_same_on_pools(&POOLS, || {
        let recs = records(&people, Level::Municipality, false, false);
        let (out, report) = MatchJob::new_transition(&recs, 9)
            .with_ratio(MatchRatio::new(2).unwrap())
            .with_birth_window(BirthDateWindowDays::new(60).unwrap())
            .with_age_limit(AgeLimitYears::new(6).unwrap())
            .with_exact_matches(["sex", "municipality"])
            .with_alive_check()
            .run_with_balance(&cases, &controls);
        assert!(!out.pairs.is_empty(), "no output, test is vacuous");
        let mut d = Digest::default();
        d.outcome(&out);
        d.debug(&report);
        d.0
    });
}

#[test]
fn propensity_estimation_and_matching() {
    let (anchors, candidates) = balance_inputs(600, 2_400, 4);
    let elastic = PropensityEstimator::ElasticNetLogit(ElasticNetLogisticConfig::default());
    for (name, estimator, scale) in [
        (
            "logit",
            PropensityEstimator::GlmLogit,
            PropensityScoreOutputScale::Probability,
        ),
        (
            "logit_lp",
            PropensityEstimator::GlmLogit,
            PropensityScoreOutputScale::LinearPredictor,
        ),
        ("elastic", elastic, PropensityScoreOutputScale::Probability),
    ] {
        let config = PropensityScoreConfig::builder()
            .estimator(estimator)
            .output_scale(scale)
            .encoding(
                CovariateEncodingConfig::builder()
                    .include_intercept(true)
                    .missing_value_policy(MissingValuePolicy::Impute)
                    .build(),
            )
            .build();
        for replacement in [false, true] {
            assert_same_on_pools(&POOLS, || {
                let mut d = Digest::default();
                let prepared =
                    prepare_propensity_distance_config(&anchors, &candidates, &config).unwrap();
                d.estimate(&prepared.estimate);
                let matched = estimate_propensity_and_match(
                    &anchors,
                    &candidates,
                    &criteria(2, replacement),
                    NearestBirthDateSelection,
                    &[MatchRatio::new(1).unwrap()],
                    &(),
                    &config,
                )
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
                assert!(!matched.outcome.pairs.is_empty(), "{name}");
                d.outcome(&matched.outcome);
                d.weights(&matched.outcome);
                d.estimate(&matched.propensity);
                let outcome = MatchJob::new_standard(&anchors, &candidates, 8)
                    .with_ratio(MatchRatio::new(2).unwrap())
                    .with_birth_window(BirthDateWindowDays::new(45).unwrap())
                    .with_exact_matches(["sex", "region"])
                    .with_distance_config(prepared.distance_config)
                    .run();
                d.outcome(&outcome);
                d.0
            });
        }
    }
}

#[test]
fn mahalanobis_preparation_and_matching() {
    let (anchors, candidates) = balance_inputs(600, 2_400, 5);
    for strategy in [
        MahalanobisCovarianceStrategy::PooledWithinGroups,
        MahalanobisCovarianceStrategy::FullSample,
    ] {
        for transform in [MahalanobisTransform::Raw, MahalanobisTransform::Rank] {
            let config = MahalanobisPreparationConfig::builder()
                .covariance_strategy(strategy)
                .transform(transform)
                .encoding(
                    CovariateEncodingConfig::builder()
                        .missing_value_policy(MissingValuePolicy::Impute)
                        .build(),
                )
                .build();
            assert_same_on_pools(&POOLS, || {
                let mut d = Digest::default();
                let prepared =
                    prepare_mahalanobis_distance_config(&anchors, &candidates, None, &config)
                        .unwrap();
                let mut vectors: Vec<_> = prepared.vectors.iter().collect();
                vectors.sort_by(|a, b| a.0.cmp(b.0));
                for (id, vector) in vectors {
                    d.text(id);
                    for v in vector {
                        d.float(*v);
                    }
                }
                for v in &prepared.inverse_covariance {
                    d.float(*v);
                }
                d.int(prepared.dimension);
                d.debug(&prepared.column_names);
                d.debug(&prepared.ridge);
                let outcome = MatchJob::new_standard(&anchors, &candidates, 8)
                    .with_ratio(MatchRatio::new(2).unwrap())
                    .with_birth_window(BirthDateWindowDays::new(45).unwrap())
                    .with_exact_matches(["sex", "region"])
                    .with_distance_config(prepared.distance_config)
                    .run();
                assert!(!outcome.pairs.is_empty(), "no output, test is vacuous");
                d.outcome(&outcome);
                d.0
            });
        }
    }
}

#[test]
fn subclassification() {
    let (anchors, candidates) = balance_inputs(1_200, 4_000, 6);
    let prepared = prepare_propensity_distance_config(
        &anchors,
        &candidates,
        &PropensityScoreConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .include_intercept(true)
                    .missing_value_policy(MissingValuePolicy::Impute)
                    .build(),
            )
            .build(),
    )
    .unwrap();
    let scores: HashMap<String, f64> = prepared
        .estimate
        .probabilities
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        for estimand in [Estimand::Att, Estimand::Atc, Estimand::Ate, Estimand::Atm] {
            for subclasses in [1, 6, 20] {
                let out = subclassify_by_propensity_score_map(
                    &anchors,
                    &candidates,
                    &scores,
                    estimand,
                    &SubclassificationConfig {
                        subclasses,
                        reference_group: None,
                        min_n: 1,
                    },
                );
                assert!(out.retained_units() > 0);
                d.int(out.requested_subclasses);
                d.int(out.realized_subclasses);
                let mut assignments: Vec<_> = out.assignments.iter().collect();
                assignments.sort();
                d.debug(&assignments);
                d.floats(&out.weights);
                let mut dropped = out.dropped_ids.clone();
                dropped.sort();
                d.debug(&dropped);
                d.debug(&out.subclasses);
            }
        }
        d.0
    });
}

#[test]
fn reconstructed_pairs() {
    let people = cohort(6_000, 120, 15, 33);
    let rows: Vec<ReconstructedEpisode> = people
        .iter()
        .enumerate()
        .map(|(i, p)| ReconstructedEpisode {
            person_id: p.id.clone(),
            index_date: p
                .first_date
                .unwrap_or(p.birth + chrono::TimeDelta::days(900)),
            exposed: p.exposed,
            sex: if i % 50 == 0 {
                None
            } else {
                p.sex.parse().ok()
            },
            municipality: (i % 40 != 0).then(|| p.municipality.clone()),
            region: Some(p.region.clone()),
            family_id: (i % 7 != 0).then(|| p.family.clone()),
            birth_date: (i % 60 != 0).then_some(p.birth),
        })
        .collect();
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        for (max_ratio, caliper) in [(1, 0), (3, 30), (5, 90)] {
            let out = reconstruct_case_control_pairs(
                &rows,
                ReconstructedMatchingOptions {
                    max_ratio,
                    match_caliper_days: caliper,
                },
            );
            assert!(!out.matched_pairs.is_empty(), "no output, test is vacuous");
            for set in [
                &out.eligible_case_ids,
                &out.matched_case_ids,
                &out.matched_control_ids,
                &out.municipality_case_ids,
                &out.municipality_control_ids,
                &out.region_case_ids,
                &out.region_control_ids,
            ] {
                let mut ids: Vec<_> = set.iter().collect();
                ids.sort();
                d.debug(&ids);
            }
            d.debug(&out.matched_pairs);
        }
        d.0
    });
}

#[test]
fn nearest_matching_in_every_order() {
    use epistat_matching::constraints::caliper_on_field;
    use epistat_matching::{Assignment, MatchOrder, RapidHashMap};
    let (mut anchors, mut candidates) = base_pair(300, 1_200, 21);
    let mut rng = common::Lcg::new(77);
    let mut scores: RapidHashMap<String, f64> = RapidHashMap::default();
    let mut vectors: RapidHashMap<String, Vec<f64>> = RapidHashMap::default();
    for record in anchors.iter_mut().chain(candidates.iter_mut()) {
        // Scores on a grid, so that many candidates share a distance.
        let score = rng.below(400) as f64 / 400.0;
        scores.insert(record.id.clone(), score);
        vectors.insert(
            record.id.clone(),
            vec![
                score + rng.below(100) as f64 / 500.0,
                rng.below(1000) as f64 / 100.0,
            ],
        );
        record
            .numerics
            .insert("age".to_string(), rng.below(60) as f64);
        record
            .strata
            .insert("fam".to_string(), format!("f{}", rng.below(40)));
    }
    let maha = || {
        DistanceConfig::mahalanobis_map(vectors.clone(), vec![1.0, 0.0, 0.0, 1.0], 2, None).unwrap()
    };
    let orders = [
        MatchOrder::Largest,
        MatchOrder::Smallest,
        MatchOrder::Data,
        MatchOrder::Random,
        MatchOrder::Closest,
        MatchOrder::Farthest,
    ];
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        for order in orders {
            for (replace, ratio, assignment) in [
                (false, 1, Assignment::Rounds),
                (false, 3, Assignment::Rounds),
                (true, 2, Assignment::Rounds),
            ] {
                let caliper = DistanceCaliper::new(0.3);
                let job = MatchJob::new_standard(&anchors, &candidates, 5)
                    .with_nearest()
                    .with_m_order(order)
                    .with_assignment(assignment)
                    .with_std_caliper(true)
                    .with_ratio(MatchRatio::new(ratio).unwrap())
                    .with_replacement(replace)
                    .with_exact_matches(["sex", "region"])
                    .with_distance_config(DistanceConfig::propensity_score_map(
                        scores.clone(),
                        caliper,
                    ))
                    .with_constraints(vec![caliper_on_field::<BaseRecord>("age", 20.0)]);
                let out = job.with_antiexact("fam").run();
                assert!(!out.pairs.is_empty(), "no output, test is vacuous");
                d.outcome(&out);
                d.weights(&out);
            }
            let out = MatchJob::new_standard(&anchors, &candidates, 5)
                .with_nearest()
                .with_m_order(order)
                .with_ratio(MatchRatio::new(2).unwrap())
                .with_exact_matches(["sex"])
                .with_distance_config(DistanceConfig::propensity_score_map(
                    scores.clone(),
                    DistanceCaliper::new(0.2),
                ))
                .with_mahvars(maha())
                .run();
            assert!(!out.pairs.is_empty(), "no output, test is vacuous");
            d.outcome(&out);
        }
        for order in [MatchOrder::Data, MatchOrder::Closest, MatchOrder::Farthest] {
            let out = MatchJob::new_standard(&anchors, &candidates[..300], 5)
                .with_nearest()
                .with_m_order(order)
                .with_ratio(MatchRatio::new(2).unwrap())
                .with_exact_matches(["sex", "region"])
                .with_distance_config(maha())
                .run();
            assert!(!out.pairs.is_empty(), "no output, test is vacuous");
            d.outcome(&out);
        }
        let out = MatchJob::new_standard(&anchors, &candidates, 5)
            .with_strategy(DeterministicSelection)
            .with_round_assignment()
            .with_ratio(MatchRatio::new(3).unwrap())
            .with_birth_window(BirthDateWindowDays::new(45).unwrap())
            .with_exact_matches(["sex", "region"])
            .run();
        assert!(!out.pairs.is_empty(), "no output, test is vacuous");
        d.outcome(&out);
        d.0
    });
}
