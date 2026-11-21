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
    CbpsConfig, CbpsEstimand, CbpsMethod, DeterministicSelection, NearestBirthDateSelection,
    ReconstructedEpisode, ReconstructedMatchingOptions, SubclassificationConfig,
    balance_report_with_options, estimate_propensity_and_match, prepare_euclidean_distance_config,
    prepare_mahalanobis_distance_config, prepare_propensity_distance_config,
    reconstruct_case_control_pairs, subclassify_by_propensity_score_map,
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
        self.debug(&estimate.coefficients);
        self.debug(&estimate.column_names);
        self.text(estimate.estimator);
        self.int(usize::from(estimate.converged));
        self.int(estimate.iterations);
        self.debug(&estimate.aliased_columns);
        self.int(usize::from(estimate.fitted_at_boundary));
        self.debug(&estimate.lambda);
        self.debug(&estimate.cross_validation);
        self.debug(&estimate.cbps);
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
        let mut recs = records(&people, Level::Municipality, true, false);
        // Staggered entry: a third of the people enter between ages 1 and 5,
        // so the entry date decides some risk sets.
        for (idx, rec) in recs.iter_mut().enumerate() {
            if idx % 3 == 0 {
                let months = 12 + u32::try_from(idx % 49).expect("small");
                rec.entry_date = rec
                    .birth_date
                    .checked_add_months(chrono::Months::new(months));
            }
        }
        assert!(recs.iter().any(|rec| rec.entry_date.is_some()));
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
fn exposure_density_sampling() {
    let people = cohort(6_000, 90, 12, 31);
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        let recs: Vec<RoleTransitionRecord<BalanceRecord>> =
            records(&people, Level::Municipality, true, false)
                .into_iter()
                .map(|row| {
                    RoleTransitionRecord::from_record(
                        BalanceRecord {
                            core: row.record,
                            covariates: HashMap::new(),
                        },
                        row.transition_date,
                    )
                })
                .collect();
        let spec = MatchSpec {
            mode: MatchMode::ExposureDensity,
            seed: 99,
            ratio: 3,
            replace: true,
            birth_window_days: 60,
            age_limit_years: Some(6),
            exact: vec!["sex".into(), "municipality".into()],
            require_alive: true,
            ..MatchSpec::default()
        };
        let SpecOutput::ExposureDensity(sets) =
            run_spec(&spec, SpecInput::RiskSet { cohort: &recs }).expect("valid spec")
        else {
            panic!("expected exposure density sets");
        };
        assert!(!sets.controls.is_empty(), "no output, test is vacuous");
        d.outcome(&sets.outcome);
        d.debug(&sets.exposed);
        d.debug(&sets.controls);
        d.0
    });
}

#[test]
fn sequential_trials() {
    let people = cohort(4_000, 90, 12, 31);
    let rows = records(&people, Level::Municipality, true, false);
    let start = NaiveDate::from_ymd_opt(2006, 1, 1).expect("valid date");
    // Eight periods of 180 days. A person starts treatment at a period set by
    // the position in the cohort, stops again after two or three periods for
    // some, and is eligible while never treated.
    let panel: Vec<TrialPerson<BalanceRecord>> = rows
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let first_on = (i * 7 + i / 3) % 11;
            let length = 2 + i % 2;
            let mut periods = Vec::new();
            for p in 0..8_usize {
                let date = start + chrono::Days::new(180 * p as u64);
                let gone = row.record.death_date.is_some_and(|d| d <= date)
                    || row.record.emigration_date.is_some_and(|d| d <= date)
                    || row.record.entry_date.is_some_and(|d| d > date);
                if gone {
                    continue;
                }
                let on = i % 5 != 0 && p >= first_on && p < first_on + length;
                periods.push(TrialPeriod {
                    period: i64::try_from(p).expect("small"),
                    date,
                    eligible: p <= first_on || i % 5 == 0,
                    treatment: Some(on),
                    event: false,
                });
            }
            TrialPerson {
                record: BalanceRecord {
                    core: row.record,
                    covariates: HashMap::new(),
                },
                periods,
            }
        })
        .collect();
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        let spec = MatchSpec {
            mode: MatchMode::SequentialTrials,
            seed: 5,
            ratio: 2,
            replace: false,
            birth_window_days: 365,
            age_limit_years: Some(12),
            exact: vec!["sex".into(), "municipality".into()],
            require_alive: true,
            require_resident: true,
            trials: TrialOptions {
                control_reuse: ControlReuse::OncePerRun,
                ..TrialOptions::default()
            },
            ..MatchSpec::default()
        };
        let SpecOutput::Trials(out) =
            run_spec(&spec, SpecInput::Trials { panel: &panel }).expect("valid spec")
        else {
            panic!("expected trials");
        };
        assert!(
            out.trials.iter().any(|t| !t.outcome.pairs.is_empty()),
            "no output, test is vacuous"
        );
        for trial in &out.trials {
            d.int(trial.n_eligible);
            d.int(trial.n_initiators);
            d.outcome(&trial.outcome);
            d.debug(&trial.members);
        }
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
        d.float(diagnostics.match_rate.unwrap());
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
    let cross_validated = PropensityEstimator::ElasticNetCv(
        ElasticNetCvConfig::builder()
            .seed(11)
            .nfolds(3)
            .nlambda(12)
            .build(),
    );
    let logit = PropensityLink::Logit;
    for (name, estimator, scale, link) in [
        (
            "logit",
            PropensityEstimator::GlmLogit,
            PropensityScoreOutputScale::Probability,
            logit,
        ),
        (
            "logit_lp",
            PropensityEstimator::GlmLogit,
            PropensityScoreOutputScale::LinearPredictor,
            logit,
        ),
        (
            "probit",
            PropensityEstimator::GlmLogit,
            PropensityScoreOutputScale::Probability,
            PropensityLink::Probit,
        ),
        (
            "cloglog",
            PropensityEstimator::GlmLogit,
            PropensityScoreOutputScale::LinearPredictor,
            PropensityLink::Cloglog,
        ),
        (
            "cbps_over_att",
            PropensityEstimator::Cbps(CbpsConfig::default()),
            PropensityScoreOutputScale::Probability,
            logit,
        ),
        (
            "cbps_exact_ate",
            PropensityEstimator::Cbps(
                CbpsConfig::builder()
                    .estimand(CbpsEstimand::Ate)
                    .method(CbpsMethod::Exact)
                    .build(),
            ),
            PropensityScoreOutputScale::LinearPredictor,
            logit,
        ),
        (
            "elastic",
            elastic,
            PropensityScoreOutputScale::Probability,
            logit,
        ),
        (
            "elastic_cv",
            cross_validated,
            PropensityScoreOutputScale::Probability,
            logit,
        ),
    ] {
        let config = PropensityScoreConfig::builder()
            .estimator(estimator)
            .output_scale(scale)
            .logistic(LogisticRegressionConfig::builder().link(link).build())
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
fn reestimated_propensity_and_nearest_matching() {
    use epistat_matching::{CommonSupport, NearestSelection};
    let (anchors, candidates) = balance_inputs(600, 2_400, 4);
    let config = PropensityScoreConfig::builder()
        .reestimate(true)
        .encoding(
            CovariateEncodingConfig::builder()
                .include_intercept(true)
                .missing_value_policy(MissingValuePolicy::Impute)
                .build(),
        )
        .build();
    for policy in [
        CommonSupport::Treated,
        CommonSupport::Control,
        CommonSupport::Both,
    ] {
        assert_same_on_pools(&POOLS, || {
            let mut d = Digest::default();
            let criteria = MatchingCriteria::builder()
                .match_ratio(2)
                .common_support(policy)
                .build();
            let matched = estimate_propensity_and_match(
                &anchors,
                &candidates,
                &criteria,
                NearestSelection::new(3),
                &[],
                &(),
                &config,
            )
            .unwrap();
            assert!(
                !matched.outcome.pairs.is_empty(),
                "no output, test is vacuous"
            );
            d.outcome(&matched.outcome);
            d.weights(&matched.outcome);
            d.estimate(&matched.propensity);
            d.0
        });
    }
}

#[test]
fn nearest_matching_on_prepared_distances() {
    use epistat_matching::MatchOrder;
    let (anchors, candidates) = balance_inputs(150, 600, 9);
    let impute = CovariateEncodingConfig::builder()
        .missing_value_policy(MissingValuePolicy::Impute)
        .build();
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        let mahalanobis = prepare_mahalanobis_distance_config(
            &anchors,
            &candidates,
            None,
            &MahalanobisPreparationConfig::builder()
                .encoding(impute.clone())
                .build(),
        )
        .unwrap();
        let robust = prepare_mahalanobis_distance_config(
            &anchors,
            &candidates,
            None,
            &MahalanobisPreparationConfig::builder()
                .encoding(impute.clone())
                .covariance_strategy(MahalanobisCovarianceStrategy::FullSample)
                .transform(MahalanobisTransform::RankTieAdjusted)
                .build(),
        )
        .unwrap();
        let euclidean = prepare_euclidean_distance_config(
            &anchors,
            &candidates,
            None,
            &EuclideanPreparationConfig::builder()
                .encoding(impute.clone())
                .scale(EuclideanScale::PooledWithinGroupSd)
                .build(),
        )
        .unwrap();
        for config in [
            mahalanobis.distance_config,
            robust.distance_config,
            euclidean.distance_config,
        ] {
            for order in [MatchOrder::Data, MatchOrder::Closest, MatchOrder::Farthest] {
                let out = MatchJob::new_standard(&anchors, &candidates, 1)
                    .with_nearest()
                    .with_m_order(order)
                    .with_ratio(MatchRatio::new(2).unwrap())
                    .with_exact_matches(["sex"])
                    .with_distance_config(config.clone())
                    .run();
                assert!(!out.pairs.is_empty(), "no output, test is vacuous");
                d.outcome(&out);
            }
        }
        d.0
    });
}

#[test]
fn mahalanobis_preparation_and_matching() {
    let (anchors, candidates) = balance_inputs(600, 2_400, 5);
    for strategy in [
        MahalanobisCovarianceStrategy::PooledWithinGroups,
        MahalanobisCovarianceStrategy::PooledByDegreesOfFreedom,
        MahalanobisCovarianceStrategy::FullSample,
    ] {
        for transform in [
            MahalanobisTransform::Raw,
            MahalanobisTransform::Rank,
            MahalanobisTransform::RankTieAdjusted,
        ] {
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
                d.int(prepared.covariance_rank);
                d.debug(&prepared.column_names);
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
fn euclidean_preparation_and_matching() {
    let (anchors, candidates) = balance_inputs(600, 2_400, 5);
    for scale in [EuclideanScale::None, EuclideanScale::PooledWithinGroupSd] {
        let config = EuclideanPreparationConfig::builder()
            .scale(scale)
            .encoding(
                CovariateEncodingConfig::builder()
                    .missing_value_policy(MissingValuePolicy::Impute)
                    .build(),
            )
            .build();
        assert_same_on_pools(&POOLS, || {
            let mut d = Digest::default();
            let prepared = prepare_euclidean_distance_config(
                &anchors,
                &candidates,
                Some(DistanceCaliper::new(2.0).unwrap()),
                &config,
            )
            .unwrap();
            let mut vectors: Vec<_> = prepared.vectors.iter().collect();
            vectors.sort_by(|a, b| a.0.cmp(b.0));
            for (id, vector) in vectors {
                d.text(id);
                for v in vector {
                    d.float(*v);
                }
            }
            for v in &prepared.divisors {
                d.float(*v);
            }
            d.int(prepared.dimension);
            d.debug(&prepared.column_names);
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
                )
                .expect("valid subclassification settings");
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
    use epistat_matching::{Assignment, MatchOrder, RapidHashMap, VariableRatio};
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
            for replace in [false, true] {
                let out = MatchJob::new_standard(&anchors, &candidates, 5)
                    .with_nearest()
                    .with_m_order(order)
                    .with_variable_ratio(VariableRatio::new(2.5, 1, 5).unwrap())
                    .with_replacement(replace)
                    .with_exact_matches(["sex"])
                    .with_distance_config(DistanceConfig::propensity_score_map(
                        scores.clone(),
                        DistanceCaliper::new(0.4),
                    ))
                    .with_std_caliper(true)
                    .run();
                assert!(!out.pairs.is_empty(), "no output, test is vacuous");
                d.outcome(&out);
                d.weights(&out);
            }
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

/// The units with their age as a numeric field, which coarsened exact matching
/// reads.
fn with_age_fields(rows: &mut [BalanceRecord]) {
    for row in rows {
        if let Some(CovariateValue::Numeric(age)) = row.covariates.get("age") {
            let age = *age;
            row.core.numerics.insert("age".into(), age);
        }
    }
}

#[test]
fn coarsened_exact_matching() {
    use epistat_matching::{CemBinning, CemRule, CemVariable, cem_k2k_pairs, cem_strata};
    let (mut anchors, mut candidates) = balance_inputs(1_200, 4_000, 11);
    with_age_fields(&mut anchors);
    with_age_fields(&mut candidates);
    let variables = vec![
        CemVariable::numeric("age", CemBinning::Rule(CemRule::FreedmanDiaconis)),
        CemVariable::categorical("sex", vec![]),
        CemVariable::categorical("region", vec![vec!["r0".into(), "r1".into(), "r2".into()]]),
    ];
    let euclidean = prepare_euclidean_distance_config(
        &anchors,
        &candidates,
        None,
        &EuclideanPreparationConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .missing_value_policy(MissingValuePolicy::Impute)
                    .build(),
            )
            .build(),
    )
    .unwrap()
    .distance_config;
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        let out = epistat_matching::cem_subclasses(&anchors, &candidates, &variables).unwrap();
        assert!(out.retained_units() > 0, "no output, test is vacuous");
        let mut assignments: Vec<_> = out.assignments.iter().collect();
        assignments.sort();
        d.debug(&assignments);
        d.floats(&out.att_weights);
        d.floats(&out.atc_weights);
        d.floats(&out.ate_weights);
        let mut dropped = out.dropped_ids.clone();
        dropped.sort();
        d.debug(&dropped);
        d.debug(&out.subclasses);
        let strata = cem_strata(&anchors, &candidates, &variables).unwrap();
        for order in [
            MatchOrder::Data,
            MatchOrder::Random,
            MatchOrder::Closest,
            MatchOrder::Farthest,
        ] {
            for distance in [Some(&euclidean), None] {
                let pairs = cem_k2k_pairs(
                    &anchors,
                    &candidates,
                    &strata,
                    distance,
                    order,
                    9,
                    Estimand::Att,
                )
                .unwrap();
                assert!(!pairs.pairs.is_empty(), "no output, test is vacuous");
                d.outcome(&pairs);
            }
        }
        d.0
    });
}

#[test]
fn nearest_matching_on_a_distance_matrix() {
    let (anchors, candidates) = balance_inputs(300, 1_200, 12);
    let age = |record: &BalanceRecord| match record.covariates.get("age") {
        Some(CovariateValue::Numeric(age)) => *age,
        _ => panic!("every unit has an age"),
    };
    let mut values = Vec::with_capacity(anchors.len() * candidates.len());
    for (row, anchor) in anchors.iter().enumerate() {
        for (column, candidate) in candidates.iter().enumerate() {
            // Some pairs are forbidden.
            values.push(if (row * 7 + column * 3) % 11 == 0 {
                f64::INFINITY
            } else {
                (age(anchor) - age(candidate)).abs()
            });
        }
    }
    let matrix = PairDistanceMatrix::new(
        anchors.iter().map(|r| r.core.id.clone()).collect(),
        candidates.iter().map(|r| r.core.id.clone()).collect(),
        values,
    )
    .unwrap();
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        for order in [MatchOrder::Data, MatchOrder::Closest, MatchOrder::Farthest] {
            for ratio in [1, 3] {
                let out = MatchJob::new_standard(&anchors, &candidates, 5)
                    .with_nearest()
                    .with_m_order(order)
                    .with_ratio(MatchRatio::new(ratio).unwrap())
                    .with_exact_matches(["sex"])
                    .with_distance_config(DistanceConfig::matrix(
                        matrix.clone(),
                        DistanceCaliper::new(1.5),
                    ))
                    .run();
                assert!(!out.pairs.is_empty(), "no output, test is vacuous");
                d.outcome(&out);
                d.weights(&out);
            }
        }
        d.0
    });
}

/// Generalised full matching on a line and in the plane, with every seed
/// method and a caliper, and on records with exact strata. The plane is large
/// enough for the k-d tree to be built in parallel.
#[test]
fn generalised_full_matching() {
    use epistat_matching::{QuickMatchOptions, SeedMethod, quick_clusters, quick_match_subclasses};
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let line: Vec<f64> = (0..20_000).map(|_| next()).collect();
    let plane: Vec<f64> = (0..2 * 12_000).map(|_| next()).collect();
    let anchor_line: Vec<bool> = line.iter().map(|x| next() < 0.2 + 0.3 * x).collect();
    let anchor_plane: Vec<bool> = (0..12_000).map(|_| next() < 0.3).collect();

    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    let mut coordinates: HashMap<String, Vec<f64>> = HashMap::new();
    for row in 0..3_000_usize {
        let mut record = BaseRecord::new(format!("r{row}"), birth);
        record
            .strata
            .insert("sex".into(), ["f", "m"][row % 2].into());
        coordinates.insert(record.id().to_string(), vec![next(), next()]);
        if next() < 0.35 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }

    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        for method in [
            SeedMethod::InwardsUpdating,
            SeedMethod::InwardsOrder,
            SeedMethod::Lexical,
        ] {
            for (min_anchors, min_candidates, size, caliper) in [
                (1, 1, None, None),
                (2, 3, Some(7), None),
                (1, 1, None, Some(0.002)),
            ] {
                let options = QuickMatchOptions {
                    min_anchors,
                    min_candidates,
                    size,
                    caliper,
                    seed_method: method,
                };
                let labels = quick_clusters(&line, 1, &anchor_line, &options).unwrap();
                assert!(labels.iter().flatten().count() > 0, "vacuous");
                d.debug(&labels);
                let caliper = caliper.map(|width| width * 40.0);
                let options = QuickMatchOptions { caliper, ..options };
                let labels = quick_clusters(&plane, 2, &anchor_plane, &options).unwrap();
                assert!(labels.iter().flatten().count() > 0, "vacuous");
                d.debug(&labels);
            }
            let out = quick_match_subclasses(
                &anchors,
                &candidates,
                |record: &BaseRecord| coordinates.get(record.id()).map(Vec::as_slice),
                2,
                &["sex"],
                &QuickMatchOptions {
                    seed_method: method,
                    ..QuickMatchOptions::default()
                },
            )
            .unwrap();
            let mut assignments: Vec<_> = out.assignments.iter().collect();
            assignments.sort();
            d.debug(&assignments);
            d.floats(&out.att_weights);
            d.floats(&out.atc_weights);
            d.floats(&out.ate_weights);
            d.debug(&out.subclasses);
        }
        d.0
    });
}

/// Optimal pair and full matching solve each exact stratum on its own and
/// join the strata in order, so the result is the same in every pool.
#[test]
fn optimal_and_full_matching() {
    use epistat_matching::{
        FullMatchConfig, OptimalRequest, RapidHashMap, match_full, match_optimal,
    };

    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut draw = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let birth = date(0);
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    let mut scores: RapidHashMap<String, f64> = RapidHashMap::default();
    for (rows, prefix, count, shift) in [
        (&mut anchors, "a", 400, 0.2),
        (&mut candidates, "c", 1_200, 0.0),
    ] {
        for index in 0..count {
            let id = format!("{prefix}{index}");
            let mut record = BaseRecord::new(id.clone(), birth);
            record
                .strata
                .insert("g".to_string(), (draw() * 8.0).floor().to_string());
            scores.insert(id, draw() + shift);
            rows.push(record);
        }
    }
    let distance = DistanceConfig::propensity_score_map(scores, None);
    let criteria = MatchingCriteria::builder()
        .match_ratio(2)
        .required_strata(vec!["g".to_string()])
        .build();
    let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        let pairs = match_optimal(&anchors, &candidates, &request).unwrap();
        assert_ne!(pairs.pairs.len(), 0);
        d.outcome(&pairs);
        d.weights(&pairs);
        for config in [
            FullMatchConfig::default(),
            FullMatchConfig {
                min_controls: 1.0,
                max_controls: 5.0,
                ..FullMatchConfig::default()
            },
        ] {
            let out = match_full(&anchors, &candidates, &request, &config).unwrap();
            let mut assignments: Vec<_> = out.assignments.iter().collect();
            assignments.sort();
            d.debug(&assignments);
            d.floats(&out.att_weights);
            d.floats(&out.atc_weights);
            d.floats(&out.ate_weights);
            d.debug(&out.dropped_ids);
            d.debug(&out.subclasses);
        }
        d.0
    });
}

/// Balanced matching builds the edges of the strata in parallel and solves one
/// network, so the result is the same in every pool.
#[test]
fn fine_balance_matching() {
    use epistat_matching::{FineBalance, OptimalRequest, RapidHashMap, match_fine_balance};

    let mut state = 0x1f83_d9ab_fb41_bd6b_u64;
    let mut draw = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let birth = date(0);
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    let mut scores: RapidHashMap<String, f64> = RapidHashMap::default();
    for (rows, prefix, count, shift) in [
        (&mut anchors, "a", 300, 0.2),
        (&mut candidates, "c", 1_000, 0.0),
    ] {
        for index in 0..count {
            let id = format!("{prefix}{index}");
            let mut record = BaseRecord::new(id.clone(), birth);
            let coarse = (draw() * 3.0).floor();
            let fine = coarse * 2.0 + (draw() * 2.0).floor();
            record
                .strata
                .insert("g".to_string(), (draw() * 4.0).floor().to_string());
            record
                .strata
                .insert("coarse".to_string(), coarse.to_string());
            record.strata.insert("fine".to_string(), fine.to_string());
            scores.insert(id, draw() + shift);
            rows.push(record);
        }
    }
    let distance = DistanceConfig::propensity_score_map(scores, None);
    let criteria = MatchingCriteria::builder()
        .match_ratio(2)
        .required_strata(vec!["g".to_string()])
        .build();
    let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
    let balance = FineBalance {
        levels: vec![vec!["coarse".into()], vec!["fine".into()]],
        ..FineBalance::default()
    };
    assert_same_on_pools(&POOLS, || {
        let mut d = Digest::default();
        let out = match_fine_balance(&anchors, &candidates, &request, &balance).unwrap();
        assert_ne!(out.outcome.pairs.len(), 0);
        d.outcome(&out.outcome);
        d.debug(&out.levels);
        d.0
    });
}
