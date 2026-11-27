//! Vector matching and matching after multiple imputation give bit-identical
//! output in rayon pools of 1, 2, 3, 4 and 8 threads.
//!
//! The vector match is on 600 simulated units in three groups. The imputation
//! match is the `within` and `across` approaches on the five data sets of
//! `tests/data/matchthem.json`, with the pooled effect.

use chrono::NaiveDate;
use epistat_matching::effects::{EffectSpec, OutcomeModel, UnitData};
use epistat_matching::imputation::{
    CompleteDataDf, ImputationApproach, ImputedSet, estimate_pooled, match_imputed,
};
use epistat_matching::multigroup::{VectorMatchSpec, vector_match};
use epistat_matching::output::MatchDataOptions;
use epistat_matching::{
    BalanceRecord, CovariateValue, MatchingCriteria, NearestSelection, PropensityScoreConfig,
    estimate_propensity_and_match,
};
use epistat_testkit::{POOLS, assert_same_on_pools, load_json};
use std::collections::BTreeMap;

fn units(n: usize) -> (Vec<BalanceRecord>, Vec<usize>) {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).unwrap();
    let mut state = 12345_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut records = Vec::new();
    let mut groups = Vec::new();
    for i in 0..n {
        let (a, b) = (next() * 2.0 - 1.0, next() * 2.0 - 1.0);
        let w1 = (0.8 * a).exp();
        let w2 = (0.8 * b - 0.4 * a).exp();
        let draw = next() * (1.0 + w1 + w2);
        let group = if draw < 1.0 {
            0
        } else if draw < 1.0 + w1 {
            1
        } else {
            2
        };
        let mut record = BalanceRecord::new(format!("u{i}"), birth);
        record
            .covariates
            .insert("a".to_string(), CovariateValue::Numeric(a));
        record
            .covariates
            .insert("b".to_string(), CovariateValue::Numeric(b));
        records.push(record);
        groups.push(group);
    }
    (records, groups)
}

#[test]
fn vector_matching_does_not_depend_on_the_number_of_threads() {
    let (records, groups) = units(600);
    assert_same_on_pools(&POOLS, || {
        let out = vector_match(&records, &groups, 3, &VectorMatchSpec::new(0, 3)).unwrap();
        let mut values = out.gps.probabilities.clone();
        for comparison in &out.comparisons {
            for pair in &comparison.outcome.pairs {
                values.push(pair.anchor_id()[1..].parse().unwrap());
                values.push(pair.comparator_id()[1..].parse().unwrap());
            }
        }
        values
    });
}

/// The anchors and the candidates of each imputed data set.
type AnchorsAndCandidates = Vec<(Vec<BalanceRecord>, Vec<BalanceRecord>)>;

/// The imputed data sets of `matchthem.json` as anchors and candidates, and
/// the outcome of each set.
fn matchthem_sets() -> (AnchorsAndCandidates, Vec<UnitData>) {
    let data: Vec<BTreeMap<String, Vec<f64>>> = {
        #[derive(serde::Deserialize)]
        struct File {
            data: Vec<BTreeMap<String, Vec<f64>>>,
        }
        let file: File = load_json(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/matchthem.json"
        ));
        file.data
    };
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).unwrap();
    let prepared: Vec<(Vec<BalanceRecord>, Vec<BalanceRecord>)> = data
        .iter()
        .map(|d| {
            let (mut anchors, mut candidates) = (Vec::new(), Vec::new());
            for (row, treat) in d["treat"].iter().enumerate() {
                let mut record = BalanceRecord::new((row + 1).to_string(), birth);
                for name in ["age", "educ", "married", "nodegree", "re74", "re75"] {
                    record
                        .covariates
                        .insert(name.to_string(), CovariateValue::Numeric(d[name][row]));
                }
                if *treat > 0.5 {
                    anchors.push(record);
                } else {
                    candidates.push(record);
                }
            }
            (anchors, candidates)
        })
        .collect();
    let outcomes: Vec<UnitData> = data
        .iter()
        .map(|d| {
            UnitData::new(
                (1..=d["treat"].len()).map(|i| i.to_string()).collect(),
                d["treat"].iter().map(|t| *t > 0.5).collect(),
                d["re78"].clone(),
            )
            .unwrap()
        })
        .collect();
    (prepared, outcomes)
}

#[test]
fn imputation_matching_does_not_depend_on_the_number_of_threads() {
    let (prepared, outcomes) = matchthem_sets();
    let sets: Vec<ImputedSet<'_, BalanceRecord>> = prepared
        .iter()
        .map(|(a, c)| ImputedSet {
            anchors: a,
            candidates: c,
        })
        .collect();
    let criteria = MatchingCriteria::builder().match_ratio(1).build();
    for approach in [ImputationApproach::Within, ImputationApproach::Across] {
        assert_same_on_pools(&POOLS, || {
            let matches = match_imputed(
                &sets,
                approach,
                &criteria,
                &NearestSelection::new(0),
                &[],
                &(),
                &PropensityScoreConfig::default(),
            )
            .unwrap();
            let effects = estimate_pooled(
                &sets,
                &matches,
                &outcomes,
                MatchDataOptions::default(),
                &EffectSpec::new(OutcomeModel::Linear),
                CompleteDataDf::Residual,
            )
            .unwrap();
            let p = effects.pooled.difference.unwrap();
            let mut values = vec![p.estimate, p.within, p.between, p.total, p.df.unwrap()];
            for set in &matches.sets {
                for pair in &set.outcome.pairs {
                    values.push(pair.anchor_id().parse().unwrap());
                    values.push(pair.comparator_id().parse().unwrap());
                }
            }
            values
        });
    }
}

/// The sets matched together, each in its own task, are those one call per set
/// gives, in the order of the sets.
#[test]
fn within_imputation_matches_each_set_as_a_single_set() {
    let (prepared, _) = matchthem_sets();
    let sets: Vec<ImputedSet<'_, BalanceRecord>> = prepared
        .iter()
        .map(|(a, c)| ImputedSet {
            anchors: a,
            candidates: c,
        })
        .collect();
    let criteria = MatchingCriteria::builder().match_ratio(1).build();
    let config = PropensityScoreConfig::default();
    let matches = match_imputed(
        &sets,
        ImputationApproach::Within,
        &criteria,
        &NearestSelection::new(0),
        &[],
        &(),
        &config,
    )
    .unwrap();
    for (set, together) in sets.iter().zip(&matches.sets) {
        let alone = estimate_propensity_and_match(
            set.anchors,
            set.candidates,
            &criteria,
            NearestSelection::new(0),
            &[],
            &(),
            &config,
        )
        .unwrap();
        assert_eq!(
            format!("{:?}", alone.outcome),
            format!("{:?}", together.outcome)
        );
        assert_eq!(
            format!(
                "{:?}",
                alone
                    .propensity
                    .ids
                    .iter()
                    .zip(alone.propensity.distance_scores.iter())
                    .collect::<BTreeMap<_, _>>()
            ),
            format!(
                "{:?}",
                together
                    .propensity
                    .ids
                    .iter()
                    .zip(together.propensity.distance_scores.iter())
                    .collect::<BTreeMap<_, _>>()
            )
        );
    }
}
