//! The bits of `balance_summary` on weights given from outside, pinned by a
//! hash of the whole summary, on one thread and on several. One problem has
//! tens of thousands of units in a `Cohort` with sampling weights, discarded
//! units, pairs and every interaction; the other has records with their own
//! covariate maps, supplemental covariates and an identifier that two
//! controls share, so that pairs are read by identifier. The hashes were
//! recorded before the report's lookups by identifier were changed, so they
//! show that the report is the same.
//!
//! The hash is of the summary written as JSON, whose numbers round-trip, so a
//! change to any last bit changes it. The test runs only on aarch64 macOS,
//! where the other pinned hashes of the workspace are recorded.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use chrono::NaiveDate;
use epistat_matching::{
    BalanceRecord, BalanceReportOptions, BalanceSummary, BalanceWeights, Cohort, CohortRow,
    CovariateValue, Estimand, MatchedPair, NumericBalanceTransform, SupplementalBalanceCovariates,
    UnitWeightSet, balance_summary,
};
use epistat_testkit::{Fingerprint, SplitMix64, on_threads};
use rapidhash::RapidHashMap;
use std::collections::HashMap;

const REGIONS: [&str; 4] = ["north", "south", "east", "west"];

fn hash(summary: &BalanceSummary) -> u64 {
    assert!(summary.pair_distances.iter().any(|d| d.raw.is_some()));
    serde_json::to_string(summary)
        .expect("a summary is JSON")
        .bytes()
        .fold(Fingerprint::new(), |h, b| h.u64(u64::from(b)))
        .finish()
}

fn date() -> NaiveDate {
    NaiveDate::from_ymd_opt(2001, 2, 3).expect("a date")
}

/// Every seventh unit a case, the next unit its paired control with weight
/// one, the one after with weight one third, and every 89th unit discarded.
fn weights(n: usize, id: impl Fn(usize) -> String) -> BalanceWeights {
    let mut anchor = RapidHashMap::default();
    let mut candidate = RapidHashMap::default();
    let mut pairs = Vec::new();
    for i in 0..n {
        match i % 7 {
            0 => {
                anchor.insert(id(i), 1.0);
                if i + 1 < n {
                    pairs.push(MatchedPair::new(id(i), id(i + 1)));
                }
            }
            1 => {
                candidate.insert(id(i), 1.0);
            }
            2 => {
                candidate.insert(id(i), 1.0 / 3.0);
            }
            _ => {}
        }
    }
    BalanceWeights::new(Estimand::Ate, UnitWeightSet::new(anchor, candidate))
        .with_pairs(pairs)
        .with_discarded((0..n).filter(|i| i % 89 == 5).map(&id))
}

fn split<T: Copy>(rows: &[T]) -> (Vec<T>, Vec<T>) {
    let cases = rows.iter().step_by(7).copied().collect();
    let controls = rows
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 7 != 0)
        .map(|(_, r)| *r)
        .collect();
    (cases, controls)
}

fn cohort(n: usize) -> Cohort {
    let mut rng = SplitMix64::new(20_261_124);
    let mut age = Vec::new();
    let mut income = Vec::new();
    let mut smoker = Vec::new();
    let mut sex = Vec::new();
    let mut region = Vec::new();
    for i in 0..n {
        let shift = if i % 7 == 0 { 0.4 } else { 0.0 };
        age.push(Some((30.0 + 20.0 * (rng.unit() + shift)).round()));
        income.push((i % 13 != 4).then(|| 100.0 * (rng.unit() + shift)));
        smoker.push(Some(f64::from(u8::from(rng.unit() < 0.3))));
        sex.push(Some(if rng.unit() < 0.5 { "F" } else { "M" }));
        region.push(
            (i % 17 != 3).then(|| REGIONS[usize::try_from(rng.below(4)).expect("a small index")]),
        );
    }
    Cohort::builder((0..n).map(|i| format!("c{i}")), vec![date(); n])
        .numeric_covariate("age", age)
        .numeric_covariate("income", income)
        .numeric_covariate("smoker", smoker)
        .categorical_covariate("sex", sex)
        .categorical_covariate("region", region)
        .build()
        .expect("a valid cohort")
}

fn records(n: usize) -> Vec<BalanceRecord> {
    let mut rng = SplitMix64::new(20_261_125);
    (0..n)
        .map(|i| {
            // Units 1 and 8 are both controls called "r1".
            let id = if i == 8 {
                "r1".to_owned()
            } else {
                format!("r{i}")
            };
            let shift = if i % 7 == 0 { 0.3 } else { 0.0 };
            BalanceRecord::builder(id, date())
                .numeric("dose", (i % 11 != 2).then(|| rng.unit() + shift))
                .categorical(
                    "grade",
                    (i % 5 != 1).then(|| {
                        ["a", "b", "c"][usize::try_from(rng.below(3)).expect("a small index")]
                            .to_owned()
                    }),
                )
                .build()
        })
        .collect()
}

fn summaries() -> Vec<u64> {
    let n = 40_007;
    let cohort = cohort(n);
    let rows: Vec<CohortRow<'_>> = cohort.rows().collect();
    let (cases, controls) = split(&rows);
    let sampling: HashMap<String, f64> = (0..n)
        .filter(|i| i % 3 == 1)
        .map(|i| {
            (
                format!("c{i}"),
                0.5 + f64::from(u8::try_from(i % 4).unwrap()),
            )
        })
        .collect();
    let on_cohort = weights(n, |i| format!("c{i}")).with_sampling_weights(sampling);
    let all_terms = BalanceReportOptions {
        numeric_transforms: NumericBalanceTransform::SquaresAndPairwiseInteractions,
        ..BalanceReportOptions::default()
    };
    let mut hashes = vec![
        hash(&balance_summary(&cases, &controls, &on_cohort, &all_terms).unwrap()),
        hash(
            &balance_summary(
                &cases,
                &controls,
                &on_cohort,
                &BalanceReportOptions::default(),
            )
            .unwrap(),
        ),
    ];

    let m = 3_001;
    let owned = records(m);
    let refs: Vec<&BalanceRecord> = owned.iter().collect();
    let (cases, controls) = split(&refs);
    let cases: Vec<BalanceRecord> = cases.into_iter().cloned().collect();
    let controls: Vec<BalanceRecord> = controls.into_iter().cloned().collect();
    let id = |i: usize| {
        if i == 8 {
            "r1".to_owned()
        } else {
            format!("r{i}")
        }
    };
    let mut supplemental = SupplementalBalanceCovariates::default();
    for i in (0..m).filter(|i| i % 4 == 0) {
        let target = if i % 7 == 0 {
            &mut supplemental.cases
        } else {
            &mut supplemental.controls
        };
        target.insert(
            id(i),
            HashMap::from([(
                "score".to_owned(),
                CovariateValue::Numeric(f64::from(u32::try_from(i % 9).unwrap())),
            )]),
        );
    }
    let options = BalanceReportOptions {
        numeric_transforms: NumericBalanceTransform::Squares,
        supplemental_covariates: supplemental,
    };
    hashes.push(hash(
        &balance_summary(&cases, &controls, &weights(m, id), &options).unwrap(),
    ));
    hashes
}

#[test]
fn the_summaries_keep_their_bits_on_every_pool() {
    let expected: [u64; 3] = [
        9_782_542_025_365_647_269,
        8_386_241_516_838_501_912,
        5_585_345_883_895_415_797,
    ];
    for (threads, got) in [1, 3, 8].into_iter().zip(on_threads(&[1, 3, 8], summaries)) {
        assert_eq!(got, expected, "{threads} threads");
    }
}
