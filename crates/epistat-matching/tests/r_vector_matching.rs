//! Matching with more than two treatment groups: the matching step checked
//! against R 4.6's `Matching::Match()`, and the rest by its properties.
//!
//! The reference is `tests/data/vector_matching.json`, written by
//! `tests/data/vector_matching.R`. It holds eight simulated data sets in three
//! groups. For each, `nnet::multinom()` gave the generalised propensity score,
//! `stats::kmeans()` clustered the units on the logit score of the third
//! group, and `Match(M = 1, replace = TRUE, caliper = 0.25 or none)` matched
//! the units of the first group to those of the second or third within each
//! cluster on the logit score of the first. The test gives
//! `match_within_clusters` the same scores, groups and clusters and requires the
//! same pairs. Scores are continuous, so no two candidates tie, and the
//! only boundary case is a candidate exactly one caliper away, which the
//! 1e-16 relative rounding of the standard deviation could move and which
//! the data do not contain.
//!
//! The procedure as a whole has no R reference. The tests below check the
//! properties the algorithm of Lopez and Gutman gives it: the pairs join the
//! focal group to the compared group inside one cluster and within the
//! caliper, the final cohort has a match in every comparison and no other
//! focal unit has pairs, the common support drops what its rule says, and the
//! match is reproducible and improves balance.

use chrono::NaiveDate;
use epistat_matching::multigroup::{
    GpsConfig, VectorMatchSpec, balance_by_group, generalised_propensity_score,
    match_within_clusters, vector_match,
};
use epistat_matching::prelude::*;
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    #[serde(rename = "Matching")]
    matching: String,
}

#[derive(Deserialize)]
struct Run {
    caliper: Option<f64>,
    ids: Vec<usize>,
    score: Vec<f64>,
    is_focal: Vec<bool>,
    cluster: Vec<usize>,
    pairs_focal: Vec<usize>,
    pairs_control: Vec<usize>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/vector_matching.json"
    ))
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let r = reference();
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.matching, "4.10.15");
}

#[test]
fn matching_within_clusters_agrees_with_matching_match() {
    let r = reference();
    let mut dropped_by_caliper = 0;
    for (k, run) in r.runs.iter().enumerate() {
        let ids: Vec<String> = run.ids.iter().map(ToString::to_string).collect();
        let pairs =
            match_within_clusters(&ids, &run.score, &run.is_focal, &run.cluster, run.caliper)
                .unwrap();
        let mut ours: Vec<(usize, usize)> = pairs
            .iter()
            .map(|p| {
                (
                    p.anchor_id().parse().unwrap(),
                    p.comparator_id().parse().unwrap(),
                )
            })
            .collect();
        let mut theirs: Vec<(usize, usize)> = run
            .pairs_focal
            .iter()
            .copied()
            .zip(run.pairs_control.iter().copied())
            .collect();
        ours.sort_unstable();
        theirs.sort_unstable();
        if ours != theirs {
            let a: HashSet<_> = ours.iter().collect();
            let b: HashSet<_> = theirs.iter().collect();
            panic!(
                "run {k}: only ours {:?}, only R {:?}",
                a.difference(&b).collect::<Vec<_>>(),
                b.difference(&a).collect::<Vec<_>>()
            );
        }
        let focal = run.is_focal.iter().filter(|f| **f).count();
        dropped_by_caliper += focal - ours.len();
    }
    assert!(dropped_by_caliper > 0, "the calipers refuse some matches");
}

/// A uniform number from a linear congruential generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u = self.next().max(1e-12);
        let v = self.next();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `n` units in `k` groups with two covariates that the group depends on.
fn simulate(n: usize, k: usize, seed: u64) -> (Vec<BalanceRecord>, Vec<usize>) {
    let mut rng = Lcg(seed);
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).unwrap();
    let mut records = Vec::new();
    let mut groups = Vec::new();
    for i in 0..n {
        let (a, b) = (rng.normal(), rng.normal());
        let mut weights = vec![1.0];
        for g in 1..k {
            let s = g as f64;
            weights.push((0.5 * s * a - 0.4 * b * (s - 1.0)).exp());
        }
        let total: f64 = weights.iter().sum();
        let mut draw = rng.next() * total;
        let mut group = k - 1;
        for (g, w) in weights.iter().enumerate() {
            if draw < *w {
                group = g;
                break;
            }
            draw -= w;
        }
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
fn the_final_cohort_has_a_match_in_every_comparison() {
    let (records, groups) = simulate(900, 3, 7);
    let spec = VectorMatchSpec::new(0, 3);
    let out = vector_match(&records, &groups, 3, &spec).unwrap();
    assert_eq!(out.comparisons.len(), 2);
    assert!(out.matched_focal.len() > 10);
    let group_of: HashMap<&str, usize> = records
        .iter()
        .zip(&groups)
        .map(|(r, g)| (r.core.id.as_str(), *g))
        .collect();
    let position: HashMap<&str, usize> = out
        .kept
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();
    let cohort: HashSet<&str> = out.matched_focal.iter().map(String::as_str).collect();
    for comparison in &out.comparisons {
        let anchors: HashSet<&str> = comparison
            .outcome
            .pairs
            .iter()
            .map(MatchedPair::anchor_id)
            .collect();
        assert_eq!(anchors, cohort, "group {}", comparison.group);
        // Standard deviation of the focal logit score within each cluster of
        // the units of the two groups.
        let mut scores: HashMap<usize, Vec<f64>> = HashMap::new();
        for (row, id) in out.kept.iter().enumerate() {
            let g = group_of[id.as_str()];
            if g == 0 || g == comparison.group {
                scores
                    .entry(comparison.cluster[row])
                    .or_default()
                    .push(out.gps.logit(row, 0).unwrap());
            }
        }
        for pair in &comparison.outcome.pairs {
            assert_eq!(group_of[pair.anchor_id()], 0);
            assert_eq!(group_of[pair.comparator_id()], comparison.group);
            let (a, c) = (position[pair.anchor_id()], position[pair.comparator_id()]);
            assert_eq!(comparison.cluster[a], comparison.cluster[c], "same cluster");
            let values = &scores[&comparison.cluster[a]];
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64)
                .sqrt();
            let gap = (out.gps.logit(a, 0).unwrap() - out.gps.logit(c, 0).unwrap()).abs();
            assert!(gap <= 0.25 * sd * (1.0 + 1e-12), "caliper {gap} vs {sd}");
        }
    }
    // With replacement, one unit serves several pairs somewhere.
    let reused = out.comparisons.iter().any(|c| {
        let mut seen = HashSet::new();
        c.outcome
            .pairs
            .iter()
            .any(|p| !seen.insert(p.comparator_id()))
    });
    assert!(reused);
}

#[test]
fn common_support_drops_units_outside_the_ranges_of_every_group() {
    let (records, groups) = simulate(900, 3, 11);
    let mut spec = VectorMatchSpec::new(0, 2);
    spec.refit = false;
    let out = vector_match(&records, &groups, 3, &spec).unwrap();
    assert!(out.discarded.len() > 10);
    assert_eq!(out.discarded.len() + out.kept.len(), records.len());
    // The rule, applied to the fit on all units.
    let gps = generalised_propensity_score(&records, &groups, 3, &GpsConfig::default()).unwrap();
    let mut expected = HashSet::new();
    for l in 0..3 {
        let range = |g: usize| {
            let v: Vec<f64> = (0..records.len())
                .filter(|&i| groups[i] == g)
                .map(|i| gps.probability(i, l))
                .collect();
            (
                v.iter().copied().fold(f64::INFINITY, f64::min),
                v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        let low = (0..3).map(|g| range(g).0).fold(f64::NEG_INFINITY, f64::max);
        let high = (0..3).map(|g| range(g).1).fold(f64::INFINITY, f64::min);
        for (i, r) in records.iter().enumerate() {
            let p = gps.probability(i, l);
            if !(p > low && p < high) {
                expected.insert(r.core.id.clone());
            }
        }
    }
    let got: HashSet<String> = out.discarded.iter().cloned().collect();
    assert_eq!(got, expected);

    let mut off = VectorMatchSpec::new(0, 2);
    off.common_support = false;
    assert_eq!(
        vector_match(&records, &groups, 3, &off)
            .unwrap()
            .discarded
            .len(),
        0
    );
}

#[test]
fn two_groups_need_no_clusters_and_the_match_is_reproducible() {
    let (records, groups) = simulate(500, 2, 3);
    let spec = VectorMatchSpec::new(1, 4);
    let first = vector_match(&records, &groups, 2, &spec).unwrap();
    let again = vector_match(&records, &groups, 2, &spec).unwrap();
    assert_eq!(first.comparisons.len(), 1);
    assert!(first.comparisons[0].cluster.iter().all(|c| *c == 0));
    assert_eq!(first.matched_focal, again.matched_focal);
    let pairs = |o: &epistat_matching::multigroup::VectorMatchOutcome| -> Vec<(String, String)> {
        o.comparisons[0]
            .outcome
            .pairs
            .iter()
            .map(|p| (p.anchor_id().to_string(), p.comparator_id().to_string()))
            .collect()
    };
    assert_eq!(pairs(&first), pairs(&again));
}

#[test]
fn matching_reduces_the_imbalance_of_each_pair_of_groups() {
    let (records, groups) = simulate(1200, 3, 5);
    let spec = VectorMatchSpec::new(0, 3);
    let out = vector_match(&records, &groups, 3, &spec).unwrap();
    let reports =
        balance_by_group(&records, &groups, 0, &out, &BalanceReportOptions::default()).unwrap();
    assert_eq!(reports.len(), 2);
    for (group, report) in &reports {
        let a = report.numeric.iter().find(|n| n.name == "a").unwrap();
        let before = a.smd_pre.unwrap().abs();
        let after = a.smd_post.unwrap().abs();
        assert!(after < before, "group {group}: {after} against {before}");
    }
}

#[test]
fn bad_input_is_an_error() {
    let (records, groups) = simulate(200, 3, 1);
    assert!(vector_match(&records, &groups, 3, &VectorMatchSpec::new(3, 2)).is_err());
    assert!(vector_match(&records, &groups, 3, &VectorMatchSpec::new(0, 0)).is_err());
    let mut spec = VectorMatchSpec::new(0, 2);
    spec.caliper = Some(0.0);
    assert!(vector_match(&records, &groups, 3, &spec).is_err());
    assert!(vector_match(&records, &groups[1..], 3, &VectorMatchSpec::new(0, 2)).is_err());
    assert!(vector_match(&records, &groups, 1, &VectorMatchSpec::new(0, 2)).is_err());
}
