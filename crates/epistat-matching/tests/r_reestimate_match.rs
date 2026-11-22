//! Matching on a propensity score that is fitted again after common-support
//! trimming, checked against R 4.6's `MatchIt` 4.8.2,
//! `matchit(method = "nearest", discard = policy, reestimate = TRUE)`.
//!
//! The reference is `tests/data/matchit_reestimate_match.json`, written by
//! `tests/data/matchit_reestimate_match.R`. It holds, for lalonde and two
//! simulated sets, each trimming policy, ratios 1 and 2 and a caliper of 0.25
//! standard deviations of the distance, the pairs of `matchit()` and the units
//! it discarded. `matchit()` finds the discarded units once, from the first
//! fit, and matches the kept units on the refitted distance. The crate must
//! give the same pairs and the same discarded units through
//! `estimate_propensity_and_match`. The scores are those of a fit that agrees
//! with R's to about 1e-13 (see `r_propensity.rs`), so a pair would differ
//! only where two candidates' distances lie within that of each other, which
//! the continuous covariates make absent.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{CommonSupport, NearestSelection, estimate_propensity_and_match};

use r_reference::{assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    data: BTreeMap<String, Data>,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    matchit: String,
}

#[derive(Deserialize)]
struct Data {
    treat: Vec<u8>,
    names: Vec<String>,
    covariates: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct Run {
    data: String,
    policy: String,
    ratio: usize,
    caliper: Option<f64>,
    treated_rows: Vec<usize>,
    matchit: Vec<Rows>,
    discarded: Vec<usize>,
}

/// R writes a vector of one element as a bare number.
#[derive(Deserialize)]
#[serde(untagged)]
enum Rows {
    Many(Vec<usize>),
    One(usize),
}

impl Rows {
    fn to_vec(&self) -> Vec<usize> {
        match self {
            Self::Many(values) => values.clone(),
            Self::One(value) => vec![*value],
        }
    }
}

fn reference() -> Reference {
    load_reference("matchit_reestimate_match.json")
}

fn policy_of(name: &str) -> CommonSupport {
    match name {
        "treated" => CommonSupport::Treated,
        "control" => CommonSupport::Control,
        "both" => CommonSupport::Both,
        other => panic!("unknown policy {other}"),
    }
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let reference = reference();
    assert_r_version(&reference.versions.r);
    assert_eq!(reference.versions.matchit, "4.8.2");
}

#[test]
fn matching_after_reestimation_keeps_the_units_matchit_keeps() {
    let reference = reference();
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut runs_with_discards = 0;
    for run in &reference.runs {
        let data = &reference.data[&run.data];
        let mut anchors = Vec::new();
        let mut candidates = Vec::new();
        for (row, label) in data.treat.iter().enumerate() {
            let mut record = BalanceRecord::new((row + 1).to_string(), birth);
            for (name, column) in data.names.iter().zip(&data.covariates) {
                record
                    .covariates
                    .insert(name.clone(), CovariateValue::Numeric(column[row]));
            }
            if *label == 1 {
                anchors.push(record);
            } else {
                candidates.push(record);
            }
        }
        let criteria = MatchingCriteria::builder()
            .match_ratio(run.ratio)
            .common_support(policy_of(&run.policy))
            .build();
        let config = PropensityScoreConfig::builder()
            .reestimate(true)
            .maybe_caliper(
                run.caliper
                    .map(|width| DistanceCaliper::new(width).expect("positive")),
            )
            .build();
        let matched = estimate_propensity_and_match(
            &anchors,
            &candidates,
            &criteria,
            NearestSelection::new(0).with_std_caliper(run.caliper.is_some()),
            &[],
            &(),
            &config,
        )
        .expect("the fit succeeds");

        let label = format!(
            "{} {} ratio {} caliper {:?}",
            run.data, run.policy, run.ratio, run.caliper
        );
        let mut discarded: Vec<usize> = matched
            .propensity
            .discarded_anchors
            .iter()
            .chain(&matched.propensity.discarded_candidates)
            .map(|id| id.parse().expect("row"))
            .collect();
        discarded.sort_unstable();
        assert_eq!(discarded, run.discarded, "{label}: discarded");

        let mut by_anchor: HashMap<usize, Vec<usize>> = HashMap::new();
        for pair in &matched.outcome.pairs {
            by_anchor
                .entry(pair.anchor_id().parse().expect("row"))
                .or_default()
                .push(pair.comparator_id().parse().expect("row"));
        }
        let ours: Vec<Vec<usize>> = run
            .treated_rows
            .iter()
            .map(|row| by_anchor.remove(row).unwrap_or_default())
            .collect();
        let expected: Vec<Vec<usize>> = run.matchit.iter().map(Rows::to_vec).collect();
        assert_eq!(ours, expected, "{label}: pairs");
        let reported = &matched.outcome.diagnostics.common_support_trimmed_ids;
        let mut reported_rows: Vec<usize> = reported
            .anchors
            .iter()
            .chain(&reported.candidates)
            .map(|id| id.parse().expect("row"))
            .collect();
        reported_rows.sort_unstable();
        assert_eq!(reported_rows, run.discarded, "{label}: reported ids");
        assert_eq!(
            matched.outcome.diagnostics.common_support_trimmed_anchors
                + matched
                    .outcome
                    .diagnostics
                    .common_support_trimmed_candidates,
            run.discarded.len(),
            "{label}: trimmed count"
        );
        if !run.discarded.is_empty() {
            runs_with_discards += 1;
        }
    }
    assert!(runs_with_discards > 10);
}
