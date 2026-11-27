//! Nearest neighbour matching on distances the crate prepares itself from the
//! covariates, checked against R 4.6's `MatchIt` 4.8.2 with
//! `distance = "mahalanobis"`, `"robust_mahalanobis"`, `"euclidean"` and
//! `"scaled_euclidean"`.
//!
//! The reference is `tests/data/matchit_prepared_distances.json`, written by
//! `tests/data/matchit_prepared_distances.R`. For a sample of lalonde and a
//! simulated set, each with a factor and binary covariates, it holds the
//! covariates and, for every distance and for `m.order` `data`, `closest` and
//! `farthest`, ratios 1 to 3, with replacement, with a caliper on a
//! covariate, with exact matching and for both estimands, the pairs
//! `matchit()` returned. No treated unit has two controls at the same
//! distance, so `matchit()` has no tie to break.
//!
//! The crate prepares each distance from the covariates with
//! `prepare_mahalanobis_distance_config` (the defaults, including the
//! pooled within-group covariance; the
//! full-sample covariance of tie-adjusted ranks for the robust distance) and
//! `prepare_euclidean_distance_config` (the plain and the pooled
//! within-group scale) and must give `matchit()`'s pairs in every run. The
//! preparation's distance values are compared with MatchIt's in
//! `r_distances.rs`; here the selection is compared.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::constraints::caliper_on_field;
use epistat_matching::prelude::*;
use epistat_matching::{
    EuclideanPreparationConfig, EuclideanScale, MahalanobisCovarianceStrategy,
    MahalanobisPreparationConfig, MahalanobisTransform, MatchOrder,
    prepare_euclidean_distance_config, prepare_mahalanobis_distance_config,
};

use r_reference::{Versions, assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Deserialize)]
struct Reference {
    refusals: BTreeMap<String, String>,
    versions: Versions,
    data: BTreeMap<String, Data>,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Data {
    treat: Vec<u8>,
    numerics: BTreeMap<String, Vec<f64>>,
    factors: BTreeMap<String, Vec<String>>,
    s: Vec<String>,
    age_sd: f64,
}

#[derive(Deserialize)]
struct Spec {
    order: String,
    ratio: usize,
    replace: bool,
    covcal: Option<f64>,
    covcal_std: Option<f64>,
    exact: bool,
    estimand: String,
}

#[derive(Deserialize)]
struct Run {
    data: String,
    distance: String,
    spec: Spec,
    focal_rows: Vec<usize>,
    matchit: Vec<Rows>,
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
    load_reference("matchit_prepared_distances.json")
}

fn order_of(name: &str) -> MatchOrder {
    match name {
        "data" => MatchOrder::Data,
        "closest" => MatchOrder::Closest,
        "farthest" => MatchOrder::Farthest,
        other => panic!("order {other}"),
    }
}

fn split(data: &Data, focal: u8) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (row, treat) in data.treat.iter().enumerate() {
        let mut record = BalanceRecord::new((row + 1).to_string(), birth);
        for (name, values) in &data.numerics {
            record
                .covariates
                .insert(name.clone(), CovariateValue::Numeric(values[row]));
        }
        for (name, values) in &data.factors {
            record.covariates.insert(
                name.clone(),
                CovariateValue::Categorical(values[row].clone()),
            );
        }
        record
            .core
            .numerics
            .insert("age".to_string(), data.numerics["age"][row]);
        record
            .core
            .strata
            .insert("s".to_string(), data.s[row].clone());
        if *treat == focal {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    (anchors, candidates)
}

fn prepared(
    distance: &str,
    anchors: &[BalanceRecord],
    candidates: &[BalanceRecord],
    caliper: Option<DistanceCaliper>,
) -> DistanceConfig {
    match distance {
        "mahalanobis" => {
            prepare_mahalanobis_distance_config(
                anchors,
                candidates,
                caliper,
                &MahalanobisPreparationConfig::default(),
            )
            .expect("preparation")
            .distance_config
        }
        "robust_mahalanobis" => {
            let config = MahalanobisPreparationConfig::builder()
                .covariance_strategy(MahalanobisCovarianceStrategy::FullSample)
                .transform(MahalanobisTransform::RankTieAdjusted)
                .build();
            prepare_mahalanobis_distance_config(anchors, candidates, caliper, &config)
                .expect("preparation")
                .distance_config
        }
        "euclidean" | "scaled_euclidean" => {
            let scale = if distance == "euclidean" {
                EuclideanScale::None
            } else {
                EuclideanScale::PooledWithinGroupSd
            };
            let config = EuclideanPreparationConfig::builder().scale(scale).build();
            prepare_euclidean_distance_config(anchors, candidates, caliper, &config)
                .expect("preparation")
                .distance_config
        }
        other => panic!("distance {other}"),
    }
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let reference = reference();
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
}

#[test]
fn prepared_distances_give_matchits_pairs() {
    let reference = reference();
    let mut compared = 0;
    for (number, run) in reference.runs.iter().enumerate() {
        let data = &reference.data[&run.data];
        let spec = &run.spec;
        let focal = u8::from(spec.estimand != "ATC");
        let (anchors, candidates) = split(data, focal);
        let config = prepared(&run.distance, &anchors, &candidates, None);
        // A caliper on a covariate in standard deviation units is the raw
        // width times the covariate's sample standard deviation over all units.
        let covariate_width = spec
            .covcal
            .or_else(|| spec.covcal_std.map(|width| width * data.age_sd));
        let calipers: Vec<_> = covariate_width
            .map(|width| caliper_on_field::<BalanceRecord>("age", width))
            .into_iter()
            .collect();
        let outcome = MatchJob::new_standard(&anchors, &candidates, 0)
            .with_nearest()
            .with_m_order(order_of(&spec.order))
            .with_ratio(MatchRatio::new(spec.ratio).expect("ratio"))
            .with_replacement(spec.replace)
            .with_estimand(if focal == 0 {
                Estimand::Atc
            } else {
                Estimand::Att
            })
            .with_exact_matches(if spec.exact { vec!["s"] } else { Vec::new() })
            .with_distance_config(config)
            .with_constraints(calipers)
            .run()
            .expect("a valid job");
        let mut by_anchor: HashMap<usize, Vec<usize>> = HashMap::new();
        for pair in &outcome.pairs {
            by_anchor
                .entry(pair.anchor_id().parse().expect("row"))
                .or_default()
                .push(pair.comparator_id().parse().expect("row"));
        }
        let ours: Vec<Vec<usize>> = run
            .focal_rows
            .iter()
            .map(|row| by_anchor.remove(row).unwrap_or_default())
            .collect();
        let expected: Vec<Vec<usize>> = run.matchit.iter().map(Rows::to_vec).collect();
        assert_eq!(
            ours,
            expected,
            "run {number} {} {} order {} ratio {} replace {} covcal {:?} exact {} {}",
            run.data,
            run.distance,
            spec.order,
            spec.ratio,
            spec.replace,
            spec.covcal,
            spec.exact,
            spec.estimand
        );
        compared += 1;
    }
    assert_eq!(compared, reference.runs.len());
}

/// `matchit()` refuses a caliper on the distance itself when the distance is
/// a covariate distance: every caliper must name a covariate. It follows that
/// `std.caliper` has no meaning there. The crate refuses a caliper in standard
/// deviation units on such a distance too. A caliper in the distance's own
/// units is accepted as an extension, since a Mahalanobis or Euclidean
/// distance has units a user can reason about, and a caliper on a covariate is
/// an ordinary constraint (above).
#[test]
fn a_caliper_in_standard_deviations_of_a_covariate_distance_is_refused() {
    let reference = reference();
    for message in reference.refusals.values() {
        assert!(message.contains("must be named"), "{message}");
    }
    let data = &reference.data["sim"];
    let (anchors, candidates) = split(data, 1);
    for distance in [
        "mahalanobis",
        "robust_mahalanobis",
        "euclidean",
        "scaled_euclidean",
    ] {
        let width = DistanceCaliper::new(2.0).expect("positive");
        let config = prepared(distance, &anchors, &candidates, Some(width));
        let refused = MatchJob::new_standard(&anchors, &candidates, 0)
            .with_nearest()
            .with_std_caliper(true)
            .with_distance_config(config)
            .run();
        assert_eq!(
            refused.unwrap_err(),
            Error::InvalidCriteria(InvalidCriteriaReason::NearestOptions),
            "{distance}"
        );
    }
}
