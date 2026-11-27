//! Variable ratio nearest neighbour matching (`min.controls`, `max.controls`)
//! checked against R 4.6's `MatchIt` 4.8.2.
//!
//! The reference is `tests/data/matchit_variable_ratio.json`, written by
//! `tests/data/matchit_variable_ratio.R`. Its runs cover every deterministic
//! `m.order`, averages of 1.5 to 3 with minimum 1 or 2 and maximum 2 to 12,
//! with and without replacement, for both estimands, with a caliper and with
//! exact matching, on a probability and linear score, on scores on a grid
//! (many ties), on a set in which the anchors have the lower scores, and with
//! `mahvars`.
//!
//! Besides what `matchit()` returned, every run holds the pairs of a short
//! loop written in the script from the paper and the documentation, the
//! numbers of controls it used and the number of ties it met. The checks are:
//!
//! * The crate gives the loop's pairs and the loop's numbers of controls in
//!   every run.
//! * In every run without a tie, and in every run where `matchit()`'s numbers
//!   of controls are the paper's, the crate gives `matchit()`'s pairs. Where
//!   the distances tie, `matchit()` does not break ties by row order (see
//!   `r_nearest.rs`), so those runs are checked against the loop only.
//! * Where `matchit()` and the crate give the same pairs, the normalized
//!   weights are the same.
//! * `matchit()` takes the number of controls of the unit between those with
//!   the minimum and those with the maximum from a formula that gives it fewer
//!   than `min.controls` for some totals when `min.controls` is above 1. The
//!   runs flagged `defect` are those. The script asserts that `matchit()`
//!   gives a unit fewer than `min.controls` where it runs with replacement
//!   and no caliper, and this test asserts it too, and that the crate keeps to
//!   the minimum and to the same total.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{DistanceCaliper, DistanceConfig, Estimand, MatchWeightMethod};
use epistat_matching::{MatchOrder, VariableRatio};
use epistat_testkit::Tolerance;
use r_reference::{Versions, assert_r_version, reference as load_reference};
use rapidhash::RapidHashMap;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

/// Weights are sums of a few reciprocals of small integers divided by a mean,
/// exact to a few units in the last place.
const WEIGHT_TOLERANCE: Tolerance = Tolerance::exact().abs(1e-12);

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    scenarios: BTreeMap<String, Scenario>,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Scenario {
    treat: Vec<u8>,
    s: Vec<String>,
    age: Vec<f64>,
    dist: Vec<f64>,
    maha: Option<Vec<Vec<f64>>>,
    kind: String,
}

#[derive(Deserialize)]
struct Spec {
    order: Option<String>,
    ratio: f64,
    min: usize,
    max: usize,
    replace: bool,
    estimand: String,
    cal: Option<f64>,
    exact: bool,
}

#[derive(Deserialize)]
struct Run {
    scenario: String,
    spec: Spec,
    defect: bool,
    matchit: Vec<Rows>,
    model: Vec<Rows>,
    counts: Vec<usize>,
    matchit_counts_ok: Option<bool>,
    matchit_weights: Vec<f64>,
    ties: usize,
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

fn expected(rows: &[Rows]) -> Vec<Vec<usize>> {
    rows.iter().map(Rows::to_vec).collect()
}

fn reference() -> Reference {
    load_reference("matchit_variable_ratio.json")
}

fn birth() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date")
}

fn split(scenario: &Scenario, focal: u8) -> (Vec<usize>, Vec<BaseRecord>, Vec<BaseRecord>) {
    let mut focal_rows = Vec::new();
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (row, treat) in scenario.treat.iter().enumerate() {
        let mut record =
            BaseRecord::new((row + 1).to_string(), birth()).with_numeric("age", scenario.age[row]);
        record
            .strata
            .insert("s".to_string(), scenario.s[row].clone());
        if *treat == focal {
            focal_rows.push(row + 1);
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    (focal_rows, anchors, candidates)
}

fn scores(scenario: &Scenario) -> RapidHashMap<String, f64> {
    scenario
        .dist
        .iter()
        .enumerate()
        .map(|(row, value)| ((row + 1).to_string(), *value))
        .collect()
}

/// The Euclidean distance between the transformed covariates of `matchit()`.
fn mahalanobis(scenario: &Scenario) -> DistanceConfig {
    let columns = scenario.maha.as_ref().expect("covariates");
    let dimension = columns.len();
    let vectors: RapidHashMap<String, Vec<f64>> = (0..scenario.treat.len())
        .map(|row| {
            (
                (row + 1).to_string(),
                columns.iter().map(|column| column[row]).collect(),
            )
        })
        .collect();
    let identity = (0..dimension * dimension)
        .map(|k| f64::from(u8::from(k / dimension == k % dimension)))
        .collect();
    DistanceConfig::mahalanobis_map(vectors, identity, dimension, None).expect("valid")
}

fn order_of(name: &str) -> MatchOrder {
    match name {
        "largest" => MatchOrder::Largest,
        "smallest" => MatchOrder::Smallest,
        "data" => MatchOrder::Data,
        "closest" => MatchOrder::Closest,
        "farthest" => MatchOrder::Farthest,
        other => panic!("order {other}"),
    }
}

fn run_ours(scenario: &Scenario, spec: &Spec) -> (Vec<usize>, MatchOutcome) {
    let focal = u8::from(spec.estimand != "ATC");
    let (focal_rows, anchors, candidates) = split(scenario, focal);
    // The caliper is on the score, in standard deviations of the score over
    // all units, as `std.caliper = TRUE` reads it.
    let caliper = spec
        .cal
        .map(|width| DistanceCaliper::new(width).expect("positive"));
    let mut job = MatchJob::new_standard(&anchors, &candidates, 0)
        .with_nearest()
        .with_variable_ratio(
            VariableRatio::new(spec.ratio, spec.min, spec.max).expect("valid variable ratio"),
        )
        .with_replacement(spec.replace)
        .with_estimand(if focal == 0 {
            Estimand::Atc
        } else {
            Estimand::Att
        })
        .with_exact_matches(if spec.exact { vec!["s"] } else { Vec::new() })
        .with_std_caliper(true)
        .with_distance_config(DistanceConfig::propensity_score_map(
            scores(scenario),
            caliper,
        ));
    if let Some(order) = spec.order.as_deref() {
        job = job.with_m_order(order_of(order));
    }
    if scenario.kind == "mahvars" {
        job = job.with_mahvars(mahalanobis(scenario));
    }
    (focal_rows, job.run().expect("a valid job"))
}

fn rows_of(outcome: &MatchOutcome, focal_rows: &[usize]) -> Vec<Vec<usize>> {
    let mut by_anchor: HashMap<usize, Vec<usize>> = HashMap::new();
    for pair in &outcome.pairs {
        by_anchor
            .entry(pair.anchor_id().parse().expect("row number"))
            .or_default()
            .push(pair.comparator_id().parse().expect("row number"));
    }
    focal_rows
        .iter()
        .map(|row| by_anchor.remove(row).unwrap_or_default())
        .collect()
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let reference = reference();
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
}

#[test]
fn pairs_follow_the_loop_always_and_matchit_where_it_is_well_defined() {
    let reference = reference();
    let (mut compared, mut with_ties, mut defects) = (0, 0, 0);
    for (number, run) in reference.runs.iter().enumerate() {
        let scenario = &reference.scenarios[&run.scenario];
        let (focal_rows, outcome) = run_ours(scenario, &run.spec);
        let ours = rows_of(&outcome, &focal_rows);
        let label = format!(
            "run {number} {} order {:?} ratio {} min {} max {} replace {} {}",
            run.scenario,
            run.spec.order,
            run.spec.ratio,
            run.spec.min,
            run.spec.max,
            run.spec.replace,
            run.spec.estimand
        );
        assert_eq!(ours, expected(&run.model), "{label}: the loop");
        let minimum = run.spec.min;
        let total: usize = run.counts.iter().sum();
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a count of units"
        )]
        let asked = (run.spec.ratio * run.counts.len() as f64).round_ties_even() as usize;
        assert_eq!(total, asked, "{label}: the total");
        assert!(
            run.counts
                .iter()
                .all(|count| (minimum..=run.spec.max).contains(count)),
            "{label}: bounds"
        );

        if run.defect {
            defects += 1;
            if run.matchit_counts_ok == Some(false) {
                let smallest = expected(&run.matchit).iter().map(Vec::len).min();
                assert!(
                    smallest < Some(minimum),
                    "{label}: matchit under the minimum"
                );
            }
            continue;
        }
        if run.ties > 0 {
            with_ties += 1;
            continue;
        }
        assert_eq!(ours, expected(&run.matchit), "{label}: matchit");
        let (treat, weights) = (&scenario.treat, &run.matchit_weights);
        let focal = u8::from(run.spec.estimand != "ATC");
        let unit_weights = outcome
            .match_weights(MatchWeightMethod::AnchorUnitCandidateFractional)
            .normalized();
        for (unit, want) in weights.iter().enumerate() {
            let set = if treat[unit] == focal {
                &unit_weights.anchor
            } else {
                &unit_weights.candidate
            };
            let got = set.get(&(unit + 1).to_string()).copied().unwrap_or(0.0);
            assert!(
                WEIGHT_TOLERANCE.accepts(got, *want),
                "{label}: unit {} weight {got}, MatchIt {want}",
                unit + 1
            );
        }
        compared += 1;
    }
    println!(
        "{} runs: {compared} agree with matchit pair for pair and in weights; \
         {with_ties} with ties not compared; {defects} with MatchIt's numbers under the minimum",
        reference.runs.len()
    );
    assert!(compared > 100, "{compared}");
    assert!(defects > 0);
}

#[test]
fn matchit_numbers_agree_where_the_minimum_is_one() {
    let reference = reference();
    for run in reference.runs.iter().filter(|run| run.spec.min == 1) {
        assert!(!run.defect, "minimum 1 has no defect");
    }
}

#[test]
fn invalid_ratios_are_errors() {
    assert!(VariableRatio::new(1.0, 1, 3).is_err());
    assert!(VariableRatio::new(2.0, 0, 3).is_err());
    assert!(VariableRatio::new(2.0, 2, 3).is_err());
    assert!(VariableRatio::new(2.0, 1, 2).is_err());
    assert!(VariableRatio::new(f64::NAN, 1, 3).is_err());
    assert!(VariableRatio::new(2.0, 1, 3).is_ok());
}
