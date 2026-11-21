//! Optimal pair matching checked against R 4.6's `MatchIt` 4.8.2 and
//! `optmatch` 0.10.8.
//!
//! The reference is `tests/data/matchit_optimal.json`, written by
//! `tests/data/matchit_optimal.R`. It holds 66 runs of
//! `matchit(method = "optimal")` on six simulated data sets, one of which has
//! exact strata, with estimands ATT and ATC, ratios 1 to 3, with and without
//! the strata. The data sets include one with fewer than `ratio` controls per
//! treated unit, one with fewer controls than treated units, and one with
//! rounded scores, so that optimal matches tie. Three further data sets give
//! the same designs with a Euclidean distance of two covariates, whose optima
//! are unique. Alongside are 22 runs with calipers. `matchit()` ignores
//! calipers with `method = "optimal"`, so those call `optmatch::fullmatch()`
//! on distances with the pairs outside the caliper forbidden.
//!
//! What is checked, on every run:
//!
//! * The total distance of the pairs. The crate solves with costs of about 40
//!   bits, so its total is the optimum to `3 n d_max / 2^40` for `n` pairs and
//!   largest distance `d_max`. optmatch rounds the distances to integers with a
//!   resolution `epsilon` (in the file), and its total is at most
//!   `1.5 epsilon` per pair above the optimum. So the crate's total is not
//!   above R's by more than its own bound, and R's is not above the crate's by
//!   more than R's bound plus the crate's.
//! * The number of pairs and, per anchor, the number of candidates.
//! * Where the optimum is unique (six perturbations of every distance by 1e-5
//!   standard deviations of the score give R's pairs each time) the pairs, and
//!   MatchIt's weights, equal the crate's. Distances on a line tie often, so
//!   only some of the runs on scores are unique, and every run on Euclidean
//!   distances is.
//! * Where R finds no feasible matching, the crate reports
//!   [`OptimalError::Infeasible`].
//!
//! The weights in the file carry 17 significant digits. The crate's weights are
//! reciprocals of small integers rescaled by a mean, so they agree with R's to
//! a few ulps and are compared to a relative 1e-12.

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{RapidHashMap, match_optimal};
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    scenarios: BTreeMap<String, Scenario>,
    runs: Vec<Run>,
    caliper_runs: Vec<CaliperRun>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    matchit: String,
    optmatch: String,
    solver: String,
}

#[derive(Deserialize)]
struct Scenario {
    treat: Vec<u8>,
    x: Vec<f64>,
    g: Vec<String>,
    x2: Vec<f64>,
}

/// R writes a vector of one element as a bare number.
#[derive(Deserialize)]
#[serde(untagged)]
enum Ints {
    Many(Vec<usize>),
    One(usize),
}

impl Ints {
    fn to_vec(&self) -> Vec<usize> {
        match self {
            Self::Many(values) => values.clone(),
            Self::One(value) => vec![*value],
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Floats {
    Many(Vec<f64>),
    One(f64),
}

impl Floats {
    fn to_vec(&self) -> Vec<f64> {
        match self {
            Self::Many(values) => values.clone(),
            Self::One(value) => vec![*value],
        }
    }
}

#[derive(Deserialize)]
struct Run {
    scenario: String,
    estimand: String,
    ratio: usize,
    exact: bool,
    error: bool,
    focal: Option<Ints>,
    other: Option<Ints>,
    weights: Option<Floats>,
    total: Option<f64>,
    r_epsilon: Option<f64>,
    unique: Option<bool>,
}

#[derive(Deserialize)]
struct CaliperRun {
    scenario: String,
    caliper: f64,
    ratio: usize,
    error: bool,
    focal: Option<Ints>,
    other: Option<Ints>,
    total: Option<f64>,
    r_epsilon: Option<f64>,
    unique: Option<bool>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/matchit_optimal.json"
    ))
}

fn euclidean(name: &str) -> bool {
    name.starts_with("euclid")
}

struct Setup {
    anchors: Vec<BaseRecord>,
    candidates: Vec<BaseRecord>,
    distance: DistanceConfig,
}

/// The units of a run with the focal group as anchors, ids the row numbers of
/// the data set from 1, and the distance of the scenario.
fn setup(name: &str, scenario: &Scenario, focal: u8, caliper: Option<f64>) -> Setup {
    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    let width = caliper.map(|width| DistanceCaliper::new(width).expect("non-negative caliper"));
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    let mut scores: RapidHashMap<String, f64> = RapidHashMap::default();
    let mut vectors: RapidHashMap<String, Vec<f64>> = RapidHashMap::default();
    for (row, treat) in scenario.treat.iter().enumerate() {
        let id = (row + 1).to_string();
        let mut record = BaseRecord::new(id.clone(), birth);
        record
            .strata
            .insert("g".to_string(), scenario.g[row].clone());
        scores.insert(id.clone(), scenario.x[row]);
        vectors.insert(id, vec![scenario.x[row], scenario.x2[row]]);
        if *treat == focal {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    let distance = if euclidean(name) {
        DistanceConfig::euclidean_map(vectors, 2, width).expect("two columns")
    } else {
        DistanceConfig::propensity_score_map(scores, width)
    };
    Setup {
        anchors,
        candidates,
        distance,
    }
}

fn run_ours(setup: &Setup, ratio: usize, exact: bool) -> Result<MatchOutcome, OptimalError> {
    let criteria = MatchingCriteria::builder()
        .match_ratio(ratio)
        .required_strata(if exact {
            vec!["g".to_string()]
        } else {
            Vec::new()
        })
        .build();
    let request = OptimalRequest::new(&criteria, &()).with_distance_config(&setup.distance);
    match_optimal(&setup.anchors, &setup.candidates, &request)
}

fn row(id: &str) -> usize {
    id.parse().expect("row number")
}

fn pairs_of(outcome: &MatchOutcome) -> BTreeSet<(usize, usize)> {
    outcome
        .pairs
        .iter()
        .map(|pair| (row(&pair.case_id), row(&pair.control_id)))
        .collect()
}

/// The distance of a pair, as the data define it.
fn distance(name: &str, scenario: &Scenario, a: usize, b: usize) -> f64 {
    let dx = scenario.x[a - 1] - scenario.x[b - 1];
    if euclidean(name) {
        let dy = scenario.x2[a - 1] - scenario.x2[b - 1];
        (dx * dx + dy * dy).sqrt()
    } else {
        dx.abs()
    }
}

fn total(name: &str, scenario: &Scenario, pairs: &BTreeSet<(usize, usize)>) -> f64 {
    pairs
        .iter()
        .map(|(a, b)| distance(name, scenario, *a, *b))
        .sum()
}

fn r_pairs(focal: &Ints, other: &Ints) -> BTreeSet<(usize, usize)> {
    focal
        .to_vec()
        .into_iter()
        .zip(other.to_vec())
        .collect::<BTreeSet<_>>()
}

fn max_distance(name: &str, scenario: &Scenario) -> f64 {
    let n = scenario.treat.len();
    let mut largest = 0.0_f64;
    for a in 1..=n {
        for b in (a + 1)..=n {
            if scenario.treat[a - 1] != scenario.treat[b - 1] {
                largest = largest.max(distance(name, scenario, a, b));
            }
        }
    }
    largest
}

/// The totals agree to the arithmetic of the two solvers.
fn check_total(label: &str, ours: f64, r_total: f64, pairs: usize, r_epsilon: f64, d_max: f64) {
    // Costs are rounded to about 2^-40 of the largest distance, and a power of
    // two below that, so each pair is off by at most 3 d_max / 2^40.
    let our_bound = 3.0 * pairs as f64 * d_max / 2_f64.powi(40) + 1e-12;
    // optmatch rounds up to 1.5 epsilon per pair.
    let r_bound = 1.5 * pairs as f64 * r_epsilon;
    assert!(
        ours <= r_total + our_bound,
        "{label}: crate total {ours} above R's {r_total}"
    );
    assert!(
        r_total <= ours + r_bound + our_bound,
        "{label}: R total {r_total} above the crate's {ours} by more than {r_bound}"
    );
}

fn check_weights(label: &str, outcome: &MatchOutcome, scenario: &Scenario, expected: &[f64]) {
    let weights = outcome
        .match_weights(MatchWeightMethod::AnchorUnitCandidateFractional)
        .normalized();
    for (index, want) in expected.iter().enumerate() {
        let id = (index + 1).to_string();
        let got = weights
            .anchor
            .get(&id)
            .or_else(|| weights.candidate.get(&id))
            .copied()
            .unwrap_or(0.0);
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "{label}: weight of row {} is {got}, R has {want}",
            index + 1
        );
    }
    assert_eq!(expected.len(), scenario.treat.len(), "{label}");
}

#[test]
fn pairs_totals_and_weights_match_matchit() {
    let reference = reference();
    println!(
        "{} / MatchIt {} / optmatch {} / solver {}",
        reference.versions.r,
        reference.versions.matchit,
        reference.versions.optmatch,
        reference.versions.solver
    );
    let (mut compared, mut pairs_compared, mut errors) = (0, 0, 0);
    for run in &reference.runs {
        let scenario = &reference.scenarios[&run.scenario];
        let label = format!(
            "{} {} ratio {} exact {}",
            run.scenario, run.estimand, run.ratio, run.exact
        );
        let focal = u8::from(run.estimand == "ATT");
        let built = setup(&run.scenario, scenario, focal, None);
        let ratio = run.ratio;
        let got = run_ours(&built, ratio, run.exact);
        if run.error {
            assert!(
                matches!(
                    got,
                    Err(OptimalError::Infeasible { .. } | OptimalError::NoMatches)
                ),
                "{label}: R failed but the crate gave {got:?}"
            );
            errors += 1;
            continue;
        }
        let outcome = got.unwrap_or_else(|error| panic!("{label}: {error}"));
        let r_set = r_pairs(
            run.focal.as_ref().expect("focal"),
            run.other.as_ref().expect("other"),
        );
        let ours = pairs_of(&outcome);
        assert_eq!(ours.len(), r_set.len(), "{label}: number of pairs");
        let mut per_anchor: BTreeMap<usize, usize> = BTreeMap::new();
        for (anchor, _) in &ours {
            *per_anchor.entry(*anchor).or_default() += 1;
        }
        let mut r_per_anchor: BTreeMap<usize, usize> = BTreeMap::new();
        for (anchor, _) in &r_set {
            *r_per_anchor.entry(*anchor).or_default() += 1;
        }
        assert_eq!(
            per_anchor.values().copied().collect::<BTreeSet<_>>(),
            r_per_anchor.values().copied().collect::<BTreeSet<_>>(),
            "{label}: candidates per anchor"
        );
        let our_total = total(&run.scenario, scenario, &ours);
        check_total(
            &label,
            our_total,
            run.total.expect("total"),
            ours.len(),
            run.r_epsilon.expect("epsilon"),
            max_distance(&run.scenario, scenario),
        );
        compared += 1;
        if run.unique == Some(true) {
            assert_eq!(ours, r_set, "{label}: pairs");
            check_weights(
                &label,
                &outcome,
                scenario,
                &run.weights.as_ref().expect("weights").to_vec(),
            );
            pairs_compared += 1;
        }
    }
    println!("{compared} totals, {pairs_compared} unique pairs, {errors} refusals compared");
    assert!(compared >= 60, "only {compared} totals compared");
    assert!(pairs_compared >= 25, "only {pairs_compared} unique runs");
}

#[test]
fn calipers_forbid_pairs_as_optmatch_does() {
    let reference = reference();
    let (mut compared, mut pairs_compared, mut errors) = (0, 0, 0);
    for run in &reference.caliper_runs {
        let scenario = &reference.scenarios[&run.scenario];
        let label = format!(
            "{} caliper {} ratio {}",
            run.scenario, run.caliper, run.ratio
        );
        let built = setup(&run.scenario, scenario, 1, Some(run.caliper));
        let ratio = run.ratio;
        let got = run_ours(&built, ratio, false);
        if run.error {
            assert!(
                matches!(
                    got,
                    Err(OptimalError::Infeasible { .. } | OptimalError::NoMatches)
                ),
                "{label}: optmatch failed but the crate gave {got:?}"
            );
            errors += 1;
            continue;
        }
        let outcome = got.unwrap_or_else(|error| panic!("{label}: {error}"));
        let ours = pairs_of(&outcome);
        let r_set = r_pairs(
            run.focal.as_ref().expect("focal"),
            run.other.as_ref().expect("other"),
        );
        assert_eq!(ours.len(), r_set.len(), "{label}: number of pairs");
        for (a, b) in &ours {
            assert!(
                distance(&run.scenario, scenario, *a, *b) <= run.caliper,
                "{label}: pair {a}-{b} is outside the caliper"
            );
        }
        check_total(
            &label,
            total(&run.scenario, scenario, &ours),
            run.total.expect("total"),
            ours.len(),
            run.r_epsilon.expect("epsilon"),
            max_distance(&run.scenario, scenario),
        );
        compared += 1;
        if run.unique == Some(true) {
            assert_eq!(ours, r_set, "{label}: pairs");
            pairs_compared += 1;
        }
    }
    println!("{compared} totals, {pairs_compared} unique pairs, {errors} refusals compared");
    assert!(compared >= 10, "only {compared} caliper totals compared");
    assert!(errors >= 1, "no caliper run was infeasible in R");
}

#[test]
fn replacement_and_unique_keys_are_refused() {
    let reference = reference();
    let scenario = &reference.scenarios["wide"];
    let built = setup("wide", scenario, 1, None);
    let criteria = MatchingCriteria::builder().allow_replacement(true).build();
    let request = OptimalRequest::new(&criteria, &()).with_distance_config(&built.distance);
    assert!(matches!(
        match_optimal(&built.anchors, &built.candidates, &request),
        Err(OptimalError::Option {
            option: "replace",
            ..
        })
    ));
}
