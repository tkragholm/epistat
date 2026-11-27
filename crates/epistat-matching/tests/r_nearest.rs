//! Nearest neighbour matching by distance checked against R 4.6's `MatchIt`
//! 4.8.2, `matchit(method = "nearest")`.
//!
//! The reference is `tests/data/matchit_nearest.json`, written by
//! `tests/data/matchit_nearest.R`. Its runs cover every deterministic
//! `m.order` (`largest`, `smallest`, `data`, `closest`, `farthest`) and the
//! default order, ratios 1 to 3 with and without replacement, a numeric
//! distance on the probability and the linear predictor scale, Mahalanobis
//! distance, `mahvars`, calipers on the distance (in standard deviation units
//! and raw) and on a covariate, `exact` and `antiexact`, and for the estimands
//! ATT and ATC. The data are a sample of `lalonde` (60 treated, 180 controls)
//! and simulated sets, one with scores on a grid so that many candidates share
//! a distance.
//!
//! Besides what `matchit()` returned, every run holds the pairs of a short
//! loop written in the script from the documented behaviour, with ties between
//! candidates at the same distance broken by row order, and the number of ties
//! that loop met. The checks are:
//!
//! * The crate gives the loop's pairs, in order of assignment per anchor, in
//!   every run, with or without ties.
//! * In every run without a tie the crate gives `matchit()`'s pairs. The script
//!   stops if the loop disagrees with `matchit()` in such a run, so the two
//!   statements are the same.
//! * Where the distances tie, `matchit()` does not break ties by row order as
//!   its documentation says. It takes the control with the highest row among
//!   those at the same distance below the treated unit's score, and the lowest
//!   row among those above it (`tie_cases`, asserted below). With both sides
//!   and the treated unit's own row in play the choice depends on the rows of
//!   every unit in between, so the crate keeps the documented rule. The runs
//!   with ties are therefore checked against the loop only, and the number
//!   that differ from `matchit()` is reported. `lalonde` has identical
//!   covariate rows and so identical scores; its `_unique` variants add a
//!   perturbation of 1e-10 to 1e-6 times the row number, far below any
//!   difference that matters, and agree with `matchit()` in every run.
//! * A random order cannot reproduce R's random numbers. It is checked for
//!   properties: no control reused without replacement, calipers and
//!   `antiexact` held, every anchor that could still have been matched is,
//!   and the same seed gives the same pairs.
//! * Rounds for the existing strategies are checked against the `matchit()`
//!   runs of `tests/data/match_pairs.json`, which give every anchor a first
//!   control before any gets a second.
//!
//! The standard deviation behind `std.caliper` is computed from the scores
//! with n - 1, as R's `sd` does, in two passes in a different order of
//! summation. The width can differ from R's in the last bit. A pair whose
//! distance equals the caliper to that bit would move, and the references
//! have none.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::constraints::caliper_on_field;
use epistat_matching::prelude::*;
use epistat_matching::{Assignment, DeterministicSelection, MatchOrder, RapidHashMap};
use epistat_testkit::load_json;
use r_reference::{Versions, assert_r_version, reference as load_reference};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    tie_cases: TieCases,
    scenarios: BTreeMap<String, Scenario>,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct TieCases {
    below: TieCase,
    above: TieCase,
}

#[derive(Deserialize)]
struct TieCase {
    treat: Vec<u8>,
    dist: Vec<f64>,
    matchit: usize,
}

#[derive(Deserialize)]
struct Scenario {
    treat: Vec<u8>,
    s: Vec<String>,
    fam: Vec<String>,
    age: Vec<f64>,
    dist: Option<Vec<f64>>,
    maha: Option<Vec<Vec<f64>>>,
    kind: String,
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "the fields are the arguments of the R call"
)]
#[derive(Deserialize)]
struct Spec {
    order: Option<String>,
    ratio: usize,
    replace: bool,
    estimand: String,
    cal: Option<f64>,
    std: bool,
    covcal: Option<f64>,
    exact: bool,
    anti: bool,
}

#[derive(Deserialize)]
struct Run {
    scenario: String,
    spec: Spec,
    matchit: Vec<Rows>,
    model: Vec<Rows>,
    ties: usize,
    agrees: bool,
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
    load_reference("matchit_nearest.json")
}

fn birth() -> NaiveDate {
    NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date")
}

/// Anchors and candidates in row order, with the rows of the anchors.
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
        record
            .strata
            .insert("fam".to_string(), scenario.fam[row].clone());
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
        .as_ref()
        .expect("scores")
        .iter()
        .enumerate()
        .map(|(row, value)| ((row + 1).to_string(), *value))
        .collect()
}

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
        .map(|k| {
            if k / dimension == k % dimension {
                1.0
            } else {
                0.0
            }
        })
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

fn run_ours(
    scenario: &Scenario,
    spec: &Spec,
    order: Option<MatchOrder>,
    seed: u64,
) -> (Vec<usize>, MatchOutcome) {
    let focal = u8::from(spec.estimand != "ATC");
    let (focal_rows, anchors, candidates) = split(scenario, focal);
    let caliper = spec
        .cal
        .map(|width| DistanceCaliper::new(width).expect("positive"));
    let ranking = match scenario.kind.as_str() {
        "vec" | "mahvars" => DistanceConfig::propensity_score_map(scores(scenario), caliper),
        _ => mahalanobis(scenario),
    };
    let mut job = MatchJob::new_standard(&anchors, &candidates, seed)
        .with_nearest()
        .with_ratio(MatchRatio::new(spec.ratio).expect("ratio"))
        .with_replacement(spec.replace)
        .with_estimand(if focal == 0 {
            Estimand::Atc
        } else {
            Estimand::Att
        })
        .with_exact_matches(if spec.exact { vec!["s"] } else { Vec::new() })
        .with_std_caliper(spec.std)
        .with_distance_config(ranking);
    if let Some(order) = order {
        job = job.with_m_order(order);
    }
    if scenario.kind == "mahvars" {
        job = job.with_mahvars(mahalanobis(scenario));
    }
    let calipers: Vec<_> = spec
        .covcal
        .map(|width| caliper_on_field::<BaseRecord>("age", width))
        .into_iter()
        .collect();
    let job = job.with_constraints(calipers);
    let outcome = if spec.anti {
        job.with_antiexact("fam").run().expect("a valid job")
    } else {
        job.run().expect("a valid job")
    };
    (focal_rows, outcome)
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

fn expected(rows: &[Rows]) -> Vec<Vec<usize>> {
    rows.iter().map(Rows::to_vec).collect()
}

#[test]
fn reference_was_made_by_the_documented_versions() {
    let reference = reference();
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["matchit"], "4.8.2");
}

#[test]
fn pairs_match_the_loop_always_and_matchit_without_ties() {
    let reference = reference();
    let mut with_ties = 0;
    let mut differ = 0;
    let mut identical_without_ties = 0;
    for (number, run) in reference.runs.iter().enumerate() {
        let scenario = &reference.scenarios[&run.scenario];
        let order = run.spec.order.as_deref().map(order_of);
        let (focal_rows, outcome) = run_ours(scenario, &run.spec, order, 0);
        let ours = rows_of(&outcome, &focal_rows);
        let label = format!(
            "run {number} {} order {:?} ratio {} replace {} {}",
            run.scenario, run.spec.order, run.spec.ratio, run.spec.replace, run.spec.estimand
        );
        assert_eq!(ours, expected(&run.model), "{label}: the loop");
        if run.ties == 0 {
            assert!(run.agrees, "{label}: the script checks this");
            assert_eq!(ours, expected(&run.matchit), "{label}: matchit");
            identical_without_ties += 1;
        } else {
            with_ties += 1;
            if !run.agrees {
                differ += 1;
            }
        }
    }
    println!(
        "{} runs: {identical_without_ties} without ties agree with matchit pair for pair; \
         {with_ties} with ties, of which {differ} differ from matchit",
        reference.runs.len()
    );
    assert!(identical_without_ties > 200);
}

#[test]
fn unique_variants_of_lalonde_agree_with_matchit_in_every_run() {
    let reference = reference();
    let mut checked = 0;
    for run in reference
        .runs
        .iter()
        .filter(|run| run.scenario.ends_with("_unique"))
    {
        let scenario = &reference.scenarios[&run.scenario];
        let order = run.spec.order.as_deref().map(order_of);
        let (focal_rows, outcome) = run_ours(scenario, &run.spec, order, 0);
        assert_eq!(
            rows_of(&outcome, &focal_rows),
            expected(&run.matchit),
            "{} {:?} ratio {}",
            run.scenario,
            run.spec.order,
            run.spec.ratio
        );
        checked += 1;
    }
    assert!(checked >= 100, "{checked}");
}

/// `matchit()` takes the highest row among tied controls below the treated
/// unit and the lowest above it. The crate takes the lowest row in both.
#[test]
fn ties_are_broken_by_row_order_where_matchit_is_not() {
    let reference = reference();
    for (case, ours_expected, matchit_expected) in [
        (&reference.tie_cases.below, 1_usize, 3_usize),
        (&reference.tie_cases.above, 2, 2),
    ] {
        assert_eq!(case.matchit, matchit_expected);
        let anchors: Vec<BaseRecord> = case
            .treat
            .iter()
            .enumerate()
            .filter(|(_, treat)| **treat == 1)
            .map(|(row, _)| BaseRecord::new((row + 1).to_string(), birth()))
            .collect();
        let candidates: Vec<BaseRecord> = case
            .treat
            .iter()
            .enumerate()
            .filter(|(_, treat)| **treat == 0)
            .map(|(row, _)| BaseRecord::new((row + 1).to_string(), birth()))
            .collect();
        let scores: RapidHashMap<String, f64> = case
            .dist
            .iter()
            .enumerate()
            .map(|(row, value)| ((row + 1).to_string(), *value))
            .collect();
        let outcome = MatchJob::new_standard(&anchors, &candidates, 0)
            .with_nearest()
            .with_replacement(true)
            .with_distance_config(DistanceConfig::propensity_score_map(scores, None))
            .run()
            .expect("a valid job");
        assert_eq!(outcome.pairs.len(), 1);
        assert_eq!(outcome.pairs[0].comparator_id(), ours_expected.to_string());
    }
}

#[test]
fn random_order_keeps_the_rules_and_is_reproducible() {
    let reference = reference();
    let mut checked = 0;
    for name in ["lalonde_prob_unique", "sim_continuous", "sim_mahalanobis"] {
        let scenario = &reference.scenarios[name];
        for (cal, covcal, exact, anti, ratio, replace) in [
            (None, None, false, false, 1, false),
            (Some(0.25), None, false, false, 2, false),
            (Some(0.25), Some(4.0), true, true, 2, false),
            (None, None, true, true, 3, false),
            (Some(0.25), None, false, true, 1, true),
        ] {
            if scenario.kind == "maha" && cal.is_some() {
                continue;
            }
            let spec = Spec {
                order: Some("random".to_string()),
                ratio,
                replace,
                estimand: "ATT".to_string(),
                cal,
                std: true,
                covcal,
                exact,
                anti,
            };
            let first = run_ours(scenario, &spec, Some(MatchOrder::Random), 7);
            let again = run_ours(scenario, &spec, Some(MatchOrder::Random), 7);
            assert_eq!(first.1.pairs, again.1.pairs, "same seed, same pairs");
            let other = run_ours(scenario, &spec, Some(MatchOrder::Random), 8);
            assert_ne!(first.1.pairs, other.1.pairs, "another seed, another order");

            let (focal_rows, outcome) = first;
            let (_, anchors, candidates) = split(scenario, 1);
            let width = cal.map(|c| {
                let scores = scenario.dist.as_ref().expect("scores");
                let n = scores.len() as f64;
                let mean = scores.iter().sum::<f64>() / n;
                c * (scores.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt()
            });
            let mut used = HashSet::new();
            let mut matched_rows = HashSet::new();
            for pair in &outcome.pairs {
                let a: usize = pair.anchor_id().parse().expect("row");
                let c: usize = pair.comparator_id().parse().expect("row");
                matched_rows.insert(a);
                if !replace {
                    assert!(used.insert(c), "{name}: control {c} used twice");
                }
                if let Some(width) = width {
                    let scores = scenario.dist.as_ref().expect("scores");
                    assert!(
                        (scores[a - 1] - scores[c - 1]).abs() <= width,
                        "{name}: caliper"
                    );
                }
                if let Some(w) = covcal {
                    assert!((scenario.age[a - 1] - scenario.age[c - 1]).abs() <= w);
                }
                if exact {
                    assert_eq!(scenario.s[a - 1], scenario.s[c - 1]);
                }
                if anti {
                    assert_ne!(scenario.fam[a - 1], scenario.fam[c - 1]);
                }
            }
            // An anchor left without a control had none left that it could take.
            for (anchor, row) in anchors.iter().zip(&focal_rows) {
                if matched_rows.contains(row) {
                    continue;
                }
                for candidate in &candidates {
                    let c: usize = candidate.id.parse().expect("row");
                    if used.contains(&c) && !replace {
                        continue;
                    }
                    let allowed = width.is_none_or(|w| {
                        let scores = scenario.dist.as_ref().expect("scores");
                        (scores[row - 1] - scores[c - 1]).abs() <= w
                    }) && covcal.is_none_or(|w| {
                        (anchor.numerics["age"] - candidate.numerics["age"]).abs() <= w
                    }) && (!exact || scenario.s[row - 1] == scenario.s[c - 1])
                        && (!anti || scenario.fam[row - 1] != scenario.fam[c - 1]);
                    assert!(!allowed, "{name}: anchor {row} could have taken {c}");
                }
            }
            checked += 1;
        }
    }
    assert!(checked >= 10);
}

#[derive(Deserialize)]
struct PairsReference {
    scenarios: BTreeMap<String, PairsScenario>,
    runs: Vec<PairsRun>,
}

#[derive(Deserialize)]
struct PairsScenario {
    treat: Vec<u8>,
    strata: BTreeMap<String, Vec<String>>,
    numerics: BTreeMap<String, Vec<f64>>,
}

#[derive(Deserialize)]
struct PairsRun {
    scenario: String,
    estimand: String,
    exact: Vec<String>,
    caliper: serde_json::Value,
    replace: bool,
    ratio: usize,
    matchit: PairsBlock,
}

#[derive(Deserialize)]
struct PairsBlock {
    rows: Vec<Rows>,
}

/// The existing strategies, asked for rounds, give `matchit()`'s pairs where
/// every anchor gets a first control before any gets a second.
#[test]
fn rounds_for_the_existing_strategies_follow_matchit() {
    let reference: PairsReference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/match_pairs.json"
    ));
    let mut compared = 0;
    for run in reference
        .runs
        .iter()
        .filter(|run| !run.replace && run.ratio > 1)
    {
        let scenario = &reference.scenarios[&run.scenario];
        let focal_value = u8::from(run.estimand == "ATT");
        let mut focal_rows = Vec::new();
        let mut anchors = Vec::new();
        let mut candidates = Vec::new();
        for (row, treat) in scenario.treat.iter().enumerate() {
            let mut record = BaseRecord::new((row + 1).to_string(), birth());
            for (name, values) in &scenario.strata {
                record.strata.insert(name.clone(), values[row].clone());
            }
            for (name, values) in &scenario.numerics {
                record = record.with_numeric(name.clone(), values[row]);
            }
            if *treat == focal_value {
                focal_rows.push(row + 1);
                anchors.push(record);
            } else {
                candidates.push(record);
            }
        }
        let calipers: Vec<_> = run
            .caliper
            .as_object()
            .map(|map| {
                map.iter()
                    .map(|(name, window)| {
                        caliper_on_field::<BaseRecord>(name, window.as_f64().expect("number"))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let outcome = MatchJob::new_standard(&anchors, &candidates, 0)
            .with_strategy(DeterministicSelection)
            .with_round_assignment()
            .with_ratio(MatchRatio::new(run.ratio).expect("ratio"))
            .with_birth_window(BirthDateWindowDays::new(36_500).expect("window"))
            .with_exact_matches(run.exact.clone())
            .with_constraints(calipers)
            .run()
            .expect("a valid job");
        assert_eq!(
            rows_of(&outcome, &focal_rows),
            expected(&run.matchit.rows),
            "{} {} ratio {}",
            run.scenario,
            run.estimand,
            run.ratio
        );
        compared += 1;
    }
    assert_eq!(compared, 44);
}

#[test]
fn sequential_assignment_is_still_available_in_the_nearest_mode() {
    let reference = reference();
    let scenario = &reference.scenarios["sim_continuous"];
    let spec = Spec {
        order: Some("largest".to_string()),
        ratio: 3,
        replace: false,
        estimand: "ATT".to_string(),
        cal: None,
        std: true,
        covcal: None,
        exact: false,
        anti: false,
    };
    let (_, anchors, candidates) = split(scenario, 1);
    let sequential = MatchJob::new_standard(&anchors, &candidates, 0)
        .with_nearest()
        .with_ratio(MatchRatio::new(3).expect("ratio"))
        .with_assignment(Assignment::Sequential)
        .with_distance_config(DistanceConfig::propensity_score_map(scores(scenario), None))
        .run()
        .expect("a valid job");
    let rounds = run_ours(scenario, &spec, Some(MatchOrder::Largest), 0).1;
    assert_eq!(sequential.pairs.len(), rounds.pairs.len());
    assert_ne!(sequential.pairs, rounds.pairs);
}
