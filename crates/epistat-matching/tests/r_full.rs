//! Optimal full matching checked against R 4.6's `MatchIt` 4.8.2 and
//! `optmatch` 0.10.8.
//!
//! The reference is `tests/data/matchit_full.json`, written by
//! `tests/data/matchit_full.R`: runs of `matchit(method = "full")` on five
//! simulated data sets with the limits of `fullmatch()` (`min.controls`,
//! `max.controls`, `omit.fraction` of either sign, `mean.controls`), the
//! estimands ATT, ATC and ATE, exact strata and calipers. For ATC MatchIt
//! takes the controls as the focal group, so the crate is given the controls
//! as anchors and its anchor-group weights are compared.
//!
//! What is checked, on every run:
//!
//! * The total distance, the sum of the distances of the treated-control pairs
//!   within each subclass, with the bounds of `r_optimal.rs`: the crate's is
//!   at most `3 n d_max / 2^40` above the optimum and optmatch's at most
//!   `1.5 epsilon` per pair.
//! * The number of units in subclasses and of subclasses.
//! * Where the partition is unique (four perturbations of every distance by
//!   1e-5 standard deviations give the same partition), the subclasses as sets
//!   of units, and MatchIt's weights against the crate's weights rescaled by
//!   [`FullMatchOutcome::match_data`].
//! * Where MatchIt finds no match, the crate reports an error.
//!
//! Weights in the file carry 17 digits and the crate's are ratios of small
//! integers rescaled by a mean, so they are compared to a relative 1e-12.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_matching::{RapidHashMap, match_full};

use r_reference::reference as load_reference;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Deserialize)]
struct Reference {
    defect: Defect,
    versions: Versions,
    scenarios: BTreeMap<String, Scenario>,
    limits: BTreeMap<String, Value>,
    runs: Vec<Run>,
}

/// optmatch's negative `omit.fraction` against the transposed problem.
#[derive(Deserialize)]
struct Defect {
    /// Rows 1 to `n_treated` are the treated, the rest the controls.
    points: Vec<Vec<f64>>,
    n_treated: usize,
    total_negative: f64,
    total_transposed: f64,
    largest_group_negative: usize,
    largest_group_transposed: usize,
    r_epsilon: f64,
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

#[derive(Deserialize)]
struct Run {
    scenario: String,
    estimand: String,
    limits: String,
    exact: bool,
    caliper: Option<f64>,
    error: bool,
    subclass: Option<Vec<Option<usize>>>,
    weights: Option<Vec<f64>>,
    total: Option<f64>,
    r_epsilon: Option<f64>,
    unique: Option<bool>,
}

fn reference() -> Reference {
    load_reference("matchit_full.json")
}

fn euclidean(name: &str) -> bool {
    name.starts_with("euclid")
}

fn number(value: &Value) -> f64 {
    match value {
        Value::Array(items) => number(&items[0]),
        other => other.as_f64().expect("number"),
    }
}

/// R writes an empty list as an empty array.
fn config(limits: &Value) -> FullMatchConfig {
    let mut config = FullMatchConfig::default();
    for (key, value) in limits.as_object().into_iter().flatten() {
        let value = number(value);
        match key.as_str() {
            "min.controls" => config.min_controls = value,
            "max.controls" => config.max_controls = value,
            "omit.fraction" => config.retain = Retain::OmitFraction(value),
            "mean.controls" => config.retain = Retain::MeanControls(value),
            other => panic!("limit {other}"),
        }
    }
    config
}

fn distance(name: &str, scenario: &Scenario, a: usize, b: usize) -> f64 {
    let dx = scenario.x[a - 1] - scenario.x[b - 1];
    if euclidean(name) {
        let dy = scenario.x2[a - 1] - scenario.x2[b - 1];
        (dx * dx + dy * dy).sqrt()
    } else {
        dx.abs()
    }
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

/// The partition as the set of its subclasses, each a set of rows.
fn partition(subclass: &[Option<usize>]) -> BTreeSet<BTreeSet<usize>> {
    let mut groups: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for (index, subclass) in subclass.iter().enumerate() {
        if let Some(subclass) = subclass {
            groups.entry(*subclass).or_default().insert(index + 1);
        }
    }
    groups.into_values().collect()
}

#[test]
fn subclasses_totals_and_weights_match_matchit() {
    let reference = reference();
    println!(
        "{} / MatchIt {} / optmatch {} / solver {}",
        reference.versions.r,
        reference.versions.matchit,
        reference.versions.optmatch,
        reference.versions.solver
    );
    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    let (mut totals, mut partitions, mut errors) = (0, 0, 0);
    for run in &reference.runs {
        let scenario = &reference.scenarios[&run.scenario];
        let label = format!(
            "{} {} {} exact {} caliper {:?}",
            run.scenario, run.estimand, run.limits, run.exact, run.caliper
        );
        let focal = u8::from(run.estimand != "ATC");
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
        let width = run
            .caliper
            .map(|w| DistanceCaliper::new(w).expect("caliper"));
        let distance_config = if euclidean(&run.scenario) {
            DistanceConfig::euclidean_map(vectors, 2, width).expect("two columns")
        } else {
            DistanceConfig::propensity_score_map(scores, width)
        };
        let criteria = MatchingCriteria::builder()
            .required_strata(if run.exact {
                vec!["g".to_string()]
            } else {
                Vec::new()
            })
            .build();
        let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance_config);
        let got = match_full(
            &anchors,
            &candidates,
            &request,
            &config(&reference.limits[&run.limits]),
        );
        if run.error {
            assert!(got.is_err(), "{label}: R failed but the crate matched");
            errors += 1;
            continue;
        }
        let outcome = got.unwrap_or_else(|error| panic!("{label}: {error}"));
        let r_subclass = run.subclass.as_ref().expect("subclass");
        let n = r_subclass.len();

        // Our subclasses by row.
        let subclass_of: HashMap<&str, usize> = outcome.assignments().collect();
        let ours: Vec<Option<usize>> = (1..=n)
            .map(|row| subclass_of.get(row.to_string().as_str()).copied())
            .collect();
        assert_eq!(
            ours.iter().flatten().count(),
            r_subclass.iter().flatten().count(),
            "{label}: matched units"
        );
        let our_partition = partition(&ours);
        let r_partition = partition(r_subclass);
        assert_eq!(
            our_partition.len(),
            r_partition.len(),
            "{label}: subclasses"
        );

        // Total distance of the treated-control pairs within subclasses.
        let mut total = 0.0;
        let mut pairs = 0_usize;
        for group in &our_partition {
            for a in group {
                for b in group {
                    if scenario.treat[a - 1] == 1 && scenario.treat[b - 1] == 0 {
                        total += distance(&run.scenario, scenario, *a, *b);
                        pairs += 1;
                    }
                }
            }
        }
        let d_max = max_distance(&run.scenario, scenario);
        let our_bound = 3.0 * pairs as f64 * d_max / 2_f64.powi(40) + 1e-12;
        let r_total = run.total.expect("total");
        let r_bound = 1.5 * pairs as f64 * run.r_epsilon.expect("epsilon");
        assert!(
            total <= r_total + our_bound,
            "{label}: crate total {total} above R's {r_total}"
        );
        // optmatch's exchange of treated and controls limits group sizes by the
        // wrong dimensions (see `optmatch_negative_omit_fraction_is_not_the_transposed_problem`),
        // so its total can be above the optimum on those runs.
        let exchanged = reference.limits[&run.limits]
            .as_object()
            .is_some_and(|map| {
                map.get("omit.fraction").is_some_and(|v| number(v) < 0.0)
                    || map.get("max.controls").is_some_and(|v| number(v) <= 0.5)
            });
        if !exchanged {
            assert!(
                r_total <= total + r_bound + our_bound,
                "{label}: R total {r_total} above the crate's {total}"
            );
        }
        totals += 1;

        if run.unique == Some(true) && !exchanged {
            assert_eq!(our_partition, r_partition, "{label}: subclasses");
            partitions += 1;
            let estimand = match run.estimand.as_str() {
                "ATE" => Estimand::Ate,
                _ => Estimand::Att,
            };
            let units = Units::new(
                (1..=n).map(|row| row.to_string()).collect(),
                (1..=n)
                    .map(|row| scenario.treat[row - 1] == focal)
                    .collect(),
            )
            .expect("unique ids");
            let data = outcome
                .match_data_for(&units, estimand, true, false)
                .expect("weights");
            let expected = run.weights.as_ref().expect("weights");
            for (position, id) in data.id.iter().enumerate() {
                let want = expected[row_of(id) - 1];
                let got = data.weight[position];
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "{label}: weight of row {id} is {got}, R has {want}"
                );
                assert_eq!(
                    data.subclass[position].is_some(),
                    r_subclass[row_of(id) - 1].is_some(),
                    "{label}: subclass of row {id}"
                );
            }
        }
    }
    println!("{totals} totals, {partitions} unique partitions, {errors} refusals compared");
    assert!(totals >= 100, "only {totals} totals compared");
    assert!(partitions >= 50, "only {partitions} unique partitions");
    assert!(errors >= 5, "only {errors} runs where R failed");
}

fn row_of(id: &str) -> usize {
    id.parse().expect("row number")
}

#[test]
fn match_data_numbers_subclasses_from_one_and_leaves_out_unmatched_units() {
    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    let rows: Vec<(&str, f64)> = vec![
        ("a1", 0.0),
        ("a2", 5.0),
        ("c1", 0.1),
        ("c2", 5.2),
        ("c3", 5.3),
    ];
    let scores: RapidHashMap<String, f64> =
        rows.iter().map(|(id, x)| ((*id).to_string(), *x)).collect();
    let make = |id: &str| BaseRecord::new(id, birth);
    let anchors = vec![make("a1"), make("a2")];
    let candidates = vec![make("c1"), make("c2"), make("c3")];
    let distance = DistanceConfig::propensity_score_map(scores, None);
    let criteria = MatchingCriteria::builder().build();
    let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
    let outcome =
        match_full(&anchors, &candidates, &request, &FullMatchConfig::default()).expect("feasible");
    assert_eq!(outcome.retained_units(), 5);
    assert_eq!(outcome.subclasses.len(), 2);
    let units = Units::new(
        vec![
            "a1".into(),
            "a2".into(),
            "c1".into(),
            "c2".into(),
            "c3".into(),
        ],
        vec![true, true, false, false, false],
    )
    .expect("unique");
    let data = outcome
        .match_data_for(&units, Estimand::Att, false, true)
        .expect("weights");
    assert_eq!(data.len(), 5);
    assert_eq!(
        data.subclass,
        vec![Some(1), Some(2), Some(1), Some(2), Some(2)]
    );
    // The second subclass has one anchor and two candidates.
    // The weight of a candidate is p / (1 - p) of the anchor share p, which
    // rounds within a few ulps of the ratio of the counts.
    for (got, want) in data.weight.iter().zip([1.0, 1.0, 1.0, 0.5, 0.5]) {
        assert!((got - want).abs() < 1e-15, "{got} against {want}");
    }
}

/// With fewer treated than controls, `fullmatch(D, omit.fraction = -0.3)`
/// caps its largest group below what the transposed problem allows, which the
/// documentation gives as the same problem, and its total is higher. The crate
/// agrees with the transposed call.
#[test]
fn optmatch_negative_omit_fraction_is_not_the_transposed_problem() {
    let reference = reference();
    let defect = &reference.defect;
    assert!(defect.total_negative > defect.total_transposed + 1e-3);
    assert!(defect.largest_group_negative < defect.largest_group_transposed);

    let birth = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    let mut vectors: RapidHashMap<String, Vec<f64>> = RapidHashMap::default();
    for (index, point) in defect.points.iter().enumerate() {
        let id = format!("u{index}");
        vectors.insert(id.clone(), point.clone());
        let record = BaseRecord::new(id, birth);
        if index < defect.n_treated {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    let distance = DistanceConfig::euclidean_map(vectors.clone(), 2, None).expect("two columns");
    let criteria = MatchingCriteria::builder().build();
    let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
    let config = FullMatchConfig {
        retain: Retain::OmitFraction(-0.3),
        ..FullMatchConfig::default()
    };
    let outcome = match_full(&anchors, &candidates, &request, &config).expect("feasible");
    let mut groups: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    for (id, subclass) in outcome.assignments() {
        groups.entry(subclass).or_default().push(id);
    }
    let mut total = 0.0;
    for members in groups.values() {
        for a in members
            .iter()
            .filter(|id| id[1..].parse::<usize>().expect("index") < defect.n_treated)
        {
            for b in members
                .iter()
                .filter(|id| id[1..].parse::<usize>().expect("index") >= defect.n_treated)
            {
                let (pa, pb) = (&vectors[*a], &vectors[*b]);
                total += ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2)).sqrt();
            }
        }
    }
    let largest = groups.values().map(Vec::len).max().expect("groups");
    let pairs_bound = 1.5 * 50.0 * defect.r_epsilon + 1e-9;
    assert!(
        (total - defect.total_transposed).abs() <= pairs_bound,
        "crate {total} against the transposed call {}",
        defect.total_transposed
    );
    assert!(total < defect.total_negative - 1e-3);
    assert!(
        largest > defect.largest_group_negative + 1,
        "largest group {largest}"
    );
}
