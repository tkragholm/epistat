//! Propensity-score subclassification checked against MatchIt 4.8.2's
//! `matchit(method = "subclass")`.
//!
//! The reference is `tests/data/matchit_subclass.json`, written by
//! `tests/data/matchit_subclass.R`. Every run hands MatchIt the scores as a
//! numeric `distance`, which it uses as given, and the Rust function the same
//! scores through its score map, so the runs also cover the numeric-vector
//! distance. The data are `lalonde` with its fitted propensity score, a
//! simulated data set with 24 distinct scores, and 30 small random data sets
//! with rounded scores. Together they give ties at the cutpoints, tied
//! cutpoints, subclasses with too few units of one group, and requests that
//! MatchIt refuses. Runs cover ATT, ATC and ATE, 2 to 10 subclasses and
//! `min.n` of 0, 1, 2 and 4.
//!
//! The comparison covers the quantile cutpoints, the subclass of every unit and
//! the weights. MatchIt rescales the weights of each group to sum to the number
//! of units of that group that keep a weight. The Rust weights are the
//! unscaled subclass weights, so the test applies the same rescaling before it
//! compares them.

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_testkit::load_json;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    matchit: String,
}

#[derive(Deserialize)]
struct Dataset {
    name: String,
    treat: Vec<f64>,
    distance: Vec<f64>,
    runs: Vec<RRun>,
}

#[derive(Deserialize)]
struct RRun {
    estimand: String,
    subclass: usize,
    min_n: usize,
    error: bool,
    q_cut: Option<Vec<f64>>,
    assignment: Option<Vec<Option<usize>>>,
    weights: Option<Vec<f64>>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/matchit_subclass.json"
    ))
}

fn id(idx: usize) -> String {
    format!("u{idx}")
}

fn estimand_of(name: &str) -> Estimand {
    match name {
        "ATT" => Estimand::Att,
        "ATC" => Estimand::Atc,
        "ATE" => Estimand::Ate,
        other => panic!("estimand {other}"),
    }
}

fn ours(data: &Dataset, run: &RRun) -> SubclassificationOutcome {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    let mut scores = HashMap::new();
    for (idx, label) in data.treat.iter().enumerate() {
        let record = BaseRecord::new(id(idx), birth);
        scores.insert(id(idx), data.distance[idx]);
        if *label > 0.5 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    let config = SubclassificationConfig {
        subclasses: run.subclass,
        reference_group: None,
        min_n: run.min_n,
    };
    subclassify_by_propensity_score_map(
        &anchors,
        &candidates,
        &scores,
        estimand_of(&run.estimand),
        &config,
    )
    .expect("valid subclassification settings")
}

/// Rescale the weights of each group to sum to the number of its units that
/// are retained, as MatchIt does after computing subclass weights.
fn rescaled(data: &Dataset, outcome: &SubclassificationOutcome) -> Vec<f64> {
    let mut sum = [0.0_f64; 2];
    let mut count = [0.0_f64; 2];
    let group = |idx: usize| usize::from(data.treat[idx] < 0.5);
    for idx in 0..data.treat.len() {
        if let Some(w) = outcome.weights.get(&id(idx)) {
            sum[group(idx)] += w;
            count[group(idx)] += 1.0;
        }
    }
    (0..data.treat.len())
        .map(|idx| {
            outcome
                .weights
                .get(&id(idx))
                .map_or(0.0, |w| w * count[group(idx)] / sum[group(idx)])
        })
        .collect()
}

/// Relative tolerance for the weights. A weight is a ratio of counts, rescaled
/// by a ratio of two sums of at most a few hundred such ratios, so it carries
/// well under 1e3 roundings of 1.1e-16 each. 1e-12 relative bounds that.
const WEIGHT_TOLERANCE: f64 = 1e-12;

/// Cutpoints, subclass membership and weights equal MatchIt's on every run
/// that MatchIt accepts.
#[test]
fn subclasses_and_weights_match_matchit() {
    let reference = reference();
    println!(
        "{} / MatchIt {}",
        reference.versions.r, reference.versions.matchit
    );
    let mut compared = 0;
    let mut scooted = 0;
    for data in &reference.datasets {
        for run in data.runs.iter().filter(|r| !r.error) {
            let label = format!(
                "{} {} k={} min.n={}",
                data.name, run.estimand, run.subclass, run.min_n
            );
            let got = ours(data, run);
            let q_cut = run.q_cut.as_ref().expect("q.cut");
            assert_eq!(got.quantile_cuts.len(), q_cut.len(), "{label}");
            for (position, (a, b)) in got.quantile_cuts.iter().zip(q_cut).enumerate() {
                // The same two multiplications and an addition, in the same
                // order, so the cutpoints agree to the bit.
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "q.cut {position} {label}: {a} against {b}"
                );
            }

            let expected = run.assignment.as_ref().expect("assignment");
            let kept: BTreeSet<usize> = got
                .assignments
                .keys()
                .map(|key| key[1..].parse().expect("unit id"))
                .collect();
            let expected_kept: BTreeSet<usize> = expected
                .iter()
                .enumerate()
                .filter_map(|(idx, s)| s.map(|_| idx))
                .collect();
            assert_eq!(kept, expected_kept, "retained units, {label}");
            // MatchIt numbers the subclasses it keeps from 1; ours are the
            // same subclasses in the same order, numbered from 0 with a gap
            // where a dropped subclass was.
            let present: BTreeSet<usize> = got.assignments.values().copied().collect();
            let rank: HashMap<usize, usize> = present
                .iter()
                .enumerate()
                .map(|(r, s)| (*s, r + 1))
                .collect();
            for idx in &kept {
                assert_eq!(
                    Some(rank[&got.assignments[&id(*idx)]]),
                    expected[*idx],
                    "subclass of unit {idx}, {label}"
                );
            }
            let min_group = |treated: bool| {
                got.subclasses
                    .iter()
                    .filter(|s| s.anchor_count > 0 && s.candidate_count > 0)
                    .map(|s| {
                        if treated {
                            s.anchor_count
                        } else {
                            s.candidate_count
                        }
                    })
                    .min()
            };
            if run.min_n > 0 {
                assert!(min_group(true) >= Some(run.min_n), "{label}");
                assert!(min_group(false) >= Some(run.min_n), "{label}");
            }

            let weights = rescaled(data, &got);
            for (idx, (a, b)) in weights
                .iter()
                .zip(run.weights.as_ref().expect("weights"))
                .enumerate()
            {
                assert!(
                    (a - b).abs() <= WEIGHT_TOLERANCE * b.abs().max(1.0),
                    "weight of unit {idx}, {label}: {a} against {b}"
                );
            }
            compared += 1;
            // A run counts as filled when the plain quantile intervals leave
            // some subclass without a unit of one group.
            if run.min_n > 0 && got.quantile_cuts.len() > 2 {
                let interior = &got.quantile_cuts[1..got.quantile_cuts.len() - 1];
                let mut seen = BTreeSet::new();
                for (label_value, score) in data.treat.iter().zip(&data.distance) {
                    seen.insert((
                        interior.partition_point(|cut| cut <= score),
                        *label_value > 0.5,
                    ));
                }
                let subclasses: BTreeSet<usize> = seen.iter().map(|(s, _)| *s).collect();
                if subclasses
                    .iter()
                    .any(|s| !seen.contains(&(*s, false)) || !seen.contains(&(*s, true)))
                {
                    scooted += 1;
                }
            }
        }
    }
    println!("{compared} runs compared, {scooted} with units moved to fill a subclass");
    assert!(compared >= 500);
    assert!(scooted >= 50, "the reference exercises min.n");
}

/// Where MatchIt stops because a group has fewer than `subclass * min.n` units,
/// every unit is dropped here, with no subclass or weight.
#[test]
fn infeasible_min_n_drops_every_unit() {
    let reference = reference();
    let mut refused = 0;
    for data in &reference.datasets {
        for run in data.runs.iter().filter(|r| r.error) {
            let got = ours(data, run);
            assert!(
                got.assignments.is_empty(),
                "{} {} {:?}",
                data.name,
                run.estimand,
                got.assignments
            );
            assert!(got.weights.is_empty(), "{:?}", got.weights);
            assert_eq!(got.dropped_ids.len(), data.treat.len());
            refused += 1;
        }
    }
    assert!(refused > 0);
}
