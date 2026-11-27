//! Exact matching into strata checked against MatchIt 4.8.2's
//! `matchit(method = "exact")`.
//!
//! The reference is `tests/data/matchit_exact.json`, written by
//! `tests/data/matchit_exact.R`. It holds `lalonde` stratified on two and on
//! three variables, the three-variable case again in random order, and small
//! simulated sets over 24 cells in which some cells are empty and many hold
//! one group only, plus a set in which no stratum holds both groups, which
//! MatchIt refuses.
//!
//! For ATT, ATC and ATE the test compares which units are kept, the subclass
//! of each, and the weights. MatchIt rescales the weights of each group to sum
//! to the number of its kept units. The script also writes the unscaled
//! subclass weights, computed from the subclass counts and checked in R
//! against MatchIt's after the same rescaling, and the test compares ours with
//! both, applying the rescaling to ours.
//!
//! MatchIt numbers the subclasses in the order of first appearance within the
//! group of the first row of its data. The function numbers them by the order
//! among the anchors, so for data that start with a control the test checks
//! that MatchIt's numbers follow the controls, and compares ours after
//! renumbering MatchIt's by the anchors.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::Error;
use epistat_matching::prelude::*;

use r_reference::{Versions, reference as load_reference, unit_id as id};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct Dataset {
    name: String,
    treat: Vec<f64>,
    vars: BTreeMap<String, Vec<String>>,
    runs: Vec<RRun>,
}

#[derive(Deserialize)]
struct RRun {
    estimand: String,
    error: bool,
    subclass: Option<Vec<Option<usize>>>,
    weights: Option<Vec<f64>>,
    unscaled_weights: Option<Vec<f64>>,
}

fn reference() -> Reference {
    load_reference("matchit_exact.json")
}

fn ours(data: &Dataset) -> Result<ExactMatchOutcome, Error> {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (idx, label) in data.treat.iter().enumerate() {
        let mut record = BaseRecord::new(id(idx), birth);
        for (name, values) in &data.vars {
            record.strata.insert(name.clone(), values[idx].clone());
        }
        if *label > 0.5 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    let names: Vec<&String> = data.vars.keys().collect();
    exact_match_subclasses(&anchors, &candidates, &names)
}

fn estimand_of(name: &str) -> Estimand {
    match name {
        "ATT" => Estimand::Att,
        "ATC" => Estimand::Atc,
        "ATE" => Estimand::Ate,
        other => panic!("estimand {other}"),
    }
}

/// MatchIt's numbers, renumbered by the order of first appearance among the
/// units of `group`.
fn renumbered(treat: &[f64], subclass: &[Option<usize>], group: f64) -> Vec<Option<usize>> {
    let mut order: Vec<usize> = Vec::new();
    for (idx, label) in treat.iter().enumerate() {
        if (*label - group).abs() < 0.5
            && let Some(s) = subclass[idx]
            && !order.contains(&s)
        {
            order.push(s);
        }
    }
    subclass
        .iter()
        .map(|s| s.map(|s| order.iter().position(|o| *o == s).expect("seen") + 1))
        .collect()
}

/// Rescale the weights of each group to sum to the number of its kept units.
fn rescaled(treat: &[f64], weights: &[Option<f64>]) -> Vec<f64> {
    let group = |idx: usize| usize::from(treat[idx] < 0.5);
    let mut sum = [0.0_f64; 2];
    let mut count = [0.0_f64; 2];
    for (idx, w) in weights.iter().enumerate() {
        if let Some(w) = w {
            sum[group(idx)] += w;
            count[group(idx)] += 1.0;
        }
    }
    weights
        .iter()
        .enumerate()
        .map(|(idx, w)| w.map_or(0.0, |w| w * count[group(idx)] / sum[group(idx)]))
        .collect()
}

/// Relative tolerance for weights. Our weight is `p / (1 - p)` for the
/// anchor share `p = n1 / n` of a subclass, and the script's is `n1 / n0`.
/// `1 - p` is at least `1 / n`, so its relative error is at most `n` roundings
/// of 1.1e-16, below 7e-14 for the 614 units of lalonde. The rescaling adds a
/// sum of at most 614 terms, the same again. 1e-12 leaves a margin of about
/// seven.
const WEIGHT_TOLERANCE: f64 = 1e-12;

fn close(got: f64, expected: f64) -> bool {
    (got - expected).abs() <= WEIGHT_TOLERANCE * expected.abs().max(1.0)
}

/// Kept units, subclasses and weights equal MatchIt's on every run, and
/// MatchIt's refusal is ours.
#[test]
fn subclasses_and_weights_match_matchit() {
    let reference = reference();
    println!(
        "{} / MatchIt {}",
        reference.versions["r"], reference.versions["matchit"]
    );
    let mut dropped_somewhere = false;
    let mut compared = 0;
    for data in &reference.datasets {
        let got = ours(data);
        if data.runs.iter().all(|run| run.error) {
            assert_eq!(got.unwrap_err(), Error::NoSharedStratum, "{}", data.name);
            compared += 1;
            continue;
        }
        let got = got.expect("MatchIt accepts the data");
        for run in &data.runs {
            let label = format!("{} {}", data.name, run.estimand);
            assert!(!run.error, "{label}");
            let r_subclass = run.subclass.as_ref().expect("subclass");
            let r_weights = run.weights.as_ref().expect("weights");
            let r_unscaled = run.unscaled_weights.as_ref().expect("unscaled");
            let n = data.treat.len();

            let by_anchors = renumbered(&data.treat, r_subclass, 1.0);
            let by_first_row = renumbered(&data.treat, r_subclass, data.treat[0]);
            // MatchIt's numbers follow the group of the first row.
            let one_based: Vec<Option<usize>> = r_subclass.clone();
            assert_eq!(by_first_row, one_based, "{label}: MatchIt numbering");

            let weights: HashMap<String, f64> = {
                let (ids, values) = got
                    .weights_sorted(estimand_of(&run.estimand))
                    .expect("weights");
                ids.into_iter().map(str::to_owned).zip(values).collect()
            };
            let assigned: HashMap<String, usize> = got
                .assignments()
                .map(|(id, subclass)| (id.to_owned(), subclass))
                .collect();
            let dropped: Vec<String> = got.dropped_ids().into_iter().map(str::to_owned).collect();
            let unscaled: Vec<Option<f64>> =
                (0..n).map(|idx| weights.get(&id(idx)).copied()).collect();
            let scaled = rescaled(&data.treat, &unscaled);
            for idx in 0..n {
                let ours_subclass = assigned.get(&id(idx)).map(|s| s + 1);
                assert_eq!(ours_subclass, by_anchors[idx], "{label}: subclass of {idx}");
                assert_eq!(
                    ours_subclass.is_some(),
                    r_subclass[idx].is_some(),
                    "{label}: kept {idx}"
                );
                assert_eq!(
                    dropped.contains(&id(idx)),
                    r_subclass[idx].is_none(),
                    "{label}: dropped {idx}"
                );
                assert!(
                    close(scaled[idx], r_weights[idx]),
                    "{label}: rescaled weight of {idx}: {} against {}",
                    scaled[idx],
                    r_weights[idx]
                );
                assert!(
                    close(unscaled[idx].unwrap_or(0.0), r_unscaled[idx]),
                    "{label}: unscaled weight of {idx}"
                );
            }
            dropped_somewhere |= !dropped.is_empty();
            compared += 1;
        }
    }
    assert!(dropped_somewhere, "no run drops a unit, test is vacuous");
    assert!(compared >= 16, "{compared} runs compared");
}

/// The number of subclasses and their group counts equal MatchIt's table, with
/// MatchIt's numbers renumbered by the anchors.
#[test]
fn subclass_counts_match_matchit() {
    let reference = reference();
    for data in reference.datasets.iter().filter(|d| !d.runs[0].error) {
        let got = ours(data).expect("accepted");
        let r_subclass = data.runs[0].subclass.as_ref().expect("subclass");
        let by_anchors = renumbered(&data.treat, r_subclass, 1.0);
        let mut table: BTreeMap<usize, [usize; 2]> = BTreeMap::new();
        for (idx, s) in by_anchors.iter().enumerate() {
            if let Some(s) = s {
                table.entry(*s - 1).or_default()[usize::from(data.treat[idx] < 0.5)] += 1;
            }
        }
        assert_eq!(got.subclasses.len(), table.len(), "{}", data.name);
        for summary in &got.subclasses {
            assert_eq!(
                [summary.anchor_count, summary.candidate_count],
                table[&summary.subclass],
                "{} subclass {}",
                data.name,
                summary.subclass
            );
        }
    }
}
