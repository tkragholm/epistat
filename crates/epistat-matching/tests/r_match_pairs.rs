//! Pairs, weights and output shapes checked against R 4.6's `MatchIt` 4.8.2.
//!
//! The reference is `tests/data/match_pairs.json`, written by
//! `tests/data/match_pairs.R`. Its runs cover both estimands, with and without
//! exact strata, calipers on named covariates on the raw scale, with and
//! without replacement and ratios 1 to 3, on a simulated data set whose exact
//! strata include one without controls and several with too few, and on a
//! sample of `lalonde`.
//!
//! `matchit(method = "nearest")` chooses the nearest control by distance and
//! the crate chooses the lowest index, so the script gives MatchIt a distance
//! that is 0 for every focal unit and rises with the row number for the other
//! group. MatchIt's nearest control is then the first available one in row
//! order, which is what `DeterministicSelection` picks. The focal group is the
//! treated group for `ATT` and the control group for `ATC`, as in MatchIt.
//!
//! What the tests check:
//!
//! * The pairs, in order of assignment per anchor, are the same as MatchIt's
//!   whenever each anchor gets one control or controls are reused.
//! * With ratio above 1 and no replacement the pairs differ by design.
//!   MatchIt gives every anchor its first control before any gets a second.
//!   The crate lets each anchor take all its controls before the next anchor
//!   starts. The reference holds the pairs of that second order, made by a
//!   short loop in the script, and the crate must reproduce them.
//! * The unit weights of `CaseUnitControlFractional` are MatchIt's weights
//!   with `normalize = FALSE`, and `UnitWeightSet::normalized` gives the
//!   default `normalize = TRUE` weights.
//! * `ControlUnitCaseFractional` is MatchIt's rule for the opposite
//!   estimand applied to the same pairs with the roles of the groups swapped.
//! * `PairCount` is not a weighting `matchit()` offers. It is compared with
//!   the stratum weights of `estimand = "ATE"`, which nearest neighbour
//!   matching itself refuses, and agrees with them once both are normalized
//!   when every anchor has the same number of controls.
//! * `match.matrix`, `get_matches()` and `match_data()` are the crate's
//!   `match_matrix`, `get_matches` and `match_data` of its pairs, and are
//!   compared with MatchIt's: the rows and columns of the matrix, the units,
//!   subclasses and weights of the long table, and the weights and subclass of
//!   the matched data, with and without rescaling.
//!
//! Weights in the file are rounded to 12 decimals, so they are compared to an
//! absolute 1e-12 (half a unit in the last place is 5e-13, and the crate's
//! weights are sums of a few reciprocals of small integers, exact to about
//! 1e-16).

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::DeterministicSelection;
use epistat_matching::MatchWeightMethod;
use epistat_matching::constraints::caliper_on_field;
use epistat_matching::output::{MatchDataOptions, Units, get_matches, match_data, match_matrix};
use epistat_matching::prelude::*;
use epistat_matching::{Estimand, UnitRole, UnitWeightSet};
use epistat_testkit::Tolerance;
use r_reference::{Versions, reference as load_reference};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    scenarios: BTreeMap<String, Scenario>,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Scenario {
    treat: Vec<u8>,
    strata: BTreeMap<String, Vec<String>>,
    numerics: BTreeMap<String, Vec<f64>>,
}

#[derive(Deserialize)]
struct Run {
    scenario: String,
    estimand: String,
    exact: Vec<String>,
    caliper: serde_json::Value,
    replace: bool,
    ratio: usize,
    matchit: Block,
    sequential: Option<Block>,
    get_matches: Option<LongTable>,
}

/// One set of pairs and everything R computed from it.
#[derive(Deserialize)]
struct Block {
    /// Units with a non-zero weight, from 1.
    idx: Vec<usize>,
    raw: Vec<f64>,
    norm: Vec<f64>,
    other_raw: Vec<f64>,
    /// One entry per focal unit in row order: the units matched to it in
    /// order of assignment.
    rows: Vec<Rows>,
    subclass_idx: Option<Vec<usize>>,
    subclass: Option<Vec<usize>>,
    ate_raw: Option<Vec<f64>>,
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

#[derive(Deserialize)]
struct LongTable {
    id: Vec<usize>,
    subclass: Vec<usize>,
    weights: Vec<f64>,
}

const WEIGHT_TOLERANCE: Tolerance = Tolerance::exact().abs(1e-12);

fn reference() -> Reference {
    load_reference("match_pairs.json")
}

fn records(scenario: &Scenario, focal_value: u8) -> (Vec<usize>, Vec<BaseRecord>, Vec<BaseRecord>) {
    let birth_date = NaiveDate::from_ymd_opt(2000, 1, 1).expect("valid date");
    let mut focal_idx = Vec::new();
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (row, treat) in scenario.treat.iter().enumerate() {
        let mut record = BaseRecord::new((row + 1).to_string(), birth_date);
        for (name, values) in &scenario.strata {
            record.strata.insert(name.clone(), values[row].clone());
        }
        for (name, values) in &scenario.numerics {
            record = record.with_numeric(name.clone(), values[row]);
        }
        if *treat == focal_value {
            focal_idx.push(row + 1);
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    (focal_idx, anchors, candidates)
}

fn run_ours(run: &Run, anchors: &[BaseRecord], candidates: &[BaseRecord]) -> MatchOutcome {
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
    MatchJob::new_standard(anchors, candidates, 0)
        .with_strategy(DeterministicSelection)
        .with_ratio(MatchRatio::new(run.ratio).expect("non-zero ratio"))
        .with_birth_window(BirthDateWindowDays::new(36_500).expect("non-negative window"))
        .with_exact_matches(run.exact.clone())
        .with_replacement(run.replace)
        .with_constraints(calipers)
        .run()
        .expect("a valid job")
}

/// The units of a run in row order, with the focal group as anchors.
fn units_of(scenario: &Scenario, focal_value: u8) -> Units {
    Units::new(
        (1..=scenario.treat.len())
            .map(|row| row.to_string())
            .collect(),
        scenario.treat.iter().map(|t| *t == focal_value).collect(),
    )
    .expect("row numbers are unique")
}

/// The controls of each anchor in order of assignment, by anchor row, read
/// from the crate's match matrix.
fn rows_of(outcome: &MatchOutcome, units: &Units, ratio: usize) -> Vec<Vec<usize>> {
    let matrix = match_matrix(outcome, units, ratio).expect("pairs are among the units");
    assert_eq!(matrix.ncol(), ratio.max(matrix.ncol()));
    (0..matrix.nrow())
        .map(|row| {
            matrix
                .controls_of_row(row)
                .into_iter()
                .map(|id| id.parse().expect("row number"))
                .collect()
        })
        .collect()
}

fn weight_of(set: &UnitWeightSet, treat: &[u8], focal_value: u8, unit: usize) -> f64 {
    let id = unit.to_string();
    if treat[unit - 1] == focal_value {
        set.case_weight(&id)
    } else {
        set.control_weight(&id)
    }
    .unwrap_or(0.0)
}

fn expected_of(idx: &[usize], values: &[f64]) -> HashMap<usize, f64> {
    assert_eq!(idx.len(), values.len());
    idx.iter().copied().zip(values.iter().copied()).collect()
}

#[track_caller]
fn assert_weights(
    label: &str,
    ours: &UnitWeightSet,
    scenario: &Scenario,
    focal_value: u8,
    expected: &HashMap<usize, f64>,
) {
    for unit in 1..=scenario.treat.len() {
        let got = weight_of(ours, &scenario.treat, focal_value, unit);
        let want = expected.get(&unit).copied().unwrap_or(0.0);
        assert!(
            WEIGHT_TOLERANCE.accepts(got, want),
            "{label}: unit {unit}: got {got}, MatchIt {want}"
        );
    }
}

/// Divide the positive weights of each group by their mean, in unit order.
fn normalized_by_group(expected: &HashMap<usize, f64>, treat: &[u8]) -> HashMap<usize, f64> {
    let mut out = expected.clone();
    for group in [0_u8, 1] {
        let mut units: Vec<usize> = expected
            .iter()
            .filter(|(unit, weight)| treat[**unit - 1] == group && **weight > 0.0)
            .map(|(unit, _)| *unit)
            .collect();
        units.sort_unstable();
        if units.is_empty() {
            continue;
        }
        let mut sum = 0.0;
        for unit in &units {
            sum += expected[unit];
        }
        #[expect(clippy::cast_precision_loss, reason = "a few hundred units")]
        let mean = sum / units.len() as f64;
        for unit in units {
            out.insert(unit, expected[&unit] / mean);
        }
    }
    out
}

/// The crate's `get_matches` as (unit, subclass, weight), in the order of the
/// table: by subclass, and in each subclass the treated unit first. The
/// units of one group keep the order the crate lists them in, which is the
/// order MatchIt lists them in.
fn long_rows(outcome: &MatchOutcome, units: &Units, treat: &[u8]) -> Vec<(usize, usize, f64)> {
    let long = get_matches(outcome, units, false).expect("pairs are among the units");
    let mut out: Vec<(usize, usize, f64)> = (0..long.len())
        .map(|row| {
            (
                long.id[row].parse().expect("row number"),
                long.pair_id[row].expect("matched rows only"),
                long.weight[row],
            )
        })
        .collect();
    out.sort_by_key(|(unit, subclass, _)| (*subclass, std::cmp::Reverse(treat[*unit - 1])));
    out
}

#[derive(Default)]
struct Tally {
    runs: usize,
    pairs_equal_to_matchit: usize,
    round_order_runs: usize,
    round_order_differs: usize,
    anchors_lost_by_sequence: usize,
    ate_differs: usize,
    ate_equal: usize,
    shortfall_anchors: usize,
    unmatched_cases: usize,
}

#[test]
fn pairs_weights_and_shapes_agree_with_matchit() {
    let reference = reference();
    assert_eq!(reference.versions["matchit"], "4.8.2");
    assert!(reference.versions["r"].starts_with("R version 4."));
    assert!(reference.versions["jsonlite"].starts_with(char::is_numeric));

    let mut tally = Tally::default();
    for run in &reference.runs {
        let scenario = &reference.scenarios[&run.scenario];
        let focal_value = u8::from(run.estimand == "ATT");
        let (focal_idx, anchors, candidates) = records(scenario, focal_value);
        let label = format!(
            "{} {} exact={:?} caliper={} replace={} ratio={}",
            run.scenario, run.estimand, run.exact, run.caliper, run.replace, run.ratio
        );
        let outcome = run_ours(run, &anchors, &candidates);
        let units = units_of(scenario, focal_value);
        let rows = rows_of(&outcome, &units, run.ratio);
        tally.runs += 1;

        // The order of assignment, and which block of the reference the crate
        // must reproduce.
        let matchit_rows: Vec<Vec<usize>> = run.matchit.rows.iter().map(Rows::to_vec).collect();
        assert_eq!(matchit_rows.len(), focal_idx.len(), "{label}");
        let expected = if let Some(sequential) = &run.sequential {
            tally.round_order_runs += 1;
            let sequential_rows: Vec<Vec<usize>> =
                sequential.rows.iter().map(Rows::to_vec).collect();
            if sequential_rows != matchit_rows {
                tally.round_order_differs += 1;
                let ours_matched = rows.iter().filter(|row| !row.is_empty()).count();
                let theirs_matched = matchit_rows.iter().filter(|row| !row.is_empty()).count();
                tally.anchors_lost_by_sequence += theirs_matched.saturating_sub(ours_matched);
            }
            sequential
        } else {
            tally.pairs_equal_to_matchit += 1;
            assert_eq!(rows, matchit_rows, "{label}: pairs differ from MatchIt");
            &run.matchit
        };
        assert_eq!(
            rows,
            expected.rows.iter().map(Rows::to_vec).collect::<Vec<_>>(),
            "{label}: pairs"
        );

        // Counts. MatchIt has no unique-control count, so used_controls is
        // compared with the number of distinct controls in the pairs.
        let matched = rows.iter().filter(|row| !row.is_empty()).count();
        assert_eq!(outcome.matched_cases, matched, "{label}");
        assert_eq!(
            outcome.unmatched_cases,
            focal_idx.len() - matched,
            "{label}"
        );
        tally.unmatched_cases += outcome.unmatched_cases;
        tally.shortfall_anchors += rows
            .iter()
            .filter(|row| !row.is_empty() && row.len() < run.ratio)
            .count();
        let distinct: HashSet<usize> = rows.iter().flatten().copied().collect();
        assert_eq!(
            outcome.used_controls,
            distinct.len(),
            "{label}: used_controls"
        );

        // Weights of the matched side by estimand.
        let weights_anchor_side =
            outcome.match_weights(MatchWeightMethod::for_estimand(Estimand::Att));
        let raw = expected_of(&expected.idx, &expected.raw);
        assert_weights(
            &format!("{label} raw"),
            &weights_anchor_side,
            scenario,
            focal_value,
            &raw,
        );
        let norm = expected_of(&expected.idx, &expected.norm);
        assert_weights(
            &format!("{label} normalized"),
            &weights_anchor_side.normalized(),
            scenario,
            focal_value,
            &norm,
        );
        let weights_candidate_side =
            outcome.match_weights(MatchWeightMethod::for_estimand(Estimand::Atc));
        let other = expected_of(&expected.idx, &expected.other_raw);
        assert_weights(
            &format!("{label} other"),
            &weights_candidate_side,
            scenario,
            focal_value,
            &other,
        );

        // PairCount is the number of pairs a unit is in.
        let weights_count = outcome.match_weights(MatchWeightMethod::for_estimand(Estimand::Ate));
        let mut counts: HashMap<usize, f64> = HashMap::new();
        for (anchor, controls) in focal_idx.iter().zip(&rows) {
            for control in controls {
                *counts.entry(*anchor).or_insert(0.0) += 1.0;
                *counts.entry(*control).or_insert(0.0) += 1.0;
            }
        }
        assert_weights(
            &format!("{label} count"),
            &weights_count,
            scenario,
            focal_value,
            &counts,
        );
        if let (Some(ate_raw), false) = (&expected.ate_raw, run.replace) {
            let ate = normalized_by_group(&expected_of(&expected.idx, ate_raw), &scenario.treat);
            let ours = weights_count.normalized();
            let same = (1..=scenario.treat.len()).all(|unit| {
                WEIGHT_TOLERANCE.accepts(
                    weight_of(&ours, &scenario.treat, focal_value, unit),
                    ate.get(&unit).copied().unwrap_or(0.0),
                )
            });
            let equal_sizes = {
                let sizes: HashSet<usize> = rows
                    .iter()
                    .filter(|row| !row.is_empty())
                    .map(Vec::len)
                    .collect();
                sizes.len() <= 1
            };
            if equal_sizes {
                assert!(same, "{label}: equal pair sizes, ATE weights differ");
            }
            if same {
                tally.ate_equal += 1;
            } else {
                tally.ate_differs += 1;
            }
        }

        // match.matrix: one row per anchor, NA for an anchor with no control.
        assert_eq!(
            expected.rows.len(),
            focal_idx.len(),
            "{label}: one row per anchor"
        );
        let matrix = match_matrix(&outcome, &units, run.ratio).expect("matrix");
        assert_eq!(matrix.ncol(), run.ratio, "{label}: columns");
        assert_eq!(
            matrix
                .case_id
                .iter()
                .map(|id| id.parse::<usize>().expect("row"))
                .collect::<Vec<_>>(),
            focal_idx,
            "{label}: anchors in input order"
        );

        // match_data(): the matched units with their weights and subclass.
        for normalize in [false, true] {
            let data = match_data(
                &outcome,
                &units,
                MatchDataOptions {
                    normalize,
                    replacement: run.replace,
                    ..MatchDataOptions::default()
                },
            )
            .expect("match data");
            let wanted = if normalize {
                &expected.norm
            } else {
                &expected.raw
            };
            let reference: HashMap<usize, f64> = expected
                .idx
                .iter()
                .copied()
                .zip(wanted.iter().copied())
                .filter(|(_, weight)| *weight > 0.0)
                .collect();
            let units_in_order: Vec<usize> =
                data.id.iter().map(|id| id.parse().expect("row")).collect();
            assert!(
                units_in_order.windows(2).all(|pair| pair[0] < pair[1]),
                "{label}: match_data keeps the order of the data"
            );
            assert_eq!(
                units_in_order.iter().copied().collect::<HashSet<_>>(),
                reference.keys().copied().collect::<HashSet<_>>(),
                "{label}: units of match_data"
            );
            for (row, unit) in units_in_order.iter().enumerate() {
                assert!(
                    WEIGHT_TOLERANCE.accepts(data.weight[row], reference[unit]),
                    "{label}: match_data weight of unit {unit}"
                );
                let expected_role = if scenario.treat[*unit - 1] == focal_value {
                    UnitRole::Case
                } else {
                    UnitRole::Control
                };
                assert_eq!(data.role[row], expected_role, "{label}: role");
            }
            if run.replace {
                assert!(
                    data.subclass.iter().all(Option::is_none),
                    "{label}: no subclass with replacement"
                );
            }
            if let (Some(subclass_idx), Some(subclass)) =
                (&expected.subclass_idx, &expected.subclass)
            {
                let reference_subclass: HashMap<usize, usize> = subclass_idx
                    .iter()
                    .copied()
                    .zip(subclass.iter().copied())
                    .collect();
                for (row, unit) in units_in_order.iter().enumerate() {
                    // MatchIt gives no subclass with replacement.
                    let want = if run.replace {
                        None
                    } else {
                        reference_subclass.get(unit).copied()
                    };
                    assert_eq!(data.subclass[row], want, "{label}: subclass of unit {unit}");
                }
            }
        }

        // get_matches(): rebuilt from the pairs when the pairs are MatchIt's.
        if let (Some(long), None) = (&run.get_matches, &run.sequential) {
            let rebuilt = long_rows(&outcome, &units, &scenario.treat);
            let mut theirs: Vec<(usize, usize, f64)> = (0..long.id.len())
                .map(|row| (long.id[row], long.subclass[row], long.weights[row]))
                .collect();
            theirs.sort_by_key(|(unit, subclass, _)| {
                (*subclass, std::cmp::Reverse(scenario.treat[*unit - 1]))
            });
            assert_eq!(rebuilt.len(), theirs.len(), "{label}: get_matches length");
            for (row, ((unit, subclass, weight), (id, sub, w))) in
                rebuilt.iter().zip(&theirs).enumerate()
            {
                assert_eq!(
                    (unit, subclass),
                    (id, sub),
                    "{label}: get_matches, row {row}"
                );
                assert!(
                    WEIGHT_TOLERANCE.accepts(*weight, *w),
                    "{label}: get_matches weight, row {row}: {weight} against {w}"
                );
            }
            // And back: the controls of each anchor from the long table, as a
            // set, since the table lists a reused control where it first occurs.
            let mut back: BTreeMap<usize, (usize, Vec<usize>)> = BTreeMap::new();
            for row in 0..long.id.len() {
                let entry = back.entry(long.subclass[row]).or_default();
                if scenario.treat[long.id[row] - 1] == focal_value {
                    entry.0 = long.id[row];
                } else {
                    entry.1.push(long.id[row]);
                }
            }
            let from_long: HashMap<usize, Vec<usize>> = back.into_values().collect();
            for (anchor, controls) in focal_idx.iter().zip(&rows) {
                let mut from_table = from_long.get(anchor).cloned().unwrap_or_default();
                from_table.sort_unstable();
                let mut ours = controls.clone();
                ours.sort_unstable();
                assert_eq!(from_table, ours, "{label}: match.matrix from get_matches");
            }
        }
    }

    println!(
        "runs {}, pairs equal to MatchIt {}, round-order runs {} (pairs differ in {}, anchors \
         matched by MatchIt and not by the crate {}), ATE equal {} differs {}, anchors short of \
         ratio {}, unmatched anchors {}",
        tally.runs,
        tally.pairs_equal_to_matchit,
        tally.round_order_runs,
        tally.round_order_differs,
        tally.anchors_lost_by_sequence,
        tally.ate_equal,
        tally.ate_differs,
        tally.shortfall_anchors,
        tally.unmatched_cases
    );
    // The reference must contain the cases the comparison is about.
    assert!(tally.pairs_equal_to_matchit > 0);
    assert!(
        tally.round_order_differs > 0,
        "order of assignment never differs"
    );
    assert!(
        tally.shortfall_anchors > 0,
        "no anchor got fewer than the ratio"
    );
    assert!(tally.unmatched_cases > 0, "no unmatched anchor");
    assert!(
        tally.ate_differs > 0,
        "ATE never differs from the pair count"
    );
}
