//! Cardinality and profile matching checked against MatchIt 4.8.2's
//! `matchit(method = "cardinality")` with the HiGHS solver.
//!
//! The reference is `tests/data/matchit_cardinality.json`, written by
//! `tests/data/matchit_cardinality.R`. It holds lalonde and a simulated cohort of
//! 1000 units, and for each case the options, the weights MatchIt returns and
//! HiGHS's status and objective in every exact stratum. The checks are:
//!
//! 1. Where HiGHS reports the optimum in every stratum and our search closes
//!    within its node limit (4000 relaxations per stratum), the number of units
//!    selected equals MatchIt's, exactly. Where the search stops at the limit
//!    its subset is no larger than HiGHS's, and the shortfall is printed, as is
//!    the number of cases that closed. The environment variable `CARD_NODES`
//!    raises the limit. Where it stopped at its time limit,
//!    ours is at least MatchIt's, since MatchIt's solution is feasible.
//! 2. A case that HiGHS did not close is reported and not asserted. Every balance constraint, as the documentation of `method_cardinality`
//!    states it, holds for our selection, in every exact stratum. The same
//!    function accepts MatchIt's selection, which shows it is the constraint
//!    MatchIt solved. HiGHS holds its rows to `1e-6` on sums, so MatchIt's
//!    selection is accepted to `1e-6` in units of a mean difference, and ours to
//!    `1e-8`, which is the solver's row tolerance `1e-9 ‖a‖₂` divided by the size of
//!    the smaller selected group at these sizes.
//! 3. The weight rule of the documentation, applied to MatchIt's own selected
//!    units, gives MatchIt's weights, and applied to ours gives ours. Where the
//!    selected set is the same as MatchIt's the weights are the same. How often
//!    the sets are the same is printed: the optimum is not unique when several
//!    units are interchangeable.
//! 4. The pairing after selection, on MatchIt's own selected units, has the
//!    total Mahalanobis distance of `optmatch::pairmatch`'s pairs, measured in
//!    MatchIt's transformed coordinates.
//! 5. The weights do not depend on the number of threads.
//! 6. Tolerances that cannot be met are `Infeasible`, and a node limit is
//!    reported as not proven optimal.

use epistat_matching::{
    CardinalityError, CardinalityLimits, CardinalityRequest, Estimand, match_cardinality,
    pair_units,
};
use epistat_testkit::{assert_same_on_pools, load_json};
use serde::{Deserialize, Deserializer};
use std::collections::{BTreeMap, HashMap};

fn one_or_many<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Either<T> {
        Many(Vec<T>),
        One(T),
    }
    Ok(match Either::deserialize(deserializer)? {
        Either::Many(v) => v,
        Either::One(x) => vec![x],
    })
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    datasets: HashMap<String, Dataset>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    matchit: String,
    highs: String,
    optmatch: String,
}

#[derive(Deserialize)]
struct Dataset {
    treat: Vec<u8>,
    #[serde(deserialize_with = "columns")]
    x: Vec<Vec<f64>>,
}

fn columns<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<Vec<f64>>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Cell {
        Column(Vec<f64>),
        Scalar(f64),
    }
    let cells: Vec<Cell> = Vec::deserialize(deserializer)?;
    Ok(cells
        .into_iter()
        .map(|c| match c {
            Cell::Column(v) => v,
            Cell::Scalar(x) => vec![x],
        })
        .collect())
}

#[derive(Deserialize)]
struct Case {
    name: String,
    dataset: String,
    estimand: String,
    ratio: Option<f64>,
    #[serde(deserialize_with = "one_or_many")]
    tols: Vec<f64>,
    #[serde(deserialize_with = "one_or_many")]
    std_tols: Vec<bool>,
    s_weights: Option<Vec<f64>>,
    exact: Option<Vec<u64>>,
    weights: Vec<f64>,
    n_selected: usize,
    #[serde(deserialize_with = "one_or_many")]
    status: Vec<i64>,
    optimal: bool,
    pair_raw: Option<Vec<Vec<f64>>>,
    pair_z: Option<Vec<Vec<f64>>>,
    pair_total: Option<f64>,
}

fn estimand(case: &Case) -> Estimand {
    match case.estimand.as_str() {
        "ATT" => Estimand::Att,
        "ATC" => Estimand::Atc,
        "ATE" => Estimand::Ate,
        other => panic!("estimand {other}"),
    }
}

fn load() -> Reference {
    load_json("tests/data/matchit_cardinality.json")
}

fn request<'a>(case: &'a Case, data: &'a Dataset, treated: &'a [bool]) -> CardinalityRequest<'a> {
    let mut request = CardinalityRequest::new(treated, &data.x);
    request.estimand = estimand(case);
    request.ratio = case.ratio;
    request.tols.clone_from(&case.tols);
    request.std_tols.clone_from(&case.std_tols);
    request.exact = case.exact.as_deref();
    request.sampling_weights = case.s_weights.as_deref();
    // A node limit, not a time limit, so the result does not depend on the machine.
    // `CARD_NODES` raises it to see how many cases the search closes.
    let nodes = std::env::var("CARD_NODES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4_000);
    request.limits = CardinalityLimits {
        time: None,
        nodes: Some(nodes),
    };
    request
}

/// The variance MatchIt gives a column: weights normalised to sum to 1, `m(1-m)`
/// for a column of zeros and ones, else `Σ w (x-m)² / (1 - Σ w²)`.
fn wvar(values: &[f64], weights: &[f64], binary: bool) -> f64 {
    let total: f64 = weights.iter().sum();
    let w: Vec<f64> = weights.iter().map(|v| v / total).collect();
    let mean: f64 = w.iter().zip(values).map(|(w, x)| w * x).sum();
    if binary {
        return mean * (1.0 - mean);
    }
    let ss: f64 = w
        .iter()
        .zip(values)
        .map(|(w, x)| w * (x - mean).powi(2))
        .sum();
    ss / (1.0 - w.iter().map(|w| w * w).sum::<f64>())
}

fn is_binary(values: &[f64]) -> bool {
    values.iter().all(|&v| v == 0.0 || v == 1.0)
}

/// The columns in the units the tolerances are in.
fn standardised(case: &Case, data: &Dataset, treated: &[bool]) -> Vec<Vec<f64>> {
    let n = treated.len();
    let sw: Vec<f64> = case.s_weights.clone().unwrap_or_else(|| vec![1.0; n]);
    let est = estimand(case);
    let focal = est != Estimand::Atc;
    data.x
        .iter()
        .enumerate()
        .map(|(c, column)| {
            if !case.std_tols[c] {
                return column.clone();
            }
            let sd = if est == Estimand::Ate {
                let binary = is_binary(column);
                let mut total = 0.0;
                for g in [false, true] {
                    let idx: Vec<usize> = (0..n).filter(|&i| treated[i] == g).collect();
                    let v: Vec<f64> = idx.iter().map(|&i| column[i]).collect();
                    let w: Vec<f64> = idx.iter().map(|&i| sw[i]).collect();
                    total += wvar(&v, &w, binary);
                }
                (total / 2.0).sqrt()
            } else {
                let idx: Vec<usize> = (0..n).filter(|&i| treated[i] == focal).collect();
                let v: Vec<f64> = idx.iter().map(|&i| column[i]).collect();
                let w: Vec<f64> = idx.iter().map(|&i| sw[i]).collect();
                wvar(&v, &w, is_binary(&v)).sqrt()
            };
            if sd >= 1e-10 {
                column.iter().map(|v| v / sd).collect()
            } else {
                column.clone()
            }
        })
        .collect()
}

/// The largest amount by which the selection breaks a balance constraint, in
/// units of a (standardised) mean difference, in every exact stratum.
fn worst_violation(case: &Case, data: &Dataset, treated: &[bool], selected: &[bool]) -> f64 {
    let n = treated.len();
    let x = standardised(case, data, treated);
    let est = estimand(case);
    let focal = est != Estimand::Atc;
    let mut strata: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        strata
            .entry(case.exact.as_ref().map_or(0, |e| e[i]))
            .or_default()
            .push(i);
    }
    let mut worst = 0.0_f64;
    for units in strata.values() {
        if !(units.iter().any(|&i| treated[i]) && units.iter().any(|&i| !treated[i])) {
            continue;
        }
        // Sampling weights scaled within the stratum so each group's sum is its size.
        let mut sw: Vec<f64> = vec![1.0; n];
        if let Some(raw) = &case.s_weights {
            for g in [false, true] {
                let members: Vec<usize> =
                    units.iter().copied().filter(|&i| treated[i] == g).collect();
                let total: f64 = members.iter().map(|&i| raw[i]).sum();
                for &i in &members {
                    sw[i] = members.len() as f64 * raw[i] / total;
                }
            }
        }
        let wmean = |members: &[usize], c: usize| -> f64 {
            let num: f64 = members.iter().map(|&i| sw[i] * x[c][i]).sum();
            let den: f64 = members.iter().map(|&i| sw[i]).sum();
            num / den
        };
        let pick = |g: bool, only_selected: bool| -> Vec<usize> {
            units
                .iter()
                .copied()
                .filter(|&i| treated[i] == g && (!only_selected || selected[i]))
                .collect()
        };
        let p = x.len();
        match (est, case.ratio) {
            (Estimand::Ate, ratio) => {
                for c in 0..p {
                    let target = wmean(units, c);
                    for g in [false, true] {
                        let sel = pick(g, true);
                        let dev = (wmean(&sel, c) - target).abs() - case.tols[c] / 2.0;
                        worst = worst.max(dev);
                    }
                }
                if let Some(r) = ratio {
                    let (nf, no) = (
                        pick(focal, true).len() as f64,
                        pick(!focal, true).len() as f64,
                    );
                    worst = worst.max((no - r * nf).abs());
                }
            }
            (_, Some(r)) => {
                let (f, o) = (pick(focal, true), pick(!focal, true));
                worst = worst.max((o.len() as f64 - r * f.len() as f64).abs());
                for c in 0..p {
                    let mean =
                        |v: &[usize]| v.iter().map(|&i| x[c][i]).sum::<f64>() / v.len() as f64;
                    worst = worst.max((mean(&f) - mean(&o)).abs() - case.tols[c]);
                }
            }
            (_, None) => {
                let all_focal = pick(focal, false);
                let o = pick(!focal, true);
                for c in 0..p {
                    let dev = (wmean(&o, c) - wmean(&all_focal, c)).abs() - case.tols[c];
                    worst = worst.max(dev);
                }
            }
        }
    }
    worst
}

/// The weights the documentation describes, from a selection: in each stratum
/// the larger group's weights are scaled to the smaller group's total (the focal
/// group's for profile matching for the ATT), then each group's weights are
/// scaled to sum to the number of its units that have a weight.
fn expected_weights(case: &Case, treated: &[bool], selected: &[bool]) -> Vec<f64> {
    let n = treated.len();
    let est = estimand(case);
    let profile_att = est != Estimand::Ate && case.ratio.is_none();
    let focal = est != Estimand::Atc;
    let mut w: Vec<f64> = selected.iter().map(|&b| f64::from(u8::from(b))).collect();
    let mut strata: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    for i in 0..n {
        strata
            .entry(case.exact.as_ref().map_or(0, |e| e[i]))
            .or_default()
            .push(i);
    }
    for units in strata.values() {
        let size = |g: bool| units.iter().filter(|&&i| treated[i] == g).count();
        if size(true) == 0 || size(false) == 0 {
            continue;
        }
        let reference = if profile_att {
            focal
        } else {
            size(true) < size(false)
        };
        let total = |g: bool| {
            units
                .iter()
                .filter(|&&i| treated[i] == g)
                .map(|&i| w[i])
                .sum::<f64>()
        };
        let (r, o) = (total(reference), total(!reference));
        if o > 0.0 {
            for &i in units.iter().filter(|&&i| treated[i] != reference) {
                w[i] *= r / o;
            }
        }
    }
    for g in [false, true] {
        let members: Vec<usize> = (0..n).filter(|&i| treated[i] == g).collect();
        let kept = members.iter().filter(|&&i| w[i] > 0.0).count() as f64;
        let total: f64 = members.iter().map(|&i| w[i]).sum();
        if total > 0.0 {
            for &i in &members {
                w[i] *= kept / total;
            }
        }
    }
    w
}

fn selected_from(weights: &[f64]) -> Vec<bool> {
    weights.iter().map(|&w| w > 0.0).collect()
}

#[test]
fn selection_sizes_and_balance_agree_with_matchit() {
    let reference = load();
    eprintln!(
        "{} MatchIt {} highs {} optmatch {}",
        reference.versions.r,
        reference.versions.matchit,
        reference.versions.highs,
        reference.versions.optmatch
    );
    let mut identical = 0;
    let mut compared = 0;
    let mut worst_ours = 0.0_f64;
    let mut worst_theirs = 0.0_f64;
    let mut optimal_cases = 0;
    let mut proven_cases = 0;
    let mut not_found = 0;
    let mut deficit = 0;
    for case in &reference.cases {
        if std::env::var("CARD_CASE").is_ok_and(|c| c != case.name) {
            continue;
        }
        let data = &reference.datasets[&case.dataset];
        let treated: Vec<bool> = data.treat.iter().map(|&t| t == 1).collect();
        let their_selected = selected_from(&case.weights);
        // The weight rule reproduces MatchIt's weights from its own selection.
        for (i, (a, b)) in expected_weights(case, &treated, &their_selected)
            .iter()
            .zip(&case.weights)
            .enumerate()
        {
            assert!(
                (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                "{}: MatchIt weight {i} is {b}, the rule gives {a}",
                case.name
            );
        }
        let theirs = worst_violation(case, data, &treated, &their_selected);
        worst_theirs = worst_theirs.max(theirs);
        assert!(
            theirs < 1e-6,
            "{}: MatchIt's own selection breaks the checker by {theirs}",
            case.name
        );
        let t0 = std::time::Instant::now();
        let outcome = match match_cardinality(&request(case, data, &treated)) {
            Ok(outcome) => outcome,
            Err(CardinalityError::NoSolutionInTime { .. }) => {
                eprintln!("{}: no subset found within the node limit", case.name);
                not_found += 1;
                continue;
            }
            Err(e) => panic!("{}: {e}", case.name),
        };
        eprintln!("{} {:?}", case.name, t0.elapsed());
        let ours = selected_from(&outcome.weights);
        let violation = worst_violation(case, data, &treated, &outcome.selected);
        worst_ours = worst_ours.max(violation);
        assert!(
            violation < 1e-8,
            "{}: balance broken by {violation}",
            case.name
        );
        for (i, (a, b)) in expected_weights(case, &treated, &outcome.selected)
            .iter()
            .zip(&outcome.weights)
            .enumerate()
        {
            assert!(
                (a - b).abs() <= 1e-9 * b.abs().max(1.0),
                "{}: our weight {i} is {b}, the rule gives {a}",
                case.name
            );
        }
        let n_ours = ours.iter().filter(|&&b| b).count();
        let proven = outcome.strata.iter().all(|s| s.proven_optimal);
        if case.optimal {
            // A search that closes proves its subset the largest, so it equals
            // HiGHS's. One that stops at its node limit can only be smaller.
            if proven {
                assert_eq!(n_ours, case.n_selected, "{}: size", case.name);
                proven_cases += 1;
            } else {
                assert!(
                    n_ours <= case.n_selected,
                    "{}: {n_ours} selected, more than HiGHS's optimum {}",
                    case.name,
                    case.n_selected
                );
                deficit += case.n_selected - n_ours;
            }
            optimal_cases += 1;
        } else {
            // HiGHS stopped at its time limit, so its subset is only a lower
            // bound on the optimum. Nothing is asserted.
            eprintln!(
                "{}: HiGHS did not close (status {:?}), it selected {}, we {n_ours}",
                case.name, case.status, case.n_selected
            );
        }
        compared += 1;
        if ours == their_selected {
            identical += 1;
            for (i, (a, b)) in outcome.weights.iter().zip(&case.weights).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                    "{}: weight {i}: {a} vs {b}",
                    case.name
                );
            }
        }
    }
    eprintln!(
        "{compared} cases, {optimal_cases} where HiGHS proved its optimum, {proven_cases} of them \
         proved by us, the others {deficit} units short in all and {not_found} with none found; same selected set in {identical}; worst balance excess ours {worst_ours:.2e}, MatchIt's {worst_theirs:.2e}"
    );
    assert!(proven_cases >= 12, "only {proven_cases} cases closed");
}

#[test]
fn pairing_matches_pairmatch_on_matchits_selection() {
    let reference = load();
    let mut checked = 0;
    for case in reference.cases.iter().filter(|c| c.pair_raw.is_some()) {
        let data = &reference.datasets[&case.dataset];
        let treated: Vec<bool> = data.treat.iter().map(|&t| t == 1).collect();
        let selected = selected_from(&case.weights);
        let n = treated.len();
        let mut by_key: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            by_key
                .entry(case.exact.as_ref().map_or(0, |e| e[i]))
                .or_default()
                .push(i);
        }
        let strata: Vec<&[usize]> = by_key.values().map(Vec::as_slice).collect();
        let ratio = case.ratio.unwrap() as i64;
        let pairs = pair_units(
            &treated,
            &selected,
            case.pair_raw.as_ref().unwrap(),
            ratio,
            &strata,
            None,
            case.s_weights.as_deref(),
        )
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        // Total distance in MatchIt's coordinates.
        let z = case.pair_z.as_ref().unwrap();
        let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, p) in pairs.iter().enumerate() {
            if let Some(p) = p {
                members.entry(*p).or_default().push(i);
            }
        }
        let mut total = 0.0;
        for units in members.values() {
            let tr: Vec<usize> = units.iter().copied().filter(|&i| treated[i]).collect();
            assert_eq!(tr.len(), 1, "{}: one treated unit per pair", case.name);
            for &j in units.iter().filter(|&&i| !treated[i]) {
                total += z
                    .iter()
                    .map(|col| (col[tr[0]] - col[j]).powi(2))
                    .sum::<f64>()
                    .sqrt();
            }
        }
        let theirs = case.pair_total.unwrap();
        // Both are optima of the same assignment problem. Ours is on costs
        // rounded to 2^-30 of the largest distance, so the totals agree to that
        // times the number of pairs.
        let n_pairs = selected
            .iter()
            .zip(&treated)
            .filter(|(s, t)| **s && **t)
            .count();
        let slack = 2.0_f64.powi(-30) * 20.0 * n_pairs as f64 * 100.0 + 1e-9 * theirs;
        assert!(
            (total - theirs).abs() <= slack,
            "{}: total distance {total} vs {theirs}",
            case.name
        );
        checked += 1;
    }
    assert!(checked >= 3);
}

#[test]
fn weights_do_not_depend_on_threads() {
    let reference = load();
    for name in ["lalonde_ate_profile", "sim_att_exact", "lalonde_att_1"] {
        let case = reference.cases.iter().find(|c| c.name == name).unwrap();
        let data = &reference.datasets[&case.dataset];
        let treated: Vec<bool> = data.treat.iter().map(|&t| t == 1).collect();
        assert_same_on_pools(&[1, 2, 8], || {
            match_cardinality(&request(case, data, &treated))
                .unwrap()
                .weights
        });
    }
}

#[test]
fn impossible_tolerances_are_infeasible() {
    // Two treated units and two controls, one covariate that cannot be balanced.
    let treated = [true, true, false, false];
    let covariates = vec![vec![0.0, 0.0, 10.0, 10.0]];
    let mut r = CardinalityRequest::new(&treated, &covariates);
    r.tols = vec![0.1];
    r.std_tols = vec![false];
    // Profile matching for the ATE needs a unit from each group, and the
    // treated mean (0) is 5 from the overall mean.
    r.estimand = Estimand::Ate;
    r.ratio = None;
    assert_eq!(
        match_cardinality(&r).unwrap_err(),
        CardinalityError::Infeasible { key: 0 }
    );
}

#[test]
fn a_node_limit_is_reported_as_not_proven() {
    let reference = load();
    let case = reference
        .cases
        .iter()
        .find(|c| c.name == "sim_att_1")
        .unwrap();
    let data = &reference.datasets[&case.dataset];
    let treated: Vec<bool> = data.treat.iter().map(|&t| t == 1).collect();
    let mut r = request(case, data, &treated);
    r.limits = CardinalityLimits {
        time: None,
        nodes: Some(1),
    };
    match match_cardinality(&r) {
        Ok(outcome) => {
            // The root relaxation was integral, or the limit left a gap.
            let report = &outcome.strata[0];
            assert!(
                report.proven_optimal
                    || report.bound >= (report.selected_treated + report.selected_control) as f64
            );
        }
        Err(CardinalityError::NoSolutionInTime { .. }) => {}
        Err(e) => panic!("{e}"),
    }
}
