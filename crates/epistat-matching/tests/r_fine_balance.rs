//! Fine, near-fine and refined balance checked against R's `rcbalance` 1.8.8
//! and `designmatch` 0.5.5.
//!
//! The reference is `tests/data/fine_balance.json`, written by
//! `tests/data/fine_balance.R` from nine runs of `rcbalance::rcbalance()` on
//! simulated data with a Euclidean distance of two covariates, whose optima
//! are unique (three perturbations of every distance by a relative 1e-4 give
//! the same pairs). The runs cover fine balance on one variable with one and
//! two candidates per anchor, near-fine balance where fine balance is
//! impossible, refined balance with two nested levels, the interaction of two
//! variables, a caliper, exact strata, strata with a caliper and refined
//! balance, and a caliper that leaves no match. `rcbalance` solves with
//! `rlemon`. Where fine balance is feasible, `designmatch::bmatch()` with
//! `fine` and the glpk solver gives a second total.
//!
//! What is checked, on every run:
//!
//! * Where R finds no match, the crate reports [`Error::Infeasible`].
//! * The balance table of every level, which is determined by the optimum.
//! * The total distance. The crate's costs are scaled to `2^b` with `b` at
//!   least 30 for these sizes, so it is within `3 n d_max / 2^30` of the
//!   optimum, and R is within `tol / 2` per pair above it.
//! * The pairs, since the optima are unique.
//! * The total of `designmatch` where it was run, within its own rounding of
//!   `tol` per pair.
//!
//! Exact mode is checked on the runs with no deviation, which it must
//! reproduce, and on those with deviation, which it must refuse.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::Error;
use epistat_matching::prelude::*;
use epistat_matching::{
    BalanceLevel, BalanceMode, FineBalance, FineBalanceOutcome, match_fine_balance,
};
use epistat_matching::{DistanceCaliper, DistanceConfig, MatchingCriteria, OptimalRequest};
use epistat_testkit::load_json;
use r_reference::Versions;
use rapidhash::RapidHashMap;
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Units {
    a: Vec<String>,
    b: Vec<String>,
    s: Vec<String>,
    x1: Vec<f64>,
    x2: Vec<f64>,
}

#[derive(Deserialize)]
struct Level {
    keys: Vec<String>,
    categories: Vec<String>,
    target: Vec<usize>,
    matched: Vec<usize>,
}

#[derive(Deserialize)]
struct DesignMatch {
    total: f64,
}

#[derive(Deserialize)]
struct Run {
    name: String,
    k: usize,
    caliper: Option<f64>,
    strata: bool,
    fb: Vec<Vec<String>>,
    t: Units,
    c: Units,
    error: bool,
    matches: Option<Vec<Vec<usize>>>,
    total: Option<f64>,
    levels: Option<Vec<Level>>,
    unique: Option<bool>,
    tol: Option<f64>,
    designmatch: Option<DesignMatch>,
}

fn record(id: String, units: &Units, i: usize) -> BaseRecord {
    let mut record = BaseRecord::new(id, NaiveDate::from_ymd_opt(2000, 1, 1).expect("date"));
    for (key, values) in [("a", &units.a), ("b", &units.b), ("s", &units.s)] {
        record.strata.insert(key.to_string(), values[i].clone());
    }
    record
}

fn run_ours(run: &Run, mode: BalanceMode) -> Result<FineBalanceOutcome, Error> {
    let anchors: Vec<BaseRecord> = (0..run.t.a.len())
        .map(|i| record(format!("t{}", i + 1), &run.t, i))
        .collect();
    let candidates: Vec<BaseRecord> = (0..run.c.a.len())
        .map(|i| record(format!("c{}", i + 1), &run.c, i))
        .collect();
    let mut vectors: RapidHashMap<String, Vec<f64>> = RapidHashMap::default();
    for i in 0..anchors.len() {
        vectors.insert(format!("t{}", i + 1), vec![run.t.x1[i], run.t.x2[i]]);
    }
    for i in 0..candidates.len() {
        vectors.insert(format!("c{}", i + 1), vec![run.c.x1[i], run.c.x2[i]]);
    }
    let width = run
        .caliper
        .map(|width| DistanceCaliper::new(width).expect("caliper"));
    let distance = DistanceConfig::euclidean_map(vectors, 2, width).expect("two columns");
    let criteria = MatchingCriteria::builder()
        .match_ratio(run.k)
        .required_strata(if run.strata {
            vec!["s".to_string()]
        } else {
            Vec::new()
        })
        .build();
    let request = OptimalRequest::new(&criteria, &()).with_distance_config(&distance);
    let balance = FineBalance {
        levels: run.fb.clone(),
        mode,
    };
    match_fine_balance(&anchors, &candidates, &request, &balance)
}

fn pairs_of(outcome: &MatchOutcome) -> BTreeSet<(usize, usize)> {
    outcome
        .pairs
        .iter()
        .map(|pair| {
            (
                pair.case_id[1..].parse().expect("row"),
                pair.control_id[1..].parse().expect("row"),
            )
        })
        .collect()
}

fn distance(run: &Run, t: usize, c: usize) -> f64 {
    let dx = run.t.x1[t - 1] - run.c.x1[c - 1];
    let dy = run.t.x2[t - 1] - run.c.x2[c - 1];
    (dx * dx + dy * dy).sqrt()
}

#[test]
fn balance_tables_totals_and_pairs_match_rcbalance() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/fine_balance.json"
    ));
    println!(
        "{} / rcbalance {} / designmatch {} / rlemon {}",
        reference.versions["r"],
        reference.versions["rcbalance"],
        reference.versions["designmatch"],
        reference.versions["rlemon"]
    );
    let (mut errors, mut compared, mut designmatch) = (0, 0, 0);
    for run in &reference.runs {
        let label = &run.name;
        let got = run_ours(run, BalanceMode::Near);
        if run.error {
            assert!(
                matches!(got, Err(Error::Infeasible { .. })),
                "{label}: R found no match but the crate gave {:?}",
                got.map(|o| o.outcome.pairs.len())
            );
            errors += 1;
            continue;
        }
        let ours = got.unwrap_or_else(|error| panic!("{label}: {error}"));
        let r_levels = run.levels.as_ref().expect("levels");
        assert_eq!(ours.levels.len(), r_levels.len(), "{label}: levels");
        for (depth, (mine, theirs)) in ours.levels.iter().zip(r_levels).enumerate() {
            assert_eq!(mine.keys, theirs.keys, "{label}: keys of level {depth}");
            assert_eq!(
                mine.categories.len(),
                theirs.categories.len(),
                "{label}: categories of level {depth}"
            );
            // R labels a category by its values joined with a dot.
            let mut ours_by_label: Vec<(String, usize, usize)> = mine
                .categories
                .iter()
                .map(|category| {
                    let label: Vec<&str> = category
                        .values
                        .iter()
                        .map(|value| value.as_deref().unwrap_or("NA"))
                        .collect();
                    (label.join("."), category.target, category.matched)
                })
                .collect();
            ours_by_label.sort();
            let mut theirs_by_label: Vec<(String, usize, usize)> = theirs
                .categories
                .iter()
                .zip(theirs.target.iter().zip(&theirs.matched))
                .map(|(name, (target, matched))| (name.clone(), *target, *matched))
                .collect();
            theirs_by_label.sort();
            assert_eq!(
                ours_by_label, theirs_by_label,
                "{label}: balance table of level {depth}"
            );
        }

        let pairs = pairs_of(&ours.outcome);
        let r_matches = run.matches.as_ref().expect("matches");
        let r_pairs: BTreeSet<(usize, usize)> = r_matches
            .iter()
            .enumerate()
            .flat_map(|(t, cs)| cs.iter().map(move |c| (t + 1, *c)))
            .collect();
        assert_eq!(pairs.len(), r_pairs.len(), "{label}: number of pairs");

        let total: f64 = pairs.iter().map(|(t, c)| distance(run, *t, *c)).sum();
        let r_total = run.total.expect("total");
        let tol = run.tol.expect("tol");
        let d_max = pairs
            .iter()
            .chain(&r_pairs)
            .map(|(t, c)| distance(run, *t, *c))
            .fold(0.0_f64, f64::max);
        // Costs are about 2^30 or finer here, so each pair is off by at most
        // 3 d_max / 2^30 in the crate's solve.
        let ours_bound = 3.0 * pairs.len() as f64 * d_max / 2_f64.powi(30) + 1e-12;
        // rcbalance rounds each distance to a multiple of tol.
        let r_bound = 0.5 * pairs.len() as f64 * tol;
        assert!(
            total <= r_total + ours_bound,
            "{label}: crate total {total} above R's {r_total}"
        );
        assert!(
            r_total <= total + r_bound + ours_bound,
            "{label}: R total {r_total} above the crate's {total}"
        );
        if run.unique == Some(true) {
            assert_eq!(pairs, r_pairs, "{label}: pairs");
        }
        if let Some(dm) = &run.designmatch {
            assert!(
                (total - dm.total).abs() <= ours_bound + r_bound,
                "{label}: designmatch total {} against {total}",
                dm.total
            );
            designmatch += 1;
        }

        // Exact mode reproduces a balanced match and refuses an unbalanced one.
        let exact = run_ours(run, BalanceMode::Exact);
        let deviation: usize = ours.levels.iter().map(BalanceLevel::total_deviation).sum();
        if deviation == 0 {
            let exact = exact.unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(pairs_of(&exact.outcome), pairs, "{label}: exact mode");
        } else {
            assert!(
                matches!(exact, Err(Error::Infeasible { .. })),
                "{label}: exact mode accepted a deviation of {deviation}"
            );
        }
        compared += 1;
    }
    assert_eq!(errors, 1);
    assert_eq!(compared, reference.runs.len() - errors);
    assert_eq!(designmatch, 2);
}

/// Levels that are missing or not nested are refused.
#[test]
fn input_errors() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/fine_balance.json"
    ));
    let run = &reference.runs[0];
    let mut bad = clone_run(run);
    bad.fb = Vec::new();
    assert!(matches!(
        run_ours(&bad, BalanceMode::Near),
        Err(Error::Option { .. })
    ));
    // "b" is not nested in "a": the units of one category of b differ in a.
    bad.fb = vec![vec!["a".into()], vec!["b".into()]];
    assert!(matches!(
        run_ours(&bad, BalanceMode::Near),
        Err(Error::Option { .. })
    ));
}

fn clone_run(run: &Run) -> Run {
    let units = |u: &Units| Units {
        a: u.a.clone(),
        b: u.b.clone(),
        s: u.s.clone(),
        x1: u.x1.clone(),
        x2: u.x2.clone(),
    };
    Run {
        name: run.name.clone(),
        k: run.k,
        caliper: run.caliper,
        strata: run.strata,
        fb: run.fb.clone(),
        t: units(&run.t),
        c: units(&run.c),
        error: run.error,
        matches: None,
        total: None,
        levels: None,
        unique: None,
        tol: None,
        designmatch: None,
    }
}
