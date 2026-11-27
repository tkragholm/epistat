//! `markov::test`, the Titman-Putter test, checked against `MarkovTest` of mstate 0.3.3 (R
//! 4.6.2), the function its author wrote, on a four-state process with
//! re-entries and tied exit times. `tests/data/markov.R` makes the data and
//! the reference, and records the versions.
//!
//! How the call maps. `transition` is the tested transition, and the stays at
//! risk are those in its origin state. R's `qualset` is the origin plus every
//! state from which the origin can be reached, sorted by state number. Rust
//! numbers the qualifying states with the origin 0 and the others in that
//! order, so the columns are matched through `col_of_code` in the reference.
//! `grid` is the grid of landmarks. `B` is the number of draws, one seed per
//! draw here. `dist = "normal"` is the standard normal multiplier the Rust
//! test uses. `fn` is the default, the mean of the absolute normalised traces,
//! which is the statistic of the Rust test. `fn2` (the chi-squared trace over
//! the other states, with its own p-value), `formula` (covariates in the Cox
//! fit of the intensity), `min_time`, `other_weights` and the optimal weights
//! of `optimal_weights_matrix` have no counterpart in Rust: the test is the
//! one without covariates and the unweighted mean.
//!
//! Definitional differences. R's p-value is the share of draws whose statistic
//! exceeds the observed one, `c / B`. The Rust p-value is `(c + 1) / (B + 1)`,
//! with at or above. The test converts it back to `c / B` and compares that. R
//! reports `zbar = trace / sqrt(variance + (variance == 0))`, so a slot of
//! zero variance has the trace itself, a rounding residue, where Rust has 0.
//! The Rust function returns the trace, the variance, the number of stays
//! occupying each state and the events after each landmark. R returns the
//! normalised trace, the variance and the events after each landmark, and the
//! occupancy counts come from a plain loop in the script.

#![expect(clippy::float_cmp, reason = "the event column holds exactly 0 and 1")]

use epistat_core::Bootstrap;
use epistat_survival::SurvData;
use epistat_survival::markov::{Options, test};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    tests: Vec<Case>,
}

#[derive(Deserialize)]
struct Versions {
    #[serde(rename = "R")]
    r: String,
    mstate: String,
}

#[derive(Deserialize)]
struct RStays {
    person: Vec<u32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    state: Vec<u32>,
    event: Vec<f64>,
}

#[derive(Deserialize)]
struct Case {
    transition: u32,
    qualset: Vec<u32>,
    col_of_code: Vec<usize>,
    grid: Vec<f64>,
    #[serde(rename = "B")]
    draws: usize,
    stays: RStays,
    zbar: Vec<Vec<f64>>,
    variance: Vec<Vec<f64>>,
    orig_stat: Vec<f64>,
    nobs_grid: Vec<f64>,
    p_stat_wb: Vec<f64>,
    occupied: Vec<Vec<f64>>,
    trace_abs: Vec<Vec<f64>>,
    h_max: f64,
    n_at_risk: usize,
}

/// The stays as follow-up data. The reference gives the state of each stay as
/// its position among the qualifying states, so position `j` is the status
/// code `j + 1`, and the code `q + 1` is the one state that is not
/// qualifying, which the tested transition may end in. A stay ends in the state of the
/// next stay of its person, and in the target when it is the last and the
/// reference marks an event. The target is the state that the stays at risk
/// move to.
struct Follow {
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    istate: Vec<u32>,
    person: Vec<u32>,
    target: u32,
}

fn follow(stays: &RStays, q: usize) -> Follow {
    let n = stays.person.len();
    let other = u32::try_from(q).unwrap() + 1;
    let code = |row: usize| stays.state[row] + 1;
    let next = |row: usize| {
        (row + 1 < n && stays.person[row + 1] == stays.person[row]).then(|| code(row + 1))
    };
    let mut targets: Vec<u32> = (0..n)
        .filter(|&row| stays.state[row] == 0 && stays.event[row] == 1.0)
        .map(|row| next(row).unwrap_or(other))
        .collect();
    targets.sort_unstable();
    targets.dedup();
    assert_eq!(targets.len(), 1, "the stays at risk end in one target");
    let target = targets[0];
    let status = (0..n)
        .map(|row| {
            let reached = next(row).unwrap_or(if stays.event[row] == 1.0 { target } else { 0 });
            i32::try_from(reached).unwrap()
        })
        .collect();
    Follow {
        start: stays.start.clone(),
        stop: stays.stop.clone(),
        status,
        istate: (0..n).map(code).collect(),
        person: stays.person.clone(),
        target,
    }
}

/// Unit roundoff of a double.
const U: f64 = f64::EPSILON / 2.0;

/// Draws of the Rust bootstrap. More than R's, which makes the comparison of
/// the two p-values less noisy than its own Monte Carlo error alone.
const RUST_DRAWS: usize = 20_000;

/// Both p-values estimate one probability from independent draws. Their
/// difference has a standard error of `sqrt(p (1 - p) (1 / B_r + 1 / B_rust))`
/// with `p` the pooled share, smoothed by one draw in each group so that it is
/// not 0 when neither group exceeded. The test allows this many of them.
const STANDARD_ERRORS: f64 = 5.0;

#[test]
fn markov_test_matches_mstate() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/markov.json"
    ));
    eprintln!(
        "{} mstate {}",
        reference.versions.r, reference.versions.mstate
    );
    let seeds = Bootstrap::new(RUST_DRAWS, 0);
    for case in &reference.tests {
        let name = format!("transition {}", case.transition);
        let k = case.grid.len();
        let q = case.qualset.len();
        let s = &case.stays;
        let f = follow(s, q);
        let data = SurvData::builder(&f.stop, &f.status)
            .start(&f.start)
            .n_codes(u32::try_from(q).unwrap() + 1)
            .id(&f.person)
            .istate(&f.istate)
            .build()
            .unwrap();
        let qualifying: Vec<u32> = (1..=u32::try_from(q).unwrap()).collect();
        let out = test(
            &data,
            &qualifying,
            f.target,
            &case.grid,
            seeds,
            &Options::default(),
        )
        .unwrap();
        assert_eq!(out.n_rows, s.person.len(), "{name}");
        assert_eq!(out.n_used, s.person.len(), "{name}");
        // The events that the status gives are the reference's, at the stays
        // at risk.
        for row in (0..s.person.len()).filter(|&row| s.state[row] == 0) {
            assert_eq!(
                f64::from(u8::from(u32::try_from(f.status[row]).unwrap() == f.target)),
                s.event[row],
                "{name}: event of stay {row}"
            );
        }

        // Deterministic outputs are compared in the order of the Rust states.
        let by_code = |m: &Vec<Vec<f64>>, landmark: usize, code: usize| {
            m[landmark][case.col_of_code[code] - 1]
        };
        // n terms in each sum, both implementations: the bounds below are
        // doubled for the pair.
        let n = case.n_at_risk as f64;
        // The variance is a sum of non-negative terms over at most n events
        // and n risk-set members each, so it carries a relative error of at
        // most (2 n + 8) u.
        let variance_bound = 2.0 * (2.0 * n + 8.0) * U;

        assert_eq!(out.n_events_after, case.nobs_grid, "{name}: events after");
        let mut worst = [0.0f64; 4];
        let mut stat_bound = vec![0.0f64; q];
        for landmark in 0..k {
            for (code, bound) in stat_bound.iter_mut().enumerate() {
                let slot = landmark * q + code;
                assert!(
                    out.n_occupied[slot].to_bits() == case.occupied[landmark][code].to_bits(),
                    "{name}: occupancy at landmark {landmark}, state {code}"
                );
                let v_r = by_code(&case.variance, landmark, code);
                let v = out.variance[slot];
                // The trace is a sum of at most n_occ products of a
                // martingale residual and 0 or 1. Each residual carries the
                // error of the Nelson-Aalen sum, at most (n + 2) u h_max
                // absolute at each of its two evaluations, and the sum adds
                // at most (n + 1) u times the sum of absolute terms.
                let occupied = out.n_occupied[slot];
                let trace_bound = 2.0
                    * U
                    * ((n + 1.0) * case.trace_abs[landmark][code]
                        + 2.0 * (n + 2.0) * case.h_max * occupied);
                let z_r = by_code(&case.zbar, landmark, code);
                if v_r == 0.0 {
                    // R reports the trace itself; Rust reports 0.
                    assert!(
                        v.to_bits() == 0,
                        "{name}: variance at landmark {landmark}, state {code}"
                    );
                    assert!(
                        z_r.abs() <= trace_bound,
                        "{name}: residue {z_r} beyond {trace_bound}"
                    );
                    *bound += trace_bound / k as f64;
                    continue;
                }
                let v_tol = variance_bound * v_r;
                worst[0] = worst[0].max((v - v_r).abs() / v_tol);
                assert!((v - v_r).abs() <= v_tol, "{name}: variance {v} vs {v_r}");
                let z = out.trace[slot] / v.sqrt();
                let z_tol = trace_bound / v_r.sqrt() + z_r.abs() * (n + 4.0) * U * 2.0;
                worst[1] = worst[1].max((z - z_r).abs() / z_tol);
                assert!(
                    (z - z_r).abs() <= z_tol,
                    "{name}: normalised trace {z} vs {z_r}, tolerance {z_tol}"
                );
                *bound += z_tol / k as f64;
            }
        }
        for (code, bound) in stat_bound.iter().enumerate() {
            let r_stat = case.orig_stat[case.col_of_code[code] - 1];
            let tol = bound + 2.0 * U * r_stat;
            let got = out.statistic[code].expect("the state has a positive variance");
            worst[2] = worst[2].max((got - r_stat).abs() / tol);
            assert!(
                (got - r_stat).abs() <= tol,
                "{name}: statistic of state {code}: {got} vs {r_stat}"
            );
        }

        // The p-values, as counts over the draws.
        let (b_r, b_rust) = (case.draws as f64, RUST_DRAWS as f64);
        for code in 0..q {
            let p_r = case.p_stat_wb[case.col_of_code[code] - 1];
            let p_rust = out.p_value[code].expect("draws were given");
            let exceeding = p_rust * (b_rust + 1.0) - 1.0;
            let share_rust = exceeding / b_rust;
            let pooled = (p_r * b_r + exceeding + 1.0) / (b_r + b_rust + 2.0);
            let se = (pooled * (1.0 - pooled) * (1.0 / b_r + 1.0 / b_rust)).sqrt();
            let deviation = (share_rust - p_r).abs() / (STANDARD_ERRORS * se);
            worst[3] = worst[3].max(deviation);
            assert!(
                deviation <= 1.0,
                "{name}: state {code}: p {p_rust} (c/B {share_rust}) vs R {p_r}, {STANDARD_ERRORS} standard errors {}",
                STANDARD_ERRORS * se
            );
        }
        eprintln!(
            "{name}: variance {:.3}, normalised trace {:.3}, statistic {:.3}, p-value {:.3} of their bounds",
            worst[0], worst[1], worst[2], worst[3]
        );
    }
}
