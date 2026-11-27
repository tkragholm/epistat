//! The smoothness (`Delta^SD(M)`) intervals against `HonestDiD` 0.2.8.
//!
//! The reference is written by `tests/data/generate_smoothness_reference.R`:
//! `BCdata_EventStudy` and `LWdata_EventStudy`, several post-period
//! functionals, and grids of `M` from 0 to beyond the point where the
//! pre-period coefficients stop contradicting the restriction.

use crate::sensitivity::{
    HonestBiasDirection, HonestEventStudyInput, HonestMonotonicityDirection,
    LeastFavorableSimulation, SensitivitySummaryMethod,
    summarize_smoothness_sensitivity_with_simulation,
};
use crate::sensitivity::{
    SmoothnessFlciConfig, SmoothnessFlciResult, build_smoothness_flci_problem,
    build_smoothness_multi_flci_problem, compute_smoothness_flci,
    compute_smoothness_flci_with_config, compute_smoothness_identified_set,
    compute_smoothness_multi_flci,
};
use crate::util::InferenceConfig;
use epistat_stats::dist::normal;
use epistat_testkit::{assert_same_on_threads, load_json, na_f64};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Reference {
    alpha: f64,
    folded_normal_draws: f64,
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct Dataset {
    name: String,
    betahat: Vec<f64>,
    sigma: Vec<Vec<f64>>,
    num_pre: usize,
    num_post: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    target: String,
    l_vec: Vec<f64>,
    m: f64,
    identified: Identified,
    flci: Option<Flci>,
    conditional: Option<BTreeMap<String, Conditional>>,
}

#[derive(Deserialize)]
struct Identified {
    feasible: bool,
    lb: f64,
    ub: f64,
}

#[derive(Deserialize)]
struct Flci {
    lb: f64,
    ub: f64,
    half_length: f64,
    optimal_vec: Vec<f64>,
}

#[derive(Deserialize)]
struct Conditional {
    #[serde(deserialize_with = "na_f64")]
    lb: f64,
    #[serde(deserialize_with = "na_f64")]
    ub: f64,
    accepted_runs: usize,
    grid_lb: f64,
    grid_ub: f64,
    grid_step: f64,
    grid_points: usize,
}

fn reference() -> Reference {
    load_json("tests/data/honest_did_smoothness_ref.json")
}

fn input(dataset: &Dataset) -> HonestEventStudyInput {
    let num_pre = i32::try_from(dataset.num_pre).expect("pre periods");
    let num_post = i32::try_from(dataset.num_post).expect("post periods");
    HonestEventStudyInput {
        betahat: dataset.betahat.clone(),
        covariance: dataset.sigma.clone(),
        pre_periods: (-num_pre - 1..-1).collect(),
        post_periods: (0..num_post).collect(),
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn standard_deviation(weights: &[f64], sigma: &[Vec<f64>]) -> f64 {
    let variance: f64 = weights
        .iter()
        .zip(sigma)
        .map(|(left, row)| left * dot(row, weights))
        .sum();
    variance.sqrt()
}

/// Standard error of `HonestDiD`'s half-length `chi h`, where its `chi` is the
/// `p` quantile of `|Z + mu|` over `draws` normal draws: the quantile's
/// standard error `sqrt(p (1 - p) / draws) / f(chi)`, with `f` the folded
/// normal density, times `h`. `h` is the standard deviation of the weights and
/// `mu` is solved from `chi = half / h`.
fn folded_quantile_standard_error(
    result: &SmoothnessFlciResult,
    sigma: &[Vec<f64>],
    p: f64,
    draws: f64,
) -> f64 {
    let h = standard_deviation(&result.optimal_vec, sigma);
    let chi = result.optimal_half_length / h;
    let quantile = |mu: f64| normal::folded_quantile(p, mu);
    let (mut low, mut high) = (0.0, chi);
    if quantile(low) < chi {
        for _ in 0..200 {
            let mid = f64::midpoint(low, high);
            if quantile(mid) < chi {
                low = mid;
            } else {
                high = mid;
            }
        }
    }
    let mu = f64::midpoint(low, high);
    let density = normal::pdf(chi - mu) + normal::pdf(chi + mu);
    h * (p * (1.0 - p) / draws).sqrt() / density
}

/// FLCI: the half-length within four Monte Carlo standard errors of
/// `HonestDiD`'s quantile, plus the solver tolerance.
///
/// `HonestDiD` minimises the half-length over `h` by bisection with its
/// simulated quantile; here the minimum is taken over a grid of `h` with the
/// exact quantile. A grid misses the minimum by about the square of its
/// spacing, so the 189-point grid, which contains the production 48-point grid
/// and has a quarter of its spacing, misses it by a sixteenth as much; its
/// minimum is compared with R directly. The production interval may be longer by what the coarser grid
/// misses, which the test computes as the difference between the two grids
/// and adds to its bound. Both sides solve the cone programs to `1e-8` (ECOS
/// and Clarabel defaults); `1e-6` of the half-length is a hundred times that.
///
/// The endpoints are the centre `a' beta_hat` plus or minus the half-length.
/// Where the minimum is flat in `h`, near-optimal weights differ between the
/// two searches, and the centres with them; that difference is computed from
/// the two weight vectors and allowed in the endpoints.
#[test]
fn smoothness_flci_matches_honestdid() {
    let reference = reference();
    let confidence = 1.0 - reference.alpha;
    let inference = InferenceConfig::new(confidence);
    let mut worst_half = 0.0_f64;
    let mut worst_endpoint = 0.0_f64;
    let mut worst_grid = 0.0_f64;
    let mut cases = 0;
    for dataset in &reference.datasets {
        let input = input(dataset);
        for case in &dataset.cases {
            let Some(expected) = &case.flci else { continue };
            cases += 1;
            let label = format!("{} {} M = {}", dataset.name, case.target, case.m);
            let problem = build_smoothness_flci_problem(&input, &case.l_vec, case.m, inference)
                .expect("problem");
            let got = compute_smoothness_flci(&problem).expect("flci");
            let fine = compute_smoothness_flci_with_config(
                &problem,
                SmoothnessFlciConfig {
                    num_grid_points: 189,
                },
            )
            .expect("fine flci");
            let solver = 1e-6 * expected.half_length;
            let monte_carlo = 4.0
                * folded_quantile_standard_error(
                    &fine,
                    &dataset.sigma,
                    confidence,
                    reference.folded_normal_draws,
                );

            let grid_excess = got.optimal_half_length - fine.optimal_half_length;
            assert!(grid_excess >= -solver, "{label}: the finer grid is longer");
            let half_gap = (fine.optimal_half_length - expected.half_length).abs();
            assert!(
                half_gap <= monte_carlo + solver,
                "{label}: half-length {} against {} (bound {monte_carlo:e})",
                fine.optimal_half_length,
                expected.half_length
            );
            let production_gap = (got.optimal_half_length - expected.half_length).abs();
            assert!(
                production_gap <= monte_carlo + grid_excess + solver,
                "{label}"
            );

            let centre_gap = (dot(&got.optimal_vec, &dataset.betahat)
                - dot(&expected.optimal_vec, &dataset.betahat))
            .abs();
            let endpoint_gap = (got.flci.0 - expected.lb)
                .abs()
                .max((got.flci.1 - expected.ub).abs());
            assert!(
                endpoint_gap <= centre_gap + production_gap + solver,
                "{label}: ({}, {}) against ({}, {})",
                got.flci.0,
                got.flci.1,
                expected.lb,
                expected.ub
            );
            worst_half = worst_half.max(half_gap / monte_carlo);
            worst_grid = worst_grid.max(grid_excess / expected.half_length);
            worst_endpoint = worst_endpoint.max(endpoint_gap / (expected.ub - expected.lb));
        }
    }
    println!(
        "smoothness FLCI, {cases} cases: half-length within {worst_half:.2} of its bound, \
         production grid longer by at most {worst_grid:.1e} relative, endpoints off by at \
         most {worst_endpoint:.1e} of the width"
    );
}

/// C-F: the conditional test with the FLCI first stage, read off a grid of
/// 1,000 points spanning the first-stage interval at level `1 - alpha / 10`.
///
/// Both sides return grid points, so each bound may sit a grid step away on
/// either side, and the grids themselves start from the two first-stage
/// intervals, whose difference the test computes; where a bound is set by the
/// first stage it moves with it. The bound is therefore one step of each grid
/// plus the larger difference between the first-stage endpoints.
///
/// `HonestDiD` reports the outermost accepted grid points. Its accepted set
/// is sometimes several separate runs, mostly at small `M` where the
/// pre-period coefficients contradict the restriction; epistat-did searches
/// outward from the estimate by bisection and stops at the edge of a run, so
/// there the interval is checked to lie within `HonestDiD`'s.
#[test]
fn smoothness_conditional_flci_matches_honestdid() {
    let reference = reference();
    let confidence = 1.0 - reference.alpha;
    let inference = InferenceConfig::new(confidence);
    let first_stage_inference = InferenceConfig::new(1.0 - reference.alpha / 10.0);
    let mut worst = 0.0_f64;
    let mut cases = 0;
    let mut separate_runs = 0;
    for dataset in &reference.datasets {
        let input = input(dataset);
        for case in &dataset.cases {
            let Some(conditional) = &case.conditional else {
                continue;
            };
            let first_stage = compute_smoothness_flci(
                &build_smoothness_flci_problem(&input, &case.l_vec, case.m, first_stage_inference)
                    .expect("first-stage problem"),
            )
            .expect("first stage");
            for (delta, expected) in conditional {
                cases += 1;
                let label = format!("{} {} {delta} M = {}", dataset.name, case.target, case.m);
                let (bias, monotone) = match delta.as_str() {
                    "DeltaSD" => (None, None),
                    "DeltaSDPB" => (Some(HonestBiasDirection::Positive), None),
                    "DeltaSDNB" => (Some(HonestBiasDirection::Negative), None),
                    "DeltaSDI" => (None, Some(HonestMonotonicityDirection::Increasing)),
                    "DeltaSDD" => (None, Some(HonestMonotonicityDirection::Decreasing)),
                    other => panic!("unknown restriction {other}"),
                };
                let got = summarize_smoothness_sensitivity_with_simulation(
                    &input,
                    inference,
                    &case.l_vec,
                    Some(SensitivitySummaryMethod::ConditionalFlci),
                    Some(&[case.m]),
                    bias,
                    monotone,
                    LeastFavorableSimulation::new(1_000, 0),
                );
                assert!(expected.lb.is_finite(), "{label}: R accepted no grid point");
                let row = &got.expect("conditional").rows[0];
                let first_stage_gap = (first_stage.flci.0 - expected.grid_lb)
                    .abs()
                    .max((first_stage.flci.1 - expected.grid_ub).abs());
                let own_step = (first_stage.flci.1 - first_stage.flci.0)
                    / f64::from(u32::try_from(expected.grid_points - 1).expect("grid"));
                let tolerance = expected.grid_step + own_step + first_stage_gap + 1e-9;
                if expected.accepted_runs == 1 {
                    let gap = (row.lb - expected.lb)
                        .abs()
                        .max((row.ub - expected.ub).abs());
                    assert!(
                        gap <= tolerance,
                        "{label}: ({}, {}) against ({}, {}), bound {tolerance:e}",
                        row.lb,
                        row.ub,
                        expected.lb,
                        expected.ub
                    );
                    worst = worst.max(gap / tolerance);
                } else {
                    separate_runs += 1;
                    assert!(
                        expected.lb - tolerance <= row.lb
                            && row.lb <= row.ub
                            && row.ub <= expected.ub + tolerance,
                        "{label}: ({}, {}) outside ({}, {})",
                        row.lb,
                        row.ub,
                        expected.lb,
                        expected.ub
                    );
                }
            }
        }
    }
    println!(
        "smoothness C-F, {cases} cases: within {worst:.2} of the bound where HonestDiD \
         accepts one run, {separate_runs} with several runs inside HonestDiD's interval"
    );
}

/// Identified sets: equal to `HonestDiD`'s linear programs where they are
/// feasible, to a hundred times Clarabel's `1e-8` tolerance on the bound's
/// scale (Rglpk's simplex is exact there); NaN where they are not, where
/// `HonestDiD` returns the point estimate with a warning.
#[test]
fn smoothness_identified_set_matches_honestdid() {
    let reference = reference();
    for dataset in &reference.datasets {
        let input = input(dataset);
        for case in &dataset.cases {
            let label = format!("{} {} M = {}", dataset.name, case.target, case.m);
            let got =
                compute_smoothness_identified_set(&input, &case.l_vec, case.m).expect("identified");
            if case.identified.feasible {
                for (ours, theirs) in [(got.lb, case.identified.lb), (got.ub, case.identified.ub)] {
                    assert!(
                        (ours - theirs).abs() <= 1e-6 * (1.0 + theirs.abs()),
                        "{label}: {ours} against {theirs}"
                    );
                }
            } else {
                assert!(got.lb.is_nan() && got.ub.is_nan(), "{label}: {got:?}");
            }
        }
    }
}

#[test]
fn smoothness_intervals_do_not_depend_on_the_thread_count() {
    let reference = reference();
    let dataset = &reference.datasets[0];
    let input = input(dataset);
    let inference = InferenceConfig::new(1.0 - reference.alpha);
    let weights = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.25; 4],
        vec![1.0, -1.0, 0.0, 0.0],
    ];
    assert_same_on_threads(|| {
        let mut out = Vec::new();
        let problem = build_smoothness_multi_flci_problem(&input, &weights, 0.05, inference)
            .expect("multi problem");
        let multi = compute_smoothness_multi_flci(&problem).expect("multi flci");
        for point in &multi.points {
            out.extend([point.flci.0, point.flci.1]);
        }
        for monotone in [None, Some(HonestMonotonicityDirection::Decreasing)] {
            let summary = summarize_smoothness_sensitivity_with_simulation(
                &input,
                inference,
                &weights[1],
                Some(SensitivitySummaryMethod::ConditionalFlci),
                Some(&[0.0, 0.1]),
                None,
                monotone,
                LeastFavorableSimulation::new(1_000, 0),
            )
            .expect("conditional");
            out.extend(summary.rows.iter().flat_map(|row| [row.lb, row.ub]));
        }
        out
    });
}
