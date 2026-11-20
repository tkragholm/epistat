//! The Fine-Gray rows and fit checked against `survival::finegray` with
//! `coxph`, and the fit against `cmprsk::crr`. The reference is written by
//! `tests/data/fine_gray.R`, which records the versions it ran under.
//!
//! Row weights are products of at most `m + 1` factors, where `m` is the
//! number of distinct censoring times: the Kaplan-Meier steps `1 - d / n` up
//! to the interval and one division by `G` before the event. Each factor has a
//! relative error of a few `epsilon`, so two implementations of the same
//! arithmetic differ by at most `(2 m + 4) epsilon` relative to the weight. A
//! sum of `k` such weights adds `k epsilon`. Starts, stops and statuses are
//! compared exactly.
//!
//! The one-covariate fit bounds are derived at `coefficient_deviation`. The
//! other one-covariate outputs are bounded by `8 epsilon` relative to the value
//! or to 1 for a value below 1, as in `r_cox.rs`, where the factor is the
//! condition number of the information, which is 1 for one covariate. `coxph`
//! and the Rust fit run with the default convergence rule, so both stop at the
//! same iteration, which the test checks.
//!
//! The fits on several covariates have the bounds of `r_cox.rs` and
//! `r_cox_control.rs` with the term for the weights derived at
//! `multi_bounds`. The comparison with `crr` runs the Rust fit to a tighter
//! `eps`, because `crr` stops at its own tolerance near the root.

#![expect(
    clippy::float_cmp,
    reason = "starts, stops and times are copied from the data and compared exactly"
)]

use std::collections::BTreeMap;

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::Ties;
use epistat_survival::fine_gray::{
    FineGrayOptions, Rows, RowsOptions, fit, float_codes, risk_rows, rows,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    r_version: String,
    packages: BTreeMap<String, String>,
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct Dataset {
    name: String,
    time: Vec<f64>,
    event: Vec<i32>,
    weight: Vec<f64>,
    x: Vec<f64>,
    bin: Vec<f64>,
    fac: Vec<u32>,
    id: Vec<u32>,
    group: Vec<u32>,
    results: Vec<CauseResult>,
}

#[derive(Deserialize)]
struct CauseResult {
    cause: i32,
    rows: RRows,
    fits: Vec<RFit>,
    rows_weights_argument: Option<RRows>,
    crr: Option<Vec<RCrr>>,
    multi: Vec<RMulti>,
    crr_multi: Option<RCrrMulti>,
}

/// `coxph` on the rows with `bin + x + factor(fac)`, whose columns are bin, x
/// and the indicators of levels 1 and 2 of fac.
#[derive(Deserialize)]
struct RMulti {
    ties: String,
    cluster: String,
    coefficients: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    info: Vec<Vec<f64>>,
    vcov_robust: Vec<Vec<f64>>,
    loglik: f64,
    loglik_null: f64,
    iterations: u32,
    wald_model: f64,
    wald: f64,
    score: f64,
    kappa: f64,
    kappa_init: f64,
    kappa_robust: Option<f64>,
    n_events: f64,
}

#[derive(Deserialize)]
struct RCrrMulti {
    coefficients: Vec<f64>,
    information: Vec<Vec<f64>>,
    loglik: f64,
}

#[derive(Deserialize)]
struct RRows {
    orig: Vec<usize>,
    start: Vec<f64>,
    stop: Vec<f64>,
    status: Vec<i32>,
    weight: Vec<f64>,
}

#[derive(Deserialize)]
struct RFit {
    covariate: String,
    ties: String,
    cluster: String,
    coefficient: f64,
    se: f64,
    robust_se: Option<f64>,
    iterations: u32,
    n_events: f64,
}

#[derive(Deserialize)]
struct RCrr {
    covariate: String,
    coefficient: f64,
    se: f64,
    information: f64,
    loglik: f64,
    replica: f64,
}

/// The tolerance `gtol` given to `crr` in the script.
const CRR_GTOL: f64 = 1e-8;

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/fine_gray.json"
    ))
}

/// The data of a reference set with the given weights and clusters. The
/// status is the cause code, and the largest code is the number of causes.
fn surv<'a>(data: &'a Dataset, weight: &'a [f64], cluster: Option<&'a [u32]>) -> SurvData<'a> {
    let n_codes = u32::try_from(data.event.iter().copied().max().unwrap()).unwrap();
    let mut builder = SurvData::builder(&data.time, &data.event)
        .n_codes(n_codes)
        .weight(weight);
    if let Some(cluster) = cluster {
        builder = builder.cluster(cluster);
    }
    builder.build().unwrap()
}

fn cause_code(result: &CauseResult) -> u32 {
    u32::try_from(result.cause).unwrap()
}

/// The columns of a design matrix with one covariate or with bin, x and the
/// indicators of levels 1 and 2 of fac.
fn design(data: &Dataset, names: &[&str]) -> Mat<f64> {
    Mat::from_fn(data.time.len(), names.len(), |i, j| match names[j] {
        "x" => data.x[i],
        "bin" => data.bin[i],
        "fac1" => f64::from(u8::from(data.fac[i] == 1)),
        "fac2" => f64::from(u8::from(data.fac[i] == 2)),
        other => panic!("unknown covariate {other}"),
    })
}

const MULTI: [&str; 4] = ["bin", "x", "fac1", "fac2"];

fn ties_of(name: &str) -> Ties {
    if name == "efron" {
        Ties::Efron
    } else {
        Ties::Breslow
    }
}

fn cluster_of<'a>(data: &'a Dataset, name: &str) -> Option<&'a [u32]> {
    match name {
        "none" => None,
        "id" => Some(data.id.as_slice()),
        "group" => Some(data.group.as_slice()),
        other => panic!("unknown cluster {other}"),
    }
}

fn covariate<'a>(data: &'a Dataset, name: &str) -> &'a [f64] {
    match name {
        "x" => &data.x,
        "bin" => &data.bin,
        other => panic!("unknown covariate {other}"),
    }
}

/// The number of distinct censoring times.
fn censoring_times(data: &Dataset) -> usize {
    let mut times: Vec<f64> = (0..data.time.len())
        .filter(|&i| data.event[i] == 0)
        .map(|i| data.time[i])
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times.len()
}

/// Times at which the cause of interest happens.
fn event_times(data: &Dataset, cause: i32) -> Vec<f64> {
    let mut times: Vec<f64> = (0..data.time.len())
        .filter(|&i| data.event[i] == cause)
        .map(|i| data.time[i])
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times
}

/// Positions of the rows of each individual, in order.
fn by_individual(rows: &[usize]) -> BTreeMap<usize, Vec<usize>> {
    let mut map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (position, &individual) in rows.iter().enumerate() {
        map.entry(individual).or_default().push(position);
    }
    map
}

fn relative(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs()
}

/// What `survival::finegray` leaves out or merges, found by comparing.
#[derive(Default)]
struct RowReport {
    /// The largest relative weight deviation, in units of its bound.
    worst: f64,
    r_rows: usize,
    merged: usize,
    uncovered: usize,
}

/// Compare the rows `expand` writes with the rows of R.
///
/// `finegray` writes one row between two censoring times only when the cause
/// of interest happens in the interval that follows, so its rows can span
/// several of ours. Each R row must then start where one of ours starts, have
/// its weight, and be covered by a chain of ours without a gap, with no event
/// time of the cause inside the chain after its first row, because the weight
/// there is not the R row's. A row of ours that R does not write, because it
/// is past the last event time of the cause or between two rows R keeps, must
/// contain no event time of the cause. `scale` multiplies the weight of our rows of
/// each individual.
fn compare_rows(
    ours: &Rows,
    theirs: &RRows,
    scale: &[f64],
    events: &[f64],
    bound: f64,
    label: &str,
) -> RowReport {
    let mut report = RowReport::default();
    let mine = by_individual(&ours.row.iter().map(|&r| r as usize).collect::<Vec<_>>());
    let r_of = by_individual(&theirs.orig.iter().map(|&o| o - 1).collect::<Vec<_>>());
    for (&individual, r_positions) in &r_of {
        let own = mine.get(&individual).unwrap_or_else(|| {
            panic!("{label}: individual {individual} has no rows of ours");
        });
        let mut next = 0usize;
        for &p in r_positions {
            report.r_rows += 1;
            // Rows of ours that R leaves out before this one hold no event time.
            while ours.start[own[next]] < theirs.start[p] {
                report.uncovered += 1;
                assert!(
                    !events
                        .iter()
                        .any(|&t| ours.start[own[next]] < t && t <= ours.stop[own[next]]),
                    "{label}: a row of ours with an event time that R does not write"
                );
                next += 1;
            }
            let first = own[next];
            assert_eq!(ours.start[first], theirs.start[p], "{label} start");
            assert_eq!(
                ours.event[first], theirs.status[p],
                "{label} status of individual {individual}"
            );
            let ours_weight = ours.weight[first] * scale[individual];
            let deviation = relative(ours_weight, theirs.weight[p]) / bound;
            report.worst = report.worst.max(deviation);
            assert!(
                deviation <= 1.0,
                "{label} weight of individual {individual}: {ours_weight} against {}",
                theirs.weight[p]
            );
            let mut last = first;
            while ours.stop[last] < theirs.stop[p] {
                next += 1;
                let following = own[next];
                assert_eq!(ours.stop[last], ours.start[following], "{label} gap");
                assert!(
                    !events
                        .iter()
                        .any(|&t| ours.start[following] < t && t <= ours.stop[following]),
                    "{label}: an event time inside a merged row of individual {individual}"
                );
                last = following;
                report.merged += 1;
            }
            assert_eq!(ours.stop[last], theirs.stop[p], "{label} stop");
            next += 1;
        }
        for &position in &own[next..] {
            report.uncovered += 1;
            assert!(
                !events
                    .iter()
                    .any(|&t| ours.start[position] < t && t <= ours.stop[position]),
                "{label}: a row of ours with an event time that R does not write"
            );
        }
    }
    for individual in mine.keys() {
        if !r_of.contains_key(individual) {
            // Nothing from R: the individual's rows lie past every event time.
            for &position in &mine[individual] {
                assert!(
                    !events
                        .iter()
                        .any(|&t| ours.start[position] < t && t <= ours.stop[position]),
                    "{label}: rows of individual {individual} missing from R"
                );
            }
        }
    }
    report
}

#[test]
fn expansion_matches_finegray() {
    let reference = reference();
    eprintln!("{} {:?}", reference.r_version, reference.packages);
    for data in &reference.datasets {
        let m = censoring_times(data);
        let bound = (2 * m + 4) as f64 * f64::EPSILON;
        let ones = vec![1.0; data.time.len()];
        for result in &data.results {
            let label = format!("{} cause {}", data.name, result.cause);
            let events = event_times(data, result.cause);
            let ours = rows(
                &surv(data, &data.weight, None),
                cause_code(result),
                &RowsOptions::default(),
            )
            .unwrap();
            let report = compare_rows(&ours, &result.rows, &ones, &events, bound, &label);
            eprintln!(
                "{label:22} R rows {:4}  merged {:3}  rows R omits {:3}  worst weight {:.2e} of its bound",
                report.r_rows, report.merged, report.uncovered, report.worst
            );
            if let Some(argument) = &result.rows_weights_argument {
                // finegray(weights =) leaves the case weights out of the
                // censoring distribution, so its rows are the unweighted rows
                // times the case weight.
                let unit = rows(
                    &surv(data, &ones, None),
                    cause_code(result),
                    &RowsOptions::default(),
                )
                .unwrap();
                let report = compare_rows(
                    &unit,
                    argument,
                    &data.weight,
                    &events,
                    bound,
                    &format!("{label} weights argument"),
                );
                eprintln!(
                    "{label:22} weights= rows: worst weight {:.2e} of its bound",
                    report.worst
                );
            }
        }
    }
}

/// The weight at risk at each event time of the cause, and the weight of the
/// events, per covariate level, from rows with their levels.
fn risk_sums(
    rows: (&[f64], &[f64], &[i32], &[f64]),
    levels: &[u32],
    events: &[f64],
) -> BTreeMap<(u64, u32), (f64, f64, usize)> {
    let (start, stop, status, weight) = rows;
    let mut sums: BTreeMap<(u64, u32), (f64, f64, usize)> = BTreeMap::new();
    for (k, &level) in levels.iter().enumerate() {
        for &t in events {
            let entry = sums.entry((t.to_bits(), level)).or_default();
            if start[k] < t && t <= stop[k] {
                entry.0 += weight[k];
                entry.2 += 1;
            }
            if status[k] == 1 && stop[k] == t {
                entry.1 += weight[k];
            }
        }
    }
    sums
}

#[test]
fn collapsed_rows_carry_the_risk_sets_of_finegray() {
    let reference = reference();
    for data in &reference.datasets {
        let m = censoring_times(data);
        for result in &data.results {
            let events = event_times(data, result.cause);
            let weighted = surv(data, &data.weight, None);
            for name in ["bin", "x"] {
                let levels = float_codes(covariate(data, name));
                let ours = risk_rows(
                    &weighted,
                    cause_code(result),
                    &levels,
                    &RowsOptions::default(),
                )
                .unwrap();
                let our_levels: Vec<u32> = ours.row.iter().map(|&r| levels[r as usize]).collect();
                let theirs = &result.rows;
                let their_levels: Vec<u32> = theirs.orig.iter().map(|&o| levels[o - 1]).collect();
                let ours_sums = risk_sums(
                    (&ours.start, &ours.stop, &ours.event, &ours.weight),
                    &our_levels,
                    &events,
                );
                let their_sums = risk_sums(
                    (&theirs.start, &theirs.stop, &theirs.status, &theirs.weight),
                    &their_levels,
                    &events,
                );
                let mut worst = 0.0f64;
                for (key, &(r_risk, r_events, terms)) in &their_sums {
                    let (risk, event_weight, our_terms) = ours_sums[key];
                    let bound = (2 * m + 4 + terms + our_terms) as f64 * f64::EPSILON;
                    let deviation = (relative_or_zero(risk, r_risk)
                        .max(relative_or_zero(event_weight, r_events)))
                        / bound;
                    worst = worst.max(deviation);
                    assert!(
                        deviation <= 1.0,
                        "{} cause {} {name}: risk {risk} against {r_risk}, events {event_weight} against {r_events}",
                        data.name,
                        result.cause
                    );
                }
                eprintln!(
                    "{:12} cause {} {name:3} collapsed {:4} rows of {:4}  worst risk-set sum {worst:.2e} of its bound",
                    data.name,
                    result.cause,
                    ours.row.len(),
                    theirs.orig.len()
                );
            }
        }
    }
}

fn relative_or_zero(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        (got - want).abs() / want.abs()
    }
}

/// Deviation in units of `8 epsilon max(|want|, 1)`.
fn fit_deviation(got: f64, want: f64) -> f64 {
    (got - want).abs() / (8.0 * f64::EPSILON * want.abs().max(1.0))
}

/// Deviation of a coefficient in units of its bound.
///
/// The root of the score moves by the error of the score over the information
/// `I`. Each event adds a term of at most `range` to the score, in the sum and
/// in the risk-set means, so with `W` the weight of the events and 8 rounding
/// steps per term the score has an error of at most `8 epsilon W range`. The
/// bound is that over `I = 1 / se^2`, or `8 epsilon |beta|` when that is
/// larger.
fn coefficient_deviation(got: f64, want: f64, events: f64, range: f64, se: f64) -> f64 {
    let bound = 8.0 * f64::EPSILON * (events * range * se * se).max(want.abs());
    (got - want).abs() / bound
}

#[test]
fn fits_match_coxph_on_finegray_rows() {
    let reference = reference();
    let mut worst = [0.0f64; 4];
    for data in &reference.datasets {
        for result in &data.results {
            for r in &result.fits {
                let x = covariate(data, &r.covariate);
                let cluster = cluster_of(data, &r.cluster);
                let column = design(data, &[r.covariate.as_str()]);
                let fit = fit(
                    &surv(data, &data.weight, cluster),
                    column.as_ref(),
                    &FineGrayOptions {
                        ties: ties_of(&r.ties),
                        ..FineGrayOptions::new(cause_code(result))
                    },
                )
                .unwrap();
                let label = format!(
                    "{} cause {} {} {} {}",
                    data.name, result.cause, r.covariate, r.ties, r.cluster
                );
                assert!(fit.converged, "{label}");
                assert_eq!(fit.iterations, r.iterations, "{label}");

                let range = x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - x.iter().copied().fold(f64::INFINITY, f64::min);
                let coefficient = coefficient_deviation(
                    fit.coefficients[0].unwrap(),
                    r.coefficient,
                    r.n_events,
                    range,
                    r.se,
                );
                let se = fit_deviation(fit.standard_errors[0].unwrap(), r.se);
                let robust = match (&fit.robust_standard_errors, r.robust_se) {
                    (Some(got), Some(want)) => fit_deviation(got[0].unwrap(), want),
                    (None, None) => 0.0,
                    other => panic!("{label}: robust standard errors {other:?}"),
                };
                let events = fit_deviation(fit.n_events, r.n_events);
                for (slot, value) in worst.iter_mut().zip([coefficient, se, robust, events]) {
                    *slot = slot.max(value);
                }
                for (what, value) in [
                    ("coefficient", coefficient),
                    ("se", se),
                    ("robust se", robust),
                    ("events", events),
                ] {
                    assert!(
                        value <= 1.0,
                        "{label} {what}: {value} of the bound, {fit:?} against {} {} {:?}",
                        r.coefficient,
                        r.se,
                        r.robust_se
                    );
                }
            }
        }
    }
    eprintln!(
        "worst deviation in bounds: coefficient {:.2e}  se {:.2e}  robust se {:.2e}  events {:.2e}",
        worst[0], worst[1], worst[2], worst[3]
    );
}

/// `crr` and the Rust fit differ in the censoring distribution when a failure
/// ties with a censoring. `crr` estimates it by `survfit(Surv(time, censored))`,
/// which keeps a failure at the censoring time in the risk set, and
/// `finegray` takes the failure out first. Without such a tie they estimate
/// the same thing, Breslow's approximation with the weight `G(t-) / G(x-)`,
/// and the coefficients agree to the tolerance `crr` stops at: its score is
/// below `gtol max(|loglik|, 1) / max(|b|, 1)`, so the coefficient is within
/// that over the information. The standard errors differ by design. `crr`
/// adds the variance from estimating the censoring distribution, which a
/// sandwich on the rows does not have.
#[test]
fn coefficients_match_crr_when_no_failure_ties_with_a_censoring() {
    let reference = reference();
    for data in &reference.datasets {
        let tie = (0..data.time.len()).any(|i| {
            data.event[i] != 0
                && (0..data.time.len()).any(|j| data.event[j] == 0 && data.time[j] == data.time[i])
        });
        for result in &data.results {
            let Some(crr) = &result.crr else { continue };
            for r in crr {
                let column = design(data, &[r.covariate.as_str()]);
                let fit = fit(
                    &surv(data, &data.weight, None),
                    column.as_ref(),
                    &FineGrayOptions {
                        ties: Ties::Breslow,
                        eps: 1e-13,
                        ..FineGrayOptions::new(cause_code(result))
                    },
                )
                .unwrap();
                let beta = fit.coefficients[0].unwrap();
                let tolerance = 2.0 * CRR_GTOL * r.loglik.abs().max(1.0) / r.information;
                let difference = (beta - r.coefficient).abs();
                eprintln!(
                    "{:11} cause {} {:3} failure ties a censoring: {tie:5}  rust {beta:.10}  crr {:.10}  |diff| {difference:.2e}  of tolerance {:.2e}  se rust {:.4} robust - crr {:.4}",
                    data.name,
                    result.cause,
                    r.covariate,
                    r.coefficient,
                    difference / tolerance,
                    fit.standard_errors[0].unwrap(),
                    r.se
                );
                if tie {
                    // The replica of crr with its own censoring distribution
                    // is checked against crr in the script; here the two
                    // estimators are only reported.
                    assert!((r.replica - r.coefficient).abs() <= tolerance);
                } else {
                    assert!(
                        difference <= tolerance,
                        "{} cause {} {}: {beta} against {}",
                        data.name,
                        result.cause,
                        r.covariate,
                        r.coefficient
                    );
                }
            }
        }
    }
}

/// Tolerances for the fits on several covariates.
///
/// Both fits take the same Newton steps and stop at the same iteration. They
/// differ by the error of the solves, `bound = 8 kappa epsilon` as in
/// `r_cox.rs`, and by the weights: the Rust rows are built by the same
/// arithmetic as `finegray`, up to the `f = (2 m + 4 + k) epsilon` relative
/// error derived at the top, with `m` censoring times and `k` rows.
///
/// A relative change `f` in every weight changes the score at a coefficient by
/// at most `2 f W r_k` in column `k`, with `W` the weight of the events and
/// `r_k` the range of the column (a sum of `W` terms, each a difference of a
/// covariate and a weighted mean, with a ratio of two sums each off by `f`).
/// It moves the root by `V` times that, `db_j = sum_k |V_jk| 2 f W r_k`. The
/// information changes by `dI_kl = r_k r_l (2 f W + W sum_m r_m db_m)`, the
/// second term because `dI_kl / d beta_m` is a sum of third central moments,
/// each at most `r_k r_l r_m`. The covariance `V I V` changes by
/// `dV_ab = sum_kl |V_ak| dI_kl |V_lb|`.
struct MultiBounds {
    solve: f64,
    db: Vec<f64>,
    di: Vec<Vec<f64>>,
    dv: Vec<Vec<f64>>,
    f: f64,
}

fn multi_bounds(
    r: &RMulti,
    ranges: &[f64],
    weight: f64,
    censoring_times: usize,
    rows: usize,
) -> MultiBounds {
    let p = r.coefficients.len();
    let f = (2 * censoring_times + 4 + rows) as f64 * f64::EPSILON;
    let db: Vec<f64> = (0..p)
        .map(|j| {
            (0..p)
                .map(|k| r.vcov[j][k].abs() * 2.0 * f * weight * ranges[k])
                .sum()
        })
        .collect();
    let beta_term: f64 = (0..p).map(|m| ranges[m] * db[m]).sum();
    let di: Vec<Vec<f64>> = (0..p)
        .map(|k| {
            (0..p)
                .map(|l| ranges[k] * ranges[l] * (2.0 * f * weight + weight * beta_term))
                .collect()
        })
        .collect();
    let dv = (0..p)
        .map(|a| {
            (0..p)
                .map(|b| {
                    (0..p)
                        .flat_map(|k| (0..p).map(move |l| (k, l)))
                        .map(|(k, l)| r.vcov[a][k].abs() * di[k][l] * r.vcov[l][b].abs())
                        .sum()
                })
                .collect()
        })
        .collect();
    MultiBounds {
        solve: 8.0 * r.kappa * f64::EPSILON,
        db,
        di,
        dv,
        f,
    }
}

/// The largest deviation in units of the tolerance, entry by entry.
fn units(got: f64, want: f64, tolerance: f64) -> f64 {
    (got - want).abs() / tolerance
}

#[test]
fn fits_on_several_covariates_match_coxph_on_finegray_rows() {
    let reference = reference();
    let mut worst: BTreeMap<&str, f64> = BTreeMap::new();
    let mut record = |what: &'static str, value: f64, label: &str| {
        assert!(value <= 1.0, "{label} {what}: {value} of the tolerance");
        let slot = worst.entry(what).or_insert(0.0);
        *slot = slot.max(value);
    };
    for data in &reference.datasets {
        let x = design(data, &MULTI);
        let ranges: Vec<f64> = (0..4)
            .map(|j| {
                let column: Vec<f64> = (0..x.nrows()).map(|i| x[(i, j)]).collect();
                column.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - column.iter().copied().fold(f64::INFINITY, f64::min)
            })
            .collect();
        let largest: Vec<f64> = (0..4)
            .map(|j| (0..x.nrows()).map(|i| x[(i, j)].abs()).fold(0.0, f64::max))
            .collect();
        let m = censoring_times(data);
        for result in &data.results {
            for r in &result.multi {
                let label = format!(
                    "{} cause {} {} {}",
                    data.name, result.cause, r.ties, r.cluster
                );
                let model = fit(
                    &surv(data, &data.weight, cluster_of(data, &r.cluster)),
                    x.as_ref(),
                    &FineGrayOptions {
                        ties: ties_of(&r.ties),
                        ..FineGrayOptions::new(cause_code(result))
                    },
                )
                .unwrap();
                assert!(model.converged && model.rank == 4, "{label}");
                assert_eq!(model.iterations, r.iterations, "{label}");
                let k = result.rows.orig.len();
                let w = r.n_events;
                let b = multi_bounds(r, &ranges, w, m, k);
                let n = k as f64;
                let p = 4.0;
                let eps = f64::EPSILON;
                assert!(units(model.n_events, w, 4.0 * b.f * w) <= 1.0, "{label}");

                let mut coefficient = 0.0f64;
                let mut se = 0.0f64;
                let mut covariance = 0.0f64;
                for a in 0..4 {
                    let beta = model.coefficients[a].unwrap();
                    let tolerance = b.db[a] + b.solve * r.coefficients[a].abs().max(1.0);
                    coefficient = coefficient.max(units(beta, r.coefficients[a], tolerance));
                    let sd = r.vcov[a][a].sqrt();
                    // sqrt is 1/2 relative of the variance.
                    let tolerance = f64::midpoint(b.dv[a][a], p * b.solve * r.vcov[a][a]) / sd;
                    se = se.max(units(model.standard_errors[a].unwrap(), sd, tolerance));
                    for c in 0..4 {
                        let tolerance =
                            b.dv[a][c] + p * b.solve * (r.vcov[a][a] * r.vcov[c][c]).sqrt();
                        covariance = covariance.max(units(
                            model.covariance[a * 4 + c].unwrap(),
                            r.vcov[a][c],
                            tolerance,
                        ));
                    }
                }
                record("coefficient", coefficient, &label);
                record("se", se, &label);
                record("covariance", covariance, &label);

                // The robust covariance V M V: the relative changes of V enter
                // twice and are amplified by at most kappa, the meat has the
                // weights squared, and the solves and sums are those of
                // `r_cox.rs`: `4 p bound + 2 n epsilon`.
                if r.cluster == "none" {
                    assert!(model.robust_covariance.is_none(), "{label}");
                    assert!(!r.vcov_robust.iter().any(|row| !row.is_empty()), "{label}");
                } else {
                    let robust = model.robust_covariance.as_ref().unwrap();
                    let relative_v = (0..4)
                        .flat_map(|a| (0..4).map(move |c| (a, c)))
                        .map(|(a, c)| b.dv[a][c] / (r.vcov[a][a] * r.vcov[c][c]).sqrt())
                        .fold(0.0, f64::max);
                    let relative =
                        4.0 * p * b.solve + 2.0 * n * eps + 2.0 * r.kappa * relative_v + 4.0 * b.f;
                    let mut worst_robust = 0.0f64;
                    for a in 0..4 {
                        for c in 0..4 {
                            let scale = (r.vcov_robust[a][a] * r.vcov_robust[c][c]).sqrt();
                            worst_robust = worst_robust.max(units(
                                robust[a * 4 + c].unwrap(),
                                r.vcov_robust[a][c],
                                relative * scale,
                            ));
                        }
                    }
                    record("robust covariance", worst_robust, &label);
                }

                // Log partial likelihoods: `W f` from the weights in the
                // log of each risk-set sum, `W f max|x| sum |beta|` in the
                // linear predictor, and the rounding of `n` terms.
                let eta: f64 = (0..4).map(|j| largest[j] * r.coefficients[j].abs()).sum();
                let tolerance_loglik =
                    2.0 * b.f * w * (1.0 + eta) + (b.solve + n * eps) * r.loglik.abs().max(1.0);
                let tolerance_null = 2.0 * b.f * w + n * eps * r.loglik_null.abs().max(1.0);
                let loglik = model.log_partial_likelihood.unwrap();
                let null = model.null_log_partial_likelihood.unwrap();
                record("loglik", units(loglik, r.loglik, tolerance_loglik), &label);
                record(
                    "null loglik",
                    units(null, r.loglik_null, tolerance_null),
                    &label,
                );
                let lr = model.tests.likelihood_ratio.unwrap();
                assert_eq!(lr.df, 4, "{label}");
                record(
                    "likelihood ratio",
                    units(
                        lr.statistic,
                        2.0 * (r.loglik - r.loglik_null),
                        2.0 * (tolerance_loglik + tolerance_null),
                    ),
                    &label,
                );

                // The Wald statistic `beta' I beta` changes by
                // `2 |I beta|' db + |beta|' dI |beta|`, and by the solve.
                let info_beta: Vec<f64> = (0..4)
                    .map(|j| {
                        (0..4)
                            .map(|k| r.info[j][k] * r.coefficients[k])
                            .sum::<f64>()
                    })
                    .collect();
                let from_weights: f64 = (0..4)
                    .map(|j| 2.0 * info_beta[j].abs() * b.db[j])
                    .sum::<f64>()
                    + (0..4)
                        .flat_map(|k| (0..4).map(move |l| (k, l)))
                        .map(|(k, l)| {
                            r.coefficients[k].abs() * b.di[k][l] * r.coefficients[l].abs()
                        })
                        .sum::<f64>();
                let wald = model.tests.wald.unwrap();
                assert_eq!(wald.df, 4, "{label}");
                record(
                    "wald",
                    units(
                        wald.statistic,
                        r.wald_model,
                        from_weights + 2.0 * p * b.solve * r.wald_model.max(1.0),
                    ),
                    &label,
                );
                if r.cluster == "none" {
                    assert!(model.tests.robust_wald.is_none(), "{label}");
                } else {
                    // `d' V^-1 d` with the robust `V`: a relative change of `V`
                    // moves it by at most kappa(V) times that change.
                    let kappa = r.kappa_robust.unwrap();
                    let relative_v = (0..4)
                        .flat_map(|a| (0..4).map(move |c| (a, c)))
                        .map(|(a, c)| b.dv[a][c] / (r.vcov[a][a] * r.vcov[c][c]).sqrt())
                        .fold(0.0, f64::max);
                    let relative = kappa
                        * (4.0 * p * b.solve
                            + 2.0 * n * eps
                            + 2.0 * r.kappa * relative_v
                            + 4.0 * b.f)
                        + 8.0 * kappa * eps
                        + 2.0 * relative_v;
                    let robust_wald = model.tests.robust_wald.unwrap();
                    record(
                        "robust wald",
                        units(robust_wald.statistic, r.wald, relative * r.wald.max(1.0)),
                        &label,
                    );
                }
                // The score test is at zero, where only the weights act:
                // `u' I^-1 u` changes by `6 f` relative, and the solve and
                // the sums add `8 kappa_0 epsilon + n epsilon`.
                let score = model.tests.score.unwrap();
                record(
                    "score",
                    units(
                        score.statistic,
                        r.score,
                        (6.0 * b.f + 8.0 * r.kappa_init * eps + n * eps) * r.score.max(1.0),
                    ),
                    &label,
                );
            }
        }
    }
    for (what, value) in &worst {
        eprintln!("several covariates, worst {what}: {value:.2e} of its tolerance");
    }
}

/// The Rust fit and `crr` with several covariates, on the data sets where no
/// failure ties with a censoring, which is the case where they estimate the
/// same censoring distribution. `crr` stops when its score is below `gtol
/// max(|loglik|, 1) / max(|b|, 1)`, so each coefficient is within that over
/// the smallest eigenvalue of the information, bounded by the trace of the
/// inverse: `|b - b*| <= ||I^-1|| ||score||`, with the score below
/// `gtol max(|loglik|, 1)` in the norm. The bound takes `||I^-1||` as the
/// largest row sum of `|I^-1|`, with `I^-1` from the inverse of the matrix
/// `crr` reports.
#[test]
fn several_coefficients_match_crr_when_no_failure_ties_with_a_censoring() {
    let reference = reference();
    let mut checked = 0;
    for data in &reference.datasets {
        let tie = (0..data.time.len()).any(|i| {
            data.event[i] != 0
                && (0..data.time.len()).any(|j| data.event[j] == 0 && data.time[j] == data.time[i])
        });
        if tie {
            continue;
        }
        let x = design(data, &MULTI);
        for result in &data.results {
            let Some(crr) = &result.crr_multi else {
                continue;
            };
            let model = fit(
                &surv(data, &data.weight, None),
                x.as_ref(),
                &FineGrayOptions {
                    ties: Ties::Breslow,
                    eps: 1e-13,
                    ..FineGrayOptions::new(cause_code(result))
                },
            )
            .unwrap();
            // The inverse of the 4 by 4 information, by Gauss-Jordan.
            let inverse = invert(&crr.information);
            let norm = inverse
                .iter()
                .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
                .fold(0.0, f64::max);
            let tolerance = 2.0 * CRR_GTOL * crr.loglik.abs().max(1.0) * norm;
            for (j, &want) in crr.coefficients.iter().enumerate() {
                let got = model.coefficients[j].unwrap();
                eprintln!(
                    "{} cause {} column {j}: rust {got:.10} crr {want:.10} |diff| {:.2e} of tolerance",
                    data.name,
                    result.cause,
                    (got - want).abs() / tolerance
                );
                assert!((got - want).abs() <= tolerance, "{} {j}", data.name);
            }
            checked += 1;
        }
    }
    assert!(checked > 0);
}

fn invert(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.extend((0..n).map(|j| f64::from(u8::from(i == j))));
            row
        })
        .collect();
    for c in 0..n {
        let pivot = (c..n)
            .max_by(|&i, &j| m[i][c].abs().total_cmp(&m[j][c].abs()))
            .unwrap();
        m.swap(c, pivot);
        let d = m[c][c];
        for v in &mut m[c] {
            *v /= d;
        }
        for i in 0..n {
            if i != c {
                let factor = m[i][c];
                let source = m[c].clone();
                for (v, s) in m[i].iter_mut().zip(&source) {
                    *v -= factor * s;
                }
            }
        }
    }
    m.into_iter().map(|row| row[n..].to_vec()).collect()
}
