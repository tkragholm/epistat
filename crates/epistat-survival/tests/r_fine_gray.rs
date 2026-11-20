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
//! The fit bounds are derived at `coefficient_deviation` and
//! `se_deviation`. The other fit outputs are bounded by `8 epsilon`
//! relative to the value or to 1 for a value below 1, as in `r_cox.rs`, where
//! the factor is the condition number of the information, which is 1 for one
//! covariate. The reference coefficient is polished in R by two Newton steps,
//! because `coxph` stops a step early.

#![expect(
    clippy::float_cmp,
    reason = "starts, stops and times are copied from the data and compared exactly"
)]

use std::collections::BTreeMap;

use epistat_survival::cox::Ties;
use epistat_survival::fine_gray::{FineGrayData, Rows, fine_gray_fit, float_codes};
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
            let fg =
                FineGrayData::new(&data.time, &data.event, &data.weight, result.cause).unwrap();
            let report = compare_rows(&fg.expand(), &result.rows, &ones, &events, bound, &label);
            eprintln!(
                "{label:22} R rows {:4}  merged {:3}  rows R omits {:3}  worst weight {:.2e} of its bound",
                report.r_rows, report.merged, report.uncovered, report.worst
            );
            if let Some(argument) = &result.rows_weights_argument {
                // finegray(weights =) leaves the case weights out of the
                // censoring distribution, so its rows are the unweighted rows
                // times the case weight.
                let unit = FineGrayData::new(&data.time, &data.event, &ones, result.cause).unwrap();
                let report = compare_rows(
                    &unit.expand(),
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
            let fg =
                FineGrayData::new(&data.time, &data.event, &data.weight, result.cause).unwrap();
            for name in ["bin", "x"] {
                let levels = float_codes(covariate(data, name));
                let ours = fg.risk_rows(&levels).unwrap();
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

/// The Newton iteration stops when a step is below this.
const STEP_TOLERANCE: f64 = 1e-10;

/// Deviation of a model-based standard error in units of its bound.
///
/// The fit reports `1 / sqrt(I)` with `I` evaluated at the iterate before its
/// last step, `R` at the estimate. `d I / d beta` is a sum of third central
/// moments of the covariate over the risk sets, each at most the range of the
/// covariate times the variance, so `|d ln I / d beta| <= range`. The last step
/// is below [`STEP_TOLERANCE`], so the standard error differs from the one at
/// the estimate by at most `range / 2 * STEP_TOLERANCE` relative to itself, on
/// top of the `8 epsilon` of the arithmetic.
fn se_deviation(got: f64, want: f64, range: f64) -> f64 {
    let bound =
        8.0 * f64::EPSILON * want.abs().max(1.0) + 0.5 * range * STEP_TOLERANCE * want.abs();
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
                let cluster = match r.cluster.as_str() {
                    "none" => None,
                    "id" => Some(data.id.as_slice()),
                    "group" => Some(data.group.as_slice()),
                    other => panic!("unknown cluster {other}"),
                };
                let ties = if r.ties == "efron" {
                    Ties::Efron
                } else {
                    Ties::Breslow
                };
                let fit = fine_gray_fit(
                    &data.time,
                    &data.event,
                    &data.weight,
                    x,
                    result.cause,
                    ties,
                    cluster,
                )
                .unwrap();
                let label = format!(
                    "{} cause {} {} {} {}",
                    data.name, result.cause, r.covariate, r.ties, r.cluster
                );
                assert!(fit.converged, "{label}");

                let range = x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - x.iter().copied().fold(f64::INFINITY, f64::min);
                let coefficient = coefficient_deviation(
                    fit.log_hazard_ratio.unwrap(),
                    r.coefficient,
                    r.n_events,
                    range,
                    r.se,
                );
                let se = se_deviation(fit.standard_error.unwrap(), r.se, range);
                let robust = match (fit.robust_standard_error, r.robust_se) {
                    (Some(got), Some(want)) => fit_deviation(got, want),
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
                let x = covariate(data, &r.covariate);
                let fit = fine_gray_fit(
                    &data.time,
                    &data.event,
                    &data.weight,
                    x,
                    result.cause,
                    Ties::Breslow,
                    None,
                )
                .unwrap();
                let beta = fit.log_hazard_ratio.unwrap();
                let tolerance = 2.0 * CRR_GTOL * r.loglik.abs().max(1.0) / r.information;
                let difference = (beta - r.coefficient).abs();
                eprintln!(
                    "{:11} cause {} {:3} failure ties a censoring: {tie:5}  rust {beta:.10}  crr {:.10}  |diff| {difference:.2e}  of tolerance {:.2e}  se rust {:.4} robust - crr {:.4}",
                    data.name,
                    result.cause,
                    r.covariate,
                    r.coefficient,
                    difference / tolerance,
                    fit.standard_error.unwrap(),
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
