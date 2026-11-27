//! The self-controlled case series: the conditional Poisson likelihood of the
//! events of each case given the number of events, the pieces of observation
//! time it is evaluated on, and the splitting of each case's follow-up into
//! those pieces.
//!
//! # The model
//!
//! Each person contributes the events of their own observation period, which is
//! cut into pieces with a constant age band and a constant exposure state.
//! Person `i` has pieces `k` of length `e_ik`, covariates `x_ik` (indicators
//! of the risk windows and of the age bands) and `y_ik` events. Events arise
//! as a Poisson process whose rate is `phi_i exp(x_ik' beta)` in piece `k`,
//! with a person-specific level `phi_i` that is left free. Conditional on the
//! total `n_i = sum_k y_ik` the events of the person are multinomial over the
//! pieces, and `phi_i` drops out \[1, 2\]:
//!
//! ```text
//! eta_ik = x_ik' beta + ln e_ik
//! p_ik   = exp(eta_ik) / sum_l exp(eta_il)
//! ln L   = sum_i [ sum_k y_ik eta_ik - n_i ln sum_l exp(eta_il) ]
//! score  = sum_i [ sum_k y_ik x_ik - n_i sum_k p_ik x_ik ]
//! info   = sum_i n_i [ sum_k p_ik (x_ik - xbar_i)(x_ik - xbar_i)' ],   xbar_i = sum_k p_ik x_ik
//! ```
//!
//! This is the profile likelihood of the Poisson regression with a free
//! intercept for each person, so the slopes equal those of `glm(y ~
//! factor(person) + x + offset(log(e)), family = poisson)`, and with one event
//! per person it is the likelihood of `survival::clogit` with the piece that
//! holds the event as the case and `offset(log(e))`. With several events per
//! person it is the product of the multinomial probabilities of the events,
//! which is the likelihood of the stacked format of the `SCCS` package: one
//! stratum per event, each over the whole observation period. `info` is
//! the exact negative Hessian.
//!
//! # Estimation
//!
//! Newton-Raphson from `beta = 0` with the information as the Hessian and step
//! halving when the log likelihood falls. The sums over persons are formed in
//! blocks of 512 persons, each person's terms in piece order and the blocks
//! added in order, so the bits do not depend on the number of threads. The
//! log-sum-exp subtracts the largest `eta_il`. The information is factorised
//! as `coxph` does it ([`crate::cox`]): a column whose pivot at `beta = 0` is
//! below `alias_tolerance` times the largest diagonal element is aliased, as a
//! column that is constant within every person is, and its coefficient is
//! `None`.
//!
//! The iteration stops when the log likelihood changes by at most `tolerance`
//! relative to `max(|ln L|, 1)`. A last step that is not small means the
//! maximum does not exist (a risk window with no events, all the events of a
//! person set in one window): the Newton step on the scaled column, `|(I^-1
//! U)_j| s_j` with `s_j` the within-person standard deviation of the column,
//! then exceeds `max(sqrt(tolerance), 1e-9)`, as `coxph` warns of a coefficient
//! that may be infinite. The column is listed in `monotone_columns`, every
//! estimate is `None` and `converged` is false. The coefficients are not
//! repaired.
//!
//! The covariance is `I^-1`. With `robust` the person-level scores `s_i` give
//! the sandwich `I^-1 (sum_i s_i s_i') I^-1`, valid when the events of a
//! person are not independent given the person's rate. The standard errors
//! are the square roots of the diagonals, and
//! [`SccsFit::relative_incidence`] gives `exp(b)` with the Wald interval
//! `exp(b -+ z se)`, `z` the normal quantile of the level, and the two-sided
//! Wald p-value. `standardsccs` prints its interval with `z = 1.96`, which
//! differs from the quantile `1.959964` in the fourth decimal of the limits.
//!
//! # The pieces
//!
//! [`build_pieces`] cuts the observation period `[start_i, end_i)` of each
//! person at the age cut points, at the boundaries of every risk window of
//! every exposure date of every exposure series, and at the calendar cuts and
//! season starts, and counts the events in each piece. Time is continuous and
//! intervals are half open: an event at a boundary belongs to the piece that
//! starts there, and for dates in days `end` is the last observed day plus
//! one. All times, the observation period, the exposure dates, the events and
//! the age cut points, are on one scale, usually age in days. A
//! [`WindowSpec`] covers `[date + from, date + to)` for each exposure date of
//! the person, and `to` may be infinite. Windows of one series may not
//! overlap. Where windows of different dates of one series overlap, the piece
//! takes the window of the latest date, so that a pre-exposure window of a
//! later dose takes precedence over the late window of an earlier dose. Each
//! [`ExposureSeries`] labels the pieces on its own, so the windows of two
//! drugs may overlap and the design then has both indicators set, the
//! multiplicative model of `standardsccs` with several drugs.
//!
//! A [`Calendar`] places each person's time on the calendar: time `t` is the
//! day `origin_i + t` from 1970-01-01, so with time as age in days `origin_i`
//! is the date of birth. Calendar cuts are dates, and each falls at the time
//! `cut - origin_i` of the person. A season starts on the same day of the
//! calendar every year, and the days before the first start of the year belong
//! to the last season, as in `standardsccs` with `seasongrp` and `dob`.
//! Neighbouring pieces with the same labels are merged, which does not change
//! the likelihood. The pieces of a person are contiguous and in time order,
//! and people appear in the order of the input.
//!
//! [`Pieces::window_summary`] and [`Pieces::exposed_cases`] count, for each
//! window, the cases observed in it, those that inform its coefficient (an
//! event, and time both in and out of the window), the events and the time in
//! it, which a report gives beside the estimate.
//!
//! # What the likelihood assumes
//!
//! The likelihood is valid when (a) the events are independent given the
//! person's rate, so that recurrent events are rare or only the first event is
//! used, (b) the observation period does not depend on the events, which is
//! violated when death or emigration follows the event, and (c) exposure after
//! an event does not depend on the event, which is violated when the exposure
//! treats the early symptoms of the event. A pre-exposure window in the design
//! shows (c). [`fit_event_dependent`] fits the extension of Farrington et al.
//! \[3\] for exposures that the event censors or stops, as
//! `SCCS::eventdepenexp`. The extension for event-dependent observation
//! periods \[4\] (`SCCS::eventdepenobs`) and the smooth age effect of \[5\]
//! (`SCCS::smoothagesccs`, and with a smooth exposure effect too,
//! `nonparasccs`) are not provided. [`semiparametric_pieces`] gives the
//! semiparametric case series, with an age effect free at every event time,
//! as `SCCS::semisccs`, and [`sample_size`] the number of cases a design
//! needs, as `SCCS::samplesize`.
//!
//! # Scale
//!
//! One pass is `O(P p^2)` for `P` pieces and `p` columns, plus the design,
//! which is held in memory: `P p` doubles, 12 GB for `10^8` pieces and 15
//! columns. A study of that size cuts the cases into parts of persons and sums
//! the person terms of the parts with its own code.
//!
//! # Agreement with R
//!
//! The tests compare coefficients, both covariances and the log likelihood with
//! `SCCS::standardsccs` (SCCS 1.8) for one event per person and for several,
//! with two drugs and with seasons, with `glm` Poisson with person dummies,
//! with `survival::clogit`, and with a hand-written R loop for the pieces,
//! with R's `Date` for calendar periods and seasons. `tests/data/sccs.R` and
//! `tests/data/sccs_calendar.R` record the versions.
//!
//! # References
//!
//! 1. Farrington CP. Relative incidence estimation from case series for
//!    vaccine safety evaluation. Biometrics 1995;51(1):228-35.
//! 2. Whitaker HJ, Farrington CP, Spiessens B, Musonda P. Tutorial in
//!    biostatistics: the self-controlled case series method. Stat Med
//!    2006;25(10):1768-97.
//! 3. Farrington CP, Whitaker HJ, Hocine MN. Case series analysis for censored,
//!    perturbed, or curtailed post-event exposures. Biostatistics
//!    2009;10(1):3-16.
//! 4. Farrington CP, Anaya-Izquierdo K, Whitaker HJ, Hocine MN, Douglas I,
//!    Smeeth L. Self-controlled case series analysis with event-dependent
//!    observation periods. J Am Stat Assoc 2011;106(494):417-26.
//! 5. Ghebremichael-Weldeselassie Y, Whitaker HJ, Farrington CP. Self-controlled
//!    case series method with smooth age effect. Stat Med 2014;33(4):639-49.

mod event_dependent;
mod sample_size;
mod semiparametric;

pub use event_dependent::{
    EventDependentFit, EventDependentOptions, EventDependentSpec, fit_event_dependent,
};
pub use sample_size::{SampleSizeSpec, sample_size};
pub use semiparametric::{SemiparametricPieces, semiparametric_pieces};

use epistat_core::Level;
use epistat_linalg::{Mat, MatRef, congruence};

use crate::error::{Error, Result};
use crate::ldl;
use crate::par::ordered_reduce;

/// Persons per block of the sums.
const PERSON_BLOCK: usize = 512;

/// The pieces of observation time on which the likelihood is evaluated.
///
/// The three columns have one entry per piece. The pieces of a person are
/// contiguous, and the pieces of two people are never interleaved.
#[derive(Debug, Clone, Copy)]
pub struct SccsData<'a> {
    /// The person of each piece.
    pub person: &'a [u32],
    /// The length of each piece, positive and finite.
    pub length: &'a [f64],
    /// The number of events in each piece, not negative and finite.
    pub events: &'a [f64],
}

/// What [`fit`] takes besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SccsOptions {
    /// The most Newton steps.
    pub max_iterations: usize,
    /// The relative change of the log likelihood at convergence.
    pub tolerance: f64,
    /// A column is aliased when its pivot at `beta = 0` is below this times
    /// the largest diagonal element of the information, as `toler.chol` of
    /// `coxph.control`.
    pub alias_tolerance: f64,
    /// Whether to form the person-level sandwich covariance.
    pub robust: bool,
}

impl Default for SccsOptions {
    fn default() -> Self {
        Self {
            max_iterations: 25,
            tolerance: 1e-10,
            alias_tolerance: f64::EPSILON.powf(0.75),
            robust: false,
        }
    }
}

/// A fitted self-controlled case series model.
#[derive(Debug, Clone)]
pub struct SccsFit {
    /// The log relative incidences, one per column, `None` for an aliased
    /// column and for every column when the fit has no estimate.
    pub coefficients: Vec<Option<f64>>,
    /// The model-based covariance, `p * p` row-major, `None` where its row or
    /// column is a coefficient that is `None`.
    pub covariance: Vec<Option<f64>>,
    /// The person-level sandwich covariance, laid out as `covariance`, when
    /// [`SccsOptions::robust`] was set and there is an estimate.
    pub robust_covariance: Option<Vec<Option<f64>>>,
    /// The standard errors of the coefficients, the square roots of the
    /// diagonal of `covariance`.
    pub standard_errors: Vec<Option<f64>>,
    /// The standard errors from `robust_covariance`, when it is formed.
    pub robust_standard_errors: Option<Vec<Option<f64>>>,
    /// The log likelihood at the estimate, `None` without an estimate.
    pub log_likelihood: Option<f64>,
    /// The log likelihood at `beta = 0`, `None` when no person has an event in
    /// a person with more than one piece.
    pub null_log_likelihood: Option<f64>,
    /// The people in the data.
    pub n_persons: usize,
    /// The people who add to the likelihood: events in at least two pieces.
    pub n_persons_used: usize,
    /// The people dropped because their events sum to 0.
    pub n_persons_without_events: usize,
    /// The people dropped because all their observation is one piece.
    pub n_persons_one_piece: usize,
    /// The events of the people used.
    pub n_events: f64,
    /// For each column, the people used whose value of the column is not the
    /// same in all their pieces. For the indicator of a risk window these are
    /// the cases with an event who were observed both in and out of the
    /// window, the exposed cases that inform its coefficient.
    pub n_informative_by_column: Vec<usize>,
    /// The columns that are not aliased.
    pub rank: usize,
    /// The Newton steps taken.
    pub iterations: usize,
    /// Whether the log likelihood stopped changing within the step limit with
    /// no diverging coefficient.
    pub converged: bool,
    /// The columns whose coefficient diverges. Empty when none do.
    pub monotone_columns: Vec<usize>,
}

/// Sums of one block of persons at one coefficient vector.
struct Sums {
    ll: f64,
    score: Vec<f64>,
    info: Vec<f64>,
    meat: Vec<f64>,
}

impl Sums {
    fn zero(p: usize, meat: bool) -> Self {
        Self {
            ll: 0.0,
            score: vec![0.0; p],
            info: vec![0.0; p * p],
            meat: vec![0.0; if meat { p * p } else { 0 }],
        }
    }

    fn add(&mut self, other: Self) {
        self.ll += other.ll;
        for (a, b) in self.score.iter_mut().zip(other.score) {
            *a += b;
        }
        for (a, b) in self.info.iter_mut().zip(other.info) {
            *a += b;
        }
        for (a, b) in self.meat.iter_mut().zip(other.meat) {
            *a += b;
        }
    }
}

/// The people of the data as ranges of pieces.
struct Persons {
    ranges: Vec<(usize, usize)>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid(message)
}

fn persons_of(data: &SccsData<'_>) -> Result<Persons> {
    let n = data.person.len();
    let mut ranges = Vec::new();
    let mut start = 0;
    for i in 1..=n {
        if i == n || data.person[i] != data.person[start] {
            ranges.push((start, i));
            start = i;
        }
    }
    let mut ids: Vec<u32> = ranges.iter().map(|&(a, _)| data.person[a]).collect();
    ids.sort_unstable();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return Err(invalid(
            "the pieces of a person are not contiguous: a person code reappears after another",
        ));
    }
    Ok(Persons { ranges })
}

/// The log likelihood, score and information of the persons `range` at `beta`
/// on the columns `cols`.
fn sums_at(
    data: &SccsData<'_>,
    x: MatRef<'_, f64>,
    cols: &[usize],
    persons: &[(usize, usize)],
    beta: &[f64],
    meat: bool,
) -> Sums {
    let p = cols.len();
    let mut sums = Sums::zero(p, meat);
    let mut eta: Vec<f64> = Vec::new();
    let mut xbar = vec![0.0; p];
    let mut centred = vec![0.0; p];
    let mut person_score = vec![0.0; p];
    for &(a, b) in persons {
        let n_i: f64 = data.events[a..b].iter().sum();
        if n_i == 0.0 || b - a < 2 {
            continue;
        }
        eta.clear();
        let mut top = f64::NEG_INFINITY;
        for r in a..b {
            let mut e = data.length[r].ln();
            for (j, &c) in cols.iter().enumerate() {
                e += x[(r, c)] * beta[j];
            }
            top = top.max(e);
            eta.push(e);
        }
        let mut total = 0.0;
        for e in &mut eta {
            *e = (*e - top).exp();
            total += *e;
        }
        let log_total = top + total.ln();
        let mut y_eta = 0.0;
        for r in a..b {
            if data.events[r] != 0.0 {
                y_eta += data.events[r] * (eta[r - a].ln() + top);
            }
        }
        sums.ll += y_eta - n_i * log_total;
        xbar.fill(0.0);
        for r in a..b {
            let w = eta[r - a] / total;
            for (j, &c) in cols.iter().enumerate() {
                xbar[j] += w * x[(r, c)];
            }
        }
        person_score.fill(0.0);
        for r in a..b {
            let w = eta[r - a] / total;
            for (j, &c) in cols.iter().enumerate() {
                centred[j] = x[(r, c)] - xbar[j];
            }
            for j in 0..p {
                person_score[j] += data.events[r] * centred[j];
                for l in 0..=j {
                    sums.info[j * p + l] += n_i * w * centred[j] * centred[l];
                }
            }
        }
        for j in 0..p {
            sums.score[j] += person_score[j];
            if meat {
                for l in 0..=j {
                    sums.meat[j * p + l] += person_score[j] * person_score[l];
                }
            }
        }
    }
    for j in 0..p {
        for l in 0..j {
            sums.info[l * p + j] = sums.info[j * p + l];
            if meat {
                sums.meat[l * p + j] = sums.meat[j * p + l];
            }
        }
    }
    sums
}

fn blocked(
    data: &SccsData<'_>,
    x: MatRef<'_, f64>,
    cols: &[usize],
    persons: &Persons,
    beta: &[f64],
    meat: bool,
) -> Sums {
    ordered_reduce(
        persons.ranges.len(),
        PERSON_BLOCK,
        64,
        |r| sums_at(data, x, cols, &persons.ranges[r], beta, meat),
        Sums::add,
    )
}

fn check_data(data: &SccsData<'_>, x: MatRef<'_, f64>) -> Result<()> {
    let n = data.person.len();
    if data.length.len() != n || data.events.len() != n || x.nrows() != n {
        return Err(invalid(format!(
            "person, length, events and the rows of x must have one entry per piece, found {}, {}, {} and {}",
            n,
            data.length.len(),
            data.events.len(),
            x.nrows()
        )));
    }
    if x.ncols() == 0 {
        return Err(invalid("x has no column"));
    }
    if let Some(i) = data
        .length
        .iter()
        .position(|&l| !(l.is_finite() && l > 0.0))
    {
        return Err(invalid(format!(
            "length of piece {i} is {}, which is not positive and finite",
            data.length[i]
        )));
    }
    if let Some(i) = data
        .events
        .iter()
        .position(|&y| !(y.is_finite() && y >= 0.0))
    {
        return Err(invalid(format!(
            "events of piece {i} is {}, which is negative or not finite",
            data.events[i]
        )));
    }
    for j in 0..x.ncols() {
        for i in 0..n {
            if !x[(i, j)].is_finite() {
                return Err(invalid(format!("x[{i}, {j}] is not finite")));
            }
        }
    }
    Ok(())
}

/// Fits the self-controlled case series model of the events in `data` on the
/// columns of `x`, one row per piece.
///
/// The columns are the risk windows and the age bands, one reference level
/// each left out, plus any piece-level covariate. See the module documentation
/// for the likelihood, the stopping rule, aliasing and what a diverging
/// coefficient gives.
///
/// # Errors
///
/// An input error when the columns differ in length, `x` has no column or a
/// value that is not finite, a length is not positive, an event count is
/// negative or not finite, the pieces of a person are not contiguous, the
/// tolerance is not positive and finite, or the information is not positive
/// semi-definite.
pub fn fit(data: &SccsData<'_>, x: MatRef<'_, f64>, opts: &SccsOptions) -> Result<SccsFit> {
    check_data(data, x)?;
    if !opts.tolerance.is_finite()
        || opts.tolerance <= 0.0
        || !opts.alias_tolerance.is_finite()
        || opts.alias_tolerance <= 0.0
        || opts.max_iterations == 0
    {
        return Err(invalid(
            "tolerance and alias_tolerance must be positive and finite, and max_iterations at least 1",
        ));
    }
    let persons = persons_of(data)?;
    let n_cols = x.ncols();
    let mut n_without = 0;
    let mut n_one = 0;
    let mut n_used = 0;
    let mut n_events = 0.0;
    for &(a, b) in &persons.ranges {
        let n_i: f64 = data.events[a..b].iter().sum();
        if n_i == 0.0 {
            n_without += 1;
        } else if b - a < 2 {
            n_one += 1;
        } else {
            n_used += 1;
            n_events += n_i;
        }
    }
    let mut result = SccsFit {
        coefficients: vec![None; n_cols],
        covariance: vec![None; n_cols * n_cols],
        robust_covariance: None,
        standard_errors: vec![None; n_cols],
        robust_standard_errors: None,
        log_likelihood: None,
        null_log_likelihood: None,
        n_persons: persons.ranges.len(),
        n_persons_used: n_used,
        n_persons_without_events: n_without,
        n_persons_one_piece: n_one,
        n_events,
        n_informative_by_column: informative_by_column(data, x, &persons),
        rank: 0,
        iterations: 0,
        converged: false,
        monotone_columns: Vec::new(),
    };
    if n_used == 0 {
        return Ok(result);
    }
    let all: Vec<usize> = (0..n_cols).collect();
    let zero = vec![0.0; n_cols];
    let start = blocked(data, x, &all, &persons, &zero, false);
    result.null_log_likelihood = Some(start.ll);
    let mut info0 = start.info.clone();
    let pivots = ldl::factorise(&mut info0, n_cols, opts.alias_tolerance);
    if !pivots.clean {
        return Err(invalid("the information is not positive semi-definite"));
    }
    let cols: Vec<usize> = all
        .iter()
        .copied()
        .filter(|&j| info0[j * n_cols + j] != 0.0)
        .collect();
    let p = cols.len();
    result.rank = p;
    if p == 0 {
        return Ok(result);
    }
    // The scale of each kept column: its within-person standard deviation.
    let scale: Vec<f64> = cols
        .iter()
        .map(|&j| (start.info[j * n_cols + j] / n_events).sqrt())
        .collect();
    let mut beta = vec![0.0; p];
    let mut current = blocked(data, x, &cols, &persons, &beta, false);
    let mut converged = false;
    let mut iterations = 0;
    for _ in 0..opts.max_iterations {
        let step = newton_step(&current, p, opts.alias_tolerance)?;
        iterations += 1;
        let mut factor = 1.0;
        let mut trial = None;
        for _ in 0..30 {
            let candidate: Vec<f64> = beta
                .iter()
                .zip(&step)
                .map(|(b, s)| b + factor * s)
                .collect();
            let sums = blocked(data, x, &cols, &persons, &candidate, false);
            if sums.ll.is_finite() && sums.ll >= current.ll - 1e-14 * current.ll.abs() {
                trial = Some((candidate, sums));
                break;
            }
            factor *= 0.5;
        }
        let Some((candidate, sums)) = trial else {
            break;
        };
        let change = (sums.ll - current.ll).abs();
        let bound = opts.tolerance * sums.ll.abs().max(1.0);
        beta = candidate;
        current = sums;
        if change <= bound {
            converged = true;
            break;
        }
    }
    result.iterations = iterations;
    // The Newton step at the estimate shows a coefficient that does not settle.
    let mut factored = current.info.clone();
    let pivots = ldl::factorise(&mut factored, p, opts.alias_tolerance);
    if !pivots.clean || pivots.rank != p {
        return Err(invalid("the information is not positive definite"));
    }
    let mut last = current.score.clone();
    ldl::solve(&factored, p, &mut last);
    let limit = opts.tolerance.sqrt().max(1e-9);
    let monotone: Vec<usize> = (0..p)
        .filter(|&j| (last[j] * scale[j]).abs() > limit)
        .map(|j| cols[j])
        .collect();
    if converged && !monotone.is_empty() {
        result.monotone_columns = monotone;
        result.converged = false;
        return Ok(result);
    }
    result.converged = converged;
    result.log_likelihood = Some(current.ll);
    let inverse = ldl::inverse(&factored, p);
    for (a, &ca) in cols.iter().enumerate() {
        result.coefficients[ca] = Some(beta[a]);
        for (b, &cb) in cols.iter().enumerate() {
            result.covariance[ca * n_cols + cb] = Some(inverse[a * p + b]);
        }
    }
    result.standard_errors = diagonal_roots(&result.covariance, n_cols);
    if opts.robust {
        let meat = blocked(data, x, &cols, &persons, &beta, true).meat;
        let sandwich = sandwich(&inverse, &meat, p)?;
        let mut full = vec![None; n_cols * n_cols];
        for (a, &ca) in cols.iter().enumerate() {
            for (b, &cb) in cols.iter().enumerate() {
                full[ca * n_cols + cb] = Some(sandwich[a * p + b]);
            }
        }
        result.robust_standard_errors = Some(diagonal_roots(&full, n_cols));
        result.robust_covariance = Some(full);
    }
    Ok(result)
}

/// The square roots of the diagonal of a `p * p` covariance.
fn diagonal_roots(covariance: &[Option<f64>], p: usize) -> Vec<Option<f64>> {
    (0..p)
        .map(|j| covariance[j * p + j].map(f64::sqrt))
        .collect()
}

/// For each column, the people with events in at least two pieces whose
/// value of the column varies between their pieces.
#[expect(
    clippy::float_cmp,
    reason = "a column varies within a person or it does not"
)]
fn informative_by_column(data: &SccsData<'_>, x: MatRef<'_, f64>, persons: &Persons) -> Vec<usize> {
    let p = x.ncols();
    ordered_reduce(
        persons.ranges.len(),
        PERSON_BLOCK,
        64,
        |r| {
            let mut count = vec![0_usize; p];
            for &(a, b) in &persons.ranges[r] {
                let n_i: f64 = data.events[a..b].iter().sum();
                if n_i == 0.0 || b - a < 2 {
                    continue;
                }
                for (j, c) in count.iter_mut().enumerate() {
                    let first = x[(a, j)];
                    *c += usize::from((a + 1..b).any(|k| x[(k, j)] != first));
                }
            }
            count
        },
        |total: &mut Vec<usize>, next| {
            for (t, n) in total.iter_mut().zip(next) {
                *t += n;
            }
        },
    )
}

/// The relative incidence of one column with its interval and test, from
/// [`SccsFit::relative_incidence`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeIncidence {
    /// The coefficient, the log relative incidence.
    pub coefficient: f64,
    /// Its standard error, model-based or robust as asked.
    pub standard_error: f64,
    /// `exp(coefficient)`.
    pub relative_incidence: f64,
    /// The lower limit, `exp(coefficient - z standard_error)`.
    pub lower: f64,
    /// The upper limit, `exp(coefficient + z standard_error)`.
    pub upper: f64,
    /// The Wald statistic, `coefficient / standard_error`.
    pub z: f64,
    /// The two-sided p-value of the Wald statistic.
    pub p_value: f64,
}

impl SccsFit {
    /// The relative incidence of each column with its Wald interval at
    /// `level`, `exp(b +- z se)` with `z` the `(1 + level) / 2` normal
    /// quantile, and the two-sided Wald test of no effect. `robust` takes the
    /// standard errors from the sandwich. A column whose coefficient is `None`
    /// gives `None`.
    ///
    /// # Errors
    ///
    /// An input error when `robust` is asked for a fit without the sandwich.
    pub fn relative_incidence(
        &self,
        level: Level,
        robust: bool,
    ) -> Result<Vec<Option<RelativeIncidence>>> {
        let se = if robust {
            self.robust_standard_errors.as_ref().ok_or_else(|| {
                invalid("the fit has no robust covariance: set SccsOptions::robust")
            })?
        } else {
            &self.standard_errors
        };
        let z = epistat_stats::dist::normal::quantile(level.upper_probability());
        Ok(self
            .coefficients
            .iter()
            .zip(se)
            .map(|(b, s)| {
                let (b, s) = (b.as_ref()?, s.as_ref()?);
                let (b, s) = (*b, *s);
                let wald = b / s;
                Some(RelativeIncidence {
                    coefficient: b,
                    standard_error: s,
                    relative_incidence: b.exp(),
                    lower: (b - z * s).exp(),
                    upper: (b + z * s).exp(),
                    z: wald,
                    p_value: 2.0 * epistat_stats::dist::normal::cdf(-wald.abs()),
                })
            })
            .collect())
    }
}

fn newton_step(sums: &Sums, p: usize, tolerance: f64) -> Result<Vec<f64>> {
    let mut factored = sums.info.clone();
    let pivots = ldl::factorise(&mut factored, p, tolerance);
    if !pivots.clean || pivots.rank != p {
        return Err(invalid("the information is not positive definite"));
    }
    let mut step = sums.score.clone();
    ldl::solve(&factored, p, &mut step);
    Ok(step)
}

/// The sandwich `H M H'` of the inverse information `H` and the meat `M`,
/// both row-major `p x p`, as linalg's `congruence` forms it.
fn sandwich(inverse: &[f64], meat: &[f64], p: usize) -> Result<Vec<f64>> {
    let h = Mat::from_fn(p, p, |i, j| inverse[i * p + j]);
    let m = Mat::from_fn(p, p, |i, j| meat[i * p + j]);
    let out = congruence(h.as_ref(), m.as_ref())?;
    Ok((0..p * p).map(|k| out[(k / p, k % p)]).collect())
}

/// A risk window: the times from `from` to `to` after an exposure date, as
/// `[date + from, date + to)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSpec {
    /// The start relative to the exposure date, negative for a pre-exposure
    /// window.
    pub from: f64,
    /// The end relative to the exposure date, greater than `from`; infinite
    /// for a window that lasts to the end of observation.
    pub to: f64,
}

/// One exposure series: the exposure dates of each person and the risk
/// windows that every date of the series opens.
///
/// The dates are ragged: person `i` has `dates[offsets[i]..offsets[i + 1]]`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExposureSeries<'a> {
    /// The offsets of each person's dates, one more entry than people.
    pub offsets: &'a [usize],
    /// The exposure dates. A date outside the observation period is allowed,
    /// since its windows may reach into it.
    pub dates: &'a [f64],
    /// The risk windows of the series, which do not overlap one another.
    pub windows: &'a [WindowSpec],
}

/// The first day of a season, as a day of the calendar year.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeasonStart {
    /// The month, 1 to 12.
    pub month: u32,
    /// The day of the month. 29 February is refused, since it is missing in
    /// most years.
    pub day: u32,
}

/// Cuts on the calendar: calendar periods and seasons.
///
/// The time of the pieces is in days, and `origin` places each person's time
/// scale on the calendar: time `t` of person `i` is the day `origin[i] + t`,
/// counted from 1970-01-01. With time as age in days, `origin` is the date of
/// birth; with time as calendar days, it is 0.
#[derive(Debug, Clone, Copy, Default)]
pub struct Calendar<'a> {
    /// The day, from 1970-01-01, at time 0 of each person.
    pub origin: &'a [f64],
    /// Strictly increasing calendar dates, as days from 1970-01-01. Calendar
    /// period `c` is `[cut_(c-1), cut_c)`, with period 0 before the first cut.
    pub cuts: &'a [f64],
    /// The first days of the seasons, in calendar order from 1 January and at
    /// least two when given. Season `s` runs from `season_starts[s]` to the day
    /// before the next start; the days of the year before the first start
    /// belong to the last season, and season 0 is the reference.
    pub season_starts: &'a [SeasonStart],
}

/// The observation, events and exposures of the people, and how to cut them.
///
/// Events are ragged: person `i` has the entries
/// `events[event_offsets[i]..event_offsets[i + 1]]`. Fields left out with
/// `..PieceSpec::default()` are empty: no exposure, no age cut, no calendar.
#[derive(Debug, Clone, Copy, Default)]
pub struct PieceSpec<'a> {
    /// The start of observation of each person.
    pub start: &'a [f64],
    /// The end of observation of each person, after `start`, exclusive.
    pub end: &'a [f64],
    /// The offsets of each person's events, one more entry than people.
    pub event_offsets: &'a [usize],
    /// The event times, each in `[start, end)` of its person.
    pub events: &'a [f64],
    /// The exposure series. Each has its own windows and its own columns in
    /// the design, and windows of different series may overlap.
    pub exposures: &'a [ExposureSeries<'a>],
    /// The strictly increasing age cut points. Age band `b` is `[cut_(b-1),
    /// cut_b)`, with band 0 below the first cut.
    pub age_cuts: &'a [f64],
    /// Calendar periods and seasons, when the time is in days.
    pub calendar: Option<Calendar<'a>>,
}

/// A column of [`Pieces::design`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    /// The indicator of risk window `window` of exposure series `series`.
    Window {
        /// The exposure series.
        series: usize,
        /// The window within the series.
        window: usize,
    },
    /// The indicator of an age band, from 1; band 0 is the reference.
    AgeBand(usize),
    /// The indicator of a season, from 1; season 0 is the reference.
    Season(usize),
    /// The indicator of a calendar period, from 1; period 0 is the reference.
    CalendarPeriod(usize),
}

/// The pieces of the observation periods of a set of people.
#[derive(Debug, Clone)]
pub struct Pieces {
    /// The position of each piece's person in the input.
    pub person: Vec<u32>,
    /// The start of each piece.
    pub start: Vec<f64>,
    /// The end of each piece, exclusive.
    pub end: Vec<f64>,
    /// The events in each piece.
    pub events: Vec<f64>,
    /// The age band of each piece, from 0.
    pub age_band: Vec<u32>,
    /// The risk window of each piece in each exposure series, `window[s][k]`
    /// for series `s` and piece `k`, `None` outside every window of the
    /// series.
    pub window: Vec<Vec<Option<u32>>>,
    /// The season of each piece, 0 without seasons.
    pub season: Vec<u32>,
    /// The calendar period of each piece, 0 without calendar cuts.
    pub calendar_period: Vec<u32>,
    /// The number of age bands, one more than the cut points.
    pub n_age_bands: usize,
    /// The number of risk windows of each exposure series.
    pub n_windows: Vec<usize>,
    /// The number of seasons, 1 without seasons.
    pub n_seasons: usize,
    /// The number of calendar periods, one more than the calendar cuts.
    pub n_calendar_periods: usize,
}

/// The cases and events in one risk window, from [`Pieces::window_summary`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSummary {
    /// The exposure series.
    pub series: usize,
    /// The window within the series.
    pub window: usize,
    /// The people with observed time in the window.
    pub cases: usize,
    /// The people with observed time in the window, at least one event and
    /// observed time outside the window too, who carry information about it.
    pub informative_cases: usize,
    /// The events in the window.
    pub events: f64,
    /// The observed time in the window.
    pub time: f64,
}

impl Pieces {
    /// The value of `per_person`, one value per person of the input, at the
    /// person of each piece, for a covariate of the person.
    ///
    /// # Errors
    ///
    /// An input error when `per_person` has no value for the person of some
    /// piece.
    pub fn gather<T: Copy>(&self, per_person: &[T]) -> Result<Vec<T>> {
        crate::expanded::gather(&self.person, per_person, "per_person")
    }

    /// The length of each piece, `end - start`.
    #[must_use]
    pub fn length(&self) -> Vec<f64> {
        self.start
            .iter()
            .zip(&self.end)
            .map(|(a, b)| b - a)
            .collect()
    }

    /// The pieces as the input of [`fit`], given the lengths from
    /// [`Pieces::length`].
    #[must_use]
    pub fn data<'a>(&'a self, length: &'a [f64]) -> SccsData<'a> {
        SccsData {
            person: &self.person,
            length,
            events: &self.events,
        }
    }

    /// What each column of [`Pieces::design`] is.
    #[must_use]
    pub fn columns(&self) -> Vec<Column> {
        let mut out = Vec::new();
        for (series, &n) in self.n_windows.iter().enumerate() {
            out.extend((0..n).map(|window| Column::Window { series, window }));
        }
        out.extend((1..self.n_age_bands).map(Column::AgeBand));
        out.extend((1..self.n_seasons).map(Column::Season));
        out.extend((1..self.n_calendar_periods).map(Column::CalendarPeriod));
        out
    }

    /// The design: one indicator column per risk window of each exposure
    /// series in turn, then one per age band after the first, one per season
    /// after the first and one per calendar period after the first, as
    /// [`Pieces::columns`] lists them.
    #[must_use]
    pub fn design(&self) -> Mat<f64> {
        let columns = self.columns();
        Mat::from_fn(self.person.len(), columns.len(), |i, j| {
            let on = match columns[j] {
                Column::Window { series, window } => {
                    self.window[series][i] == u32::try_from(window).ok()
                }
                Column::AgeBand(b) => self.age_band[i] as usize == b,
                Column::Season(s) => self.season[i] as usize == s,
                Column::CalendarPeriod(c) => self.calendar_period[i] as usize == c,
            };
            f64::from(u8::from(on))
        })
    }

    /// The people with observed time in any risk window of exposure series
    /// `series`: the cases exposed while observed.
    ///
    /// # Panics
    ///
    /// When `series` is not a series of the pieces.
    #[must_use]
    pub fn exposed_cases(&self, series: usize) -> usize {
        let window = &self.window[series];
        let mut count = 0;
        let mut k = 0;
        while k < self.person.len() {
            let p = self.person[k];
            let mut exposed = false;
            while k < self.person.len() && self.person[k] == p {
                exposed |= window[k].is_some();
                k += 1;
            }
            count += usize::from(exposed);
        }
        count
    }

    /// The cases, events and time of each risk window of each series, in the
    /// order of the window columns of [`Pieces::design`].
    #[must_use]
    pub fn window_summary(&self) -> Vec<WindowSummary> {
        let mut out = Vec::new();
        for (series, &n) in self.n_windows.iter().enumerate() {
            let label = &self.window[series];
            for window in 0..n {
                let w = u32::try_from(window).ok();
                let mut summary = WindowSummary {
                    series,
                    window,
                    cases: 0,
                    informative_cases: 0,
                    events: 0.0,
                    time: 0.0,
                };
                let mut k = 0;
                while k < self.person.len() {
                    let p = self.person[k];
                    let (mut inside, mut outside, mut events) = (false, false, 0.0);
                    while k < self.person.len() && self.person[k] == p {
                        if label[k] == w {
                            inside = true;
                            summary.events += self.events[k];
                            summary.time += self.end[k] - self.start[k];
                        } else {
                            outside = true;
                        }
                        events += self.events[k];
                        k += 1;
                    }
                    summary.cases += usize::from(inside);
                    summary.informative_cases += usize::from(inside && outside && events > 0.0);
                }
                out.push(summary);
            }
        }
        out
    }
}

fn check_offsets(o: &[usize], n: usize, len: usize) -> bool {
    o.len() == n + 1 && o[0] == 0 && o[n] == len && o.windows(2).all(|w| w[0] <= w[1])
}

fn check_windows(series: usize, windows: &[WindowSpec]) -> Result<()> {
    for (k, w) in windows.iter().enumerate() {
        if w.from.is_nan()
            || w.to.is_nan()
            || w.from >= w.to
            || w.from == f64::INFINITY
            || w.from == f64::NEG_INFINITY
        {
            return Err(invalid(format!(
                "window {k} of exposure series {series} is [{}, {}), which is not a non-empty interval with a finite start",
                w.from, w.to
            )));
        }
        for (l, v) in windows.iter().enumerate().take(k) {
            if w.from < v.to && v.from < w.to {
                return Err(invalid(format!(
                    "windows {l} and {k} of exposure series {series} overlap"
                )));
            }
        }
    }
    Ok(())
}

/// Whether `(month, day)` is a day of the calendar other than 29 February.
fn valid_season_start(s: SeasonStart) -> bool {
    const DAYS: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    (1..=12).contains(&s.month) && s.day >= 1 && s.day <= DAYS[s.month as usize - 1]
}

fn check_calendar(cal: &Calendar<'_>, n: usize) -> Result<()> {
    if cal.origin.len() != n {
        return Err(invalid("the calendar origin needs one entry per person"));
    }
    if let Some(i) = cal.origin.iter().position(|o| !o.is_finite()) {
        return Err(invalid(format!(
            "the calendar origin of person {i} is {}, which is not finite",
            cal.origin[i]
        )));
    }
    if !cal.cuts.iter().all(|c| c.is_finite()) || cal.cuts.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid(
            "the calendar cuts must be finite and strictly increasing",
        ));
    }
    if cal.season_starts.len() == 1 {
        return Err(invalid("seasons need at least two season starts"));
    }
    if let Some(s) = cal.season_starts.iter().find(|&&s| !valid_season_start(s)) {
        return Err(invalid(format!(
            "the season start {}-{} is not a day of every year",
            s.month, s.day
        )));
    }
    if cal
        .season_starts
        .windows(2)
        .any(|w| (w[0].month, w[0].day) >= (w[1].month, w[1].day))
    {
        return Err(invalid(
            "the season starts must be in calendar order without repeats",
        ));
    }
    Ok(())
}

pub(crate) fn check_spec(spec: &PieceSpec<'_>) -> Result<usize> {
    let n = spec.start.len();
    if spec.end.len() != n || spec.event_offsets.len() != n + 1 {
        return Err(invalid(
            "start and end need one entry per person and the event offsets one more",
        ));
    }
    if u32::try_from(n).is_err() {
        return Err(invalid("more people than a u32 addresses"));
    }
    if !check_offsets(spec.event_offsets, n, spec.events.len()) {
        return Err(invalid(
            "the event offsets must start at 0, not decrease and end at the number of events",
        ));
    }
    for (s, series) in spec.exposures.iter().enumerate() {
        if !check_offsets(series.offsets, n, series.dates.len()) {
            return Err(invalid(format!(
                "the offsets of exposure series {s} must have one more entry than people, start at 0, not decrease and end at the number of its dates"
            )));
        }
        if let Some(&t) = series.dates.iter().find(|t| !t.is_finite()) {
            return Err(invalid(format!(
                "exposure series {s} has the date {t}, which is not finite"
            )));
        }
        check_windows(s, series.windows)?;
    }
    for i in 0..n {
        if !(spec.start[i].is_finite() && spec.end[i].is_finite() && spec.start[i] < spec.end[i]) {
            return Err(invalid(format!(
                "person {i} has the observation period [{}, {}), which is not a finite, non-empty interval",
                spec.start[i], spec.end[i]
            )));
        }
    }
    if let Some(&t) = spec.events.iter().find(|t| !t.is_finite()) {
        return Err(invalid(format!("an event time is {t}")));
    }
    if !spec.age_cuts.iter().all(|c| c.is_finite())
        || spec.age_cuts.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(invalid(
            "the age cut points must be finite and strictly increasing",
        ));
    }
    if let Some(cal) = &spec.calendar {
        check_calendar(cal, n)?;
    }
    Ok(n)
}

/// The day count of a time on the calendar, rounded down.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a finite day count far inside the i64 range"
)]
fn day_of(t: f64) -> i64 {
    t.floor() as i64
}

/// The season boundaries of person `i` on their own time scale, as `(time,
/// season)` in time order, from the last start on or before `start` to the
/// last start before `end`.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "day counts are far below 2^53 and seasons are few"
)]
fn season_boundaries(cal: &Calendar<'_>, origin: f64, start: f64, end: f64) -> Vec<(f64, u32)> {
    let first_year = epistat_core::day::to_civil(day_of(origin + start)).0 - 1;
    let last_year = epistat_core::day::to_civil(day_of(origin + end)).0;
    let mut out = Vec::new();
    for year in first_year..=last_year {
        for (s, start_day) in cal.season_starts.iter().enumerate() {
            let day = epistat_core::day::from_civil(
                year,
                i64::from(start_day.month),
                i64::from(start_day.day),
            );
            out.push((day as f64 - origin, s as u32));
        }
    }
    out
}

/// One person's pieces appended to the output.
#[expect(
    clippy::cast_possible_truncation,
    reason = "window, age band, season and period counts are small"
)]
fn cut_person(spec: &PieceSpec<'_>, i: usize, out: &mut Pieces) -> Result<()> {
    let (start, end) = (spec.start[i], spec.end[i]);
    let events = &spec.events[spec.event_offsets[i]..spec.event_offsets[i + 1]];
    for &t in events {
        if !(t >= start && t < end) {
            return Err(invalid(format!(
                "person {i} has an event at {t} outside the observation period [{start}, {end})"
            )));
        }
    }
    let dates: Vec<&[f64]> = spec
        .exposures
        .iter()
        .map(|s| &s.dates[s.offsets[i]..s.offsets[i + 1]])
        .collect();
    let mut cuts = vec![start, end];
    let mut push = |c: f64| {
        if c > start && c < end {
            cuts.push(c);
        }
    };
    for &c in spec.age_cuts {
        push(c);
    }
    for (series, dates) in spec.exposures.iter().zip(&dates) {
        for &d in *dates {
            for w in series.windows {
                push(d + w.from);
                push(d + w.to);
            }
        }
    }
    let mut seasons = Vec::new();
    let origin = spec.calendar.map_or(0.0, |cal| cal.origin[i]);
    if let Some(cal) = &spec.calendar {
        for &c in cal.cuts {
            push(c - origin);
        }
        if !cal.season_starts.is_empty() {
            seasons = season_boundaries(cal, origin, start, end);
            for &(t, _) in &seasons {
                push(t);
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let person = u32::try_from(i).map_err(|_| invalid("more people than a u32 addresses"))?;
    let first = out.person.len();
    let mut labels: Vec<Option<u32>> = vec![None; spec.exposures.len()];
    for pair in cuts.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let band = spec.age_cuts.partition_point(|&c| c <= a) as u32;
        let period =
            spec.calendar
                .map_or(0, |cal| cal.cuts.partition_point(|&c| c - origin <= a)) as u32;
        let season = seasons
            .iter()
            .rev()
            .find(|&&(t, _)| t <= a)
            .map_or(0, |&(_, s)| s);
        // In each series, the latest exposure date with a window that covers
        // the piece.
        for ((series, dates), label) in spec.exposures.iter().zip(&dates).zip(&mut labels) {
            let mut best: Option<(f64, u32)> = None;
            for &d in *dates {
                for (k, w) in series.windows.iter().enumerate() {
                    if d + w.from <= a && a < d + w.to && best.is_none_or(|(top, _)| d > top) {
                        best = Some((d, k as u32));
                    }
                }
            }
            *label = best.map(|(_, k)| k);
        }
        let count = events.iter().filter(|&&t| t >= a && t < b).count() as f64;
        let last = out.person.len();
        let same = last > first
            && out.age_band[last - 1] == band
            && out.season[last - 1] == season
            && out.calendar_period[last - 1] == period
            && out
                .window
                .iter()
                .zip(&labels)
                .all(|(w, l)| w[last - 1] == *l);
        if same {
            out.end[last - 1] = b;
            out.events[last - 1] += count;
        } else {
            out.person.push(person);
            out.start.push(a);
            out.end.push(b);
            out.events.push(count);
            out.age_band.push(band);
            out.season.push(season);
            out.calendar_period.push(period);
            for (w, l) in out.window.iter_mut().zip(&labels) {
                w.push(*l);
            }
        }
    }
    Ok(())
}

/// Cuts the observation period of each person into pieces of constant age
/// band, risk window of each exposure series, season and calendar period, and
/// counts the events in each. See the module documentation for the rules.
///
/// # Errors
///
/// An input error when the columns and offsets disagree in length, a time is
/// not finite, an observation period is empty, an event lies outside its
/// person's observation period, the age or calendar cut points are not
/// strictly increasing, a window is empty, two windows of one series overlap,
/// or a season start is not a day of every year or out of calendar order.
pub fn build_pieces(spec: &PieceSpec<'_>) -> Result<Pieces> {
    let n = check_spec(spec)?;
    let blocks = ordered_reduce(
        n,
        PERSON_BLOCK,
        64,
        |r| {
            let mut part = empty(spec);
            let mut status = Ok(());
            for i in r {
                if let Err(e) = cut_person(spec, i, &mut part) {
                    status = Err(e);
                    break;
                }
            }
            (part, status)
        },
        |total, next| {
            if total.1.is_ok() {
                if next.1.is_err() {
                    total.1 = next.1;
                } else {
                    total.0.person.extend(next.0.person);
                    total.0.start.extend(next.0.start);
                    total.0.end.extend(next.0.end);
                    total.0.events.extend(next.0.events);
                    total.0.age_band.extend(next.0.age_band);
                    total.0.season.extend(next.0.season);
                    total.0.calendar_period.extend(next.0.calendar_period);
                    for (w, v) in total.0.window.iter_mut().zip(next.0.window) {
                        w.extend(v);
                    }
                }
            }
        },
    );
    blocks.1?;
    Ok(blocks.0)
}

fn empty(spec: &PieceSpec<'_>) -> Pieces {
    let (n_seasons, n_periods) = spec.calendar.map_or((1, 1), |cal| {
        (cal.season_starts.len().max(1), cal.cuts.len() + 1)
    });
    Pieces {
        person: Vec::new(),
        start: Vec::new(),
        end: Vec::new(),
        events: Vec::new(),
        age_band: Vec::new(),
        window: vec![Vec::new(); spec.exposures.len()],
        season: Vec::new(),
        calendar_period: Vec::new(),
        n_age_bands: spec.age_cuts.len() + 1,
        n_windows: spec.exposures.iter().map(|s| s.windows.len()).collect(),
        n_seasons,
        n_calendar_periods: n_periods,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_pieces_one_event_each_is_a_logit_of_the_rate_ratio() {
        // Person 0 has pieces of length 2 and 1, with the event in the second
        // piece, exposed. beta = ln(odds) with offsets: p = e^b * 1 / (e^b * 1 + 2).
        let person = [0, 0];
        let length = [2.0, 1.0];
        let events = [0.0, 1.0];
        let x = Mat::from_fn(2, 1, |i, _| i as f64);
        let data = SccsData {
            person: &person,
            length: &length,
            events: &events,
        };
        let opts = SccsOptions {
            max_iterations: 200,
            ..SccsOptions::default()
        };
        let f = fit(&data, x.as_ref(), &opts).unwrap();
        // The likelihood is p(b); its supremum is at p = 1, which is separation.
        assert!(!f.converged);
        assert_eq!(f.monotone_columns, vec![0]);
        assert!(f.coefficients[0].is_none());
    }

    #[test]
    fn pieces_are_cut_at_cuts_and_windows() {
        let spec = PieceSpec {
            start: &[0.0],
            end: &[100.0],
            event_offsets: &[0, 2],
            events: &[10.0, 30.0],
            exposures: &[ExposureSeries {
                offsets: &[0, 1],
                dates: &[20.0],
                windows: &[WindowSpec {
                    from: 0.0,
                    to: 10.0,
                }],
            }],
            age_cuts: &[50.0],
            calendar: None,
        };
        let pieces = build_pieces(&spec).unwrap();
        assert_eq!(pieces.start, vec![0.0, 20.0, 30.0, 50.0]);
        assert_eq!(pieces.end, vec![20.0, 30.0, 50.0, 100.0]);
        assert_eq!(pieces.events, vec![1.0, 0.0, 1.0, 0.0]);
        assert_eq!(pieces.window, vec![vec![None, Some(0), None, None]]);
        assert_eq!(pieces.age_band, vec![0, 0, 0, 1]);
    }

    #[test]
    fn seasons_and_calendar_periods_cut_on_the_calendar() {
        // Time is age in days from a birth on 2000-01-01, day 10_957. Seasons
        // start on 1 March and 1 September; a calendar cut on 2001-01-01.
        let birth = epistat_core::day::from_civil(2000, 1, 1) as f64;
        let march = epistat_core::day::from_civil(2000, 3, 1) as f64 - birth;
        let september = epistat_core::day::from_civil(2000, 9, 1) as f64 - birth;
        let new_year = epistat_core::day::from_civil(2001, 1, 1) as f64;
        let march_next = epistat_core::day::from_civil(2001, 3, 1) as f64 - birth;
        let spec = PieceSpec {
            start: &[10.0],
            end: &[500.0],
            event_offsets: &[0, 1],
            events: &[100.0],
            calendar: Some(Calendar {
                origin: &[birth],
                cuts: &[new_year],
                season_starts: &[
                    SeasonStart { month: 3, day: 1 },
                    SeasonStart { month: 9, day: 1 },
                ],
            }),
            ..PieceSpec::default()
        };
        let p = build_pieces(&spec).unwrap();
        let year = new_year - birth;
        assert_eq!(p.start, vec![10.0, march, september, year, march_next]);
        assert_eq!(p.end, vec![march, september, year, march_next, 500.0]);
        // January and February belong to the last season, the autumn one.
        assert_eq!(p.season, vec![1, 0, 1, 1, 0]);
        assert_eq!(p.calendar_period, vec![0, 0, 0, 1, 1]);
        assert_eq!(
            p.columns(),
            vec![Column::Season(1), Column::CalendarPeriod(1)]
        );
    }
}
