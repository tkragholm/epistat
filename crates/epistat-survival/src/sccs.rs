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
//! person are not independent given the person's rate.
//!
//! # The pieces
//!
//! [`build_pieces`] cuts the observation period `[start_i, end_i)` of each
//! person at the age cut points and at the boundaries of every risk window of
//! every exposure date, and counts the events in each piece. Time is
//! continuous and intervals are half open: an event at a boundary belongs to
//! the piece that starts there, and for dates in days `end` is the last
//! observed day plus one. All times, the observation period, the exposure
//! dates, the events and the age cut points, are on one scale, usually age in
//! days. A [`WindowSpec`] covers `[date + from, date + to)` for each exposure
//! date of the person, and `to` may be infinite. Windows of one exposure may
//! not overlap. Where windows of different exposure dates overlap, the piece
//! takes the window of the latest exposure date, so that a pre-exposure window
//! of a later dose takes precedence over the late window of an earlier dose.
//! Neighbouring pieces with the same age band and window are merged, which
//! does not change the likelihood. The pieces of a person are contiguous and
//! in time order, and people appear in the order of the input.
//!
//! # What the likelihood assumes
//!
//! The likelihood is valid when (a) the events are independent given the
//! person's rate, so that recurrent events are rare or only the first event is
//! used, (b) the observation period does not depend on the events, which is
//! violated when death or emigration follows the event, and (c) exposure after
//! an event does not depend on the event, which is violated when the exposure
//! treats the early symptoms of the event. A pre-exposure window in the design
//! shows (c). The extensions for event-dependent exposures and observation
//! periods of Farrington et al. \[3, 4\] change the likelihood by a factor that
//! depends on the person's own times, which per-piece lengths cannot express,
//! and are not provided. Nor is the smooth age effect of \[5\].
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
//! with `glm` Poisson with person dummies, with `survival::clogit`, and with a
//! hand-written R loop for the pieces. `tests/data/sccs.R` records the
//! versions.
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

use epistat_linalg::{Mat, MatRef};

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
    Error::input(message)
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
/// [`Error::Input`] when the columns differ in length, `x` has no column or a
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
        log_likelihood: None,
        null_log_likelihood: None,
        n_persons: persons.ranges.len(),
        n_persons_used: n_used,
        n_persons_without_events: n_without,
        n_persons_one_piece: n_one,
        n_events,
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
    if opts.robust {
        let meat = blocked(data, x, &cols, &persons, &beta, true).meat;
        let sandwich = sandwich(&inverse, &meat, p);
        let mut full = vec![None; n_cols * n_cols];
        for (a, &ca) in cols.iter().enumerate() {
            for (b, &cb) in cols.iter().enumerate() {
                full[ca * n_cols + cb] = Some(sandwich[a * p + b]);
            }
        }
        result.robust_covariance = Some(full);
    }
    Ok(result)
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

fn sandwich(inverse: &[f64], meat: &[f64], p: usize) -> Vec<f64> {
    let mut half = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..p {
                s += inverse[i * p + k] * meat[k * p + j];
            }
            half[i * p + j] = s;
        }
    }
    let mut out = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..p {
                s += half[i * p + k] * inverse[k * p + j];
            }
            out[i * p + j] = s;
            out[j * p + i] = s;
        }
    }
    out
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

/// The observation, events and exposures of the people, and how to cut them.
///
/// Events and exposure dates are ragged: person `i` has the entries
/// `events[event_offsets[i]..event_offsets[i + 1]]`, and the same for
/// exposures.
#[derive(Debug, Clone, Copy)]
pub struct PieceSpec<'a> {
    /// The start of observation of each person.
    pub start: &'a [f64],
    /// The end of observation of each person, after `start`, exclusive.
    pub end: &'a [f64],
    /// The offsets of each person's events, one more entry than people.
    pub event_offsets: &'a [usize],
    /// The event times, each in `[start, end)` of its person.
    pub events: &'a [f64],
    /// The offsets of each person's exposure dates, one more entry than people.
    pub exposure_offsets: &'a [usize],
    /// The exposure dates. A date outside the observation period is allowed,
    /// since its windows may reach into it.
    pub exposures: &'a [f64],
    /// The strictly increasing age cut points. Age band `b` is `[cut_(b-1),
    /// cut_b)`, with band 0 below the first cut.
    pub age_cuts: &'a [f64],
    /// The risk windows, which do not overlap.
    pub windows: &'a [WindowSpec],
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
    /// The risk window of each piece, `None` outside every window.
    pub window: Vec<Option<u32>>,
    /// The number of age bands, one more than the cut points.
    pub n_age_bands: usize,
    /// The number of risk windows.
    pub n_windows: usize,
}

impl Pieces {
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

    /// The design: one indicator column per risk window, then one per age band
    /// after the first, which is the reference.
    #[must_use]
    pub fn design(&self) -> Mat<f64> {
        let (w, a) = (self.n_windows, self.n_age_bands.saturating_sub(1));
        Mat::from_fn(self.person.len(), w + a, |i, j| {
            if j < w {
                f64::from(u8::from(self.window[i] == u32::try_from(j).ok()))
            } else {
                f64::from(u8::from(self.age_band[i] as usize == j - w + 1))
            }
        })
    }
}

fn check_spec(spec: &PieceSpec<'_>) -> Result<usize> {
    let n = spec.start.len();
    if spec.end.len() != n
        || spec.event_offsets.len() != n + 1
        || spec.exposure_offsets.len() != n + 1
    {
        return Err(invalid(
            "start and end need one entry per person and the offsets one more",
        ));
    }
    if u32::try_from(n).is_err() {
        return Err(invalid("more people than a u32 addresses"));
    }
    let offsets_ok =
        |o: &[usize], len: usize| o[0] == 0 && o[n] == len && o.windows(2).all(|w| w[0] <= w[1]);
    if !offsets_ok(spec.event_offsets, spec.events.len())
        || !offsets_ok(spec.exposure_offsets, spec.exposures.len())
    {
        return Err(invalid(
            "the offsets must start at 0, not decrease and end at the length of their column",
        ));
    }
    for i in 0..n {
        if !(spec.start[i].is_finite() && spec.end[i].is_finite() && spec.start[i] < spec.end[i]) {
            return Err(invalid(format!(
                "person {i} has the observation period [{}, {}), which is not a finite, non-empty interval",
                spec.start[i], spec.end[i]
            )));
        }
    }
    if let Some(&t) = spec
        .events
        .iter()
        .chain(spec.exposures)
        .find(|t| !t.is_finite())
    {
        return Err(invalid(format!("an event or exposure time is {t}")));
    }
    if !spec.age_cuts.iter().all(|c| c.is_finite())
        || spec.age_cuts.windows(2).any(|w| w[0] >= w[1])
    {
        return Err(invalid(
            "the age cut points must be finite and strictly increasing",
        ));
    }
    for (k, w) in spec.windows.iter().enumerate() {
        if w.from.is_nan()
            || w.to.is_nan()
            || w.from >= w.to
            || w.from == f64::INFINITY
            || w.from == f64::NEG_INFINITY
        {
            return Err(invalid(format!(
                "window {k} is [{}, {}), which is not a non-empty interval with a finite start",
                w.from, w.to
            )));
        }
        for (l, v) in spec.windows.iter().enumerate().take(k) {
            if w.from < v.to && v.from < w.to {
                return Err(invalid(format!("windows {l} and {k} overlap")));
            }
        }
    }
    Ok(n)
}

/// One person's pieces appended to the output.
#[expect(
    clippy::cast_possible_truncation,
    reason = "window and age band counts are small"
)]
fn cut_person(spec: &PieceSpec<'_>, i: usize, out: &mut Pieces) -> Result<()> {
    let (start, end) = (spec.start[i], spec.end[i]);
    let events = &spec.events[spec.event_offsets[i]..spec.event_offsets[i + 1]];
    let dates = &spec.exposures[spec.exposure_offsets[i]..spec.exposure_offsets[i + 1]];
    for &t in events {
        if !(t >= start && t < end) {
            return Err(invalid(format!(
                "person {i} has an event at {t} outside the observation period [{start}, {end})"
            )));
        }
    }
    let mut cuts = vec![start, end];
    for &c in spec.age_cuts {
        if c > start && c < end {
            cuts.push(c);
        }
    }
    for &d in dates {
        for w in spec.windows {
            for c in [d + w.from, d + w.to] {
                if c > start && c < end {
                    cuts.push(c);
                }
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    let person = u32::try_from(i).map_err(|_| invalid("more people than a u32 addresses"))?;
    let first = out.person.len();
    for pair in cuts.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let band = spec.age_cuts.partition_point(|&c| c <= a);
        // The latest exposure date with a window that covers the piece.
        let mut label: Option<(f64, u32)> = None;
        for &d in dates {
            for (k, w) in spec.windows.iter().enumerate() {
                if d + w.from <= a && a < d + w.to {
                    let better = label.is_none_or(|(best, _)| d > best);
                    if better {
                        label = Some((d, k as u32));
                    }
                }
            }
        }
        let window = label.map(|(_, k)| k);
        let band = band as u32;
        let count = events.iter().filter(|&&t| t >= a && t < b).count() as f64;
        let last = out.person.len();
        if last > first && out.age_band[last - 1] == band && out.window[last - 1] == window {
            out.end[last - 1] = b;
            out.events[last - 1] += count;
        } else {
            out.person.push(person);
            out.start.push(a);
            out.end.push(b);
            out.events.push(count);
            out.age_band.push(band);
            out.window.push(window);
        }
    }
    Ok(())
}

/// Cuts the observation period of each person into pieces of constant age band
/// and risk window, and counts the events in each. See the module
/// documentation for the rules.
///
/// # Errors
///
/// [`Error::Input`] when the columns and offsets disagree in length, a time is
/// not finite, an observation period is empty, an event lies outside its
/// person's observation period, the age cut points are not strictly increasing,
/// a window is empty or two windows overlap.
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
                    total.0.window.extend(next.0.window);
                }
            }
        },
    );
    blocks.1?;
    Ok(blocks.0)
}

fn empty(spec: &PieceSpec<'_>) -> Pieces {
    Pieces {
        person: Vec::new(),
        start: Vec::new(),
        end: Vec::new(),
        events: Vec::new(),
        age_band: Vec::new(),
        window: Vec::new(),
        n_age_bands: spec.age_cuts.len() + 1,
        n_windows: spec.windows.len(),
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
            exposure_offsets: &[0, 1],
            exposures: &[20.0],
            age_cuts: &[50.0],
            windows: &[WindowSpec {
                from: 0.0,
                to: 10.0,
            }],
        };
        let pieces = build_pieces(&spec).unwrap();
        assert_eq!(pieces.start, vec![0.0, 20.0, 30.0, 50.0]);
        assert_eq!(pieces.end, vec![20.0, 30.0, 50.0, 100.0]);
        assert_eq!(pieces.events, vec![1.0, 0.0, 1.0, 0.0]);
        assert_eq!(pieces.window, vec![None, Some(0), None, None]);
        assert_eq!(pieces.age_band, vec![0, 0, 0, 1]);
    }
}
