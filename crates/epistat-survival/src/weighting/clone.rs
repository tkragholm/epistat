//! Clone-censor-weight analysis of sustained treatment strategies with grace
//! periods, with a bootstrap over persons.
//!
//! # Design
//!
//! A strategy such as "start treatment within 6 months of diagnosis and stay on
//! it" cannot be assigned to a person at time zero from the data of time zero,
//! since whether the person follows it depends on what happens in the next
//! months. Assigning it by what happens later gives immortal time bias: a person
//! who starts treatment in month 5 has lived to month 5. The way round this is
//! to give every person a copy for each strategy at time zero (a clone), to
//! follow every clone until its person's observed treatment first departs from
//! the strategy, to censor it there, and to weight the clones that remain by
//! the inverse probability of not having been censored (Cain et al. 2010,
//! Hernan 2016, Maringe et al. 2020).
//!
//! # Cloning and censoring
//!
//! The data has one row for each person and period, the periods numbered from 0
//! at time zero. The treatment `a_j` of a period is the treatment at its start
//! and the outcome `y_j` happens during it. A clone of a strategy takes the rows
//! of its person in order. A row is the first deviation of the clone, and the
//! clone is censored at the start of it, when
//!
//! - for [`Strategy::InitiateBetween`] with `first = f` and `last = m`: the
//!   person has not started treatment before the row and either `a_j = 1` and
//!   `j < f` (started too early) or `a_j = 0` and `j = m` (the period `m` is
//!   the last in which starting is allowed); or the person started in period
//!   `s` and the adherence forbids stopping in the row while `a_j = 0`:
//!   [`Adherence::Continuous`] in every row after the start,
//!   [`Adherence::For`] with `d` in the rows `s < j < s + d`, and
//!   [`Adherence::None`] in none. [`Strategy::InitiateWithin`] with `grace = m`
//!   is `f = 0`;
//! - for [`Strategy::InitiateWhen`] with `grace = g`: as `InitiateBetween` with
//!   `f = k` and `m = k + g`, where `k` is the first period in which the
//!   trigger holds on the covariate of the row. Before `k` every period is one
//!   in which starting deviates, and a person in whom the trigger never holds
//!   follows the strategy untreated;
//! - for [`Strategy::NeverInitiate`] with `until = u`: `a_j = 1` and `j <= u`, or
//!   at any period without `until`.
//!
//! A clone ends at the row of its first outcome. The row of a deviation is kept
//! in the output, marked as a deviation, since the model of the weights is
//! fitted on it, and it carries no outcome. A person with a deviation in period
//! 0 for every strategy has no clone row with an outcome and is counted in
//! [`Cloned::n_without_follow_up`].
//!
//! # Weights
//!
//! A clone is at risk of deviating in a row when a deviation can happen there:
//! in the periods before the first allowed start and in the last allowed one
//! before treatment is started, in the periods after the start that the
//! adherence constrains, and for `NeverInitiate` in every period it covers. The
//! deviation in a row at risk is a binary variable `d_j`. For
//! each strategy a pooled logistic model of `d_j` is fitted on the rows at risk of
//! the clones of the strategy, with a denominator design `z` of the covariates
//! and the time, and optionally a numerator design that stabilises the weights.
//! The weight of a row is the product over the rows of the clone up to and
//! including it of
//!
//! ```text
//! (1 - p_n(j)) / (1 - p_d(j))     in a row at risk,
//! 1                               otherwise,
//! ```
//!
//! with the numerator probability `p_n` equal to 0 without stabilisation. This is
//! the inverse of the probability of remaining uncensored through the row given
//! the history in the model. In a strategy with a grace period the clones that
//! start treatment within the period are not at risk of the deviation of period
//! `m`, so they carry the weight of their adherence only, and the persons who
//! have not started at `m` are the clones whose censoring the weights account
//! for through their covariates before `m`. For this to be valid the covariates
//! of the denominator model must contain everything that predicts both the
//! starting of treatment and the outcome; for a dynamic strategy that includes
//! the covariate of its trigger, which decides the rows at risk.
//!
//! # The outcome
//!
//! The outcome model is the weighted pooled logistic model of
//! [`trials::fit_msm`](super::trials::fit_msm) on the rows of the clones that are
//! not a deviation, with a design that holds the strategy and the time, and the
//! cumulative incidence of each strategy comes from
//! [`trials::cumulative_incidence`](super::trials::cumulative_incidence). The
//! two clones of a person are two rows at each period, so the covariance is
//! clustered by person. It treats the weights as known. [`fit_msm_clones`] fits
//! the same model without building its design, from the clones and a function
//! that forms the design of a clone row. The person bootstrap that refits
//! everything, which is the variance to report, is
//! [`trials::bootstrap_trials`](super::trials::bootstrap_trials) on the ids of
//! the input, with [`Cloned::case_weights`] turning a resample into weights of
//! the clone rows.
//!
//! # References
//!
//! 1. Cain LE, Robins JM, Lanoy E, Logan R, Costagliola D, Hernan MA. When to
//!    start treatment? A systematic approach to the comparison of dynamic
//!    regimes using observational data. Int J Biostat 2010;6(2):Article 18.
//! 2. Hernan MA, Sauer BC, Hernandez-Diaz S, Platt R, Shrier I. Specifying a
//!    target trial prevents immortal time bias and other self-inflicted injuries
//!    in observational analyses. J Clin Epidemiol 2016;79:70-5.
//! 3. Maringe C, Benitez Majano S, Exarchakou A, et al. Reflection on modern
//!    methods: trial emulation in the presence of immortal-time bias. Int J
//!    Epidemiol 2020;49(5):1719-29.
//! 4. Young JG, Cain LE, Robins JM, O'Reilly EJ, Hernan MA. Comparative
//!    effectiveness of dynamic treatment regimes: an application of the
//!    parametric g-formula. Stat Biosci 2011;3(1):119-43.

use epistat_glm::rows::{Block, Rows};
use epistat_glm::{Family, Glm, Link};
use epistat_linalg::{Mat, MatRef, ROW_BLOCK};
use rayon::prelude::*;

use super::trials::PersonCounts;
use crate::error::{Error, Result};

/// Whether a person who has started treatment must stay on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adherence {
    /// Stopping treatment is a deviation.
    Continuous,
    /// Treatment after the start is not constrained.
    None,
    /// Treatment must go on for this many periods from the start, the period
    /// of the start included; stopping in them is a deviation, and treatment
    /// after them is not constrained. `For(1)` is `None`.
    For(u32),
}

/// When a covariate sets off a dynamic strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Trigger {
    /// The covariate is below the threshold.
    Below {
        /// The position of the covariate in [`CloneInput::covariates`].
        covariate: usize,
        /// The threshold.
        threshold: f64,
    },
    /// The covariate is at or above the threshold. A 0 or 1 indicator, such as
    /// a flare in the period, sets it off with a threshold of 1.
    AtLeast {
        /// The position of the covariate in [`CloneInput::covariates`].
        covariate: usize,
        /// The threshold.
        threshold: f64,
    },
}

impl Trigger {
    const fn covariate(self) -> usize {
        match self {
            Self::Below { covariate, .. } | Self::AtLeast { covariate, .. } => covariate,
        }
    }

    fn holds(self, value: f64) -> bool {
        match self {
            Self::Below { threshold, .. } => value < threshold,
            Self::AtLeast { threshold, .. } => value >= threshold,
        }
    }
}

/// A treatment strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Strategy {
    /// Start treatment in one of the periods `0..=grace`, and then follow
    /// `adherence`.
    InitiateWithin {
        /// The last period in which treatment may be started.
        grace: u32,
        /// The adherence after the start.
        adherence: Adherence,
    },
    /// Start treatment in one of the periods `first..=last`, not before, and
    /// then follow `adherence`. `InitiateWithin { grace }` is `first = 0` and
    /// `last = grace`.
    InitiateBetween {
        /// The first period in which treatment may be started.
        first: u32,
        /// The last period in which treatment may be started.
        last: u32,
        /// The adherence after the start.
        adherence: Adherence,
    },
    /// Do not start treatment, in any period or up to and including `until`.
    NeverInitiate {
        /// The last period in which treatment must not be given, `None` for all.
        until: Option<u32>,
    },
    /// Start treatment in one of the `grace + 1` periods from the first period
    /// in which `trigger` holds, not before it, and then follow `adherence`; a
    /// person in whom it never holds follows the strategy untreated. This is
    /// the dynamic strategy "start within `grace` periods of the first flare"
    /// or "of the covariate first falling below a threshold" (Cain et al.
    /// 2010).
    InitiateWhen {
        /// What sets the strategy off.
        trigger: Trigger,
        /// The periods after the first one in which the trigger holds in which
        /// treatment may still be started.
        grace: u32,
        /// The adherence after the start.
        adherence: Adherence,
    },
}

/// A person-period data set. The rows of a person are next to each other, in
/// order, and start at period 0.
#[derive(Debug, Clone, Copy)]
pub struct CloneInput<'a> {
    /// The person of each row.
    pub id: &'a [u64],
    /// The period of each row, 0, 1, 2, ... within a person.
    pub period: &'a [i64],
    /// Whether the person is treated at the start of the period.
    pub treatment: &'a [bool],
    /// Whether the outcome happens in the period.
    pub outcome: &'a [bool],
    /// The time-varying covariates that the triggers of dynamic strategies
    /// read, one slice of rows each, measured at the start of the period.
    /// Empty when no strategy has a trigger.
    pub covariates: &'a [&'a [f64]],
}

/// The clones of all persons under all strategies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cloned {
    /// The row of the input each clone row is made from, for building the
    /// designs of the models.
    pub source: Vec<usize>,
    /// The strategy of each clone row, an index into the strategies.
    pub arm: Vec<usize>,
    /// The person of each clone row, numbered from 0 in the order of the input,
    /// which is increasing id order.
    pub person: Vec<u32>,
    /// The id of each person, in increasing order: `ids[person[k]]` is the id
    /// of clone row `k`.
    pub ids: Vec<u64>,
    /// Whether the row is the first deviation of its clone.
    pub deviation: Vec<bool>,
    /// Whether the row is in the model of the weights of its strategy.
    pub at_risk: Vec<bool>,
    /// Whether the outcome happens in the row. False in a deviation row.
    pub outcome: Vec<bool>,
    /// The number of persons.
    pub n_persons: usize,
    /// The persons with no clone row that carries an outcome observation, for
    /// any strategy.
    pub n_without_follow_up: usize,
}

impl Cloned {
    /// The value of `column`, one value per row of the input, at the input row
    /// of each clone row.
    ///
    /// # Errors
    ///
    /// An input error when `column` has no value at some input row.
    pub fn gather<T: Copy>(&self, column: &[T]) -> Result<Vec<T>> {
        match self.source.iter().find(|&&r| r >= column.len()) {
            Some(_) => {
                let needed = self.source.iter().max().map_or(0, |&r| r + 1);
                Err(epistat_core::InputError::Length {
                    what: "column",
                    expected: needed,
                    found: column.len(),
                }
                .into())
            }
            None => Ok(self.source.iter().map(|&r| column[r]).collect()),
        }
    }

    /// The number of clone rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.source.len()
    }

    /// Whether there are no clone rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.source.is_empty()
    }

    /// The case weight of each clone row in a resample of the persons: the
    /// count of its person in `counts`, from
    /// [`trials::bootstrap_trials`](super::trials::bootstrap_trials) or
    /// [`PersonCounts::draw`] on the ids of the input.
    ///
    /// # Errors
    ///
    /// An input error for a person whose id was not among the ids resampled.
    pub fn case_weights(&self, counts: &PersonCounts) -> Result<Vec<f64>> {
        let by_person = counts.case_weights(&self.ids)?;
        Ok(self.person.iter().map(|&p| by_person[p as usize]).collect())
    }
}

/// Clone every person under every strategy and censor each clone at its first
/// deviation.
///
/// # Errors
///
/// An input error for columns of different lengths, no strategy, rows that
/// are not ordered by person with the periods 0, 1, 2, ... in a person, and a
/// grace period or `until` beyond the longest follow-up, which no person can
/// reach.
pub fn clone_censor(input: &CloneInput<'_>, strategies: &[Strategy]) -> Result<Cloned> {
    let n = input.id.len();
    for (name, len) in [
        ("period", input.period.len()),
        ("treatment", input.treatment.len()),
        ("outcome", input.outcome.len()),
    ] {
        if len != n {
            return Err(Error::invalid(format!("{name}: {len} values for {n} rows")));
        }
    }
    if strategies.is_empty() {
        return Err(Error::invalid("strategies: at least one is needed"));
    }
    let mut n_persons = 0usize;
    let mut starts = Vec::new();
    let mut ids = Vec::new();
    for i in 0..n {
        if i == 0 || input.id[i] != input.id[i - 1] {
            if input.period[i] != 0 {
                return Err(Error::invalid(format!(
                    "period: person {} does not start at period 0 in row {i}",
                    input.id[i]
                )));
            }
            if i > 0 && input.id[i] < input.id[i - 1] {
                return Err(Error::invalid(format!(
                    "id: the rows of a person must be next to each other and ordered, row {i}"
                )));
            }
            starts.push(i);
            ids.push(input.id[i]);
            n_persons += 1;
        } else if input.period[i] != input.period[i - 1] + 1 {
            return Err(Error::invalid(format!(
                "period: row {i} does not follow the row before it"
            )));
        }
    }
    starts.push(n);
    let longest = starts.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    for s in strategies {
        let reach = match *s {
            Strategy::InitiateWithin { grace, .. } => Some(grace),
            Strategy::InitiateBetween { first, last, .. } => {
                if first > last {
                    return Err(Error::invalid(format!(
                        "strategy {s:?}: the first period of starting is after the last"
                    )));
                }
                Some(last)
            }
            Strategy::NeverInitiate { until } => until,
            Strategy::InitiateWhen { trigger, .. } => {
                let k = trigger.covariate();
                let Some(column) = input.covariates.get(k) else {
                    return Err(Error::invalid(format!(
                        "strategy {s:?}: covariate {k} is not among the {} covariates",
                        input.covariates.len()
                    )));
                };
                if column.len() != n {
                    return Err(Error::invalid(format!(
                        "covariate {k}: {} values for {n} rows",
                        column.len()
                    )));
                }
                if let Some(r) = column.iter().position(|v| !v.is_finite()) {
                    return Err(Error::invalid(format!(
                        "covariate {k}: the value of row {r} is not finite"
                    )));
                }
                None
            }
        };
        if let Some(r) = reach
            && r as usize >= longest
        {
            return Err(Error::invalid(format!(
                "strategy {s:?}: period {r} is beyond the longest follow-up of {longest} periods"
            )));
        }
        if let Strategy::InitiateWithin { adherence, .. }
        | Strategy::InitiateBetween { adherence, .. }
        | Strategy::InitiateWhen { adherence, .. } = *s
            && adherence == Adherence::For(0)
        {
            return Err(Error::invalid(format!(
                "strategy {s:?}: treatment must go on for at least one period"
            )));
        }
    }
    // The clone rows are counted first, so that each column is allocated once
    // at its length.
    let mut rows = 0_usize;
    let mut without = 0_usize;
    for_each_clone_row(input, strategies, &starts, &mut |_| rows += 1, &mut || {
        without += 1;
    });
    let mut out = Cloned {
        source: Vec::with_capacity(rows),
        arm: Vec::with_capacity(rows),
        person: Vec::with_capacity(rows),
        ids,
        deviation: Vec::with_capacity(rows),
        at_risk: Vec::with_capacity(rows),
        outcome: Vec::with_capacity(rows),
        n_persons,
        n_without_follow_up: without,
    };
    for_each_clone_row(
        input,
        strategies,
        &starts,
        &mut |c| {
            out.source.push(c.row);
            out.arm.push(c.arm);
            out.person.push(c.person);
            out.deviation.push(c.deviation);
            out.at_risk.push(c.at_risk);
            out.outcome.push(c.outcome);
        },
        &mut || {},
    );
    Ok(out)
}

/// Whether the row of period `j`, with treatment `a`, deviates from a
/// strategy that starts treatment in one of the periods `first..=last` and
/// then follows `adherence`, and whether a deviation can happen in it (the row
/// is at risk). `started` is the period of the start before the row.
fn initiation(
    started: Option<i64>,
    j: i64,
    first: i64,
    last: i64,
    adherence: Adherence,
    a: bool,
) -> (bool, bool) {
    match started {
        Some(s) => {
            let risk = match adherence {
                Adherence::Continuous => true,
                Adherence::None => false,
                Adherence::For(d) => j - s < i64::from(d),
            };
            (risk && !a, risk)
        }
        None if j < first => (a, true),
        None if j == last => (!a, true),
        None => (false, false),
    }
}

/// One clone row as [`clone_censor`] forms it.
struct CloneRow {
    row: usize,
    arm: usize,
    person: u32,
    deviation: bool,
    at_risk: bool,
    outcome: bool,
}

/// Calls `row` for each clone row in order, and `unobserved` for each person
/// without a clone row that carries an outcome observation.
fn for_each_clone_row(
    input: &CloneInput<'_>,
    strategies: &[Strategy],
    starts: &[usize],
    row: &mut impl FnMut(CloneRow),
    unobserved: &mut impl FnMut(),
) {
    for (p, w) in starts.windows(2).enumerate() {
        let mut observed = false;
        for (k, strategy) in strategies.iter().enumerate() {
            // The period in which treatment was started, and the first period
            // in which the trigger held, before the row.
            let mut started: Option<i64> = None;
            let mut triggered: Option<i64> = None;
            for r in w[0]..w[1] {
                let j = input.period[r];
                let a = input.treatment[r];
                let (deviates, risk) = match *strategy {
                    Strategy::InitiateWithin { grace, adherence } => {
                        initiation(started, j, 0, i64::from(grace), adherence, a)
                    }
                    Strategy::InitiateBetween {
                        first,
                        last,
                        adherence,
                    } => initiation(started, j, i64::from(first), i64::from(last), adherence, a),
                    Strategy::NeverInitiate { until } => {
                        let risk = until.is_none_or(|u| j <= i64::from(u));
                        (risk && a, risk)
                    }
                    Strategy::InitiateWhen {
                        trigger,
                        grace,
                        adherence,
                    } => {
                        if triggered.is_none()
                            && trigger.holds(input.covariates[trigger.covariate()][r])
                        {
                            triggered = Some(j);
                        }
                        match triggered {
                            Some(t) => {
                                initiation(started, j, t, t + i64::from(grace), adherence, a)
                            }
                            // Before the trigger every period is one in which
                            // starting deviates; once started, the adherence
                            // applies as in the other strategies.
                            None => initiation(started, j, i64::MAX, i64::MAX, adherence, a),
                        }
                    }
                };
                let event = !deviates && input.outcome[r];
                row(CloneRow {
                    row: r,
                    arm: k,
                    person: p as u32,
                    deviation: deviates,
                    at_risk: risk,
                    outcome: event,
                });
                if deviates {
                    break;
                }
                observed = true;
                if a && started.is_none() {
                    started = Some(j);
                }
                if event {
                    break;
                }
            }
        }
        if !observed {
            unobserved();
        }
    }
}

/// The designs of the models of the deviation of one strategy, with one row for
/// each clone row of every strategy.
#[derive(Debug, Clone, Copy)]
pub struct DeviationModel<'a> {
    /// The denominator design of every clone row, with an intercept.
    pub denominator: MatRef<'a, f64>,
    /// The numerator design of every clone row, with an intercept, for
    /// stabilised weights.
    pub numerator: Option<MatRef<'a, f64>>,
}

fn subset(x: MatRef<'_, f64>, rows: &[usize]) -> Mat<f64> {
    Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)])
}

fn fit_deviation(
    x: MatRef<'_, f64>,
    d: &[bool],
    rows: &[usize],
    case: Option<&[f64]>,
    what: &str,
) -> Result<Vec<f64>> {
    let positives = rows.iter().filter(|&&r| d[r]).count();
    if positives == 0 || positives == rows.len() {
        return Err(Error::invalid(format!(
            "{what}: the rows at risk need both deviations and rows that are not"
        )));
    }
    let xs = subset(x, rows);
    let ys: Vec<f64> = rows.iter().map(|&r| f64::from(u8::from(d[r]))).collect();
    let ws: Option<Vec<f64>> = case.map(|c| rows.iter().map(|&r| c[r]).collect());
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit(&crate::logistic::data(
            xs.as_ref(),
            &ys,
            ws.as_deref(),
            None,
            None,
        )?)
        .map_err(Error::model(what))?;
    let coef = fit.full_rank().map_err(Error::model(what))?;
    Ok(coef)
}

fn probability(x: MatRef<'_, f64>, coef: &[f64], row: usize) -> f64 {
    let eta: f64 = (0..coef.len()).map(|j| x[(row, j)] * coef[j]).sum();
    Link::Logit.inverse(eta)
}

/// The weights of the clones of one fit of the deviation models.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviationWeights {
    /// The weight of each clone row through the row.
    pub weight: Vec<f64>,
    /// The fitted probability of deviating, under the denominator model, in each
    /// clone row (also those not at risk).
    pub probability: Vec<f64>,
    /// The coefficients of the denominator model of each strategy.
    pub coefficients: Vec<Vec<f64>>,
}

/// The weights of the clone rows from pooled logistic models of the deviation,
/// one for each strategy fitted on its rows at risk.
///
/// `models` has one entry for each strategy, with designs of one row per clone
/// row. `case_weight` weights the rows in the fits, for the resamples of a
/// bootstrap, and is `None` otherwise.
///
/// # Errors
///
/// An input error for models or designs that do not match the clones, a
/// strategy without rows at risk or whose rows at risk are all deviations or none,
/// a model that has an aliased column or does not converge, and a case weight of
/// the wrong length, negative or not finite.
pub fn deviation_weights(
    cloned: &Cloned,
    models: &[DeviationModel<'_>],
    case_weight: Option<&[f64]>,
) -> Result<DeviationWeights> {
    let n = cloned.len();
    let n_arms = cloned.arm.iter().copied().max().map_or(0, |a| a + 1);
    if models.len() < n_arms {
        return Err(Error::invalid(format!(
            "models: {} for {n_arms} strategies",
            models.len()
        )));
    }
    if let Some(c) = case_weight
        && (c.len() != n || c.iter().any(|v| !(v.is_finite() && *v >= 0.0)))
    {
        return Err(Error::invalid(
            "case_weight: one non-negative finite value per clone row",
        ));
    }
    let mut p_d = vec![0.0; n];
    let mut p_n = vec![0.0; n];
    let mut coefficients = Vec::new();
    for (k, model) in models.iter().enumerate().take(n_arms) {
        for (name, m) in [
            ("denominator", Some(model.denominator)),
            ("numerator", model.numerator),
        ] {
            if let Some(m) = m {
                if m.nrows() != n || m.ncols() == 0 {
                    return Err(Error::invalid(format!(
                        "strategy {k} {name}: {} rows and {} columns for {n} clone rows",
                        m.nrows(),
                        m.ncols()
                    )));
                }
                if !epistat_linalg::is_finite(m) {
                    return Err(Error::invalid(format!(
                        "strategy {k} {name}: a value is not finite"
                    )));
                }
            }
        }
        let rows: Vec<usize> = (0..n)
            .filter(|&r| cloned.arm[r] == k && cloned.at_risk[r])
            .collect();
        let coef = fit_deviation(
            model.denominator,
            &cloned.deviation,
            &rows,
            case_weight,
            &format!("strategy {k} denominator"),
        )?;
        for (r, p) in p_d.iter_mut().enumerate() {
            if cloned.arm[r] == k {
                *p = probability(model.denominator, &coef, r);
            }
        }
        coefficients.push(coef);
        if let Some(num) = model.numerator {
            let c = fit_deviation(
                num,
                &cloned.deviation,
                &rows,
                case_weight,
                &format!("strategy {k} numerator"),
            )?;
            for (r, p) in p_n.iter_mut().enumerate() {
                if cloned.arm[r] == k {
                    *p = probability(num, &c, r);
                }
            }
        }
    }
    let mut weight = vec![0.0; n];
    let mut cum = 1.0;
    for r in 0..n {
        if r == 0 || cloned.person[r] != cloned.person[r - 1] || cloned.arm[r] != cloned.arm[r - 1]
        {
            cum = 1.0;
        }
        if cloned.at_risk[r] {
            cum *= (1.0 - p_n[r]) / (1.0 - p_d[r]);
        }
        weight[r] = cum;
    }
    Ok(DeviationWeights {
        weight,
        probability: p_d,
        coefficients,
    })
}

/// The design of one model of the deviation, formed for a clone row when it
/// is needed instead of held for every clone row.
#[derive(Clone, Copy)]
pub struct DeviationDesign<'a> {
    /// The number of columns, with the intercept.
    pub columns: usize,
    /// Writes the design of clone row `row` into `out`, `columns` values.
    pub fill: &'a (dyn Fn(usize, &mut [f64]) + Sync),
}

impl std::fmt::Debug for DeviationDesign<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviationDesign")
            .field("columns", &self.columns)
            .finish_non_exhaustive()
    }
}

/// The designs of the models of the deviation of one strategy as
/// [`DeviationDesign`]s, the form of [`DeviationModel`] for
/// [`deviation_weights_rows`].
#[derive(Debug, Clone, Copy)]
pub struct DeviationRows<'a> {
    /// The denominator design, with an intercept.
    pub denominator: DeviationDesign<'a>,
    /// The numerator design, with an intercept, for stabilised weights.
    pub numerator: Option<DeviationDesign<'a>>,
}

/// [`deviation_weights`] with designs formed row by row: the same weights,
/// probabilities and coefficients, bit for bit, as from matrices that hold the
/// values `fill` writes, without the matrices or the copies of their rows at
/// risk. The models are fitted by [`Glm::fit_rows`], which forms the rows at
/// risk block by block, and the numerator probabilities are formed where the
/// weights are, so the memory beyond the clones is the weights and the
/// probabilities. The fits read their rows about `p^2` times for `p` columns,
/// which [`deviation_weights`] does not.
///
/// # Errors
///
/// As [`deviation_weights`], for a design with no column or a value that is
/// not finite in any clone row.
pub fn deviation_weights_rows(
    cloned: &Cloned,
    models: &[DeviationRows<'_>],
    case_weight: Option<&[f64]>,
) -> Result<DeviationWeights> {
    let n = cloned.len();
    let n_arms = cloned.arm.iter().copied().max().map_or(0, |a| a + 1);
    if models.len() < n_arms {
        return Err(Error::invalid(format!(
            "models: {} for {n_arms} strategies",
            models.len()
        )));
    }
    if let Some(c) = case_weight
        && (c.len() != n || c.iter().any(|v| !(v.is_finite() && *v >= 0.0)))
    {
        return Err(Error::invalid(
            "case_weight: one non-negative finite value per clone row",
        ));
    }
    let mut p_d = vec![0.0; n];
    let mut coefficients = Vec::new();
    let mut numerators: Vec<Option<(DeviationDesign<'_>, Vec<f64>)>> = Vec::new();
    for (k, model) in models.iter().enumerate().take(n_arms) {
        for (name, m) in [
            ("denominator", Some(model.denominator)),
            ("numerator", model.numerator),
        ] {
            if let Some(m) = m {
                if m.columns == 0 {
                    return Err(Error::invalid(format!(
                        "strategy {k} {name}: {n} rows and 0 columns for {n} clone rows"
                    )));
                }
                if !finite_design(m, n) {
                    return Err(Error::invalid(format!(
                        "strategy {k} {name}: a value is not finite"
                    )));
                }
            }
        }
        let at_risk = AtRisk::new(cloned, k, case_weight);
        let coef = fit_deviation_rows(
            &at_risk,
            model.denominator,
            &format!("strategy {k} denominator"),
        )?;
        fill_probabilities(cloned, k, model.denominator, &coef, &mut p_d);
        coefficients.push(coef);
        let numerator = match model.numerator {
            Some(num) => {
                let c = fit_deviation_rows(&at_risk, num, &format!("strategy {k} numerator"))?;
                Some((num, c))
            }
            None => None,
        };
        numerators.push(numerator);
    }
    // The factor of each row at risk, formed in parallel, then multiplied in
    // order within each clone.
    let mut weight = vec![1.0; n];
    weight
        .par_chunks_mut(ROW_BLOCK)
        .enumerate()
        .for_each(|(b, chunk)| {
            let mut values = Vec::new();
            for (i, w) in chunk.iter_mut().enumerate() {
                let r = b * ROW_BLOCK + i;
                if !cloned.at_risk[r] {
                    continue;
                }
                let p_n = match &numerators[cloned.arm[r]] {
                    Some((design, coef)) => {
                        values.resize(design.columns, 0.0);
                        (design.fill)(r, &mut values);
                        probability_of(&values, coef)
                    }
                    None => 0.0,
                };
                *w = (1.0 - p_n) / (1.0 - p_d[r]);
            }
        });
    let mut cum = 1.0;
    for (r, w) in weight.iter_mut().enumerate() {
        if r == 0 || cloned.person[r] != cloned.person[r - 1] || cloned.arm[r] != cloned.arm[r - 1]
        {
            cum = 1.0;
        }
        if cloned.at_risk[r] {
            cum *= *w;
        }
        *w = cum;
    }
    Ok(DeviationWeights {
        weight,
        probability: p_d,
        coefficients,
    })
}

/// Whether `design` gives finite values in every one of `n` clone rows.
fn finite_design(design: DeviationDesign<'_>, n: usize) -> bool {
    (0..n.div_ceil(ROW_BLOCK)).into_par_iter().all(|b| {
        let mut values = vec![0.0; design.columns];
        (b * ROW_BLOCK..((b + 1) * ROW_BLOCK).min(n)).all(|r| {
            (design.fill)(r, &mut values);
            values.iter().all(|v| v.is_finite())
        })
    })
}

/// The probability of the logistic model with coefficients `coef` for the
/// design row `x`, as `probability` forms it from a matrix.
fn probability_of(x: &[f64], coef: &[f64]) -> f64 {
    let eta: f64 = (0..coef.len()).map(|j| x[j] * coef[j]).sum();
    Link::Logit.inverse(eta)
}

/// The denominator probability of every clone row of strategy `k`.
fn fill_probabilities(
    cloned: &Cloned,
    k: usize,
    design: DeviationDesign<'_>,
    coef: &[f64],
    p_d: &mut [f64],
) {
    p_d.par_chunks_mut(ROW_BLOCK)
        .enumerate()
        .for_each(|(b, chunk)| {
            let mut values = vec![0.0; design.columns];
            for (i, p) in chunk.iter_mut().enumerate() {
                let r = b * ROW_BLOCK + i;
                if cloned.arm[r] == k {
                    (design.fill)(r, &mut values);
                    *p = probability_of(&values, coef);
                }
            }
        });
}

/// The rows at risk of one strategy, in order, with the clone row at which
/// each block of them starts and the number of deviations among them.
struct AtRisk<'a> {
    cloned: &'a Cloned,
    arm: usize,
    case: Option<&'a [f64]>,
    blocks: Vec<usize>,
    rows: usize,
    deviations: usize,
}

impl<'a> AtRisk<'a> {
    fn new(cloned: &'a Cloned, arm: usize, case: Option<&'a [f64]>) -> Self {
        let (mut blocks, mut rows, mut deviations) = (Vec::new(), 0, 0);
        for r in 0..cloned.len() {
            if cloned.arm[r] == arm && cloned.at_risk[r] {
                if rows % ROW_BLOCK == 0 {
                    blocks.push(r);
                }
                rows += 1;
                deviations += usize::from(cloned.deviation[r]);
            }
        }
        Self {
            cloned,
            arm,
            case,
            blocks,
            rows,
            deviations,
        }
    }
}

/// The rows at risk of one strategy with the design of one of its models.
struct AtRiskRows<'a> {
    at_risk: &'a AtRisk<'a>,
    design: DeviationDesign<'a>,
}

impl Rows for AtRiskRows<'_> {
    fn n_rows(&self) -> usize {
        self.at_risk.rows
    }

    fn n_columns(&self) -> usize {
        self.design.columns
    }

    fn fill(&self, start: usize, mut block: Block<'_>) {
        assert_eq!(
            start % ROW_BLOCK,
            0,
            "a block starts at a multiple of ROW_BLOCK"
        );
        let a = self.at_risk;
        let c = a.cloned;
        let mut r = a.blocks[start / ROW_BLOCK];
        let mut values = vec![0.0; self.design.columns];
        for i in 0..block.y.len() {
            while !(c.arm[r] == a.arm && c.at_risk[r]) {
                r += 1;
            }
            (self.design.fill)(r, &mut values);
            for (j, &v) in values.iter().enumerate() {
                block.x[(i, j)] = v;
            }
            block.y[i] = f64::from(u8::from(c.deviation[r]));
            block.weight[i] = a.case.map_or(1.0, |w| w[r]);
            block.cluster[i] = 0;
            r += 1;
        }
    }
}

/// `fit_deviation` on the rows at risk formed by `design`.
fn fit_deviation_rows(
    at_risk: &AtRisk<'_>,
    design: DeviationDesign<'_>,
    what: &str,
) -> Result<Vec<f64>> {
    if at_risk.deviations == 0 || at_risk.deviations == at_risk.rows {
        return Err(Error::invalid(format!(
            "{what}: the rows at risk need both deviations and rows that are not"
        )));
    }
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit_rows(&AtRiskRows { at_risk, design })
        .map_err(Error::model(what))?;
    let coef = crate::logistic::full_rank(&fit, what)?;
    Ok(coef)
}

/// [`trials::fit_msm`](super::trials::fit_msm) on the clone rows that are not a
/// deviation without building their design: the rows in order, with the design
/// that `design(row, out)` writes into `out` for clone row `row`, the outcome
/// of the row, its `weight` (one per clone row, such as
/// [`DeviationWeights::weight`]) and its person as its cluster.
///
/// The result has the bits of building the design, outcomes, weights and
/// persons of those rows and fitting them with `fit_msm`, and the memory is
/// that of the clones: the rows are formed block by block as the fit needs
/// them, by [`Glm::fit_rows`](epistat_glm::Glm::fit_rows). See
/// [`epistat_glm::rows`].
///
/// # Errors
///
/// An input error for a `weight` of another length than the clone rows,
/// clone rows whose persons are not in order, `columns` of 0, and the errors
/// of `fit_msm`.
pub fn fit_msm_clones(
    cloned: &Cloned,
    weight: &[f64],
    columns: usize,
    design: &(dyn Fn(usize, &mut [f64]) + Sync),
) -> Result<super::trials::Msm> {
    let n = cloned.len();
    if weight.len() != n {
        return Err(Error::invalid(format!(
            "weight: {} values for {n} clone rows",
            weight.len()
        )));
    }
    for (name, len) in [
        ("arm", cloned.arm.len()),
        ("person", cloned.person.len()),
        ("deviation", cloned.deviation.len()),
        ("outcome", cloned.outcome.len()),
    ] {
        if len != n {
            return Err(Error::invalid(format!(
                "{name}: {len} values for {n} clone rows"
            )));
        }
    }
    let mut blocks = Vec::new();
    let (mut kept, mut clusters) = (0_usize, 0_usize);
    let mut last: Option<u32> = None;
    for r in 0..n {
        if cloned.deviation[r] {
            continue;
        }
        let person = cloned.person[r];
        if let Some(l) = last
            && person < l
        {
            return Err(Error::invalid(format!(
                "person: clone row {r} is before the rows of an earlier person"
            )));
        }
        if last != Some(person) {
            clusters += 1;
            last = Some(person);
        }
        if kept % ROW_BLOCK == 0 {
            blocks.push((r, clusters - 1));
        }
        kept += 1;
    }
    let rows = CloneRows {
        cloned,
        weight,
        columns,
        design,
        blocks,
        rows: kept,
        clusters,
    };
    super::trials::fit_msm_rows(&rows, columns)
}

/// The clone rows that are not a deviation, formed from the clones.
struct CloneRows<'a> {
    cloned: &'a Cloned,
    weight: &'a [f64],
    columns: usize,
    design: &'a (dyn Fn(usize, &mut [f64]) + Sync),
    /// The clone row and the cluster of the first row of each block.
    blocks: Vec<(usize, usize)>,
    rows: usize,
    clusters: usize,
}

impl Rows for CloneRows<'_> {
    fn n_rows(&self) -> usize {
        self.rows
    }

    fn n_columns(&self) -> usize {
        self.columns
    }

    fn fill(&self, start: usize, mut block: Block<'_>) {
        assert_eq!(
            start % ROW_BLOCK,
            0,
            "a block starts at a multiple of ROW_BLOCK"
        );
        let (mut r, mut cluster) = self.blocks[start / ROW_BLOCK];
        let c = self.cloned;
        let mut values = vec![0.0; self.columns];
        let mut person = c.person[r];
        for i in 0..block.y.len() {
            while c.deviation[r] {
                r += 1;
            }
            if c.person[r] != person {
                cluster += 1;
                person = c.person[r];
            }
            (self.design)(r, &mut values);
            for (j, &v) in values.iter().enumerate() {
                block.x[(i, j)] = v;
            }
            block.y[i] = f64::from(u8::from(c.outcome[r]));
            block.weight[i] = self.weight[r];
            block.cluster[i] = cluster;
            r += 1;
        }
    }
}

impl super::trials::Clustered for CloneRows<'_> {
    fn n_clusters(&self) -> usize {
        self.clusters
    }
}
