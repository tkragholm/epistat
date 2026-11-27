//! Per-protocol weights and a marginal structural model for sequential trials,
//! as `TrialEmulation::initiators(estimand_type = "PP")`.
//!
//! # Design
//!
//! A sequential trial emulation starts a trial at every period at which a person
//! is eligible and follows the person in it until the treatment differs from the
//! treatment assigned at the start of the trial (the rows of the trials are made
//! by `epistat_matching::trials::expand_trials`). The analysis of the stacked
//! trials by protocol adherence censors a person at the deviation and weights the
//! persons who remain, so that they stand for everyone who would have followed
//! the protocol. This module holds the weights and the outcome model. The
//! expansion into trials is not done here.
//!
//! # Weights
//!
//! The data has one row for each person and period, ordered by period within the
//! person, with the treatment `a_t` of the period, the treatment `a_{t-1}` of the
//! one before it (0 in the first row) and the covariates of the period. For each
//! value `g` of the previous treatment a logistic model of the treatment is
//! fitted among the rows with `a_{t-1} = g`, once on the covariates of a
//! denominator model and once on those of a numerator model, `p_d(g, x_t)` and
//! `p_n(g, x_t)` of the treatment being 1. The weight of the row is
//!
//! ```text
//! w_t = p_n / p_d             if a_t = 1
//! w_t = (1 - p_n) / (1 - p_d) if a_t = 0
//! ```
//!
//! [`switch_weights`]. The weights of censoring have the same form with the
//! probability of remaining uncensored, `w_t = p_n / p_d` whatever the status of
//! the row, from models fitted either by the previous treatment or on all rows
//! ([`censoring_weights`]). The weight of a row in a trial is the product of the
//! weights of the periods after the start of the trial up to the row,
//!
//! ```text
//! W_(s, t) = (prod over k <= t of w_k) / (prod over k <= s of w_k),
//! ```
//!
//! for the trial that started in period `s` ([`trial_weights`]), formed as a
//! ratio of the two cumulative products as `TrialEmulation` does.
//!
//! # The outcome model
//!
//! The outcome model is a logistic regression of the event on the design of the
//! rows of the trials (the assigned treatment, the follow-up time, the trial
//! period and the covariates at the start), weighted by `W` ([`fit_msm`]). Its
//! covariance is the sandwich covariance clustered by person, multiplied by
//! `G / (G - 1)` for `G` persons, which is `sandwich::vcovCL(type = NULL)` as
//! `TrialEmulation` calls it. The cumulative
//! incidence of a treatment is the mean over the persons of a population of
//! `1 - prod over s <= t of (1 - p_s)` with `p_s` the fitted probability at
//! follow-up time `s` ([`cumulative_incidence`]). Its standard error is the
//! delta-method one: with `S_t = prod (1 - p_s)`, `d log S_t / d b = - sum over s
//! <= t of p_s x_s`, so the derivative of the mean is `- mean(S_t d log S_t / d
//! b)`, and the variance is that of the robust covariance of the coefficients.
//! `TrialEmulation` gives limits by drawing coefficients from the same
//! covariance.
//!
//! # Memory
//!
//! The stacked trials have about `T (T + 1) / 2` rows for each person followed
//! for `T` periods, so a design built for them is several times the data.
//! [`fit_msm_trials`] takes the trials as spans of the rows of the data and
//! forms their rows block by block while it fits, with the bits of
//! [`fit_msm`] on the built rows, so that the memory is that of the
//! person-period data and its weights. The models of the weights read their
//! rows from the design given, without a copy of the rows of each fit.
//!
//! # What the variance leaves out
//!
//! The covariance of the outcome model treats the weights as known, as
//! `TrialEmulation` does, so it does not account for their estimation. A
//! bootstrap over persons that refits the weights does.
//!
//! # References
//!
//! 1. Maringe C, Benitez Majano S, Exarchakou A, et al. Reflection on modern
//!    methods: trial emulation in the presence of immortal-time bias. Int J
//!    Epidemiol 2020;49(5):1719-29.
//! 2. Danaei G, Rodriguez LAG, Cantero OF, Logan R, Hernan MA. Observational data
//!    for comparative effectiveness research: an emulation of randomised trials
//!    of statins and primary prevention of coronary heart disease. Stat Methods
//!    Med Res 2013;22(1):70-96.
//! 3. TrialEmulation: Causal Analysis of Observational Time-to-Event Data. R
//!    package version 0.0.5.
//! 4. Zeileis A, Koll S, Graham N. Various versatile variances: an
//!    object-oriented implementation of clustered covariances in R. J Stat Softw
//!    2020;95(1):1-36.

use epistat_core::{Covariance, InputError};
use epistat_glm::robust::{SandwichType, robust_covariance};
use epistat_glm::rows::{Block, Rows};
use epistat_glm::{Family, Glm, Link};
use epistat_inference::Clusters;
use epistat_linalg::{Mat, MatRef, ROW_BLOCK};
use rayon::prelude::*;

use crate::error::{Error, Result};

mod bootstrap;
mod competing;
mod limits;
mod protocol;

pub use bootstrap::{PersonCounts, bootstrap_trials};
pub use competing::{
    CompetingIncidence, MsmCompeting, cumulative_incidence_competing, fit_msm_competing,
};
pub use limits::WeightLimits;
pub use protocol::{ProtocolPanel, ProtocolRows, protocol_rows};

/// The rows of a person-period data set, ordered by person and then period.
#[derive(Debug, Clone, Copy)]
pub struct PersonPeriod<'a> {
    /// The person of each row. The rows of a person are next to each other.
    pub id: &'a [u64],
    /// The period of each row, increasing within a person.
    pub period: &'a [i64],
    /// The treatment in the period.
    pub treatment: &'a [bool],
    /// The treatment in the period before, `false` in the first row of a person.
    pub previous: &'a [bool],
}

impl PersonPeriod<'_> {
    fn check(&self) -> Result<usize> {
        let n = self.id.len();
        for (name, len) in [
            ("period", self.period.len()),
            ("treatment", self.treatment.len()),
            ("previous", self.previous.len()),
        ] {
            if len != n {
                return Err(Error::invalid(format!("{name}: {len} values for {n} rows")));
            }
        }
        for i in 1..n {
            if self.id[i] == self.id[i - 1] && self.period[i] <= self.period[i - 1] {
                return Err(Error::invalid(format!(
                    "period: row {i} is not after the row before it for person {}",
                    self.id[i]
                )));
            }
            if self.id[i] < self.id[i - 1] {
                return Err(Error::invalid(format!(
                    "id: the rows of a person must be next to each other and ordered, row {i}"
                )));
            }
        }
        Ok(n)
    }
}

/// The designs of the numerator and denominator models of a weight.
#[derive(Debug, Clone, Copy)]
pub struct WeightModel<'a> {
    /// The design of the numerator model, with an intercept column.
    pub numerator: MatRef<'a, f64>,
    /// The design of the denominator model, with an intercept column.
    pub denominator: MatRef<'a, f64>,
}

/// How the models of a weight are fitted over the values of the previous
/// treatment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialWeightFit {
    /// One model for each value of the previous treatment.
    ByPreviousTreatment,
    /// One model on all rows.
    Pooled,
}

/// The weights of each row and the fitted models.
#[derive(Debug, Clone, PartialEq)]
pub struct RowWeights {
    /// The weight of each row.
    pub weight: Vec<f64>,
    /// The probability of the numerator model of each row, of the value 1 of the
    /// response.
    pub numerator_probability: Vec<f64>,
    /// The probability of the denominator model of each row.
    pub denominator_probability: Vec<f64>,
    /// The coefficients of the numerator models, for the previous treatment 0 and
    /// 1, or one model when pooled.
    pub numerator_coefficients: Vec<Vec<f64>>,
    /// The coefficients of the denominator models, as `numerator_coefficients`.
    pub denominator_coefficients: Vec<Vec<f64>>,
}

fn check_design(name: &str, x: MatRef<'_, f64>, n: usize) -> Result<()> {
    if x.nrows() != n {
        return Err(Error::invalid(format!(
            "{name}: {} rows for {n} rows of data",
            x.nrows()
        )));
    }
    if x.ncols() == 0 {
        return Err(Error::invalid(format!("{name}: no columns")));
    }
    if !epistat_linalg::is_finite(x) {
        return Err(Error::invalid(format!("{name}: a value is not finite")));
    }
    Ok(())
}

/// Fit a logistic model of `y` on `x` over `rows`, with the case weights
/// `case` when given, and give the coefficients and the fitted probability of
/// each of the rows.
fn fit_logistic(
    x: MatRef<'_, f64>,
    y: &[bool],
    rows: &[usize],
    case: Option<&[f64]>,
    what: &str,
) -> Result<(Vec<f64>, Vec<f64>)> {
    // A row of case weight 0 is not in the fit, as it is not in a resample.
    let counted = |r: usize| case.is_none_or(|c| c[r] > 0.0);
    let total = rows.iter().filter(|&&r| counted(r)).count();
    let positives = rows.iter().filter(|&&r| counted(r) && y[r]).count();
    if positives == 0 || positives == total {
        return Err(Error::invalid(format!(
            "{what}: the rows of the fit need both values of the response"
        )));
    }
    let subset = Subset { x, y, rows, case };
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit_rows(&subset)
        .map_err(Error::model(what))?;
    let coefficients = crate::logistic::full_rank(&fit, what)?;
    Ok((coefficients, fit.fitted(&subset)))
}

/// The rows `rows` of a design and a binary response as the rows of a fit,
/// in place of a copy of them.
struct Subset<'a> {
    x: MatRef<'a, f64>,
    y: &'a [bool],
    rows: &'a [usize],
    case: Option<&'a [f64]>,
}

impl Rows for Subset<'_> {
    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn n_columns(&self) -> usize {
        self.x.ncols()
    }

    fn fill(&self, start: usize, mut block: Block<'_>) {
        for k in 0..block.y.len() {
            let r = self.rows[start + k];
            for j in 0..self.x.ncols() {
                block.x[(k, j)] = self.x[(r, j)];
            }
            block.y[k] = f64::from(u8::from(self.y[r]));
            block.weight[k] = self.case.map_or(1.0, |c| c[r]);
            block.cluster[k] = 0;
        }
    }
}

/// Which rows the models of a weight are fitted on, and with which case
/// weights.
#[derive(Debug, Clone, Copy, Default)]
pub struct WeightRows<'a> {
    /// Whether each row is in the fits, all rows when `None`. A row that is not
    /// has the weight 1 and probabilities of NaN, so that it leaves the
    /// cumulative products of [`trial_weights`] as they are. [`protocol_rows`]
    /// gives the rows of `TrialEmulation`.
    pub fitted: Option<&'a [bool]>,
    /// The case weight of each row in the fits, 1 for every row when `None`. A
    /// row of weight 0 does not count towards the two values of the response a
    /// fit needs. The counts of a resample of persons (see [`bootstrap_trials`])
    /// give the fits of the resample.
    pub case_weight: Option<&'a [f64]>,
}

impl WeightRows<'_> {
    fn check(&self, n: usize) -> Result<()> {
        if let Some(f) = self.fitted
            && f.len() != n
        {
            return Err(Error::from(InputError::Length {
                what: "fitted",
                expected: n,
                found: f.len(),
            }));
        }
        if let Some(c) = self.case_weight
            && (c.len() != n || c.iter().any(|v| !(v.is_finite() && *v >= 0.0)))
        {
            return Err(Error::invalid(
                "case_weight: one non-negative finite value per row",
            ));
        }
        Ok(())
    }
}

/// The weights `p_n / p_d` of the value observed (`observed = true`) or of the
/// value 1 of the response.
fn ratio_weights(
    previous: &[bool],
    y: &[bool],
    model: &WeightModel<'_>,
    fit: TrialWeightFit,
    observed_value: bool,
    rows: &WeightRows<'_>,
    what: &str,
) -> Result<RowWeights> {
    let n = y.len();
    check_design("numerator", model.numerator, n)?;
    check_design("denominator", model.denominator, n)?;
    rows.check(n)?;
    let fitted = |r: usize| rows.fitted.is_none_or(|f| f[r]);
    let groups: Vec<Vec<usize>> = match fit {
        TrialWeightFit::Pooled => vec![(0..n).filter(|&r| fitted(r)).collect()],
        TrialWeightFit::ByPreviousTreatment => (0..2)
            .map(|g| {
                (0..n)
                    .filter(|&r| fitted(r) && previous[r] == (g == 1))
                    .collect()
            })
            .collect(),
    };
    let mut p_n = vec![f64::NAN; n];
    let mut p_d = vec![f64::NAN; n];
    let mut numerator_coefficients = Vec::new();
    let mut denominator_coefficients = Vec::new();
    for (g, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(Error::invalid(format!(
                "{what}: no rows with previous treatment {g}"
            )));
        }
        let (cn, pn) = fit_logistic(
            model.numerator,
            y,
            group,
            rows.case_weight,
            &format!("{what} numerator {g}"),
        )?;
        let (cd, pd) = fit_logistic(
            model.denominator,
            y,
            group,
            rows.case_weight,
            &format!("{what} denominator {g}"),
        )?;
        for (k, &r) in group.iter().enumerate() {
            p_n[r] = pn[k];
            p_d[r] = pd[k];
        }
        numerator_coefficients.push(cn);
        denominator_coefficients.push(cd);
    }
    let weight = (0..n)
        .map(|r| {
            if !fitted(r) {
                1.0
            } else if !observed_value || y[r] {
                p_n[r] / p_d[r]
            } else {
                (1.0 - p_n[r]) / (1.0 - p_d[r])
            }
        })
        .collect();
    Ok(RowWeights {
        weight,
        numerator_probability: p_n,
        denominator_probability: p_d,
        numerator_coefficients,
        denominator_coefficients,
    })
}

/// The weights of the treatment switches of each row: the logistic models of the
/// treatment among the rows of each previous treatment, `p_n / p_d` for a
/// treated row and `(1 - p_n) / (1 - p_d)` for an untreated one.
///
/// The models are fitted on the rows of `rows` with its case weights;
/// [`WeightRows::default`] fits them on every row with weight 1. The designs
/// have one row for every row of `data`.
///
/// # Errors
///
/// An input error for rows that are not ordered by person and period, for
/// slices or designs of another length than the rows, a design with no column or
/// a value that is not finite, a group of the previous treatment without rows or
/// without both treated and untreated rows, `rows` of another length than the
/// data or a case weight that is negative or not finite, and a model that has
/// an aliased column or does not converge.
pub fn switch_weights(
    data: &PersonPeriod<'_>,
    model: &WeightModel<'_>,
    rows: &WeightRows<'_>,
) -> Result<RowWeights> {
    data.check()?;
    ratio_weights(
        data.previous,
        data.treatment,
        model,
        TrialWeightFit::ByPreviousTreatment,
        true,
        rows,
        "switch",
    )
}

/// The weights of censoring of each row, `p_n / p_d` with the probability of
/// remaining uncensored from the models of the numerator and the denominator,
/// fitted by the previous treatment or on all rows.
///
/// `censored` marks the rows of persons who are censored in the period. The
/// models are fitted on the rows of `rows` with its case weights. With
/// [`WeightRows::default`] every row is in the fit, so that a data set from
/// which the rows after a deviation have been removed gives the weights of
/// `TrialEmulation`.
///
/// # Errors
///
/// As [`switch_weights`], with `censored` of another length than the rows.
pub fn censoring_weights(
    data: &PersonPeriod<'_>,
    censored: &[bool],
    model: &WeightModel<'_>,
    fit: TrialWeightFit,
    rows: &WeightRows<'_>,
) -> Result<RowWeights> {
    let n = data.check()?;
    if censored.len() != n {
        return Err(Error::from(InputError::Length {
            what: "censored",
            expected: n,
            found: censored.len(),
        }));
    }
    let uncensored: Vec<bool> = censored.iter().map(|&c| !c).collect();
    ratio_weights(
        data.previous,
        &uncensored,
        model,
        fit,
        false,
        rows,
        "censoring",
    )
}

/// The rows of the stacked trials: the person, the period the trial started and
/// the follow-up time of each row.
#[derive(Debug, Clone, Copy)]
pub struct TrialRows<'a> {
    /// The person of each row.
    pub id: &'a [u64],
    /// The period in which the trial of the row started.
    pub trial_period: &'a [i64],
    /// The periods since the start of the trial, 0 in the first row.
    pub follow_up: &'a [i64],
}

/// The weight of every row of the trials, the product of the weights of the
/// periods after the start of the trial up to the row, as the ratio of the
/// cumulative product of the person to that at the start of the trial.
///
/// `row_weight` has the weight of each row of `data`, for instance the product
/// of [`switch_weights`] and [`censoring_weights`].
///
/// # Errors
///
/// An input error for rows or weights that do not match, and for a row of a
/// trial whose person has no row in the period of the start of the trial or of
/// the row.
pub fn trial_weights(
    data: &PersonPeriod<'_>,
    row_weight: &[f64],
    rows: &TrialRows<'_>,
) -> Result<Vec<f64>> {
    let n = data.check()?;
    if row_weight.len() != n {
        return Err(Error::from(InputError::Length {
            what: "row_weight",
            expected: n,
            found: row_weight.len(),
        }));
    }
    let m = rows.id.len();
    if rows.trial_period.len() != m || rows.follow_up.len() != m {
        return Err(Error::invalid("trial rows: columns of different lengths"));
    }
    let mut cumulative = vec![0.0; n];
    for i in 0..n {
        cumulative[i] = if i > 0 && data.id[i] == data.id[i - 1] {
            cumulative[i - 1] * row_weight[i]
        } else {
            row_weight[i]
        };
    }
    // The first row of every person.
    let mut starts: Vec<(u64, usize, usize)> = Vec::new();
    for i in 0..n {
        if i == 0 || data.id[i] != data.id[i - 1] {
            starts.push((data.id[i], i, i + 1));
        } else if let Some(last) = starts.last_mut() {
            last.2 = i + 1;
        }
    }
    let find = |id: u64, period: i64| -> Option<usize> {
        let k = starts.binary_search_by_key(&id, |s| s.0).ok()?;
        let (_, a, b) = starts[k];
        let at = data.period[a..b].binary_search(&period).ok()?;
        Some(a + at)
    };
    (0..m)
        .map(|r| {
            let (id, s) = (rows.id[r], rows.trial_period[r]);
            let t = s + rows.follow_up[r];
            let (Some(a), Some(b)) = (find(id, s), find(id, t)) else {
                return Err(Error::invalid(format!(
                    "trial row {r}: person {id} has no row in period {s} or {t}"
                )));
            };
            Ok(cumulative[b] / cumulative[a])
        })
        .collect()
}

/// The data of the outcome model: one row for each row of the trials.
#[derive(Debug, Clone, Copy)]
pub struct MsmInput<'a> {
    /// The design: an intercept, the assigned treatment, the follow-up time and
    /// the covariates of the model.
    pub design: MatRef<'a, f64>,
    /// Whether the event happened in the row.
    pub outcome: &'a [bool],
    /// The weight of the row.
    pub weight: &'a [f64],
    /// The person of the row, the cluster of the covariance.
    pub person: &'a [u64],
}

/// The outcome model of the stacked trials.
#[derive(Debug, Clone, PartialEq)]
pub struct Msm {
    /// The coefficients.
    pub coefficients: Vec<f64>,
    /// The covariance of the coefficients clustered by person, `p` by `p`.
    pub covariance: Covariance,
    /// The number of rows.
    pub n_rows: usize,
    /// The number of persons.
    pub n_persons: usize,
}

impl Msm {
    /// The robust standard errors of the coefficients, `None` where a
    /// variance is undefined.
    #[must_use]
    pub fn se(&self) -> Vec<Option<f64>> {
        self.covariance
            .variances()
            .into_iter()
            .map(|v| v.map(|v| v.max(0.0).sqrt()))
            .collect()
    }
}

/// Fit the weighted logistic outcome model of the stacked trials, with the
/// covariance clustered by person.
///
/// # Errors
///
/// An input error for inputs of different lengths, a weight that is negative
/// or not finite, a model that has an aliased column or does not converge, and
/// fewer than two persons.
pub fn fit_msm(input: &MsmInput<'_>) -> Result<Msm> {
    let n = input.outcome.len();
    check_design("design", input.design, n)?;
    if input.weight.len() != n || input.person.len() != n {
        return Err(Error::invalid("weight and person need one value per row"));
    }
    let y: Vec<f64> = input
        .outcome
        .iter()
        .map(|&o| f64::from(u8::from(o)))
        .collect();
    let model = Glm::new(Family::Binomial, Link::Logit);
    let fit = model
        .fit(&crate::logistic::data(
            input.design,
            &y,
            Some(input.weight),
            None,
            None,
        )?)
        .map_err(Error::model("outcome model"))?;
    let p = input.design.ncols();
    let coefficients = fit.full_rank().map_err(Error::model("outcome model"))?;
    let clusters = Clusters::from_labels_first_seen(input.person);
    let cov = robust_covariance(
        &model,
        &fit,
        input.design,
        &y,
        Some(&clusters),
        SandwichType::Cr1,
    )
    .map_err(Error::model("outcome model"))?;
    let g = clusters.count();
    if g < 2 {
        return Err(Error::invalid(
            "outcome model: at least two persons are needed",
        ));
    }
    if cov.as_slice().iter().any(Option::is_none) {
        return Err(Error::undefined(
            "outcome model: the covariance is not defined",
        ));
    }
    debug_assert_eq!(cov.dim(), p);
    Ok(Msm {
        coefficients,
        covariance: cov,
        n_rows: n,
        n_persons: clusters.count(),
    })
}

/// The trials of a stacked analysis as spans of the rows of the person-period
/// data: trial `k` starts at row `start[k]` and follows its person through the
/// rows before `end[k]`.
///
/// The trials are ordered by their first row, so that the rows of a person are
/// next to each other in the stack. A trial that runs to the last row of its
/// person and one censored at a deviation are both a span.
#[derive(Debug, Clone, Copy)]
pub struct TrialSpans<'a> {
    /// The row of the data at which each trial starts.
    pub start: &'a [usize],
    /// The row after the last row of each trial.
    pub end: &'a [usize],
}

/// [`fit_msm`] on the stacked trials of `trials` without building them: the
/// rows of the trials in order, trial by trial and in each from its first row,
/// with the design that `design(start, row, out)` writes into `out` for the
/// trial that starts at row `start` of the data and its row `row`, the outcome
/// `outcome[row]`, the weight [`trial_weights`] gives the row and the person of
/// the row as its cluster.
///
/// The result has the bits of building the rows of the trials with
/// [`TrialRows`], weighting them with [`trial_weights`] and fitting them with
/// [`fit_msm`], and the memory is that of the data: the rows of the trials are
/// formed block by block as the fit needs them, by
/// [`Glm::fit_rows`](epistat_glm::Glm::fit_rows). The time is that of the
/// rows of the trials formed about `2 i + p^2` times for `i` iterations and `p`
/// columns, which the fit of a built design does not spend. See
/// [`epistat_glm::rows`].
///
/// # Errors
///
/// An input error for rows that are not ordered by person and period, a
/// `row_weight` or `outcome` of another length than the rows, spans of
/// different lengths, a trial that is empty, ends beyond the data or leaves
/// its person, trials not ordered by their first row, `columns` of 0, and the
/// errors of [`fit_msm`].
pub fn fit_msm_trials(
    data: &PersonPeriod<'_>,
    row_weight: &[f64],
    trials: &TrialSpans<'_>,
    outcome: &[bool],
    columns: usize,
    design: &(dyn Fn(usize, usize, &mut [f64]) + Sync),
) -> Result<Msm> {
    let n = data.check()?;
    if row_weight.len() != n {
        return Err(Error::from(InputError::Length {
            what: "row_weight",
            expected: n,
            found: row_weight.len(),
        }));
    }
    if outcome.len() != n {
        return Err(Error::from(InputError::Length {
            what: "outcome",
            expected: n,
            found: outcome.len(),
        }));
    }
    if trials.start.len() != trials.end.len() {
        return Err(Error::invalid("trial spans: columns of different lengths"));
    }
    for (k, (&a, &b)) in trials.start.iter().zip(trials.end).enumerate() {
        if !(a < b && b <= n) {
            return Err(Error::invalid(format!(
                "trial {k}: the rows {a} to {b} are not a span of the {n} rows"
            )));
        }
        if data.id[a] != data.id[b - 1] {
            return Err(Error::invalid(format!(
                "trial {k}: the rows {a} to {b} have more than one person"
            )));
        }
        if k > 0 && a < trials.start[k - 1] {
            return Err(Error::invalid(format!(
                "trial {k}: the trials must be ordered by their first row"
            )));
        }
    }
    let mut cumulative = vec![0.0; n];
    for i in 0..n {
        cumulative[i] = if i > 0 && data.id[i] == data.id[i - 1] {
            cumulative[i - 1] * row_weight[i]
        } else {
            row_weight[i]
        };
    }
    let stack = Stack::new(
        trials.start.len(),
        |k| trials.end[k] - trials.start[k],
        |k| data.id[trials.start[k]],
    );
    let rows = TrialStack {
        trials,
        outcome,
        cumulative,
        person: data.id,
        columns,
        design,
        stack,
    };
    fit_msm_rows(&rows, columns)
}

/// Where each block of rows of a stack of spans starts: the span, the row in
/// it and the cluster of that row, with the number of rows and of clusters.
struct Stack {
    blocks: Vec<(usize, usize, usize)>,
    rows: usize,
    clusters: usize,
}

impl Stack {
    /// The stack of `spans` spans of `length(k)` rows each, with the clusters
    /// numbered in the order in which the `label` of a span first appears, the
    /// labels of a cluster being next to each other.
    fn new(spans: usize, length: impl Fn(usize) -> usize, label: impl Fn(usize) -> u64) -> Self {
        let mut blocks = Vec::new();
        let (mut rows, mut clusters) = (0_usize, 0_usize);
        let mut next = 0_usize;
        let mut last: Option<u64> = None;
        for k in 0..spans {
            let len = length(k);
            if len == 0 {
                continue;
            }
            if last != Some(label(k)) {
                clusters += 1;
                last = Some(label(k));
            }
            while next < rows + len {
                blocks.push((k, next - rows, clusters - 1));
                next += ROW_BLOCK;
            }
            rows += len;
        }
        Self {
            blocks,
            rows,
            clusters,
        }
    }

    /// The span, row in it and cluster of the first row of the block that
    /// starts at `start`, a multiple of [`ROW_BLOCK`].
    fn at(&self, start: usize) -> (usize, usize, usize) {
        assert_eq!(
            start % ROW_BLOCK,
            0,
            "a block starts at a multiple of ROW_BLOCK"
        );
        self.blocks[start / ROW_BLOCK]
    }
}

/// The rows of the trials, formed from the data.
struct TrialStack<'a> {
    trials: &'a TrialSpans<'a>,
    outcome: &'a [bool],
    cumulative: Vec<f64>,
    person: &'a [u64],
    columns: usize,
    design: &'a (dyn Fn(usize, usize, &mut [f64]) + Sync),
    stack: Stack,
}

impl Rows for TrialStack<'_> {
    fn n_rows(&self) -> usize {
        self.stack.rows
    }

    fn n_columns(&self) -> usize {
        self.columns
    }

    fn fill(&self, start: usize, mut block: Block<'_>) {
        let (mut k, mut offset, mut cluster) = self.stack.at(start);
        let mut values = vec![0.0; self.columns];
        for i in 0..block.y.len() {
            while offset == self.trials.end[k] - self.trials.start[k] {
                let next = k + 1;
                if self.person[self.trials.start[next]] != self.person[self.trials.start[k]] {
                    cluster += 1;
                }
                (k, offset) = (next, 0);
            }
            let first = self.trials.start[k];
            let row = first + offset;
            (self.design)(first, row, &mut values);
            for (j, &v) in values.iter().enumerate() {
                block.x[(i, j)] = v;
            }
            block.y[i] = f64::from(u8::from(self.outcome[row]));
            block.weight[i] = self.cumulative[row] / self.cumulative[first];
            block.cluster[i] = cluster;
            offset += 1;
        }
    }
}

/// [`fit_msm`] on rows formed by `rows`, each clustered by the person its
/// cluster numbers, with the number of persons `rows` has.
pub(crate) fn fit_msm_rows(rows: &(impl Rows + Clustered), columns: usize) -> Result<Msm> {
    if columns == 0 {
        return Err(Error::invalid("design has no columns"));
    }
    let n = rows.n_rows();
    let finite = (0..n.div_ceil(ROW_BLOCK)).into_par_iter().all(|b| {
        let start = b * ROW_BLOCK;
        let len = ROW_BLOCK.min(n - start);
        let mut x = Mat::<f64>::zeros(len, columns);
        let (mut y, mut w, mut c) = (vec![0.0; len], vec![0.0; len], vec![0; len]);
        rows.fill(
            start,
            Block {
                x: x.as_mut(),
                y: &mut y,
                weight: &mut w,
                cluster: &mut c,
            },
        );
        epistat_linalg::is_finite(x.as_ref())
    });
    if !finite {
        return Err(Error::invalid("design: a value is not finite"));
    }
    let model = Glm::new(Family::Binomial, Link::Logit);
    let fit = model
        .fit_rows(rows)
        .map_err(Error::model("outcome model"))?;
    let coefficients = crate::logistic::full_rank(&fit, "outcome model")?;
    let cov = fit
        .robust_covariance(rows, true, SandwichType::Cr1)
        .map_err(Error::model("outcome model"))?;
    let g = rows.n_clusters();
    if g < 2 {
        return Err(Error::invalid(
            "outcome model: at least two persons are needed",
        ));
    }
    if cov.as_slice().iter().any(Option::is_none) {
        return Err(Error::undefined(
            "outcome model: the covariance is not defined",
        ));
    }
    Ok(Msm {
        coefficients,
        covariance: cov,
        n_rows: n,
        n_persons: g,
    })
}

/// Rows whose clusters are counted in advance.
pub(crate) trait Clustered {
    /// The number of clusters among the rows.
    fn n_clusters(&self) -> usize;
}

impl Clustered for TrialStack<'_> {
    fn n_clusters(&self) -> usize {
        self.stack.clusters
    }
}

/// The cumulative incidence at each follow-up time, with its standard error.
#[derive(Debug, Clone, PartialEq)]
pub struct CumulativeIncidence {
    /// `1 - mean(prod over s <= t of (1 - p_s))` at each time.
    pub risk: Vec<f64>,
    /// The delta-method standard error at each time.
    pub se: Vec<f64>,
}

/// The cumulative incidence under one treatment from the model, for a
/// population of rows: `designs[t]` has the design of every row at follow-up
/// time `t`, with the assigned treatment and the time set, and the same rows in
/// the same order for every time.
///
/// [`cumulative_incidence_rows`] gives the same, bit for bit, from a function
/// that writes the design of a row at a time, without a matrix per time.
///
/// # Errors
///
/// An input error for no time, designs with another number of columns than the
/// model or of different numbers of rows, no row, and a covariance that is not
/// `p` by `p` or has an undefined entry.
pub fn cumulative_incidence(msm: &Msm, designs: &[MatRef<'_, f64>]) -> Result<CumulativeIncidence> {
    let p = msm.coefficients.len();
    let Some(first) = designs.first() else {
        return Err(Error::invalid("designs: no times"));
    };
    let rows = first.nrows();
    for (t, d) in designs.iter().enumerate() {
        if d.nrows() != rows || d.ncols() != p {
            return Err(Error::invalid(format!(
                "designs[{t}]: {} by {} where {rows} by {p} is needed",
                d.nrows(),
                d.ncols()
            )));
        }
    }
    cumulative_incidence_rows(msm, rows, designs.len(), &|r, t, out| {
        for (j, v) in out.iter_mut().enumerate() {
            *v = designs[t][(r, j)];
        }
    })
}

/// [`cumulative_incidence`] with the designs formed by `design(row, time,
/// out)`, which writes into `out` the design of row `row` of the population at
/// follow-up time `time` (from 0 to `times - 1`), with the assigned treatment
/// and the time set. The memory is one design row and the sums of each time.
///
/// The risk at time `t` is `1 - mean over rows of S_r(t)`, `S_r(t) = prod over
/// s <= t of (1 - p_r(s))`, with the delta-method standard error of the module
/// documentation. The sum over rows is taken in the order of the rows, so the
/// result does not depend on the number of threads and equals that of
/// [`cumulative_incidence`] on the same designs.
///
/// # Errors
///
/// An input error for no time, no row, a design value that is not finite,
/// and a covariance that is not `p` by `p` or has an undefined entry.
pub fn cumulative_incidence_rows(
    msm: &Msm,
    rows: usize,
    times: usize,
    design: &(dyn Fn(usize, usize, &mut [f64]) + Sync),
) -> Result<CumulativeIncidence> {
    let p = msm.coefficients.len();
    if times == 0 {
        return Err(Error::invalid("designs: no times"));
    }
    if rows == 0 {
        return Err(Error::from(InputError::Empty { what: "designs" }));
    }
    if msm.covariance.dim() != p {
        return Err(Error::invalid("msm: the covariance is not p by p"));
    }
    let covariance = dense(&msm.covariance, "msm")?;
    let mut mean = vec![0.0; times];
    let mut grad = vec![0.0; times * p];
    let mut x = vec![0.0; p];
    let mut dlog = vec![0.0; p];
    for r in 0..rows {
        // The running product of survival and the running sum of p_s x_s of
        // the row.
        let mut surv = 1.0;
        dlog.fill(0.0);
        for t in 0..times {
            design(r, t, &mut x);
            if x.iter().any(|v| !v.is_finite()) {
                return Err(Error::invalid(format!(
                    "design: a value of row {r} at time {t} is not finite"
                )));
            }
            let eta: f64 = (0..p).map(|j| x[j] * msm.coefficients[j]).sum();
            let prob = Link::Logit.inverse(eta);
            surv *= 1.0 - prob;
            for j in 0..p {
                dlog[j] += prob * x[j];
            }
            mean[t] += surv;
            for j in 0..p {
                grad[t * p + j] += surv * dlog[j];
            }
        }
    }
    let nf = rows as f64;
    let mut risk = Vec::with_capacity(times);
    let mut std_err = Vec::with_capacity(times);
    for t in 0..times {
        risk.push(1.0 - mean[t] / nf);
        // The risk is 1 - mean S, so its derivative is + mean(S dlog).
        let g: Vec<f64> = grad[t * p..(t + 1) * p].iter().map(|v| v / nf).collect();
        std_err.push(quadratic_form(&g, &covariance).max(0.0).sqrt());
    }
    Ok(CumulativeIncidence { risk, se: std_err })
}

/// The entries of `covariance`, row-major, refusing an undefined one.
pub(crate) fn dense(covariance: &Covariance, what: &str) -> Result<Vec<f64>> {
    covariance
        .as_slice()
        .iter()
        .map(|v| v.ok_or_else(|| Error::invalid(format!("{what}: the covariance is not defined"))))
        .collect()
}

/// `g' V g` for a `p * p` matrix `V` in row order, summed as `a` then `b`.
pub(crate) fn quadratic_form(g: &[f64], v: &[f64]) -> f64 {
    let p = g.len();
    let mut var = 0.0;
    for a in 0..p {
        for b in 0..p {
            var += g[a] * v[a * p + b] * g[b];
        }
    }
    var
}
