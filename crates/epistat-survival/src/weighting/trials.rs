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

use epistat_glm::robust::{SandwichType, robust_covariance};
use epistat_glm::{Family, Glm, Link};
use epistat_inference::Clusters;
use epistat_linalg::{Mat, MatRef};

use crate::error::{Error, Result};

/// The rows of a person-period data set, ordered by person and then period.
#[derive(Debug, Clone, Copy)]
pub struct PersonPeriod<'a> {
    /// The person of each row. The rows of a person are next to each other.
    pub id: &'a [u32],
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
                return Err(Error::input(format!("{name}: {len} values for {n} rows")));
            }
        }
        for i in 1..n {
            if self.id[i] == self.id[i - 1] && self.period[i] <= self.period[i - 1] {
                return Err(Error::input(format!(
                    "period: row {i} is not after the row before it for person {}",
                    self.id[i]
                )));
            }
            if self.id[i] < self.id[i - 1] {
                return Err(Error::input(format!(
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
pub enum Fit {
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

fn subset(x: MatRef<'_, f64>, rows: &[usize]) -> Mat<f64> {
    Mat::from_fn(rows.len(), x.ncols(), |i, j| x[(rows[i], j)])
}

fn check_design(name: &str, x: MatRef<'_, f64>, n: usize) -> Result<()> {
    if x.nrows() != n {
        return Err(Error::input(format!(
            "{name}: {} rows for {n} rows of data",
            x.nrows()
        )));
    }
    if x.ncols() == 0 {
        return Err(Error::input(format!("{name}: no columns")));
    }
    if !epistat_linalg::is_finite(x) {
        return Err(Error::input(format!("{name}: a value is not finite")));
    }
    Ok(())
}

/// Fit a logistic model of `y` on `x` over `rows` and give the coefficients and
/// the fitted probability of each of the rows.
fn fit_logistic(
    x: MatRef<'_, f64>,
    y: &[bool],
    rows: &[usize],
    what: &str,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let positives = rows.iter().filter(|&&r| y[r]).count();
    if positives == 0 || positives == rows.len() {
        return Err(Error::input(format!(
            "{what}: the rows of the fit need both values of the response"
        )));
    }
    let xs = subset(x, rows);
    let ys: Vec<f64> = rows.iter().map(|&r| f64::from(u8::from(y[r]))).collect();
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit(xs.as_ref(), &ys, None, None, None)
        .map_err(|e| Error::input(format!("{what}: {e}")))?;
    if !fit.converged {
        return Err(Error::input(format!("{what}: no convergence")));
    }
    let mut coefficients = Vec::with_capacity(fit.coefficients.len());
    for (j, c) in fit.coefficients.iter().enumerate() {
        coefficients.push(
            c.ok_or_else(|| Error::input(format!("{what}: design column {j} is aliased")))?,
        );
    }
    Ok((coefficients, fit.fitted))
}

/// The weights `p_n / p_d` of the value observed (`observed = true`) or of the
/// value 1 of the response.
fn ratio_weights(
    previous: &[bool],
    y: &[bool],
    model: &WeightModel<'_>,
    fit: Fit,
    observed_value: bool,
    what: &str,
) -> Result<RowWeights> {
    let n = y.len();
    check_design("numerator", model.numerator, n)?;
    check_design("denominator", model.denominator, n)?;
    let groups: Vec<Vec<usize>> = match fit {
        Fit::Pooled => vec![(0..n).collect()],
        Fit::ByPreviousTreatment => (0..2)
            .map(|g| (0..n).filter(|&r| previous[r] == (g == 1)).collect())
            .collect(),
    };
    let mut p_n = vec![f64::NAN; n];
    let mut p_d = vec![f64::NAN; n];
    let mut numerator_coefficients = Vec::new();
    let mut denominator_coefficients = Vec::new();
    for (g, rows) in groups.iter().enumerate() {
        if rows.is_empty() {
            return Err(Error::input(format!(
                "{what}: no rows with previous treatment {g}"
            )));
        }
        let (cn, pn) = fit_logistic(model.numerator, y, rows, &format!("{what} numerator {g}"))?;
        let (cd, pd) = fit_logistic(
            model.denominator,
            y,
            rows,
            &format!("{what} denominator {g}"),
        )?;
        for (k, &r) in rows.iter().enumerate() {
            p_n[r] = pn[k];
            p_d[r] = pd[k];
        }
        numerator_coefficients.push(cn);
        denominator_coefficients.push(cd);
    }
    let weight = (0..n)
        .map(|r| {
            if !observed_value || y[r] {
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
/// # Errors
///
/// [`Error::Input`] for rows that are not ordered by person and period, for
/// slices or designs of another length than the rows, a design with no column or
/// a value that is not finite, a group of the previous treatment without rows or
/// without both treated and untreated rows, a model that has an aliased column
/// or does not converge.
pub fn switch_weights(data: &PersonPeriod<'_>, model: &WeightModel<'_>) -> Result<RowWeights> {
    data.check()?;
    ratio_weights(
        data.previous,
        data.treatment,
        model,
        Fit::ByPreviousTreatment,
        true,
        "switch",
    )
}

/// The weights of censoring of each row, `p_n / p_d` with the probability of
/// remaining uncensored from the models of the numerator and the denominator,
/// fitted by the previous treatment or on all rows.
///
/// `censored` marks the rows of persons who are censored in the period. The
/// rows are all used in the fit, so that a data set from which the rows after a
/// deviation have been removed gives the weights of `TrialEmulation`.
///
/// # Errors
///
/// As [`switch_weights`], with `censored` of another length than the rows.
pub fn censoring_weights(
    data: &PersonPeriod<'_>,
    censored: &[bool],
    model: &WeightModel<'_>,
    fit: Fit,
) -> Result<RowWeights> {
    let n = data.check()?;
    if censored.len() != n {
        return Err(Error::input(format!(
            "censored: {} values for {n} rows",
            censored.len()
        )));
    }
    let uncensored: Vec<bool> = censored.iter().map(|&c| !c).collect();
    ratio_weights(data.previous, &uncensored, model, fit, false, "censoring")
}

/// The rows of the stacked trials: the person, the period the trial started and
/// the follow-up time of each row.
#[derive(Debug, Clone, Copy)]
pub struct TrialRows<'a> {
    /// The person of each row.
    pub id: &'a [u32],
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
/// [`Error::Input`] for rows or weights that do not match, and for a row of a
/// trial whose person has no row in the period of the start of the trial or of
/// the row.
pub fn trial_weights(
    data: &PersonPeriod<'_>,
    row_weight: &[f64],
    rows: &TrialRows<'_>,
) -> Result<Vec<f64>> {
    let n = data.check()?;
    if row_weight.len() != n {
        return Err(Error::input(format!(
            "row_weight: {} values for {n} rows",
            row_weight.len()
        )));
    }
    let m = rows.id.len();
    if rows.trial_period.len() != m || rows.follow_up.len() != m {
        return Err(Error::input("trial rows: columns of different lengths"));
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
    let mut starts: Vec<(u32, usize, usize)> = Vec::new();
    for i in 0..n {
        if i == 0 || data.id[i] != data.id[i - 1] {
            starts.push((data.id[i], i, i + 1));
        } else if let Some(last) = starts.last_mut() {
            last.2 = i + 1;
        }
    }
    let find = |id: u32, period: i64| -> Option<usize> {
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
                return Err(Error::input(format!(
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
    pub person: &'a [u32],
}

/// The outcome model of the stacked trials.
#[derive(Debug, Clone, PartialEq)]
pub struct Msm {
    /// The coefficients.
    pub coefficients: Vec<f64>,
    /// The covariance of the coefficients clustered by person, `p * p` values.
    pub covariance: Vec<f64>,
    /// The number of rows.
    pub n_rows: usize,
    /// The number of persons.
    pub n_persons: usize,
}

impl Msm {
    /// The robust standard errors of the coefficients.
    #[must_use]
    pub fn std_err(&self) -> Vec<f64> {
        let p = self.coefficients.len();
        (0..p)
            .map(|i| self.covariance[i * p + i].max(0.0).sqrt())
            .collect()
    }
}

/// Fit the weighted logistic outcome model of the stacked trials, with the
/// covariance clustered by person.
///
/// # Errors
///
/// [`Error::Input`] for inputs of different lengths, a weight that is negative
/// or not finite, a model that has an aliased column or does not converge, and
/// fewer than two persons.
pub fn fit_msm(input: &MsmInput<'_>) -> Result<Msm> {
    let n = input.outcome.len();
    check_design("design", input.design, n)?;
    if input.weight.len() != n || input.person.len() != n {
        return Err(Error::input("weight and person need one value per row"));
    }
    let y: Vec<f64> = input
        .outcome
        .iter()
        .map(|&o| f64::from(u8::from(o)))
        .collect();
    let model = Glm::new(Family::Binomial, Link::Logit);
    let fit = model
        .fit(input.design, &y, Some(input.weight), None, None)
        .map_err(|e| Error::input(format!("outcome model: {e}")))?;
    if !fit.converged {
        return Err(Error::input("outcome model: no convergence"));
    }
    let p = input.design.ncols();
    let coefficients: Vec<f64> = fit
        .coefficients
        .iter()
        .enumerate()
        .map(|(j, c)| {
            c.ok_or_else(|| Error::input(format!("outcome model: design column {j} is aliased")))
        })
        .collect::<Result<_>>()?;
    let clusters = Clusters::from_labels(input.person);
    let cov = robust_covariance(
        &model,
        &fit,
        input.design,
        &y,
        Some(&clusters),
        SandwichType::Hc0,
    )
    .map_err(|e| Error::input(format!("outcome model: {e}")))?;
    let g = clusters.count();
    if g < 2 {
        return Err(Error::input("outcome model: at least two persons are needed"));
    }
    let factor = g as f64 / (g - 1) as f64;
    let covariance: Vec<f64> = cov
        .iter()
        .map(|v| {
            v.map(|v| v * factor)
                .ok_or_else(|| Error::input("outcome model: the covariance is not defined"))
        })
        .collect::<Result<_>>()?;
    debug_assert_eq!(covariance.len(), p * p);
    Ok(Msm {
        coefficients,
        covariance,
        n_rows: n,
        n_persons: clusters.count(),
    })
}

/// The cumulative incidence at each follow-up time, with its standard error.
#[derive(Debug, Clone, PartialEq)]
pub struct CumulativeIncidence {
    /// `1 - mean(prod over s <= t of (1 - p_s))` at each time.
    pub risk: Vec<f64>,
    /// The delta-method standard error at each time.
    pub std_err: Vec<f64>,
}

/// The cumulative incidence under one treatment from the model, for a
/// population of rows: `designs[t]` has the design of every row at follow-up
/// time `t`, with the assigned treatment and the time set, and the same rows in
/// the same order for every time.
///
/// # Errors
///
/// [`Error::Input`] for no time, designs with another number of columns than the
/// model or of different numbers of rows, or no row.
pub fn cumulative_incidence(
    msm: &Msm,
    designs: &[MatRef<'_, f64>],
) -> Result<CumulativeIncidence> {
    let p = msm.coefficients.len();
    let Some(first) = designs.first() else {
        return Err(Error::input("designs: no times"));
    };
    let rows = first.nrows();
    if rows == 0 {
        return Err(Error::input("designs: no rows"));
    }
    for (t, d) in designs.iter().enumerate() {
        if d.nrows() != rows || d.ncols() != p {
            return Err(Error::input(format!(
                "designs[{t}]: {} by {} where {rows} by {p} is needed",
                d.nrows(),
                d.ncols()
            )));
        }
    }
    // For each row the running product of survival and the running sum of
    // p_s x_s.
    let mut surv = vec![1.0; rows];
    let mut dlog = vec![0.0; rows * p];
    let mut risk = Vec::with_capacity(designs.len());
    let mut std_err = Vec::with_capacity(designs.len());
    for d in designs {
        let mut mean = 0.0;
        let mut grad = vec![0.0; p];
        for r in 0..rows {
            let eta: f64 = (0..p).map(|j| d[(r, j)] * msm.coefficients[j]).sum();
            let prob = 1.0 / (1.0 + (-eta).exp());
            surv[r] *= 1.0 - prob;
            for j in 0..p {
                dlog[r * p + j] += prob * d[(r, j)];
            }
            mean += surv[r];
            for j in 0..p {
                grad[j] += surv[r] * dlog[r * p + j];
            }
        }
        let nf = rows as f64;
        risk.push(1.0 - mean / nf);
        // The risk is 1 - mean S, so its derivative is + mean(S dlog).
        let g: Vec<f64> = grad.iter().map(|v| v / nf).collect();
        let mut var = 0.0;
        for a in 0..p {
            for b in 0..p {
                var += g[a] * msm.covariance[a * p + b] * g[b];
            }
        }
        std_err.push(var.max(0.0).sqrt());
    }
    Ok(CumulativeIncidence { risk, std_err })
}
