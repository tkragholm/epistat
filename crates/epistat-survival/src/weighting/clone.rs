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
//! - for [`Strategy::InitiateWithin`] with `grace = m`: the person has not
//!   started treatment before the row, `a_j = 0` and `j = m` (the period `m` is
//!   the last in which starting is allowed); or the person has started, the
//!   adherence is [`Adherence::Continuous`] and `a_j = 0`;
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
//! for `InitiateWithin` in the period `m` before treatment is started and in every
//! period after it with continuous adherence, for `NeverInitiate` in every period
//! it covers. The deviation in a row at risk is a binary variable `d_j`. For
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
//! starting of treatment and the outcome.
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
//! that forms the design of a clone row. [`bootstrap_persons`]
//! gives the person bootstrap that refits everything, which is the variance to
//! report.
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

use epistat_glm::rows::{Block, Rows};
use epistat_glm::{Family, Glm, Link};
use epistat_linalg::{Mat, MatRef, ROW_BLOCK};
use rayon::prelude::*;

use crate::error::{Error, Result};

/// Whether a person who has started treatment must stay on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Adherence {
    /// Stopping treatment is a deviation.
    Continuous,
    /// Treatment after the start is not constrained.
    None,
}

/// A treatment strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Start treatment in one of the periods `0..=grace`, and then follow
    /// `adherence`.
    InitiateWithin {
        /// The last period in which treatment may be started.
        grace: u32,
        /// The adherence after the start.
        adherence: Adherence,
    },
    /// Do not start treatment, in any period or up to and including `until`.
    NeverInitiate {
        /// The last period in which treatment must not be given, `None` for all.
        until: Option<u32>,
    },
}

/// A person-period data set. The rows of a person are next to each other, in
/// order, and start at period 0.
#[derive(Debug, Clone, Copy)]
pub struct CloneInput<'a> {
    /// The person of each row.
    pub id: &'a [u32],
    /// The period of each row, 0, 1, 2, ... within a person.
    pub period: &'a [u32],
    /// Whether the person is treated at the start of the period.
    pub treatment: &'a [bool],
    /// Whether the outcome happens in the period.
    pub outcome: &'a [bool],
}

/// The clones of all persons under all strategies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cloned {
    /// The row of the input each clone row is made from, for building the
    /// designs of the models.
    pub source: Vec<usize>,
    /// The strategy of each clone row, an index into the strategies.
    pub arm: Vec<usize>,
    /// The person of each clone row, numbered from 0 in the order of the input.
    pub person: Vec<u32>,
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
}

/// Clone every person under every strategy and censor each clone at its first
/// deviation.
///
/// # Errors
///
/// [`Error::Input`] for columns of different lengths, no strategy, rows that
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
            return Err(Error::input(format!("{name}: {len} values for {n} rows")));
        }
    }
    if strategies.is_empty() {
        return Err(Error::input("strategies: at least one is needed"));
    }
    let mut n_persons = 0usize;
    let mut starts = Vec::new();
    for i in 0..n {
        if i == 0 || input.id[i] != input.id[i - 1] {
            if input.period[i] != 0 {
                return Err(Error::input(format!(
                    "period: person {} does not start at period 0 in row {i}",
                    input.id[i]
                )));
            }
            if i > 0 && input.id[i] < input.id[i - 1] {
                return Err(Error::input(format!(
                    "id: the rows of a person must be next to each other and ordered, row {i}"
                )));
            }
            starts.push(i);
            n_persons += 1;
        } else if input.period[i] != input.period[i - 1] + 1 {
            return Err(Error::input(format!(
                "period: row {i} does not follow the row before it"
            )));
        }
    }
    starts.push(n);
    let longest = starts.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    for s in strategies {
        let reach = match *s {
            Strategy::InitiateWithin { grace, .. } => Some(grace),
            Strategy::NeverInitiate { until } => until,
        };
        if let Some(r) = reach
            && r as usize >= longest
        {
            return Err(Error::input(format!(
                "strategy {s:?}: period {r} is beyond the longest follow-up of {longest} periods"
            )));
        }
    }
    let mut out = Cloned {
        source: Vec::new(),
        arm: Vec::new(),
        person: Vec::new(),
        deviation: Vec::new(),
        at_risk: Vec::new(),
        outcome: Vec::new(),
        n_persons,
        n_without_follow_up: 0,
    };
    for (p, w) in starts.windows(2).enumerate() {
        let mut observed = false;
        for (k, strategy) in strategies.iter().enumerate() {
            let mut started = false;
            for row in w[0]..w[1] {
                let j = input.period[row];
                let a = input.treatment[row];
                let (deviates, risk) = match *strategy {
                    Strategy::InitiateWithin { grace, adherence } => {
                        if started {
                            let risk = adherence == Adherence::Continuous;
                            (risk && !a, risk)
                        } else if j == grace {
                            (!a, true)
                        } else {
                            (false, false)
                        }
                    }
                    Strategy::NeverInitiate { until } => {
                        let risk = until.is_none_or(|u| j <= u);
                        (risk && a, risk)
                    }
                };
                out.source.push(row);
                out.arm.push(k);
                out.person.push(p as u32);
                out.deviation.push(deviates);
                out.at_risk.push(risk);
                let event = !deviates && input.outcome[row];
                out.outcome.push(event);
                if deviates {
                    break;
                }
                observed = true;
                started |= a;
                if event {
                    break;
                }
            }
        }
        if !observed {
            out.n_without_follow_up += 1;
        }
    }
    Ok(out)
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
        return Err(Error::input(format!(
            "{what}: the rows at risk need both deviations and rows that are not"
        )));
    }
    let xs = subset(x, rows);
    let ys: Vec<f64> = rows.iter().map(|&r| f64::from(u8::from(d[r]))).collect();
    let ws: Option<Vec<f64>> = case.map(|c| rows.iter().map(|&r| c[r]).collect());
    let fit = Glm::new(Family::Binomial, Link::Logit)
        .fit(xs.as_ref(), &ys, ws.as_deref(), None, None)
        .map_err(|e| Error::input(format!("{what}: {e}")))?;
    if !fit.converged {
        return Err(Error::input(format!("{what}: no convergence")));
    }
    let mut coef = Vec::with_capacity(fit.coefficients.len());
    for (j, c) in fit.coefficients.iter().enumerate() {
        coef.push(c.ok_or_else(|| Error::input(format!("{what}: design column {j} is aliased")))?);
    }
    Ok(coef)
}

fn probability(x: MatRef<'_, f64>, coef: &[f64], row: usize) -> f64 {
    let eta: f64 = (0..coef.len()).map(|j| x[(row, j)] * coef[j]).sum();
    1.0 / (1.0 + (-eta).exp())
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
/// [`Error::Input`] for models or designs that do not match the clones, a
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
        return Err(Error::input(format!(
            "models: {} for {n_arms} strategies",
            models.len()
        )));
    }
    if let Some(c) = case_weight
        && (c.len() != n || c.iter().any(|v| !(v.is_finite() && *v >= 0.0)))
    {
        return Err(Error::input(
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
                    return Err(Error::input(format!(
                        "strategy {k} {name}: {} rows and {} columns for {n} clone rows",
                        m.nrows(),
                        m.ncols()
                    )));
                }
                if !epistat_linalg::is_finite(m) {
                    return Err(Error::input(format!(
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
/// [`Error::Input`] for a `weight` of another length than the clone rows,
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
        return Err(Error::input(format!(
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
            return Err(Error::input(format!(
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
            return Err(Error::input(format!(
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

/// Apply `analysis` to resamples of the persons: for each seed the number of
/// times each person is drawn in a resample of the same size with replacement,
/// as case weights of the persons, which `analysis` turns into case weights of
/// the clone rows and refits the weights and the outcome with.
///
/// The counts of a replicate depend on its seed only, so the results do not depend
/// on the number of threads. The results are in the order of the seeds.
pub fn bootstrap_persons<T: Send>(
    n_persons: usize,
    seeds: &[u64],
    analysis: impl Fn(&[u32]) -> Result<T> + Sync,
) -> Vec<Result<T>> {
    seeds
        .par_iter()
        .map(|&seed| analysis(&epistat_rng::resample_counts(n_persons, seed)))
        .collect()
}

/// The case weight of each clone row from the counts of the persons.
#[must_use]
pub fn row_counts(cloned: &Cloned, counts: &[u32]) -> Vec<f64> {
    cloned
        .person
        .iter()
        .map(|&p| f64::from(counts[p as usize]))
        .collect()
}
