//! Effect estimates after matching.
//!
//! The functions here estimate the average effect of the treatment in a
//! matched sample from a regression of the outcome on the treatment, fitted
//! to the matched rows with the matching weights, and give its variance by the
//! delta method from a cluster-robust sandwich covariance of the coefficients.
//! The bootstrap of matched sets gives an interval that does not rest on the
//! covariance. Both follow the MatchIt vignette "Estimating effects after
//! matching", which uses `marginaleffects::avg_comparisons()` with a
//! cluster-robust variance, and the bootstrap of Austin and Small \[1\].
//!
//! # Input
//!
//! A [`MatchedSample`] is a table with one row per matched unit and pair, with
//! the treatment, the weight, the outcome, optional covariates, the pair and
//! the unit of the row. [`MatchedSample::from_match_data`] builds it from
//! [`crate::output::match_data`], whose rows are units. [`MatchedSample::from_long_matches`]
//! builds it from [`crate::output::get_matches`], whose rows are units within
//! pairs, so a control that is matched to several cases has a row in each.
//! Use that form after matching with replacement. The outcome, the treatment
//! and the covariates come from a [`UnitData`] keyed by the id of the unit,
//! so every matching method gives the same input. The cases need not be the
//! treated units: the treatment is read from the [`UnitData`], so a match for
//! the effect in the controls works the same way.
//!
//! # Method
//!
//! The outcome model is linear (identity link, Gaussian) or logistic. Its
//! design has an intercept, the treatment `A`, the covariates `X` and, with
//! interactions, the products `A X`. The model is fitted by weighted least
//! squares or iteratively reweighted least squares with the matching weights
//! `w_i` as prior weights.
//!
//! The estimand is an average over the rows of a target population, the
//! treated rows for [`Estimand::Att`], the controls for [`Estimand::Atc`] and all
//! rows for [`Estimand::Ate`]. With `x_i(a)` the design row of unit `i` with the
//! treatment set to `a`, `g` the inverse link, and weights `w_i` over the
//! target rows,
//!
//! ```text
//! m_a = sum_i w_i g(x_i(a)' b) / sum_i w_i
//! ```
//!
//! is the average prediction with everyone treated (`a = 1`) or untreated
//! (`a = 0`). The average effect is the difference `m_1 - m_0`. For a logistic
//! model the module also gives the ratio `m_1 / m_0` and the odds ratio
//! `m_1 (1 - m_0) / (m_0 (1 - m_1))`, both of the averaged predictions, each
//! with its logarithm. This is g-computation, and it is what
//! `avg_comparisons()` reports for `comparison` equal to `"difference"`,
//! `"ratio"`, `"lnratio"` and `"lnor"`.
//!
//! The covariance of the coefficients is the cluster-robust sandwich
//!
//! ```text
//! V = c A^-1 (sum_g s_g s_g') A^-1,   s_g = sum_{i in g} w_i (y_i - mu_i) x_i
//! ```
//!
//! with `A = sum_i v_i x_i x_i'`, where `v_i` is `w_i` for the linear model and
//! `w_i mu_i (1 - mu_i)` for the logistic one. The factor is
//! `c = G / (G - 1)` for `G` clusters, and for the linear model also
//! `(n - 1) / (n - p)` for `n` rows and `p` coefficients. These are the
//! defaults of `sandwich::vcovCL()`, which `marginaleffects` calls: type HC1
//! for a linear model and HC0 for any other, each with the cluster adjustment.
//! With two clustering variables the covariance is `V_a + V_b - V_ab`, each
//! term formed as above with its own number of clusters, where `V_ab`
//! clusters by the combination of the two \[3\]. The variance of an effect
//! `f(b)` is `d' V d`, with `d` its gradient in the coefficients, by the delta
//! method. A variance that is not positive, which the two-way form can give,
//! has no standard error.
//!
//! # Clustering
//!
//! [`Clustering::Pair`] clusters the rows by the pair or subclass, as the
//! vignette does after matching without replacement. After matching with
//! replacement the vignette clusters by pair and by unit,
//! [`Clustering::PairAndUnit`], since a control that is in several pairs
//! links them. [`Clustering::Unit`] is the one-way variance by unit.
//!
//! # Bootstrap
//!
//! [`bootstrap`] resamples the pairs or subclasses with replacement, the
//! cluster bootstrap that Austin and Small \[1\] recommend, keeps every row of a
//! drawn pair with its weight and refits the model. A pair drawn twice enters
//! twice. Replicate `r` counts how often each pair is drawn with
//! `epistat_rng::resample_counts` from the stream seeded with
//! `replicate_seed(seed, r)`, so the replicates do not depend on the number of
//! threads or on their order, and its rows are those of the pairs in pair
//! order, each repeated as often as it was drawn. The bootstrap standard error is the standard deviation of the
//! replicates, the normal interval is the estimate plus and minus the normal
//! quantile times that error, and the percentile interval uses R's type 7
//! quantiles of the replicates. The replicates of a statistic that is
//! undefined in any replicate give no summary for it.
//!
//! # Errors
//!
//! A fit that does not converge, an aliased design column, a target group
//! with no weight, and a replicate whose model cannot be fitted are errors and
//! are not repaired. A variance needs two clusters and is `None` with fewer.
//!
//! # References
//!
//! 1. Austin PC, Small DS. The use of bootstrapping when using
//!    propensity-score matching without replacement: a simulation study.
//!    Stat Med. 2014;33(24):4306-19.
//! 2. Abadie A, Spiess J. Robust post-matching inference. J Am Stat Assoc.
//!    2022;117(538):983-95.
//! 3. Cameron AC, Gelbach JB, Miller DL. Robust inference with multiway
//!    clustering. J Bus Econ Stat. 2011;29(2):238-49.
//! 4. Zeileis A, Koell S, Graham N. Various versatile variances: an
//!    object-oriented implementation of clustered covariances in R. J Stat
//!    Softw. 2020;95(1):1-36.

#![expect(
    deprecated,
    reason = "the fits take their data as separate arguments, the epistat-glm 0.2 form"
)]

use crate::Error;
use epistat_core::Estimand;
use std::collections::HashMap;

use epistat_core::{Bootstrap, Interval, Level};
use epistat_glm::{Family, Glm, Link, expit};
use epistat_inference::Clusters;
use epistat_inference::interval::{self, Percentile};
use epistat_inference::sandwich::{meat, sandwich};
use epistat_linalg::{Attempt, Mat, row_scaled_gram};
use epistat_rng::{Stream, replicate_seed, resample_counts};
use rayon::prelude::*;

use crate::output::{LongMatches, MatchData};

/// The outcome model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeModel {
    /// Linear regression, fitted by weighted least squares.
    Linear,
    /// Logistic regression for an outcome in `[0, 1]`.
    Logistic,
}

/// What the cluster-robust variance clusters by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clustering {
    /// The pair or subclass, for a match without replacement.
    Pair,
    /// The unit.
    Unit,
    /// The pair and the unit, two-way, for a match with replacement.
    PairAndUnit,
}

/// The model and the variance of an effect estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectSpec {
    /// The outcome model.
    pub model: OutcomeModel,
    /// The rows the effect is averaged over.
    pub target: Estimand,
    /// Whether the model has the products of the treatment and the
    /// covariates.
    pub interactions: bool,
    /// The clusters of the variance.
    pub clustering: Clustering,
}

impl EffectSpec {
    /// A model of the outcome, the effect in the treated and the variance by
    /// pair, without interactions.
    #[must_use]
    pub const fn new(model: OutcomeModel) -> Self {
        Self {
            model,
            target: Estimand::Att,
            interactions: false,
            clustering: Clustering::Pair,
        }
    }
}

/// The treatment, the outcome and the covariates of every unit, keyed by id.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitData {
    index: HashMap<String, usize>,
    treated: Vec<bool>,
    outcome: Vec<f64>,
    covariate_names: Vec<String>,
    covariates: Vec<Vec<f64>>,
}

impl UnitData {
    /// The units with their treatment and outcome.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the lists differ in length,
    /// [`Error::DuplicateId`] for a repeated id and
    /// [`Error::Value`] for an outcome that is not finite.
    pub fn new(ids: Vec<String>, treated: Vec<bool>, outcome: Vec<f64>) -> Result<Self, Error> {
        check_length("treated", ids.len(), treated.len())?;
        check_length("outcome", ids.len(), outcome.len())?;
        if let Some(row) = outcome.iter().position(|v| !v.is_finite()) {
            return Err(Error::Value {
                what: "outcome",
                row,
            });
        }
        let mut index = HashMap::with_capacity(ids.len());
        for (row, id) in ids.into_iter().enumerate() {
            if index.contains_key(&id) {
                return Err(Error::DuplicateId { id });
            }
            index.insert(id, row);
        }
        Ok(Self {
            index,
            treated,
            outcome,
            covariate_names: Vec::new(),
            covariates: Vec::new(),
        })
    }

    /// Adds a covariate, one value per unit in the order of the ids.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for a column of the wrong length or [`Error::Value`].
    pub fn with_covariate(
        mut self,
        name: impl Into<String>,
        values: Vec<f64>,
    ) -> Result<Self, Error> {
        check_length("covariate", self.treated.len(), values.len())?;
        if let Some(row) = values.iter().position(|v| !v.is_finite()) {
            return Err(Error::Value {
                what: "covariate",
                row,
            });
        }
        self.covariate_names.push(name.into());
        self.covariates.push(values);
        Ok(self)
    }

    /// The treatment and outcome of the unit with this id.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownUnit`] when there is no such unit.
    pub fn treatment_and_outcome(&self, id: &str) -> Result<(bool, f64), Error> {
        let row = self.row_of(id)?;
        Ok((self.treated[row], self.outcome[row]))
    }

    pub(crate) fn treated_flags(&self) -> &[bool] {
        &self.treated
    }

    pub(crate) fn outcomes(&self) -> &[f64] {
        &self.outcome
    }

    pub(crate) fn covariate_columns(&self) -> &[Vec<f64>] {
        &self.covariates
    }

    /// The treatment and outcome of the unit in row `row`.
    pub(crate) fn treatment_and_outcome_at(&self, row: usize) -> (bool, f64) {
        (self.treated[row], self.outcome[row])
    }

    /// The row of a unit.
    pub(crate) fn row_of(&self, id: &str) -> Result<usize, Error> {
        self.index
            .get(id)
            .copied()
            .ok_or_else(|| Error::UnknownUnit { id: id.to_string() })
    }
}

/// The rows of a matched sample.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchedSample {
    treated: Vec<bool>,
    weight: Vec<f64>,
    outcome: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    pair: Vec<Option<usize>>,
    unit: Vec<usize>,
}

impl MatchedSample {
    /// A sample from its columns. `unit` numbers the units, so that rows of
    /// the same unit share a number, and `pair` the pairs or subclasses, `None`
    /// for a row in no pair. `covariates` holds one column per covariate.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] for columns of different length and
    /// [`Error::Value`] for a weight that is negative or not finite, or
    /// an outcome or covariate that is not finite.
    pub fn new(
        treated: Vec<bool>,
        weight: Vec<f64>,
        outcome: Vec<f64>,
        covariates: Vec<Vec<f64>>,
        pair: Vec<Option<usize>>,
        unit: Vec<usize>,
    ) -> Result<Self, Error> {
        let n = treated.len();
        check_length("weight", n, weight.len())?;
        check_length("outcome", n, outcome.len())?;
        check_length("pair", n, pair.len())?;
        check_length("unit", n, unit.len())?;
        for column in &covariates {
            check_length("covariate", n, column.len())?;
            if let Some(row) = column.iter().position(|v| !v.is_finite()) {
                return Err(Error::Value {
                    what: "covariate",
                    row,
                });
            }
        }
        if let Some(row) = weight.iter().position(|w| !(w.is_finite() && *w >= 0.0)) {
            return Err(Error::Value {
                what: "weight",
                row,
            });
        }
        if let Some(row) = outcome.iter().position(|v| !v.is_finite()) {
            return Err(Error::Value {
                what: "outcome",
                row,
            });
        }
        Ok(Self {
            treated,
            weight,
            outcome,
            covariates,
            pair,
            unit,
        })
    }

    /// The sample of [`crate::output::match_data`]. Rows with weight 0 are
    /// left out. The pair is the subclass of the row, which is `None` for
    /// every row after matching with replacement.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownUnit`] for a unit not in `units`, and those of
    /// [`MatchedSample::new`].
    pub fn from_match_data(data: &MatchData, units: &UnitData) -> Result<Self, Error> {
        let keep: Vec<usize> = (0..data.len()).filter(|&i| data.weight[i] > 0.0).collect();
        let rows = keep
            .iter()
            .map(|&i| units.row_of(&data.id[i]))
            .collect::<Result<Vec<_>, _>>()?;
        let pair = keep.iter().map(|&i| data.subclass[i]).collect();
        let weight = keep.iter().map(|&i| data.weight[i]).collect();
        Self::from_unit_rows(&rows, weight, pair, units)
    }

    /// The sample of [`crate::output::get_matches`], one row per unit and
    /// pair. Rows with weight 0 are left out.
    ///
    /// # Errors
    ///
    /// As [`MatchedSample::from_match_data`].
    pub fn from_long_matches(data: &LongMatches, units: &UnitData) -> Result<Self, Error> {
        let keep: Vec<usize> = (0..data.len()).filter(|&i| data.weight[i] > 0.0).collect();
        let rows = keep
            .iter()
            .map(|&i| units.row_of(&data.id[i]))
            .collect::<Result<Vec<_>, _>>()?;
        let pair = keep.iter().map(|&i| data.pair_id[i]).collect();
        let weight = keep.iter().map(|&i| data.weight[i]).collect();
        Self::from_unit_rows(&rows, weight, pair, units)
    }

    fn from_unit_rows(
        rows: &[usize],
        weight: Vec<f64>,
        pair: Vec<Option<usize>>,
        units: &UnitData,
    ) -> Result<Self, Error> {
        Self::new(
            rows.iter().map(|&r| units.treated[r]).collect(),
            weight,
            rows.iter().map(|&r| units.outcome[r]).collect(),
            units
                .covariates
                .iter()
                .map(|column| rows.iter().map(|&r| column[r]).collect())
                .collect(),
            pair,
            rows.to_vec(),
        )
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.treated.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.treated.is_empty()
    }

    /// The rows at `rows`, in that order, which may repeat.
    fn select(&self, rows: &[usize]) -> Self {
        Self {
            treated: rows.iter().map(|&i| self.treated[i]).collect(),
            weight: rows.iter().map(|&i| self.weight[i]).collect(),
            outcome: rows.iter().map(|&i| self.outcome[i]).collect(),
            covariates: self
                .covariates
                .iter()
                .map(|c| rows.iter().map(|&i| c[i]).collect())
                .collect(),
            pair: rows.iter().map(|&i| self.pair[i]).collect(),
            unit: rows.iter().map(|&i| self.unit[i]).collect(),
        }
    }

    fn pair_labels(&self) -> Result<Vec<usize>, Error> {
        self.pair
            .iter()
            .enumerate()
            .map(|(row, p)| p.ok_or(Error::MissingPair { row }))
            .collect()
    }
}

/// An effect and its standard error.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Effect {
    /// The estimate.
    pub estimate: f64,
    /// The standard error by the delta method, `None` when the covariance is
    /// not defined or the variance is not positive.
    pub se: Option<f64>,
}

/// The effect estimates of one fit.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EffectEstimates {
    /// The weighted average prediction of the target with everyone treated.
    pub mean_treated: f64,
    /// The weighted average prediction of the target with everyone untreated.
    pub mean_control: f64,
    /// The difference of the two means.
    pub difference: Effect,
    /// Their ratio, for a logistic model with a positive control mean.
    pub ratio: Option<Effect>,
    /// The log of the ratio, for a logistic model with positive means.
    pub log_ratio: Option<Effect>,
    /// The odds ratio, for a logistic model with both means in `(0, 1)`.
    pub odds_ratio: Option<Effect>,
    /// The log of the odds ratio, under the same condition.
    pub log_odds_ratio: Option<Effect>,
    /// The coefficients: intercept, treatment, covariates, then the products
    /// of the treatment with the covariates.
    pub coefficients: Vec<f64>,
    /// The covariance of the coefficients, row-major, `None` with fewer than
    /// two clusters.
    pub covariance: Option<Vec<f64>>,
    /// The number of rows of the fit.
    pub rows: usize,
}

struct Fitted {
    beta: Vec<f64>,
    means: Vec<f64>,
    design: Mat<f64>,
}

fn check_length(what: &'static str, expected: usize, found: usize) -> Result<(), Error> {
    if expected == found {
        Ok(())
    } else {
        Err(Error::length(what, expected, found))
    }
}

/// The design row of `row` with the treatment set to `a`.
fn design_value(sample: &MatchedSample, interactions: bool, row: usize, a: f64, col: usize) -> f64 {
    let q = sample.covariates.len();
    match col {
        0 => 1.0,
        1 => a,
        c if c < 2 + q => sample.covariates[c - 2][row],
        c => {
            debug_assert!(interactions);
            a * sample.covariates[c - 2 - q][row]
        }
    }
}

fn width(sample: &MatchedSample, interactions: bool) -> usize {
    let q = sample.covariates.len();
    2 + q + if interactions { q } else { 0 }
}

fn design(
    sample: &MatchedSample,
    interactions: bool,
    rows: &[usize],
    force: Option<f64>,
) -> Mat<f64> {
    let p = width(sample, interactions);
    Mat::from_fn(rows.len(), p, |i, c| {
        let row = rows[i];
        let a = force.unwrap_or_else(|| f64::from(u8::from(sample.treated[row])));
        design_value(sample, interactions, row, a, c)
    })
}

fn fit_model(sample: &MatchedSample, spec: &EffectSpec) -> Result<Fitted, Error> {
    let weighted = |treated: bool| {
        sample
            .treated
            .iter()
            .zip(&sample.weight)
            .any(|(t, w)| *t == treated && *w > 0.0)
    };
    if !weighted(true) || !weighted(false) {
        return Err(Error::EmptyTarget);
    }
    let n = sample.len();
    let p = width(sample, spec.interactions);
    if n < p {
        return Err(Error::TooFewRows);
    }
    if spec.model == OutcomeModel::Logistic
        && let Some(row) = sample.outcome.iter().position(|y| !(0.0..=1.0).contains(y))
    {
        return Err(Error::Outcome { row });
    }
    let all: Vec<usize> = (0..n).collect();
    let x = design(sample, spec.interactions, &all, None);
    let (family, link) = match spec.model {
        OutcomeModel::Linear => (Family::Gaussian, Link::Identity),
        OutcomeModel::Logistic => (Family::Binomial, Link::Logit),
    };
    // A tight tolerance puts the iteration error well below the tolerances
    // of the comparison with R, whose default stops at a relative deviance
    // change of 1e-8.
    let mut glm = Glm::new(family, link);
    glm.tolerance = 1e-12;
    glm.max_iterations = 100;
    let fit = glm.fit_slices(
        x.as_ref(),
        &sample.outcome,
        Some(&sample.weight),
        None,
        None,
    )?;
    if !fit.converged {
        return Err(Error::NotConverged);
    }
    let beta = fit
        .coefficients
        .iter()
        .map(|c| c.ok_or(Error::Aliased))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Fitted {
        beta,
        means: fit.fitted,
        design: x,
    })
}

/// The mean predictions of the target under each treatment and their
/// gradients in the coefficients.
struct Means {
    m1: f64,
    m0: f64,
    g1: Vec<f64>,
    g0: Vec<f64>,
}

fn means(sample: &MatchedSample, spec: &EffectSpec, beta: &[f64]) -> Result<Means, Error> {
    let all = match spec.target {
        Estimand::Att | Estimand::Atc => false,
        Estimand::Ate => true,
        other => {
            return Err(Error::UnsupportedEstimand {
                estimand: other,
                method: "the effect estimate",
            });
        }
    };
    let rows: Vec<usize> = (0..sample.len())
        .filter(|&i| all || sample.treated[i] == (spec.target == Estimand::Att))
        .collect();
    let total: f64 = rows.iter().map(|&i| sample.weight[i]).sum();
    if total <= 0.0 {
        return Err(Error::EmptyTarget);
    }
    let p = beta.len();
    let one = |a: f64| -> (f64, Vec<f64>) {
        let x = design(sample, spec.interactions, &rows, Some(a));
        let mut mean = 0.0;
        let mut gradient = vec![0.0; p];
        for (k, &row) in rows.iter().enumerate() {
            let eta: f64 = (0..p).map(|c| x[(k, c)] * beta[c]).sum();
            let (mu, slope) = match spec.model {
                OutcomeModel::Linear => (eta, 1.0),
                OutcomeModel::Logistic => {
                    let mu = expit(eta);
                    (mu, mu * (1.0 - mu))
                }
            };
            let w = sample.weight[row];
            mean += w * mu;
            for (c, g) in gradient.iter_mut().enumerate() {
                *g += w * slope * x[(k, c)];
            }
        }
        for g in &mut gradient {
            *g /= total;
        }
        (mean / total, gradient)
    };
    let (m1, g1) = one(1.0);
    let (m0, g0) = one(0.0);
    Ok(Means { m1, m0, g1, g0 })
}

/// A point estimate and its gradient in the coefficients.
type Contrast = (f64, Vec<f64>);

/// The point estimates of the effect and their gradients in the coefficients.
struct Contrasts {
    difference: Contrast,
    ratio: Option<Contrast>,
    log_ratio: Option<Contrast>,
    odds_ratio: Option<Contrast>,
    log_odds_ratio: Option<Contrast>,
}

fn combine(a: &[f64], ka: f64, b: &[f64], kb: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| ka * x + kb * y).collect()
}

fn contrasts(m: &Means, model: OutcomeModel) -> Contrasts {
    let difference = (m.m1 - m.m0, combine(&m.g1, 1.0, &m.g0, -1.0));
    let (mut ratio, mut log_ratio, mut odds_ratio, mut log_odds_ratio) = (None, None, None, None);
    if model == OutcomeModel::Logistic {
        if m.m0 > 0.0 {
            let r = m.m1 / m.m0;
            ratio = Some((r, combine(&m.g1, 1.0 / m.m0, &m.g0, -r / m.m0)));
        }
        if m.m0 > 0.0 && m.m1 > 0.0 {
            log_ratio = Some((
                (m.m1 / m.m0).ln(),
                combine(&m.g1, 1.0 / m.m1, &m.g0, -1.0 / m.m0),
            ));
        }
        if m.m0 > 0.0 && m.m0 < 1.0 && m.m1 > 0.0 && m.m1 < 1.0 {
            let lor = (m.m1 / (1.0 - m.m1)).ln() - (m.m0 / (1.0 - m.m0)).ln();
            let d = combine(
                &m.g1,
                1.0 / (m.m1 * (1.0 - m.m1)),
                &m.g0,
                -1.0 / (m.m0 * (1.0 - m.m0)),
            );
            let or = lor.exp();
            odds_ratio = Some((or, d.iter().map(|v| or * v).collect()));
            log_odds_ratio = Some((lor, d));
        }
    }
    Contrasts {
        difference,
        ratio,
        log_ratio,
        odds_ratio,
        log_odds_ratio,
    }
}

/// The cluster-robust covariance of the coefficients, `None` with fewer than
/// two clusters in any of the clusterings.
fn covariance(
    sample: &MatchedSample,
    spec: &EffectSpec,
    fitted: &Fitted,
) -> Result<Option<Mat<f64>>, Error> {
    let n = sample.len();
    let p = fitted.beta.len();
    let pairs = match spec.clustering {
        Clustering::Unit => None,
        _ => Some(sample.pair_labels()?),
    };
    // (clusters, sign) of each term.
    let mut terms: Vec<(Clusters, f64)> = Vec::new();
    match spec.clustering {
        Clustering::Pair => {
            terms.push((
                Clusters::from_labels_first_seen(pairs.as_deref().unwrap_or(&[])),
                1.0,
            ));
        }
        Clustering::Unit => terms.push((Clusters::from_labels_first_seen(&sample.unit), 1.0)),
        Clustering::PairAndUnit => {
            let pairs = pairs.unwrap_or_default();
            let both: Vec<(usize, usize)> = pairs
                .iter()
                .copied()
                .zip(sample.unit.iter().copied())
                .collect();
            terms.push((Clusters::from_labels_first_seen(&pairs), 1.0));
            terms.push((Clusters::from_labels_first_seen(&sample.unit), 1.0));
            terms.push((Clusters::from_labels_first_seen(&both), -1.0));
        }
    }
    if terms.iter().any(|(c, _)| c.count() < 2) {
        return Ok(None);
    }
    let info_weight: Vec<f64> = (0..n)
        .map(|i| match spec.model {
            OutcomeModel::Linear => sample.weight[i],
            OutcomeModel::Logistic => sample.weight[i] * fitted.means[i] * (1.0 - fitted.means[i]),
        })
        .collect();
    let bread = row_scaled_gram(fitted.design.as_ref(), |i| info_weight[i].sqrt());
    let small_sample = match spec.model {
        OutcomeModel::Linear => {
            if n <= p {
                return Err(Error::TooFewRows);
            }
            (n as f64 - 1.0) / (n as f64 - p as f64)
        }
        OutcomeModel::Logistic => 1.0,
    };
    let mut total = Mat::<f64>::zeros(p, p);
    for (clusters, sign) in &terms {
        let g = clusters.count() as f64;
        let scale = sign * small_sample * g / (g - 1.0);
        let m = meat(
            fitted.design.as_ref(),
            |i| sample.weight[i] * (sample.outcome[i] - fitted.means[i]),
            Some(clusters),
        )?;
        let v = sandwich(bread.as_ref(), m.as_ref(), &Attempt::SYMMETRIC)?;
        for a in 0..p {
            for b in 0..p {
                total[(a, b)] += scale * v[(a, b)];
            }
        }
    }
    Ok(Some(total))
}

fn standard_error(gradient: &[f64], v: Option<&Mat<f64>>) -> Option<f64> {
    let v = v?;
    let p = gradient.len();
    let variance: f64 = (0..p)
        .map(|a| gradient[a] * (0..p).map(|b| v[(a, b)] * gradient[b]).sum::<f64>())
        .sum();
    (variance.is_finite() && variance > 0.0).then(|| variance.sqrt())
}

/// The average effect of the treatment in a matched sample and its variance.
///
/// # Errors
///
/// See [`Error`].
pub fn estimate_effects(
    sample: &MatchedSample,
    spec: &EffectSpec,
) -> Result<EffectEstimates, Error> {
    let fitted = fit_model(sample, spec)?;
    let m = means(sample, spec, &fitted.beta)?;
    let c = contrasts(&m, spec.model);
    let v = covariance(sample, spec, &fitted)?;
    let effect = |pair: &(f64, Vec<f64>)| Effect {
        estimate: pair.0,
        se: standard_error(&pair.1, v.as_ref()),
    };
    let p = fitted.beta.len();
    Ok(EffectEstimates {
        mean_treated: m.m1,
        mean_control: m.m0,
        difference: effect(&c.difference),
        ratio: c.ratio.as_ref().map(effect),
        log_ratio: c.log_ratio.as_ref().map(effect),
        odds_ratio: c.odds_ratio.as_ref().map(effect),
        log_odds_ratio: c.log_odds_ratio.as_ref().map(effect),
        coefficients: fitted.beta,
        covariance: v.map(|v| (0..p * p).map(|k| v[(k / p, k % p)]).collect()),
        rows: sample.len(),
    })
}

/// The pair draws of a bootstrap: `bootstrap.replicates` lists of
/// `n_clusters` indices in `0..n_clusters`, the list of replicate `r` drawn
/// from the stream seeded with `replicate_seed(bootstrap.seed, r)`.
///
/// [`bootstrap`] reads the same draws as counts, which
/// `epistat_rng::resample_counts` gives.
#[deprecated(
    note = "use epistat_rng::resample_counts(n_clusters, epistat_rng::replicate_seed(seed, r)), \
            the same draws as counts"
)]
#[must_use]
pub fn cluster_resamples(n_clusters: usize, bootstrap: Bootstrap) -> Vec<Vec<usize>> {
    (0..bootstrap.replicates)
        .map(|r| {
            let mut stream = Stream::seeded(replicate_seed(bootstrap.seed, r as u64));
            (0..n_clusters).map(|_| stream.below(n_clusters)).collect()
        })
        .collect()
}

/// The settings of a bootstrap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapSpec {
    /// The number of replicates and the root seed of their draws.
    pub bootstrap: Bootstrap,
    /// The level of the two intervals.
    pub level: Level,
}

impl BootstrapSpec {
    /// The replicates of `bootstrap`, intervals at `Level::alpha(0.05)`, the
    /// 95 percent level in the form R's `alpha = 0.05` gives it.
    #[must_use]
    pub fn new(bootstrap: Bootstrap) -> Self {
        Self {
            bootstrap,
            level: Level::alpha(0.05).unwrap_or_default(),
        }
    }
}

/// The bootstrap distribution of one statistic.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BootstrapSummary {
    /// The estimate in the matched sample.
    pub estimate: f64,
    /// The standard deviation of the replicates.
    pub se: f64,
    /// The estimate plus and minus the normal quantile times `se`.
    pub normal: Interval,
    /// The type 7 quantiles of the replicates at the lower and the upper
    /// probability of the level.
    pub percentile: Interval,
}

/// The values of one statistic in every replicate, with its summary.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BootstrapStatistic {
    /// The replicates in the order of their draws.
    pub replicates: Vec<f64>,
    /// The summary.
    pub summary: BootstrapSummary,
}

/// The bootstrap of the matched sets.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BootstrapEffects {
    /// The difference of the average predictions.
    pub difference: BootstrapStatistic,
    /// Their ratio, for a logistic model, when defined in every replicate and
    /// in the sample.
    pub ratio: Option<BootstrapStatistic>,
    /// The odds ratio, under the same condition.
    pub odds_ratio: Option<BootstrapStatistic>,
}

fn point_contrasts(sample: &MatchedSample, spec: &EffectSpec) -> Result<Contrasts, Error> {
    let fitted = fit_model(sample, spec)?;
    let m = means(sample, spec, &fitted.beta)?;
    Ok(contrasts(&m, spec.model))
}

/// The cluster bootstrap of the effect over pairs or subclasses.
///
/// Every row must have a pair. At least two replicates are needed, since the
/// standard error is their standard deviation. The variance settings of
/// `spec` do not enter.
///
/// # Errors
///
/// [`Error::MissingPair`], [`Error::Bootstrap`] and the errors of
/// the fit of the sample. A replicate that cannot be fitted is
/// [`Error::Replicate`].
pub fn bootstrap(
    sample: &MatchedSample,
    spec: &EffectSpec,
    settings: &BootstrapSpec,
) -> Result<BootstrapEffects, Error> {
    let labels = sample.pair_labels()?;
    let clusters = Clusters::from_sorted_labels(&labels);
    let g = clusters.count();
    if settings.bootstrap.replicates < 2 || g < 2 {
        return Err(Error::Bootstrap);
    }
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); g];
    for (row, &c) in clusters.of().iter().enumerate() {
        members[c].push(row);
    }
    let observed = point_contrasts(sample, spec)?;
    let seed = settings.bootstrap.seed;
    let points: Vec<Contrasts> = (0..settings.bootstrap.replicates)
        .into_par_iter()
        .map(|index| {
            let counts = resample_counts(g, replicate_seed(seed, index as u64));
            let rows: Vec<usize> = counts
                .iter()
                .zip(&members)
                .flat_map(|(&count, rows)| {
                    std::iter::repeat_n(rows, count as usize).flatten().copied()
                })
                .collect();
            point_contrasts(&sample.select(&rows), spec).map_err(|e| Error::Replicate {
                index,
                reason: e.to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let summarise = |estimate: f64, values: Vec<f64>| -> Result<BootstrapStatistic, Error> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let se = (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)).sqrt();
        let normal = interval::normal(estimate, se, settings.level);
        let mut sorted = values.clone();
        let percentile = interval::percentile(&mut sorted, settings.level, Percentile::Type7)?;
        Ok(BootstrapStatistic {
            replicates: values,
            summary: BootstrapSummary {
                estimate,
                se,
                normal,
                percentile,
            },
        })
    };
    let optional =
        |pick: fn(&Contrasts) -> Option<&Contrast>| -> Result<Option<BootstrapStatistic>, Error> {
            let Some(estimate) = pick(&observed) else {
                return Ok(None);
            };
            let values: Option<Vec<f64>> = points.iter().map(|c| pick(c).map(|v| v.0)).collect();
            values.map(|v| summarise(estimate.0, v)).transpose()
        };
    Ok(BootstrapEffects {
        difference: summarise(
            observed.difference.0,
            points.iter().map(|c| c.difference.0).collect(),
        )?,
        ratio: optional(|c| c.ratio.as_ref())?,
        odds_ratio: optional(|c| c.odds_ratio.as_ref())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eight pairs of one treated and one control row, outcomes chosen so that
    /// the weighted means are easy to read.
    fn pairs_sample() -> MatchedSample {
        let treated = (0..16).map(|i| i % 2 == 0).collect();
        let outcome = vec![
            3.0, 1.0, 4.0, 2.0, 5.0, 2.0, 6.0, 3.0, 3.0, 1.0, 4.0, 2.0, 5.0, 2.0, 7.0, 3.0,
        ];
        MatchedSample::new(
            treated,
            vec![1.0; 16],
            outcome,
            Vec::new(),
            (0..16).map(|i| Some(i / 2)).collect(),
            (0..16).collect(),
        )
        .unwrap()
    }

    #[test]
    fn a_bootstrap_needs_two_replicates() {
        let sample = pairs_sample();
        let spec = EffectSpec::new(OutcomeModel::Linear);
        assert_eq!(
            bootstrap(&sample, &spec, &BootstrapSpec::new(Bootstrap::new(1, 7))).unwrap_err(),
            Error::Bootstrap
        );
        let two = bootstrap(&sample, &spec, &BootstrapSpec::new(Bootstrap::new(2, 7))).unwrap();
        assert!(two.difference.summary.se.is_finite());
    }

    #[test]
    fn a_model_without_covariates_gives_the_difference_in_means() {
        let sample = pairs_sample();
        let out = estimate_effects(&sample, &EffectSpec::new(OutcomeModel::Linear)).unwrap();
        // Treated mean 4.625, control mean 2.0.
        assert!((out.difference.estimate - 2.625).abs() < 1e-12);
        assert!((out.mean_treated - 4.625).abs() < 1e-12);
        assert!(out.difference.se.is_some());
        assert!(out.ratio.is_none());
    }

    #[test]
    fn one_cluster_has_no_covariance() {
        let mut sample = pairs_sample();
        sample.pair = vec![Some(0); 16];
        let out = estimate_effects(&sample, &EffectSpec::new(OutcomeModel::Linear)).unwrap();
        assert!(out.covariance.is_none());
        assert_eq!(out.difference.se, None);
    }

    #[test]
    fn rows_without_a_pair_need_clustering_by_unit() {
        let mut sample = pairs_sample();
        sample.pair = vec![None; 16];
        let pair = estimate_effects(&sample, &EffectSpec::new(OutcomeModel::Linear));
        assert_eq!(pair.unwrap_err(), Error::MissingPair { row: 0 });
        let mut spec = EffectSpec::new(OutcomeModel::Linear);
        spec.clustering = Clustering::Unit;
        assert!(
            estimate_effects(&sample, &spec)
                .unwrap()
                .difference
                .se
                .is_some()
        );
        let boot = bootstrap(&sample, &spec, &BootstrapSpec::new(Bootstrap::new(10, 1)));
        assert_eq!(boot.unwrap_err(), Error::MissingPair { row: 0 });
    }

    #[test]
    fn a_target_without_weighted_controls_is_an_error() {
        let mut sample = pairs_sample();
        for (w, t) in sample.weight.iter_mut().zip(&sample.treated) {
            if !*t {
                *w = 0.0;
            }
        }
        let out = estimate_effects(&sample, &EffectSpec::new(OutcomeModel::Linear));
        assert_eq!(out.unwrap_err(), Error::EmptyTarget);
    }

    #[test]
    fn a_logistic_outcome_outside_the_unit_interval_is_an_error() {
        let sample = pairs_sample();
        let out = estimate_effects(&sample, &EffectSpec::new(OutcomeModel::Logistic));
        assert_eq!(out.unwrap_err(), Error::Outcome { row: 0 });
    }

    #[test]
    fn unit_data_rejects_repeated_ids_and_unknown_units_are_named() {
        let dup = UnitData::new(
            vec!["a".into(), "a".into()],
            vec![true, false],
            vec![1.0, 2.0],
        );
        assert_eq!(dup.unwrap_err(), Error::DuplicateId { id: "a".into() });
        let units = UnitData::new(vec!["a".into()], vec![true], vec![1.0]).unwrap();
        let data = MatchData {
            id: vec!["b".into()],
            role: vec![crate::UnitRole::Case],
            weight: vec![1.0],
            subclass: vec![Some(1)],
        };
        assert_eq!(
            MatchedSample::from_match_data(&data, &units).unwrap_err(),
            Error::UnknownUnit { id: "b".into() }
        );
    }

    #[test]
    #[expect(
        deprecated,
        reason = "the released draws are checked against the counts"
    )]
    fn resamples_are_the_draws_the_counts_tally() {
        let a = cluster_resamples(5, Bootstrap::new(20, 3));
        assert_eq!(a, cluster_resamples(5, Bootstrap::new(20, 3)));
        assert_ne!(a, cluster_resamples(5, Bootstrap::new(20, 4)));
        for (r, draw) in a.iter().enumerate() {
            let mut counts = vec![0_u32; 5];
            for &c in draw {
                counts[c] += 1;
            }
            assert_eq!(counts, resample_counts(5, replicate_seed(3, r as u64)));
        }
    }

    #[test]
    fn the_bootstrap_summarises_its_replicates() {
        let sample = pairs_sample();
        let spec = EffectSpec::new(OutcomeModel::Linear);
        let out = bootstrap(&sample, &spec, &BootstrapSpec::new(Bootstrap::new(50, 11))).unwrap();
        let d = &out.difference;
        assert_eq!(d.replicates.len(), 50);
        assert!((d.summary.estimate - 2.625).abs() < 1e-12);
        assert!(d.summary.percentile.lower.unwrap() <= d.summary.estimate + 1.0);
        assert!(
            d.summary.normal.lower.unwrap() < d.summary.estimate
                && d.summary.estimate < d.summary.normal.upper.unwrap()
        );
        assert!(out.ratio.is_none() && out.odds_ratio.is_none());
        let few = bootstrap(&sample, &spec, &BootstrapSpec::new(Bootstrap::new(0, 11)));
        assert_eq!(few.unwrap_err(), Error::Bootstrap);
    }
}
